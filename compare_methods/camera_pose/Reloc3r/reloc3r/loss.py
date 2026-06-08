# references: DUSt3R: https://github.com/naver/dust3r


from copy import copy, deepcopy
import math
import torch
import torch.nn as nn


class LLoss (nn.Module):
    """ L-norm loss
    """

    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, a, b):
        assert a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3, f'Bad shape = {a.shape}'
        dist = self.distance(a, b)
        assert dist.ndim == a.ndim-1  # one dimension less
        if self.reduction == 'none':
            return dist
        if self.reduction == 'sum':
            return dist.sum()
        if self.reduction == 'mean':
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f'bad {self.reduction=} mode')

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss (LLoss):
    """ Euclidean distance between 3d points  """

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


L21 = L21Loss()


class Criterion (nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(criterion, LLoss), f'{criterion} is not a proper criterion!'+bb()
        self.criterion = copy(criterion)

    def get_name(self):
        return f'{type(self).__name__}({self.criterion})'

    def with_reduction(self, mode):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = 'none'  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss (nn.Module):
    """ Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res
    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f'{self._alpha:g}*{name}'
        if self._loss2:
            name = f'{name} + {self._loss2}'
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)

        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details


class RelativeCameraPoseRegression(Criterion, MultiLoss): 
    def __init__(self, criterion):
        super().__init__(criterion)
        self.PoseLoss = Reloc3rPoseLoss() 

    def get_poses(self, gt1, gt2, pose1, pose2):
        gt_pose2to1 = torch.inverse(gt1['camera_pose']) @ gt2['camera_pose']
        gt_pose1to2 = torch.inverse(gt2['camera_pose']) @ gt1['camera_pose']
        pr_pose2to1 = pose2['pose'] 
        pr_pose1to2 = pose1['pose']
        return gt_pose2to1, pr_pose2to1, gt_pose1to2, pr_pose1to2, {}

    def compute_loss(self, gt1, gt2, pose1, pose2, **kw):

        gt_pose2to1, pr_pose2to1, gt_pose1to2, pr_pose1to2, monitoring = self.get_poses(gt1, gt2, pose1, pose2)

        # compute loss
        loss_pose2, loss_Terr2, loss_Rerr2 = self.PoseLoss(pr_pose2to1, gt_pose2to1)
        loss_pose1, loss_Terr1, loss_Rerr1 = self.PoseLoss(pr_pose1to2, gt_pose1to2)

        # record and return details
        self_name = type(self).__name__
        details = {
                   self_name+'_terr1': float(loss_Terr1*180 / math.pi),
                   self_name+'_rerr1': float(loss_Rerr1*180 / math.pi),
                   self_name+'_terr2': float(loss_Terr2*180 / math.pi),
                   self_name+'_rerr2': float(loss_Rerr2*180 / math.pi),
                  }
        return loss_pose1 + loss_pose2, dict(pose_loss = float(loss_pose1 + loss_pose2), **(details | monitoring))


class Reloc3rPoseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pose_pred, pose_gt):
        t = pose_pred[:,0:3,-1]
        tgt = pose_gt[:,0:3,-1]
        R = pose_pred[:, :3, :3]
        Rgt = pose_gt[:, :3, :3]

        trans_loss = self.transl_ang_loss(t, tgt)
        rot_loss = self.rot_ang_loss(R, Rgt)
        loss =  trans_loss + rot_loss
        return loss, trans_loss, rot_loss

    def transl_ang_loss(self, t, tgt, eps=1e-6):
        """
        Args: 
            t: estimated translation vector [B, 3]
            tgt: ground-truth translation vector [B, 3]
        Returns: 
            T_err: translation direction angular error 
        """
        t_norm = torch.norm(t, dim=1, keepdim=True)
        t_normed = t / (t_norm + eps)
        tgt_norm = torch.norm(tgt, dim=1, keepdim=True)
        tgt_normed = tgt / (tgt_norm + eps)
        cosine = torch.sum(t_normed * tgt_normed, dim=1)
        T_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return T_err.mean()

    def rot_ang_loss(self, R, Rgt, eps=1e-6):
        """
        Args:
            R: estimated rotation matrix [B, 3, 3]
            Rgt: ground-truth rotation matrix [B, 3, 3]
        Returns:  
            R_err: rotation angular error 
        """
        residual = torch.matmul(R.transpose(1, 2), Rgt)
        trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
        cosine = (trace - 1) / 2
        R_err = torch.acos(torch.clamp(cosine, -1.0 + eps, 1.0 - eps))  # handle numerical errors and NaNs
        return R_err.mean()


def loss_of_one_batch(batch, model, criterion, device, use_amp=False, ret=None):
    view1, view2 = batch
    for view in batch:
        for name in 'img camera_intrinsics camera_pose'.split(): 
            if name not in view:
                continue
            view[name] = view[name].to(device, non_blocking=True)

    with torch.cuda.amp.autocast(enabled=bool(use_amp)):

        pose1, pose2 = model(view1, view2)

        # loss is supposed to be symmetric
        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(view1, view2, pose1, pose2) if criterion is not None else None

    result = dict(view1=view1, view2=view2, pose1=pose1, pose2=pose2, loss=loss)
    return result[ret] if ret else result

