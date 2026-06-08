# code adapted from DUSt3R
from .utils.transforms import *
from .base.batched_sampler import BatchedRandomSampler  # noqa: F401

# datasets for training
from .co3d import Co3d
from .scannetpp import ScanNetpp # noqa: F401
from .arkitscenes import ARKitScenes
from .blendedmvs import BlendedMVS
from .megadepth import MegaDepth
from .dl3dv import DL3DV
from .realestate10k import RealEstate 

# datasets for evaluation 
from .scannet1500 import ScanNet1500
from .megadepth_valid import MegaDepth_valid
# from .cambridge_retrieval import *
from .cambridge import CambridgeRelpose
from .sevenscenes import SevenScenesRelpose


def get_data_loader(dataset, batch_size, num_workers=8, shuffle=True, drop_last=True, pin_mem=True):
    import torch
    from croco.utils.misc import get_world_size, get_rank

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    try:
        sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                       rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return data_loader

