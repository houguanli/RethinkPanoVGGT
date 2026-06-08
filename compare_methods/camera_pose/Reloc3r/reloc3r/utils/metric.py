# import torch  # useless
import numpy as np
import cv2
from pdb import set_trace as bb


def get_rot_err(rot_a, rot_b):
    rot_err = rot_a.T.dot(rot_b)
    rot_err = cv2.Rodrigues(rot_err)[0]
    rot_err = np.reshape(rot_err, (1,3))
    rot_err = np.reshape(np.linalg.norm(rot_err, axis = 1), -1) / np.pi * 180.
    return rot_err[0]

def get_transl_ang_err(dir_a, dir_b):
    dot_product = np.sum(dir_a * dir_b)
    cos_angle = dot_product / (np.linalg.norm(dir_a) * np.linalg.norm(dir_b))
    angle = np.arccos(cos_angle)
    err = np.degrees(angle)
    return err


def error_auc(rError, tErrors, thresholds):
    """
    Args:
        Error (list): [N,]
        tErrors (list): [N,]
        thresholds (list)
    """
    error_matrix = np.concatenate((rError[:, None], tErrors[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    errors = [0] + sorted(list(max_errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    # thresholds = [5, 10, 20, 30]
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


# def calculate_auc(r_error, t_error, max_threshold=30):
#     """
#     Calculate the Area Under the Curve (AUC) for the given error arrays using PyTorch.

#     :param r_error: torch.Tensor representing R error values (Degree).
#     :param t_error: torch.Tensor representing T error values (Degree).
#     :param max_threshold: maximum threshold value for binning the histogram.
#     :return: cumulative sum of normalized histogram of maximum error values.
#     """
#     # Concatenate the error tensors along a new axis
#     error_matrix = torch.stack((r_error, t_error), dim=1)

#     # Compute the maximum error value for each pair
#     max_errors, _ = torch.max(error_matrix, dim=1)

#     # Define histogram bins
#     bins = torch.arange(max_threshold + 1)

#     # Calculate histogram of maximum error values
#     histogram = torch.histc(max_errors, bins=max_threshold + 1, min=0, max=max_threshold)

#     # Normalize the histogram
#     num_pairs = float(max_errors.size(0))
#     normalized_histogram = histogram / num_pairs

#     # Compute and return the cumulative sum of the normalized histogram
#     return torch.cumsum(normalized_histogram, dim=0).mean()


def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays.

    :param r_error: numpy array representing R error values (Degree).
    :param t_error: numpy array representing T error values (Degree).
    :param max_threshold: maximum threshold value for binning the histogram.
    :return: cumulative sum of normalized histogram of maximum error values.
    """

    # Concatenate the error arrays along a new axis
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)

    # Compute the maximum error value for each pair
    max_errors = np.max(error_matrix, axis=1)

    # Define histogram bins
    bins = np.arange(max_threshold + 1)

    # Calculate histogram of maximum error values
    histogram, _ = np.histogram(max_errors, bins=bins)

    # Normalize the histogram
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs

    # Compute and return the cumulative sum of the normalized histogram
    return np.mean(np.cumsum(normalized_histogram))

