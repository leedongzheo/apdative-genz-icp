import numpy as np

from genz_icp.pybind import genz_icp_pybind


def sequence_error(gt_poses: np.ndarray, poses: np.ndarray):
    return genz_icp_pybind._kitti_seq_error(gt_poses, poses)


def absolute_trajectory_error(gt_poses: np.ndarray, poses: np.ndarray):
    return genz_icp_pybind._absolute_trajectory_error(gt_poses, poses)
