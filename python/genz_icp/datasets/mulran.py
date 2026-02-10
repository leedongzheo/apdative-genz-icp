import glob
import os
from pathlib import Path

import numpy as np


class MulranDataset:
    def __init__(self, data_dir: Path, *_, **__):
        self.sequence_id = os.path.basename(data_dir)
        self.sequence_dir = os.path.realpath(data_dir)
        self.velodyne_dir = os.path.join(self.sequence_dir, "Ouster/")

        self.scan_files = sorted(glob.glob(self.velodyne_dir + "*.bin"))
        if not self.scan_files:
            raise FileNotFoundError(f"No Mulran scans found in {self.velodyne_dir}")
        self.scan_timestamps = [int(os.path.basename(t).split(".")[0]) for t in self.scan_files]

        gt_file = os.path.join(self.sequence_dir, "global_pose.csv")
        if os.path.exists(gt_file):
            self.gt_poses = self.load_gt_poses(gt_file)

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, idx):
        return self.read_point_cloud(self.scan_files[idx])

    def read_point_cloud(self, file_path: str):
        points = np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))[:, :3]
        timestamps = self.get_timestamps()
        if points.shape[0] != timestamps.shape[0]:
            return points.astype(np.float64), np.array([])
        return points.astype(np.float64), timestamps

    @staticmethod
    def get_timestamps():
        h, w = 64, 1024
        return np.floor(np.arange(h * w) / h) / w

    def load_gt_poses(self, poses_file: str):
        """Mulran has more poses than scans; match by timestamp."""

        def read_csv(_poses_file: str):
            poses = np.loadtxt(_poses_file, delimiter=",")
            timestamps = poses[:, 0]
            poses = poses[:, 1:]
            n = poses.shape[0]
            poses = np.concatenate(
                (poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)),
                axis=1,
            )
            poses = poses.reshape((n, 4, 4))
            return poses, timestamps

        poses, timestamps = read_csv(poses_file)
        poses = poses[[np.argmin(abs(timestamps - t)) for t in self.scan_timestamps]]

        first_pose = poses[0, :, :]
        poses = np.linalg.inv(first_pose) @ poses

        t_lidar_to_base, t_base_to_lidar = self._get_calibration()
        return t_lidar_to_base @ poses @ t_base_to_lidar

    def _get_calibration(self):
        t_lidar_to_base = np.array(
            [
                [-9.9998295e-01, -5.8398386e-03, -5.2257060e-06, 1.7042000e00],
                [5.8398386e-03, -9.9998295e-01, 1.7758769e-06, -2.1000000e-02],
                [-5.2359878e-06, 1.7453292e-06, 1.0000000e00, 1.8047000e00],
                [0.0000000e00, 0.0000000e00, 0.0000000e00, 1.0000000e00],
            ]
        )
        t_base_to_lidar = np.linalg.inv(t_lidar_to_base)
        return t_lidar_to_base, t_base_to_lidar
