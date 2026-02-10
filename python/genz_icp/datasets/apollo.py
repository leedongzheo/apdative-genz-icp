import glob
import importlib
import os
from pathlib import Path

import numpy as np


class ApolloDataset:
    def __init__(self, data_dir: Path, *_, **__):
        try:
            self.o3d = importlib.import_module("open3d")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                'pcd files require open3d. Install with: pip install "genz-icp[all]"'
            ) from exc

        try:
            natsort = importlib.import_module("natsort")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                'Apollo loader requires natsort. Install with: pip install "genz-icp[all]"'
            ) from exc

        self.scan_files = natsort.natsorted(glob.glob(f"{data_dir}/pcds/*.pcd"))
        if not self.scan_files:
            raise FileNotFoundError(f"No Apollo .pcd scans found in {data_dir}/pcds")

        gt_file = f"{data_dir}/poses/gt_poses.txt"
        if os.path.exists(gt_file):
            self.gt_poses = self.read_poses(gt_file)
        self.sequence_id = os.path.basename(data_dir)

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, idx):
        return self.get_scan(self.scan_files[idx]), np.array([])

    def get_scan(self, scan_file: str):
        points = np.asarray(self.o3d.io.read_point_cloud(scan_file).points, dtype=np.float64)
        return points.astype(np.float64)

    @staticmethod
    def read_poses(file):
        try:
            Quaternion = importlib.import_module("pyquaternion").Quaternion
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                'Apollo GT pose parsing requires pyquaternion. Install with: pip install "genz-icp[all]"'
            ) from exc

        data = np.loadtxt(file)
        _, _, translations, qxyzw = np.split(data, [1, 2, 5], axis=1)
        rotations = np.array([Quaternion(x=x, y=y, z=z, w=w).rotation_matrix for x, y, z, w in qxyzw])
        poses = np.zeros([rotations.shape[0], 4, 4])
        poses[:, :3, -1] = translations
        poses[:, :3, :3] = rotations
        poses[:, -1, -1] = 1
        first_pose = poses[0, :, :]
        return np.linalg.inv(first_pose) @ poses
