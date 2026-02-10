import glob
import os
from pathlib import Path

import numpy as np

class KITTIOdometryDataset:
    def __init__(self, data_dir: Path, sequence: str = "00", *_, **__):
        self.sequence_id = str(sequence).zfill(2)
        self.kitti_sequence_dir = os.path.join(data_dir, "sequences", self.sequence_id)
        self.velodyne_dir = os.path.join(self.kitti_sequence_dir, "velodyne/")

        self.scan_files = sorted(glob.glob(self.velodyne_dir + "*.bin"))
        if not self.scan_files:
            raise FileNotFoundError(f"No KITTI scans found in {self.velodyne_dir}")

        self.calibration = self.read_calib_file(os.path.join(self.kitti_sequence_dir, "calib.txt"))

        # Load GT poses when available
        if int(sequence) < 11:
            self.poses_fn = os.path.join(data_dir, f"poses/{self.sequence_id}.txt")
            if os.path.exists(self.poses_fn):
                self.gt_poses = self.load_poses(self.poses_fn)

        # --- SỬA Ở ĐÂY: Xóa đoạn gọi C++ binding ---
        # from genz_icp.pybind import genz_icp_pybind
        # self.correct_kitti_scan = lambda frame: ...
        # -------------------------------------------

    def __getitem__(self, idx):
        return self.scans(idx)

    def __len__(self):
        return len(self.scan_files)

    def scans(self, idx):
        return self.read_point_cloud(self.scan_files[idx]), np.array([])

    def apply_calibration(self, poses: np.ndarray) -> np.ndarray:
        """Converts from Velodyne to Camera Frame."""
        tr = np.eye(4, dtype=np.float64)
        tr[:3, :4] = self.calibration["Tr"].reshape(3, 4)
        return tr @ poses @ np.linalg.inv(tr)

    def read_point_cloud(self, scan_file: str):
        points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))[:, :3].astype(np.float64)
        # Gọi hàm sửa lỗi bằng Python thuần
        return self._correct_scan(points)

    # --- HÀM MỚI: Sửa lỗi góc nghiêng KITTI bằng Numpy ---
    def _correct_scan(self, points):
        """
        Sửa lỗi góc nghiêng 0.205 độ đặc thù của KITTI (Vertical Angle Offset).
        """
        VERTICAL_ANGLE_OFFSET = (0.205 * np.pi) / 180.0
        
        # Tạo ma trận xoay bù trừ (Xoay quanh trục Y/X tùy hệ tọa độ, thường là Pitch)
        # Vector xoay chuẩn hóa (dựa trên logic KISS-ICP)
        # Cách đơn giản nhất là xoay quanh trục ngang của xe.
        
        # Logic tương đương C++: 
        # rotationVector = pt.cross(Eigen::Vector3d(0., 0., 1.)) -> Xoay quanh vector vuông góc với Z và điểm đó
        # Nhưng để đơn giản và nhanh trong Python, ta có thể dùng ma trận xoay cố định nếu giả sử LiDAR đặt thẳng.
        # Tuy nhiên, để chính xác 100% như C++, ta dùng công thức Rodrigues hoặc xấp xỉ.
        
        # Cách KISS-ICP làm trong Python (nếu không dùng C++):
        # correction_angle = 0.205 * np.pi / 180.0
        # points[:, 2] += np.linalg.norm(points[:, :2], axis=1) * np.tan(correction_angle)
        # Cách trên là xấp xỉ (small angle approximation), rất nhanh và đủ tốt.
        
        correction_angle = 0.205 * np.pi / 180.0
        # Nâng điểm Z lên dựa theo khoảng cách XY (để bù độ chúi xuống)
        points[:, 2] += np.linalg.norm(points[:, :2], axis=1) * np.tan(correction_angle)
        return points
    # ----------------------------------------------------

    def load_poses(self, poses_file):
        def _lidar_pose_gt(poses_gt):
            _tr = self.calibration["Tr"].reshape(3, 4)
            tr = np.eye(4, dtype=np.float64)
            tr[:3, :4] = _tr
            left = np.einsum("...ij,...jk->...ik", np.linalg.inv(tr), poses_gt)
            right = np.einsum("...ij,...jk->...ik", left, tr)
            return right

        poses = np.loadtxt(poses_file, delimiter=" ")
        n = poses.shape[0]
        poses = np.concatenate(
            (poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)), axis=1
        )
        poses = poses.reshape((n, 4, 4))
        return _lidar_pose_gt(poses)

    def get_frames_timestamps(self) -> np.ndarray:
        times_file = os.path.join(self.kitti_sequence_dir, "times.txt")
        return np.loadtxt(times_file).reshape(-1, 1) if os.path.exists(times_file) else np.array([])

    @staticmethod
    def read_calib_file(file_path: str) -> dict:
        calib_dict = {}
        with open(file_path, "r") as calib_file:
            for line in calib_file.readlines():
                tokens = line.split(" ")
                if tokens[0] == "calib_time:":
                    continue
                if len(tokens) > 0:
                    values = np.array([float(token) for token in tokens[1:] if token != ""], dtype=np.float32)
                    key = tokens[0][:-1]
                    calib_dict[key] = values
        return calib_dict
# class KITTIOdometryDataset:
#     def __init__(self, data_dir: Path, sequence: str = "00", *_, **__):
#         self.sequence_id = str(sequence).zfill(2)
#         self.kitti_sequence_dir = os.path.join(data_dir, "sequences", self.sequence_id)
#         self.velodyne_dir = os.path.join(self.kitti_sequence_dir, "velodyne/")

#         self.scan_files = sorted(glob.glob(self.velodyne_dir + "*.bin"))
#         if not self.scan_files:
#             raise FileNotFoundError(f"No KITTI scans found in {self.velodyne_dir}")

#         self.calibration = self.read_calib_file(os.path.join(self.kitti_sequence_dir, "calib.txt"))

#         # Load GT poses when available
#         if int(sequence) < 11:
#             self.poses_fn = os.path.join(data_dir, f"poses/{self.sequence_id}.txt")
#             if os.path.exists(self.poses_fn):
#                 self.gt_poses = self.load_poses(self.poses_fn)

#         from genz_icp.pybind import genz_icp_pybind

#         self.correct_kitti_scan = lambda frame: np.asarray(
#             genz_icp_pybind._correct_kitti_scan(genz_icp_pybind._Vector3dVector(frame))
#         )

#     def __getitem__(self, idx):
#         return self.scans(idx)

#     def __len__(self):
#         return len(self.scan_files)

#     def scans(self, idx):
#         return self.read_point_cloud(self.scan_files[idx]), np.array([])

#     def apply_calibration(self, poses: np.ndarray) -> np.ndarray:
#         """Converts from Velodyne to Camera Frame."""
#         tr = np.eye(4, dtype=np.float64)
#         tr[:3, :4] = self.calibration["Tr"].reshape(3, 4)
#         return tr @ poses @ np.linalg.inv(tr)

#     def read_point_cloud(self, scan_file: str):
#         points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))[:, :3].astype(np.float64)
#         return self.correct_kitti_scan(points)

#     def load_poses(self, poses_file):
#         def _lidar_pose_gt(poses_gt):
#             _tr = self.calibration["Tr"].reshape(3, 4)
#             tr = np.eye(4, dtype=np.float64)
#             tr[:3, :4] = _tr
#             left = np.einsum("...ij,...jk->...ik", np.linalg.inv(tr), poses_gt)
#             right = np.einsum("...ij,...jk->...ik", left, tr)
#             return right

#         poses = np.loadtxt(poses_file, delimiter=" ")
#         n = poses.shape[0]
#         poses = np.concatenate(
#             (poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)), axis=1
#         )
#         poses = poses.reshape((n, 4, 4))
#         return _lidar_pose_gt(poses)

#     def get_frames_timestamps(self) -> np.ndarray:
#         times_file = os.path.join(self.kitti_sequence_dir, "times.txt")
#         return np.loadtxt(times_file).reshape(-1, 1) if os.path.exists(times_file) else np.array([])

#     @staticmethod
#     def read_calib_file(file_path: str) -> dict:
#         calib_dict = {}
#         with open(file_path, "r") as calib_file:
#             for line in calib_file.readlines():
#                 tokens = line.split(" ")
#                 if tokens[0] == "calib_time:":
#                     continue
#                 if len(tokens) > 0:
#                     values = np.array([float(token) for token in tokens[1:] if token != ""], dtype=np.float32)
#                     key = tokens[0][:-1]
#                     calib_dict[key] = values
#         return calib_dict
