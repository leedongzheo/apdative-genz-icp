from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from genz_icp.pybind import genz_icp_pybind


@dataclass
class GenZConfig:
    max_range: float = 100.0
    min_range: float = 0.5
    map_cleanup_radius: float = 400.0
    max_points_per_voxel: int = 1
    voxel_size: float = 0.25
    desired_num_voxelized_points: int = 2000
    min_motion_th: float = 0.1
    initial_threshold: float = 2.0
    
    # [CŨ] Dùng cho Baseline (Fixed Threshold)
    planarity_threshold: float = 0.1 

    # === [THÊM MỚI] Các tham số cho Adaptive Threshold ===
    use_adaptive_planarity: bool = True
    adaptive_threshold_base: float = 0.06
    min_adaptive_threshold: float = 0.001
    max_adaptive_threshold: float = 0.2
    # ====================================================

    deskew: bool = False
    max_num_iterations: int = 150
    convergence_criterion: float = 0.0001

    def _to_cpp(self):
        config = genz_icp_pybind._GenZConfig()
        for key, value in self.__dict__.items():
            # Hàm này sẽ tự động copy các biến mới (use_adaptive_planarity...)
            # sang C++ object nhờ vào việc bạn đã bind chúng trong genz_icp_pybind.cpp
            setattr(config, key, value)
        return config


def _to_cpp_points(frame: np.ndarray):
    points = np.asarray(frame, dtype=np.float64)
    return genz_icp_pybind._Vector3dVector(points)


class GenZICP:
    def __init__(self, config: Optional[GenZConfig] = None):
        self.config = config or GenZConfig()
        self._odometry = genz_icp_pybind._GenZICP(self.config._to_cpp())

    def register_frame(
        self, frame: np.ndarray, timestamps: Optional[Iterable[float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        points = _to_cpp_points(frame)
        if timestamps is None:
            planar, non_planar = self._odometry._register_frame(points)
        else:
            planar, non_planar = self._odometry._register_frame(points, list(timestamps))
        return np.asarray(planar), np.asarray(non_planar)

    @property
    def poses(self) -> List[np.ndarray]:
        return [np.asarray(pose) for pose in self._odometry._poses()]

    @property
    def last_pose(self) -> np.ndarray:
        return np.asarray(self._odometry._last_pose())

    @property
    def local_map(self) -> np.ndarray:
        return np.asarray(self._odometry._local_map())


def voxel_down_sample(frame: np.ndarray, voxel_size: float) -> np.ndarray:
    return np.asarray(genz_icp_pybind._voxel_down_sample(_to_cpp_points(frame), voxel_size))
