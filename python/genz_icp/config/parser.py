from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

from genz_icp.genz_icp import GenZConfig

from .config import AdaptiveThresholdConfig, DataConfig, MappingConfig, RegistrationConfig


class GenZPipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="genz_icp_")
    out_dir: str = "results"
    data: DataConfig = DataConfig()
    mapping: MappingConfig = MappingConfig()
    registration: RegistrationConfig = RegistrationConfig()
    adaptive_threshold: AdaptiveThresholdConfig = AdaptiveThresholdConfig()


# def _yaml_source(config_file: Optional[Path]) -> Dict[str, Any]:
#     if config_file is None:
#         return {}
#     yaml = importlib.import_module("yaml")
#     with open(config_file) as cfg_file:
#         return yaml.safe_load(cfg_file) or {}
def _yaml_source(config_file: Optional[Path]) -> Dict[str, Any]:
    if config_file is None:
        return {}
    try:
        yaml = importlib.import_module("yaml")
    except ModuleNotFoundError:
        # Fallback cơ bản nếu không có pyyaml (dù nên cài)
        return {}
        
    with open(config_file) as cfg_file:
        return yaml.safe_load(cfg_file) or {}

# def load_config(config_file: Optional[Path]) -> GenZPipelineConfig:
#     config = GenZPipelineConfig(**_yaml_source(config_file))
#     if config.data.max_range < config.data.min_range:
#         config.data.min_range = 0.0
#     return config
def load_config(config_file: Optional[Path]) -> GenZPipelineConfig:
    """Load configuration from an optional yaml file."""
    config = GenZPipelineConfig(**_yaml_source(config_file))

    # 1. Logic sửa lỗi min range (giữ nguyên)
    if config.data.max_range < config.data.min_range:
        config.data.min_range = 0.0

    # 2. Logic TỰ ĐỘNG TÍNH VOXEL SIZE (Mới thêm vào)
    # Giống hệt KISS-ICP: Nếu không set voxel_size, nó sẽ lấy max_range / 100
    if config.mapping.voxel_size is None:
        config.mapping.voxel_size = float(config.data.max_range / 100.0)

    return config

# def to_genz_config(config: GenZPipelineConfig) -> GenZConfig:
#     assert config.mapping.voxel_size is not None, "Voxel size has not been computed!"
#     return GenZConfig(
#         max_range=config.data.max_range,
#         min_range=config.data.min_range,
#         map_cleanup_radius=config.mapping.map_cleanup_radius,
#         max_points_per_voxel=config.mapping.max_points_per_voxel,
#         voxel_size=config.mapping.voxel_size,
#         desired_num_voxelized_points=config.mapping.desired_num_voxelized_points,
#         min_motion_th=config.adaptive_threshold.min_motion_th,
#         initial_threshold=config.adaptive_threshold.initial_threshold,
#         planarity_threshold=config.adaptive_threshold.planarity_threshold,
#         deskew=config.data.deskew,
#         max_num_iterations=config.registration.max_num_iterations,
#         convergence_criterion=config.registration.convergence_criterion,
#     )
def to_genz_config(config: GenZPipelineConfig) -> GenZConfig:
    assert config.mapping.voxel_size is not None, "Voxel size has not been computed!"
    
    # [QUAN TRỌNG] Vì bên C++ chỉ có constructor rỗng, ta phải gán từng dòng
    # Thay vì return GenZConfig(max_range=..., ...) hãy làm như sau:
    
    c_config = GenZConfig()
    
    # 1. Gán các tham số cũ
    c_config.max_range = config.data.max_range
    c_config.min_range = config.data.min_range
    c_config.map_cleanup_radius = config.mapping.map_cleanup_radius
    c_config.max_points_per_voxel = config.mapping.max_points_per_voxel
    c_config.voxel_size = config.mapping.voxel_size
    c_config.desired_num_voxelized_points = config.mapping.desired_num_voxelized_points
    c_config.min_motion_th = config.adaptive_threshold.min_motion_th
    c_config.initial_threshold = config.adaptive_threshold.initial_threshold
    c_config.planarity_threshold = config.adaptive_threshold.planarity_threshold
    c_config.deskew = config.data.deskew
    c_config.max_num_iterations = config.registration.max_num_iterations
    c_config.convergence_criterion = config.registration.convergence_criterion
    
    # 2. [THÊM MỚI] Gán các tham số Adaptive
    c_config.use_adaptive_planarity = config.adaptive_threshold.use_adaptive_planarity
    c_config.adaptive_threshold_base = config.adaptive_threshold.adaptive_threshold_base
    c_config.min_adaptive_threshold = config.adaptive_threshold.min_adaptive_threshold
    c_config.max_adaptive_threshold = config.adaptive_threshold.max_adaptive_threshold

    return c_config

def write_config(config: GenZPipelineConfig = GenZPipelineConfig(), filename: str = "genz_icp.yaml"):
    with open(filename, "w") as outfile:
        try:
            yaml = importlib.import_module("yaml")
            yaml.dump(config.model_dump(), outfile, default_flow_style=False)
        except ModuleNotFoundError:
            outfile.write(str(config.model_dump()))
