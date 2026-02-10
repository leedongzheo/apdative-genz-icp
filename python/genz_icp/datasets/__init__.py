from pathlib import Path
from typing import Dict, List


def supported_file_extensions():
    return ["bin", "pcd", "ply", "xyz", "obj", "ctm", "off", "stl", "npy"]


def available_dataloaders() -> List[str]:
    return ["generic", "kitti", "mulran", "apollo", "ouster"]


def dataloader_types() -> Dict[str, str]:
    return {
        "generic": "GenericDataset",
        "kitti": "KITTIOdometryDataset",
        "mulran": "MulranDataset",
        "apollo": "ApolloDataset",
        "ouster": "OusterDataloader",
    }


def dataset_factory(dataloader: str, data_dir: Path, *args, **kwargs):
    import importlib

    if dataloader not in available_dataloaders():
        raise ValueError(f"Unsupported dataloader '{dataloader}'. Available: {available_dataloaders()}")
    module = importlib.import_module(f".{dataloader}", __name__)
    dataset = getattr(module, dataloader_types()[dataloader])
    return dataset(data_dir=data_dir, *args, **kwargs)


__all__ = [
    "supported_file_extensions",
    "available_dataloaders",
    "dataloader_types",
    "dataset_factory",
]
