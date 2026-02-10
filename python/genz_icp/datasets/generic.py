from pathlib import Path
from typing import Tuple

import numpy as np


class GenericDataset:
    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.scan_files = sorted(self.data_dir.glob("*.npy"))
        if not self.scan_files:
            raise FileNotFoundError(f"No .npy frames found in {self.data_dir}")

    def __len__(self):
        return len(self.scan_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        points = np.load(self.scan_files[idx]).astype(np.float64)
        timestamps = np.array([], dtype=np.float64)
        return points, timestamps
