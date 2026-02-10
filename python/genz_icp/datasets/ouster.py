import os
from typing import Optional

import numpy as np


class OusterDataloader:
    """Ouster pcap dataloader."""

    def __init__(self, data_dir: str, meta: Optional[str] = None, *_, **__):
        try:
            from ouster.sdk import client, open_source
        except ImportError as exc:
            raise ImportError('ouster-sdk is not installed, run: pip install "genz-icp[all]"') from exc

        assert os.path.isfile(data_dir), "Ouster dataloader expects an existing PCAP file"

        source = open_source(str(data_dir), meta=[meta] if meta else [], index=True)

        self._client = client
        self.data_dir = os.path.dirname(data_dir)
        self._xyz_lut = client.XYZLut(source.metadata)
        self._pcap_file = str(data_dir)
        self._scans_num = len(source)
        self._timestamps = np.linspace(0, self._scans_num, self._scans_num, endpoint=False)
        self._source = source

    def __getitem__(self, idx):
        scan = self._source[idx]
        self._timestamps[idx] = 1e-9 * scan.timestamp[0]

        timestamps = np.tile(np.linspace(0, 1.0, scan.w, endpoint=False), (scan.h, 1))
        sel_flag = scan.field(self._client.ChanField.RANGE) != 0
        xyz = self._xyz_lut(scan)[sel_flag]
        timestamps = timestamps[sel_flag]
        return xyz, timestamps

    def get_frames_timestamps(self):
        return self._timestamps

    def __len__(self):
        return self._scans_num
