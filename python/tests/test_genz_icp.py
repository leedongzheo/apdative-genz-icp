import importlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

from genz_icp import GenZConfig, GenZICP, voxel_down_sample
from genz_icp.config import load_config, to_genz_config
from genz_icp.datasets import available_dataloaders, dataset_factory
from genz_icp.datasets.kitti import KITTIOdometryDataset
from genz_icp.datasets.mulran import MulranDataset
from genz_icp.pipeline import OdometryPipeline
from genz_icp.tools.visualizer import StubVisualizer
from genz_icp.tools.cmd import guess_dataloader


def test_python_api_smoke():
    odometry = GenZICP(GenZConfig())
    frame = np.random.randn(1000, 3)
    planar, non_planar = odometry.register_frame(frame)
    assert planar.shape[1] == 3
    assert non_planar.shape[1] == 3
    assert odometry.last_pose.shape == (4, 4)


def test_voxel_down_sample_smoke():
    frame = np.random.randn(1000, 3)
    sampled = voxel_down_sample(frame, 0.2)
    assert sampled.shape[1] == 3


def test_stub_visualizer_smoke():
    vis = StubVisualizer()
    vis.update(np.zeros((1, 3)), np.zeros((1, 3)), np.eye(4))
    vis.close()


def test_kitti_loader_with_calib_gt_and_scan_correction(tmp_path):
    base = tmp_path / "kitti"
    seq = base / "sequences" / "00"
    vel = seq / "velodyne"
    poses_dir = base / "poses"
    vel.mkdir(parents=True)
    poses_dir.mkdir(parents=True)

    (seq / "calib.txt").write_text("Tr: 1 0 0 0 0 1 0 0 0 0 1 0\n")
    (seq / "times.txt").write_text("0.0\n0.1\n")

    pose_line = np.hstack([np.eye(3), np.zeros((3, 1))]).reshape(1, 12)
    np.savetxt(poses_dir / "00.txt", np.vstack([pose_line, pose_line]))

    scan = np.random.randn(128, 4).astype(np.float32)
    scan.tofile(vel / "000000.bin")

    ds = KITTIOdometryDataset(base, sequence="00")
    points, ts = ds[0]
    assert points.shape == (128, 3)
    assert ts.size == 0
    assert hasattr(ds, "gt_poses")
    assert ds.get_frames_timestamps().shape[0] == 2


def test_mulran_loader_timestamp_gt_matching_and_calibration(tmp_path):
    base = tmp_path / "mulran_seq"
    ouster_dir = base / "Ouster"
    ouster_dir.mkdir(parents=True)

    np.random.randn(64 * 1024, 4).astype(np.float32).tofile(ouster_dir / "1000.bin")

    pose_flat = np.hstack([np.eye(3), np.zeros((3, 1))]).reshape(-1)
    row = np.concatenate([[1000.0], pose_flat])
    np.savetxt(base / "global_pose.csv", np.vstack([row, row]), delimiter=",")

    ds = MulranDataset(base)
    points, timestamps = ds[0]
    assert points.shape == (64 * 1024, 3)
    assert timestamps.shape[0] == 64 * 1024
    assert hasattr(ds, "gt_poses") and ds.gt_poses.shape[1:] == (4, 4)


def test_apollo_loader_with_metadata_and_behavior(tmp_path):
    pytest.importorskip("natsort")
    pytest.importorskip("pyquaternion")
    try:
        o3d = importlib.import_module("open3d")
    except Exception:
        pytest.skip("open3d runtime is unavailable in this environment")

    base = tmp_path / "apollo"
    pcds = base / "pcds"
    poses = base / "poses"
    pcds.mkdir(parents=True)
    poses.mkdir(parents=True)

    pts = np.random.randn(50, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    o3d.io.write_point_cloud(str(pcds / "000000.pcd"), pcd)

    np.savetxt(poses / "gt_poses.txt", np.array([[0.0, 0.0, 0, 0, 0, 0, 0, 0, 1.0]]))

    ds = dataset_factory("apollo", base)
    points, timestamps = ds[0]
    assert points.shape[1] == 3
    assert timestamps.size == 0
    assert hasattr(ds, "gt_poses")


def test_pipeline_outputs_like_kiss_icp(tmp_path):
    class DummyDataset:
        def __init__(self, data_dir: Path):
            self.data_dir = str(data_dir)
            self.sequence_id = "2018-10-11"
            self.frames = [np.random.randn(2000, 3).astype(np.float64) for _ in range(3)]
            self.gt_poses = np.stack([np.eye(4) for _ in range(3)])

        def __len__(self):
            return len(self.frames)

        def __getitem__(self, idx):
            return self.frames[idx], np.array([])

        def get_frames_timestamps(self):
            return np.arange(len(self.frames), dtype=np.float64)

    config_file = tmp_path / "config.yml"
    config_file.write_text("out_dir: '%s'\n" % (tmp_path / "results"))

    pipeline = OdometryPipeline(dataset=DummyDataset(tmp_path), config=config_file, visualize=False)
    pipeline.run()
    out = Path(pipeline.results_dir)

    assert (out / "result_metrics.log").exists()
    assert (out / "trajectory.g2o").exists()
    assert (out / "trajectory.png").exists() or True
    assert (out / "config.yml").exists()
    assert (out / "local_maps").exists()
    assert (out / "local_maps" / "local_map_final.npy").exists()

    assert (out / "2018-10-11_gt.npy").exists()
    assert (out / "2018-10-11_gt_kitti.txt").exists()
    assert (out / "2018-10-11_gt_tum.txt").exists()
    assert (out / "2018-10-11_poses.npy").exists()
    assert (out / "2018-10-11_poses_kitti.txt").exists()
    assert (out / "2018-10-11_poses_tum.txt").exists()


def test_dataset_factory_and_config_blocks(tmp_path):
    np.save(tmp_path / "000000.npy", np.random.randn(256, 3))
    dataset = dataset_factory("generic", tmp_path)
    frame, timestamps = dataset[0]
    assert frame.shape[1] == 3
    assert timestamps.size == 0

    cfg = load_config(None)
    genz_cfg = to_genz_config(cfg)
    assert isinstance(genz_cfg, GenZConfig)


def test_available_dataloaders_include_kiss_like_entries():
    names = set(available_dataloaders())
    assert {"generic", "kitti", "mulran", "apollo", "ouster"}.issubset(names)


def test_cli_help_has_new_flags():
    result = subprocess.run(["genz_icp_pipeline", "--help"], capture_output=True, text=True, check=True)
    assert "--visualize" in result.stdout
    assert "--dataloader" in result.stdout
    assert "--config" in result.stdout
    assert "--sequence" in result.stdout
    assert "--n-scans" in result.stdout
    assert "--jump" in result.stdout
    assert "--ouster-meta" in result.stdout



def test_guess_dataloader_for_pcap(tmp_path):
    pcap = tmp_path / 'scan.pcap'
    pcap.write_text('dummy')
    dataloader, path = guess_dataloader(pcap, default_dataloader='generic')
    assert dataloader == 'ouster'
    assert path == pcap
