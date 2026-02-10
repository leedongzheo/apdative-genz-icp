# GenZ-ICP Python Package

This folder provides a pip-installable Python package for the GenZ-ICP core.

## Step-by-step: build C++ core, then run the Python pipeline

### 1) Install system dependencies

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build libtbb-dev
```

(Optional but recommended for visualizer and dataset loaders)

```bash
pip install "genz-icp[all]"
```

### 2) Build and install the Python package (builds C++ core via CMake)

```bash
cd /workspace/pipeline_core/genz-icp/python
python3 -m pip install -e .
```

This step compiles the C++ core and installs the Python bindings in editable mode.

### 3) Run the Python pipeline

#### 3.1 Generic dataset (folder of `*.npy` frames)

```bash
genz_icp_pipeline /path/to/frames --dataloader generic --config /path/to/config.yml
```

#### 3.2 KITTI

```bash
genz_icp_pipeline /path/to/kitti --dataloader kitti --sequence 00
```

#### 3.3 Mulran

```bash
genz_icp_pipeline /path/to/mulran_seq --dataloader mulran
```

#### 3.4 Apollo

```bash
genz_icp_pipeline /path/to/apollo_root --dataloader apollo
```

#### 3.5 Ouster (pcap)

```bash
genz_icp_pipeline /path/to/scan.pcap --dataloader ouster --ouster-meta /path/to/metadata.json
```

#### 3.6 Visualization

```bash
genz_icp_pipeline /path/to/frames --visualize
```

## Output artifacts (KISS-ICP-like)

After each run, results are written to `<out_dir>/<timestamp>/` (with a `latest` symlink), including:
- `<sequence>_poses.npy`, `<sequence>_poses_kitti.txt`, `<sequence>_poses_tum.txt`
- `<sequence>_gt.npy`, `<sequence>_gt_kitti.txt`, `<sequence>_gt_tum.txt` (when GT exists)
- `config.yml`
- `result_metrics.log`
- `trajectory.g2o`
- `trajectory.png`
- `local_maps/local_map_final.npy`
