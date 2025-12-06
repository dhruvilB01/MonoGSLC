# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

MonoGS (Gaussian Splatting SLAM) is a dense SLAM system based on 3D Gaussian Splatting that supports monocular, stereo, and RGB-D inputs. Published at CVPR 2024 as a highlight paper.

## Environment Setup

### Installation
```bash
conda env create -f environment.yml
conda activate MonoGS
```

**Important:** PyTorch/CUDA versions must match your system. The default is `pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6`. Adjust in `environment.yml` if needed (reference: https://pytorch.org/get-started/previous-versions/).

### Dependencies
- Python 3.7.13
- Two custom submodules in `submodules/` must be installed:
  - `simple-knn` (k-nearest neighbors)
  - `diff-gaussian-rasterization` (differential Gaussian rasterization)
- These are installed via pip during conda environment creation

## Common Commands

### Run SLAM System

**Monocular:**
```bash
python slam.py --config configs/mono/tum/fr3_office.yaml
```

**RGB-D:**
```bash
python slam.py --config configs/rgbd/tum/fr3_office.yaml
python slam.py --config configs/rgbd/replica/office0.yaml
```

**Stereo (experimental):**
```bash
python slam.py --config configs/stereo/euroc/mh02.yaml
```

### Evaluation Mode
Add `--eval` flag to run headless without GUI and log rendering metrics:
```bash
python slam.py --config configs/mono/tum/fr3_office.yaml --eval
```

### Live Demo with RealSense
```bash
pip install pyrealsense2  # First time only
python slam.py --config configs/live/realsense.yaml
```

### Download Datasets
```bash
bash scripts/download_tum.sh       # TUM-RGBD dataset
bash scripts/download_replica.sh   # Replica dataset
bash scripts/download_euroc.sh     # EuRoC MAV dataset
```

### Code Quality
```bash
ruff check .                # Lint codebase
ruff format .               # Format code (Black-style, 88 char line length)
```

## Architecture Overview

### Multi-Process Pipeline
The system uses Python multiprocessing to run frontend and backend concurrently:

- **Main Process** (`slam.py`): Orchestrates the SLAM system, initializes components, manages communication queues
- **Frontend Process** (`utils/slam_frontend.py`): Handles camera tracking, keyframe selection, and real-time pose estimation
- **Backend Process** (`utils/slam_backend.py`): Performs mapping (bundle adjustment), Gaussian densification/pruning, and map optimization
- **GUI Process** (optional, `gui/slam_gui.py`): Real-time visualization of reconstruction and tracking

### Communication
Components communicate via `mp.Queue`:
- `frontend_queue`: Backend → Frontend communication
- `backend_queue`: Frontend → Backend communication  
- `q_main2vis`: Main → GUI visualization data
- `q_vis2main`: GUI → Main user interactions

### Core Components

**Gaussian Model** (`gaussian_splatting/scene/gaussian_model.py`):
- Manages 3D Gaussian primitives representing the scene
- Handles densification, pruning, and optimization of Gaussians
- Extends from 3D Gaussian Splatting codebase

**Frontend** (`utils/slam_frontend.py`):
- Processes incoming frames sequentially
- Performs camera tracking by optimizing pose parameters (rotation delta, translation delta, exposure)
- Keyframe selection based on translation, rotation, and overlap criteria
- Initializes new Gaussians for keyframes

**Backend** (`utils/slam_backend.py`):
- Runs mapping optimization on keyframe windows
- Updates Gaussian parameters and camera poses
- Handles Gaussian densification and pruning
- Initial map building for monocular mode

**Rendering** (`gaussian_splatting/gaussian_renderer/__init__.py`):
- Differentiable rasterization of 3D Gaussians
- Outputs RGB image, depth, opacity for tracking and mapping losses

**Dataset Loaders** (`utils/dataset.py`):
- `TUMParser`: TUM RGB-D dataset format
- `ReplicaParser`: Replica dataset format  
- `EuRoCParser`: EuRoC stereo dataset format
- `BaseDataset`: Generic dataset wrapper

### Configuration System

Configs use YAML with inheritance via `inherit_from`:
```yaml
inherit_from: "configs/mono/tum/base_config.yaml"
```

Key config sections:
- `Dataset`: Dataset path, sensor type (monocular/rgbd/stereo), calibration
- `Training`: Tracking/mapping iterations, keyframe intervals, window size, learning rates
- `Results`: Save options, GUI, evaluation flags, wandb logging
- `opt_params`: Gaussian optimization parameters (densification, learning rates)
- `model_params`: Model configuration (SH degree, resolution)
- `pipeline_params`: Rendering pipeline options

### Monocular vs RGB-D Mode

**Monocular:**
- Requires initialization phase to build initial map (`init_itr_num` iterations)
- Depth is estimated from rendered Gaussians during tracking
- New keyframes initialize depth with noise around median depth

**RGB-D:**
- Uses observed depth directly from sensors
- Skips initialization phase, starts mapping immediately
- More stable tracking and mapping

## Key Implementation Details

### Tracking Loop
1. Initialize pose from previous frame
2. Optimize camera pose parameters (rotation delta, translation delta, exposure) via gradient descent
3. Render scene from current pose estimate
4. Compute tracking loss (photometric + depth alignment)
5. Update pose until convergence or max iterations

### Keyframe Selection
Keyframe triggered if:
- Sufficient translation from last keyframe
- Sufficient rotation change  
- Overlap with existing map drops below threshold
- Minimum frames since last keyframe

### Mapping Loop
1. Render all keyframes in current window
2. Compute mapping loss (L1 + SSIM on RGB, depth alignment)
3. Update Gaussian parameters and keyframe poses
4. Periodically densify (split/clone) and prune Gaussians based on gradient statistics

### Gaussian Management
- **Densification**: Split large Gaussians or clone small ones in high-gradient areas
- **Pruning**: Remove Gaussians with low opacity or too large scale
- **Opacity Reset**: Periodically reset opacity to prevent degeneration

## Code Style

Configured via `pyproject.toml`:
- Formatter: ruff (Black-compatible)
- Line length: 88 characters
- Indent: 4 spaces
- Target: Python 3.7
- Quote style: double quotes

## Important Notes

- The codebase has a `dev.speedup` branch with performance improvements (up to 10 FPS)
- Tested on RTX 4090; performance varies by GPU
- Multi-process performance has randomness due to GPU utilization
- For live demos, avoid aggressive camera motion before initial BA completes (first ~15 seconds)
- Use USB-3 port for RealSense cameras
- Disable GUI (`use_gui: False`) for maximum GPU utilization during benchmarking
- Results saved in `save_dir` (default: `results/`)

## External Dependencies

Components from external projects (see `Dependencies.md` and `LICENSE.md`):
- `gaussian_splatting/`: From 3D Gaussian Splatting (graphdeco-inria)
- `gui/gl_render/`: From Tiny Gaussian Splatting Viewer
- `submodules/`: diff-gaussian-rasterization, simple-knn

When modifying these, follow their respective licenses.
