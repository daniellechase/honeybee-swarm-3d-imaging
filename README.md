# honeybee-swarm-3d-imaging

Analysis and figure generation pipelines for honeybee swarm assembly experiments.

> **Data availability:** The `data/` directory in this repository contains pre-computed outputs (calibration parameters, tracks, morphology metrics) for session S02/0722. Pipelines marked runs with provided data work out of the box. Pipelines marked requires raw data need access to the original raw video files to run. The code used to produce the provided data is included.

---

## Setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

After installation all `run-*` commands are available in the terminal.

---

## Project structure

```
configs/                          YAML configs — one per session / pipeline run
  calibration/
  morphology/
  trajectories/
  figures/
    figmorph/
    figtraj/
    frame_sequences/
    ml_identification/
    quiver/

data/                             Pre-computed outputs included in this repo
  calibration/S02/0722/           Stereo calibration parameters (JSON)
  morphology/S02/0722/            Morphology metrics (parquet)
  trajectories/S02/0722/          3D tracks and 2D tracks (parquet)

src/swarm_assembly_methods/       Package source code
  calibration/                    Stereo camera calibration (AprilTag-based)
  morphology/                     Swarm morphology metrics (volume, diameter, length)
  trajectories/                   2D tracking + stereo triangulation
  figures/
    figmorph/                     Morphology time-series figures
    figtraj/                      Trajectory overlay and quiver figures
    frame_sequences/              Camcorder frame-strip figures
    ml_identification/            ML-based identification figures
```

---

## Pipelines

Each pipeline require a single YAML config.

---

### Morphology figures — *runs with provided data*

Generate time-series figures (normalized volume/mass, diameter/length, scatter plots, flying bee count) from a pre-computed metrics parquet.

```bash
run-morphology-figures --config configs/figures/figmorph/S02_0722_gate.yaml
```

Key config sections:
- `input.metrics_cache` — path to parquet produced by `run-morphology`
- `input.weight_csv` — optional scale data
- `session` — phase boundary timestamps
- `plots` — output format, smoothing windows, normalization mode

---

### Trajectory figures — *runs with provided data*

Trajectory overlay figures from 3D tracks. Set `video_left`/`video_right` to `null` in the config to draw trajectories on blank frames without the raw video.

```bash
run-trajectory --config configs/figures/figtraj/S02_0722.yaml
```

---

### Quiver figures — *runs with provided data*

Quiver / velocity-field plots from 3D tracks.

```bash
run-quiver --config configs/figures/quiver/S02_0722_gate.yaml
```

---

### Morphology — *requires raw data*

Compute morphology metrics (volume, diameter, length) from swarm mask NPZ files on the lab storage drives.

```bash
run-morphology --config configs/morphology/S02_0722.yaml
```

---

### Trajectories — *requires raw data*

Run 2D tracking and stereo triangulation from GoPro video pairs on the lab storage drives.

```bash
run-trajectories --config configs/trajectories/S02_0722.yaml
```

---

### Calibration — *requires raw data*

Stereo camera calibration pipeline (AprilTag-based), steps 1–6. Requires the raw calibration videos on the lab storage drives.

Run all steps:
```bash
run-calibration --config configs/calibration/S02/0722/S02_0722_gate.yaml
```

Run individual steps:
```bash
run-calibration-s1 --config configs/calibration/S02/0722/S02_0722_gate.yaml   # Export frames
run-calibration-s2 --config configs/calibration/S02/0722/S02_0722_gate.yaml   # Intrinsics
run-calibration-s3 --config configs/calibration/S02/0722/S02_0722_gate.yaml   # Sweep dk
run-calibration-s4 --config configs/calibration/S02/0722/S02_0722_gate.yaml   # Extrinsics
run-calibration-s5 --config configs/calibration/S02/0722/S02_0722_gate.yaml   # Check rectification
run-calibration-s6 --config configs/calibration/S02/0722/S02_0722_gate.yaml   # View rectified
```

Skip or isolate steps:
```bash
run-calibration --config configs/calibration/S02/0722/S02_0722_gate.yaml --skip 1 6
run-calibration --config configs/calibration/S02/0722/S02_0722_gate.yaml --only 2 4
```

---

### Frame sequence strips — *requires raw data*

Extract frames from a camcorder video at specified timestamps and assemble them into a horizontal strip figure (PDF + PNG). Requires the raw camcorder video on the lab storage drives.

```bash
run-frame-sequences --config configs/figures/frame_sequences/S202_0725.yaml
```

Key config sections:
- `save_dir` — output directory
- `strip` — figure width, DPI, padding, font size
- `sequences` — one entry per strip:
  - `video` — path to camcorder video
  - `times` — list of timestamps `"HH:MM:SS"`
  - `label_unit` — `"min"` or `"sec"`
  - `outputs` — output filenames (`.pdf`, `.png`)

---

### ML identification figures — *requires raw data*

```bash
run-ml-identification --config configs/figures/ml_identification/S02_0722.yaml
```

---
