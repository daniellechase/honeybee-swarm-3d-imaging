# honeybee-swarm-3d-imaging

Analysis and figure generation pipelines for the paper: 3D Imaging of Honeybee Swarm Assembly and Disassembly

> **Data availability:** Data is available at Zenodo: [https://doi.org/10.5281/zenodo.18992442](https://doi.org/10.5281/zenodo.18992442) and should be extracted to the `data/` folder.

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

data/                             access data at zenodo and upload all folders to this directory

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

### Morphology figures

Generate time-series figures (normalized volume/mass, diameter/length, scatter plots, flying bee count) from cached data

```bash
run-morphology-figures --config configs/figures/figmorph/S02_0722_gate.yaml
```
---

### Trajectory figures

Trajectory overlay figures from cached trajectories

```bash
run-trajectory --config configs/figures/figtraj/S02_0722.yaml
```

---

### Quiver figures

Quiver / velocity-field plots from 3D tracks.

```bash
run-quiver --config configs/figures/quiver/S02_0722_gate.yaml
```

---

### Morphology

Compute morphology metrics (volume, diameter, height) from swarm masks

```bash
run-morphology --config configs/morphology/S02_0722.yaml
```

---

### Trajectories

Run 2D tracking and stereo triangulation from GoPro video pairs

```bash
run-trajectories --config configs/trajectories/S02_0722.yaml
```

---

### Calibration

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

### Frame sequence strips

```bash
run-frame-sequences --config configs/figures/frame_sequences/S202_0725.yaml
```
---

### ML identification figures

```bash
run-ml-identification --config configs/figures/ml_identification/S02_0722.yaml
```

---
