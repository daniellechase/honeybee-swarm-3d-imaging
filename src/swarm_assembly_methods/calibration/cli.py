import argparse
from .config import load_config
from .export_frames import run_export_frames
from .intrinsics import run_intrinsics
from .sweep_dk import run_sweep_dk
from .extrinsics import run_extrinsics
from .check_rectification import run_check_rectification
from .view_rectified import run_view_rectified

STEPS = {
    1: ("Export frames",       run_export_frames),
    2: ("Intrinsics",          run_intrinsics),
    3: ("Sweep dk",            run_sweep_dk),
    4: ("Extrinsics",          run_extrinsics),
    5: ("Check rectification", run_check_rectification),
    6: ("View rectified",      run_view_rectified),
}


def main():
    parser = argparse.ArgumentParser(description="Run calibration pipeline steps 1-6.")
    parser.add_argument("--config", required=True, help="Path to calibration YAML config")
    parser.add_argument("--skip", type=int, nargs="*", default=[],
                        help="Step numbers to skip (e.g. --skip 1 3)")
    parser.add_argument("--only", type=int, nargs="*", default=[],
                        help="Only run these steps (e.g. --only 2 4)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    steps_to_run = list(STEPS.keys())
    if args.only:
        steps_to_run = [s for s in steps_to_run if s in args.only]
    if args.skip:
        steps_to_run = [s for s in steps_to_run if s not in args.skip]

    for step_num in steps_to_run:
        name, fn = STEPS[step_num]
        print(f"\n{'='*60}")
        print(f"  Step {step_num}: {name}")
        print(f"{'='*60}\n")
        fn(cfg)

    print("\nPipeline complete.")


def main_step1():
    name, fn = STEPS[1]
    parser = argparse.ArgumentParser(description=f"Calibration step 1: {name}.")
    parser.add_argument("--config", required=True, help="Path to calibration YAML config")
    parser.add_argument("--camera", choices=["left", "right", "both"], default="both",
                        help="Which camera to export frames for (default: both)")
    args = parser.parse_args()
    fn(load_config(args.config), camera=args.camera)

def main_step2():
    name, fn = STEPS[2]
    parser = argparse.ArgumentParser(description=f"Calibration step 2: {name}.")
    parser.add_argument("--config", required=True, help="Path to calibration YAML config")
    parser.add_argument("--camera", choices=["left", "right", "both"], default="both",
                        help="Which camera to calibrate (default: both)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.camera == "both":
        fn(cfg)
    else:
        from .intrinsics import run_intrinsics_one_camera
        run_intrinsics_one_camera(cfg, args.camera)

def main_step3():
    name, fn = STEPS[3]
    parser = argparse.ArgumentParser(description=f"Calibration step 3: {name}.")
    parser.add_argument("--config", required=True, help="Path to calibration YAML config")
    args = parser.parse_args()
    from .yaml_utils import update_yaml
    from swarm_assembly_methods.utils import resolve_config_path
    config_path = resolve_config_path(args.config)
    results = fn(load_config(config_path))
    if results:
        update_yaml(config_path, {"sweep_dk": {"results": {k: int(v) for k, v in results.items()}}})
        print(f"\nAuto-populated sweep_dk.results in {config_path}")

def main_step4():
    _run_single(4)

def main_step5():
    _run_single(5)

def main_step6():
    _run_single(6)


def _run_single(step_num: int):
    name, fn = STEPS[step_num]
    parser = argparse.ArgumentParser(description=f"Calibration step {step_num}: {name}.")
    parser.add_argument("--config", required=True, help="Path to calibration YAML config")
    args = parser.parse_args()
    fn(load_config(args.config))
