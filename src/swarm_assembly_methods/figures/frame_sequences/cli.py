import argparse
import yaml
from .pipeline import run_frame_sequence_figures
from swarm_assembly_methods.utils import rename_config_with_run_id, resolve_config_path


def main():
    parser = argparse.ArgumentParser(
        description="Build camcorder frame-strip figures from a YAML config."
    )
    parser.add_argument("--config", required=True,
                        help="Path to frame-sequences YAML config")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    config_path = rename_config_with_run_id(config_path, cfg)
    run_frame_sequence_figures(cfg, config_path=config_path)
