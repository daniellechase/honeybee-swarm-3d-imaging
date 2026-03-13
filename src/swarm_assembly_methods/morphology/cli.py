import argparse
from .config import load_config
from .pipeline import run_morphology
from swarm_assembly_methods.utils import rename_config_with_run_id, resolve_config_path


def main():
    parser = argparse.ArgumentParser(description="Swarm morphology analysis pipeline.")
    parser.add_argument("--config", required=True, help="Path to morphology YAML config")
    args = parser.parse_args()
    config_path = resolve_config_path(args.config)
    cfg = load_config(config_path)
    config_path = rename_config_with_run_id(config_path, cfg)
    run_morphology(cfg, config_path=config_path)
