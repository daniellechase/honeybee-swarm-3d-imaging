import argparse
import yaml
from .pipeline import run_morphology_figures
from swarm_assembly_methods.utils import rename_config_with_run_id, resolve_config_path


def main():
    parser = argparse.ArgumentParser(description="Morphology figures from YAML config.")
    parser.add_argument("--config", required=True,
                        help="Path to figures/figmorph YAML config")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    config_path = rename_config_with_run_id(config_path, cfg)
    run_morphology_figures(cfg, config_path=config_path)
