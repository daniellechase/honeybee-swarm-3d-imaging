"""Config loading for figures pipeline."""

from pathlib import Path
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[4]  # src/swarm_assembly_methods/figures/figtraj → repo root


def load_config(yaml_path):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["_yaml_stem"] = Path(yaml_path).stem
    # Default data_root and output_root to repo root so all relative paths
    # (data/, figures/) resolve against the repo without explicit config entries.
    cfg.setdefault("data_root",   str(_REPO_ROOT))
    cfg.setdefault("output_root", str(_REPO_ROOT))
    return cfg


def _session_subpath(cfg):
    """
    Return a Path fragment  subject/date[/run_id]  from the session block.
    Falls back to empty Path if no session block is present.
    """
    sess = cfg.get("session", {})
    subject = sess.get("subject")
    date    = sess.get("date")
    if not subject or not date:
        return Path("")
    p = Path(str(subject)) / str(date)
    stereo_side = sess.get("stereo_side")
    if stereo_side:
        p = p / str(stereo_side)
    run_id = cfg.get("run_id") or sess.get("run_id")
    if run_id is not None and str(run_id).strip():
        p = p / str(run_id)
    return p


def get_tracks_dir(cfg):
    """
    Directory containing *_3d.parquet files.

    Resolution order:
      1. input.tracks_dir  — explicit absolute or data_root-relative path
      2. data_root / tracks_base / subject / date [/ run_id]
         where tracks_base defaults to "data/trajectories"

    Example (session-based):
        data_root:   "/mnt/peleg2/Danielle"
        session:     {subject: S02, date: "0722"}
        run_id:      13
        → /mnt/peleg2/Danielle/data/trajectories/S02/0722/13/
    """
    inp = cfg.get("input", {})
    data_root = cfg.get("data_root")

    explicit = inp.get("tracks_dir")
    if explicit:
        base = Path(explicit)
        out = Path(data_root) / base if (data_root and not base.is_absolute()) else base
    else:
        tracks_base = inp.get("tracks_base", "data/trajectories")
        out = Path(data_root or ".") / tracks_base / _session_subpath(cfg)

    if not out.exists():
        raise FileNotFoundError(f"tracks_dir not found: {out}")
    return out


def get_figures_dir(cfg):
    """
    Output directory for figures.

    Resolution order:
      1. output.figures_dir — explicit absolute or output_root-relative path
         (yaml stem is still appended as the final subfolder)
      2. output_root / figures_base / subject / date [/ run_id] / yaml_stem
         where figures_base defaults to "figures/quiver"

    Example (session-based):
        output_root:  "/mnt/peleg2/Danielle"
        session:      {subject: S02, date: "0722"}
        run_id:       13
        yaml file:    diagnostics_13.yaml
        → /mnt/peleg2/Danielle/figures/quiver/S02/0722/13/diagnostics_13/
    """
    out_cfg = cfg.get("output") or {}
    output_root = cfg.get("output_root")

    run_id = cfg.get("run_id") or (cfg.get("session") or {}).get("run_id")
    has_run_id = run_id is not None and str(run_id).strip()

    explicit = out_cfg.get("figures_dir")
    if explicit:
        base = Path(explicit)
        if output_root and not base.is_absolute():
            base = Path(output_root) / base
        out = base / cfg["_yaml_stem"] if not has_run_id else base
    else:
        figures_base = out_cfg.get("figures_base", "figures/quiver")
        out = Path(output_root or ".") / figures_base / _session_subpath(cfg)

    out.mkdir(parents=True, exist_ok=True)
    return out
