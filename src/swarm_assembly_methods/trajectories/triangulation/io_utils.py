"""I/O for triangulation: load 2D tracks, save/load 3D trajectory DataFrames."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_tracks_parquet(path):
    """
    Load a _tracks.parquet file saved by tracking_2d.io_utils.save_tracks().

    Returns
    -------
    dict  {tid: np.ndarray shape (N, 3) columns [frame, x, y]}
    """
    df = pd.read_parquet(path)
    if df.empty:
        return {}
    tracks = {}
    for row in df.itertuples(index=False):
        tracks.setdefault(row.track_id, []).append((row.frame, row.x, row.y))
    return {tid: np.array(pts, dtype=float) for tid, pts in tracks.items()}


def save_tracks_3d(df, path):
    """Save 3D trajectory DataFrame to parquet."""
    path = Path(path).with_suffix(".parquet")
    df.to_parquet(path, index=False)
    return path


def load_tracks_3d(path):
    """Load 3D trajectory DataFrame from parquet."""
    return pd.read_parquet(path)
