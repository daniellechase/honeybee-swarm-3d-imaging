"""
I/O utilities for 2D tracking: load detections from .npy, save/load tracks as parquet.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def load_detections_npy(path):
    """
    Load detections from a .npy file with shape (N, 7).
    Columns used: 0=frame, 1=cx, 2=cy.

    Returns pd.DataFrame with columns ["t", "x", "y"].
    """
    arr = np.load(path)
    return pd.DataFrame({
        "t": arr[:, 0].astype(int),
        "x": arr[:, 1].astype(float),
        "y": arr[:, 2].astype(float),
    })


def save_tracks(tracks, path):
    """
    Save tracks dict to parquet with columns (track_id, frame, x, y, interpolated).

    Parameters
    ----------
    tracks : dict  {tid: [(frame, x, y) or (frame, x, y, interp), ...]}
    path   : str or Path  — .parquet extension added if not present
    """
    rows = []
    for tid, traj in tracks.items():
        for pt in traj:
            rows.append({
                "track_id":     int(tid),
                "frame":        int(pt[0]),
                "x":            float(pt[1]),
                "y":            float(pt[2]),
                "interpolated": bool(pt[3]) if len(pt) > 3 else False,
            })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["track_id", "frame", "x", "y", "interpolated"])
    path = Path(path).with_suffix(".parquet")
    df.to_parquet(path, index=False)
    return path


def load_tracks(path):
    """
    Load tracks from a parquet file saved by save_tracks().

    Returns dict  {tid: [(frame, x, y, interp), ...]}
    """
    df = pd.read_parquet(path)
    tracks = {}
    for row in df.itertuples(index=False):
        tracks.setdefault(row.track_id, []).append(
            (row.frame, row.x, row.y, int(row.interpolated))
        )
    return tracks
