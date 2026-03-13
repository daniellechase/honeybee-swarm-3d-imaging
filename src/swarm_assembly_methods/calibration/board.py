"""Calibration board geometry — tag sizes, grid layout, object points."""

from typing import NamedTuple

import numpy as np


class BoardParams(NamedTuple):
    tag_size_mm: float
    spacing_mm: float
    rows: int
    cols: int
    id_offset: int


_BOARDS = {
    "large": BoardParams(tag_size_mm=75, spacing_mm=40, rows=4, cols=5, id_offset=0),
    "small": BoardParams(tag_size_mm=49, spacing_mm=11, rows=3, cols=4, id_offset=0),
}


def get_board_params(board_type: str) -> BoardParams:
    if board_type not in _BOARDS:
        raise ValueError(f"Unknown board_type '{board_type}'. Choose from: {list(_BOARDS)}")
    return _BOARDS[board_type]


def grid_object_pts(tag_id: int, board: BoardParams) -> np.ndarray:
    """Return the 4 corner 3-D positions (mm) for *tag_id* on *board*."""
    idx = tag_id - board.id_offset
    r, c = divmod(idx, board.cols)
    bx = c * (board.tag_size_mm + board.spacing_mm)
    by = r * (board.tag_size_mm + board.spacing_mm)
    s = board.tag_size_mm
    return np.array(
        [[bx, by, 0.0], [bx + s, by, 0.0], [bx + s, by + s, 0.0], [bx, by + s, 0.0]],
        dtype=np.float32,
    )
