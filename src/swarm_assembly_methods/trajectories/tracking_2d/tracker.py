"""
2D tracking: SimpleTracker (Hungarian-assignment tracker with velocity prediction)
and track analysis utilities.
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


class SimpleTracker:
    def __init__(self, max_dist=10, max_frame_skip=1, use_velocity=True,
                 angle_penalty_weight=0.0, min_speed_for_angle=2.0,
                 vel_ema_alpha=0.3):
        """
        Parameters
        ----------
        max_dist : float
            Maximum pixel distance from prediction to accept a match.
        max_frame_skip : int
            Frames a track can survive without a detection.
        use_velocity : bool
            Use smoothed velocity for prediction and angle penalty.
        angle_penalty_weight : float
            Extra cost (in pixels) added for a full 180° direction reversal.
            0 = disabled (original behavior).  A value of 40-80 works well
            for typical bee tracking — sharp turns get penalized, making
            the tracker prefer continuing in the current direction even if
            another bee is slightly closer.
            The penalty scales smoothly: 0 for same direction, weight/2 for
            a 90° turn, full weight for a 180° reversal.
        min_speed_for_angle : float
            Minimum speed (px/frame) below which the angle penalty is not
            applied.  Avoids penalizing near-stationary bees that have no
            meaningful direction.
        vel_ema_alpha : float
            EMA smoothing factor for velocity (0–1). Higher = faster adaptation,
            lower = smoother velocity estimate.
        """
        self.max_dist = float(max_dist)
        self.max_frame_skip = int(max_frame_skip)
        self.use_velocity = bool(use_velocity)
        self.angle_penalty_weight = float(angle_penalty_weight)
        self.min_speed_for_angle = float(min_speed_for_angle)
        self.vel_ema_alpha = float(vel_ema_alpha)

        self.tracks = {}         # tid -> list of (frame, x, y)
        self.active_tracks = {}  # tid -> last frame seen
        self.vel_state = {}      # tid -> smoothed (vx, vy) in px/frame
        self.next_id = 0
        self.pred_residuals = [] # ||detection - prediction|| at each match (px)

    def _predict(self, tid, traj, frame_num):
        """
        Predict (x,y) at frame_num using the smoothed velocity.
        Falls back to last position if no velocity is available.
        """
        last_f, last_x, last_y = traj[-1]
        dt = frame_num - last_f
        if not self.use_velocity or dt <= 0:
            return last_x, last_y
        vx, vy = self.vel_state.get(tid, (0.0, 0.0))
        return last_x + vx * dt, last_y + vy * dt

    def update(self, points, frame_num):
        """
        points: array-like of shape (N,2) for this frame in pixel coords
        Hungarian matching version: optimal assignment of tracks to detections.
        """
        points = np.asarray(points, float)
        to_terminate = []

        # --- build list of active track IDs and their predictions ---
        active_tids = []
        predictions = []

        for tid in list(self.active_tracks.keys()):
            traj = self.tracks[tid]
            last_f, last_x, last_y = traj[-1]

            if frame_num - last_f > self.max_frame_skip:
                to_terminate.append(tid)
                continue

            px, py = self._predict(tid, traj, frame_num)
            active_tids.append(tid)
            predictions.append([px, py])

        # --- terminate tracks that exceeded skip ---
        for tid in to_terminate:
            self.active_tracks.pop(tid, None)

        # --- build cost matrix (num_tracks x num_points) ---
        n_tracks = len(active_tids)
        n_points = len(points)

        if n_tracks == 0:
            # No active tracks, spawn new ones for all points
            for i, (x, y) in enumerate(points):
                tid = self.next_id
                self.tracks[tid] = [(frame_num, float(x), float(y))]
                self.active_tracks[tid] = frame_num
                self.next_id += 1
            return

        if n_points == 0:
            # No detections this frame; tracks survive (if not exceeding skip)
            return

        # Cost = Euclidean distance from prediction + optional angle penalty
        predictions = np.asarray(predictions, float)   # (n_tracks, 2)

        # --- KD-tree pre-filter: drop tracks with no detection within max_dist ---
        tree = cKDTree(points)
        nearby = tree.query_ball_point(predictions, r=self.max_dist)  # list of lists

        # Collect only (track, point) pairs that are within max_dist
        pairs_i, pairs_j = [], []
        for i, nbrs in enumerate(nearby):
            for j in nbrs:
                pairs_i.append(i)
                pairs_j.append(j)

        cost_matrix = np.full((n_tracks, n_points), 1e9, dtype=float)

        if pairs_i:
            pi = np.array(pairs_i)
            pj = np.array(pairs_j)

            # Euclidean distance for all candidate pairs
            diffs = predictions[pi] - points[pj]            # (K, 2)
            dists = np.hypot(diffs[:, 0], diffs[:, 1])      # (K,)

            # Angle penalty (vectorised)
            use_angle = self.use_velocity and self.angle_penalty_weight > 0
            if use_angle:
                # Use smoothed velocity for direction — not raw last-two-point diff
                last_pos = np.empty((n_tracks, 2), dtype=float)
                vel_xy   = np.zeros((n_tracks, 2), dtype=float)
                vel_spd  = np.zeros(n_tracks, dtype=float)
                for i, tid in enumerate(active_tids):
                    traj = self.tracks[tid]
                    last_pos[i] = (traj[-1][1], traj[-1][2])
                    vx_c, vy_c = self.vel_state.get(tid, (0.0, 0.0))
                    vel_xy[i]  = (vx_c, vy_c)
                    vel_spd[i] = np.hypot(vx_c, vy_c)

                # Direction from last position → candidate detection
                dx_new = points[pj, 0] - last_pos[pi, 0]   # (K,)
                dy_new = points[pj, 1] - last_pos[pi, 1]   # (K,)
                spd_new = np.hypot(dx_new, dy_new)          # (K,)

                trk_spd = vel_spd[pi]                       # (K,)
                apply_mask = (trk_spd > 1e-9) & (spd_new > 1e-9) & \
                             (trk_spd >= self.min_speed_for_angle) & \
                             (spd_new >= self.min_speed_for_angle)

                if apply_mask.any():
                    vx_k = vel_xy[pi[apply_mask], 0]
                    vy_k = vel_xy[pi[apply_mask], 1]
                    dx_k = dx_new[apply_mask]
                    dy_k = dy_new[apply_mask]
                    cos_a = (vx_k * dx_k + vy_k * dy_k) / \
                            (trk_spd[apply_mask] * spd_new[apply_mask])
                    cos_a = np.clip(cos_a, -1.0, 1.0)
                    dists[apply_mask] += self.angle_penalty_weight * (1.0 - cos_a) / 2.0

            cost_matrix[pi, pj] = dists

        # Replace any NaN/inf that slipped through with a large cost
        cost_matrix = np.where(np.isfinite(cost_matrix), cost_matrix, 1e9)

        # --- solve assignment problem ---
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assigned_points = set()

        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] <= self.max_dist:
                tid = active_tids[i]
                x, y = float(points[j, 0]), float(points[j, 1])
                px, py = predictions[i]
                self.pred_residuals.append(float(np.hypot(x - px, y - py)))
                traj = self.tracks[tid]
                last_f, last_x, last_y = traj[-1]
                dt = frame_num - last_f
                # Update velocity with EMA
                if self.use_velocity and dt > 0:
                    vx, vy = self.vel_state.get(tid, (0.0, 0.0))
                    raw_vx = (x - last_x) / dt
                    raw_vy = (y - last_y) / dt
                    self.vel_state[tid] = (
                        (1.0 - self.vel_ema_alpha) * vx + self.vel_ema_alpha * raw_vx,
                        (1.0 - self.vel_ema_alpha) * vy + self.vel_ema_alpha * raw_vy,
                    )
                traj.append((frame_num, x, y))
                self.active_tracks[tid] = frame_num
                assigned_points.add(j)

        # --- spawn new tracks for unmatched points ---
        for j, (x, y) in enumerate(points):
            if j not in assigned_points:
                tid = self.next_id
                self.tracks[tid] = [(frame_num, float(x), float(y))]
                self.active_tracks[tid] = frame_num
                self.next_id += 1


def track_2d_points(
    df_pts,
    t_col="t",
    t_start=None,
    t_end=None,
    min_len=0,
    max_dist=25,
    max_frame_skip=8,
    use_velocity=True,
    angle_penalty_weight=0.0,
    min_speed_for_angle=2.0,
    vel_ema_alpha=0.3,
    verbose=True,
    camera_name="",
):
    tracker = SimpleTracker(
        max_dist=max_dist,
        max_frame_skip=max_frame_skip,
        use_velocity=use_velocity,
        angle_penalty_weight=angle_penalty_weight,
        min_speed_for_angle=min_speed_for_angle,
        vel_ema_alpha=vel_ema_alpha,
    )

    # ---- restrict time window ----
    if t_start is not None:
        df_pts = df_pts[df_pts[t_col] >= t_start]
    if t_end is not None:
        df_pts = df_pts[df_pts[t_col] <= t_end]

    if len(df_pts) == 0:
        return {}, np.array([])

    # Ensure integer frame index (important for skip logic)
    # If your t_col is already int frames, this is harmless.
    df_pts = df_pts.copy()
    df_pts[t_col] = df_pts[t_col].astype(int)

    # ---- pre-group (critical for speed) ----
    pts_by_frame = {
        int(f): g[["x", "y"]].to_numpy(dtype=float)
        for f, g in df_pts.groupby(t_col, sort=False)
    }
    frames = sorted(pts_by_frame.keys())

    for f in frames:
        tracker.update(pts_by_frame[f], f)

    # ---- keep only real tracks ----
    tracks = {
        tid: traj
        for tid, traj in tracker.tracks.items()
        if len(traj) >= min_len
    }

    if verbose and tracks:
        lengths = np.array([len(t) for t in tracks.values()])
        tag = f"[{camera_name}] " if camera_name else ""
        print(f"  {tag}2D tracks: n={len(tracks)}, "
              f"len p50={np.median(lengths):.0f} p75={np.percentile(lengths,75):.0f} "
              f"p90={np.percentile(lengths,90):.0f} max={lengths.max()}")

    return tracks, np.array(tracker.pred_residuals)


def interpolate_track_gaps(tracks):
    """
    Fill frame gaps in 2D tracks with linearly interpolated points.

    If a track jumps from frame 100 to frame 103, this inserts interpolated
    (x, y) entries for frames 101 and 102.  This ensures continuous per-frame
    coverage for downstream stereo matching / 3D triangulation.

    Parameters
    ----------
    tracks : dict  {tid: [(frame, x, y), ...]}

    Returns
    -------
    dict  {tid: [(frame, x, y), ...]}  — new dict with gaps filled.
    """
    out = {}
    for tid, traj in tracks.items():
        if len(traj) <= 1:
            out[tid] = list(traj)
            continue

        # Unpack either (frame, x, y) or (frame, x, y, interp) tuples
        def _as4(pt):
            return (pt[0], pt[1], pt[2], pt[3] if len(pt) > 3 else 0)

        filled = [_as4(traj[0])]
        for i in range(1, len(traj)):
            f_prev, x_prev, y_prev, _ = _as4(traj[i - 1])
            f_curr, x_curr, y_curr, flag_curr = _as4(traj[i])
            gap = int(f_curr - f_prev)
            if gap > 1:
                for g in range(1, gap):
                    alpha = g / gap
                    f_interp = int(f_prev + g)
                    x_interp = x_prev + alpha * (x_curr - x_prev)
                    y_interp = y_prev + alpha * (y_curr - y_prev)
                    filled.append((f_interp, float(x_interp), float(y_interp), 1))
            filled.append((f_curr, x_curr, y_curr, flag_curr))
        out[tid] = filled
    return out


def compute_tracked_displacements(tracks, camera_name="Camera"):
    """
    Compute real per-frame displacement from tracked trajectories.
    Only uses consecutive frames (dt=1) for accurate measurement.

    NOTE on max_dist:
      max_dist is the tolerance between the *predicted* position and the
      actual detection — NOT the raw displacement.  With velocity prediction,
      the residual error is much smaller than the displacement itself.
      Use this output to set an upper bound for max_dist (no-prediction case),
      or tune max_dist empirically by inspecting tracking quality.
    """
    displacements = []
    for tid, traj in tracks.items():
        for i in range(1, len(traj)):
            t_prev, x_prev, y_prev = traj[i-1]
            t_curr, x_curr, y_curr = traj[i]
            dt = t_curr - t_prev
            if dt == 1:  # consecutive frames only
                d = np.hypot(x_curr - x_prev, y_curr - y_prev)
                displacements.append(d)
    displacements = np.array(displacements)

    if len(displacements) == 0:
        print(f"[{camera_name}] No consecutive-frame displacements found!")
        return displacements

    p50, p90, p95, p99 = np.percentile(displacements, [50, 90, 95, 99])
    print(f"[{camera_name}] Tracked per-frame displacement (consecutive frames only):")
    print(f"  N samples: {len(displacements)}")
    print(f"  p50={p50:.1f} px, p90={p90:.1f} px, p95={p95:.1f} px, p99={p99:.1f} px")
    print(f"  max={displacements.max():.1f} px")
    print(f"  max_dist upper bound (no prediction) ≈ p99 = {p99:.1f} px")
    print(f"  max_dist with velocity prediction: tune empirically — prediction residual << displacement")

    return displacements


def analyze_2d_tracks(tracks, camera_name="Camera"):
    """
    Compute statistics on 2D trajectories.
    tracks: dict {tid: [(frame, x, y), ...]}
    """
    lengths = np.array([len(traj) for traj in tracks.values()])

    if len(lengths) == 0:
        print(f"[{camera_name}] No tracks found!")
        return

    print(f"\n[{camera_name}] 2D Trajectory Statistics:")
    print(f"  Total tracks: {len(tracks)}")
    print(f"  Length - min: {lengths.min()}, max: {lengths.max()}, mean: {lengths.mean():.1f}, median: {np.median(lengths):.1f}")
    print(f"  Percentiles - p25: {np.percentile(lengths, 25):.1f}, p50: {np.percentile(lengths, 50):.1f}, p75: {np.percentile(lengths, 75):.1f}, p90: {np.percentile(lengths, 90):.1f}")

    # Frame span per trajectory
    frame_spans = []
    for traj in tracks.values():
        frames = [pt[0] for pt in traj]
        frame_spans.append(max(frames) - min(frames))
    frame_spans = np.array(frame_spans)
    print(f"  Frame span - min: {frame_spans.min()}, max: {frame_spans.max()}, mean: {frame_spans.mean():.1f}")

    return lengths, frame_spans
