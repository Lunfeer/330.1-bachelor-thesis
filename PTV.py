import cv2, numpy as np
import trackpy as tp, pandas as pd
tp.ignore_logging()

SEARCH_RANGE = 15               # Max distance for linking particles between frames
OUTLIER_FACTOR = 3              # Multiplier on median for outlier removal
MIN_GLOBAL_DISP_PX = 0.1        # Min mean displacement to consider frames distinct

# Link particles between two frames and return position + displacement vectors
def run_ptv(ptc_frame_1, ptc_frame_2, search_range=SEARCH_RANGE):
    ptc_frame_1 = ptc_frame_1.copy()
    ptc_frame_2 = ptc_frame_2.copy()
    ptc_frame_1["frame"] = 0
    ptc_frame_2["frame"] = 1

    linked = tp.link_df(pd.concat([ptc_frame_1, ptc_frame_2], ignore_index=True), search_range=search_range, link_strategy="auto")

    ptcX, ptcY, dx, dy = [], [], [], []
    for p in linked["particle"].unique():
        pair = linked[linked["particle"] == p]

        ptc0 = pair[pair["frame"] == 0]
        ptc1 = pair[pair["frame"] == 1]
        if len(ptc0) != 1 or len(ptc1) != 1:
            continue

        x0 = ptc0.iloc[0]["x"]
        y0 = ptc0.iloc[0]["y"]
        x1 = ptc1.iloc[0]["x"]
        y1 = ptc1.iloc[0]["y"]
        ptcX.append(x0)
        ptcY.append(y0)
        dx.append(x1 - x0)
        dy.append(y1 - y0)

    ptcX, ptcY, dx, dy = np.array(ptcX), np.array(ptcY), np.array(dx), np.array(dy)
    if len(dx) > 3:
        norms = np.sqrt(dx**2 + dy**2)
        med = np.median(norms)

        if med < 0.1:
            print("Duplicate frames detected")

            return np.array([]), np.array([]), np.array([]), np.array([])

        mask = norms < OUTLIER_FACTOR * med
        ptcX, ptcY = ptcX[mask], ptcY[mask]
        dx, dy = dx[mask], dy[mask]

    return ptcX, ptcY, dx, dy

# Link detections across multiple frames into full trajectories
def track_trajectories(ptc_frames_list, search_range=SEARCH_RANGE):
    for i, d in enumerate(ptc_frames_list):
        d = d.copy()
        d["frame"] = i
        ptc_frames_list[i] = d

    linked = tp.link_df(pd.concat(ptc_frames_list, ignore_index=True), search_range=search_range, link_strategy="auto")

    return linked

# Detect duplicate frames (mean displacement < threshold) and return their indices
def detect_duplicate_frames(linked):
    all_frames = sorted(linked["frame"].unique())
    dup_frames = set()
    for i in range(len(all_frames) - 1):
        f_a, f_b = all_frames[i], all_frames[i + 1]
        dfa = linked[linked["frame"] == f_a]
        dfb = linked[linked["frame"] == f_b]
        common = pd.merge(dfa, dfb, on="particle", suffixes=("_a", "_b"))

        dx = common["x_b"].values - common["x_a"].values    # type: ignore
        dy = common["y_b"].values - common["y_a"].values    # type: ignore
        mean_disp = float(np.mean(np.sqrt(dx**2 + dy**2)))
        if mean_disp < MIN_GLOBAL_DISP_PX:
            dup_frames.add(f_a)

    return dup_frames

# Remove frames from the DataFrame and reassign particle IDs
def remove_frames(linked, frames_to_remove):
    kept = linked[~linked["frame"].isin(frames_to_remove)].copy()

    old_pids = kept["particle"].unique()
    mapping = {old: new for new, old in enumerate(old_pids)}
    kept["particle"] = kept["particle"].map(mapping)

    return kept

# Compute velocities from trajectories using homography and timestamps
def trajectory_velocities(linked, orig_idx, H, ts, fps=60):
    results = []
    for p in linked["particle"].unique():
        traj = linked[linked["particle"] == p].sort_values("frame")
        frames = traj["frame"].values
        xy = traj[["x", "y"]].values.astype(np.float32)
        for i in range(len(traj) - 1):
            f_a = int(frames[i])
            f_b = int(frames[i + 1])
            idx_a, idx_b = orig_idx[f_a], orig_idx[f_b]
            gap_s = (ts[idx_b] - ts[idx_a]) / 1000.0
            if gap_s <= 0:
                continue

            pts = np.array([[xy[i]], [xy[i+1]]], dtype=np.float32)
            cm = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
            dx_cm, dy_cm = cm[0, 0] - cm[1, 0], cm[0, 1] - cm[1, 1]
            v = np.sqrt(dx_cm**2 + dy_cm**2) / gap_s
            vx = (cm[1, 0] - cm[0, 0]) / gap_s
            vy = (cm[1, 1] - cm[0, 1]) / gap_s

            results.append({"particle": p, "frame": f_a, "x_cm": float(cm[0,0]), "y_cm": float(cm[0,1]), "vx_cm_s": float(vx), "vy_cm_s": float(vy), "speed_cm_s": float(v)})

    return pd.DataFrame(results)