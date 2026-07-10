import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

MIN_SEGMENT_LEN = 8       # Minimum number of frames for a trajectory
MIN_TOTAL_DISP_PX = 5.0   # Minimum total displacement to keep a trajectory
OUTLIER_THRESH = 2.5      # Normalized residual treshold for outlier detection

# Clean trajectories: remove outlier steps, short segments, and low-range tracks
def clean_trajectories(linked):
    n_raw = int(linked["particle"].nunique())

    linked["vx"] = 0.0
    linked["vy"] = 0.0
    linked["outlier"] = False
    for p in linked["particle"].unique():
        idx = linked["particle"] == p
        traj = linked.loc[idx].sort_values("frame")
        for i in range(len(traj) - 1):
            frame = traj.iloc[i]["frame"]
            dx = float(traj.iloc[i + 1]["x"] - traj.iloc[i]["x"])
            dy = float(traj.iloc[i + 1]["y"] - traj.iloc[i]["y"])
            linked.loc[(linked["particle"] == p) & (linked["frame"] == frame), "vx"] = dx
            linked.loc[(linked["particle"] == p) & (linked["frame"] == frame), "vy"] = dy

    n_out = 0
    for frame in sorted(linked["frame"].unique()):
        fdata = linked[linked["frame"] == frame]
        if len(fdata) < 5:
            continue

        pts = fdata[["x", "y"]].values
        vels = fdata[["vx", "vy"]].values

        # Adapted from https://stackoverflow.com/questions/12374781 
        try:
            tri = Delaunay(pts)
        except Exception:
            print(f"Delaunay failed for frame {frame}")
            continue

        indptr, indices = tri.vertex_neighbor_vertices
        for i in range(len(fdata)):
            neighbors = indices[indptr[i]:indptr[i + 1]]
            if len(neighbors) < 2:
                continue

            nv = vels[list(neighbors)]
            med = np.array([float(np.median(nv[:, 0])), float(np.median(nv[:, 1]))])
            res = float(np.linalg.norm(vels[i] - med))
            loc = [float(np.linalg.norm(v - med)) for v in nv]
            rmed = float(np.median(loc))

            # Adapted from https://stackoverflow.com/questions/22354094
            rhat = rmed / 0.6745 if rmed > 0 else 0.0
            if rhat > 0 and res / rhat > OUTLIER_THRESH:
                idx = fdata.index[i]
                linked.loc[idx, "outlier"] = True
                n_out += 1

    clean = linked[~linked["outlier"]].copy()
    print(f"universal outlier: {n_out} steps removed")

    candidates = []
    for p in clean["particle"].unique():
        traj = clean[clean["particle"] == p].sort_values("frame")
        pts = traj[["x", "y"]].values.astype(np.float32)
        n = len(pts)
        if n < MIN_SEGMENT_LEN:
            continue

        x_range = float(np.max(pts[:, 0]) - np.min(pts[:, 0]))
        y_range = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        total_range = np.sqrt(x_range**2 + y_range**2)
        if total_range >= MIN_TOTAL_DISP_PX:
            candidates.append((pts, traj, n))
    if not candidates:
        return pd.DataFrame(), {"original": 0, "kept": 0, "outliers": n_out, "n_raw": n_raw}

    new_rows = []
    next_pid = 0
    kept = 0
    for pts, traj, n in candidates:
        for i, (_, row) in enumerate(traj.iterrows()):
            row = row.copy()
            row["x"] = float(pts[i, 0])
            row["y"] = float(pts[i, 1])
            row["particle"] = next_pid
            new_rows.append(row)

        next_pid += 1
        kept += 1

    result = pd.DataFrame(new_rows)
    stats = {"original": len(candidates), "kept": kept, "outliers": n_out, "n_raw": n_raw}

    return result, stats