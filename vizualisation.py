import cv2, numpy as np, pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The following visualization functions was heavily coded with ai tools (Chat GPT)

MIN_BRIGHT = 80                # Minimum brightness for trajectory lines
MAX_BRIGHT = 255               # Maximum brightness for trajectory lines
SPEED_THRESHOLD = 0.05         # Speed below wich a step is considered zero
HIST_BINS = 200                # Number of bins for speed histogram
NX = 30                        # Grid cells in x for ensemble field
NY = 20                        # Grid cells in y for ensemble field
SCALE_VEC = 1.5                # Scale factor for quiver arrows

# Generate all PNG images from trajectories and velocities
def render_all(linked, vel_df, frame_shape, scale, out_dir):
    h, w = frame_shape
    vis_h, vis_w = h * scale, w * scale
    dir_img = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
    spd_img = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
    v_max = np.percentile(vel_df["speed_cm_s"], 99) if vel_df is not None and len(vel_df) > 0 else 1.0
    thick = max(1, scale)
    for p in linked["particle"].unique():
        traj = linked[linked["particle"] == p].sort_values("frame")
        pts = traj[["x", "y"]].values.astype(np.int32) * scale
        for i in range(len(pts) - 1):
            alpha = i / len(pts)
            bright = int(MIN_BRIGHT + (MAX_BRIGHT - MIN_BRIGHT) * alpha)
            dx = pts[i+1][0] - pts[i][0]
            dy = pts[i+1][1] - pts[i][1]
            angle = (np.degrees(np.arctan2(dy, dx)) + 180) % 360
            hue_dir = int(angle / 2)
            c_dir = cv2.cvtColor(np.uint8([[[hue_dir, 255, bright]]]), cv2.COLOR_HSV2BGR)[0, 0] # type: ignore
            cv2.line(dir_img, tuple(pts[i]), tuple(pts[i+1]), tuple(int(x) for x in c_dir), thick)
            if vel_df is not None:
                row = vel_df[(vel_df["particle"] == p) & (vel_df["frame"] == traj.iloc[i]["frame"])]
                hue_spd = 60
                if not row.empty:
                    v = row.iloc[0]["speed_cm_s"]
                    hue_spd = int(120 - 90 * min(v / v_max, 1.0))
            else:
                hue_spd = 60

            c_spd = cv2.cvtColor(np.uint8([[[hue_spd, 255, bright]]]), cv2.COLOR_HSV2BGR)[0, 0] # type: ignore
            cv2.line(spd_img, tuple(pts[i]), tuple(pts[i+1]), tuple(int(x) for x in c_spd), thick)

    cv2.putText(dir_img, "Color = movement direction", (10 * scale, 25 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (200, 200, 200), thick)
    cv2.putText(dir_img, "Brightness = time (brighter = later)", (10 * scale, 45 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (180, 180, 180), thick)
    cv2.putText(spd_img, "Color = speed (blue slow -> yellow fast)", (10 * scale, 25 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (200, 200, 200), thick)
    cv2.putText(spd_img, "Brightness = time (brighter = later)", (10 * scale, 45 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (180, 180, 180), thick)

    cv2.imwrite(str(out_dir / "trajectories_direction.png"), dir_img)
    cv2.imwrite(str(out_dir / "trajectories_speed.png"), spd_img)

    v = vel_df["speed_cm_s"].values
    p95 = np.percentile(v, 95)
    p99 = np.percentile(v, 99)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(v, bins=HIST_BINS, range=(0, min(v.max(), p99 * 1.5)), color="steelblue", edgecolor="none")
    ax.axvline(v.mean(), color="red", ls="--", lw=1.5, label=f"mean={v.mean():.2f}")
    ax.axvline(np.median(v), color="orange", ls="--", lw=1.5, label=f"median={np.median(v):.2f}")
    ax.axvline(p95, color="green", ls=":", lw=1.5, label=f"p95={p95:.2f}")
    ax.axvline(p99, color="purple", ls=":", lw=1.5, label=f"p99={p99:.2f}")
    ax.set_xlabel("speed (cm/s)")
    ax.set_ylabel("count")
    ax.set_title(f"Speed distribtion  (n={len(v)}, max={v.max():.2f} cm/s)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(str(out_dir / "speed_distribution.png"), dpi=150)
    plt.close(fig)
    _render_zero_steps(linked, vel_df, frame_shape, scale, out_dir)

# Draw trajectories with zero-speed steps from pre-computed data
def _render_zero_steps(linked, vel_df, frame_shape, scale, out_dir):
    zero_particles = set(vel_df[vel_df["speed_cm_s"] < SPEED_THRESHOLD]["particle"].unique())
    n_steps_total = len(vel_df)
    n_steps_zero = (vel_df["speed_cm_s"] < SPEED_THRESHOLD).sum()
    print(f"zero steps: {n_steps_zero}/{n_steps_total} steps, {len(zero_particles)} trajectories affected")

    h, w = frame_shape
    vis_h, vis_w = h * scale, w * scale
    img = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)
    thick = max(1, scale)
    n_zero = 0
    for p in linked["particle"].unique():
        if p not in zero_particles:
            continue

        traj = linked[linked["particle"] == p].sort_values("frame")
        pts = traj[["x", "y"]].values.astype(np.int32) * scale
        for i in range(len(pts) - 1):
            cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), (60, 60, 60), thick)
        for i in range(len(pts) - 1):
            row = vel_df[(vel_df["particle"] == p) & (vel_df["frame"] == traj.iloc[i]["frame"])]
            if not row.empty and row.iloc[0]["speed_cm_s"] < SPEED_THRESHOLD:
                cv2.line(img, tuple(pts[i]), tuple(pts[i+1]), (0, 0, 255), thick + 1)
                n_zero += 1

    cv2.putText(img, f"trajectories with zero steps: {len(zero_particles)}", (10 * scale, 30 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (255, 255, 255), thick)
    cv2.putText(img, f"{n_zero} zero steps (red)", (10 * scale, 55 * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * scale, (200, 200, 200), thick)

    cv2.imwrite(str(out_dir / "trajectories_zeros.png"), img)
