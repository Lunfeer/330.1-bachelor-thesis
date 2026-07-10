import numpy as np
import cv2, pandas as pd
from pathlib import Path
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIN_TITLE = "Exit line"         # Window name for line selection
CIRCLE_R = 6                    # Radius of drawn circles
LINE_THICK = 2                  # Thicknes of drawn lines
COLOR_RED = (0, 0, 255)         # BGR red for drawing
COLOR_GRAY = (200, 200, 200)    # Gray text
TAIL_SIZE = 15                  # Number of last frames for correlation
MIN_TAIL = 8                    # Min frames required for correlation
P_THRESH = 0.05                 # P-value treshold for correlation
HIST_BINS = 25                  # Bins for correlation histogram

# Let the user click two points to define the exit line and return them
def select_exit_line(frame):
    print("Add two point to draw the exit line and press enter")

    img = frame.copy()
    pts = []

    # Adapted from https://www.geeksforgeeks.org/python/handle-mouse-events-in-python-opencv/
    def click(event, x, y, flags, param):
        nonlocal pts
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))
            vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for p in pts:
                cv2.circle(vis, p, CIRCLE_R, COLOR_RED, -1)
            if len(pts) == 2:
                cv2.line(vis, pts[0], pts[1], COLOR_RED, LINE_THICK)

            cv2.imshow(WIN_TITLE, vis)

    cv2.namedWindow(WIN_TITLE)
    cv2.setMouseCallback(WIN_TITLE, click)

    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.imshow(WIN_TITLE, vis)
    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == 13:
            break
        
    cv2.destroyAllWindows()

    return pts

# Perpendicular distance from point p to the infinite line (a, b)
def line_dist(p, a, b):
    x0, y0 = p
    x1, y1 = a
    x2, y2 = b

    return abs((x2 - x1) * (y1 - y0) - (x1 - x0) * (y2 - y1)) / np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# Split trajectories into groups: A = near exit line, B = the rest
def classify_trajectories(linked, exit_line_px, dist_thresh_px=30):
    a_px, b_px = exit_line_px
    categories = {}
    for p in linked["particle"].unique():
        traj = linked[linked["particle"] == p].sort_values("frame")
        last = traj.iloc[-1]
        if line_dist((last["x"], last["y"]), a_px, b_px) < dist_thresh_px:
            categories[p] = "A"
    for p in linked["particle"].unique():
        if p not in categories:
            categories[p] = "B"

    n_a = sum(1 for c in categories.values() if c == "A")
    n_b = sum(1 for c in categories.values() if c == "B")
    print(f"A (near exit): {n_a}   B (not studied): {n_b}")

    return categories

# Classify trajectories, compute A correlation stats, and save category image
def classify_and_save(vel_df, linked, exit_line_px, out_dir, dist_thresh_px=30, scale=4):
    categories = classify_trajectories(linked, exit_line_px, dist_thresh_px)

    a_corrs, n_too_short, n_pvalue = [], 0, 0
    for p in linked["particle"].unique():
        if categories[p] != "A":
            continue

        sub = vel_df[vel_df["particle"] == p].sort_values("frame")
        tail = sub.tail(TAIL_SIZE)
        if len(tail) < MIN_TAIL:
            n_too_short += 1
            continue

        corr, pval = pearsonr(tail["frame"], tail["speed_cm_s"])
        if pval >= P_THRESH: # type: ignore
            n_pvalue += 1
            continue

        a_corrs.append(corr)
    if a_corrs:
        print(f"A - correlation (tail {TAIL_SIZE}, p<{P_THRESH}): mean={np.mean(a_corrs):.2f}  std={np.std(a_corrs):.2f}  n={len(a_corrs)}")
        print(f"excluded: {n_too_short} (<{MIN_TAIL} frames)  {n_pvalue} (p>={P_THRESH})")

        plt.figure(figsize=(6, 4))
        plt.hist(a_corrs, bins=HIST_BINS, color="steelblue", edgecolor="white")
        plt.axvline(np.mean(a_corrs), color="red", ls="--", label=f'mean={np.mean(a_corrs):.2f}')
        plt.xlabel("Pearson r (speed vs frame)")
        plt.ylabel("Number of trajectories A")
        plt.title(f"Distribution of A correlations (tail {TAIL_SIZE} frames, p<{P_THRESH})")
        plt.legend()

        hist_path = str(Path(out_dir) / "a_correlation_hist.png")
        plt.savefig(hist_path, dpi=150, bbox_inches="tight")
        plt.close()

    # The following code was heavily coded with ai tools (Chat GPT)
    max_x = int(linked["x"].max()) + 10
    max_y = int(linked["y"].max()) + 10
    vis_h, vis_w = max_y * scale, max_x * scale
    vis = np.zeros((vis_h, vis_w, 3), dtype=np.uint8)

    colors = {"A": (255, 0, 0), "B": (0, 0, 255)}
    labels = {"A": "A - near exit (correlation)", "B": "B - not studied"}
    thick = max(1, scale)
    for p in linked["particle"].unique():
        cat = categories[p]
        traj = linked[linked["particle"] == p].sort_values("frame")
        pts = traj[["x", "y"]].values.astype(int) * scale

        cv2.polylines(vis, [pts], False, colors[cat], thick)
    for i, (k, lbl) in enumerate(labels.items()):
        cv2.putText(vis, f"{lbl} ({sum(1 for c in categories.values() if c == k)})",
                    (10 * scale, (30 + i * 25) * scale), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, colors[k], thick)

    out_path = str(Path(out_dir) / "trajectories_categories.png")
    cv2.imwrite(out_path, vis)

    return categories