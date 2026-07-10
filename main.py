import cv2, numpy as np
from pathlib import Path
from crop_tools import process_video_with_mask
from calibrate import calibrate, load_calibration
from detect import detect, show_detection
from compare_PIV_PTV import show_comparison
from vizualisation import render_all
from PTV import track_trajectories, trajectory_velocities, detect_duplicate_frames, remove_frames
from selection import clean_trajectories
from files_utils import load_frames
from evaluate import select_exit_line, classify_and_save

DIAM = 9                # Particle diameter in pixels for detection
MIN_INTENSITY = 100     # Minimum integrated brightness
ECC = 0.3               # Max eccentricity
SZ_MIN = 1.0            # Min radius after detection
SZ_MAX = 2.0            # Max radius after detection
DIST_THRESH_PX = 30     # Max distance from exit line for group A
SCALE = 4               # Upscale factor for output images
REMOVE_DUPLICATE = True # Try to remove duplicated frame during the extraction

CURRENT_DIR = Path(__file__).parent
VIDEO = Path(CURRENT_DIR) / "v3_slow.mp4"
REF_IMG = Path(CURRENT_DIR) / "before.tif"
OUT_DIR = Path(CURRENT_DIR) / "results"
FRAMES_DIR = Path(CURRENT_DIR) / "frames"
CALIB_FILE = Path(FRAMES_DIR) / "calibration.json"
TS_FILE = Path(FRAMES_DIR) / "timestamps.npy"

REF_HEIGHT = 25
REF_WIDTH = 29.5
FPS = 60

print("========= 1 CALIBRATION =========")
if not CALIB_FILE.exists():
    calibrate(REF_IMG, REF_WIDTH, REF_HEIGHT, str(CALIB_FILE))

H, W_CM, H_CM = load_calibration(str(CALIB_FILE))

print("========= 2 EXTRACTION =========")
if not TS_FILE.exists():
    n_frames = process_video_with_mask(video_path=VIDEO, output_dir=str(FRAMES_DIR), remove_duplicates=REMOVE_DUPLICATE)

ts = np.load(str(TS_FILE))

print("========= 3 DETECTION =========")
frames, orig_idx = load_frames(FRAMES_DIR)
im_a, im_b = frames[0], frames[1]

f_a = show_detection(im_a, f"Frame {orig_idx[0]}", diam=DIAM, min_intensity=MIN_INTENSITY, ecc=ECC, sz_min=SZ_MIN, sz_max=SZ_MAX)
f_b = show_detection(im_b, f"Frame {orig_idx[1]}", diam=DIAM, min_intensity=MIN_INTENSITY, ecc=ECC, sz_min=SZ_MIN, sz_max=SZ_MAX)

print("========= 4 PIV vs PTV =========")
show_comparison(im_a, im_b, f_a, f_b)

print("========= 5 EXIT LINE =========")
exit_line = select_exit_line(frames[0])

print("========= 6 FULL PTV =========")
detections = [detect(f) for f in frames]
print(f"{len(detections)} frames detected")

linked = track_trajectories(detections)
print(f"{linked['particle'].nunique()} trajectories (raw)")

linked, _ = clean_trajectories(linked)
print(f"{linked['particle'].nunique()} trajectories (cleaned)")

dup_frames = detect_duplicate_frames(linked)

if dup_frames:
    linked = remove_frames(linked, dup_frames)
    ids = sorted(int(f) for f in dup_frames)
    print(f"duplicate frames: {len(dup_frames)} ids={ids}")

print(f"{linked['particle'].nunique()} trajectories (no dup)")

vel_df = trajectory_velocities(linked, orig_idx, H, ts=ts, fps=FPS)
print(f"{len(vel_df)} velocity steps computed")

render_all(linked, vel_df, frames[0].shape, scale=SCALE, out_dir=OUT_DIR) # type: ignore

classify_and_save(vel_df, linked, exit_line, OUT_DIR, dist_thresh_px=DIST_THRESH_PX, scale=SCALE)
