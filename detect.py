import cv2, numpy as np, warnings
from pathlib import Path
import trackpy as tp
warnings.filterwarnings("ignore")

DIAM = 9                # bandpass filter window (particle diameter in px)
MIN_INTENSITY = 100     # minimum integrated brightness (trackpy minmass)
ECC = 0.3               # max eccentricity (0 = circle and 1 = line)
SZ_MIN = 1.0            # min radius particle after detection
SZ_MAX = 2.0            # max radius particle after detection

# Using trackpy to detect particle and filter them before returning the result
def detect(frame, diam=DIAM, min_intensity=MIN_INTENSITY, ecc=ECC, sz_min=SZ_MIN, sz_max=SZ_MAX):
    f = tp.locate(frame, diameter=diam, minmass=min_intensity, invert=False, preprocess=True)
    f = f[(f.ecc < ecc) & (f["size"] >= sz_min) & (f["size"] <= sz_max)]
    
    return f

# Display the particles after detection 
def show_detection(frame, frame_label="frame", diam=DIAM, min_intensity=MIN_INTENSITY, ecc=ECC, sz_min=SZ_MIN, sz_max=SZ_MAX):
    f = detect(frame, diam, min_intensity, ecc, sz_min, sz_max)
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for i, p in f.iterrows():
        cv2.circle(vis, (int(p["x"]), int(p["y"])), 3, (0, 0, 255), 1)

    cv2.imshow(frame_label, vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return f
