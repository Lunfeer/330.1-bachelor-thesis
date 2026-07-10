import cv2, numpy as np
from pathlib import Path

# Return (images list, original_indexes) from a .tif folder sorted by index
def load_frames(frames_dir):
    paths = sorted(Path(frames_dir).glob("*.tif"), key=lambda p: int(p.stem.split("_")[1]))
    orig_idx = np.array([int(p.stem.split("_")[1]) for p in paths])
    imgs = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in paths]

    return imgs, orig_idx
