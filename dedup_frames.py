import cv2, numpy as np, os, sys
from pathlib import Path

THR_BRIGHT = 100    # Minimum brightess to consider a pixel as "particle"
DUP_DIFF = 3.0      # Max mean absolute difference to consider frames similar
DUP_NCC = 0.97      # Minimum NCC to consider frames similar

# Mean of absolute difference on bright pixels only
def _masked_diff(im_a, im_b):
    mask = (im_a > THR_BRIGHT) | (im_b > THR_BRIGHT)

    return np.mean(np.abs(im_a.astype(int) - im_b.astype(int))[mask])

# Normalized cross-correlation on bright pixels only
def _ncc_on_particles(im_a, im_b):
    mask = (im_a > THR_BRIGHT) | (im_b > THR_BRIGHT)
    n = mask.sum()
    if n < 10:
        return 1.0

    a = im_a[mask].astype(float)
    b = im_b[mask].astype(float)
    a -= a.mean()
    b -= b.mean()

    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom > 0:
        return np.sum(a * b) / denom
    else:
        return 1.0

# Remove duplicate frames based on image similarity and verify timestamps
def dedup_frames(frames_dir):
    paths = sorted(Path(frames_dir).glob("*.tif"))
    prev_im = None
    kept_path = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if prev_im is not None:
            masked_diff = _masked_diff(prev_im, im)
            ncc = _ncc_on_particles(prev_im, im)
            if masked_diff >= DUP_DIFF or ncc <= DUP_NCC:
                kept_path.append(p)
        else:
            kept_path.append(p)

        prev_im = im

    n_dup = len(paths) - len(kept_path)
    print(f"Duplicate images: {n_dup} on {len(paths)} images")
    for p in paths:
        if p not in kept_path:
            os.remove(p)

    return kept_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run dedup_frames.py <frames_dir>")
        sys.exit(1)

    dedup_frames(sys.argv[1])