import cv2, numpy as np, os, json
from dedup_frames import dedup_frames

WIN_TITLE = "Draw polygon"      # Window name for polygone selection
WIN_W = 1280                    # Window width
WIN_H = 720                     # Window heigth
CIRCLE_R = 5                    # Radius of drawn circles
LINE_THICK = 2                  # Thicknes of drawn lines
COLOR_GREEN = (0, 255, 0)       # BGR green for drawing

# Create and return the mask based on the points and the image shape
def _create_polygon_mask(image_shape, points):
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(mask, [pts], color=255)

    return mask

# Apply mask and replace zone outside of the mask by a color (black by default)
def _apply_mask(frame, mask, fill_value=0):
    if frame.ndim == 3:
        mask_3ch = cv2.merge([mask, mask, mask])
        result = np.where(mask_3ch == 255, frame, fill_value).astype(frame.dtype)
    else:
        result = np.where(mask == 255, frame, fill_value).astype(frame.dtype)

    return result

# Return image cropped to the polygone bounding rectangel
def _crop_to_mask_rectangle(frame, mask, points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x1, x2 = max(0, min(xs)), min(frame.shape[1], max(xs))
    y1, y2 = max(0, min(ys)), min(frame.shape[0], max(ys))
    cropped = frame[y1:y2, x1:x2]

    return cropped

# Extract only the blue channel from a BGR image
def _extract_blue_channel(frame):
    if frame.ndim == 3:
        return frame[:, :, 0]

    return frame

# Let the user draw a polygone on the first video frame and return the points
def _pick_points(video_path, frame_index=0):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    res, frame = cap.read()
    cap.release()

    points = []
    display = frame.copy()

    # Adapted from https://www.geeksforgeeks.org/python/handle-mouse-events-in-python-opencv/
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display, (x, y), CIRCLE_R, COLOR_GREEN, -1)
            if len(points) > 1:
                cv2.line(display, points[-2], points[-1], COLOR_GREEN, LINE_THICK)
            cv2.imshow(WIN_TITLE, display)
            print(f"Point {len(points)}: ({x}, {y})")

    cv2.namedWindow(WIN_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_TITLE, WIN_W, WIN_H)
    cv2.imshow(WIN_TITLE, display)
    cv2.setMouseCallback(WIN_TITLE, click)
    while True:
        key = cv2.waitKey(1)
        if key == 13:
            break
    if len(points) >= 3:
        cv2.polylines(display, [np.array(points)], isClosed=True, color=COLOR_GREEN, thickness=LINE_THICK)
        cv2.imshow(WIN_TITLE, display)
        cv2.waitKey(1000)
    else:
        raise RuntimeError(f"Please provide at least 3 points")

    cv2.destroyAllWindows()

    return points

# Convert video to tif images after going throught the mask and dedup pipeline
def process_video_with_mask(video_path, output_dir, blue_channel=True, remove_duplicates=False):
    points = _pick_points(video_path)

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    res, first_frame = cap.read()

    mask = _create_polygon_mask(first_frame.shape, points)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    n_frames = 0
    timestamps = []
    while True:
        t = cap.get(cv2.CAP_PROP_POS_MSEC)
        res, frame = cap.read()
        if res == False:
            break
        if blue_channel:
            frame = _extract_blue_channel(frame)

        frame_masked = _apply_mask(frame, mask)
        frame_croped = _crop_to_mask_rectangle(frame_masked, mask, points)

        cv2.imwrite(f"{output_dir}/frame_{n_frames:04d}.tif", frame_croped)
        timestamps.append(t)
        n_frames += 1

    cap.release()
    timestamps = np.array(timestamps)
    if timestamps[1] == timestamps[0]:
        timestamps[1] = 1000.0 / fps

    np.save(f"{output_dir}/timestamps.npy", timestamps)
    print(f"{n_frames} frames extracted")
    if remove_duplicates:
        dedup_frames(output_dir)

    return n_frames