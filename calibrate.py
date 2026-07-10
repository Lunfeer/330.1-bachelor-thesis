import cv2, numpy as np, json
from pathlib import Path

WIN_TITLE = "Calibration"       # Window name for point selection
WIN_W = 1280                    # Window width
WIN_H = 720                     # Window heigth
CIRCLE_R = 5                    # Radius of drawn circles
LINE_THICK = 2                  # Thicknes of drawn lines
COLOR_GREEN = (0, 255, 0)       # BGR green for drawing

# Pick 4 points on the reference image to define the calibration rectangle
def _pick_points(image_path):
    im = cv2.imread(str(image_path))

    points = []
    display = im.copy() # type: ignore

    # Adapted from https://www.geeksforgeeks.org/python/handle-mouse-events-in-python-opencv/
    def click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(display, (x, y), CIRCLE_R, COLOR_GREEN, -1)
            if len(points) > 1:
                cv2.line(display, points[-2], points[-1], COLOR_GREEN, LINE_THICK)
            if len(points) == 4:
                cv2.polylines(display, [np.array(points)], isClosed=True, color=COLOR_GREEN, thickness=LINE_THICK)
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

    cv2.destroyAllWindows()

    return np.array(points, dtype=np.float32)

# Save the homography matrix from the 4 calibration points
def calibrate(image_path, width_cm, height_cm, save_dir=None):
    pts = _pick_points(image_path)

    top = pts[np.argsort(pts[:, 1])][:2]
    bot = pts[np.argsort(pts[:, 1])][2:]
    top_left = top[np.argsort(top[:, 0])][0]
    top_right = top[np.argsort(top[:, 0])][1]
    bot_left = bot[np.argsort(bot[:, 0])][0]
    bot_right = bot[np.argsort(bot[:, 0])][1]
    src = np.array([top_left, top_right, bot_right, bot_left], dtype=np.float32)

    dst = np.array([[0, 0], [width_cm, 0], [width_cm, height_cm], [0, height_cm]], dtype=np.float32)

    H, _ = cv2.findHomography(src, dst)
    if save_dir:
        data = {"width_cm": width_cm, "height_cm": height_cm, "H": H.tolist()}
        Path(save_dir).write_text(json.dumps(data, indent=2))

    print("Homography matrix")
    print(f"{H[0,0]}  {H[0,1]}  {H[0,2]}")
    print(f"{H[1,0]}  {H[1,1]}  {H[1,2]}")
    print(f"{H[2,0]}  {H[2,1]}  {H[2,2]}")

    return H

# Load a previously saved calibration from a JSON file
def load_calibration(path):
    data = json.loads(Path(path).read_text())

    return np.array(data["H"], dtype=np.float32), data["width_cm"], data["height_cm"]