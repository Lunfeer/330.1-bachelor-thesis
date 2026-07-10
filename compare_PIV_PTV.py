import cv2, numpy as np
from PTV import run_ptv
from PIV import run_piv, WINDOW_SIZE, OVERLAP # type: ignore

SCALE_FACTOR = 10       # Target mean pixel length for arrow scaling
TIP_LENGTH = 0.3        # Arrow tip length ratio
LINE_THICK = 1          # Thickness of drawn lines

# Display PIV grid, PIV vectors, and PTV vectors side by side
def show_comparison(im_a, im_b, f_a, f_b):
    piv_x, piv_y, piv_dx, piv_dy = run_piv(im_a, im_b)
    ptv_x, ptv_y, ptv_dx, ptv_dy = run_ptv(f_a, f_b) # type: ignore

    print(f"PIV: {piv_x.shape[0]*piv_x.shape[1]} vectors  |  PTV: {len(ptv_x)} tracked particles")

    h, w = im_a.shape

    # The following code was heavily coded with ai tools (Chat GPT)
    mean_piv = np.mean(np.sqrt(piv_dx**2 + piv_dy**2))
    scale_piv = max(1, SCALE_FACTOR / mean_piv)
    mean_ptv = np.mean(np.sqrt(ptv_dx**2 + ptv_dy**2))
    scale_ptv = max(1, SCALE_FACTOR / mean_ptv)

    vis = cv2.cvtColor(im_a, cv2.COLOR_GRAY2BGR)
    step = WINDOW_SIZE - OVERLAP
    for gx in range(0, w, step):
        cv2.line(vis, (gx, 0), (gx, h), (0, 255, 0), LINE_THICK)
    for gy in range(0, h, step):
        cv2.line(vis, (0, gy), (w, gy), (0, 255, 0), LINE_THICK)

    piv_vis = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(piv_x.shape[0]):
        for j in range(piv_x.shape[1]):
            xi, yi = int(piv_x[i,j]), int(piv_y[i,j])
            ddx, ddy = int(piv_dx[i,j]*scale_piv), int(piv_dy[i,j]*scale_piv)
            cv2.arrowedLine(piv_vis, (xi, yi), (xi+ddx, yi+ddy), (0, 0, 255), LINE_THICK, tipLength=TIP_LENGTH)

    ptv_vis = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(ptv_x)):
        ddx, ddy = int(ptv_dx[i]*scale_ptv), int(ptv_dy[i]*scale_ptv)
        cv2.arrowedLine(ptv_vis, (int(ptv_x[i]), int(ptv_y[i])), (int(ptv_x[i])+ddx, int(ptv_y[i])+ddy), (255, 255, 0), LINE_THICK, tipLength=TIP_LENGTH)

    cv2.imshow("PIV GRID", vis)
    cv2.imshow("PIV", piv_vis)
    cv2.imshow("PTV", ptv_vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()