from openpiv import pyprocess as piv, validation as piv_val, filters as piv_filt

WINDOW_SIZE = 32    # Size of interrogation windows in pixels
OVERLAP = 16        # Overlap between windows
SEARCH_AREA = 48    # Search area size for cross-correlation

# Run PIV cross-correlation between two frames, return the vectors
def run_piv(frame_a, frame_b):
    dx, dy, sig = piv.extended_search_area_piv(frame_a, frame_b, window_size=WINDOW_SIZE, overlap=OVERLAP, dt=1, search_area_size=SEARCH_AREA)

    flags = piv_val.global_val(dx, dy, (-10, 10), (-10, 10))
    dx, dy = piv_filt.replace_outliers(dx, dy, flags, method="localmean", max_iter=3, kernel_size=2)[:2]
    x, y = piv.get_coordinates(frame_a.shape, search_area_size=SEARCH_AREA, overlap=OVERLAP)

    return x, y, dx, dy
