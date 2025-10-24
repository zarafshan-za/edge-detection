# algorithms.py
# Edge detection functions used by the GUI.
# Requires: opencv-python, numpy

import cv2
import numpy as np

def ensure_odd(k):
    """Ensure kernel size is an odd integer >= 1."""
    k = int(k)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k -= 1
        if k < 1:
            k = 1
    return k

def to_display_bgr(edge_map):
    """
    Convert single-channel edge map (0..255) to 3-channel BGR for display.
    Accepts either bool or uint8 single-channel array.
    """
    if edge_map is None:
        return None
    if len(edge_map.shape) == 2:
        return cv2.cvtColor(edge_map, cv2.COLOR_GRAY2BGR)
    elif len(edge_map.shape) == 3 and edge_map.shape[2] == 1:
        return cv2.cvtColor(edge_map[:, :, 0], cv2.COLOR_GRAY2BGR)
    else:
        return edge_map

def sobel_edges(src_bgr, kernel_size=3, direction='both'):
    """
    Sobel edge detection.
    src_bgr: input BGR image (numpy array)
    kernel_size: odd int (1,3,5,7 ...)
    direction: 'x', 'y', or 'both'
    returns BGR image with edges drawn (uint8)
    """
    if src_bgr is None:
        return None
    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    k = ensure_odd(kernel_size)
    # If kernel 1, Sobel behaves like Scharr? But cv2 requires ksize in {1,3,5,7}
    k = max(1, k)
    # Compute grads
    dx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=k)
    dy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=k)
    # Convert to absolute then uint8
    abs_dx = cv2.convertScaleAbs(dx)
    abs_dy = cv2.convertScaleAbs(dy)
    if direction == 'x':
        edges = abs_dx
    elif direction == 'y':
        edges = abs_dy
    else:  # both
        edges = cv2.addWeighted(abs_dx, 0.5, abs_dy, 0.5, 0)
    return to_display_bgr(edges)

def laplacian_edges(src_bgr, kernel_size=3):
    """
    Laplacian edge detection.
    kernel_size must be odd and >=1 (cv2 supports 1,3,5,...)
    """
    if src_bgr is None:
        return None
    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    k = ensure_odd(kernel_size)
    # Use CV_16S to capture negative edges
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=k)
    lap_abs = cv2.convertScaleAbs(lap)
    return to_display_bgr(lap_abs)

def canny_edges(src_bgr, low_threshold=50, high_threshold=150, blur_ksize=5, sigma=1.0):
    """
    Canny edge detection with optional Gaussian blur.
    blur_ksize must be odd; if blur_ksize <= 1, no blur applied.
    sigma: float (Gaussian sigma)
    low_threshold, high_threshold: ints (0..255)
    """
    if src_bgr is None:
        return None
    gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
    k = ensure_odd(blur_ksize)
    if k > 1:
        # OpenCV requires odd ksize; ensure it's reasonable (1,3,5,7,...)
        gray = cv2.GaussianBlur(gray, (k, k), sigmaX=float(sigma), sigmaY=float(sigma))
    # Canny expects thresholds 0..255
    lt = int(max(0, min(255, low_threshold)))
    ht = int(max(0, min(255, high_threshold)))
    if ht < lt:
        # swap to make sense
        lt, ht = ht, lt
    edges = cv2.Canny(gray, lt, ht)
    return to_display_bgr(edges)

# small helper to resize image maintaining original processing resolution but returning scaled copy for display
def scale_for_display(img, display_w, display_h):
    """
    Scale BGR image (numpy) to fit within display_w x display_h while preserving aspect ratio.
    Returns a BGR numpy image (uint8) sized to fit inside the display box.
    """
    if img is None:
        return None
    h, w = img.shape[:2]
    # if already smaller/equal, we still scale to fit display box for consistent look
    scale = min(display_w / w, display_h / h)
    if scale <= 0:
        scale = 1.0
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized
