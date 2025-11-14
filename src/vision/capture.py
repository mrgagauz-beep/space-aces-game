import mss
import cv2
import numpy as np


def _get_roi_value(roi, key):
    """
    Extract coordinate value from roi which can be a dict or an object
    with attributes x, y, width, height.
    """
    if isinstance(roi, dict):
        return roi[key]
    return getattr(roi, key)


def capture_screen(roi):
    """
    Capture a screenshot of the specified region of the screen.

    Parameters
    ----------
    roi : dict or object
        Region of interest with fields/keys: x, y, width, height.

    Returns
    -------
    numpy.ndarray
        Image in BGR format suitable for OpenCV.
    """
    region = {
        "left": _get_roi_value(roi, "x"),
        "top": _get_roi_value(roi, "y"),
        "width": _get_roi_value(roi, "width"),
        "height": _get_roi_value(roi, "height"),
    }

    with mss.mss() as sct:
        sct_img = sct.grab(region)

    # mss returns image in BGRA format; convert to BGR for OpenCV
    img = np.array(sct_img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img_bgr

