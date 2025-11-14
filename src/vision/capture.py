"""
Screen capture helpers for Space Aces Bot.

Provides thin wrappers around MSS to grab specific ROIs
as OpenCV-compatible BGR NumPy arrays.
"""

from typing import Tuple

import cv2
import mss
import numpy as np
import logging

from utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)


def capture_roi(roi: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Capture a screenshot of the given ROI.

    Parameters
    ----------
    roi : tuple[int, int, int, int]
        Region of interest as (x, y, width, height).

    Returns
    -------
    numpy.ndarray
        Captured image in BGR format.
    """
    x, y, w, h = roi
    region = {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}

    with mss.mss() as sct:
        sct_img = sct.grab(region)

    # MSS returns BGRA; convert to BGR for OpenCV
    img = np.array(sct_img)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img_bgr


def grab_main(cfg) -> np.ndarray:
    """
    Grab main game area using MAIN ROI from config.

    Config can be a Config object or a plain dict
    with structure cfg["ROI"]["MAIN"] = [x, y, w, h].
    """
    if hasattr(cfg, "roi_main"):
        roi = cfg.roi_main()
    else:
        roi = tuple(cfg["ROI"]["MAIN"])

    logger.debug("Grabbing MAIN ROI: %s", roi)
    return capture_roi(roi)


def grab_minimap(cfg) -> np.ndarray:
    """
    Grab minimap region using MINIMAP ROI from config.

    Config can be a Config object or a plain dict
    with structure cfg["ROI"]["MINIMAP"] = [x, y, w, h].
    """
    if hasattr(cfg, "roi_minimap"):
        roi = cfg.roi_minimap()
    else:
        roi = tuple(cfg["ROI"]["MINIMAP"])

    logger.debug("Grabbing MINIMAP ROI: %s", roi)
    return capture_roi(roi)


# Backwards-compatible helper if needed elsewhere
def capture_screen(roi_dict):
    """
    Backwards-compatible wrapper: roi as dict {x, y, width, height}.
    """
    roi = (
        roi_dict["x"],
        roi_dict["y"],
        roi_dict["width"],
        roi_dict["height"],
    )
    return capture_roi(roi)


if __name__ == "__main__":
    from config.config import Config

    cfg = Config()
    main = grab_main(cfg)
    mini = grab_minimap(cfg)

    logger.info("Sanity: MAIN frame shape: %s", getattr(main, "shape", None))
    logger.info("Sanity: MINIMAP frame shape: %s", getattr(mini, "shape", None))
