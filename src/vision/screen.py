"""
Main screen vision utilities for Space Aces Bot.

This module is responsible for detecting enemies and bonus crates
on the main game screen using classical computer vision methods.
"""

from typing import List, Tuple

import cv2
import numpy as np
import logging

from config.config import Config
from utils.logger import setup_logger
from vision.capture import grab_main

setup_logger()
logger = logging.getLogger(__name__)


def _get_hsv(cfg, key: str):
    if isinstance(cfg, Config):
        return cfg.hsv()[key]
    return cfg["HSV"][key]


def find_crates_main(
    main_bgr: np.ndarray, cfg
) -> List[Tuple[int, int, int, int]]:
    """
    Detect bonus crates on the main screen using yellow HSV range.

    Returns list of bounding boxes (x, y, w, h).
    """
    if main_bgr is None or main_bgr.size == 0:
        logger.warning("Empty main frame passed to find_crates_main")
        return []

    hsv = cv2.cvtColor(main_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array(_get_hsv(cfg, "crate_main_yellow_min"), dtype=np.uint8)
    upper = np.array(_get_hsv(cfg, "crate_main_yellow_max"), dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))

    return boxes


def find_mobs_main(
    main_bgr: np.ndarray,
) -> List[Tuple[int, int, int, int]]:
    """
    Basic enemy detection on the main screen.

    Uses edge/gradient information and contour filtering as a simple
    placeholder before integrating a proper detector (e.g. YOLO).
    """
    if main_bgr is None or main_bgr.size == 0:
        logger.warning("Empty main frame passed to find_mobs_main")
        return []

    gray = cv2.cvtColor(main_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes: List[Tuple[int, int, int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, w, h))

    # TODO: integrate template / ML-based classifier (e.g., YOLO)
    # to filter out non-mob objects.
    return boxes


if __name__ == "__main__":
    cfg = Config()
    main = grab_main(cfg)

    crates = find_crates_main(main, cfg)
    mobs = find_mobs_main(main)

    logger.info(
        "Sanity main screen: crates=%d, mobs=%d", len(crates), len(mobs)
    )
