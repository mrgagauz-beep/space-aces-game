"""
Mini-map vision utilities for Space Aces Bot.

This module handles detection of player and enemies on the mini-map
using HSV thresholds and simple contour analysis.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import logging

from config.config import Config
from utils.logger import setup_logger
from vision.capture import grab_minimap

setup_logger()
logger = logging.getLogger(__name__)


def _get_hsv(cfg, key: str):
    if isinstance(cfg, Config):
        return cfg.hsv()[key]
    return cfg["HSV"][key]


def find_player_crosshair(minimap_bgr: np.ndarray) -> Optional[Tuple[int, int]]:
    """
    Detect the cyan player crosshair on the mini-map.

    Uses a rough HSV range for cyan and simple morphology to isolate
    the crosshair blob.
    """
    if minimap_bgr is None or minimap_bgr.size == 0:
        logger.warning("Empty minimap frame passed to find_player_crosshair")
        return None

    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)

    # Rough cyan range; can be moved to profile later if needed
    lower_cyan = np.array([80, 80, 120], dtype=np.uint8)
    upper_cyan = np.array([100, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_cyan, upper_cyan)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Choose largest cyan blob as player marker
    cnt = max(contours, key=cv2.contourArea)
    m = cv2.moments(cnt)
    if m["m00"] == 0:
        x, y, w, h = cv2.boundingRect(cnt)
        cx, cy = x + w // 2, y + h // 2
    else:
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])

    return cx, cy


def find_enemies(
    minimap_bgr: np.ndarray, cfg
) -> List[Tuple[int, int]]:
    """
    Detect enemy dots on the mini-map using red HSV thresholds.

    Uses two red ranges (around 0 and 179 in H) from config["HSV"].
    """
    if minimap_bgr is None or minimap_bgr.size == 0:
        logger.warning("Empty minimap frame passed to find_enemies")
        return []

    hsv = cv2.cvtColor(minimap_bgr, cv2.COLOR_BGR2HSV)

    lower1 = np.array(_get_hsv(cfg, "enemy_minimap_red_1"), dtype=np.uint8)
    upper1 = np.array(_get_hsv(cfg, "enemy_minimap_red_1_max"), dtype=np.uint8)
    lower2 = np.array(_get_hsv(cfg, "enemy_minimap_red_2"), dtype=np.uint8)
    upper2 = np.array(_get_hsv(cfg, "enemy_minimap_red_2_max"), dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    centers: List[Tuple[int, int]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 4:
            continue
        m = cv2.moments(cnt)
        if m["m00"] == 0:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
        else:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
        centers.append((cx, cy))

    return centers


if __name__ == "__main__":
    cfg = Config()
    minimap = grab_minimap(cfg)

    player = find_player_crosshair(minimap)
    enemies = find_enemies(minimap, cfg)

    logger.info("Sanity minimap: player=%s, enemies=%d", player, len(enemies))
