"""
Navigation helpers for clicking on the mini-map and waiting for arrival.

This module encapsulates basic pathing logic on the mini-map:
we click near a target position and poll mini-map frames until
the player crosshair stops moving.
"""

from __future__ import annotations

import time
from math import sqrt
from typing import Callable, Tuple

import numpy as np
import logging

from controls.input import click_in_roi
from utils.logger import setup_logger
from vision.minimap import find_player_crosshair

setup_logger()
logger = logging.getLogger(__name__)


def go_to_minimap_point(
    minimap_roi: Tuple[int, int, int, int],
    target_xy: Tuple[int, int],
    poll_fn: Callable[[], np.ndarray],
    timeout: float = 8.0,
) -> bool:
    """
    Click a target point on the mini-map and wait until the ship arrives.

    We deliberately click *near* the target (via click jitter) rather than
    exactly on the same pixel to avoid pattern-like behavior.

    Parameters
    ----------
    minimap_roi : tuple[int, int, int, int]
        Mini-map ROI in screen coordinates (x, y, w, h).
    target_xy : tuple[int, int]
        Target point in mini-map local coordinates.
    poll_fn : Callable[[], np.ndarray]
        Function returning a fresh full-screen BGR frame each poll.
    timeout : float
        Maximum time to wait in seconds.

    Returns
    -------
    bool
        True if arrival detected before timeout, otherwise False.
    """
    logger.info("Nav: go_to_minimap_point roi=%s target=%s", minimap_roi, target_xy)

    # Initial click towards the target; click_in_roi will add jitter.
    click_in_roi(minimap_roi, target_xy)

    start = time.time()
    last_pos = None
    still_counter = 0
    still_threshold = 3  # consecutive frames within 2px considered "stopped"

    while time.time() - start < timeout:
        frame = poll_fn()
        if frame is None or frame.size == 0:
            time.sleep(0.2)
            continue

        x, y, w, h = minimap_roi
        minimap = frame[y : y + h, x : x + w]
        me = find_player_crosshair(minimap)

        if me is None:
            still_counter = 0
        else:
            if last_pos is not None:
                dx = me[0] - last_pos[0]
                dy = me[1] - last_pos[1]
                dist = sqrt(dx * dx + dy * dy)
                if dist < 2.0:
                    still_counter += 1
                else:
                    still_counter = 0
            last_pos = me

        if still_counter >= still_threshold:
            logger.info(
                "Nav: arrived at minimap target=%s current=%s after %.1fs",
                target_xy,
                last_pos,
                time.time() - start,
            )
            return True

        time.sleep(0.2)

    logger.warning(
        "Nav: timeout (%.1fs) waiting for arrival at target=%s", timeout, target_xy
    )
    return False


def approach_enemy(
    minimap_roi: Tuple[int, int, int, int],
    me_xy: Tuple[int, int],
    enemy_xy: Tuple[int, int],
    factor: float = 0.8,
) -> Tuple[int, int]:
    """
    Compute a point on the mini-map at `factor` of the path to an enemy.

    Clicking slightly before the enemy marker tends to be more stable
    than clicking exactly on it, especially with latency and inertia.
    """
    mx, my = me_xy
    ex, ey = enemy_xy

    tx = int(mx + (ex - mx) * factor)
    ty = int(my + (ey - my) * factor)

    logger.debug(
        "Nav: approach_enemy roi=%s me=%s enemy=%s factor=%.2f target=(%d,%d)",
        minimap_roi,
        me_xy,
        enemy_xy,
        factor,
        tx,
        ty,
    )
    return tx, ty
