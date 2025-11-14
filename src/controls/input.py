"""
Input helpers: mouse and keyboard with light randomization.

These wrappers are intended to look slightly less robotic by adding
small jitter in timing and cursor position, and to centralize safety
checks (emergency stop).
"""

from __future__ import annotations

import random
import time
from typing import Tuple

import keyboard
import pyautogui
import logging

from utils.logger import setup_logger

setup_logger()
logger = logging.getLogger(__name__)

pyautogui.PAUSE = 0
EXIT_KEY = "esc"


def press_key(key: str, jitter_ms: Tuple[int, int] = (35, 80)) -> None:
    """
    Press and release a keyboard key with a small random delay.

    Parameters
    ----------
    key : str
        Key name according to `keyboard` / `pyautogui` conventions.
    jitter_ms : tuple[int, int]
        Min/max delay in milliseconds between press and release.
    """
    delay = random.uniform(jitter_ms[0], jitter_ms[1]) / 1000.0
    logger.debug("press_key: key=%s delay=%.3f", key, delay)

    try:
        keyboard.press(key)
        time.sleep(delay)
        keyboard.release(key)
    except Exception as exc:  # pragma: no cover - platform dependent
        logger.warning("press_key failed for %s: %s", key, exc)


def move_click_abs(
    x: int,
    y: int,
    jitter_px: Tuple[int, int] = (0, 2),
) -> None:
    """
    Move cursor to approximate absolute position and perform a left click.

    A small random jitter is added to the target point to avoid pixel-perfect
    repetition of clicks.
    """
    dx = random.randint(jitter_px[0], jitter_px[1])
    dy = random.randint(jitter_px[0], jitter_px[1])
    tx, ty = x + dx, y + dy

    logger.debug(
        "move_click_abs: target=(%d,%d) jitter=(%d,%d) final=(%d,%d)",
        x,
        y,
        dx,
        dy,
        tx,
        ty,
    )

    try:
        pyautogui.moveTo(tx, ty)
        pyautogui.click()
    except Exception as exc:  # pragma: no cover - GUI dependent
        logger.warning("move_click_abs failed at (%d,%d): %s", tx, ty, exc)


def click_in_roi(roi: Tuple[int, int, int, int], pt: Tuple[int, int]) -> None:
    """
    Click a point inside a given ROI (x, y, w, h).

    Parameters
    ----------
    roi : tuple[int, int, int, int]
        Region as (x, y, width, height).
    pt : tuple[int, int]
        Point inside ROI in local coordinates (relative to ROI origin).
    """
    rx, ry, _, _ = roi
    px, py = pt
    ax, ay = rx + px, ry + py

    logger.debug("click_in_roi: roi=%s local_pt=%s abs=(%d,%d)", roi, pt, ax, ay)
    move_click_abs(ax, ay)


def emergency_stop() -> bool:
    """
    Check if the emergency key (ESC) is held.

    Designed to be safe on platforms where low-level keyboard hooks
    are restricted: any exception is swallowed and treated as False.
    """
    try:
        return keyboard.is_pressed(EXIT_KEY)
    except Exception:  # pragma: no cover - platform dependent
        return False


# Backwards compatible alias
def is_exit_pressed() -> bool:
    """Alias to emergency_stop for older code paths."""
    return emergency_stop()
