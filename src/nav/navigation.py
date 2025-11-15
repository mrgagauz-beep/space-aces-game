"""
Navigation helpers for clicking on the mini-map and waiting for arrival.

This module encapsulates basic pathing logic on the mini-map:
we compute approach points, click on the mini-map and poll until
the player crosshair stops moving.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Tuple, Optional

import numpy as np
import logging

from controls.input import click_in_roi
from utils.logger import setup_logger
from vision.minimap import find_player_crosshair

setup_logger()
logger = logging.getLogger(__name__)


def compute_approach_point(
    me_xy: Tuple[int, int],
    enemy_xy: Tuple[int, int],
    factor: float = 0.8,
) -> Tuple[int, int]:
    """
    Возвращает точку на отрезке от позиции игрока (me_xy) до врага (enemy_xy),
    на factor доле пути (по умолчанию 80%). Используется как цель для клика
    по миникарте.
    """
    mx, my = me_xy
    ex, ey = enemy_xy
    ax = int(mx + (ex - mx) * factor)
    ay = int(my + (ey - my) * factor)
    return ax, ay


def go_to_minimap_point(
    minimap_roi: Tuple[int, int, int, int],
    click_xy: Tuple[int, int],
    cfg,
    grab_minimap_fn: Callable[[], np.ndarray],
    find_player_fn: Callable[[np.ndarray, object], Optional[Tuple[int, int]]],
    timeout: float = 8.0,
    idle_frames: int = 8,
    min_move_px: float = 2.0,
) -> bool:
    """
    Делает клик ЛКМ по миникарте в точку click_xy и ждёт прилёта.

    Parameters
    ----------
    minimap_roi : tuple[int, int, int, int]
        ROI миникарты в координатах экрана (x, y, w, h).
    click_xy : tuple[int, int]
        Точка клика в координатах миникарты (относительно ROI).
    cfg : object
        Конфигурация (пробрасывается в find_player_fn).
    grab_minimap_fn : Callable[[], np.ndarray]
        Функция, возвращающая свежий кадр миникарты (BGR).
    find_player_fn : Callable[[np.ndarray, object], Optional[tuple[int, int]]]
        Функция, находящая (x, y) игрока на миникарте.
    timeout : float
        Максимум секунд ожидания.
    idle_frames : int
        Сколько подряд кадров с маленьким движением считаем признаком
        «прилетели».
    min_move_px : float
        Минимальное изменение позиции между кадрами, чтобы считать,
        что корабль ещё летит.

    Returns
    -------
    bool
        True, если прилетели (движение остановилось), иначе False (таймаут).
    """
    logger.info("Nav: go_to_minimap_point click_xy=%s roi=%s", click_xy, minimap_roi)

    # Стартовый клик по миникарте
    click_in_roi(minimap_roi, click_xy)

    start_time = time.time()
    last_pos: Optional[Tuple[int, int]] = None
    stagnant = 0

    while time.time() - start_time < timeout:
        mm = grab_minimap_fn()
        if mm is None or mm.size == 0:
            time.sleep(0.1)
            continue

        me = find_player_fn(mm, cfg)
        if me is None:
            # Игрок не найден — просто ждём дальше
            time.sleep(0.1)
            continue

        if last_pos is not None:
            dx = me[0] - last_pos[0]
            dy = me[1] - last_pos[1]
            dist = math.hypot(dx, dy)
            if dist < min_move_px:
                stagnant += 1
            else:
                stagnant = 0

            if stagnant >= idle_frames:
                logger.info(
                    "Nav: arrived at click_xy=%s current=%s after %.1fs",
                    click_xy,
                    me,
                    time.time() - start_time,
                )
                return True

        last_pos = me
        time.sleep(0.1)

    logger.warning(
        "Nav: timeout (%.1fs) waiting for arrival at click_xy=%s",
        timeout,
        click_xy,
    )
    return False
