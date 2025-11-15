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
    main_bgr: np.ndarray, cfg
) -> List[Tuple[int, int, int, int]]:
    """
    Детекция мобов (NPC) на основном экране по цвету (синий/голубой).

    Игнорирует HUD по краям, вычитает красный корабль игрока и
    фильтрует кандидатов по размеру и форме. Возвращает список боксов
    (x, y, w, h) в координатах main_bgr.
    """
    if main_bgr is None or main_bgr.size == 0:
        logger.warning("Empty main frame passed to find_mobs_main")
        return []

    h, w = main_bgr.shape[:2]

    md = getattr(cfg, "mobs_main_detector", lambda: {})()
    margin = md.get("play_area_margin_pct", {})
    top_pct = margin.get("top", 0.18)
    bottom_pct = margin.get("bottom", 0.18)
    left_pct = margin.get("left", 0.10)
    right_pct = margin.get("right", 0.25)

    x0 = int(w * left_pct)
    x1 = int(w * (1.0 - right_pct))
    y0 = int(h * top_pct)
    y1 = int(h * (1.0 - bottom_pct))

    sub = main_bgr[y0:y1, x0:x1]

    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    hsv_cfg = cfg.hsv() if hasattr(cfg, "hsv") else {}

    mob_min = np.array(hsv_cfg.get("mob_main_blue_min", [80, 80, 110]), dtype=np.uint8)
    mob_max = np.array(hsv_cfg.get("mob_main_blue_max", [120, 255, 255]), dtype=np.uint8)
    mob_mask = cv2.inRange(hsv, mob_min, mob_max)

    red1_min = np.array(hsv_cfg.get("player_main_red_1_min", [0, 120, 120]), dtype=np.uint8)
    red1_max = np.array(hsv_cfg.get("player_main_red_1_max", [10, 255, 255]), dtype=np.uint8)
    red2_min = np.array(hsv_cfg.get("player_main_red_2_min", [170, 120, 120]), dtype=np.uint8)
    red2_max = np.array(hsv_cfg.get("player_main_red_2_max", [179, 255, 255]), dtype=np.uint8)

    red_mask1 = cv2.inRange(hsv, red1_min, red1_max)
    red_mask2 = cv2.inRange(hsv, red2_min, red2_max)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    mob_mask = cv2.bitwise_and(mob_mask, cv2.bitwise_not(red_mask))

    kernel = np.ones((3, 3), np.uint8)
    mob_mask = cv2.morphologyEx(mob_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mob_mask = cv2.morphologyEx(mob_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mob_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = md.get("min_area", 80)
    max_area = md.get("max_area", 3000)
    min_ar = md.get("min_aspect_ratio", 0.4)
    max_ar = md.get("max_aspect_ratio", 2.5)

    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w_box, h_box = cv2.boundingRect(c)
        area = w_box * h_box
        if area < min_area or area > max_area:
            continue
        if h_box == 0:
            continue
        aspect = w_box / float(h_box)
        if aspect < min_ar or aspect > max_ar:
            continue

        global_x = x0 + x
        global_y = y0 + y
        boxes.append((global_x, global_y, w_box, h_box))

    merged: List[Tuple[int, int, int, int]] = []
    for x, y, wb, hb in boxes:
        cx = x + wb // 2
        cy = y + hb // 2
        keep = True
        for i, (X, Y, W, H) in enumerate(merged):
            CX = X + W // 2
            CY = Y + H // 2
            if abs(cx - CX) <= 10 and abs(cy - CY) <= 10:
                nx = min(x, X)
                ny = min(y, Y)
                mx = max(x + wb, X + W)
                my = max(y + hb, Y + H)
                merged[i] = (nx, ny, mx - nx, my - ny)
                keep = False
                break
        if keep:
            merged.append((x, y, wb, hb))

    return merged


def find_mob_labels_main(
    main_bgr: np.ndarray, cfg
) -> List[Tuple[int, int, int, int]]:
    """
    Находит надписи имён мобов на основном экране вида
    \"-=Luminid=-\", \".::Boss Luminid::.\".

    Работает только в центральной игровой области, ищет красный
    текст и фильтрует кандидатов по размеру и форме. Возвращает
    список боксов (x, y, w, h) в координатах main_bgr.
    """
    if main_bgr is None or main_bgr.size == 0:
        logger.warning("Empty main frame passed to find_mob_labels_main")
        return []

    h, w = main_bgr.shape[:2]

    nd = (
        cfg.mob_name_detector_main()
        if hasattr(cfg, "mob_name_detector_main")
        else {}
    )
    margin = nd.get("play_area_margin_pct", {})
    top_pct = margin.get("top", 0.12)
    bottom_pct = margin.get("bottom", 0.25)
    left_pct = margin.get("left", 0.05)
    right_pct = margin.get("right", 0.20)

    x0 = int(w * left_pct)
    x1 = int(w * (1.0 - right_pct))
    y0 = int(h * top_pct)
    y1 = int(h * (1.0 - bottom_pct))

    sub = main_bgr[y0:y1, x0:x1]

    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    hsv_cfg = cfg.hsv() if hasattr(cfg, "hsv") else {}

    red1_min = np.array(
        hsv_cfg.get("mob_name_red_1_min", [0, 80, 80]), dtype=np.uint8
    )
    red1_max = np.array(
        hsv_cfg.get("mob_name_red_1_max", [10, 255, 255]), dtype=np.uint8
    )
    red2_min = np.array(
        hsv_cfg.get("mob_name_red_2_min", [170, 80, 80]), dtype=np.uint8
    )
    red2_max = np.array(
        hsv_cfg.get("mob_name_red_2_max", [179, 255, 255]), dtype=np.uint8
    )

    mask1 = cv2.inRange(hsv, red1_min, red1_max)
    mask2 = cv2.inRange(hsv, red2_min, red2_max)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    area_min = nd.get("area_min", 80)
    area_max = nd.get("area_max", 12000)
    w_min = nd.get("width_min", 20)
    w_max = nd.get("width_max", 500)
    h_min = nd.get("height_min", 6)
    h_max = nd.get("height_max", 50)
    ar_min = nd.get("min_aspect_ratio", 2.0)
    ar_max = nd.get("max_aspect_ratio", 18.0)

    boxes: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, w_box, h_box = cv2.boundingRect(c)
        area = w_box * h_box
        if area < area_min or area > area_max:
            continue
        if w_box < w_min or w_box > w_max:
            continue
        if h_box < h_min or h_box > h_max:
            continue
        aspect = w_box / float(h_box) if h_box > 0 else 999.0
        if aspect < ar_min or aspect > ar_max:
            continue

        gx = x0 + x
        gy = y0 + y
        boxes.append((gx, gy, w_box, h_box))

    return boxes


def mob_label_box_to_click_point(
    box: Tuple[int, int, int, int], cfg
) -> Tuple[int, int]:
    """
    Преобразует бокс имени моба (x, y, w, h) в точку клика по самому мобу.

    Клик ставится по центру по X и выше надписи на
    click_offset_factor * h.
    """
    x, y, w, h = box
    nd = (
        cfg.mob_name_detector_main()
        if hasattr(cfg, "mob_name_detector_main")
        else {}
    )
    offset_factor = nd.get("click_offset_factor", 0.9)
    cx = x + w // 2
    cy = int(y - h * offset_factor)
    return cx, cy


if __name__ == "__main__":
    cfg = Config()
    main = grab_main(cfg)

    crates = find_crates_main(main, cfg)
    mobs = find_mobs_main(main, cfg)

    logger.info(
        "Sanity main screen: crates=%d, mobs=%d", len(crates), len(mobs)
    )
