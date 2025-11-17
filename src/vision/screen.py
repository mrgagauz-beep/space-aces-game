"""
Main screen vision utilities for Space Aces Bot.

This module is responsible for detecting enemies and bonus crates
on the main game screen using classical computer vision methods.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import cv2
import numpy as np
import logging

from config.config import Config
from utils.logger import setup_logger
from vision.capture import grab_main

setup_logger()
logger = logging.getLogger(__name__)


# Simple alias for rectangular bounding boxes in MAIN coordinates.
# Convention: (x1, y1, x2, y2) with x2 > x1, y2 > y1.
BBox = Tuple[float, float, float, float]


class MainTargetKind(str, Enum):
    ENEMY = "enemy"
    BOX = "box"
    OTHER = "other"


@dataclass
class MainTarget:
    """
    MAIN-screen target detected by name label.

    kind : MainTargetKind
        High-level type: enemy ship / bonus box / other.
    bbox_name : BBox
        Bounding box of the name label (text + frame) in MAIN-local coords.
    bbox_ship : BBox
        Approximate bounding box of the ship or box inferred from name.
    text : str
        Recognized text (placeholder for future OCR-based classification).
    """

    kind: MainTargetKind
    bbox_name: BBox
    bbox_ship: BBox
    text: str = ""


@dataclass
class MainEnemy:
    """
    Backwards compatible enemy representation (subset of MainTarget).
    """

    bbox_name: BBox
    bbox_ship: BBox
    text: str = ""


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


def _safe_float(d: dict, key: str, default: float) -> float:
    try:
        value = float(d.get(key, default))
    except (TypeError, ValueError):
        value = default
    return value


def _safe_int(d: dict, key: str, default: int) -> int:
    try:
        value = int(d.get(key, default))
    except (TypeError, ValueError):
        value = default
    return max(0, value)


def _infer_ship_bbox_geometry_from_name_bbox(name_bbox: BBox, cfg) -> BBox:
    """
    Geometric fallback: approximate ship bbox above the name label.

    Uses fixed width/height and vertical offset from ``mob_name_detector_main``.
    """
    nx1, ny1, nx2, ny2 = name_bbox
    name_w = max(1.0, float(nx2 - nx1))
    name_h = max(1.0, float(ny2 - ny1))
    cx_name = (float(nx1) + float(nx2)) * 0.5

    nd = (
        cfg.mob_name_detector_main()
        if hasattr(cfg, "mob_name_detector_main")
        else {}
    )

    ship_w = _safe_float(nd, "ship_width_px", max(name_w * 2.5, 80.0))
    ship_h = _safe_float(nd, "ship_height_px", max(name_h * 2.0, 40.0))
    offset_legacy = _safe_float(
        nd,
        "name_to_ship_offset_px",
        max(name_h * 0.3, 5.0),
    )
    offset_up = _safe_float(
        nd,
        "name_to_ship_offset_up_px",
        offset_legacy,
    )

    sx1 = cx_name - ship_w * 0.5
    sx2 = cx_name + ship_w * 0.5
    sy2 = float(ny1) - offset_up
    sy1 = sy2 - ship_h

    return (sx1, sy1, sx2, sy2)


def _infer_ship_bbox_blob_from_name_bbox(
    frame_bgr: np.ndarray,
    name_bbox: BBox,
    cfg,
) -> Optional[BBox]:
    """
    Try to infer ship bbox as the largest bright blob above the name label.

    The search window is defined relative to the name bbox and tuned via
    ``mob_name_detector_main`` config parameters.

    Returns
    -------
    Optional[BBox]
        Ship bbox in MAIN coordinates, or None if a reliable blob was not found.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    h_frame, w_frame = frame_bgr.shape[:2]

    nx1, ny1, nx2, ny2 = name_bbox
    name_w = max(1.0, float(nx2 - nx1))
    name_h = max(1.0, float(ny2 - ny1))

    nd = (
        cfg.mob_name_detector_main()
        if hasattr(cfg, "mob_name_detector_main")
        else {}
    )

    margin_x_k = _safe_float(nd, "ship_search_margin_x_k", 0.35)
    height_up_k = _safe_float(nd, "ship_search_height_up_k", 3.0)
    offset_up_px = _safe_float(nd, "ship_search_offset_up_px", 5.0)
    blob_thr = _safe_int(nd, "ship_blob_threshold", 170)
    min_area = _safe_int(nd, "ship_blob_min_area_px", 50)
    max_area = _safe_int(nd, "ship_blob_max_area_px", 5000)
    min_w = _safe_int(nd, "ship_min_width_px", 10)
    min_h = _safe_int(nd, "ship_min_height_px", 10)

    margin_x = int(name_w * margin_x_k)
    height_up = int(name_h * height_up_k)

    sx1 = max(int(nx1) - margin_x, 0)
    sx2 = min(int(nx2) + margin_x, w_frame)
    sy2 = int(max(0.0, float(ny1) - offset_up_px))
    sy1 = max(sy2 - height_up, 0)

    if sx2 <= sx1 or sy2 <= sy1:
        return None

    roi = frame_bgr[sy1:sy2, sx1:sx2]
    if roi is None or roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    _, binary = cv2.threshold(
        gray, float(blob_thr), 255, cv2.THRESH_BINARY
    )

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary
    )
    if num_labels <= 1:
        return None

    roi_cx = (sx2 - sx1) * 0.5
    roi_cy = (sy2 - sy1) * 0.5

    best_idx: Optional[int] = None
    best_score: Optional[float] = None

    for label in range(1, num_labels):
        x, y, w_box, h_box, area = stats[label]
        if area < min_area or area > max_area:
            continue
        if w_box < min_w or h_box < min_h:
            continue

        cx, cy = centroids[label]
        dx = float(cx) - roi_cx
        dy = float(cy) - roi_cy
        d2 = dx * dx + dy * dy

        # Prefer blobs closer to ROI center; smaller score is better.
        score = d2
        if best_idx is None or best_score is None or score < best_score:
            best_idx = label
            best_score = score

    if best_idx is None:
        return None

    x, y, w_box, h_box, area = stats[best_idx]
    if area < min_area or w_box < min_w or h_box < min_h:
        return None

    ship_x1 = float(sx1 + x)
    ship_y1 = float(sy1 + y)
    ship_x2 = ship_x1 + float(w_box)
    ship_y2 = ship_y1 + float(h_box)

    return (ship_x1, ship_y1, ship_x2, ship_y2)


def infer_ship_bbox_from_name_bbox(
    frame_bgr: np.ndarray,
    name_bbox: BBox,
    cfg,
) -> BBox:
    """
    Infer ship bbox on MAIN from a mob name bbox using image blobs.

    The helper first tries to locate a bright blob above the name label.
    If no reliable blob is found, it falls back to a purely geometric
    bbox above the name.
    """
    blob_bbox = _infer_ship_bbox_blob_from_name_bbox(frame_bgr, name_bbox, cfg)
    if blob_bbox is not None:
        return blob_bbox

    return _infer_ship_bbox_geometry_from_name_bbox(name_bbox, cfg)


def _xywh_to_bbox(x: float, y: float, w: float, h: float) -> BBox:
    return (x, y, x + w, y + h)


def _bbox_from_crate_xywh(crate: Tuple[int, int, int, int]) -> BBox:
    cx, cy, cw, ch = crate[:4]
    return _xywh_to_bbox(float(cx), float(cy), float(cw), float(ch))


def _bboxes_intersect(a: BBox, b: BBox) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return ix2 > ix1 and iy2 > iy1


def detect_main_targets_main(
    main_bgr: np.ndarray,
    cfg,
    crates: Optional[List[Tuple[int, int, int, int]]] = None,
) -> List[MainTarget]:
    """
    Detect MAIN-screen targets (enemies and bonus boxes) by their name labels.

    Classification heuristic:
    - labels whose name/ship bbox intersects a detected crate bbox are treated
      as ``BOX``;
    - remaining labels are treated as ``ENEMY``.

    Parameters
    ----------
    main_bgr : np.ndarray
        MAIN screen frame (already cropped to MAIN ROI).
    cfg : Config
        Configuration object with HSV thresholds and geometry.
    crates : list[tuple[int, int, int, int]] | None
        Optional list of crate bboxes (x, y, w, h). If omitted, crates are
        detected internally via ``find_crates_main``.
    """
    if main_bgr is None or main_bgr.size == 0:
        logger.warning("Empty main frame passed to detect_main_targets_main")
        return []

    if crates is None:
        crates = find_crates_main(main_bgr, cfg)

    crate_bboxes: List[BBox] = [
        _bbox_from_crate_xywh(c)
        for c in crates
        if isinstance(c, (tuple, list)) and len(c) >= 4
    ]

    label_boxes = find_mob_labels_main(main_bgr, cfg)
    targets: List[MainTarget] = []

    for lb in label_boxes:
        if not isinstance(lb, (tuple, list)) or len(lb) < 4:
            continue
        x, y, w_box, h_box = lb[:4]
        nx1 = float(x)
        ny1 = float(y)
        nx2 = float(x + w_box)
        ny2 = float(y + h_box)
        name_bbox: BBox = (nx1, ny1, nx2, ny2)
        ship_bbox = infer_ship_bbox_from_name_bbox(main_bgr, name_bbox, cfg)

        kind = MainTargetKind.ENEMY
        for cb in crate_bboxes:
            if _bboxes_intersect(name_bbox, cb) or _bboxes_intersect(
                ship_bbox, cb
            ):
                kind = MainTargetKind.BOX
                break

        targets.append(
            MainTarget(
                kind=kind,
                bbox_name=name_bbox,
                bbox_ship=ship_bbox,
                text="",
            )
        )

    return targets


def detect_main_enemies_by_name(
    main_bgr: np.ndarray, cfg
) -> List[MainEnemy]:
    """
    Return a list of enemies on the MAIN screen detected by their name labels.

    Thin wrapper over ``detect_main_targets_main`` that keeps only ENEMY targets
    and converts them to ``MainEnemy`` structures for backwards compatibility.
    """
    targets = detect_main_targets_main(main_bgr, cfg)
    enemies: List[MainEnemy] = []

    for t in targets:
        if t.kind != MainTargetKind.ENEMY:
            continue
        enemies.append(
            MainEnemy(
                bbox_name=t.bbox_name,
                bbox_ship=t.bbox_ship,
                text=t.text,
            )
        )

    return enemies


if __name__ == "__main__":
    cfg = Config()
    main = grab_main(cfg)

    crates = find_crates_main(main, cfg)
    mobs = find_mobs_main(main, cfg)

    logger.info(
        "Sanity main screen: crates=%d, mobs=%d", len(crates), len(mobs)
    )
