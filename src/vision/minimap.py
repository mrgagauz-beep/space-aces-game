"""
Mini-map vision utilities for Space Aces Bot.

Detection of player and enemies on the mini-map using HSV thresholds
and simple contour analysis.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import logging

import cv2
import numpy as np

from config.config import Config
from utils.logger import setup_logger
from vision.capture import grab_minimap

setup_logger()
logger = logging.getLogger(__name__)

# Last known player position on minimap (in minimap ROI coordinates).
_LAST_PLAYER_MM: Optional[Tuple[int, int]] = None
_PLAYER_FRAME_IDX: int = 0

# Area bounds for player blob in inner minimap coordinates (kept for legacy detectors).
PLAYER_AREA_MIN: int = 20
PLAYER_AREA_MAX: int = 2000

# Default peak prominence ratio for line-based player detector.
PLAYER_LINE_PEAK_RATIO: float = 2.5


def _extract_inner_minimap(
    minimap_bgr: np.ndarray, cfg: Optional[Config] = None
) -> Tuple[Optional[np.ndarray], int, int]:
    """
    Cut out inner part of minimap (without frame, header and scales) using
    percentage margins from cfg.minimap_inner_margin_pct().

    Returns (inner_bgr, x0, y0) where inner_bgr is the cropped minimap region,
    and (x0, y0) is the top-left offset in the original minimap ROI.
    """
    if minimap_bgr is None or minimap_bgr.size == 0:
        return None, 0, 0

    h, w = minimap_bgr.shape[:2]

    margins = (
        cfg.minimap_inner_margin_pct()
        if cfg is not None and hasattr(cfg, "minimap_inner_margin_pct")
        else {}
    )
    # Defaults tuned for current profile (fractions of width/height).
    top = float(margins.get("top", 0.23))
    bottom = float(margins.get("bottom", 0.09))
    left = float(margins.get("left", 0.163))
    right = float(margins.get("right", 0.06))

    # Initial (unclamped) inner rectangle in minimap coordinates.
    x0_raw = int(round(w * left))
    y0_raw = int(round(h * top))
    inner_w_raw = int(round(w * (1.0 - left - right)))
    inner_h_raw = int(round(h * (1.0 - top - bottom)))

    logger.debug(
        "minimap inner_raw: w=%d h=%d left=%.3f right=%.3f top=%.3f bottom=%.3f "
        "x0=%d y0=%d w_in=%d h_in=%d",
        w,
        h,
        left,
        right,
        top,
        bottom,
        x0_raw,
        y0_raw,
        inner_w_raw,
        inner_h_raw,
    )

    # Clamp to minimap bounds.
    x0 = max(0, min(x0_raw, w))
    y0 = max(0, min(y0_raw, h))

    inner_w = max(0, inner_w_raw)
    inner_h = max(0, inner_h_raw)

    if x0 + inner_w > w:
        inner_w = max(0, w - x0)
    if y0 + inner_h > h:
        inner_h = max(0, h - y0)

    if inner_w <= 0 or inner_h <= 0:
        logger.debug(
            "minimap inner_clamped invalid: w=%d h=%d x0=%d y0=%d w_in=%d h_in=%d",
            w,
            h,
            x0,
            y0,
            inner_w,
            inner_h,
        )
        return None, 0, 0

    logger.debug(
        "minimap inner_clamped: w=%d h=%d x0=%d y0=%d w_in=%d h_in=%d",
        w,
        h,
        x0,
        y0,
        inner_w,
        inner_h,
    )

    inner = minimap_bgr[y0 : y0 + inner_h, x0 : x0 + inner_w]
    return inner, x0, y0


def _build_player_mask(
    minimap_bgr: np.ndarray, cfg: Optional[Config] = None
) -> Tuple[Optional[np.ndarray], Tuple[int, int, int, int]]:
    """
    Build binary mask for player crosshair on the minimap.

    Returns
    -------
    mask : np.ndarray | None
        Binary mask (0/255) in inner-minimap coordinates, or None on failure.
    rect : tuple[int, int, int, int]
        Inner minimap rectangle in minimap coordinates: (x0, y0, w_in, h_in).
    """
    inner, x0, y0 = _extract_inner_minimap(minimap_bgr, cfg)
    if inner is None or inner.size == 0:
        return None, (x0, y0, 0, 0)

    h_in, w_in = inner.shape[:2]
    if h_in < 5 or w_in < 5:
        return None, (x0, y0, w_in, h_in)

    # Convert inner minimap to grayscale and threshold bright gray crosshair lines.
    try:
        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    except Exception:
        logger.exception("build_player_mask: failed to convert inner minimap to grayscale")
        return None, (x0, y0, w_in, h_in)

    gray_blur = cv2.GaussianBlur(gray, (3, 3), 0)

    thr = 50
    if cfg is not None and hasattr(cfg, "player_minimap_gray_thr"):
        try:
            thr = int(cfg.player_minimap_gray_thr())  # type: ignore[call-arg]
        except Exception:
            thr = 50
    thr = max(0, min(255, thr))

    _, mask = cv2.threshold(gray_blur, thr, 255, cv2.THRESH_BINARY)

    # Optional: small dilation to make thin lines more robust.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    return mask, (x0, y0, w_in, h_in)


def _find_player_crosshair_legacy(
    minimap_bgr: np.ndarray, cfg: Optional[Config] = None
) -> Optional[Tuple[int, int]]:
    """
    Legacy player detector kept for reference / experimentation.

    Not used by the main pipeline.
    """
    inner, x0, y0 = _extract_inner_minimap(minimap_bgr, cfg)
    if inner is None or inner.size == 0:
        return None

    try:
        gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY)
    except Exception:
        return None

    try:
        _, thr = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
    except Exception:
        return None

    col_sum = thr.sum(axis=0).astype(np.float32)
    row_sum = thr.sum(axis=1).astype(np.float32)

    if col_sum.max() == 0 or row_sum.max() == 0:
        return None

    vert_x = int(np.argmax(col_sum))
    horiz_y = int(np.argmax(row_sum))

    global_x = x0 + vert_x
    global_y = y0 + horiz_y
    return global_x, global_y


def _is_near_corner(x: int, y: int, w: int, h: int, radius: int = 20) -> bool:
    """
    Return True if point (x, y) lies within `radius` pixels of any corner
    of a rectangle with size (w, h), in minimap ROI coordinates.
    """
    corners = (
        (0, 0),
        (w - 1, 0),
        (0, h - 1),
        (w - 1, h - 1),
    )
    r2 = radius * radius
    for cx, cy in corners:
        dx = x - cx
        dy = y - cy
        if dx * dx + dy * dy <= r2:
            return True
    return False


def _point_in_rect(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
    rx, ry, rw, rh = rect
    return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)


def _is_in_player_ignore_region(
    x: int,
    y: int,
    cfg: Optional[Config],
    w_mm: int,
    h_mm: int,
) -> bool:
    """
    Check if point (x, y) lies inside any "forbidden" region for player
    position on the minimap.

    Priority:
    1) If profile defines minimap_player_ignore_regions as
       {"id": [x, y, w, h], ...} in minimap ROI coordinates, use them.
    2) Otherwise, if minimap_enemy_detector().get("ignore_regions") exists,
       reuse those regions for the player.
    3) If nothing is configured, use four default corner rectangles
       (portal zones) as fractions of minimap width/height.
    """
    regions: dict[str, list[int]] = {}

    # 1) Explicit regions for player.
    if cfg is not None and hasattr(cfg, "minimap_player_ignore_regions"):
        try:
            regions = cfg.minimap_player_ignore_regions() or {}
        except Exception:
            regions = {}

    # 2) Fallback: reuse enemy ignore_regions.
    if not regions and cfg is not None and hasattr(cfg, "minimap_enemy_detector"):
        try:
            det_cfg = cfg.minimap_enemy_detector() or {}
            regions = det_cfg.get("ignore_regions", {}) or {}
        except Exception:
            regions = {}

    # 3) Regions defined in profile.
    for rect in regions.values():
        if len(rect) != 4:
            continue
        if _point_in_rect(x, y, tuple(rect)):  # type: ignore[arg-type]
            return True

    # 4) Default: four corner rectangles (approximate portal zones).
    if not regions:
        margin_x = int(w_mm * 0.22)
        margin_y = int(h_mm * 0.35)

        corner_rects = [
            (0, 0, margin_x, margin_y),  # top-left
            (w_mm - margin_x, 0, margin_x, margin_y),  # top-right
            (0, h_mm - margin_y, margin_x, margin_y),  # bottom-left
            (w_mm - margin_x, h_mm - margin_y, margin_x, margin_y),  # bottom-right
        ]
        for rect in corner_rects:
            if _point_in_rect(x, y, rect):
                return True

    return False


def find_player_crosshair(
    minimap_bgr: np.ndarray, cfg: Optional[Config] = None
) -> Optional[Tuple[int, int]]:
    """
    Find player position on minimap as intersection of two cyan crosshair lines.

    Algorithm:
    - Work on inner minimap region (without frame / header) using _extract_inner_minimap.
    - Build HSV mask for the cyan lines from profile thresholds.
    - Sum mask over rows/columns to obtain line strength profiles.
    - Take argmax over row_sum/col_sum as line positions and apply simple peak-ratio checks.
    - Convert intersection point back to minimap ROI coordinates and return (x, y).
    """
    global _LAST_PLAYER_MM, _PLAYER_FRAME_IDX

    _PLAYER_FRAME_IDX += 1

    if minimap_bgr is None or minimap_bgr.size == 0:
        logger.debug("find_player_crosshair: empty minimap frame")
        return None

    h_mm, w_mm = minimap_bgr.shape[:2]

    # Reuse common mask builder so that debug tools can visualize the same mask.
    mask, (x0, y0, w_in, h_in) = _build_player_mask(minimap_bgr, cfg)
    if mask is None or mask.size == 0 or w_in <= 0 or h_in <= 0:
        logger.debug(
            "find_player_crosshair: empty/invalid player mask (full=%dx%d inner=%dx%d)",
            w_mm,
            h_mm,
            w_in,
            h_in,
        )
        return None

    # Sum along rows and columns to find strongest horizontal/vertical lines.
    row_sum = mask.sum(axis=1).astype(np.float32)  # shape (h_in,)
    col_sum = mask.sum(axis=0).astype(np.float32)  # shape (w_in,)

    row_max = float(row_sum.max())
    col_max = float(col_sum.max())
    row_mean = float(row_sum.mean())
    col_mean = float(col_sum.mean())

    logger.debug(
        "find_player_crosshair: inner=%dx%d row_max=%.1f row_mean=%.1f "
        "col_max=%.1f col_mean=%.1f",
        w_in,
        h_in,
        row_max,
        row_mean,
        col_max,
        col_mean,
    )

    if row_max == 0.0 or col_max == 0.0:
        if _PLAYER_FRAME_IDX % 30 == 0:
            logger.info(
                "find_player_crosshair: no line signal in mask "
                "(row_max=%.1f col_max=%.1f)",
                row_max,
                col_max,
            )
        return None

    ratio_cfg = PLAYER_LINE_PEAK_RATIO
    if cfg is not None and hasattr(cfg, "player_minimap_line_ratio"):
        try:
            ratio_cfg = float(cfg.player_minimap_line_ratio())  # type: ignore[call-arg]
        except Exception:
            ratio_cfg = PLAYER_LINE_PEAK_RATIO

    # Lines are the global maxima over rows/columns.
    y_line = int(np.argmax(row_sum))
    x_line = int(np.argmax(col_sum))

    row_peak = float(row_sum[y_line])
    col_peak = float(col_sum[x_line])

    # Simple peak prominence checks to avoid random noise.
    if row_mean > 0.0 and row_peak < ratio_cfg * row_mean:
        logger.debug(
            "find_player_crosshair: weak horizontal line peak row_peak=%.1f "
            "row_mean=%.1f ratio=%.2f thr=%.2f",
            row_peak,
            row_mean,
            row_peak / max(row_mean, 1e-6),
            ratio_cfg,
        )
        return None

    if col_mean > 0.0 and col_peak < ratio_cfg * col_mean:
        logger.debug(
            "find_player_crosshair: weak vertical line peak col_peak=%.1f "
            "col_mean=%.1f ratio=%.2f thr=%.2f",
            col_peak,
            col_mean,
            col_peak / max(col_mean, 1e-6),
            ratio_cfg,
        )
        return None

    px = x0 + x_line
    py = y0 + y_line
    player_mm = (int(px), int(py))

    logger.debug(
        "find_player_crosshair: minimap=%dx%d inner=%dx%d x0=%d y0=%d "
        "x_line=%d y_line=%d player_mm=%s",
        w_mm,
        h_mm,
        w_in,
        h_in,
        x0,
        y0,
        x_line,
        y_line,
        player_mm,
    )

    if _PLAYER_FRAME_IDX % 30 == 0:
        logger.info("find_player_crosshair: player_mm=%s", player_mm)

    _LAST_PLAYER_MM = player_mm
    return player_mm


def find_enemies(
    minimap_bgr: np.ndarray,
    cfg: Config,
    map_name: Optional[str] = None,
) -> List[Tuple[int, int]]:
    """
    Simple enemy detector (red dots) on minimap.

    - Uses HSV mask for two red ranges.
    - Works on inner minimap.
    - Filters by basic contour size.
    - Returns coordinates in minimap ROI system.
    """
    inner, x0, y0 = _extract_inner_minimap(minimap_bgr, cfg)
    if inner is None or inner.size == 0:
        logger.warning("Empty minimap frame passed to find_enemies")
        return []

    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    hsv_cfg = cfg.hsv() if hasattr(cfg, "hsv") else {}

    def _val(key: str, default: List[int]) -> np.ndarray:
        return np.array(hsv_cfg.get(key, default), dtype=np.uint8)

    red1_min = _val("enemy_minimap_red_1_min", [0, 102, 64])
    red1_max = _val("enemy_minimap_red_1_max", [8, 255, 255])
    red2_min = _val("enemy_minimap_red_2_min", [170, 102, 64])
    red2_max = _val("enemy_minimap_red_2_max", [179, 255, 255])

    mask1 = cv2.inRange(hsv, red1_min, red1_max)
    mask2 = cv2.inRange(hsv, red2_min, red2_max)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    det_cfg = (
        cfg.minimap_enemy_detector()
        if hasattr(cfg, "minimap_enemy_detector")
        else {}
    )
    area_min = det_cfg.get("area_min", 3)
    area_max = det_cfg.get("area_max", 500)
    bbox_min = det_cfg.get("bbox_min", 2)
    bbox_max = det_cfg.get("bbox_max", 25)

    enemies: List[Tuple[int, int]] = []

    for c in contours:
        x, y, w_box, h_box = cv2.boundingRect(c)
        area = w_box * h_box
        if area < area_min or area > area_max:
            continue
        if w_box < bbox_min or w_box > bbox_max:
            continue
        if h_box < bbox_min or h_box > bbox_max:
            continue

        cx = x + w_box // 2
        cy = y + h_box // 2

        gx = x0 + cx
        gy = y0 + cy

        enemies.append((gx, gy))

    return enemies


if __name__ == "__main__":
    cfg = Config()
    minimap_frame = grab_minimap(cfg)

    player = find_player_crosshair(minimap_frame, cfg)
    enemies = find_enemies(minimap_frame, cfg)

    logger.info("Sanity minimap: player=%s, enemies=%d", player, len(enemies))
