"""
Debug tool for MAIN click geometry.

- Loads screenshots from assets/screenshots.
- Crops MAIN ROI using profile ROI.
- Runs vision.screen detectors for mobs / crates / name-based enemies.
- Selects a target similarly to FSM farming logic (enemy > box, nearest to center).
- Computes click_point_screen using the same aim resolver and coordinate
  mapping as in FSM (Bot._resolve_enemy_click_point + screen_from_main_vision_point).
- Draws all detected enemy ship bboxes, highlight of the chosen target
  and click point overlay for visual inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from config.config import Config
from core.fsm import Bot
from utils.logger import setup_logger
from utils.coords import screen_from_main_vision_point
from vision import screen


def iter_screenshots(folder: Path):
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for path in sorted(folder.glob(ext)):
            yield path


def draw_cross(img, center, color, size: int = 5, thickness: int = 1) -> None:
    x, y = int(center[0]), int(center[1])
    cv2.line(img, (x - size, y), (x + size, y), color, thickness)
    cv2.line(img, (x, y - size), (x, y + size), color, thickness)


def select_main_target(
    main_bgr,
    enemies,
    crates,
) -> Tuple[Optional[str], Optional[Tuple[float, float, float, float]]]:
    """
    Select a MAIN target with simple heuristic:
    - prefer enemies detected by NAME labels;
    - if no enemies, fall back to crates;
    - within chosen type, pick bbox closest to screen center.
    """
    h, w = main_bgr.shape[:2]
    cx_screen = w // 2
    cy_screen = h // 2

    def _closest_box_xywh(boxes):
        best = None
        best_d2 = None
        for b in boxes:
            if not isinstance(b, (tuple, list)) or len(b) < 4:
                continue
            x, y, bw, bh = b[:4]
            cx = x + bw * 0.5
            cy = y + bh * 0.5
            dx = cx - cx_screen
            dy = cy - cy_screen
            d2 = dx * dx + dy * dy
            if best is None or best_d2 is None or d2 < best_d2:
                best = (x, y, bw, bh)
                best_d2 = d2
        return best

    def _closest_enemy(enemies_list):
        best_enemy = None
        best_d2 = None
        for e in enemies_list:
            x1, y1, x2, y2 = e.bbox_ship
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            dx = cx - cx_screen
            dy = cy - cy_screen
            d2 = dx * dx + dy * dy
            if best_enemy is None or best_d2 is None or d2 < best_d2:
                best_enemy = e
                best_d2 = d2
        if best_enemy is None:
            return None
        x1, y1, x2, y2 = best_enemy.bbox_ship
        return (x1, y1, x2 - x1, y2 - y1)

    if enemies:
        return "enemy", _closest_enemy(enemies)
    if crates:
        return "box", _closest_box_xywh(crates)
    return None, None


def run_offline_debug(cfg: Config, logger, no_gui: bool = False) -> None:
    shots_dir = Path("assets/screenshots")
    if not shots_dir.exists():
        logger.error("Screenshots directory does not exist: %s", shots_dir)
        return

    logger.info("debug_main_click_overlay: using screenshots from %s", shots_dir)
    if not no_gui:
        logger.info("Controls: ESC/q - exit; SPACE/ENTER - next frame.")

    bot = Bot(cfg)
    roi_main = cfg.roi_main()

    for img_path in iter_screenshots(shots_dir):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Failed to load screenshot: %s", img_path)
            continue

        x_m, y_m, w_m, h_m = roi_main
        main_bgr = img[y_m : y_m + h_m, x_m : x_m + w_m]
        if main_bgr is None or getattr(main_bgr, "size", 0) == 0:
            logger.warning("Empty MAIN ROI for %s", img_path.name)
            continue

        mobs = screen.find_mobs_main(main_bgr, cfg)
        crates = screen.find_crates_main(main_bgr, cfg)
        targets = screen.detect_main_targets_main(main_bgr, cfg, crates)
        enemies = [t for t in targets if t.kind == screen.MainTargetKind.ENEMY]
        boxes = [t for t in targets if t.kind == screen.MainTargetKind.BOX]

        target_type, target_box = select_main_target(main_bgr, enemies, crates)
        if target_box is None:
            logger.info(
                "frame=%s: no MAIN targets (enemies=%d mobs=%d crates=%d)",
                img_path.name,
                len(enemies),
                len(mobs),
                len(crates),
            )
            if no_gui:
                continue

            vis = img.copy()
            # Draw MAIN ROI rectangle for context.
            cv2.rectangle(
                vis,
                (int(x_m), int(y_m)),
                (int(x_m + w_m), int(y_m + h_m)),
                (255, 255, 0),
                1,
            )
        else:
            x, y, w_box, h_box = target_box
            # Compute click point in MAIN coords using the same helper as FSM.
            click_px, click_py = bot._resolve_enemy_click_point(target_box)  # type: ignore[attr-defined]
            # Map to screen coordinates.
            world_x, world_y = screen_from_main_vision_point(
                (click_px, click_py),
                roi_main,
            )

            logger.info(
                "frame=%s type=%s bbox=(%d,%d,%d,%d) click_screen=(%d,%d) enemies=%d mobs=%d crates=%d",
                img_path.name,
                target_type,
                int(x),
                int(y),
                int(w_box),
                int(h_box),
                int(world_x),
                int(world_y),
                len(enemies),
                len(mobs),
                len(crates),
            )

            if no_gui:
                continue

            vis = img.copy()

            # Draw MAIN ROI rectangle.
            cv2.rectangle(
                vis,
                (int(x_m), int(y_m)),
                (int(x_m + w_m), int(y_m + h_m)),
                (255, 255, 0),
                1,
            )

            # Draw all detected enemy ship bboxes in screen coordinates (thin green).
            for enemy in enemies:
                ex1, ey1, ex2, ey2 = enemy.bbox_ship
                cv2.rectangle(
                    vis,
                    (int(x_m + ex1), int(y_m + ey1)),
                    (int(x_m + ex2), int(y_m + ey2)),
                    (0, 255, 0),
                    1,
                )
                label_pt = (int(x_m + ex1), int(y_m + ey1) - 4)
                cv2.putText(
                    vis,
                    "enemy",
                    label_pt,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            # Draw all detected bonus boxes (BOX kind) in yellow.
            for box in boxes:
                bx1, by1, bx2, by2 = box.bbox_ship
                cv2.rectangle(
                    vis,
                    (int(x_m + bx1), int(y_m + by1)),
                    (int(x_m + bx2), int(y_m + by2)),
                    (0, 255, 255),
                    1,
                )
                label_pt = (int(x_m + bx1), int(y_m + by1) - 4)
                cv2.putText(
                    vis,
                    "box",
                    label_pt,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Draw target bbox in screen coordinates with thicker outline.
            cv2.rectangle(
                vis,
                (int(x_m + x), int(y_m + y)),
                (int(x_m + x + w_box), int(y_m + y + h_box)),
                (0, 0, 255) if target_type == "enemy" else (0, 255, 255),
                2,
            )

            # Draw click point cross.
            draw_cross(vis, (world_x, world_y), (0, 255, 0), size=6, thickness=1)

            # Put small text label with coordinates.
            label = f"{target_type} ({world_x},{world_y})"
            cv2.putText(
                vis,
                label,
                (world_x + 5, world_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        if no_gui:
            continue

        window_name = "Space Aces MAIN click overlay"
        cv2.imshow(window_name, vis)

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                cv2.destroyWindow(window_name)
                return
            if key in (32, 13):  # SPACE or ENTER
                break

    if not no_gui:
        try:
            cv2.destroyWindow("Space Aces MAIN click overlay")
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug MAIN click geometry using saved screenshots."
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without OpenCV windows; only log target and click coordinates.",
    )
    args = parser.parse_args()

    logger = setup_logger()
    cfg = Config()

    profile_name = getattr(getattr(cfg, "path", None), "stem", None)
    profile_path = getattr(cfg, "path", None)

    try:
        roi_main = cfg.roi_main()
    except Exception:
        logger.exception("Failed to get MAIN ROI from profile")
        return

    logger.info(
        "Profile loaded: name=%s path=%s roi.main=%s",
        profile_name,
        profile_path,
        roi_main,
    )

    run_offline_debug(cfg=cfg, logger=logger, no_gui=args.no_gui)


if __name__ == "__main__":
    main()
