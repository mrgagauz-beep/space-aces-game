"""
Self-test harness for Space Aces Bot vision modules.

Loads PNG screenshots from assets/screenshots/ and runs:
- mini-map enemy / player detection;
- main screen crate / mob detection.

Prints aggregate counts and average processing FPS.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from config.config import Config
from utils.logger import setup_logger
from vision import minimap, screen


def iter_screenshots(folder: Path):
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for path in folder.glob(ext):
            yield path


def main():
    logger = setup_logger()

    cfg = Config()
    shots_dir = Path("assets/screenshots")

    if not shots_dir.exists():
        logger.error("Screenshots directory does not exist: %s", shots_dir)
        return

    total_frames = 0
    total_time = 0.0
    total_crates = 0
    total_mobs = 0
    total_enemies_mm = 0

    for img_path in iter_screenshots(shots_dir):
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Failed to load screenshot: %s", img_path)
            continue

        x_m, y_m, w_m, h_m = cfg.roi_main()
        x_mm, y_mm, w_mm, h_mm = cfg.roi_minimap()

        main_bgr = img[y_m : y_m + h_m, x_m : x_m + w_m]
        mm_bgr = img[y_mm : y_mm + h_mm, x_mm : x_mm + w_mm]

        t0 = time.perf_counter()
        crates = screen.find_crates_main(main_bgr, cfg)
        mobs = screen.find_mobs_main(main_bgr)
        enemies_mm = minimap.find_enemies(mm_bgr, cfg)
        player_mm = minimap.find_player_crosshair(mm_bgr)
        dt = time.perf_counter() - t0

        total_frames += 1
        total_time += dt
        total_crates += len(crates)
        total_mobs += len(mobs)
        total_enemies_mm += len(enemies_mm)

        logger.info(
            "Shot %s: crates=%d mobs=%d enemies_mm=%d player_mm=%s time=%.4fs",
            img_path.name,
            len(crates),
            len(mobs),
            len(enemies_mm),
            player_mm,
            dt,
        )

    if total_frames == 0:
        logger.warning("No screenshots processed from %s", shots_dir)
        return

    fps = total_frames / total_time if total_time > 0 else 0.0

    logger.info("Self-test summary:")
    logger.info("  Frames: %d", total_frames)
    logger.info("  Total crates: %d", total_crates)
    logger.info("  Total mobs: %d", total_mobs)
    logger.info("  Total enemies on minimap: %d", total_enemies_mm)
    logger.info("  Avg processing FPS: %.2f", fps)


if __name__ == "__main__":
    main()

