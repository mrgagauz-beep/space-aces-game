import argparse
import logging
import pathlib
import sys

import cv2
import numpy as np

# Ensure that `src` is on sys.path so that `config`, `vision`, etc. can be imported
ROOT_DIR = pathlib.Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.config import Config
from core.fsm import Bot
from utils.logger import setup_logger
from vision import capture, minimap, screen
from vision.screen import find_mob_labels_main, mob_label_box_to_click_point


def run_test_mode() -> None:
    """
    Visual test mode:
    - захватывает MAIN и MINIMAP;
    - детектирует ящики, мобов, подписи мобов и врагов на миникарте;
    - рисует оверлеи и показывает окна OpenCV до нажатия Esc / q.
    """
    logger = setup_logger()
    cfg = Config()

    logger.info("Test mode: starting visual debug loop")

    # Create named windows and move them outside the capture ROI to avoid recursion
    window_main = "Space Aces MAIN (crates/mobs/labels)"
    window_minimap = "Space Aces MINIMAP (player/enemies)"

    cv2.namedWindow(window_main, cv2.WINDOW_NORMAL)
    cv2.namedWindow(window_minimap, cv2.WINDOW_NORMAL)

    # Move windows to safe position (bottom right, outside typical game area)
    # Adjust these coordinates if they still conflict with your game window
    cv2.moveWindow(window_main, 2000, 100)
    cv2.moveWindow(window_minimap, 2000, 700)

    logger.info("OpenCV windows positioned at x=2000 to avoid capture recursion")

    while True:
        main_bgr = capture.grab_main(cfg)
        minimap_bgr = capture.grab_minimap(cfg)

        if main_bgr is None or minimap_bgr is None:
            logger.error("Failed to grab frames from screen")
            break

        crates = screen.find_crates_main(main_bgr, cfg)
        mobs = screen.find_mobs_main(main_bgr, cfg)
        labels = find_mob_labels_main(main_bgr, cfg)

        enemies_mm = minimap.find_enemies(minimap_bgr, cfg)
        player_mm = minimap.find_player_crosshair(minimap_bgr, cfg)

        main_vis = main_bgr.copy()
        mm_vis = minimap_bgr.copy()

        # Draw crates (yellow)
        for (x, y, w, h) in crates:
            cv2.rectangle(main_vis, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Draw mobs (red)
        for (x, y, w, h) in mobs:
            cv2.rectangle(main_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Draw mob labels (magenta) and click points (cyan)
        for (x, y, w, h) in labels:
            cv2.rectangle(main_vis, (x, y), (x + w, y + h), (255, 0, 255), 1)
            cx, cy = mob_label_box_to_click_point((x, y, w, h), cfg)
            cv2.circle(main_vis, (cx, cy), 3, (255, 255, 0), -1)

        # Draw enemies on minimap (red dots)
        for (ex, ey) in enemies_mm:
            cv2.circle(mm_vis, (ex, ey), 3, (0, 0, 255), -1)

        # Draw player on minimap (cyan dot)
        if player_mm is not None:
            px, py = player_mm
            cv2.circle(mm_vis, (px, py), 4, (255, 255, 0), -1)

        # Add info overlay to main screen
        info_text = f"Crates: {len(crates)} | Mobs: {len(mobs)} | Labels: {len(labels)}"
        cv2.putText(main_vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(main_vis, "Press ESC or Q to exit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Add info overlay to minimap
        mm_info = f"Enemies: {len(enemies_mm)} | Player: {'Yes' if player_mm else 'No'}"
        cv2.putText(mm_vis, mm_info, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(window_main, main_vis)
        cv2.imshow(window_minimap, mm_vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyAllWindows()
    logger.info("Test mode: finished")


def run_bot_mode() -> None:
    """
    Production bot mode: starts FSM loop.
    """
    logger = setup_logger()
    cfg = Config()
    bot = Bot(cfg)
    logger.info("Bot mode: starting FSM loop")
    bot.run()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["test", "bot"],
        default="test",
        help="Режим работы бота: test или bot",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logger = logging.getLogger("space_aces")
    logger.info("run.py started with mode=%s", args.mode)

    try:
        if args.mode == "test":
            run_test_mode()
        else:
            run_bot_mode()
    except Exception:
        logger.exception("Fatal error in main()")
        raise


if __name__ == "__main__":
    main()

