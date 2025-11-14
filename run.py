import argparse

import cv2

from config.config import Config
from core.fsm import Bot
from utils.logger import setup_logger
from vision import capture, minimap, screen


def parse_args():
    parser = argparse.ArgumentParser(description="Space Aces Bot launcher")
    parser.add_argument(
        "--mode",
        choices=["calibrate", "test", "bot"],
        default="bot",
        help="Run mode: calibrate ROI, test vision, or start bot",
    )
    return parser.parse_args()


def run_calibration():
    from tools import calibrate_roi

    calibrate_roi.main()


def run_test():
    logger = setup_logger()
    cfg = Config()

    main_frame = capture.grab_main(cfg)
    mini_frame = capture.grab_minimap(cfg)

    if main_frame is None or main_frame.size == 0:
        logger.error("No MAIN frame captured for test mode")
        return
    if mini_frame is None or mini_frame.size == 0:
        logger.error("No MINIMAP frame captured for test mode")
        return

    crates = screen.find_crates_main(main_frame, cfg)
    mobs = screen.find_mobs_main(main_frame)
    enemies_mm = minimap.find_enemies(mini_frame, cfg)
    player_mm = minimap.find_player_crosshair(mini_frame)

    vis_main = main_frame.copy()
    vis_mm = mini_frame.copy()

    for x, y, w, h in crates:
        cv2.rectangle(vis_main, (x, y), (x + w, y + h), (0, 255, 255), 2)
    for x, y, w, h in mobs:
        cv2.rectangle(vis_main, (x, y), (x + w, y + h), (0, 0, 255), 1)

    for ex, ey in enemies_mm:
        cv2.circle(vis_mm, (ex, ey), 3, (0, 0, 255), -1)
    if player_mm is not None:
        cv2.circle(vis_mm, player_mm, 4, (255, 255, 0), -1)

    logger.info(
        "Test mode: crates=%d mobs=%d enemies_mm=%d player_mm=%s",
        len(crates),
        len(mobs),
        len(enemies_mm),
        player_mm,
    )

    cv2.imshow("Space Aces MAIN (crates/mobs)", vis_main)
    cv2.imshow("Space Aces MINIMAP (player/enemies)", vis_mm)
    print("Press any key in an image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_bot():
    setup_logger()
    bot = Bot()
    try:
        bot.run()
    except KeyboardInterrupt:
        print("Bot stopped by user (KeyboardInterrupt).")


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "calibrate":
        run_calibration()
    elif args.mode == "test":
        run_test()
    else:
        run_bot()
