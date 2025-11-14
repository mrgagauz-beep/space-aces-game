import json
import time
from pathlib import Path

import pyautogui


def get_point(prompt: str):
    """Ask user to hover a point and press Enter, then read cursor position."""
    print(prompt)
    input("Press Enter when ready...")
    time.sleep(0.2)
    x, y = pyautogui.position()
    print(f"Recorded position: x={x}, y={y}")
    return x, y


def get_roi(label: str):
    """Interactively capture top-left and bottom-right corners for a ROI."""
    print(f"\n--- Calibrating {label} ROI ---")
    x1, y1 = get_point(
        f"Hover the mouse on the TOP-LEFT corner of the {label} area."
    )
    x2, y2 = get_point(
        f"Hover the mouse on the BOTTOM-RIGHT corner of the {label} area."
    )

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    print(f"{label} ROI: x={x1}, y={y1}, w={width}, h={height}")
    return [x1, y1, width, height]


def load_profile(path: Path):
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_profile(path: Path, data: dict):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("=== Space Aces Bot ROI Calibration ===")
    print("This tool will help you set MAIN and MINIMAP ROIs.\n")

    profile_path = Path(__file__).resolve().parent.parent / "profiles" / "default.json"

    data = load_profile(profile_path)
    data.setdefault("ROI", {})

    main_roi = get_roi("MAIN (game window)")
    minimap_roi = get_roi("MINIMAP")

    data["ROI"]["MAIN"] = main_roi
    data["ROI"]["MINIMAP"] = minimap_roi

    save_profile(profile_path, data)

    print("\nNew ROIs saved to profiles/default.json:")
    print(json.dumps(data["ROI"], ensure_ascii=False, indent=2))
    print("\nCalibration complete.")


if __name__ == "__main__":
    main()
