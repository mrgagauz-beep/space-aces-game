import json
import time
from pathlib import Path

import pyautogui


def get_point(prompt):
    print(prompt)
    input("Press Enter when ready...")
    # Небольшая задержка, чтобы избежать захвата позиции во время нажатия Enter
    time.sleep(0.2)
    x, y = pyautogui.position()
    print(f"Recorded position: x={x}, y={y}")
    return x, y


def load_profile(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_profile(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    print("=== Space Aces Bot ROI Calibration ===")
    print("This tool will help you set the screen region (ROI) for the game window.\n")

    x1, y1 = get_point(
        "Hover the mouse on the TOP-LEFT corner of the game window."
    )
    x2, y2 = get_point(
        "Hover the mouse on the BOTTOM-RIGHT corner of the game window."
    )

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    print(f"\nCalculated ROI: x={x1}, y={y1}, width={width}, height={height}")

    profile_path = Path(__file__).resolve().parent.parent / "profiles" / "default.json"

    try:
        data = load_profile(profile_path)
    except FileNotFoundError:
        print(f"Profile file not found: {profile_path}")
        return

    data.setdefault("roi", {})
    data["roi"]["x"] = x1
    data["roi"]["y"] = y1
    data["roi"]["width"] = width
    data["roi"]["height"] = height

    save_profile(profile_path, data)

    print("\nNew ROI saved to profiles/default.json:")
    print(json.dumps(data["roi"], ensure_ascii=False, indent=2))
    print("\nCalibration complete.")


if __name__ == "__main__":
    main()

