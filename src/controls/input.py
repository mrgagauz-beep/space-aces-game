import pyautogui
import keyboard


# Disable built-in pauses between pyautogui actions for faster response
pyautogui.PAUSE = 0

EXIT_KEY = "esc"


def move_and_click(x, y):
    """
    Move cursor to (x, y) screen coordinates and perform a left click.
    """
    pyautogui.moveTo(x, y)
    pyautogui.click()


def press_key(key):
    """
    Press and release a keyboard key.
    """
    keyboard.press_and_release(key)


def is_exit_pressed():
    """
    Check if the exit key is currently pressed.
    """
    try:
        return keyboard.is_pressed(EXIT_KEY)
    except RuntimeError:
        # In some environments keyboard may not have permissions
        return False

