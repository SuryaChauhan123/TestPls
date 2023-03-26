import pyautogui
# Get a list of windows that contain the word "Chrome" in their title
chrome_windows = pyautogui.getWindowsWithTitle("Chrome")

# If there are no Chrome windows, exit the script
if not chrome_windows:
    print("No Chrome windows found")
    exit()

# Activate the first Chrome window in the list
chrome_windows[0].activate()