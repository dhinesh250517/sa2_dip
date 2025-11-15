import cv2
import os
import numpy as np


IMAGE_FILE = "input.png"              # your main image
TEMPLATE_FILE = "template.png"           # your template image

METHOD = cv2.TM_CCOEFF_NORMED   # matching method
THRESHOLD = 0.8                 # match threshold (0-1, higher = stricter)

def save_same_folder(inp_path, img, suffix):
    base, ext = os.path.splitext(inp_path)
    out_path = f"{base}_{suffix}{ext}"
    cv2.imwrite(out_path, img)
    return out_path

def main():
    # Load images (same folder as this .py file)
    if not os.path.exists(IMAGE_FILE):
        raise FileNotFoundError(f"Main image not found: {IMAGE_FILE}")

    if not os.path.exists(TEMPLATE_FILE):
        raise FileNotFoundError(f"Template image not found: {TEMPLATE_FILE}")

    img = cv2.imread(IMAGE_FILE)
    tpl = cv2.imread(TEMPLATE_FILE)

    if img is None:
        raise IOError(f"Cannot read image: {IMAGE_FILE}")

    if tpl is None:
        raise IOError(f"Cannot read template: {TEMPLATE_FILE}")

    # Convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    tpl_gray = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)

    h, w = tpl_gray.shape

    # Template matching
    result = cv2.matchTemplate(img_gray, tpl_gray, METHOD)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # For TM_CCOEFF_NORMED: higher = better
    best_loc = max_loc
    match_val = max_val

    out = img.copy()

    if match_val >= THRESHOLD:
        cv2.rectangle(out, best_loc, (best_loc[0]+w, best_loc[1]+h), (0, 255, 0), 2)

    # Save output next to input image
    out_path = save_same_folder(IMAGE_FILE, out, "template_match")
    print(f"Output saved at: {out_path}")
    print(f"Match Value: {match_val}")

if __name__ == "__main__":
    main()
