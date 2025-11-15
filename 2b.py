import argparse, os
import cv2
import numpy as np
def save_same_folder(inp_path, img, suffix):
    base, ext = os.path.splitext(inp_path)
    out_path = f"{base}_{suffix}{ext}"
    cv2.imwrite(out_path, img)
    return out_path
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='DSC_0218.JPG')
    p.add_argument('--min_area', type=int, default=100)
    args = p.parse_args()
    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(args.input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = [c for c in contours if cv2.contourArea(c) >= args.min_area]
    overlay = img.copy()
    cv2.drawContours(overlay, filtered, -1, (0,255,0), 2)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, filtered, -1, 255, 1)
    out1 = save_same_folder(args.input, overlay, "contours_drawn")
    out2 = save_same_folder(args.input, mask, "contour_mask")
    print("Saved:", out1, out2)
if __name__ == '__main__':
    main()
