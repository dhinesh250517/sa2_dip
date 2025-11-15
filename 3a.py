
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
    p.add_argument('--input', '-i', default='DSC_0220.JPG')
    p.add_argument('--simple_thresh', type=int, default=120)
    args = p.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(args.input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, simple = cv2.threshold(gray, args.simple_thresh, 255, cv2.THRESH_BINARY)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    out1 = save_same_folder(args.input, simple, f"simple_thresh_{args.simple_thresh}")
    out2 = save_same_folder(args.input, otsu, "otsu_thresh")
    print("Saved:", out1, out2)

if __name__ == '__main__':
    main()
