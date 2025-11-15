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
    p.add_argument('--input', '-i', default='DSC_0171.JPG')
    p.add_argument('--kernel', '-k', type=int, default=5)
    p.add_argument('--iterations', type=int, default=1)
    args = p.parse_args()

    img = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read {args.input}")

    # operate on grayscale if single-channel desired; else operate per-channel
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.kernel, args.kernel))
    dilated = cv2.dilate(gray, kernel, iterations=args.iterations)
    eroded = cv2.erode(gray, kernel, iterations=args.iterations)

    saved1 = save_same_folder(args.input, dilated, "dilated")
    saved2 = save_same_folder(args.input, eroded, "eroded")
    print("Saved:", saved1, saved2)

if __name__ == '__main__':
    main()
