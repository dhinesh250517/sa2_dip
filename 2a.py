
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
    p.add_argument('--input', '-i', default='DSC_0192.JPG')
    p.add_argument('--block', type=int, default=2)
    p.add_argument('--ksize', type=int, default=3)
    p.add_argument('--k', type=float, default=0.04)
    p.add_argument('--threshold', type=float, default=0.01)
    args = p.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(args.input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_f = np.float32(gray)

    dst = cv2.cornerHarris(gray_f, args.block, args.ksize, args.k)
    dst = cv2.dilate(dst, None)

    out = img.copy()
    mask = dst > args.threshold * dst.max()
    out[mask] = [0, 0, 255]  # mark corners red

    out_path = save_same_folder(args.input, out, "harris_corners")
    print("Saved:", out_path)

if __name__ == '__main__':
    main()
