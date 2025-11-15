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
    p.add_argument('--input', '-i', default='DSC_0176.JPG')
    p.add_argument('--kernel', '-k', type=int, default=3)
    args = p.parse_args()
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(args.input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.kernel, args.kernel))
    morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    _, edges_bin = cv2.threshold(morph_grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    overlay = img.copy()
    overlay[edges_bin == 255] = (0, 0, 255)  # red edges
    out1 = save_same_folder(args.input, morph_grad, "morph_gradient")
    out2 = save_same_folder(args.input, edges_bin, "edges_binary")
    out3 = save_same_folder(args.input, overlay, "boundary_overlay")
    print("Saved:", out1, out2, out3)
if __name__ == '__main__':
    main()
