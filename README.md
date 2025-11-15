# 1.Morphological Operations:
## a.How do you apply dilation and erosion on an image using OpenCV in Python? (5)
```
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

```
### output:
## b.Write a Python program to detect image boundaries using morphological operations. (15)
```
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

```
### output:


# 2.Feature Detection:
## a.How can you detect corners in an image using the Harris Corner Detector in Python? (5)
```
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

```
### output:
## b.Write a Python program to detect and draw contours in an image. (15)
```
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

```

### output:
# 3.Image Segmentation:
## a.How would you perform image segmentation using thresholding in Python? (5)
```
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

```
### output:
## b.Write a Python program to implement k-means clustering for image segmentation. (15)
```
import argparse
import os
import sys
import traceback
import cv2
import numpy as np
def save_same_folder(inp_path, img, suffix):
    base, ext = os.path.splitext(inp_path)
    out_path = f"{base}_{suffix}{ext}"
    ok = cv2.imwrite(out_path, img)
    if not ok:
        raise IOError(f"Failed to write output image to {out_path}")
    return out_path
def run_kmeans(img, K=4, attempts=10, max_samples=1000000):
    h, w = img.shape[:2]
    n_pixels = h * w
    data = img.reshape((-1, 3)).astype(np.float32)
    sampled_idx = None
    if n_pixels > max_samples:
        rng = np.random.default_rng(seed=42)
        sampled_idx = rng.choice(n_pixels, size=max_samples, replace=False)
        data_sample = data[sampled_idx]
        print(f"[INFO] Image has {n_pixels} pixels — sampling {max_samples} pixels for kmeans init.")
    else:
        data_sample = data

    if K <= 0:
        raise ValueError("K must be > 0")
    if data_sample.shape[0] < K:
        raise ValueError(f"Not enough samples ({data_sample.shape[0]}) for K={K}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)

    try:
        compactness, labels_sample, centers = cv2.kmeans(
            data_sample,
            K,
            None,
            criteria,
            attempts,
            flags=cv2.KMEANS_PP_CENTERS
        )
    except cv2.error as e
        raise RuntimeError(f"cv2.kmeans failed: {e}")

    labels_sample = labels_sample.reshape(-1)

    if sampled_idx is not None:
        print("[INFO] Assigning full pixel set to nearest centers (no reclustering).")
        centers_f = centers.astype(np.float32)
        full_labels = np.empty((n_pixels,), dtype=np.int32)

        batch = 2000000  # process in batches of pixels to control memory
        for start in range(0, n_pixels, batch):
            end = min(n_pixels, start + batch)
            block = data[start:end]  # shape (block_size, 3)
            # compute squared distances to centers: (block_size, K)
            dists = np.sum((block[:, None, :] - centers_f[None, :, :]) ** 2, axis=2)
            full_labels[start:end] = np.argmin(dists, axis=1)
        labels = full_labels
    else:
        # We did not sample: labels_sample corresponds to all pixels
        labels = labels_sample

    # Build result image using centers and labels
    centers_uint8 = np.clip(centers, 0, 255).astype(np.uint8)
    segmented = centers_uint8[labels].reshape((h, w, 3))
    return segmented
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='DSC_0237.JPG', help='Input image path')
    parser.add_argument('--k', type=int, default=4, help='Number of clusters')
    parser.add_argument('--attempts', type=int, default=10, help='KMeans attempts')
    parser.add_argument('--max-samples', type=int, default=1000000, help='Max pixels to sample for kmeans init')
    args = parser.parse_args()

    inp = args.input
    if not os.path.isfile(inp):
        print(f"[ERROR] Input file not found: {inp}")
        print(" - Make sure you run script from the folder containing the image,")
        print("   or pass full path with --input \"C:/path/to/image.jpg\"")
        sys.exit(2)

    # Read image
    img = cv2.imread(inp, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[ERROR] cv2.imread returned None for: {inp}")
        sys.exit(3)

    try:
        segmented = run_kmeans(img, K=args.k, attempts=args.attempts, max_samples=args.max_samples)
        out = save_same_folder(inp, segmented, f"kmeans_k{args.k}")
        print("[OK] Saved:", out)
    except Exception as e:
        print("[ERROR] Exception raised during processing:")
        traceback.print_exc()

if __name__ == '__main__':
    main()

```
### output:
# 4.Object Detection:
## a.How can you use template matching in Python to detect objects in an image? (5)
```
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

```
### output:
## b.Write a Python program to detect faces in an image using OpenCV’s pre-trained models. (15)
```

import argparse, os
import cv2

def save_same_folder(inp_path, img, suffix):
    base, ext = os.path.splitext(inp_path)
    out_path = f"{base}_{suffix}{ext}"
    cv2.imwrite(out_path, img)
    return out_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='DSC_0356.JPG')
    p.add_argument('--scale', type=float, default=1.1)
    p.add_argument('--min_neighbors', type=int, default=5)
    p.add_argument('--min_size', type=int, default=30)
    args = p.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(args.input)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = cascade.detectMultiScale(gray, scaleFactor=args.scale, minNeighbors=args.min_neighbors,
                                     minSize=(args.min_size, args.min_size))

    out = img.copy()
    for (x,y,w,h) in faces:
        cv2.rectangle(out, (x,y), (x+w, y+h), (0,255,0), 2)

    out_path = save_same_folder(args.input, out, "faces_detected")
    print(f"Saved: {out_path} -- detected {len(faces)} faces")

if __name__ == '__main__':
    main()

```
### output:
# 5.Image Compression:
## a.How would you compress an image using Discrete Cosine Transform (DCT) in Python? (5)
```

import argparse, os
import cv2
import numpy as np

def save_same_folder(inp_path, img, suffix):
    base, ext = os.path.splitext(inp_path)
    out_path = f"{base}_{suffix}{ext}"
    cv2.imwrite(out_path, img)
    return out_path

def block_process_dct(channel, block_size=8, keep_fraction=0.1):
    h, w = channel.shape
    pad_h = (block_size - (h % block_size)) % block_size
    pad_w = (block_size - (w % block_size)) % block_size
    padded = np.pad(channel, ((0,pad_h),(0,pad_w)), mode='reflect').astype(np.float32)
    H, W = padded.shape
    out = np.zeros_like(padded, dtype=np.float32)
    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            block = padded[i:i+block_size, j:j+block_size]
            d = cv2.dct(block)
            flat = np.abs(d).flatten()
            k = max(1, int(len(flat) * keep_fraction))
            thresh = np.partition(flat, -k)[-k]
            d[np.abs(d) < thresh] = 0
            block_idct = cv2.idct(d)
            out[i:i+block_size, j:j+block_size] = block_idct
    return out[:h, :w]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='DSC_0261.JPG')
    p.add_argument('--q', type=float, default=0.1)
    p.add_argument('--block', type=int, default=8)
    args = p.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        raise FileNotFoundError(args.input)

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    Y_rec = block_process_dct(Y.astype(np.float32), block_size=args.block, keep_fraction=args.q)
    Y_rec = np.clip(Y_rec, 0, 255).astype(np.uint8)
    rec = cv2.merge([Y_rec, Cr, Cb])
    rec_bgr = cv2.cvtColor(rec, cv2.COLOR_YCrCb2BGR)

    out_path = save_same_folder(args.input, rec_bgr, f"dct_q{str(args.q).replace('.', '_')}")
    print("Saved:", out_path)

if __name__ == '__main__':
    main()

```
### output:
## b.Write a Python program to save an image with reduced file size without losing quality. (15)
```

import argparse, os
from PIL import Image

def save_same_folder(inp_path, pil_img, suffix, ext=None, **save_kwargs):
    base, orig_ext = os.path.splitext(inp_path)
    ext = ext if ext else orig_ext
    out_path = f"{base}_{suffix}{ext}"
    pil_img.save(out_path, **save_kwargs)
    return out_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='DSC_0237.JPG')
    p.add_argument('--jpeg_quality', type=int, default=95)
    args = p.parse_args()

    img = Image.open(args.input)
    base, ext = os.path.splitext(args.input)
    extl = ext.lower()
    if extl == '.png':
        out = save_same_folder(args.input, img, "optimized", ext=".png", optimize=True, compress_level=9)
    else:
        rgb = img.convert('RGB')
        out = save_same_folder(args.input, rgb, "optimized", ext=".jpg", optimize=True,
                               quality=args.jpeg_quality, progressive=True)
    print("Saved:", out)

if __name__ == '__main__':
    main()

```
### output:
