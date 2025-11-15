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

    # reshape pixels to list of samples
    data = img.reshape((-1, 3)).astype(np.float32)

    # If image is extremely large, sample pixels to speed up kmeans initialization.
    sampled_idx = None
    if n_pixels > max_samples:
        # choose a representative random sample for clustering initialization
        rng = np.random.default_rng(seed=42)
        sampled_idx = rng.choice(n_pixels, size=max_samples, replace=False)
        data_sample = data[sampled_idx]
        print(f"[INFO] Image has {n_pixels} pixels — sampling {max_samples} pixels for kmeans init.")
    else:
        data_sample = data

    # Basic validation
    if K <= 0:
        raise ValueError("K must be > 0")
    if data_sample.shape[0] < K:
        raise ValueError(f"Not enough samples ({data_sample.shape[0]}) for K={K}")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.2)

    # cv2.kmeans may raise cv2.error on failure, so keep in try
    try:
        compactness, labels_sample, centers = cv2.kmeans(
            data_sample,
            K,
            None,
            criteria,
            attempts,
            flags=cv2.KMEANS_PP_CENTERS
        )
    except cv2.error as e:
        # Re-raise with additional context
        raise RuntimeError(f"cv2.kmeans failed: {e}")

    # labels_sample shape might be (N,1) or (N,) depending on OpenCV build
    labels_sample = labels_sample.reshape(-1)

    # If we sampled, we need labels for all pixels.
    if sampled_idx is not None:
        # Assign each full pixel to nearest center (faster than re-running kmeans on full data)
        # centers is shape (K,3)
        # compute distances (could be memory heavy if many pixels) — do in chunks
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
