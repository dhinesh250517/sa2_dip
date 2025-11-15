
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
