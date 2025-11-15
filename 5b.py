
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
