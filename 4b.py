
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
