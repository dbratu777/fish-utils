import cv2
import os


def img_combine(img_dir, out_dir, fps=24):
    imgs = [img for img in os.listdir(img_dir) if img.endswith(
        ".jpg") or img.endswith(".png")]
    imgs.sort(key=lambda x: int(
        ''.join(filter(str.isdigit, os.path.splitext(x)[0]))))
    if not imgs:
        print(f"ERROR: no images found: {img_dir}.")
        return

    img_path = os.path.join(img_dir, imgs[0])
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_writer = cv2.VideoWriter(out_dir, fourcc, fps, (width, height))

    for img_name in imgs:
        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)

        if img is not None:
            vid_writer.write(img)

    vid_writer.release()


if __name__ == "__main__":
    img_combine("input", "output/test.avi", 24)
