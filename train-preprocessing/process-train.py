import os
import cv2
import numpy
from PIL import Image
import albumentations
from random import choice


def resize_img(img, size=(1024, 1024)):
    return cv2.resize(img, size)


def normalize_img(img):
    return img.astype(numpy.float32) / 255.0


def grayscale_img(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def augment_img(img):
    transform = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(p=0.2),
        albumentations.GaussianBlur(blur_limit=(3, 7), p=0.5),
        albumentations.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5)
    ])
    augmented = transform(image=img)
    return augmented['image']


def apply_motion_detection(prev_frame, curr_frame):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def process_batch(batch, size=(1024, 1024)):
    processed_imgs = []
    for img in batch:
        try:
            img = resize_img(img, size)
            img = normalize_img(img)
            img = augment_img(img)

            processed_imgs.append(img)
        except Exception as e:
            print(f"ERROR: {e}")
    return processed_imgs


def preprocess_images(input_dir, output_dir, size=(1024, 1024), batch_size=32):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    imgs = [f for f in os.listdir(input_dir) if f.endswith(
        ('.jpg', '.png', '.jpeg'))]

    prev_frame = None
    batch = []
    for i, img_name in enumerate(imgs):
        image_path = os.path.join(input_dir, img_name)

        try:
            img = cv2.imread(image_path)
            batch.append(img)

            if len(batch) == batch_size or i == len(imgs) - 1:
                processed_imgs = process_batch(batch, size=size)

                for j, processed_img in enumerate(processed_imgs):
                    output_path = os.path.join(
                        output_dir, f"{i-batch_size+j+1}.png")
                    Image.fromarray(
                        (processed_img * 255).astype(numpy.uint8)).save(output_path)

                batch.clear()

            if prev_frame is not None:
                flow = apply_motion_detection(prev_frame, img)

            prev_frame = img

        except Exception as e:
            print(f"ERROR: {img_name} - {e}")


if __name__ == "__main__":
    preprocess_images("input", "output")
