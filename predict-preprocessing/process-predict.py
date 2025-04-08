import argparse
import cv2
import numpy
import os
import time
from PIL import Image

# Set a fixed random seed for reproducibility
SEED = 42
numpy.random.seed(SEED)

def resize_img(img, size=(1024, 1024)):
    return cv2.resize(img, size)

def normalize_img(img):
    return img.astype(numpy.float32) / 255.0

def grayscale_img(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def process_batch(batch, size=(1024, 1024)):
    processed_imgs = []
    for img in batch:
        try:
            # Resize, grayscale, and normalize
            img = resize_img(img, size)
            img = normalize_img(img)
            processed_imgs.append(img)
        except Exception as e:
            print(f"ERROR: {e}")
    return processed_imgs

def preprocess_images_or_video(input_path, output_dir, size=(1024, 1024), batch_size=32):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if os.path.isdir(input_path):
        # Process images in a directory
        input_files = [f for f in os.listdir(input_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        batch = []
        
        for i, img_name in enumerate(input_files):
            img_path = os.path.join(input_path, img_name)
            try:
                img = cv2.imread(img_path)
                batch.append(img)
                
                if len(batch) == batch_size or i == len(input_files) - 1:
                    processed_imgs = process_batch(batch, size=size)
                    for j, processed_img in enumerate(processed_imgs):
                        output_path = os.path.join(output_dir, f"{time.time()}.png")
                        Image.fromarray((processed_img * 255).astype(numpy.uint8)).save(output_path)
                    batch.clear()

            except Exception as e:
                print(f"ERROR: {img_name} - {e}")
    
    elif os.path.isfile(input_path) and input_path.endswith(('.mp4', '.avi', '.mov')):
        # Process video
        cap = cv2.VideoCapture(input_path)
        batch = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            batch.append(frame)

            if len(batch) == batch_size or not cap.isOpened():
                processed_frames = process_batch(batch, size=size)
                for j, processed_frame in enumerate(processed_frames):
                    output_path = os.path.join(output_dir, f"{frame_idx + j + 1}.png")
                    Image.fromarray((processed_frame * 255).astype(numpy.uint8)).save(output_path)
                batch.clear()

            frame_idx += 1

        cap.release()

    print("Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fish Friend's Prediction Material Pre-Processing Utility.")
    parser.add_argument("input", nargs="?", default="input")
    parser.add_argument("output", nargs="?", default="output")
    args = parser.parse_args()

    preprocess_images_or_video(args.input, args.output)
