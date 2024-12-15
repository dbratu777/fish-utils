import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

# Set a fixed random seed for reproducibility
SEED = 42
np.random.seed(SEED)

def resize_image(image, size=(1080, 1080)):
    return cv2.resize(image, size)

def normalize_image(image):
    return image.astype(np.float32) / 255.0

def grayscale_image(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def process_batch(images_batch, size=(1080, 1080)):
    processed_images = []
    for image in images_batch:
        try:
            # Resize, grayscale, and normalize
            image_resized = resize_image(image, size)
            image_grayscale = grayscale_image(image_resized)
            image_normalized = normalize_image(image_grayscale)
            processed_images.append(image_normalized)
        except Exception as e:
            print(f"Error processing image: {e}")
    return processed_images

def preprocess_images_or_video(input_path, output_dir, size=(1080, 1080), batch_size=32):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if os.path.isdir(input_path):
        # Process images in a directory
        input_files = [f for f in os.listdir(input_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
        batch = []
        
        for i, image_name in tqdm(enumerate(input_files), total=len(input_files)):
            image_path = os.path.join(input_path, image_name)
            try:
                image = cv2.imread(image_path)
                batch.append(image)
                
                if len(batch) == batch_size or i == len(input_files) - 1:
                    processed_images = process_batch(batch, size=size)
                    for j, processed_image in enumerate(processed_images):
                        output_path = os.path.join(output_dir, f"processed_{i-batch_size+j+1}.png")
                        Image.fromarray((processed_image * 255).astype(np.uint8)).save(output_path)
                    batch.clear()

            except Exception as e:
                print(f"Error loading or processing {image_name}: {e}")
    
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
                    output_path = os.path.join(output_dir, f"processed_frame_{frame_idx + j + 1}.png")
                    Image.fromarray((processed_frame * 255).astype(np.uint8)).save(output_path)
                batch.clear()

            frame_idx += 1

        cap.release()

    print("Preprocessing complete.")

input_path = '/home/p3/code/utils/predict-preprocessing/input/gray2.avi'
output_dir = '/home/p3/code/utils/predict-preprocessing/output/'

preprocess_images_or_video(input_path, output_dir)
