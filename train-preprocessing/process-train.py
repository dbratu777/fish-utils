import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from random import choice
from tqdm import tqdm  # To show a progress bar during batch processing


def resize_image(image, size=(1024, 1024)):
    """Resize image to the specified size."""
    return cv2.resize(image, size)


def normalize_image(image):
    """Normalize pixel values to range [0, 1]."""
    return image / 255.0


def convert_to_grayscale(image):
    """Convert image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def augment_image(image):
    """Apply random augmentation using Albumentations."""
    transform = A.Compose([
        A.PadIfNeeded(min_height=1024, min_width=1024, p=1),
        A.RandomCrop(width=1024, height=1024),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.7),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussianBlur(blur_limit=(3, 7), p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5)
    ])
    augmented = transform(image=image)
    return augmented['image']


def apply_motion_detection(prev_frame, curr_frame):
    """Calculate optical flow between two consecutive frames."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def process_batch(images_batch, size=(1024, 1024)):
    """Preprocess a batch of images."""
    processed_images = []
    for image in images_batch:
        try:
            # Resize, normalize, and apply augmentations
            image_resized = resize_image(image, size)
            image_normalized = normalize_image(image_resized)
            image_augmented = augment_image(image_resized)

            # Optionally save the augmented images (for debugging purposes)
            # Save the processed image (augmented or grayscale based on requirement)
            processed_images.append(image_augmented)
        except Exception as e:
            print(f"Error processing image: {e}")
    return processed_images


def preprocess_images(input_dir, output_dir, size=(1024, 1024), batch_size=32):
    """Preprocess images in batches and apply advanced augmentation."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # List all image files in the input directory
    images = [f for f in os.listdir(input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    prev_frame = None
    batch = []
    
    # Use tqdm for a progress bar
    for i, image_name in tqdm(enumerate(images), total=len(images)):
        image_path = os.path.join(input_dir, image_name)
        
        try:
            image = cv2.imread(image_path)
            batch.append(image)
            
            # If batch is filled, process the batch
            if len(batch) == batch_size or i == len(images) - 1:
                processed_images = process_batch(batch, size=size)
                
                # Save processed images (or do further operations)
                for j, processed_image in enumerate(processed_images):
                    output_path = os.path.join(output_dir, f"processed_{i-batch_size+j+1}.png")
                    Image.fromarray((processed_image * 255).astype(np.uint8)).save(output_path)
                
                # Clear batch for next set of images
                batch.clear()

            # Motion detection (optical flow) between consecutive frames
            if prev_frame is not None:
                flow = apply_motion_detection(prev_frame, image)
                print(f"Flow calculated for frame {i+1} and {i}")
            
            # Update the previous frame for motion detection
            prev_frame = image

        except Exception as e:
            print(f"Error loading or processing {image_name}: {e}")
    
    print("Preprocessing complete.")


# Directory paths (modify with actual directories)
input_dir = '/home/p3/code/utils/train-preprocessing/input/'
output_dir = '/home/p3/code/utils/train-preprocessing/output/'

# Call the preprocess function
preprocess_images(input_dir, output_dir)
