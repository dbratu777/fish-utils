import cv2
import os

import os
import cv2

def create_video_from_images(image_folder, output_video_path, frame_rate=24):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".png")]
    images.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0]))))
    if not images:
        print("No images found in the folder.")
        return

    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)
    height, width, channels = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can use different codecs
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    for image_name in images:
        image_path = os.path.join(image_folder, image_name)
        image = cv2.imread(image_path)
        
        if image is not None:
            video_writer.write(image)
        else:
            print(f"Skipping image {image_name} due to error reading it.")
    
    video_writer.release()
    print(f"Video saved as {output_video_path}")


# Usage example
input_dir = '/home/p3/code/utils/vid-combine/input/'
output_dir = '/home/p3/code/utils/vid-combine/output/gray2.avi'
frame_rate = 24

create_video_from_images(input_dir, output_dir, frame_rate)
