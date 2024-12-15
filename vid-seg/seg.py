import cv2
import os
import glob

def video_to_frames(video_path, output_dir, frame_rate=1, resize_dim=(640, 640)):
    """
    Convert an MP4 video file to JPG images, saving frames at the specified frame rate,
    and resizing them to the specified dimensions.
    
    Args:
    - video_path: Path to the input video file.
    - output_dir: Directory to save the JPG frames.
    - frame_rate: Interval (in seconds) between frames to save. Default is 1 (save every frame).
    - resize_dim: Tuple of (width, height) to resize frames. Default is (640, 640).
    """
    # Get the base name (without extension) of the video file
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    
    if not video_capture.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video_capture.get(cv2.CAP_PROP_FPS)  # Frames per second
    print(f"Processing {base_name}: {total_frames} frames, {fps} FPS")
    
    # Create a directory to save frames (named after the video file)
    video_output_dir = os.path.join(output_dir, base_name)
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)
    
    # Process frames
    frame_number = 0
    saved_frame_count = 0
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break  # End of video

        # Resize the frame to the specified dimensions
        resized_frame = cv2.resize(frame, resize_dim)
        
        # Save frame at the specified interval
        if frame_number % int(fps * frame_rate) == 0:  # Check if it's time to save a frame
            frame_filename = os.path.join(video_output_dir, f"{base_name}_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, resized_frame)
            saved_frame_count += 1
            print(f"Saved {frame_filename}")
        
        frame_number += 1

    # Release video capture
    video_capture.release()
    print(f"Video {base_name} processing complete.")

def process_videos_in_directory(input_dir, output_dir, frame_rate=1, resize_dim=(640, 640)):
    """
    Process all MP4 video files in the specified directory and save frames as JPGs,
    resizing them to the specified dimensions.
    
    Args:
    - input_dir: Directory containing the video files.
    - output_dir: Directory to save the JPG frames.
    - frame_rate: Interval (in seconds) between frames to save. Default is 1 (save every frame).
    - resize_dim: Tuple of (width, height) to resize frames. Default is (640, 640).
    """
    # Get all MP4 files in the directory
    video_files = glob.glob(os.path.join(input_dir, '*.mp4'))
    
    if not video_files:
        print(f"No MP4 files found in {input_dir}")
        return
    
    print(f"Found {len(video_files)} video(s) in {input_dir}")

    # Process each video file
    for video_path in video_files:
        video_to_frames(video_path, output_dir, frame_rate, resize_dim)

if __name__ == "__main__":
    input_dir = '/home/p3/code/utils/vid-seg/input/'
    output_dir = '/home/p3/code/utils/vid-seg/output/'
    frame_rate = 1
    resize_dim = (640, 640)

    process_videos_in_directory(input_dir, output_dir, frame_rate, resize_dim)
