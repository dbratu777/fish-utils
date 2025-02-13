import cv2
import os
import glob

def seg_vid(vid_path, output_dir, fps=1, resize_dim=(1024, 1024)):
    base_name = os.path.splitext(os.path.basename(vid_path))[0]
    cap = cv2.VideoCapture(vid_path)
    
    if not cap.isOpened():
        print(f"Error: could not open video {vid_path}")
        return
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    
    vid_output_dir = os.path.join(output_dir, base_name)
    if not os.path.exists(vid_output_dir):
        os.makedirs(vid_output_dir)
    
    frame_count = 0
    saved_frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % int(vid_fps * fps) == 0:
            frame_filename = os.path.join(vid_output_dir, f"{base_name}_{saved_frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, cv2.resize(frame, resize_dim))
            saved_frame_count += 1
        
        frame_count += 1

    cap.release()

def seg_all_vids(input_dir, output_dir, fps=1, resize_dim=(1024, 1024)):
    vids = glob.glob(os.path.join(input_dir, '*.mp4'))
    if not vids:
        print(f"ERROR: no videos found: {input_dir}")
        return

    for vid_path in vids:
        seg_vid(vid_path, output_dir, fps, resize_dim)

if __name__ == "__main__":
    seg_all_vids("input", "output")
