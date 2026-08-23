import os
import cv2
import numpy as np

def create_virtual_dataset(video_path, output_dir="fish_dataset"):
    # Create directory structure for YOLO
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    print("Processing source video to build virtual dataset frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize standard to 640x640
        frame = cv2.resize(frame, (640, 640))
        
        # Split data 80% train, 20% val
        split = "val" if frame_count % 5 == 0 else "train"
        
        # 1. Save Normal Variation
        img_name = f"fish_normal_{frame_count}.jpg"
        img_path = os.path.join(output_dir, "images", split, img_name)
        cv2.imwrite(img_path, frame)
        
        # Create a dummy YOLO label file (class 0)
        label_path = os.path.join(output_dir, "labels", split, f"fish_normal_{frame_count}.txt")
        with open(label_path, "w") as f:
            f.write("0 0.5 0.5 0.3 0.3\n")

        # 2. Simulate "Unhealthy/Abnormal" Variation via Code Augmentation
        abnormal_frame = frame.copy()
        if frame_count % 2 == 0:
            # Draw random white noise spots on the fish body to mimic disease
            cv2.circle(abnormal_frame, (300 + (frame_count % 20), 320), 4, (255, 255, 255), -1)
            cv2.circle(abnormal_frame, (315 + (frame_count % 15), 310), 3, (255, 255, 255), -1)
            
            ab_img_name = f"fish_unhealthy_{frame_count}.jpg"
            ab_img_path = os.path.join(output_dir, "images", split, ab_img_name)
            cv2.imwrite(ab_img_path, abnormal_frame)
            
            ab_label_path = os.path.join(output_dir, "labels", split, f"fish_unhealthy_{frame_count}.txt")
            with open(ab_label_path, "w") as f:
                f.write("1 0.5 0.5 0.3 0.3\n") # Class 1: Unhealthy/Abnormal

        frame_count += 1
        if frame_count > 300: # Limit to 300 frames for quick testing
            break

    cap.release()
    print(f"Dataset generation complete! Saved to '{output_dir}' directory.")

if __name__ == "__main__":
    sample_video = "sample_fish.mp4" 
    if os.path.exists(sample_video):
        create_virtual_dataset(sample_video)
    else:
        print(f"Please rename your video file in VS Code to '{sample_video}'.")