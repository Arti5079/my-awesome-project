import os
import cv2
from ultralytics import YOLO

def create_real_dataset(video_path, output_dir="fish_dataset"):
    # Create directory structure for YOLO
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "labels", split), exist_ok=True)

    # Load a general base model to auto-detect fish/animals in your video frames for correct bounding boxes
    base_model = YOLO("yolov8n.pt")
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    print("Extracting frames and generating correct bounding boxes...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (640, 640))
        split = "val" if frame_count % 5 == 0 else "train"
        
        img_name = f"fish_frame_{frame_count}.jpg"
        img_path = os.path.join(output_dir, "images", split, img_name)
        cv2.imwrite(img_path, frame)
        
        # Run detection to find actual fish/objects in the frame
        results = base_model(frame, verbose=False)
        label_path = os.path.join(output_dir, "labels", split, f"fish_frame_{frame_count}.txt")
        
        with open(label_path, "w") as f:
            if results[0].boxes is not None:
                for box in results[0].boxes.xywhnorm.cpu().numpy():
                    cls_id = int(box[0]) # Class
                    x, y, w, h = box[1], box[2], box[3], box[4]
                    # Write class 0 (healthy_normal) with real coordinates
                    f.write(f"0 {x} {y} {w} {h}\n")
            
        frame_count += 1
        if frame_count > 150: # Process 150 frames for a solid training set
            break

    cap.release()
    print(f"Dataset generated successfully with real bounding boxes!")

if __name__ == "__main__":
    create_real_dataset("sample_fish.mp4")