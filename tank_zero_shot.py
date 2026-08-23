import cv2
from ultralytics import YOLO

def monitor_full_tank_zeroshot(video_path):
    # Load YOLO-World model (capable of detecting custom text prompts instantly)
    model = YOLO("yolov8s-world.pt")
    
    # Set the custom class you want to detect in the tank
    model.set_classes(["fish"])
    
    cap = cv2.VideoCapture(video_path)
    unique_fish_tracker = set()
    
    print("Starting full-tank zero-shot tracking and counting...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Run tracking with a lower confidence threshold to catch schooling fish
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.15, verbose=False)
        
        current_frame_fish_count = 0
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            
            current_frame_fish_count = len(track_ids)
            
            for box, track_id, conf in zip(boxes, track_ids, confidences):
                unique_fish_tracker.add(track_id)
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding boxes and unique tracking ID for every fish in the tank
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{track_id} fish ({conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Display live tank counts on screen
        total_unique_counted = len(unique_fish_tracker)
        cv2.putText(frame, f"Fish in View: {current_frame_fish_count}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Unique Fish Counted: {total_unique_counted}", (20, 80), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow("Full Tank Smart Monitoring Dashboard", frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor_full_tank_zeroshot("sample_fish.mp4")