import cv2
from ultralytics import YOLO

def run_tank_counter():
    # Load the model
    model = YOLO("yolov8m.pt")
    
    cap = cv2.VideoCapture("sample_fish.mp4")
    unique_ids = set()
    
    print("Running widescreen tank tracking and counting...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # --- CHANGE FRAME SIZE HERE ---
        # Change (1280, 720) to a wider widescreen resolution (Width, Height)
        frame = cv2.resize(frame, (1280, 720))
        
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=0.15, verbose=False)
        
        current_count = 0
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            current_count = len(track_ids)
            
            for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                class_name = model.names[int(cls)]
                
                # Automatically override or display custom label since general models misclassify fish
                display_label = "fish" if class_name in [ "fish"] else class_name
                
                unique_ids.add(track_id)
                x1, y1, x2, y2 = map(int, box)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{track_id} {display_label} ({conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Dashboard display
        total_counted = len(unique_ids)
        cv2.putText(frame, f"Fish in View: {current_count}", (30, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"Total Unique Fish Counted: {total_counted}", (30, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow("Smart Tank Counter & Tracker", frame)
        
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tank_counter()