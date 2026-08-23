import cv2
import streamlit as st
from ultralytics import YOLO

# Page Configuration for a widescreen layout
st.set_page_config(
    page_title="Smart Aquarium Dashboard",
    page_icon="🐠",
    layout="wide"
)

st.title("🐠 Smart Aquarium Health & Tracking Dashboard")
st.markdown("Real-time automated computer vision monitoring system for fish counting, unique identification, and health status tracking.")

# Sidebar controls
st.sidebar.header("Dashboard Controls")
confidence_threshold = st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.15, 0.05)
run_button = st.sidebar.button("Start Live Monitoring")
stop_button = st.sidebar.button("Stop Monitoring")

# Dashboard Metrics Layout
col1, col2, col3 = st.columns(3)
with col1:
    metric_fish_view = st.metric(label="Fish in Current View", value="0")
with col2:
    metric_total_unique = st.metric(label="Total Unique Fish Counted", value="0")
with col3:
    metric_status = st.metric(label="System Status", value="Ready")

st.markdown("---")

# Video frame placeholder in the main dashboard view
video_placeholder = st.empty()

if run_button:
    metric_status.metric(label="System Status", value="Running")
    
    # Load model
    model = YOLO("yolov8m.pt")
    cap = cv2.VideoCapture("sample_fish.mp4")
    unique_ids = set()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Resize frame for clean web presentation
        frame = cv2.resize(frame, (960, 540))
        
        # Run YOLO tracking
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=confidence_threshold, verbose=False)
        
        current_count = 0
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.int().cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            current_count = len(track_ids)
            
            for box, track_id, conf, cls in zip(boxes, track_ids, confidences, classes):
                unique_ids.add(track_id)
                x1, y1, x2, y2 = map(int, box)
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"ID:{track_id} fish ({conf:.2f})"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update metrics dynamically on the dashboard
        total_counted = len(unique_ids)
        metric_fish_view.metric(label="Fish in Current View", value=str(current_count))
        metric_total_unique.metric(label="Total Unique Fish Counted", value=str(total_counted))
        
        # Convert OpenCV BGR frame to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB")

    cap.release()
    metric_status.metric(label="System Status", value="Finished")