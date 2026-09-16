import cv2
import streamlit as st
from ultralytics import YOLO
import paho.mqtt.client as mqtt
import threading
import time
import uuid
import plotly.express as px
import pandas as pd


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Aquirism - Live Fish Health Monitoring",
    layout="wide"
)


# ============================================================
# MQTT CONFIGURATION
# ============================================================

MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883

TOPIC_TEMP = "aquirism/sensor/temperature"
TOPIC_PH = "aquirism/sensor/ph"
TOPIC_TDS = "aquirism/sensor/tds"


# ============================================================
# MQTT MANAGER
# ============================================================

class MQTTManager:

    def __init__(self):
        self.lock = threading.Lock()
        self.temp = "--"
        self.ph = "--"
        self.tds = "--"
        self.connected = False
        self.last_message_time = 0
        self.client = None
        self.start()

    def start(self):
        client_id = "aquirism_dashboard_" + uuid.uuid4().hex[:8]
        self.client = mqtt.Client(
            client_id=client_id,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2
        )
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message

        try:
            print("Connecting to MQTT broker...")
            self.client.connect(MQTT_BROKER, MQTT_PORT, 120)
            self.client.loop_start()
            print("MQTT client started.")
        except Exception as e:
            print("MQTT connection error:", e)

    def on_connect(self, client, userdata, flags, reason_code, properties=None):
        print("MQTT CONNECTED!", reason_code)
        client.subscribe(TOPIC_TEMP)
        client.subscribe(TOPIC_PH)
        client.subscribe(TOPIC_TDS)
        with self.lock:
            self.connected = True

    def on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties=None):
        print("MQTT DISCONNECTED:", reason_code)
        with self.lock:
            self.connected = False

    def on_message(self, client, userdata, msg):
        try:
            payload = msg.payload.decode("utf-8", errors="ignore").strip()
            with self.lock:
                if msg.topic == TOPIC_TEMP:
                    self.temp = payload
                elif msg.topic == TOPIC_PH:
                    self.ph = payload
                elif msg.topic == TOPIC_TDS:
                    self.tds = payload
                self.last_message_time = time.time()
        except Exception as e:
            print("Message processing error:", e)

    def get_data(self):
        with self.lock:
            return {
                "temp": self.temp,
                "ph": self.ph,
                "tds": self.tds,
                "connected": self.connected,
                "last_message_time": self.last_message_time
            }


@st.cache_resource
def get_mqtt_manager():
    return MQTTManager()


mqtt = get_mqtt_manager()


# ============================================================
# SESSION STATE FOR HISTORICAL DATA & TIMESTAMPS
# ============================================================

if "history_time" not in st.session_state:
    st.session_state.history_time = []
if "history_temp" not in st.session_state:
    st.session_state.history_temp = []
if "history_ph" not in st.session_state:
    st.session_state.history_ph = []
if "history_tds" not in st.session_state:
    st.session_state.history_tds = []


# ============================================================
# SENSOR DASHBOARD & PROFESSIONAL PLOTLY CHARTS
# ============================================================

@st.fragment(run_every=1)
def sensor_dashboard():
    data = mqtt.get_data()

    temp = data["temp"]
    ph = data["ph"]
    tds = data["tds"]
    connected = data["connected"]
    last_message = data["last_message_time"]

    # Append data with timestamp for professional charting
    if temp != "--":
        try:
            current_time_label = time.strftime("%H:%M:%S")
            st.session_state.history_time.append(current_time_label)
            st.session_state.history_temp.append(float(temp))
            st.session_state.history_ph.append(float(ph))
            st.session_state.history_tds.append(float(tds))

            # Keep only the last 20 points for clean viewing
            if len(st.session_state.history_temp) > 20:
                st.session_state.history_time.pop(0)
                st.session_state.history_temp.pop(0)
                st.session_state.history_ph.pop(0)
                st.session_state.history_tds.pop(0)
        except ValueError:
            pass

    st.subheader("📊 Live Water Quality Telemetry")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🌡 Water Temperature", f"{temp} °C" if temp != "--" else "-- °C")
    with col2:
        st.metric("🧪 pH Level", ph)
    with col3:
        st.metric("💧 TDS", f"{tds} ppm" if tds != "--" else "-- ppm")

    if last_message > 0:
        seconds_since_data = time.time() - last_message
    else:
        seconds_since_data = 999

    if seconds_since_data < 60:
        st.success("🟢 MQTT Connected | Live sensor data receiving")
    elif connected:
        st.warning("🟡 MQTT Connected | Waiting for sensor data")
    else:
        st.error("🔴 MQTT Disconnected")

    # Professional Charts using Plotly (similar to your reference image with markers & grid)
    st.markdown("### 📈 Real-Time Water Parameter Trends")
    g_col1, g_col2, g_col3 = st.columns(3)

    if st.session_state.history_temp:
        df = pd.DataFrame({
            "Time": st.session_state.history_time,
            "Temperature": st.session_state.history_temp,
            "pH": st.session_state.history_ph,
            "TDS": st.session_state.history_tds
        })

        with g_col1:
            fig_temp = px.line(df, x="Time", y="Temperature", markers=True, title="Temperature Trend (°C)")
            fig_temp.update_traces(line_color="#00ADB5", marker=dict(size=8))
            fig_temp.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_temp, use_container_width=True)

        with g_col2:
            fig_ph = px.line(df, x="Time", y="pH", markers=True, title="pH Level Trend")
            fig_ph.update_traces(line_color="#FF5722", marker=dict(size=8))
            fig_ph.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_ph, use_container_width=True)

        with g_col3:
            fig_tds = px.line(df, x="Time", y="TDS", markers=True, title="TDS Trend (ppm)")
            fig_tds.update_traces(line_color="#4CAF50", marker=dict(size=8))
            fig_tds.update_layout(height=220, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_tds, use_container_width=True)
    else:
        g_col1.info("Waiting for telemetry data...")


sensor_dashboard()

st.markdown("---")


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🎛 Dashboard Controls")

confidence_threshold = st.sidebar.slider(
    "Detection Confidence",
    0.10,
    0.90,
    0.25,
    0.05
)

if "is_running" not in st.session_state:
    st.session_state.is_running = False

if "unique_ids" not in st.session_state:
    st.session_state.unique_ids = set()

start = st.sidebar.button("▶ Start Live Monitoring", use_container_width=True)
stop = st.sidebar.button("⏹ Stop Monitoring", use_container_width=True)

if start:
    st.session_state.is_running = True
if stop:
    st.session_state.is_running = False


# ============================================================
# FISH METRICS
# ============================================================

st.subheader("🐟 Live Fish Health Monitoring")

col1, col2, col3, col4 = st.columns(4)

fish_metric = col1.empty()
unique_metric = col2.empty()
healthy_metric = col3.empty()
unhealthy_metric = col4.empty()

fish_metric.metric("Fish in Current View", "0")
unique_metric.metric("Total Unique Fish", "0")
healthy_metric.metric("Healthy Fish", "0")
unhealthy_metric.metric("⚠️ Unhealthy Fish", "0")

st.markdown("---")
video_placeholder = st.empty()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    print("Loading YOLO model...")
    detector = YOLO("yolov8n.pt")
    health_model = None
    try:
        health_model = YOLO("runs/classify/fish_health_classifier-10/weights/best.pt")
        print("Health classifier loaded.")
    except Exception as e:
        print("Health classifier error:", e)
    return detector, health_model


detector, health_model = load_models()


import os

# ============================================================
# VIDEO CAPTURE INITIALIZATION (Robust Path Handling)
# ============================================================

if "video_cap" not in st.session_state:
    st.session_state.video_cap = None

if st.session_state.video_cap is None or not st.session_state.video_cap.isOpened():
    # Get absolute path relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, "sample_fish.mp4")
    
    cap = cv2.VideoCapture(video_path)
    
    # Fallback if absolute path fails
    if not cap.isOpened():
        cap = cv2.VideoCapture("sample_fish.mp4")
        
    st.session_state.video_cap = cap


# ============================================================
# METRICS CONTAINERS (Fragment ke bahar rakhein taaki blink na karein)
# ============================================================
col1, col2, col3, col4 = st.columns(4)
with col1:
    fish_metric = st.empty()
with col2:
    unique_metric = st.empty()
with col3:
    healthy_metric = st.empty()
with col4:
    unhealthy_metric = st.empty()

# Video display placeholder
video_placeholder = st.empty()

# ============================================================
# FISH VIDEO STREAM FRAGMENT (Sirf video aur tracking ke liye)
# ============================================================
@st.fragment(run_every=1.5)
def fish_monitoring():
    if not st.session_state.is_running:
        video_placeholder.info("Click **▶ Start Live Monitoring** in the sidebar to start fish monitoring.")
        return

    if "video_cap" not in st.session_state or st.session_state.video_cap is None:
        video_placeholder.error("❌ Video capture not initialized.")
        return

    cap = st.session_state.video_cap

    if not cap.isOpened():
        video_placeholder.error("❌ Cannot open 'sample_fish.mp4'.")
        return

    ret, frame = cap.read()

    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    if not ret or frame is None:
        video_placeholder.error("❌ Cannot read frames from video file.")
        return

    # Optimized resolution for smooth cloud streaming
    frame = cv2.resize(frame, (640, 360))

    try:
        results = detector.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence_threshold,
            iou=0.5,
            imgsz=640,
            verbose=False
        )
    except Exception as e:
        video_placeholder.error(f"YOLO Tracking Error: {e}")
        return

    current_count = 0
    healthy_count = 0
    unhealthy_count = 0

    if results and results[0].boxes and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()
        current_count = len(track_ids)

        for box, track_id in zip(boxes, track_ids):
            track_id = int(track_id)
            st.session_state.unique_ids.add(track_id)
            x1, y1, x2, y2 = map(int, box)

            status = "Healthy"
            box_color = (0, 255, 0) # Green

            if health_model is not None:
                h, w, _ = frame.shape
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                if crop.size > 0:
                    try:
                        prediction = health_model(crop, verbose=False)
                        if prediction and prediction[0].probs is not None:
                            top_index = prediction[0].probs.top1
                            class_name = str(health_model.names[top_index])
                            name_lower = class_name.lower()

                            if "unhealthy" in name_lower or "sick" in name_lower or "abnormal" in name_lower:
                                status = "⚠️ UNHEALTHY"
                                box_color = (0, 0, 255) # Red
                                unhealthy_count += 1
                            else:
                                healthy_count += 1
                    except Exception as e:
                        healthy_count += 1
                else:
                    healthy_count += 1
            else:
                healthy_count += 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            label = f"ID:{track_id} {status}"
            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2
            )

    # Update metric UI components smoothly using placeholders
    fish_metric.metric("Fish in Current View", str(current_count))
    unique_metric.metric("Total Unique Fish", str(len(st.session_state.unique_ids)))
    healthy_metric.metric("Healthy Fish", str(healthy_count))
    unhealthy_metric.metric("⚠️ Unhealthy Fish", str(unhealthy_count))

    # Render frame cleanly
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

# Call the fragment function to render
fish_monitoring()

fish_monitoring()
