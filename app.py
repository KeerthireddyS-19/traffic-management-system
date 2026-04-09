import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import math
import time
import base64

# ---------------- Background ----------------
def set_bg():
    try:
        with open("bg.jpg","rb") as f:
            data = f.read()

        encoded = base64.b64encode(data).decode()

        st.markdown(f"""
        <style>
        .stApp {{
            background:
            linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.75)),
            url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True)
    except:
        pass

set_bg()

# ---------------- Title ----------------
st.title("🚦 AI-Based Smart Traffic Management System")

uploaded = st.file_uploader("Upload Traffic Video", type=["mp4","avi","mov"])

# object colors
object_colors = {
    "car": (0,255,0),
    "motorcycle": (255,255,0),
    "bus": (255,0,255),
    "truck": (0,255,255),
    "person": (255,0,0)
}

# store detected objects
if "objects_data" not in st.session_state:
    st.session_state.objects_data = {}

# ---------------- Detection ----------------
if uploaded:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded.read())

    st.success("Video uploaded")

    if st.button("Run Detection"):

        model = YOLO("yolov8n.pt")

        cap = cv2.VideoCapture(tfile.name)

        frame_window = st.empty()

        prev_positions = {}
        prev_time = time.time()

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.4)

            object_count = 0
            current_time = time.time()
            time_diff = current_time - prev_time

            for r in results:
                for box in r.boxes:

                    cls = int(box.cls[0])
                    label = model.names[cls]

                    if label in ["car","motorcycle","bus","truck","person"]:

                        object_count += 1
                        obj_id = object_count

                        x1,y1,x2,y2 = map(int, box.xyxy[0])

                        cx = int((x1+x2)/2)
                        cy = int((y1+y2)/2)

                        speed = 0

                        if obj_id in prev_positions:
                            px,py = prev_positions[obj_id]
                            distance = math.sqrt((cx-px)**2 + (cy-py)**2)
                            speed = int((distance/time_diff)*3)

                            if speed > 120:
                                speed = 120

                        prev_positions[obj_id] = (cx,cy)

                        color = object_colors.get(label,(0,255,0))

                        # draw box
                        cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)

                        # text display
                        if label == "person":

                            cv2.putText(frame,
                                        "Person",
                                        (x1,y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,color,2)

                            st.session_state.objects_data[obj_id] = {
                                "type": "person"
                            }

                        else:

                            cv2.putText(frame,
                                        f"{label} Speed:{speed}",
                                        (x1,y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.6,color,2)

                            st.session_state.objects_data[obj_id] = {
                                "type": label,
                                "speed": speed
                            }

            prev_time = current_time

            # ---------------- Traffic Signal Logic ----------------
            if object_count <= 5:
                signal = "GREEN"
                signal_color = (0,255,0)

            elif object_count <= 15:
                signal = "YELLOW"
                signal_color = (0,255,255)

            else:
                signal = "RED"
                signal_color = (0,0,255)

            # display object count
            cv2.putText(frame,
                        f"Objects Count: {object_count}",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        3)

            # display signal
            cv2.putText(frame,
                        f"Signal: {signal}",
                        (20,80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        signal_color,
                        3)

            frame_window.image(frame, channels="BGR", use_container_width=True)

            time.sleep(0.03)

        cap.release()

        st.success("Detection Completed")

# ---------------- Object Selection ----------------

if st.session_state.objects_data:

    st.subheader("🔍 Select Object")

    selected_id = st.selectbox(
        "Select Object",
        list(st.session_state.objects_data.keys())
    )

    data = st.session_state.objects_data[selected_id]

    st.markdown("### Object Details")

    if data["type"] == "person":

        st.info("""
👤 Person Detected
Type : Person
""")

    else:

        st.success(f"""
🚗 Vehicle Detected

Type : {data['type']}
Speed : {data['speed']} km/h
""")