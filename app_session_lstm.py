import av
import cv2
import queue
import yaml
import numpy as np
import mediapipe as mp
import tensorflow as tf
import streamlit as st
from collections import Counter, deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# CONFIG 
st.set_page_config(page_title="Live Exercise Classifier", layout="wide")
st.title("Gym Tracking App")

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

MODEL_PATH = config["lstm_bidir_6cl"]["path"]
CLASSES = config["lstm_bidir_6cl"]["class_names"]

KEYPOINT_DIM    = 132
SEQUENCE_LENGTH = 30
DROP_START      = 2
DROP_END        = 2

for key, default in {
    'session_active': False,
    'prev_session_active': False,       # track previous run’s flag
    'exercise_executed': [],
    'exercise_results': None,
    'current_exercise': "No exercise detected",
    'current_confidence': 0.0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# CALLBACKS 
def start_session():
    st.session_state.session_active = True
    st.session_state.exercise_executed = []
    st.session_state.exercise_results = None
    st.session_state.current_exercise = "No exercise detected"
    st.session_state.current_confidence = 0.0
    st.sidebar.success("Session started.")

def stop_session():
    st.session_state.session_active = False
    st.sidebar.success("Session ended.")

st.sidebar.title("Controls")
st.sidebar.button("Start Session", on_click=start_session)
st.sidebar.button("End Session", on_click=stop_session)

# LOAD MODEL 
@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model(MODEL_PATH)

mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

# VIDEO PROCESSOR
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp_drawing
        self.mp_styles  = mp_styles
        self.result_queue = queue.Queue()
        self.session_active = False
        self.sequence = deque(maxlen=SEQUENCE_LENGTH)

    def extract_keypoints(self, results):
        return np.array(
            [v for lm in results.pose_landmarks.landmark for v in (lm.x, lm.y, lm.z, lm.visibility)],
            dtype=np.float32
        )

    def recv(self, frame):
        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        # draw skeleton
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                img_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style()
            )

        # overlay active/inactive
        cv2.putText(img_bgr,
                    "Active" if self.session_active else "Inactive",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # if active, collect & predict
        if self.session_active and results.pose_landmarks:
            self.sequence.append(self.extract_keypoints(results))
            cv2.putText(img_bgr,
                        f"{len(self.sequence)}/{SEQUENCE_LENGTH}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            if len(self.sequence) == SEQUENCE_LENGTH:
                seq_np = np.array(self.sequence).reshape(1, SEQUENCE_LENGTH, KEYPOINT_DIM)
                probs  = model.predict(seq_np, verbose=0)[0]
                idx    = int(np.argmax(probs))
                label  = CLASSES[idx]
                conf   = float(probs[idx])

                # send back to main
                self.result_queue.put((label, conf))
                self.sequence.clear()

                # show it on the frame
                cv2.putText(img_bgr,
                            f"{label} ({conf:.2f})",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")


# LAYOUT
col1, col2 = st.columns(2)

with col1:
    st.write("### Live Feed")
    ctx = webrtc_streamer(
        key="exercise-classifier",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video":True, "audio":False},
        video_processor_factory=VideoProcessor,
        async_processing=True,
    )

    if ctx.video_processor:
        # sync the flag
        ctx.video_processor.session_active = st.session_state.session_active

        # on rising edge, clear old buffers
        if (not st.session_state.prev_session_active) and st.session_state.session_active:
            ctx.video_processor.sequence.clear()
            while not ctx.video_processor.result_queue.empty():
                ctx.video_processor.result_queue.get()

        # always drain any new predictions
        try:
            while True:
                lbl, conf = ctx.video_processor.result_queue.get_nowait()
                st.session_state.current_exercise   = lbl
                st.session_state.current_confidence = conf
                st.session_state.exercise_executed.append(lbl)
        except queue.Empty:
            pass

if st.session_state.prev_session_active and not st.session_state.session_active:
    preds = st.session_state.exercise_executed
    trimmed = preds[DROP_START:len(preds)-DROP_END] if len(preds) > (DROP_START+DROP_END) else []
    st.session_state.exercise_results = Counter(trimmed).most_common(1)[0][0] \
        if trimmed else "Not enough data"

# update prev flag
st.session_state.prev_session_active = st.session_state.session_active

with col2:
    st.write("### Session Results")

    if st.session_state.exercise_results is not None:
        st.write(f"**Primary Exercise Detected:** {st.session_state.exercise_results}")
        
        st.write("**Detailed Exercise Summary:**")
        counts = Counter(st.session_state.exercise_executed)
        for ex, cnt in counts.items():
            percentage = (cnt / len(st.session_state.exercise_executed) * 100) if st.session_state.exercise_executed else 0
            st.write(f"- **{ex.replace('_', ' ').title()}:** {cnt} repetitions ({percentage:.1f}%)")
        
        st.write(f"**Total Predictions:** {len(st.session_state.exercise_executed)}")
