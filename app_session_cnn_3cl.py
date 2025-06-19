import time
import av
import cv2
import queue
import numpy as np
import mediapipe as mp
import tensorflow as tf
import streamlit as st
from collections import Counter
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# CONFIG 
st.set_page_config(page_title="Live Exercise Classifier", layout="wide")
st.title("Gym Tracking App")

MODEL_PATH      = "skeleton_cnn_multiclass3.h5"
CLASSES         = ["pull_up", "push_up", "squat"]  # pull_up: 0, push_up: 1, squat: 2 

KEYPOINT_DIM    = 132
SEQUENCE_LENGTH = 1
# timing parameters (in seconds)
WARMUP_SECONDS  = 5
ACTIVE_SECONDS  = 10

# SESSION STATE
for key, default in {
    'session_active': False,      # manual flag
    'start_time': None,           # when Start was pressed
    'prev_session_active': False, # for falling-edge detection
    'exercise_executed': [],
    'exercise_results': None,
    'current_exercise': "No exercise detected",
    'current_confidence': 0.0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# CALLBACKS
def start_session():
    # begin warm-up
    st.session_state.start_time     = time.time()
    st.session_state.session_active = True
    st.session_state.exercise_executed = []
    st.session_state.exercise_results  = None
    st.session_state.current_exercise  = "No exercise detected"
    st.session_state.current_confidence = 0.0
    st.sidebar.success(f"Session starts in {WARMUP_SECONDS}s")


def interrupt_session():
    # user abort
    st.session_state.session_active = False
    st.session_state.start_time      = None
    st.session_state.exercise_executed = []
    st.session_state.exercise_results  = None
    st.session_state.current_exercise  = "No exercise detected"
    st.session_state.current_confidence = 0.0
    st.session_state.prev_session_active = False
    st.sidebar.success("Results Generated")

st.sidebar.title("Controls")
st.sidebar.button("Start Session", on_click=start_session)
st.sidebar.button("Get Results", on_click=interrupt_session)

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
        self.pose           = mp_pose.Pose(min_detection_confidence=0.5,
                                           min_tracking_confidence=0.5)
        self.mp_drawing     = mp_drawing
        self.mp_styles      = mp_styles
        self.result_queue   = queue.Queue()
        self.session_active = False
        self.manual_flag    = False
        self.start_time     = None

    def extract_keypoints(self, results):
        kpts = []
        for lm in results.pose_landmarks.landmark:
            kpts.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(kpts, dtype=np.float32)

    def recv(self, frame):
        # pull in manual flag + start_time from main thread
        flag       = self.manual_flag
        start_time = self.start_time

        # compute timer-based activity
        if flag and start_time is not None:
            elapsed = time.time() - start_time
            self.session_active = (WARMUP_SECONDS <= elapsed < (WARMUP_SECONDS + ACTIVE_SECONDS))
            if not self.session_active and elapsed >= (WARMUP_SECONDS + ACTIVE_SECONDS):
                # Signal session end by putting a special marker in the queue
                self.result_queue.put(("SESSION_END", 1.0))
        else:
            self.session_active = False

        img_bgr = frame.to_ndarray(format="bgr24")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        # skeletal overlay
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                img_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_styles.get_default_pose_landmarks_style()
            )

            if self.session_active:
                # single-frame CNN inference
                kpts  = self.extract_keypoints(results)
                inp   = kpts.reshape(1, KEYPOINT_DIM, 1)
                probs = model.predict(inp, verbose=0)[0]
                idx   = int(np.argmax(probs))
                label = CLASSES[idx]
                conf  = float(probs[idx])

                # enqueue + onscreen
                self.result_queue.put((label, conf))
                cv2.putText(
                    img_bgr,
                    f"{label} ({conf:.2f})",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                )

        # status overlay
        cv2.putText(
            img_bgr,
            "Active" if self.session_active else "Inactive",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# LAYOUT
col1, col2 = st.columns(2)

# sidebar timer info
if st.session_state.start_time is not None:
    elapsed = time.time() - st.session_state.start_time
    if elapsed < WARMUP_SECONDS + ACTIVE_SECONDS:
        st.session_state.session_active = True
    else:
        # auto-stop
        st.session_state.session_active = False

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
        # push flags into processor
        ctx.video_processor.manual_flag = st.session_state.session_active
        ctx.video_processor.start_time  = st.session_state.start_time

        # on rising edge, clear old
        if (not st.session_state.prev_session_active) and ctx.video_processor.session_active:
            while not ctx.video_processor.result_queue.empty():
                ctx.video_processor.result_queue.get()

        # drain predictions
        try:
            while True:
                lbl, conf = ctx.video_processor.result_queue.get_nowait()
                if lbl == "SESSION_END":
                    # Session has ended, compute results
                    preds = st.session_state.exercise_executed
                    if preds:
                        st.session_state.exercise_results = Counter(preds).most_common(1)[0][0]
                    else:
                        st.session_state.exercise_results = "Not enough data"
                else:
                    st.session_state.current_exercise   = lbl
                    st.session_state.current_confidence = conf
                    st.session_state.exercise_executed.append(lbl)
        except queue.Empty:
            pass

# update prev flag only if processor exists
if ctx and ctx.video_processor:
    st.session_state.prev_session_active = ctx.video_processor.session_active

with col2:
    st.write("### Session Results")
    if st.session_state.exercise_results is not None:
        st.write(f"**Primary Exercise Detected:** {st.session_state.exercise_results}")
        st.write("**Detailed Exercise Summary:**")
        counts = Counter(st.session_state.exercise_executed)
        for ex, cnt in counts.items():
            pct = (cnt / len(st.session_state.exercise_executed) * 100) if st.session_state.exercise_executed else 0
            st.write(f"- **{ex.replace('_', ' ').title()}:** {cnt} repetitions ({pct:.1f}%)")
        st.write(f"**Total Predictions:** {len(st.session_state.exercise_executed)}")