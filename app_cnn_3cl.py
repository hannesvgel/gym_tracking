import av
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode


st.set_page_config(page_title="Live Exercise Classifier", layout="wide")
st.title("Gym Tracking App")

MODEL_PATH = "skeleton_cnn_multiclass3.h5"
CLASSES     = ["pull_up", "push_up", "split_squat"]
KEYPOINT_DIM = 132
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

model = load_model(MODEL_PATH)

# -------------- Video Processor --------------
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def extract_keypoints(self, results):
        kpts = []
        for lm in results.pose_landmarks.landmark:
            kpts += [lm.x, lm.y, lm.z, lm.visibility]
        arr = np.array(kpts, dtype=np.float32)
        return arr.reshape(1, KEYPOINT_DIM, 1)  # shape = (1,132,1)

    def recv(self, frame):
        # 1) Get BGR frame
        img_bgr = frame.to_ndarray(format="bgr24")
        # 2) Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self.pose.process(img_rgb)

        # if pose detected, draw & predict
        if results.pose_landmarks:
            # draw skeleton
            self.mp_drawing.draw_landmarks(
                img_bgr,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # extract keypoints & run CNN
            inp = self.extract_keypoints(results)
            probs = model.predict(inp, verbose=0)[0]
            idx = np.argmax(probs)
            label = CLASSES[idx]
            conf = probs[idx]

            # overlay prediction text
            text = f"{label} ({conf:.2f})"
            cv2.putText(
                img_bgr, text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA
            )

        return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

# -------------- Streamlit Layout --------------
col1, col2 = st.columns(2)

# Left: video feed
with col1:
    st.write("### Live Feed")
    webrtc_streamer(
        key="exercise-classifier",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# Right: information
with col2:
    st.write("### Exercise Classes")
    for i, exercise in enumerate(CLASSES, 1):
        st.write(f"{i}. {exercise}") 