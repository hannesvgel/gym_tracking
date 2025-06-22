import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import mediapipe as mp

st.set_page_config(page_title="WebRTC Test", layout="wide")
st.title("WebRTC Camera Test with MediaPipe Skeleton")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_img)
        
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                img,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


col1, col2 = st.columns(2)

with col1:
    st.write("### Live Feed")
    webrtc_streamer(
        key="test",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

with col2:
    st.write("### Instructions")
    st.write("1. Click 'Start' to begin the webcam feed")
    st.write("2. The skeleton overlay will appear automatically")
    st.write("3. Click 'Stop' to end the feed")