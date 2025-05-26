import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf

# ——— CONFIG ———
# MODEL_PATH = "skeleton_cnn_multiclass3.h5"
# MODEL_PATH = "skeleton_resnet_multiclass3.h5"
# CLASSES    = ["push_up", "squat", "pull_up"]

MODEL_PATH = "skeleton_cnn_multiclass6.h5"
# MODEL_PATH = "skeleton_resnet_multiclass6.h5"
CLASSES    = ["push_up", "squat", "pull_up", "bench_press", "lat_pulldown", "deadlift"]

KEYPOINT_DIM = 132  # 33 landmarks with x,y,z,visibility

# ——— load trained model ———
model = tf.keras.models.load_model(MODEL_PATH)

# ——— init Mediapipe Pose & etc. ———
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ——— helper to extract keypoints array from Mediapipe results ———
def extract_keypoints_from_results(results):
    kpts = []
    for lm in results.pose_landmarks.landmark:
        kpts += [lm.x, lm.y, lm.z, lm.visibility]
    arr = np.array(kpts, dtype=np.float32)
    return arr.reshape(1, KEYPOINT_DIM, 1)  # shape = (1,132,1)

# ——— start webcam feed ———
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # use cv2.CAP_DSHOW to avoid camera access issues on Windows
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # run Mediapipe once per frame 
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # if pose detected, draw & predict
        if results.pose_landmarks:
            # draw skeleton
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_styles.get_default_pose_landmarks_style()
            )

            # extract keypoints & run CNN
            inp   = extract_keypoints_from_results(results)
            probs = model.predict(inp, verbose=0)[0]
            idx   = np.argmax(probs)
            label = CLASSES[idx]
            conf  = probs[idx]

            # overlay prediction text
            text = f"{label} ({conf:.2f})"
            cv2.putText(frame, text,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

        # show the annotated frame
        cv2.imshow("Live Exercise Classifier", frame)

        # exit on 'Esc' key
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
