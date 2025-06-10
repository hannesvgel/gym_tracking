import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import Counter

# ——— CONFIG ———
# MODEL_PATH = "skeleton_cnn_multiclass3.h5"
MODEL_PATH = "skeleton_cnn_multiclass6.h5"
# bench_press: 0, bulgarian_squat: 1, lat_machine: 2, pull_up: 3, push_up: 4, split_squat: 5
CLASSES      = ["bench_press","bulgarian_squat","lat_machine", "pull_up", "push_up", "split_squat"]

KEYPOINT_DIM  = 132  # 33 landmarks × (x,y,z,visibility)
DROP_START    = 5    # number of initial frames to drop
DROP_END      = 5    # number of final frames to drop

# ——— load trained model ———
model = tf.keras.models.load_model(MODEL_PATH)

# ——— init Mediapipe Pose & drawing utils ———
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

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
    return arr.reshape(1, KEYPOINT_DIM, 1)  # (1,132,1)

# ——— start webcam feed ———
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # use cv2.CAP_DSHOW to avoid camera access issues on Windows
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

session_active    = False
exercise_executed = []

print("Press 's' to START session, 'e' to END & show feedback, 'Esc' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # run Mediapipe once per frame
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # single waitKey per frame
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == ord('s') and not session_active:
            session_active = True
            exercise_executed.clear()
            print("Session STARTED.")
        if key == ord('e') and session_active:
            session_active = False
            print("Session ENDED.")
            break

        # draw skeleton if present
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_styles.get_default_pose_landmarks_style()
            )

            # overlay instruction
            if not session_active:
                cv2.putText(frame, "Press 's' to start",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Press 'e' to end",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # classify this frame
                inp   = extract_keypoints_from_results(results)
                probs = model.predict(inp, verbose=0)[0]
                idx   = np.argmax(probs)
                label = CLASSES[idx]
                conf  = probs[idx]
                exercise_executed.append(label)
                # overlay prediction text
                cv2.putText(frame, f"{label} ({conf:.2f})",
                            (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # show frame
        cv2.imshow("Live Exercise Classifier", frame)

    # after loop: if we ended a session, drop first & last frames then give feedback
    if exercise_executed:
        # drop initial and final frames
        start = DROP_START
        end   = -DROP_END if DROP_END > 0 else None
        trimmed = exercise_executed[start:end]

        if trimmed:
            most_common = Counter(trimmed).most_common(1)[0][0]
            print(f"Most common exercise performed: {most_common}")
        else:
            print("Not enough frames after dropping start/end to give feedback.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    pose.close()
