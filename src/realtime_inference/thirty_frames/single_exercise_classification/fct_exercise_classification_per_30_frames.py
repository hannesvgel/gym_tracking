import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf


def classify_per_30_frames(MODEL_PATH: str, CLASSES: list[str]):
    KEYPOINT_DIM = 132  # 33 landmarks with x,y,z,visibility
    sequence = []
    SEQUENCE_LENGTH = 30

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
        # build list in the same order as training: (lm0.x, lm0.y, lm0.z, lm0.v, lm1.x, …)
        kpts = []
        for lm in results.pose_landmarks.landmark:
            kpts.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(kpts, dtype=np.float32)  # shape = (132,)

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

            if not results.pose_landmarks:
                # still show the frame
                cv2.imshow("Live Exercise Classifier", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                continue

            # if pose detected, draw & predict
            if results.pose_landmarks:
                # draw skeleton
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_styles.get_default_pose_landmarks_style()
                )

                # extract keypoints & run LSTM
                keypoints = extract_keypoints_from_results(results)
                sequence.append(keypoints)

                if len(sequence) == SEQUENCE_LENGTH:
                    inp = np.array(sequence).reshape(1, SEQUENCE_LENGTH, KEYPOINT_DIM)
                    probs = model.predict(inp, verbose=0)[0]
                    idx = np.argmax(probs)
                    label = CLASSES[idx]
                    conf = probs[idx]

                    # overlay prediction text
                    text = f"{label} ({conf:.2f})"
                    cv2.putText(frame, text,
                                (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 0), 2, cv2.LINE_AA)

                    # Slide the window (optional)
                    sequence.pop(0)

            # show the annotated frame
            cv2.imshow("Live Exercise Classifier", frame)

            # exit on 'Esc' key
            if cv2.waitKey(1) & 0xFF == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
