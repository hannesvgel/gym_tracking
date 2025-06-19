import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import Counter


def classify_session_per_30_frames(MODEL_PATH: str, CLASSES: list[str]):
    KEYPOINT_DIM  = 132  # 33 landmarks × (x,y,z,visibility)
    DROP_START    = 5    # number of initial frames to drop
    DROP_END      = 5    # number of final frames to drop
    sequence = []
    SEQUENCE_LENGTH = 30

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
        # build list in the same order as training: (lm0.x, lm0.y, lm0.z, lm0.v, lm1.x, …)
        kpts = []
        for lm in results.pose_landmarks.landmark:
            kpts.extend([lm.x, lm.y, lm.z, lm.visibility])
        return np.array(kpts, dtype=np.float32)  # shape = (132,)

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
                sequence.clear()
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
                                (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # extract keypoints & run LSTM
                    keypoints = extract_keypoints_from_results(results)
                    sequence.append(keypoints)

                    if len(sequence) == SEQUENCE_LENGTH:
                        inp = np.array(sequence).reshape(1, SEQUENCE_LENGTH, KEYPOINT_DIM)
                        probs = model.predict(inp, verbose=0)[0]
                        idx = np.argmax(probs)
                        label = CLASSES[idx]
                        conf = probs[idx]
                        exercise_executed.append(label)

                        # overlay prediction text
                        text = f"{label} ({conf:.2f})"
                        cv2.putText(frame, text,
                                    (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2, cv2.LINE_AA)

                        # Slide the window (optional)
                        sequence.pop(0)


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
