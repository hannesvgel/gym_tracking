import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import Counter
import math

# Configurations
MODEL_PATH = "skeleton_lstm_multiclass6.h5"
CLASSES = ["bench_press","bulgarian_squat", "lat_machine", "pull_up","push_up", "split_squat"]
KEYPOINT_DIM = 132
SEQUENCE_LENGTH = 30
DROP_START = 5
DROP_END = 5

# Initialization MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def analyze_exercise(landmarks, exercise_id):
    feedback = ""
    color = (0, 255, 0)  
    
    try:
        if exercise_id == 1:  # bulgarian squat
           
            hip_L = [landmarks[24].x, landmarks[24].y]
            hip_R = [landmarks[23].x, landmarks[23].y]
            knee_L = [landmarks[26].x, landmarks[26].y]
            knee_R = [landmarks[25].x, landmarks[25].y]
            Toe_L = [landmarks[32].x, landmarks[32].y]
            Toe_R =  [landmarks[31].x, landmarks[31].y]
            ankle_L = [landmarks[28].x, landmarks[28].y]
            ankle_R = [landmarks[27].x, landmarks[27].y]

            angle_L = calculate_angle(hip_L, knee_L, ankle_L)
            angle_R = calculate_angle(hip_R, knee_R, ankle_R)

            cond1 = Toe_L[0] >= knee_L[0] 
            cond2 = Toe_R[0] >= Toe_R[0]
            cond3 =  angle_L <= 90 
            cond4 =angle_R <= 90

            if Toe_L > Toe_R:

                if cond1 and cond3:
                    feedback = "Bulgarian squat left leg - Correct Execution!"
                    color = (0, 255, 20)
                elif not cond1 and cond3:
                    feedback = "Left knee further forward than the tip of the foot"
                    color = (0, 165, 255)
                elif not cond3:
                    feedback = "Bend the left leg more, until reaching parallel"
                    color = (0, 165, 255)
            else: 
                if cond2 and cond4:
                    feedback = "Bulgarian squat right leg - Correct Execution!"
                    color = (0, 255, 20)
                elif not cond2 and cond4:
                    feedback = "Right knee further forward than the tip of the foot"
                    color = (0, 165, 255)
                elif not cond4:
                    feedback = "Bend the right leg more, until reaching parallel"
                    color = (0, 165, 255)

        elif exercise_id == 5:  # Split squat
            
            hip_L = [landmarks[24].x, landmarks[24].y]
            hip_R = [landmarks[23].x, landmarks[23].y]
            knee_L = [landmarks[26].x, landmarks[26].y]
            knee_R = [landmarks[25].x, landmarks[25].y]
            Toe_L = [landmarks[32].x, landmarks[32].y]
            Toe_R =  [landmarks[31].x, landmarks[31].y]
            ankle_L = [landmarks[28].x, landmarks[28].y]
            ankle_R = [landmarks[27].x, landmarks[27].y]

            angle_L = calculate_angle(hip_L, knee_L, ankle_L)
            angle_R = calculate_angle(hip_R, knee_R, ankle_R)

            cond1 = Toe_L[0] >= knee_L[0] 
            cond2 = Toe_R[0] >= Toe_R[0]
            cond3 =  angle_L <= 90 
            cond4 =angle_R <= 90

            if Toe_L > Toe_R:

                if cond1 and cond3:
                    feedback = "Split squat left leg - Correct Execution!"
                    color = (0, 255, 20)
                elif not cond1 and cond3:
                    feedback = "Left knee further forward than the tip of the foot"
                    color = (0, 165, 255)
                elif cond1 and not cond3:
                    feedback = "Bend the left leg more, until reaching parallel"
                    color = (0, 165, 255)
            else: 
                if cond2 and cond4:
                    feedback = "Split squat right leg - Correct Execution!"
                    color = (0, 255, 20)
                elif not cond2 and cond4:
                    feedback = "Right knee further forward than the tip of the foot"
                    color = (0, 165, 255)
                elif not cond4:
                    feedback = "Bend the right leg more, until reaching parallel"
                    color = (0, 165, 255)

        elif exercise_id == 4:  # Push up
           
            l_shoulder = [landmarks[12].x, landmarks[12].y]
            r_shoulder = [landmarks[11].x, landmarks[11].y]
            l_hip = [landmarks[24].x, landmarks[24].y]
            r_hip = [landmarks[23].x, landmarks[23].y]
            l_knee = [landmarks[26].x, landmarks[26].y]
            r_knee = [landmarks[25].x, landmarks[25].y]
            l_ankle = [landmarks[28].x, landmarks[28].y]
            r_ankle = [landmarks[27].x, landmarks[27].y]
            l_elbow = [landmarks[14].x, landmarks[14].y]
            r_elbow = [landmarks[13].x, landmarks[13].y]
            l_wrist = [landmarks[16].x, landmarks[16].y]
            r_wrist = [landmarks[15].x, landmarks[15].y]

            ang1 = calculate_angle(l_shoulder, l_hip, l_ankle)
            ang2 = calculate_angle(r_shoulder, r_hip, r_ankle)
            ang3 = calculate_angle(l_hip, l_knee, l_ankle)
            ang4 = calculate_angle(r_hip, r_knee, r_ankle)
            ang5 = calculate_angle(l_wrist, l_elbow, l_shoulder)
            ang6 = calculate_angle(r_wrist, r_elbow, r_shoulder)

            cond1 = 170 <= ang1 <= 190 or 170 <= ang2 <= 190
            cond4 = 170 >= ang1 or 170 >= ang2
            cond5 = ang1 >= 190 or ang2 >= 190
            cond2 = 150 <= ang3 <= 190 or 150 <= ang4 <= 190
            cond3 = ang5 <= 90 or ang6 <= 90

            if cond1 and cond2 and cond3:
                feedback = "Push-up - Correct Execution!"
                color = (0, 255, 20)
            elif cond4:
                feedback = "Raise your pelvis!"
                color = (0, 165, 255)
            elif cond5:
                feedback = "Lower your pelvis!"
                color = (0, 165, 255)
            elif not cond3:
                feedback = "Bend your arms more until they reach parallel!"
                color = (0, 165, 255)

        elif exercise_id == 0:  # Bench press
            l_shoulder = [landmarks[12].x, landmarks[12].y]
            r_shoulder = [landmarks[11].x, landmarks[11].y]
            l_hip = [landmarks[24].x, landmarks[24].y]
            r_hip = [landmarks[23].x, landmarks[23].y]
            l_elbow = [landmarks[14].x, landmarks[14].y]
            r_elbow = [landmarks[13].x, landmarks[13].y]
            l_wrist = [landmarks[16].x, landmarks[16].y]
            r_wrist = [landmarks[15].x, landmarks[15].y]
            l_knee = [landmarks[26].x, landmarks[26].y]
            r_knee = [landmarks[25].x, landmarks[25].y]

            ang1 = calculate_angle(l_elbow, l_shoulder, l_hip)
            ang2 = calculate_angle(r_elbow, r_shoulder, r_hip)
            ang3 = calculate_angle(l_wrist, l_elbow, l_shoulder)
            ang4 = calculate_angle(r_wrist, r_elbow, r_shoulder)
            ang5 = calculate_angle(l_shoulder, l_hip, l_knee)
            ang6 = calculate_angle(r_shoulder, r_hip, r_knee)

            cond1 = ang1 <= 80 or ang2 <= 80
            cond2 = ang3 <= 90 or ang4 <= 90
            cond3 = 160 <= ang5 <= 200 or 160 <= ang6 <= 200

            if cond1 and cond2 and cond3:
                feedback = "Bench-press - Correct Execution!"
                color = (0, 255, 20)
            elif not cond2:
                feedback = "Bend your arms more until they reach parallel!"
                color = (0, 165, 255)

        elif exercise_id == 3:  # pull up
            l_shoulder = [landmarks[12].x, landmarks[12].y]
            r_shoulder = [landmarks[11].x, landmarks[11].y]
            l_hip = [landmarks[24].x, landmarks[24].y]
            r_hip = [landmarks[23].x, landmarks[23].y]
            l_elbow = [landmarks[14].x, landmarks[14].y]
            r_elbow = [landmarks[13].x, landmarks[13].y]
            l_wrist = [landmarks[16].x, landmarks[16].y]
            r_wrist = [landmarks[15].x, landmarks[15].y]
            l_knee = [landmarks[26].x, landmarks[26].y]
            r_knee = [landmarks[25].x, landmarks[25].y]
            l_lowhead = [landmarks[10].x, landmarks[10].y]
            r_lowhead = [landmarks[9].x, landmarks[9].y]

            ang1 = calculate_angle(l_wrist, l_elbow, l_shoulder)
            ang2 = calculate_angle(r_wrist, r_elbow, r_shoulder)
            ang3 = calculate_angle(l_shoulder, l_hip, l_knee)
            ang4 = calculate_angle(r_shoulder, r_hip, r_knee)

            cond1 = l_lowhead[1] < l_wrist[1] and r_lowhead[1] < r_wrist[1] and r_lowhead[1] < l_wrist[1] and l_lowhead[1] < r_wrist[1]
            cond2 = ang1 <= 90 or ang2 <= 90
            cond3 = 160 <= ang3 <= 200 or 160 <= ang4 <= 200

            if cond1 and cond2 and cond3:
                feedback = "Pull-up - Correct Execution!"
                color = (0, 255, 20)
            elif not cond2 or not cond1:
                feedback = "Bend your arms more to overcome the bar with your chin!"
                color = (0, 165, 255)
            elif not cond3:
                feedback = "Align the body!"
                color = (0, 165, 255)

        elif exercise_id == 2:  # Lat machine
            l_shoulder = [landmarks[12].x, landmarks[12].y]
            r_shoulder = [landmarks[11].x, landmarks[11].y]
            l_hip = [landmarks[24].x, landmarks[24].y]
            r_hip = [landmarks[23].x, landmarks[23].y]
            l_elbow = [landmarks[14].x, landmarks[14].y]
            r_elbow = [landmarks[13].x, landmarks[13].y]
            l_wrist = [landmarks[16].x, landmarks[16].y]
            r_wrist = [landmarks[15].x, landmarks[15].y]
            l_knee = [landmarks[26].x, landmarks[26].y]
            r_knee = [landmarks[25].x, landmarks[25].y]
            l_lowhead = [landmarks[10].x, landmarks[10].y]
            r_lowhead = [landmarks[9].x, landmarks[9].y]

            ang1 = calculate_angle(l_wrist, l_elbow, l_shoulder)
            ang2 = calculate_angle(r_wrist, r_elbow, r_shoulder)
            ang3 = calculate_angle(l_shoulder, l_hip, l_knee)
            ang4 = calculate_angle(r_shoulder, r_hip, r_knee)

            cond1 = l_lowhead[1] < l_wrist[1] and r_lowhead[1] < r_wrist[1] and r_lowhead[1] < l_wrist[1] and l_lowhead[1] < r_wrist[1]
            cond2 = ang1 <= 90 or ang2 <= 90
            cond3 = 90 <= ang3 <= 130 or 90 <= ang4 <= 130
            cond4 = 90 >= ang3 or 90 >= ang4
            cond5 = ang3 >= 130 or ang4 >= 130

            if cond1 and cond2 and cond3:
                feedback = "Lat machine - Correct Execution!"
                color = (0, 255, 20)
            elif not cond2:
                feedback = "Bend your arms more to overcome the bar with your chin!"
                color = (0, 165, 255)
            elif cond4:
                feedback = "Tilt your chest back!"
                color = (0, 165, 255)
            elif cond5:
                feedback = "Tilt your chest forward!"
                color = (0, 165, 255)


    except Exception as e:
        print(f"Processing error: {e}")
        feedback = "Analysis not available"
        color = (255, 255, 255)

    return feedback, color

def main():
    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Initialize webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Error opening webcam")

    # Set MediaPipe Pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    sequence = []
    session_active = False
    exercise_history = []
    current_exercise = None
    feedback_text = "Press 's' to start"
    feedback_color = (255, 255, 255)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert and process image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # User input
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):
                session_active = True
                exercise_history = []
                sequence = []
                feedback_text = "Do the exercise"
                print("Session started")
            elif key == ord('e') and session_active:
                session_active = False
                if exercise_history:
                    trimmed = exercise_history[DROP_START:-DROP_END] if DROP_END > 0 else exercise_history[DROP_START:]
                    if trimmed:
                        current_exercise = Counter(trimmed).most_common(1)[0][0]
                        print(f"Main exercise: {CLASSES[current_exercise]}")
                feedback_text = "Analysis completed"
                print("Session ended")

            if session_active and results.pose_landmarks:
                # Obtain keypoints
                kpts = []
                for lm in results.pose_landmarks.landmark:
                    kpts.extend([lm.x, lm.y, lm.z, lm.visibility])
                sequence.append(np.array(kpts, dtype=np.float32))

                # Classification only when we have enugh frame
                if len(sequence) == SEQUENCE_LENGTH:
                    # Model input
                    inp = np.array(sequence).reshape(1, SEQUENCE_LENGTH, KEYPOINT_DIM)
                    
                    # Prevision
                    probs = model.predict(inp, verbose=0)[0]
                    idx = np.argmax(probs)
                    confidence = probs[idx]
                    
                    # update history only if confidance> 0.7
                    if confidence > 0.7:
                        current_exercise = idx
                        exercise_history.append(current_exercise)
                    
                    # quality movement analyses
                    if current_exercise is not None:
                        feedback_text, feedback_color = analyze_exercise(results.pose_landmarks.landmark, current_exercise)
                    
                    # Sliding window
                    sequence.pop(0)

                # Draw stickman
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_styles.get_default_pose_landmarks_style()
                )

            # information overlay
            if current_exercise is not None:
                cv2.putText(image, f"Exercise: {CLASSES[current_exercise]}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(image, feedback_text, 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2)
            
            if session_active:
                cv2.putText(image, "Active session - Press 'e' to end", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(image, "Press 's' to start a new session", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow('Exercise - Classification and analysis', image)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

if __name__ == "__main__":
    main()





























