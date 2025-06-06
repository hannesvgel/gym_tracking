import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import Counter
import math

# Configurazioni
MODEL_PATH = "skeleton_lstm_multiclass6.h5"
CLASSES = ["push_up", "split_squat", "pull_up", "bench_press", "bulgarian_squat", "lat_machine"]
KEYPOINT_DIM = 132
SEQUENCE_LENGTH = 30
DROP_START = 5
DROP_END = 5

# Inizializzazione MediaPipe
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
    color = (0, 255, 0)  # Verde di default
    
    try:
        if exercise_id == 0:  # Push-up
            # Implementa l'analisi per i piegamenti
            shoulder_L = [landmarks[12].x, landmarks[12].y]
            shoulder_R = [landmarks[11].x, landmarks[11].y]
            hip_L = [landmarks[24].x, landmarks[24].y]
            hip_R = [landmarks[23].x, landmarks[23].y]
            elbow_L = [landmarks[14].x, landmarks[14].y]
            elbow_R = [landmarks[13].x, landmarks[13].y]
            wrist_L = [landmarks[16].x, landmarks[16].y]
            wrist_R = [landmarks[15].x, landmarks[15].y]

            back_angle_L = calculate_angle(shoulder_L, hip_L, [hip_L[0], hip_L[1]-0.1])  # Angolo verticale
            back_angle_R = calculate_angle(shoulder_R, hip_R, [hip_R[0], hip_R[1]-0.1])
            elbow_angle_L = calculate_angle(wrist_L, elbow_L, shoulder_L)
            elbow_angle_R = calculate_angle(wrist_R, elbow_R, shoulder_R)

            if abs(back_angle_L - 180) > 10 or abs(back_angle_R - 180) > 10:
                feedback = "Mantieni la schiena dritta"
                color = (0, 0, 255)
            elif elbow_angle_L > 90 and elbow_angle_R > 90:
                feedback = "Scendi di più"
                color = (0, 165, 255)  # Arancione
            else:
                feedback = "Forma corretta"
                color = (0, 255, 0)

        elif exercise_id == 1:  # Split squat
            # Implementa l'analisi per l'affondo normale
            hip_L = [landmarks[24].x, landmarks[24].y]
            hip_R = [landmarks[23].x, landmarks[23].y]
            knee_L = [landmarks[26].x, landmarks[26].y]
            knee_R = [landmarks[25].x, landmarks[25].y]
            ankle_L = [landmarks[28].x, landmarks[28].y]
            ankle_R = [landmarks[27].x, landmarks[27].y]

            angle_L = calculate_angle(hip_L, knee_L, ankle_L)
            angle_R = calculate_angle(hip_R, knee_R, ankle_R)

            if angle_L < 80 or angle_R < 80:
                feedback = "Troppo in basso"
                color = (0, 0, 255)
            elif angle_L > 100 or angle_R > 100:
                feedback = "Poco profondo"
                color = (0, 165, 255)
            else:
                feedback = "Profondità ottimale"
                color = (0, 255, 0)

        elif exercise_id == 2:  # Piegamenti
            # Estrai i punti necessari
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

            # Calcola gli angoli
            ang1 = calculate_angle(l_shoulder, l_hip, l_ankle)
            ang2 = calculate_angle(r_shoulder, r_hip, r_ankle)
            ang3 = calculate_angle(l_hip, l_knee, l_ankle)
            ang4 = calculate_angle(r_hip, r_knee, r_ankle)
            ang5 = calculate_angle(l_wrist, l_elbow, l_shoulder)
            ang6 = calculate_angle(r_wrist, r_elbow, r_shoulder)

            # Verifica condizioni
            cond1 = 170 <= ang1 <= 190 or 170 <= ang2 <= 190
            cond2 = 150 <= ang3 <= 190 or 150 <= ang4 <= 190
            cond3 = ang5 <= 90 or ang6 <= 90

            if cond1 and cond2 and cond3:
                feedback = "Piegamento corretto!"
                color = (0, 255, 0)
            else:
                feedback = "Correggi la forma!"
                color = (0, 0, 255)

        elif exercise_id == 3:  # Panca
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
                feedback = "Bench-press ok!"
                color = (0, 255, 0)
            else:
                feedback = "Correggi la forma!"
                color = (0, 0, 255)

        elif exercise_id == 4:  # Trazioni
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
                feedback = "Pull-up ok!"
                color = (0, 255, 0)
            else:
                feedback = "Correggi la forma!"
                color = (0, 0, 255)

        elif exercise_id == 5:  # Lat machine
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

            if cond1 and cond2 and cond3:
                feedback = "Lat machine ok!"
                color = (0, 255, 0)
            else:
                feedback = "Correggi la forma!"
                color = (0, 0, 255)


    except Exception as e:
        print(f"Errore nell'analisi: {e}")
        feedback = "Analisi non disponibile"
        color = (255, 255, 255)

    return feedback, color

def main():
    # Carica il modello
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Inizializza webcam
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Errore nell'apertura della webcam")

    # Inizializza MediaPipe Pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Variabili di stato
    sequence = []
    session_active = False
    exercise_history = []
    current_exercise = None
    feedback_text = "Premi 's' per iniziare"
    feedback_color = (255, 255, 255)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Converti e processa l'immagine
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Gestione input utente
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('s'):
                session_active = True
                exercise_history = []
                sequence = []
                feedback_text = "Esegui l'esercizio"
                print("Sessione iniziata")
            elif key == ord('e') and session_active:
                session_active = False
                if exercise_history:
                    trimmed = exercise_history[DROP_START:-DROP_END] if DROP_END > 0 else exercise_history[DROP_START:]
                    if trimmed:
                        current_exercise = Counter(trimmed).most_common(1)[0][0]
                        print(f"Esercizio principale: {CLASSES[current_exercise]}")
                feedback_text = "Analisi completata"
                print("Sessione terminata")

            # Se è attiva una sessione e sono rilevati landmarks
            if session_active and results.pose_landmarks:
                # Estrai keypoints
                kpts = []
                for lm in results.pose_landmarks.landmark:
                    kpts.extend([lm.x, lm.y, lm.z, lm.visibility])
                sequence.append(np.array(kpts, dtype=np.float32))

                # Quando abbiamo abbastanza frame, classifichiamo
                if len(sequence) == SEQUENCE_LENGTH:
                    # Prepara l'input per il modello
                    inp = np.array(sequence).reshape(1, SEQUENCE_LENGTH, KEYPOINT_DIM)
                    
                    # Previsione
                    probs = model.predict(inp, verbose=0)[0]
                    idx = np.argmax(probs)
                    confidence = probs[idx]
                    
                    # Aggiorna lo storico solo se la confidenza è sufficiente
                    if confidence > 0.7:
                        current_exercise = idx
                        exercise_history.append(current_exercise)
                    
                    # Analisi qualità movimento
                    if current_exercise is not None:
                        feedback_text, feedback_color = analyze_exercise(results.pose_landmarks.landmark, current_exercise)
                    
                    # Sliding window
                    sequence.pop(0)

                # Disegna lo scheletro
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_styles.get_default_pose_landmarks_style()
                )

            # Overlay informazioni
            if current_exercise is not None:
                cv2.putText(image, f"Esercizio: {CLASSES[current_exercise]}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.putText(image, feedback_text, 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2)
            
            if session_active:
                cv2.putText(image, "Sessione attiva - Premi 'e' per terminare", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                cv2.putText(image, "Premi 's' per iniziare una nuova sessione", 
                           (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            cv2.imshow('Esercizio - Classificazione e Analisi', image)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()

if __name__ == "__main__":
    main()





























