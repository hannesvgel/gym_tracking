import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- PARAMETRI DI CONFIGURAZIONE ---
SEGMENT_LENGTH = 30
MIN_FINAL_SEGMENT = 20
SCALE_FACTOR = 0.5
INPUT_ROOT = r'C:\Users\edoar\gym_tracking\data\raw\drive-download\vertical'
OUTPUT_ROOT = r'C:\Users\edoar\gym_tracking\data\processed\own_DS\30_frame_segments\vertical'
SHOW_3D = False  # Abilita la visualizzazione 3D
# Parametri per miglioramento immagine
CONTRAST_ALPHA = 1.0  # Valore tra 1.0-3.0 (1.0 = nessun cambiamento)
BRIGHTNESS_BETA = 0  # Valore tra 0-100 (0 = nessun cambiamento)
# --- INIZIALIZZAZIONE MEDIAPIPE ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True
)
mp_draw = mp.solutions.drawing_utils

# Connessioni per la visualizzazione 3D
connections = mp_pose.POSE_CONNECTIONS

# --- FUNZIONI DI SUPPORTO ---
def upsample_segment(segment, target_length=30):
    """Esegue l'upsampling di un segmento usando interpolazione lineare"""
    if len(segment) >= target_length:
        return segment[:target_length]
    
    x = np.arange(len(segment))
    x_new = np.linspace(0, len(segment)-1, target_length)
    
    segment_array = np.array(segment)
    upsampled = np.zeros((target_length, segment_array.shape[1]))
    
    for col in range(segment_array.shape[1]):
        upsampled[:, col] = np.interp(x_new, x, segment_array[:, col])
    
    return upsampled.tolist()

def save_segment(segment, label, output_folder, base_name, video_file, is_final=False):
    """Salva il segmento come file CSV con la struttura richiesta"""
    df = pd.DataFrame(segment, columns=[
        f"{coord}_{i}" for i in range(33) for coord in ['x', 'y', 'z', 'vis']
    ])
    df['label'] = label
    
    video_base = os.path.splitext(video_file)[0]
    output_path = os.path.join(output_folder, f"{base_name}_{video_base}.csv")
    
    df.to_csv(output_path, index=False)
    status = "finale upsampled" if is_final else "completo"
    print(f"Salvato {status}: {output_path}")

def enhance_image(frame):
    """Migliora contrasto e luminosità dell'immagine"""
    # Converti a float per operazioni più precise
    frame = frame.astype('float32')
    frame = frame * CONTRAST_ALPHA + BRIGHTNESS_BETA
    # Clip i valori tra 0 e 255 e riconverti a uint8
    frame = np.clip(frame, 0, 255).astype('uint8')
    return frame

def draw_stickman_3d(landmarks, ax):
    """Disegna lo stickman 3D con la visualizzazione specificata"""
    ax.clear()
    ax.view_init(elev=90, azim=-90)  # Vista dall'alto
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect([1, 1, 0.3])  # Rapporto d'aspetto personalizzato
    
    # Estrai e normalizza le coordinate
    xs = [lm.x - 0.5 for lm in landmarks.landmark]
    ys = [-lm.y + 0.5 for lm in landmarks.landmark]
    zs = [-lm.z for lm in landmarks.landmark]
    
    # Disegna i punti
    ax.scatter(xs, ys, zs, c='red', s=20)
    
    # Disegna le connessioni
    for conn in connections:
        ax.plot([xs[conn[0]], xs[conn[1]]],
                [ys[conn[0]], ys[conn[1]]],
                [zs[conn[0]], zs[conn[1]]], 'b-', linewidth=1)
    
    plt.draw()
    plt.pause(0.01)

# --- FUNZIONE PRINCIPALE DI ELABORAZIONE ---
def process_folder(folder_path, label, output_base):
    """Elabora tutti i video in una cartella assegnando lo stesso label"""
    cap = None
    try:
        for video_file in os.listdir(folder_path):
            if video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(folder_path, video_file)
                cap = cv2.VideoCapture(video_path)
                
                folder_name = os.path.basename(folder_path)
                output_folder = os.path.join(output_base, folder_name)
                os.makedirs(output_folder, exist_ok=True)
                
                frame_idx = 0
                segment_counter = 1
                current_segment = []
                
                # Configura finestra di visualizzazione
                window_name = f"Analisi: {folder_name} - Label {label}"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                
                # Setup plot 3D se abilitato
                if SHOW_3D:
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    plt.ion()  # Modalità interattiva
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Preprocess frame
                    # Preprocess frame - Fase 1: Migliora qualità
                    frame = enhance_image(frame)
                    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Rilevamento landmark
                    results = pose.process(img_rgb)
                    
                    # Disegna stickman 2D
                    if results.pose_landmarks:
                        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        
                        # Visualizzazione 3D
                        if SHOW_3D:
                            draw_stickman_3d(results.pose_landmarks, ax)
                    
                    # Visualizzazione frame con overlay
                    display_frame = cv2.resize(frame, (0,0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
                    cv2.putText(display_frame, 
                               f"{folder_name} - Frame: {frame_idx}", 
                               (10,30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, 
                               (0,255,0), 
                               2)
                    cv2.imshow(window_name, display_frame)
                    
                    # Estrazione dati landmark
                    frame_data = []
                    if results.pose_landmarks:
                        for lm in results.pose_landmarks.landmark:
                            frame_data.extend([lm.x, lm.y, lm.z, lm.visibility])
                    else:
                        frame_data = [0.0] * (33 * 4)
                    current_segment.append(frame_data)
                    
                    # Gestione segmenti completi
                    if len(current_segment) == SEGMENT_LENGTH:
                        save_segment(current_segment, label, output_folder, 
                                   f"{folder_name}_part{segment_counter}", video_file)
                        segment_counter += 1
                        current_segment = []
                    
                    frame_idx += 1
                    
                    # Controllo uscita anticipata
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Gestione ultimo segmento
                if len(current_segment) >= MIN_FINAL_SEGMENT:
                    upsampled = upsample_segment(current_segment)
                    save_segment(upsampled, label, output_folder,
                               f"{folder_name}_part{segment_counter}", video_file, True)
                
                cap.release()
                cv2.destroyWindow(window_name)
                if SHOW_3D:
                    plt.close(fig)
                
    except Exception as e:
        print(f"Errore elaborazione {folder_path}: {str(e)}")
    finally:
        if cap and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        if SHOW_3D and 'fig' in locals():
            plt.close(fig)

# --- ESECUZIONE PRINCIPALE ---
if __name__ == "__main__":
    # Crea struttura directory di output
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    # Ordina le cartelle per nome per mantenere l'ordine dei label
    folders = sorted([
        f for f in os.listdir(INPUT_ROOT) 
        if os.path.isdir(os.path.join(INPUT_ROOT, f))
    ])
    
    # Elabora ogni categoria
    for label, folder_name in enumerate(folders):
        folder_path = os.path.join(INPUT_ROOT, folder_name)
        print(f"\nElaborazione categoria {label}: {folder_name}")
        process_folder(folder_path, label, OUTPUT_ROOT)
    
    print("\nElaborazione completata per tutte le categorie!")