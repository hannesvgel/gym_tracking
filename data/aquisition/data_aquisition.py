import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
import csv
from mpl_toolkits.mplot3d import Axes3D
import csv

# Inizializzazione MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_draw = mp.solutions.drawing_utils
connections = list(mp_pose.POSE_CONNECTIONS)

# Caricamento video da file
import tkinter as tk
from tkinter import filedialog

# Finestra per selezionare un file video
root = tk.Tk()
root.withdraw()  # Nasconde la finestra principale

video_path = filedialog.askopenfilename(
    title="Seleziona un file video",
    filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
)

if not video_path:
    print("Nessun file selezionato. Programma terminato.")
    exit()


# Crea e prepara file CSV
csv_file = open("data/raw/pose_data.csv", mode='w', newline='')
csv_writer = csv.writer(csv_file)

# Scrive l'intestazione: x_0, y_0, z_0, visibility_0, ..., x_32, y_32, z_32, visibility_32
header = []
for i in range(33):  # 33 landmarks in MediaPipe Pose
    header += [f"x_{i}", f"y_{i}", f"z_{i}", f"visibility_{i}"]
csv_writer.writerow(header)

cap = cv2.VideoCapture(video_path)

# File CSV per salvare i dati dello scheletro
csv_file = open("data/raw/pose_data.csv", mode='w', newline='')
csv_writer = csv.writer(csv_file)
# Intestazione CSV
header = []
for i in range(33):  # 33 punti nello scheletro MediaPipe
    header += [f"x_{i}", f"y_{i}", f"z_{i}", f"visibility_{i}"]
csv_writer.writerow(header)

# Inizializzazione Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update_plot(pose_landmarks):
    ax.clear()
    ax.view_init(elev=90, azim=-90)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 0.3])

    xs, ys, zs = [], [], []
    for landmark in pose_landmarks.landmark:
        xs.append(landmark.x - 0.5)
        ys.append(-landmark.y + 0.5)
        zs.append(-landmark.z)

    ax.scatter(xs, ys, zs, c='red')
    for connection in connections:
        start, end = connection
        ax.plot([xs[start], xs[end]], [ys[start], ys[end]], [zs[start], zs[end]], 'b')

    plt.draw()
    plt.pause(0.01)

# Ciclo sul video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        update_plot(results.pose_landmarks)
        # Salva dati del frame nel CSV
        row = []
        for lm in results.pose_landmarks.landmark:
            row += [lm.x, lm.y, lm.z, lm.visibility]
        csv_writer.writerow(row)

        # Salva i dati della posa nel CSV
        row = []
        for lm in results.pose_landmarks.landmark:
            row += [lm.x, lm.y, lm.z, lm.visibility]
        csv_writer.writerow(row)

    # Mostra il frame con overlay opzionale
    mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow("Video Pose Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
csv_file.close()
cv2.destroyAllWindows()
plt.close()
csv_file.close()
