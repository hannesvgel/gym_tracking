import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
from tkinter import filedialog, Tk, simpledialog
import matplotlib.pyplot as plt

# PARAMETERS
SEGMENT_FRAME_COUNT = 100
OUTPUT_FOLDER = "data\processed\kaggle_DS"
SHOW_3D = False  # change to True to show 3D stickman

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# MEDIAPIPE SETUP
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)
mp_draw = mp.solutions.drawing_utils
connections = list(mp_pose.POSE_CONNECTIONS)

# GUI FOR FILE AND LABEL
Tk().withdraw()
video_path = filedialog.askopenfilename(title="Select exercise video")
if not video_path:
    print("No video selected.")
    exit()

# exctract video name from path (e.g. videos/pu/push-up_44.mp4 -> push-up_44)
video_name = os.path.basename(video_path).split('.')[0]

# select label (& output folder) based on video name
if 'plank' in video_name:
    label = 'plank'
elif 'push-up' in video_name:
    label = 'push_up'
elif 'russian' in video_name:
    label = 'russian_twist'
elif 'squat' in video_name:
    label = 'squat'
elif 'tricep' in video_name:
    label = 'triceps_dips'

# VIDEO AND STORAGE
cap = cv2.VideoCapture(video_path)
cv2.namedWindow("Segment the video", cv2.WINDOW_NORMAL)
frame_idx = 0
segments = []
current_start = None
all_landmarks = []

# STICKMAN 3D SETUP
if SHOW_3D:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()

def draw_stickman_3d(landmarks):
    ax.clear()
    ax.view_init(elev=90, azim=-90)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect([1, 1, 0.3])
    xs, ys, zs = [], [], []
    for lm in landmarks.landmark:
        xs.append(lm.x - 0.5)
        ys.append(-lm.y + 0.5)
        zs.append(-lm.z)
    ax.scatter(xs, ys, zs, c='red')
    for conn in connections:
        ax.plot([xs[conn[0]], xs[conn[1]]],
                [ys[conn[0]], ys[conn[1]]],
                [zs[conn[0]], zs[conn[1]]], 'b')
    plt.draw()
    plt.pause(0.001)

print("Press 's' to START segment, 'e' to END segment, 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    frame_landmarks = []
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            frame_landmarks.extend([lm.x, lm.y, lm.z, lm.visibility])
    else:
        frame_landmarks = [0.0] * (33 * 4)
    all_landmarks.append(frame_landmarks)

    # Draw stickman over the subject
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        if SHOW_3D:
            draw_stickman_3d(results.pose_landmarks)

    # Display the frame
    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Segment the video", frame)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('s'):
        current_start = frame_idx
        print(f"[Start] Segment started at frame {frame_idx}")
    elif key == ord('e') and current_start is not None:
        segments.append((current_start, frame_idx))
        print(f"[End] Segment saved from {current_start} to {frame_idx}")
        current_start = None
    elif key == ord('q'):
        break

    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
if SHOW_3D:
    plt.ioff()
    plt.close()

# SAVE THE SEGMENTS AS CSV
for i, (start, end) in enumerate(segments):
    segment = all_landmarks[start:end+1]

    if len(segment) < SEGMENT_FRAME_COUNT:
        repeat_factor = SEGMENT_FRAME_COUNT // len(segment) + 1
        segment = (segment * repeat_factor)[:SEGMENT_FRAME_COUNT]
    else:
        indices = np.linspace(0, len(segment)-1, SEGMENT_FRAME_COUNT).astype(int)
        segment = [segment[idx] for idx in indices]

    df = pd.DataFrame(segment, columns=[
        f"{coord}_{i}" for i in range(33) for coord in ['x', 'y', 'z', 'vis']
    ])
    df["label"] = label

    output_path = os.path.join(OUTPUT_FOLDER, label, f"{video_name}_segment_{i+1}.csv")
    df.to_csv(output_path, index=False)
    print(f"Segment {i+1} saved: {output_path}")
