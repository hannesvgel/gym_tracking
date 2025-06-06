import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark, NormalizedLandmarkList


# ——— CONFIG ———
CSV_PATH    = "data/processed/own_DS/30_frame_segments/vertical/pull_up/pull_up_part12_7.csv"
OUTPUT_DIR  = Path("data/validation/own_DS/pull_up")
OUTPUT_DIR.mkdir(exist_ok=True)
IMAGE_W, IMAGE_H = 640, 480  # canvas size for drawing

# ——— Mediapipe setup ———
mp_pose    = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles
# We'll only use the drawing utils, no actual model needed here.

# ——— Read & parse the CSV into shape (30, 33, 4) ———
with open(CSV_PATH, "r") as f:
    lines = [line.strip() for line in f if line.strip()]
    lines = lines[1:]  # skip header line
    # skip last column
    lines = [line.rsplit(",", 1)[0] for line in lines]  # drop last column
# each line has 132 comma-separated floats
data = np.array([list(map(float, line.split(","))) for line in lines], dtype=np.float32)
data = data.reshape(-1, 33, 4)  # → (30 frames, 33 landmarks, 4 values)

# ——— Draw & save each skeleton frame ———
for i, frame_kpts in enumerate(data):
    # Build a NormalizedLandmarkList
    lm_list = NormalizedLandmarkList(
        landmark=[
            NormalizedLandmark(x=pt[0], y=pt[1], z=pt[2], visibility=pt[3])
            for pt in frame_kpts
        ]
    )
    # Blank canvas
    canvas = np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8)

    # Draw skeleton
    mp_drawing.draw_landmarks(
        canvas,
        lm_list,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
    )

    # Save
    out_path = OUTPUT_DIR / f"skeleton_{i:02d}.png"
    cv2.imwrite(str(out_path), canvas)

print(f"Saved {len(data)} skeleton images to {OUTPUT_DIR}")
