import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
import csv

# === CONFIGURATION ===
# change these to point at your local Kaggle download
INPUT_DIRS = {
    "push_up": r"C:\Users\Lenovo\.cache\kagglehub\datasets\hasyimabdillah\workoutexercises-images\versions\15\push up",
    "squat":   r"C:\Users\Lenovo\.cache\kagglehub\datasets\hasyimabdillah\workoutexercises-images\versions\15\squat",
    "pull_up": r"C:\Users\Lenovo\.cache\kagglehub\datasets\hasyimabdillah\workoutexercises-images\versions\15\pull up",
    "bench_press": r"C:\Users\Lenovo\.cache\kagglehub\datasets\hasyimabdillah\workoutexercises-images\versions\15\bench press",
    "lat_pulldown": r"C:\Users\Lenovo\.cache\kagglehub\datasets\hasyimabdillah\workoutexercises-images\versions\15\lat pulldown",
    "deadlift": r"C:\Users\Lenovo\.cache\kagglehub\datasets\hasyimabdillah\workoutexercises-images\versions\15\deadlift"
}

# base output folder
OUTPUT_BASE = Path("data/processed/kaggle_img_DS")

# create output subdirs
for cls in INPUT_DIRS:
    (OUTPUT_BASE / cls).mkdir(parents=True, exist_ok=True)

# Mediapipe setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5)

def extract_keypoints_from_image(img_path: Path) -> np.ndarray:
    """Run Mediapipe Pose on an image, return flattened keypoints + visibilities."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read {img_path!r}")
    # convert BGR→RGB
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None
    # each landmark has x,y,z,visibility
    keypoints = []
    for lm in results.pose_landmarks.landmark:
        keypoints.extend([lm.x, lm.y, lm.z, lm.visibility])
    return np.array(keypoints, dtype=np.float32)

# === PROCESS ALL ===
for cls, in_dir in INPUT_DIRS.items():
    in_path = Path(in_dir)
    out_path = OUTPUT_BASE / cls

    print(f"\nProcessing class {cls!r}: {in_path} → {out_path}")
    for img_file in in_path.glob("*.jpg"):
        kp = extract_keypoints_from_image(img_file)
        if kp is None:
            print(f"no landmarks for {img_file.name}, skipping")
            continue

        # write one row CSV: [x1, y1, z1, vis1, x2, y2, ...]
        csv_file = out_path / f"{img_file.stem}.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(kp.tolist())

    print(f"done {cls}: {len(list(out_path.glob('*.csv')))} files")

# clean up
pose.close()
print("\nAll done!")
