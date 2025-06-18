from fct_exercise_classification_per_30_frames import classify_per_30_frames

MODEL_PATH = "skeleton_lstm_multiclass3.h5"
CLASSES    = ["pull_up", "push_up", "squat"]  # pull_up: 0, push_up: 1, squat: 2

classify_per_30_frames(MODEL_PATH, CLASSES)