from fct_exercise_classification_per_30_frames import classify_per_30_frames

MODEL_PATH = "skeleton_lstm_multiclass3.h5"
CLASSES = ["push_up","split_squat","pull_up"]

classify_per_30_frames(MODEL_PATH, CLASSES)