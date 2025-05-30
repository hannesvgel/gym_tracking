from fct_exercise_session_per_30_frames import classify_session_per_30_frames

MODEL_PATH = "skeleton_lstm_multiclass3.h5"
CLASSES = ["push_up","split_squat","pull_up"]

classify_session_per_30_frames(MODEL_PATH, CLASSES)