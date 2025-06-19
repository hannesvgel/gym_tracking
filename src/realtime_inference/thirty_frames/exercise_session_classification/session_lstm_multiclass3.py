from fct_exercise_session_per_30_frames import classify_session_per_30_frames

MODEL_PATH = "skeleton_lstm_multiclass3.h5"
CLASSES    = ["pull_up", "push_up", "squat"]  # pull_up: 0, push_up: 1, squat: 2

classify_session_per_30_frames(MODEL_PATH, CLASSES)