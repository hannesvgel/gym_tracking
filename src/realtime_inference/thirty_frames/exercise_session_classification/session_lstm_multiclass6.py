from fct_exercise_session_per_30_frames import classify_session_per_30_frames

MODEL_PATH = "skeleton_lstm_bidir_multiclass6.h5"
# bench_press: 0, squat: 1, lat_machine: 2, pull_up: 3, push_up: 4, split_squat: 5
CLASSES      = ["bench_press", "squat","lat_machine", "pull_up", "push_up", "split_squat"]

classify_session_per_30_frames(MODEL_PATH, CLASSES)