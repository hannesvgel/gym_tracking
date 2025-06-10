from fct_exercise_classification_per_30_frames import classify_per_30_frames

MODEL_PATH = "skeleton_lstm_multiclass6.h5"
# bench_press: 0, bulgarian_squat: 1, lat_machine: 2, pull_up: 3, push_up: 4, split_squat: 5
CLASSES      = ["bench_press","bulgarian_squat","lat_machine", "pull_up", "push_up", "split_squat"]

classify_per_30_frames(MODEL_PATH, CLASSES)