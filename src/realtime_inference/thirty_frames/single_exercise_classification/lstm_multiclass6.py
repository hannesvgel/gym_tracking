from fct_exercise_classification_per_30_frames import classify_per_30_frames

MODEL_PATH = "skeleton_lstm_multiclass6.h5"
# bench_press: 0, lat_machine: 1, pull_up: 2, push_up: 3, squat: 4, split_squat: 5
CLASSES      = ["bench_press" ,"lat_machine", "pull_up", "push_up", "squat", "split_squat"]

classify_per_30_frames(MODEL_PATH, CLASSES)