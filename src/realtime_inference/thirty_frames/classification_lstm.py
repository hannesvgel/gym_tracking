import yaml
from fct_exercise_classification_per_30_frames import classify_per_30_frames

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# MODEL_PATH = config["lstm_bidir_3cl"]["path"]
# CLASSES = config["lstm_bidir_3cl"]["class_names"]
MODEL_PATH = config["lstm_bidir_6cl_v1"]["path"]
CLASSES = config["lstm_bidir_6cl_v1"]["class_names"]

classify_per_30_frames(MODEL_PATH, CLASSES)