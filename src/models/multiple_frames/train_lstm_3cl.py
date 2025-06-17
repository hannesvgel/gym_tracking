import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ——— CONFIG ———
DATA_DIR     = Path("data/processed/combined_DS/30_frame_segments")
CLASSES      = ["pull_up", "push_up", "split_squat"]  # pull_up: 0, push_up: 1, split_squat: 2
NUM_CLASSES  = len(CLASSES)
KEYPOINT_DIM = 132

# 1. Gather all files & labels
all_files, all_labels = [], []
for idx, cls in enumerate(CLASSES):
    for p in (DATA_DIR/cls).glob("*.csv"):
        all_files.append(str(p))
        all_labels.append(idx)

# 2. Split into train / val+test, then val / test
files_train, files_tmp, labels_train, labels_tmp = train_test_split(
    all_files, all_labels,
    test_size=0.3,             # 70% train, 30% tmp
    stratify=all_labels,
    random_state=42
)
files_val, files_test, labels_val, labels_test = train_test_split(
    files_tmp, labels_tmp,
    test_size=0.5,             # split tmp into 50/50 → 15% val, 15% test
    stratify=labels_tmp,
    random_state=42
)

# 3. tf.data pipeline - parse into (30, 132) tesnors (30 frames, 132 keypoints)
def parse_fn(path, label):
    content = tf.io.read_file(path)
    lines = tf.strings.strip(tf.strings.split(content, "\n"))
    lines = lines[1:]
    lines = tf.boolean_mask(lines, tf.strings.length(lines) > 0)

    def process_line(line):
        parts = tf.strings.split(line, ",")      
        parts = parts[:-1]                      
        tf.debugging.assert_equal(
            tf.shape(parts)[0], KEYPOINT_DIM,
            message="Line does not have 132 values after dropping last column"
        )
        return tf.strings.to_number(parts, tf.float32)

    parts = tf.map_fn(
        process_line,
        lines,
        fn_output_signature=tf.TensorSpec([KEYPOINT_DIM], tf.float32)
    )
    return parts, label


def make_ds(files, labels, batch_size=32, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((files, labels))
    if shuffle:
        ds = ds.shuffle(len(files), seed=42)
    return (ds
            .map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE))

train_ds = make_ds(files_train, labels_train, shuffle=True)
val_ds   = make_ds(files_val,   labels_val,   shuffle=False)
test_ds  = make_ds(files_test,  labels_test,  shuffle=False)

# 4. LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(30, 132)),
    tf.keras.layers.LSTM(128, return_sequences=True),  # keep full sequence
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.LSTM(64, return_sequences=False),  # reduce to summary
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
])


model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"]
)
model.summary()

# 5. Train with early stopping and best‐model checkpointing
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "skeleton_lstm_multiclass3.h5",
        save_best_only=True,
        monitor="val_loss"
    )
]
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=callbacks
)

# plot training accuracy and loss in subplots
def plot_history(history):
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    axs[0].plot(history.history['accuracy'], label='Train Accuracy')
    axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[0].set_title('Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].legend()

    axs[1].plot(history.history['loss'], label='Train Loss')
    axs[1].plot(history.history['val_loss'], label='Validation Loss')
    axs[1].set_title('Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].legend()

    plt.tight_layout()
    plt.show()

plot_history(history)

# 6. Evaluate on the test set
print("\n=== Test Set Evaluation ===")
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# 7. Predictions and classification report
y_pred = []
y_true = []
for X_batch, y_batch in test_ds:
    preds = model.predict(X_batch)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(y_batch.numpy())

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))
