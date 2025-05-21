import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# ——— CONFIG ———
DATA_DIR     = Path("data/processed/kaggle_img_DS")
CLASSES      = ["push_up","squat","pull_up"]
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

# 3. tf.data pipeline
def parse_fn(path, label):
    content = tf.io.read_file(path)
    lines   = tf.strings.split(content, "\n")
    nonempty= tf.boolean_mask(lines, tf.strings.length(lines)>0)
    first   = nonempty[0]
    parts   = tf.strings.split(first, ",")
    x       = tf.strings.to_number(parts, tf.float32)        # (132,)
    x       = tf.reshape(x, [KEYPOINT_DIM, 1])               # (132,1)
    return x, label

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

# 4. 1D‐CNN
model = tf.keras.Sequential([
    tf.keras.layers.Input((KEYPOINT_DIM,1)),
    tf.keras.layers.Conv1D(32, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPool1D(2),
    tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu"),
    tf.keras.layers.MaxPool1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"),
])
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
model.summary()

# 5. Train with early stopping and best‐model checkpointing
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint("skeleton_cnn_multiclass3.h5",
                                       save_best_only=True),
]
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=callbacks
)

# plot training accuracy and loss in subplots
history = model.history.history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='train accuracy')
plt.plot(history['val_accuracy'], label='val accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='train loss')
plt.plot(history['val_loss'], label='val loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.show()

# 6. Evaluate on the test set
print("\n=== Test Set Evaluation ===")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test accuracy: {test_acc:.2%}")

# 7. Detailed metrics (confusion matrix, classification report)
# Gather all test predictions & true labels
y_true = []
y_pred = []
for X_batch, y_batch in test_ds:
    preds = model.predict(X_batch)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(y_batch.numpy())

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))
