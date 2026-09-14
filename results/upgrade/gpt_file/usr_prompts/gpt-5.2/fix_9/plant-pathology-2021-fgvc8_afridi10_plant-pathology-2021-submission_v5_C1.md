# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.0779132040627885

# 6. Current score

0.7174

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.71403) has done: 'I fix the initial TensorFlow/Keras import crash by forcing the notebook to use `tf.keras` consistently and avoiding the standalone `keras` package that triggers the protobuf `MessageFactory` error in this environment. Because the referenced pre-trained `.h5` file doesn’t exist, I replace the load with a minimal, standard `tf.keras.applications.MobileNetV2` multi-label head trained quickly on the provided `train.csv`/`train_images`, preserving the same “predict then round” core inference semantics. I also fix label encoding/decoding so the submission uses the required space-delimited single-disease labels (not composite strings), and ensure the output CSV matches `sample_submission.csv` ordering and column names. The result run end-to-end and write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.71403) has done: 'The crash happens before any training because TensorFlow import triggers a protobuf incompatibility (`MessageFactory.GetPrototype`) in this runtime. The minimal fix is to force TensorFlow to use the pure-Python protobuf implementation *before* importing TensorFlow, which avoids the failing C++ protobuf path. I also keep everything else the same (model, training loop, rounding-based inference), but add a small safety step to ensure the prediction rows are aligned to `sample_submission.csv` order and the CSV is written with the exact required columns. These changes are score-neutral; they primarily unblock end-to-end execution and submission creation.'
- What this solution (achieved 0.71403) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf runtime *and* ensuring it takes effect before TensorFlow loads (plus disabling C++ protobuf), which is the root cause of the `MessageFactory.GetPrototype` error. I also make the submission formatting stricter: trim whitespace, ensure exactly the required columns, and align output order to `sample_submission.csv` to avoid any hidden-test ordering mismatch. Because your current score (0.71403) is far above the target (0.0779) and higher-is-better, I not make any model/training changes that would intentionally degrade score; the changes are intended to be score-neutral and purely for stability/correctness.'
- What this solution (achieved 0.71403) has done: 'You’re crashing before training because TensorFlow is still picking up the incompatible C++ protobuf bindings despite the environment variables; the most reliable fix in this Kaggle runtime is to force the pure-Python protobuf implementation and ensure it’s applied before TensorFlow loads, plus proactively remove any already-imported protobuf modules if the notebook is re-run. I keep your model/training/inference logic identical, only adjusting the import/bootstrap cell so TensorFlow can import cleanly. I also add a small guard to ensure the output submission rows exactly follow `sample_submission.csv` order (score-neutral but avoids hidden-test ordering issues). No changes are made that intentionally move the score toward the target (your current score is already far above target, and higher-is-better).'
- What this solution (achieved 0.71403) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf C++ binding by forcing the pure-Python protobuf implementation *before* TensorFlow loads, and by removing any already-imported protobuf modules in case of re-runs. I also make the import bootstrap more robust by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` (and not relying on attributes that may be missing) while keeping your model/training/inference logic unchanged (so score impact should be negligible). The rest of the pipeline (data loading, model definition, prediction, and submission formatting) is preserved, ensuring a valid `/kaggle/working/submission.csv` is always produced with the correct columns and row order.'
- What this solution (achieved 0.71403) has done: 'I fix the runtime crash happening at TensorFlow import by forcing the pure-Python protobuf implementation in a way that reliably takes effect (and by sanitizing any already-loaded protobuf modules), which is the root cause of the `MessageFactory.GetPrototype` error. I not change your model, training loop, inference, or submission formatting logic, since your current score (0.71403) is already far above the target and score changes are not needed. The remaining cells are kept the same so the notebook runs end-to-end and writes a valid `/kaggle/working/submission.csv` with the required columns and ordering.'
- What this solution (achieved 0.71403) has done: 'I fix the TensorFlow import crash (`MessageFactory` / protobuf mismatch) by forcing the pure-Python protobuf implementation *before* TensorFlow loads and by restarting the process once if TensorFlow was already imported in this kernel state. This is the minimal stability fix needed to run end-to-end; the model/training/inference logic is kept identical to avoid intentional score changes (your current score is already far above the target). I also keep the submission-writing path and formatting the same, ensuring `/kaggle/working/submission.csv` is produced with the required `image,labels` columns and sample order alignment.'
- What this solution (achieved 0.7174) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation *and* preventing the standalone `keras` package from being imported (it can trigger the `MessageFactory.GetPrototype` protobuf error in this environment). This is done entirely in the first bootstrap cell and doesn’t change your model, training loop, or inference logic, so it should be score-neutral (your current score is already far above the target). I also keep the submission-writing logic unchanged, ensuring `/kaggle/working/submission.csv` is created with the required `image,labels` columns and sample order alignment.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_USE_LEGACY_KERAS"] = "1"  # prefer tf.keras, avoid standalone keras

for m in list(sys.modules.keys()):
    if m.startswith(("google.protobuf", "tensorflow", "keras")):
        del sys.modules[m]

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import backend as K

print("TensorFlow:", tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val




## === cell 2
DATA_DIR = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

train_df["image"] = train_df["image"].astype(str).str.strip()
train_df["labels"] = train_df["labels"].astype(str).str.strip()
sub_df["image"] = sub_df["image"].astype(str).str.strip()

print(train_df.shape, sub_df.shape)
train_df.head()



## === cell 3
CLASSES = ["healthy", "scab", "frog_eye_leaf_spot", "rust", "powdery_mildew", "complex"]
class2idx = {c: i for i, c in enumerate(CLASSES)}
idx2class = {i: c for c, i in class2idx.items()}


def encode_labels(label_str: str) -> np.ndarray:
    y = np.zeros(len(CLASSES), dtype=np.float32)
    if isinstance(label_str, str) and label_str.strip():
        for tok in label_str.split():
            if tok in class2idx:
                y[class2idx[tok]] = 1.0
    return y


y_all = np.stack([encode_labels(s) for s in train_df["labels"].tolist()], axis=0)
print("Encoded y:", y_all.shape, "positives per class:", y_all.sum(axis=0))



## === cell 4
IMSIZE = 128
BATCH = 32


def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMSIZE, IMSIZE])
    img = tf.cast(img, tf.float32) / 255.0
    return img


def train_parse(image_name, y):
    path = tf.strings.join([TRAIN_IMG_DIR, "/", image_name])
    img = load_image(path)
    return img, y


def test_parse(image_name):
    path = tf.strings.join([TEST_IMG_DIR, "/", image_name])
    img = load_image(path)
    return img


n = len(train_df)
idx = np.arange(n)
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.1
n_val = int(n * val_frac)
val_idx = idx[:n_val]
trn_idx = idx[n_val:]

x_trn = train_df.iloc[trn_idx]["image"].values
y_trn = y_all[trn_idx]
x_val = train_df.iloc[val_idx]["image"].values
y_val = y_all[val_idx]

train_ds = tf.data.Dataset.from_tensor_slices((x_trn, y_trn)).map(
    train_parse, num_parallel_calls=tf.data.AUTOTUNE
)
train_ds = (
    train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).map(
    train_parse, num_parallel_calls=tf.data.AUTOTUNE
)
val_ds = val_ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)

print(
    "Train batches:",
    tf.data.experimental.cardinality(train_ds).numpy(),
    "Val batches:",
    tf.data.experimental.cardinality(val_ds).numpy(),
)



## === cell 5
base = tf.keras.applications.MobileNetV2(
    input_shape=(IMSIZE, IMSIZE, 3), include_top=False, weights="imagenet"
)
base.trainable = False

inp = tf.keras.Input(shape=(IMSIZE, IMSIZE, 3))
x = base(inp, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(len(CLASSES), activation="sigmoid")(x)
model = tf.keras.Model(inp, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[f1],
)

model.summary()

EPOCHS = 2
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)



## === cell 6
test_names = sub_df["image"].astype(str).str.strip().values

test_ds = tf.data.Dataset.from_tensor_slices(tf.constant(test_names))
test_ds = (
    test_ds.map(test_parse, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

y_pred = model.predict(test_ds, verbose=1)
print("Pred shape:", y_pred.shape)



## === cell 7
y_bin = np.around(y_pred).astype(int)

labels_out = []
for i in range(y_bin.shape[0]):
    chosen = [idx2class[j] for j in range(len(CLASSES)) if y_bin[i, j] == 1]
    if len(chosen) == 0:
        chosen = [idx2class[int(np.argmax(y_pred[i]))]]
    labels_out.append(" ".join(chosen))

pred_df = pd.DataFrame({"image": test_names, "labels": labels_out})
pred_df.head()



## === cell 8
pred_df["image"] = pred_df["image"].astype(str).str.strip()
pred_df["labels"] = pred_df["labels"].astype(str).str.strip()
pred_df = pred_df.set_index("image").reindex(sub_df["image"]).reset_index()

pred_df = pred_df[["image", "labels"]]

out_path = "/kaggle/working/submission.csv"
pred_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(pred_df))



## === cell 9
with open("/kaggle/working/submission.csv", "r") as f:
    for _ in range(5):
        print(f.readline().rstrip())
