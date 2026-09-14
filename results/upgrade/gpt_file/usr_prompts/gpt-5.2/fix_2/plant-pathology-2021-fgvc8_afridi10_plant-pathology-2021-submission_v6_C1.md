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

0.3943859649122804

# 6. Current score

0.31188

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.31188) has done: 'I fix the environment-breaking TensorFlow import error by switching to the Kaggle-stable `tf.keras` stack and forcing the pure-Python protobuf implementation before importing TensorFlow. Since the referenced pretrained `.h5` file doesn’t exist in your provided `/kaggle/input/...` paths, I keep the same “MobileNet-style classifier” core idea but build a MobileNetV2-based multi-label model and train it quickly on `train.csv` + `train_images`, then run inference on `test_images`. I also fix label post-processing to output a space-delimited list, ensure at least one label is predicted (fallback to best class), and match the exact `sample_submission.csv` image order so the submission is valid. Finally, the script always write `/kaggle/working/submission.csv` with the required `image,labels` columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

import tensorflow as tf

print("TensorFlow:", tf.__version__)

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV = f"{BASE}/train.csv"
SAMPLE_SUB = f"{BASE}/sample_submission.csv"
TRAIN_DIR = f"{BASE}/train_images"
TEST_DIR = f"{BASE}/test_images"

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.exists(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing {TEST_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

train_df.head(), sub_df.head(), train_df.shape, sub_df.shape



## === cell 2
label_names = [
    "complex",
    "frog_eye_leaf_spot",
    "frog_eye_leaf_spot complex",
    "healthy",
    "powdery_mildew",
    "powdery_mildew complex",
    "rust",
    "rust complex",
    "rust frog_eye_leaf_spot",
    "scab",
    "scab frog_eye_leaf_spot",
    "scab frog_eye_leaf_spot complex",
]
label_to_idx = {l: i for i, l in enumerate(label_names)}
num_classes = len(label_names)


def encode_labels(label_str: str) -> np.ndarray:
    y = np.zeros(num_classes, dtype=np.float32)
    s = str(label_str)
    for i, name in enumerate(label_names):
        if name in s:
            y[i] = 1.0
    return y


y_all = np.stack([encode_labels(s) for s in train_df["labels"].values])
assert y_all.shape == (len(train_df), num_classes)

empty = (y_all.sum(axis=1) == 0).sum()
print("Rows with 0 parsed labels:", empty)



## === cell 3
IMSIZE = 128
BATCH = 32


def read_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, (IMSIZE, IMSIZE), method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


idx = np.arange(len(train_df))
rng = np.random.default_rng(42)
rng.shuffle(idx)

val_frac = 0.1
val_n = int(len(idx) * val_frac)
val_idx = idx[:val_n]
trn_idx = idx[val_n:]

trn_paths = (TRAIN_DIR + "/" + train_df.iloc[trn_idx]["image"].values).tolist()
val_paths = (TRAIN_DIR + "/" + train_df.iloc[val_idx]["image"].values).tolist()
trn_y = y_all[trn_idx]
val_y = y_all[val_idx]


def make_ds(paths, labels=None, training=False):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(read_image, num_parallel_calls=tf.data.AUTOTUNE)
    if labels is not None:
        ds_y = tf.data.Dataset.from_tensor_slices(labels)
        ds = tf.data.Dataset.zip((ds, ds_y))
    if training:
        ds = ds.shuffle(2048, seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds


trn_ds = make_ds(trn_paths, trn_y, training=True)
val_ds = make_ds(val_paths, val_y, training=False)



## === cell 4
import keras.backend as K


def f1(y_true, y_pred):  # taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


base = tf.keras.applications.MobileNetV2(
    input_shape=(IMSIZE, IMSIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False  # stabilize + speed; avoids changing "approach/loops" complexity

inputs = tf.keras.Input(shape=(IMSIZE, IMSIZE, 3))
x = inputs
x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[f1],
)
model.summary()



## === cell 5
EPOCHS = 3
history = model.fit(
    trn_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3670339352.py in <cell line: 0>()
      1 # Train (no early stopping; fixed epochs for determinism)
      2 EPOCHS = 3
----> 3 history = model.fit(
      4     trn_ds,
      5     validation_data=val_ds,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/2802619377.py in f1(y_true, y_pred)
      4 
      5 def f1(y_true, y_pred):  # taken from old keras source code
----> 6     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
      7     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
      8     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

AttributeError: module 'keras.backend' has no attribute 'sum'

## === cell 6
val_pred = model.predict(val_ds, verbose=0)
val_true = val_y


def mean_f1_at_threshold(y_true, y_prob, thr: float) -> float:
    y_hat = (y_prob >= thr).astype(np.int32)
    empty = y_hat.sum(axis=1) == 0
    if empty.any():
        best = y_prob.argmax(axis=1)
        y_hat[empty, best[empty]] = 1

    tp = (y_hat & (y_true > 0.5)).sum(axis=1)
    fp = (y_hat & (y_true <= 0.5)).sum(axis=1)
    fn = ((1 - y_hat) & (y_true > 0.5)).sum(axis=1)
    f1s = (2 * tp) / (2 * tp + fp + fn + 1e-9)
    return float(np.mean(f1s))


thresholds = np.linspace(0.1, 0.9, 17)
scores = [mean_f1_at_threshold(val_true, val_pred, t) for t in thresholds]
best_i = int(np.argmax(scores))
best_thr = float(thresholds[best_i])
print("Best threshold:", best_thr, "val mean F1:", scores[best_i])



## === cell 7
test_images = sub_df["image"].astype(str).values.tolist()
test_paths = [f"{TEST_DIR}/{n}" for n in test_images]

test_ds = make_ds(test_paths, labels=None, training=False)
test_prob = model.predict(test_ds, verbose=1)



## === cell 8
y_hat = (test_prob >= best_thr).astype(np.int32)

empty = y_hat.sum(axis=1) == 0
if empty.any():
    best = test_prob.argmax(axis=1)
    y_hat[empty, best[empty]] = 1

pred_labels = []
for i in range(y_hat.shape[0]):
    labs = [label_names[j] for j in range(num_classes) if y_hat[i, j] == 1]
    pred_labels.append(" ".join(labs))

submission = pd.DataFrame({"image": test_images, "labels": pred_labels})
submission.head()



## === cell 9
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", submission.shape)



## === cell 10
with open("/kaggle/working/submission.csv", "r") as f:
    for _ in range(10):
        print(f.readline().rstrip("\n"))
