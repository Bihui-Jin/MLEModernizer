# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.1000302206104563

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.63303) has done: 'I first fix the environment-breaking import error by switching from standalone `keras` (Keras 3) to `tf_keras`, which avoids the protobuf `MessageFactory.GetPrototype` crash in this Kaggle image setup. Next, I remove the dependency on missing external model files (`../input/model-4/...` and `../input/new-model/...`) and instead build a small CNN classifier that can train quickly on the provided `train_images/` data. Finally, I ensure test inference matches the submission format by reading `sample_submission.csv` to get the exact test `image_id` order and then writing a valid `submission.csv` with `image_id,label`. These changes are required to run end-to-end and yield a non-trivial accuracy score rather than failing before producing a submission.'
- What this solution (achieved 0.63191) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing TensorFlow to use the pure-Python protobuf implementation *before* importing `tensorflow`/`tf_keras`, which is a common workaround in Kaggle images when compiled protobuf and TF versions mismatch. I also keep your model/training/inference logic the same, but add a safe fallback so the notebook still runs even if the protobuf workaround doesn’t fully resolve the issue (it retry imports cleanly). Finally, I ensure the script always writes a properly formatted `submission.csv` with the exact `image_id` order from `sample_submission.csv`.'
- What this solution (achieved 0.63154) has done: 'I fix the immediate runtime crash by applying a more robust protobuf/TensorFlow import workaround that forces the pure-Python protobuf implementation and additionally disables the C++ protobuf backend (a common cause of `MessageFactory.GetPrototype` errors in Kaggle images). I keep your model, training loop, and inference logic unchanged, only adjusting the import section to reliably load `tensorflow`/`tf_keras` without triggering the protobuf AttributeError. I also add a small defensive check that ensures the submission length matches `sample_submission.csv` order before writing, so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.63079) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by ensuring the pure-Python protobuf runtime is selected *before* any protobuf-related imports, and by force-reloading `google.protobuf` and clearing conflicting modules before importing TensorFlow. This is the minimal change needed to make the notebook run end-to-end and reliably produce `submission.csv`. I keep the model/training/inference logic unchanged (same CNN, same epochs, same data pipeline) so the score behavior stays essentially the same. Finally, I add a tiny safety check to confirm `submission.csv` matches the sample submission ordering/length.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import random
import importlib

import numpy as np
import pandas as pd
from PIL import Image

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]
importlib.invalidate_caches()

try:
    import tf_keras as keras

    tf = keras.backend.tf
except Exception:
    for m in list(sys.modules.keys()):
        if m.startswith(("tensorflow", "tf_keras", "keras", "google.protobuf")):
            del sys.modules[m]
    importlib.invalidate_caches()
    import tf_keras as keras

    tf = keras.backend.tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
if not os.path.exists(DATA_ROOT):
    alt_root = "/kaggle/data/cassava-leaf-disease-classification"
    if os.path.exists(alt_root):
        DATA_ROOT = alt_root

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing {TEST_DIR}"

print("Using DATA_ROOT:", DATA_ROOT)
print("tf_keras version:", getattr(keras, "__version__", "unknown"))
print("TF version (via tf_keras):", getattr(tf, "__version__", "unknown"))



## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
num_classes = int(train_df["label"].nunique())
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

sample_sub = pd.read_csv(SAMPLE_SUB)
test_image_ids = sample_sub["image_id"].tolist()

print("Train rows:", len(train_df), "Test rows:", len(test_image_ids))



## === cell 2
IMG_SIZE = (128, 128)  # smaller to stay within 600s budget
BATCH_SIZE = 32
EPOCHS = 2

TRAIN_FRACTION = (
    0.02  # 2% of data (minimal, direct way to reduce score without changing model/loss)
)


def load_image(path):
    img = Image.open(path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


idx = np.arange(len(train_df))
np.random.shuffle(idx)

use_n = max(1, int(len(idx) * TRAIN_FRACTION))
idx = idx[:use_n]

split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

train_df_tr = train_df.iloc[tr_idx].reset_index(drop=True)
train_df_va = train_df.iloc[va_idx].reset_index(drop=True)

print("Using TRAIN_FRACTION:", TRAIN_FRACTION)
print("Subsampled train rows:", len(train_df_tr), "val rows:", len(train_df_va))


def make_generator(df, shuffle=True):
    paths = [os.path.join(TRAIN_DIR, x) for x in df["image_id"].tolist()]
    labels = df["label"].astype(int).to_numpy()
    n = len(df)

    def gen():
        order = np.arange(n)
        while True:
            if shuffle:
                np.random.shuffle(order)
            for i in range(0, n, BATCH_SIZE):
                batch_idx = order[i : i + BATCH_SIZE]
                x = np.stack([load_image(paths[j]) for j in batch_idx], axis=0)
                y = labels[batch_idx]
                yield x, y

    steps = int(np.ceil(n / BATCH_SIZE))
    return gen(), steps


train_gen, train_steps = make_generator(train_df_tr, shuffle=True)
val_gen, val_steps = make_generator(train_df_va, shuffle=False)



## === cell 3
inputs = keras.layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = keras.layers.Conv2D(1, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(1, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(1, 3, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.0)(x)
outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 4
history = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=EPOCHS,
    verbose=2,
)



## === cell 5
preds = []
for start in range(0, len(test_image_ids), BATCH_SIZE):
    batch_ids = test_image_ids[start : start + BATCH_SIZE]
    batch_paths = [os.path.join(TEST_DIR, x) for x in batch_ids]
    x = np.stack([load_image(p) for p in batch_paths], axis=0)
    p = model.predict(x, verbose=0)
    preds.extend(np.argmax(p, axis=1).tolist())

assert len(preds) == len(
    test_image_ids
), f"Pred length {len(preds)} != test ids {len(test_image_ids)}"

sub = pd.DataFrame({"image_id": test_image_ids, "label": preds})

sub["label"] = sub["label"].astype(int).clip(0, num_classes - 1)

print(sub.head())
print("Submission rows:", len(sub), "Unique labels:", sub["label"].nunique())

sub = sub[["image_id", "label"]]
assert len(sub) == len(
    sample_sub
), "Submission row count must match sample_submission.csv"
assert (
    sub["image_id"].values == sample_sub["image_id"].values
).all(), "image_id order must match sample_submission.csv"

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
