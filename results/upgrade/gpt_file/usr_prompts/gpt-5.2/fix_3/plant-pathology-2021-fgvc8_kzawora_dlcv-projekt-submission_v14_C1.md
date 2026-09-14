# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

np.random.seed(42)
tf.random.set_seed(42)

BASE_CANDIDATES = [
    Path("/kaggle/input/plant-pathology-2021-fgvc8"),
    Path("/kaggle/data/plant-pathology-2021-fgvc8"),
    Path("../input/plant-pathology-2021-fgvc8"),
]
BASE_DIR = next((p for p in BASE_CANDIDATES if p.exists()), None)
if BASE_DIR is None:
    raise FileNotFoundError(f"Could not find dataset dir in any of: {BASE_CANDIDATES}")

TRAIN_CSV = BASE_DIR / "train.csv"
SAMPLE_SUB = BASE_DIR / "sample_submission.csv"
TRAIN_IMAGES_DIR = BASE_DIR / "train_images"
TEST_IMAGES_DIR = BASE_DIR / "test_images"

print("BASE_DIR:", BASE_DIR)
print("TRAIN_CSV exists:", TRAIN_CSV.exists())
print("SAMPLE_SUB exists:", SAMPLE_SUB.exists())
print("TRAIN_IMAGES_DIR exists:", TRAIN_IMAGES_DIR.exists())
print("TEST_IMAGES_DIR exists:", TEST_IMAGES_DIR.exists())



## === cell 1
df_train = pd.read_csv(TRAIN_CSV)
print("Train rows:", len(df_train))
print(df_train.head())

labels = ["complex", "frog_eye_leaf_spot", "powdery_mildew", "rust", "scab"]
label_to_idx = {l: i for i, l in enumerate(labels)}


def labels_to_vec(s: str):
    s = str(s).strip()
    vec = np.zeros(len(labels), dtype=np.float32)
    if not s:
        return vec
    parts = s.split()
    for p in parts:
        if p in label_to_idx:
            vec[label_to_idx[p]] = 1.0
    return vec


y = np.stack([labels_to_vec(s) for s in df_train["labels"].values], axis=0)
df_train = df_train.copy()
df_train["filepath"] = df_train["image"].apply(lambda x: str(TRAIN_IMAGES_DIR / x))

missing = (~df_train["filepath"].apply(lambda p: Path(p).exists())).sum()
if missing:
    raise FileNotFoundError(
        f"{missing} training images referenced in train.csv do not exist under {TRAIN_IMAGES_DIR}"
    )

for i, l in enumerate(labels):
    df_train[l] = y[:, i].astype(np.float32)

df_train = df_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
val_frac = 0.1
n_val = int(len(df_train) * val_frac)
df_val = df_train.iloc[:n_val].reset_index(drop=True)
df_trn = df_train.iloc[n_val:].reset_index(drop=True)

print("Train split:", len(df_trn), "Val split:", len(df_val))



## === cell 2
IMG_SIZE = (380, 380)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

y_cols = labels

train_generator = train_datagen.flow_from_dataframe(
    df_trn,
    x_col="filepath",
    y_col=y_cols,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=True,
    seed=42,
)

val_generator = val_datagen.flow_from_dataframe(
    df_val,
    x_col="filepath",
    y_col=y_cols,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False,
)



## === cell 3
inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(labels), activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

EPOCHS = 3
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 4
tmp_root = Path("/kaggle/tmp/test_dataset")
target_dir = tmp_root / "test"
target_dir.mkdir(parents=True, exist_ok=True)

test_images = sorted(TEST_IMAGES_DIR.glob("*.jpg"))
if len(list(target_dir.glob("*.jpg"))) != len(test_images):
    for f in target_dir.glob("*.jpg"):
        try:
            f.unlink()
        except OSError:
            pass
    import shutil

    for src in test_images:
        shutil.copy2(src, target_dir / src.name)

print("Test images copied:", len(list(target_dir.glob("*.jpg"))))

test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_directory(
    str(tmp_root),
    class_mode=None,
    target_size=IMG_SIZE,
    shuffle=False,
    batch_size=128,
)



## === cell 5
x = model.predict(test_generator, verbose=1)
print("Pred shape:", x.shape)

threshold = 0.7
z = (x > threshold).astype(np.int32)

predictions = [[labels[i] for i, j in enumerate(row) if j != 0] for row in z]
predictions_str = [" ".join(p) if len(p) else "healthy" for p in predictions]

filenames = [Path(p).name for p in test_generator.filenames]
df_pred = pd.DataFrame({"image": filenames, "labels": predictions_str})

sample = pd.read_csv(SAMPLE_SUB)
df = sample[["image"]].merge(df_pred, on="image", how="left")
df["labels"] = df["labels"].fillna("healthy")

print(df.head())
print("Rows:", len(df), "Unique images:", df["image"].nunique())



## === cell 6
out_path = Path("submission.csv")
df.to_csv(out_path, index=False)
print("Wrote:", out_path.resolve())
print("Submission columns:", list(df.columns))
print("Any missing labels:", df["labels"].isna().sum())
print(
    "Non-string labels:", (df["labels"].apply(lambda v: not isinstance(v, str))).sum()
)
