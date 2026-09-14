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

import random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

print("TF:", tf.__version__)



## === cell 1
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
train.head()



## === cell 2
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
classes = list(mlb.classes_)

labels = pd.DataFrame(y, columns=classes)
print("Classes:", classes)
labels.head()



## === cell 3
h_target = 256
w_target = 256
batch_size = 32

thresh = {
    "complex": 0.14,
    "frog_eye_leaf_spot": 0.3,
    "healthy": 0.33,
    "powdery_mildew": 0.08,
    "rust": 0.13,
    "scab": 0.443,
}
thresh_labels = list(thresh.keys())
print("Threshold label order:", thresh_labels)

missing_in_mlb = [c for c in thresh_labels if c not in classes]
extra_in_mlb = [c for c in classes if c not in thresh_labels]
print("Missing in mlb:", missing_in_mlb)
print("Extra in mlb:", extra_in_mlb)

y_thresh_order = labels[thresh_labels].values.astype(np.float32)
print("y shape:", y_thresh_order.shape)



## === cell 4
idx = np.arange(len(train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.1
val_size = int(len(train) * val_frac)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

train_df = train.iloc[tr_idx].reset_index(drop=True)
val_df = train.iloc[val_idx].reset_index(drop=True)

y_train = y_thresh_order[tr_idx]
y_val = y_thresh_order[val_idx]

print("Train:", train_df.shape, y_train.shape)
print("Val:", val_df.shape, y_val.shape)



## === cell 5
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True,
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_IMG_DIR,
    x_col="image",
    y_col=None,
    target_size=(h_target, w_target),
    color_mode="rgb",
    class_mode=None,
    batch_size=batch_size,
    shuffle=True,
    seed=SEED,
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=TRAIN_IMG_DIR,
    x_col="image",
    y_col=None,
    target_size=(h_target, w_target),
    color_mode="rgb",
    class_mode=None,
    batch_size=batch_size,
    shuffle=False,
)

test_generator = test_datagen.flow_from_dataframe(
    dataframe=submissions,
    directory=TEST_IMG_DIR,
    x_col="image",
    y_col=None,
    target_size=(h_target, w_target),
    color_mode="rgb",
    class_mode=None,
    batch_size=batch_size,
    shuffle=False,
)




## === cell 6
class XySequence(keras.utils.Sequence):
    """
    Fix: ensure y aligns with the same samples that the ImageDataGenerator yields.
    The previous version paired `next(x_gen)` with sequential y slices, which becomes wrong when
    the underlying generator shuffles (training generator does shuffle=True).
    Here we use the generator's current batch indices to pick matching y rows.
    """

    def __init__(self, x_gen, y_array):
        self.x_gen = x_gen
        self.y = y_array
        self.batch_size = getattr(x_gen, "batch_size", None) or 32
        self.n = len(y_array)
        self.steps = int(np.ceil(self.n / self.batch_size))

    def __len__(self):
        return self.steps

    def __getitem__(self, idx):
        x_batch = next(self.x_gen)

        if hasattr(self.x_gen, "index_array") and hasattr(self.x_gen, "batch_index"):
            bi = int(getattr(self.x_gen, "batch_index", 0)) - 1
            if bi < 0:
                bi = self.steps - 1
            start = bi * self.batch_size
            end = min(start + len(x_batch), self.n)
            batch_ids = self.x_gen.index_array[start:end]
            y_batch = self.y[batch_ids]
        else:
            start = idx * self.batch_size
            end = min(start + len(x_batch), self.n)
            y_batch = self.y[start:end]

        if len(y_batch) != len(x_batch):
            y_batch = y_batch[: len(x_batch)]
        return x_batch, y_batch

    def on_epoch_end(self):
        pass


steps_per_epoch = int(np.ceil(len(train_df) / batch_size))
val_steps = int(np.ceil(len(val_df) / batch_size))

train_seq = XySequence(train_generator, y_train)
val_seq = XySequence(val_generator, y_val)

print("steps_per_epoch:", steps_per_epoch, "val_steps:", val_steps)



## === cell 7
inputs = keras.Input(shape=(h_target, w_target, 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(len(thresh_labels), activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 8
EPOCHS = 3

history = model.fit(
    train_seq,
    validation_data=val_seq,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)



## === cell 9
preds = model.predict(test_generator, verbose=1)
print("preds shape:", preds.shape)
print(preds[:2])



## === cell 10
submissions = submissions.copy()
healthy_idx = thresh_labels.index("healthy")
thresh_vec = np.array([thresh[lbl] for lbl in thresh_labels], dtype=np.float32)

argmax_idx = np.argmax(preds, axis=1)
row_max = preds[np.arange(len(preds)), argmax_idx]
healthy_score = preds[:, healthy_idx]

labels_out = np.empty(len(preds), dtype=object)

mask_healthy_max = healthy_score == row_max
labels_out[mask_healthy_max] = "healthy"

mask_else = ~mask_healthy_max
if np.any(mask_else):
    P = preds[mask_else]
    above = P > thresh_vec  # shape: (n_else, n_labels)

    else_indices = np.where(mask_else)[0]
    for k, i in enumerate(else_indices):
        row_labels = [thresh_labels[j] for j in np.flatnonzero(above[k])]
        lbl = " ".join(row_labels)
        if (lbl == "") or ("healthy" in lbl):
            lbl = thresh_labels[int(argmax_idx[i])]
        labels_out[i] = lbl

submissions["labels"] = labels_out
submissions.head()

out_path = "submission.csv"
submissions[["image", "labels"]].to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submissions))
print(submissions[["image", "labels"]].head())
