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
import random
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

tf.random.set_seed(SEED)
print("TF:", tf.__version__)



## === cell 1
BASE_DIR = "../input/plant-pathology-2021-fgvc8"
if not os.path.exists(os.path.join(BASE_DIR, "train.csv")):
    BASE_DIR = "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8"

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
print(train.head())



## === cell 2
h_target = 512
w_target = 512
batch_size = 32

label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer().fit(label_split)
class_names = list(mlb.classes_)
n_classes = len(class_names)

print("n_classes:", n_classes)
print("classes:", class_names)



## === cell 3

train_df = train.copy()
train_df["labels_list"] = train_df["labels"].apply(lambda x: x.split())
Y = mlb.transform(train_df["labels_list"].values).astype("float32")

rng = np.random.RandomState(SEED)
idx = np.arange(len(train_df))
rng.shuffle(idx)
val_size = int(0.1 * len(idx))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

train_images = train_df.loc[trn_idx, "image"].values
valid_images = train_df.loc[val_idx, "image"].values
Y_train = Y[trn_idx]
Y_valid = Y[val_idx]


class MultiLabelImageSequence(keras.utils.Sequence):
    def __init__(
        self, image_names, y, image_dir, batch_size, target_size, shuffle, seed
    ):
        self.image_names = np.asarray(image_names)
        self.y = None if y is None else np.asarray(y, dtype=np.float32)
        self.image_dir = image_dir
        self.batch_size = int(batch_size)
        self.target_size = tuple(target_size)
        self.shuffle = bool(shuffle)
        self.rng = np.random.RandomState(seed)
        self.indices = np.arange(len(self.image_names))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.image_names) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __getitem__(self, batch_idx):
        sl = slice(batch_idx * self.batch_size, (batch_idx + 1) * self.batch_size)
        batch_ids = self.indices[sl]
        batch_names = self.image_names[batch_ids]

        X = np.empty(
            (len(batch_names), self.target_size[0], self.target_size[1], 3),
            dtype=np.float32,
        )
        for i, name in enumerate(batch_names):
            path = os.path.join(self.image_dir, name)
            img = keras.preprocessing.image.load_img(
                path, target_size=self.target_size, color_mode="rgb"
            )
            arr = keras.preprocessing.image.img_to_array(img).astype(np.float32) / 255.0
            X[i] = arr

        if self.y is None:
            return X
        yb = self.y[batch_ids].astype(np.float32)
        return X, yb


train_generator = MultiLabelImageSequence(
    train_images,
    Y_train,
    TRAIN_IMG_DIR,
    batch_size,
    (h_target, w_target),
    shuffle=True,
    seed=SEED,
)
valid_generator = MultiLabelImageSequence(
    valid_images,
    Y_valid,
    TRAIN_IMG_DIR,
    batch_size,
    (h_target, w_target),
    shuffle=False,
    seed=SEED,
)

test_generator = MultiLabelImageSequence(
    submissions["image"].values,
    None,
    TEST_IMG_DIR,
    batch_size,
    (h_target, w_target),
    shuffle=False,
    seed=SEED,
)

print(
    "Train batches:",
    len(train_generator),
    "Valid batches:",
    len(valid_generator),
    "Test batches:",
    len(test_generator),
)



## === cell 4
base = tf.keras.applications.MobileNetV2(
    input_shape=(h_target, w_target, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)
inputs = keras.Input(shape=(h_target, w_target, 3))
x = base(inputs, training=False)
outputs = keras.layers.Dense(n_classes, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
)

EPOCHS = 3
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=len(train_generator),
    validation_data=valid_generator,
    verbose=1,
)



## === cell 5
preds = model.predict(test_generator, verbose=1)
print("preds shape:", preds.shape)
print(preds[:2])



## === cell 6
thresh = 0.5

pred_labels = []
healthy_idx = class_names.index("healthy") if "healthy" in class_names else None

for i in range(preds.shape[0]):
    p = preds[i]
    top_idx = int(np.argmax(p))

    chosen = np.where(p >= thresh)[0]
    lab_list = [class_names[j] for j in chosen]

    if len(lab_list) == 0:
        lab_list = [class_names[top_idx]]

    if "healthy" in lab_list and len(lab_list) > 1:
        lab_list = [l for l in lab_list if l != "healthy"]
        if len(lab_list) == 0:
            lab_list = ["healthy"]

    lab = " ".join(lab_list)
    pred_labels.append(lab)

submissions = submissions.copy()
submissions["labels"] = pred_labels
print(submissions.head())



## === cell 7
out_path = "submission.csv"
submissions[["image", "labels"]].to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submissions))
print(submissions.head(10))
