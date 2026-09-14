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

import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF choose
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF:", tf.__version__)



## === cell 1
BASE_PATH = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

train.head(), submissions.head(), train.shape, submissions.shape



## === cell 2
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
classes = mlb.classes_.tolist()

labels_df = pd.DataFrame(y, columns=classes)
labels_df.head(), len(classes), classes



## === cell 3
train_df = train.copy()
for c in classes:
    train_df[c] = labels_df[c].values

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.1
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

tr_df.shape, va_df.shape



## === cell 4
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255, rotation_range=15, horizontal_flip=True, zoom_range=0.1
)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_dataframe(
    tr_df,
    directory=TRAIN_IMG_DIR,
    x_col="image",
    y_col=classes,
    target_size=IMG_SIZE,
    color_mode="rgb",
    class_mode="raw",
    shuffle=True,
    batch_size=BATCH_SIZE,
    seed=SEED,
    drop_remainder=True,
)

val_generator = val_datagen.flow_from_dataframe(
    va_df,
    directory=TRAIN_IMG_DIR,
    x_col="image",
    y_col=classes,
    target_size=IMG_SIZE,
    color_mode="rgb",
    class_mode="raw",
    shuffle=False,
    batch_size=BATCH_SIZE,
    drop_remainder=True,
)

test_generator = test_datagen.flow_from_dataframe(
    submissions,
    directory=TEST_IMG_DIR,
    x_col="image",
    y_col=None,
    target_size=IMG_SIZE,
    color_mode="rgb",
    class_mode=None,
    shuffle=False,
    batch_size=BATCH_SIZE,
    drop_remainder=False,  # keep all test rows
)

AUTOTUNE = tf.data.AUTOTUNE


def dataset_from_generator(gen, has_labels: bool):
    if has_labels:
        output_signature = (
            tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, len(classes)), dtype=tf.float32),
        )
        ds = tf.data.Dataset.from_generator(
            lambda: gen, output_signature=output_signature
        )
    else:
        output_signature = tf.TensorSpec(
            shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32
        )
        ds = tf.data.Dataset.from_generator(
            lambda: gen, output_signature=output_signature
        )
    return ds.prefetch(AUTOTUNE)


train_ds = dataset_from_generator(train_generator, has_labels=True)
val_ds = dataset_from_generator(val_generator, has_labels=True)
test_ds = dataset_from_generator(test_generator, has_labels=False)



## === cell 5
base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights="imagenet"
)
base.trainable = False  # keep fast and stable

inp = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inp, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(len(classes), activation="sigmoid")(x)
model = tf.keras.Model(inp, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 6
EPOCHS = 3

steps_per_epoch = int(np.ceil(train_generator.n / BATCH_SIZE))
validation_steps = int(np.ceil(val_generator.n / BATCH_SIZE))

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 7
test_steps = int(np.ceil(test_generator.n / BATCH_SIZE))

preds = model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
)
preds = preds[: len(submissions)]  # safety if generator pads last batch
preds.shape, preds[:2]



## === cell 8
thresh = 0.5

mask = preds >= thresh
any_pos = mask.any(axis=1)

top_idx = np.argmax(preds, axis=1)
rows, cols = np.where(mask)

pred_labels = [""] * preds.shape[0]
tmp = pd.DataFrame({"row": rows, "col": cols})
grp = tmp.groupby("row")["col"].apply(list)
for r, cs in grp.items():
    pred_labels[int(r)] = " ".join(classes[c] for c in cs)

no_pos_rows = np.where(~any_pos)[0]
for r in no_pos_rows:
    pred_labels[int(r)] = classes[int(top_idx[int(r)])]

submissions_out = submissions.copy()
submissions_out["labels"] = pred_labels

submissions_out.to_csv("submission.csv", index=False)
submissions_out.head(), submissions_out.shape



## === cell 9
assert os.path.exists("submission.csv")
assert list(submissions_out.columns) == ["image", "labels"]
assert len(submissions_out) == len(submissions)
print("Wrote submission.csv with", len(submissions_out), "rows")
print(submissions_out.sample(5, random_state=SEED))
