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

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

try:
    import tensorflow as tf
except Exception as e:
    raise RuntimeError(
        "TensorFlow import failed in this environment. "
        "This notebook requires TensorFlow to run."
    ) from e

tf.random.set_seed(SEED)

try:
    import keras as keras  # Keras 3
except Exception:
    keras = tf.keras

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)
print("Using keras module:", getattr(keras, "__name__", str(type(keras))))



## === cell 1
train = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
train.head()



## === cell 2
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
submissions.head()



## === cell 3
h_target = 384
w_target = 384
BATCH_SIZE = 16

TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"

assert os.path.isdir(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing test dir: {TEST_DIR}"



## === cell 4
from sklearn.preprocessing import MultiLabelBinarizer

label_split = train["labels"].str.split()
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
class_names = list(mlb.classes_)
num_classes = len(class_names)

print("Num classes:", num_classes)
print("Classes:", class_names)

train_df = pd.concat([train, pd.DataFrame(y, columns=class_names)], axis=1)
train_df.head()



## === cell 5
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.1,
    horizontal_flip=True,
    vertical_flip=True,
    rotation_range=20,
    zoom_range=0.1,
)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255, validation_split=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=TRAIN_DIR,
    x_col="image",
    y_col=class_names,
    target_size=(h_target, w_target),
    color_mode="rgb",
    class_mode="raw",
    subset="training",
    shuffle=True,
    seed=SEED,
    batch_size=BATCH_SIZE,
)

valid_generator = valid_datagen.flow_from_dataframe(
    train_df,
    directory=TRAIN_DIR,
    x_col="image",
    y_col=class_names,
    target_size=(h_target, w_target),
    color_mode="rgb",
    class_mode="raw",
    subset="validation",
    shuffle=False,
    batch_size=BATCH_SIZE,
)

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_dataframe(
    submissions,
    directory=TEST_DIR,
    x_col="image",
    y_col=None,
    target_size=(h_target, w_target),
    color_mode="rgb",
    class_mode=None,
    shuffle=False,
    batch_size=BATCH_SIZE,
)

WORKERS = min(8, (os.cpu_count() or 2))
USE_MULTIPROCESSING = True
MAX_QUEUE_SIZE = 32



## === cell 6
base = tf.keras.applications.MobileNetV2(
    input_shape=(h_target, w_target, 3), include_top=False, weights="imagenet"
)
base.trainable = False  # keep training light and stable

inputs = keras.Input(shape=(h_target, w_target, 3))
x = base(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 7
EPOCHS = 2

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 8
preds = model.predict(
    test_generator,
    verbose=1,
)
print("preds shape:", preds.shape)



## === cell 9
thresh = 0.5
picked_mask = preds >= thresh
has_any = picked_mask.any(axis=1)
argmax_idx = np.argmax(preds, axis=1)

pred_labels = []
for i in range(preds.shape[0]):
    if has_any[i]:
        idxs = np.flatnonzero(picked_mask[i])
    else:
        idxs = np.array([int(argmax_idx[i])], dtype=np.int64)
    pred_labels.append(" ".join([class_names[j] for j in idxs]))

submissions["labels"] = pred_labels
submissions = submissions[["image", "labels"]]
submissions.to_csv("submission.csv", index=False)
submissions.head()



## === cell 10
print(submissions.shape)
print(submissions.columns.tolist())
print(submissions.iloc[:5])
print("Wrote: submission.csv")
