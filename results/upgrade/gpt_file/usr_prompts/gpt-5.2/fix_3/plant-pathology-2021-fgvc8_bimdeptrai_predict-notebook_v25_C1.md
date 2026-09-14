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
from tensorflow import keras

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)



## === cell 1
train = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
train.head()



## === cell 2
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
submissions.head()



## === cell 3
h_target = 384
w_target = 384
batch_size = 32



## === cell 4
test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_data_generator.flow_from_dataframe(
    submissions,
    directory="../input/plant-pathology-2021-fgvc8/test_images",
    x_col="image",
    y_col=None,
    target_size=(h_target, w_target),
    color_mode="rgb",
    classes=None,
    class_mode=None,
    shuffle=False,
    batch_size=batch_size,
)



## === cell 5
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer().fit(label_split)
class_names = list(mlb.classes_)
labels_df = pd.DataFrame(mlb.transform(label_split), columns=class_names)

print("Num classes:", len(class_names))
print("Classes:", class_names)



## === cell 6
model_path = "../input/resnet101-512-to-384/resnet101.h5"

model = None
loaded_external = False
if os.path.exists(model_path):
    model = keras.models.load_model(model_path, compile=False)
    loaded_external = True
    print("Loaded model from:", model_path)
else:
    print("WARNING: Model file not found:", model_path)
    print(
        "Falling back to ImageNet-pretrained EfficientNetB0 with a new classification head."
    )
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(h_target, w_target, 3),
        pooling="avg",
    )
    x = tf.keras.layers.Dropout(0.2)(base.output)
    out = tf.keras.layers.Dense(len(class_names), activation="sigmoid")(x)
    model = tf.keras.Model(inputs=base.input, outputs=out)




## === cell 7
def multilabel_f1_micro(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
    return f1


if not loaded_external:
    idx = np.arange(len(train))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    val_size = int(0.10 * len(idx))
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    train_df = train.iloc[tr_idx].reset_index(drop=True).copy()
    val_df = train.iloc[val_idx].reset_index(drop=True).copy()

    for c in class_names:
        train_df[c] = labels_df.iloc[tr_idx][c].values
        val_df[c] = labels_df.iloc[val_idx][c].values

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        vertical_flip=False,
        rotation_range=10,
        zoom_range=0.10,
    )
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        directory="../input/plant-pathology-2021-fgvc8/train_images",
        x_col="image",
        y_col=class_names,
        target_size=(h_target, w_target),
        color_mode="rgb",
        class_mode="raw",
        shuffle=True,
        seed=SEED,
        batch_size=batch_size,
    )

    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        directory=(
            "../input/plant-pathology-2021-fgvcvc8/train_images"
            if False
            else "../input/plant-pathology-2021-fgvc8/train_images"
        ),
        x_col="image",
        y_col=class_names,
        target_size=(h_target, w_target),
        color_mode="rgb",
        class_mode="raw",
        shuffle=False,
        batch_size=batch_size,
    )

    for layer in model.layers:
        layer.trainable = True
    if hasattr(model.layers[0], "layers") or isinstance(
        model.layers[0], tf.keras.Model
    ):
        for lyr in model.layers:
            if isinstance(lyr, tf.keras.layers.Dense):
                lyr.trainable = True
            else:
                lyr.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[multilabel_f1_micro],
    )

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=3,
        verbose=1,
    )

    for lyr in model.layers:
        lyr.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[multilabel_f1_micro],
    )
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=1,
        verbose=1,
    )

    val_preds = model.predict(val_gen, verbose=1)
    val_true = val_df[class_names].values.astype(np.int32)

    grid = np.arange(0.05, 0.60, 0.05)

    best_thresh = np.full(len(class_names), 0.25, dtype=np.float32)
    for j in range(len(class_names)):
        yt = val_true[:, j]
        yp = val_preds[:, j]
        best_f1 = -1.0
        best_t = 0.25
        for t in grid:
            yhat = (yp >= t).astype(np.int32)
            tp = np.sum((yt == 1) & (yhat == 1))
            fp = np.sum((yt == 0) & (yhat == 1))
            fn = np.sum((yt == 1) & (yhat == 0))
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_t = t
        best_thresh[j] = best_t

    print("Per-class thresholds calibrated on validation.")
else:
    best_thresh = None



## === cell 8
preds = model.predict(test_generator, verbose=1)
print("preds shape:", preds.shape)
print(preds[:2])



## === cell 9
assert preds.shape[0] == len(
    submissions
), "Prediction count must match submission rows."



## === cell 10
default_thresh = 0.25
healthy_idx = class_names.index("healthy") if "healthy" in class_names else None

final_labels = []
for i in range(len(submissions)):
    row = preds[i]

    if healthy_idx is not None and int(np.argmax(row)) == healthy_idx:
        lbl = "healthy"
    else:
        chosen = []
        for j in range(len(class_names)):
            t = float(best_thresh[j]) if best_thresh is not None else default_thresh
            if row[j] >= t:
                chosen.append(class_names[j])

        if (len(chosen) == 0) or ("healthy" in chosen):
            chosen = [class_names[int(np.argmax(row))]]

        lbl = " ".join(chosen)

    final_labels.append(lbl)

sub_out = submissions.copy()
sub_out["labels"] = final_labels

sub_out[["image", "labels"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)



## === cell 11
sub_out.head()
