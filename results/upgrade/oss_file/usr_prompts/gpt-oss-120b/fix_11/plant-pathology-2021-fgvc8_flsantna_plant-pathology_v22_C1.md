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

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_PLATFORM"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score




## === cell 1
base_input = "/kaggle/input/plant-pathology-2021-fgvc8"
test_dir = os.path.join(base_input, "test_images")
train_dir = os.path.join(base_input, "train_images")
model_dir = "/kaggle/input/model-effb7-01/epoch-7/epoch-7"

output_dir = "./"
image_dims = (300, 300, 3)

train_df = pd.read_csv(os.path.join(base_input, "train.csv"))
one_hot = train_df["labels"].str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.tolist()
num_classes = len(dataset_labels)

np.random.seed(42)
tf.random.set_seed(42)


def build_fallback_model():
    """Lightweight EfficientNetB7 multi‑label model (no fine‑tuning)."""
    inputs = tf.keras.Input(shape=image_dims, dtype=tf.float32, name="images")
    base = tf.keras.applications.EfficientNetB7(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling="avg",
    )
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(base.output)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


if tf.io.gfile.isdir(model_dir) and tf.io.gfile.exists(
    os.path.join(model_dir, "saved_model.pb")
):
    model = tf.keras.Sequential(
        [tf.keras.layers.TFSMLayer(model_dir, call_endpoint="serving_default")]
    )
else:
    model = build_fallback_model()

if not isinstance(model.layers[0], tf.keras.layers.TFSMLayer):
    for layer in model.layers:
        layer.trainable = False
    if hasattr(model, "layers") and len(model.layers) > 0:
        model.layers[-1].trainable = True


def tf_preprocess_path_label(path, label):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_dims[0], image_dims[1]])
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, label


train_paths = [os.path.join(train_dir, img_id) for img_id in train_df["image"]]
train_labels = one_hot.values.astype(np.float32)

train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(tf_preprocess_path_label, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()  # <-- added caching
    .shuffle(1024, seed=42)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
)

model.fit(train_ds, epochs=5, verbose=2)




## === cell 2
def tf_preprocess(path):
    raw = tf.io.read_file(path)
    img = tf.image.decode_image(raw, channels=3, expand_animations=False)
    img = tf.image.resize(img, [image_dims[0], image_dims[1]])
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


val_df = train_df.sample(frac=0.10, random_state=42).reset_index(drop=True)
val_paths = [os.path.join(train_dir, img_id) for img_id in val_df["image"]]
val_labels = val_df["labels"].str.get_dummies(sep=" ").values

val_ds = (
    tf.data.Dataset.from_tensor_slices(val_paths)
    .map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()  # <-- added caching
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

all_preds = []
for batch in val_ds:
    preds_orig = model(batch, training=False)
    preds_flip = model(tf.image.flip_left_right(batch), training=False)
    preds_avg = (preds_orig + preds_flip) / 2.0
    all_preds.append(preds_avg.numpy())
val_preds = np.concatenate(all_preds, axis=0)

thresholds = np.arange(0.1, 0.61, 0.05)
per_class_thr = np.full(num_classes, 0.3)
best_f1 = 0.0
for thr in thresholds:
    binarized = (val_preds > thr).astype(int)
    f1 = f1_score(val_labels, binarized, average="macro")
    if f1 > best_f1:
        best_f1 = f1
        per_class_thr[:] = thr

for c in range(num_classes):
    best_f1_c = 0.0
    best_thr_c = 0.3
    for thr in thresholds:
        pred_bin = (val_preds[:, c] > thr).astype(int)
        f1_c = f1_score(val_labels[:, c], pred_bin, average="binary")
        if f1_c > best_f1_c:
            best_f1_c = f1_c
            best_thr_c = thr
    per_class_thr[c] = best_thr_c

print(f"Selected per‑class thresholds (first 5): {per_class_thr[:5]}")
print(f"Validation macro‑F1 (uniform best_thr): {best_f1:.4f}")




## === cell 3
test_images = sorted(
    [
        f
        for f in os.listdir(test_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
        and os.path.isfile(os.path.join(test_dir, f))
    ]
)
test_paths = [os.path.join(test_dir, name) for name in test_images]

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(tf_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()  # <-- added caching
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

submission_rows = []
for batch_paths, batch_imgs in zip(
    tf.data.Dataset.from_tensor_slices(test_paths).batch(32),
    test_ds,
):
    preds_orig = model(batch_imgs, training=False)
    preds_flip = model(tf.image.flip_left_right(batch_imgs), training=False)
    preds_avg = (preds_orig + preds_flip) / 2.0
    preds_np = preds_avg.numpy()

    for i in range(preds_np.shape[0]):
        preds_vec = preds_np[i]
        idxs = [idx for idx, p in enumerate(preds_vec) if p > per_class_thr[idx]]
        labels_str = " ".join([dataset_labels[idx] for idx in idxs])
        img_name = os.path.basename(batch_paths[i].numpy().decode())
        submission_rows.append([img_name, labels_str])

submission_df = pd.DataFrame(submission_rows, columns=["image", "labels"])
submission_path = os.path.join(output_dir, "submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
