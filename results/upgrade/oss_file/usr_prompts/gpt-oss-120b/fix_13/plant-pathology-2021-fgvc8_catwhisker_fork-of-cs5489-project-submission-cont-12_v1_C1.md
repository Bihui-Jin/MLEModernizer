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
numpy==1.26.4
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
import pandas as pd
import numpy as np

train_csv_path = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"
train_images_path = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/"
test_images_path = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"

print("number of training images:", len(os.listdir(train_images_path)))
print("number of test images:", len(os.listdir(test_images_path)))

training_csv = pd.read_csv(train_csv_path)

training_class = np.array([])
for labels in pd.unique(training_csv["labels"]):
    training_class = np.append(training_class, labels.split())
training_class = np.unique(training_class)
print("\nnumber of classes:", len(training_class))
print(training_class)




## === cell 1
def predict2strings(pred, threshold):
    """
    Convert sigmoid predictions to space‑delimited label string.
    `threshold` can be a scalar or an array of the same length as `training_class`.
    """
    t = np.array(threshold)
    if t.size == 1:
        t = np.full_like(pred, t.item())
    return " ".join(training_class[np.where(pred >= t)])


def predict2n_hot(pred, threshold):
    t = np.array(threshold)
    if t.size == 1:
        t = np.full_like(pred, t.item())
    return np.where(pred >= t, np.ones(pred.shape), np.zeros(pred.shape))


def n_hot2string(n_hot):
    return " ".join(training_class[np.where(n_hot >= 1)])




## === cell 2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    training_csv,
    test_size=0.10,
    random_state=42,
    stratify=training_csv["labels"],
)

label_counts = {lbl: 0 for lbl in training_class}
total_images = len(training_csv)
for lbls in training_csv["labels"]:
    for l in lbls.split():
        label_counts[l] += 1
label_freq = {lbl: cnt / total_images for lbl, cnt in label_counts.items()}

mlb = MultiLabelBinarizer(classes=training_class)
y_val_true = mlb.fit_transform(val_df["labels"].str.split())

best_f1 = -1.0
best_labels = []

sorted_labels = [
    lbl for lbl, _ in sorted(label_freq.items(), key=lambda x: x[1], reverse=True)
]

for k in range(1, len(sorted_labels) + 1):
    selected = sorted_labels[:k]
    idxs = [np.where(training_class == lbl)[0][0] for lbl in selected]
    y_pred = np.zeros_like(y_val_true)
    y_pred[:, idxs] = 1
    score = f1_score(y_val_true, y_pred, average="samples")
    if score > best_f1:
        best_f1 = score
        best_labels = selected

most_common_combo = training_csv["labels"].value_counts().idxmax()
combo_labels = most_common_combo.split()
combo_idxs = [np.where(training_class == lbl)[0][0] for lbl in combo_labels]
y_pred_combo = np.zeros_like(y_val_true)
y_pred_combo[:, combo_idxs] = 1
combo_f1 = f1_score(y_val_true, y_pred_combo, average="samples")

if combo_f1 > best_f1:
    best_f1 = combo_f1
    best_labels = combo_labels

if not best_labels:
    most_common_lbl = max(label_counts, key=label_counts.get)
    best_labels = [most_common_lbl]
    best_f1 = 0.0

baseline_pred_str = " ".join(best_labels)
print(f"Chosen baseline labels (constant prediction): {baseline_pred_str}")
print(f"Validation mean F1 of this baseline: {best_f1:.4f}")

threshold = [0.5] * len(training_class)




## === cell 3
def write_csv_kaggle_tags():
    import csv

    rows = [["image", "labels"]]
    for filename in os.listdir(test_images_path):
        if os.path.isdir(os.path.join(test_images_path, filename)):
            continue

        if model_available:
            import tensorflow as tf

            img_path = os.path.join(test_images_path, filename)
            image = tf.keras.preprocessing.image.load_img(
                img_path, target_size=(224, 224)
            )
            image = tf.keras.preprocessing.image.img_to_array(image) / 255.0
            image = tf.expand_dims(image, axis=0)
            pred_score = model(image)[0].numpy()
            label_str = predict2strings(pred_score, threshold=threshold)
        else:
            label_str = baseline_pred_str

        rows.append([filename, label_str])

    submission_path = "/kaggle/working/submission.csv"
    with open(submission_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"submission.csv written to {submission_path} with {len(rows) - 1} records.")




## === cell 4
model = None
model_available = False
try:
    import tensorflow as tf

    num_classes = len(training_class)

    def _load_image(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = img / 255.0
        return img

    train_paths = (
        train_df["image"].apply(lambda x: os.path.join(train_images_path, x)).values
    )
    val_paths = (
        val_df["image"].apply(lambda x: os.path.join(train_images_path, x)).values
    )

    train_labels = mlb.transform(train_df["labels"].str.split()).astype(np.float32)
    val_labels = mlb.transform(val_df["labels"].str.split()).astype(np.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    train_ds = train_ds.map(
        lambda p, l: (_load_image(p), l), num_parallel_calls=tf.data.AUTOTUNE
    )
    train_ds = train_ds.shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(
        lambda p, l: (_load_image(p), l), num_parallel_calls=tf.data.AUTOTUNE
    )
    val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
    )
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(base.output)
    model = tf.keras.Model(inputs=base.input, outputs=outputs)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="binary_crossentropy")

    model.fit(train_ds, validation_data=val_ds, epochs=2, verbose=2)

    model_available = True
    print("Model trained and ready for inference.")
except Exception as e:
    print("Model training failed, falling back to constant baseline. Error:", e)
    model = None
    model_available = False



## === cell 5
write_csv_kaggle_tags()
