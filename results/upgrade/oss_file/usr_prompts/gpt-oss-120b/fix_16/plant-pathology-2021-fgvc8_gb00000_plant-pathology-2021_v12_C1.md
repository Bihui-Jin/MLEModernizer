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
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    import tensorflow as tf
except Exception as e:
    print("TensorFlow import failed:", e)
    tf = None

if tf is not None:
    np.random.seed(42)
    tf.random.set_seed(42)
    tf.config.threading.set_intra_op_parallelism_threads(8)
    tf.config.threading.set_inter_op_parallelism_threads(8)
    try:
        if tf.config.list_physical_devices("GPU"):
            from tensorflow.keras import mixed_precision

            mixed_precision.set_global_policy("mixed_float16")
    except Exception:
        pass

train_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"
train_csv_path = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"

train_df = pd.read_csv(train_csv_path)
train_df["label_list"] = train_df["labels"].apply(lambda x: x.split())
train_df = train_df.drop(columns=["labels"])

most_common_label = train_df["label_list"].explode().value_counts().idxmax()



## === cell 1
if tf is not None:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MultiLabelBinarizer

    train_df["first_label"] = train_df["label_list"].apply(lambda x: x[0])
    train_df_split, val_df_split = train_test_split(
        train_df,
        test_size=0.1,
        stratify=train_df["first_label"],
        random_state=42,
    )

    all_labels = pd.Series(train_df["label_list"].tolist()).explode().unique()
    class_names = sorted(all_labels)
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_indices)

    target_sz = (224, 224)
    batch_sz = 64

    def _load_image(path):
        full_path = tf.strings.join([train_dir, "/", path])
        img = tf.io.read_file(full_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, target_sz)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img

    mlb = MultiLabelBinarizer(classes=class_names)
    train_onehot = mlb.fit_transform(train_df_split["label_list"]).astype(np.float32)
    val_onehot = mlb.transform(val_df_split["label_list"]).astype(np.float32)

    train_paths = tf.constant(train_df_split["image"].values)
    val_paths = tf.constant(val_df_split["image"].values)

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_onehot))
        .map(lambda p, l: (_load_image(p), l), num_parallel_calls=tf.data.AUTOTUNE)
        .cache()  # <<< added cache to speed up repeated epochs
        .shuffle(buffer_size=1000, seed=42, reshuffle_each_iteration=True)
        .batch(batch_sz)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_dataset = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_onehot))
        .map(lambda p, l: (_load_image(p), l), num_parallel_calls=tf.data.AUTOTUNE)
        .cache()
        .batch(batch_sz)
        .prefetch(tf.data.AUTOTUNE)
    )
else:
    train_dataset = None
    val_dataset = None
    num_classes = None
    class_indices = {}
    target_sz = (224, 224)



## === cell 2
if tf is not None and num_classes:
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model

    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="sigmoid")(x)  # sigmoid for multi‑label

    model = Model(inputs=base_model.input, outputs=outputs)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_dataset,
        epochs=3,
        validation_data=val_dataset,
        verbose=2,
    )

    for layer in model.layers:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_dataset,
        epochs=2,
        validation_data=val_dataset,
        verbose=2,
    )
else:
    model = None



## === cell 3
test_files = []
for dirname, _, filenames in os.walk(test_dir):
    for fname in filenames:
        test_files.append(fname)

test_df = pd.DataFrame({"image": test_files})

if tf is not None and model is not None:

    def _load_test(path):
        full_path = tf.strings.join([test_dir, "/", path])
        img = tf.io.read_file(full_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, target_sz)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img

    test_paths = tf.constant(test_df["image"].values)
    test_dataset = (
        tf.data.Dataset.from_tensor_slices(test_paths)
        .map(_load_test, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(32)
        .prefetch(tf.data.AUTOTUNE)
    )

    pred_probs = model.predict(test_dataset, verbose=0)

    idx_to_label = {v: k for k, v in class_indices.items()}
    pred_labels = []
    for probs in pred_probs:
        indices = np.where(probs > 0.5)[0]
        if len(indices) == 0:
            indices = [int(np.argmax(probs))]
        labels = [idx_to_label[i] for i in indices]
        pred_labels.append(" ".join(labels))
else:
    pred_labels = [most_common_label] * len(test_df)



## === cell 4
submission = pd.DataFrame({"image": test_df["image"], "labels": pred_labels})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
