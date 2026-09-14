# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.7229362880886431

# 6. Current score

0.16287

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.16287) has done: 'The update replaces the slow Python‑based ImageDataGenerator loaders with a pure tf.data pipeline that reads and preprocesses images using TensorFlow’s fast I/O ops.  It recreates the exact alphabetical class‑index mapping that the original generator used, so label encoding stays identical.  All other model architecture, training epochs, and evaluation steps remain unchanged, but data loading becomes much more efficient, allowing the whole notebook to finish well under the 600‑second limit.'

# 9. Code solution

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
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(4)
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
train_df["label"] = train_df["labels"].apply(lambda x: x.split()[0])
train_df = train_df.drop(columns=["labels"])

most_common_label = train_df["label"].value_counts().idxmax()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
if tf is not None:
    from sklearn.model_selection import train_test_split

    train_df_split, val_df_split = train_test_split(
        train_df,
        test_size=0.1,
        stratify=train_df["label"],
        random_state=42,
    )

    class_names = sorted(train_df["label"].unique())
    class_indices = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_indices)

    target_sz = (224, 224)
    batch_sz = 64

    def _load_and_preprocess(path, label):
        full_path = tf.strings.join([train_dir, "/", path])
        img = tf.io.read_file(full_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, target_sz)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        label_onehot = tf.one_hot(label, num_classes)
        return img, label_onehot

    train_paths = tf.constant(train_df_split["image"].values)
    train_labels = tf.constant(
        train_df_split["label"].map(class_indices).values, dtype=tf.int32
    )

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
        .shuffle(buffer=len(train_paths), seed=42, reshuffle_each_iteration=True)
        .map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_sz)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_paths = tf.constant(val_df_split["image"].values)
    val_labels = tf.constant(
        val_df_split["label"].map(class_indices).values, dtype=tf.int32
    )

    val_dataset = (
        tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
        .map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_sz)
        .prefetch(tf.data.AUTOTUNE)
    )
else:
    train_dataset = None
    val_dataset = None
    num_classes = None
    class_indices = {}
    target_sz = (224, 224)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3885089981.py in <cell line: 0>()
     36     train_dataset = (
     37         tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
---> 38         .shuffle(buffer=len(train_paths), seed=42, reshuffle_each_iteration=True)
     39         .map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
     40         .batch(batch_sz)

TypeError: DatasetV2.shuffle() got an unexpected keyword argument 'buffer'

## === cell 2
if tf is not None:
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from tensorflow.keras.models import Model

    base_model = EfficientNetB0(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3)
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
else:
    model = None




## === cell 3
if tf is not None and model is not None:
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
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_dataset,
        epochs=2,
        validation_data=val_dataset,
        verbose=2,
    )




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3949436615.py in <cell line: 0>()
      2     # phase 1: train top layers
      3     model.fit(
----> 4         train_dataset,
      5         epochs=3,
      6         validation_data=val_dataset,

NameError: name 'train_dataset' is not defined

## === cell 4
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
    pred_indices = np.argmax(pred_probs, axis=1)
    idx_to_label = {v: k for k, v in class_indices.items()}
    pred_labels = [idx_to_label[idx] for idx in pred_indices]
else:
    pred_labels = [most_common_label] * len(test_df)




## === cell 5
submission = pd.DataFrame({"image": test_df["image"], "labels": pred_labels})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
