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

0.8218651892890126

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I remove the unnecessary KaggleDatasets import that crashes, correctly build the test image path list, add a safe fallback that creates a dummy model when the pretrained *.h5 file is missing, and ensure the prediction loop runs on the dummy output so a submission CSV is always written with matching lengths. These minimal fixes unblock execution and produce a valid `submission.csv` file.'
- What this solution (achieved 0.24507) has done: 'We replace the disk‑based cache with an in‑memory cache ( .cache() ) for both training and test pipelines and set TensorFlow’s thread pool to use all CPU cores, which removes repeated disk reads across epochs and speeds up data loading without altering the model, loss, or training schedule.'
- What this solution (achieved 0.24507) has done: 'The fix adds a protobuf‑compatibility setting before importing TensorFlow so the actual model can be built and trained, changes the confidence threshold to a more typical 0.5, and replaces the notebook‑only `display` call with a plain `print` to avoid a NameError. These minimal edits unblock the pipeline, enable real predictions, and should raise the F1 score toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import re
import numpy as np
import pandas as pd

try:
    import tensorflow as tf

    print("TensorFlow version:", tf.__version__)
    tf.config.threading.set_intra_op_parallelism_threads(
        tf.config.threading.cpu_count()
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        tf.config.threading.cpu_count()
    )
except Exception as e:
    tf = None
    print("TensorFlow import failed (", e, "); proceeding with dummy model.")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def decode_image(filename, label=None, image_size=(224, 224)):
    if tf is None:
        raise RuntimeError("decode_image requires TensorFlow which is unavailable.")
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)  # scales to [0,1]
    image = tf.image.resize(image, image_size)
    return (image, label) if label is not None else image




## === cell 2
BATCH_SIZE = 32
AUTOTUNE = tf.data.experimental.AUTOTUNE if tf is not None else None

base_path = "/kaggle/input/plant-pathology-2021-fgvc8"
train_dir = os.path.join(base_path, "train_images")
test_dir = os.path.join(base_path, "test_images")

train_df = pd.read_csv(os.path.join(base_path, "train.csv"))
train_df["image_path"] = train_df["image"].apply(lambda x: os.path.join(train_dir, x))

CLASSES = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew", "healthy"]
class_to_idx = {c: i for i, c in enumerate(CLASSES)}


def encode_labels(label_str):
    labels = label_str.split()
    vec = np.zeros(len(CLASSES), dtype=np.float32)
    for l in labels:
        if l in class_to_idx:
            vec[class_to_idx[l]] = 1.0
    return vec


train_df["label_vec"] = train_df["labels"].apply(encode_labels)

IMAGE_PATHS = sorted(
    [
        os.path.join(test_dir, f)
        for f in os.listdir(test_dir)
        if re.search(
            r"([a-zA-Z0-9\s_\\.\-\(\):])+(\.jpg|\.jpeg|\.png)$", f, re.IGNORECASE
        )
    ]
)
print(f"Found {len(IMAGE_PATHS)} test images.")

if tf is not None:
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(
            (train_df["image_path"].values, np.stack(train_df["label_vec"].values))
        )
        .map(lambda x, y: (decode_image(x), y), num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(10000, reshuffle_each_iteration=True)
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    test_dataset = (
        tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
        .map(decode_image, num_parallel_calls=AUTOTUNE)
        .cache()
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
else:
    train_dataset = None
    test_dataset = None




## === cell 3
if tf is not None:
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False  # freeze backbone

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(CLASSES), activation="sigmoid")(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
    model.summary()
else:
    model = None  # dummy placeholder




## === cell 4
EPOCHS = 5
if tf is not None and model is not None and train_dataset is not None:
    model.fit(train_dataset, epochs=EPOCHS, verbose=2)
else:
    print("Skipping training – using dummy predictions.")




## === cell 5
if tf is not None and model is not None and test_dataset is not None:
    probs = model.predict(test_dataset, verbose=0)
else:
    probs = np.zeros((len(IMAGE_PATHS), len(CLASSES)), dtype=np.float32)
    healthy_idx = class_to_idx["healthy"]
    probs[:, healthy_idx] = 1.0




## === cell 6
threshold = 0.5  # adjusted threshold for better precision/recall balance
pred_string = []
for line in probs:
    labels = [CLASSES[i] for i, p in enumerate(line) if p >= threshold]
    if not labels:
        labels = ["healthy"]
    pred_string.append(" ".join(labels))




## === cell 7
submission_df = pd.DataFrame(
    {"image": [os.path.basename(p) for p in IMAGE_PATHS], "labels": pred_string}
)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
print(submission_df.head())
