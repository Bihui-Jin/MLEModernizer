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

# 5. Target score

0.7456140350877198

# 6. Current score

0.21757

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.21436) has done: 'I fix the crashes by (1) replacing the Xception backbone with a tiny pure‑CNN that works with the installed TensorFlow version, (2) removing the unsupported weight‑loading step, and (3) guarding the test‑image loop against sub‑folders so the image loader receives only actual files. These minimal changes unblock the notebook, let it write a correctly formatted `submission.csv`, and keep the original multi‑label handling logic intact.'
- What this solution (achieved 0.07232) has done: 'I remove the early TensorFlow import that caused a protobuf error, add the import later where needed, and introduce a simple training loop that loads the images, creates multi‑hot label vectors, and fits the tiny CNN for a few epochs. This enables the model to learn from the data, which should raise the mean F1‑Score toward the target while keeping the original architecture and inference code intact. The script now also reliably writes a correctly‑formatted `submission.csv`.'
- What this solution (achieved 0.05331) has done: 'The changes add a cache to the training pipeline so images are read from disk only once, and replace the slow Python‑loop inference with a batched tf.data pipeline that processes all test images in parallel while preserving the exact same preprocessing and prediction logic. These adjustments keep the model architecture, loss, and training epochs unchanged, but dramatically cut I/O and per‑image Python overhead, allowing the whole script to finish well within the 600‑second limit.'
- What this solution (achieved 0.12246) has done: 'I guard the TensorFlow import with a fallback that skips model building and training when TF cannot be imported (the protobuf error). Instead, I compute per‑class frequencies from the training labels and use these frequencies as constant prediction scores for every test image. This eliminates the crash, ensures a valid `submission.csv` is written, and provides a simple heuristic that should raise the mean F1‑Score compared to the previous near‑zero result while keeping the original pipeline structure intact.'
- What this solution (achieved 0.12035) has done: 'I fixed the TensorFlow import reliability by validating the import with a tiny operation; if it fails we safely fall back to the non‑TF path. Since the fallback predicts the same class‑frequency vector for every image, I lowered the decision threshold to the mean class frequency (instead of the original 0.5) so that more frequent labels are emitted, which improves the mean F1‑Score while keeping the original pipeline logic intact. The script now always writes a correctly formatted `submission.csv` regardless of TF availability.'
- What this solution (achieved 0.18301) has done: 'I added a safe fallback that activates when TensorFlow cannot be imported. Instead of using only class‑frequency guesses, the fallback now trains a very lightweight One‑Vs‑Rest logistic‑regression model on simple image‑level features (mean RGB values) extracted with Pillow. This provides per‑image predictions and raises the mean F1‑Score toward the target while keeping the original TensorFlow workflow unchanged for environments where TF works. All other logic, file paths, and submission formatting remain identical.'
- What this solution (achieved 0.08293) has done: 'I added a robust guard for the TensorFlow import so that any import‑related error (including protobuf mismatches) safely disables TF usage and defines `tf = None`. The fallback path now extracts richer image features by resizing each image to 32×32 pixels and flattening the RGB values (instead of just mean RGB), which gives the logistic‑regression model far more information. I also lowered the probability threshold to 0.3 to emit more labels, helping lift the mean F1‑Score toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 0.21757) has done: 'The changes upgrade the fallback model: instead of a simple logistic‑regression, we train a One‑Vs‑Rest RandomForest classifier on 32×32 flattened RGB features, and we automatically select the probability threshold that maximizes macro‑averaged F1 on a held‑out validation split. This keeps the original TensorFlow‑fallback logic intact while substantially improving the multi‑label predictions, moving the score toward the target. The script also renumbers cells to start at 1 as required.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import pandas as pd
import numpy as np

print("number of training images")
train_img_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/"
test_img_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
print(
    len(
        [
            name
            for name in os.listdir(train_img_dir)
            if os.path.isfile(os.path.join(train_img_dir, name))
        ]
    )
)
print(
    len(
        [
            name
            for name in os.listdir(test_img_dir)
            if os.path.isfile(os.path.join(test_img_dir, name))
        ]
    )
)

training_csv = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
training_class = np.array([])
for labels in pd.unique(training_csv["labels"]):
    training_class = np.append(training_class, labels.split())
training_class = np.unique(training_class)
print("\nnumber of class")
print(training_class)
print(len(training_class))




## === cell 1
def predict2strings(pred, threshold):
    """
    Convert sigmoid outputs to a space‑delimited string of class names.
    `threshold` is a scalar applied to every class.
    """
    return " ".join(training_class[pred >= threshold])


def predict2n_hot(pred, threshold):
    """
    Return a binary (n‑hot) array using the same scalar threshold.
    """
    return np.where(pred >= threshold, 1, 0)


def n_hot2string(n_hot):
    """Convert an n‑hot vector back to the space‑delimited label string."""
    return " ".join(training_class[n_hot == 1])




## === cell 2
try:
    import tensorflow as tf

    _ = tf.constant(0)
    tf_available = True
    print("TensorFlow imported and sanity‑checked successfully.")
except Exception as e:
    tf = None  # ensure tf name exists later
    tf_available = False
    print("TensorFlow import failed or is not functional; using fallback model.")
    print("Import error:", e)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
train_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/"
train_df = training_csv.copy()
train_df["filepath"] = train_dir + train_df["image"]

label_to_idx = {c: i for i, c in enumerate(training_class)}


def labels_to_multi_hot(label_str):
    vec = np.zeros(len(training_class), dtype=np.float32)
    for lbl in label_str.split():
        vec[label_to_idx[lbl]] = 1.0
    return vec


train_df["multi_hot"] = train_df["labels"].apply(labels_to_multi_hot)

paths = train_df["filepath"].values
labels = np.stack(train_df["multi_hot"].values)

if tf_available:
    AUTOTUNE = tf.data.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _process(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [299, 299])
        img = tf.image.per_image_standardization(img)
        return img, label

    dataset = dataset.map(_process, num_parallel_calls=AUTOTUNE).cache()
    dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(AUTOTUNE)

    model_input = tf.keras.layers.Input(shape=(299, 299, 3))
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(model_input)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    model_output = tf.keras.layers.Dense(len(training_class), activation="sigmoid")(x)

    model = tf.keras.Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(dataset, epochs=12, verbose=2)

    fallback_threshold = 0.5  # not used when TF is available
else:
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score

    def img_features(path):
        """Resize to 32×32 and flatten RGB channels (values in [0, 1])."""
        with Image.open(path) as img:
            img = img.convert("RGB")
            img = img.resize((32, 32))
            arr = np.array(img).astype(np.float32) / 255.0
            return arr.flatten()

    print("Extracting low‑dimensional features for training images...")
    train_features = np.stack([img_features(p) for p in paths])

    X_tr, X_val, y_tr, y_val = train_test_split(
        train_features, labels, test_size=0.1, random_state=42
    )

    print("Training One‑Vs‑Rest RandomForest classifier...")
    rf_clf = OneVsRestClassifier(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
        )
    )
    rf_clf.fit(X_tr, y_tr)

    val_probs = rf_clf.predict_proba(X_val)
    best_thr = 0.5
    best_f1 = 0.0
    for thr in np.arange(0.1, 0.6, 0.05):
        val_pred = (val_probs >= thr).astype(int)
        f1 = f1_score(y_val, val_pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    print(f"Best validation macro‑F1: {best_f1:.4f} at threshold {best_thr:.2f}")

    fallback_model = rf_clf
    fallback_threshold = best_thr  # use the optimised threshold



## === cell 4
threshold = 0.5  # default for the TF path
test_img_path = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"


def write_csv_kaggle_tags():
    import csv
    from PIL import Image

    test_files = sorted(
        [
            f
            for f in os.listdir(test_img_path)
            if os.path.isfile(os.path.join(test_img_path, f))
        ]
    )
    test_paths = [os.path.join(test_img_path, f) for f in test_files]

    if tf_available:
        AUTOTUNE = tf.data.AUTOTUNE
        test_ds = tf.data.Dataset.from_tensor_slices(test_paths)

        def _process_test(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [299, 299])
            img = tf.image.per_image_standardization(img)
            return img

        test_ds = (
            test_ds.map(_process_test, num_parallel_calls=AUTOTUNE)
            .batch(32)
            .prefetch(AUTOTUNE)
        )

        preds = model.predict(test_ds, verbose=0)
        used_threshold = threshold
    else:

        def img_features(path):
            with Image.open(path) as img:
                img = img.convert("RGB")
                img = img.resize((32, 32))
                arr = np.array(img).astype(np.float32) / 255.0
                return arr.flatten()

        print("Extracting features for test images...")
        test_features = np.stack([img_features(p) for p in test_paths])
        preds = fallback_model.predict_proba(test_features)
        used_threshold = fallback_threshold

    rows = [["image", "labels"]]
    for filename, pred_scores in zip(test_files, preds):
        label_str = predict2strings(pred_scores, used_threshold)
        rows.append([filename, label_str])

    output_path = "/kaggle/working/submission.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print(f"Submission written to {output_path}")


write_csv_kaggle_tags()
