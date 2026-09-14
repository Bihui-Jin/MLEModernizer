# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.24507

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'Your notebook currently can’t yield a score because it fails before creating a valid submission: the weights path points to a dataset that isn’t present in the provided file tree, and even if it ran, the CSV writing risks wrong row order and label decoding issues due to Tensor/NumPy mixing. I make the smallest changes to (1) robustly locate and load the weight file if it exists in the competition input directory, otherwise fall back to a “healthy-only” submission (valid CSV so you can obtain a score), and (2) ensure predictions are converted to NumPy consistently and written in the exact sample_submission order. These changes preserve your model and thresholds exactly and only touch loading + submission generation correctness. This should move you from “no score” to a valid scored submission, which is the necessary first step toward the target 0.7456.'
- What this solution (achieved 0.24507) has done: 'I fix the TensorFlow import crash caused by an incompatible protobuf version by pinning protobuf to a TF‑compatible release at runtime (a minimal environment fix that unblocks execution). Then I keep your exact model/threshold logic, but make sure the label list is never empty (fallback to `healthy`) so the submission matches the competition’s expected format and typically improves F1 versus blank labels. Finally, I make data paths robust (`/kaggle/input/...`) and keep the submission row order exactly as `sample_submission.csv` to avoid silent misalignment.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import numpy as np
import pandas as pd


def _ensure_tf_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            raise RuntimeError(f"Incompatible protobuf version detected: {pb_ver}")
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        import importlib
        import google.protobuf  # noqa: F401

        importlib.reload(google.protobuf)


_ensure_tf_protobuf_compat()

print("number of training image")
train_img_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/"
test_img_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
print(len([name for name in os.listdir(train_img_dir)]))
print(len([name for name in os.listdir(test_img_dir)]))

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
    pred = np.asarray(pred).reshape(-1)
    t = np.asarray(threshold).reshape(-1)
    s = " ".join(training_class[np.where(pred >= t)[0]])
    return s if s.strip() else "healthy"


def predict2n_hot(pred, threshold):
    pred = np.asarray(pred)
    t = np.asarray(threshold)
    return np.where(pred >= t, np.ones(pred.shape), np.zeros(pred.shape))


def n_hot2string(n_hot):
    n_hot = np.asarray(n_hot).reshape(-1)
    s = " ".join(training_class[np.where(n_hot >= 1)[0]])
    return s if s.strip() else "healthy"




## === cell 2
import tensorflow as tf

model_input = tf.keras.layers.Input(shape=(299, 299, 3))
xception_layer = tf.keras.applications.Xception(
    include_top=False,
    weights=None,
    input_shape=(299, 299, 3),
    input_tensor=model_input,
    pooling="max",
).output
fc_1 = tf.keras.layers.Dense(1024)(xception_layer)
fc_2 = tf.keras.layers.Dense(512)(fc_1)
model_output = tf.keras.layers.Dense(len(training_class), activation="sigmoid")(fc_2)
model = tf.keras.Model(inputs=model_input, outputs=model_output)




## === cell 3
def _find_weights_path():
    p0 = "../input/cs5489-project-train-baseline/my_model/model_1"
    if tf.io.gfile.exists(p0) or tf.io.gfile.exists(p0 + ".index"):
        return p0

    for root, _, files in os.walk("/kaggle/input"):
        if "model_1.index" in files:
            return os.path.join(root, "model_1")
        if "model_1" in files:
            return os.path.join(root, "model_1")
    return None


weights_path = _find_weights_path()
if weights_path is not None:
    model.load_weights(weights_path)
    print(f"Loaded weights from: {weights_path}")
else:
    print(
        "WARNING: model weights not found under /kaggle/input; will write a valid fallback submission."
    )



## === cell 4
threshold = [
    0.5469251871109009,
    0.4226226806640625,
    0.9167647957801819,
    0.1412835568189621,
    0.22426636517047882,
    0.4077064096927643,
]
test_img_path = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"


def write_csv_kaggle_tags():
    sample = pd.read_csv(
        "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
    )

    if weights_path is None:
        sub = sample.copy()
        sub["labels"] = "healthy"
        sub.to_csv("submission.csv", index=False)
        return

    labels_out = []
    for fname in sample["image"].tolist():
        img_file = os.path.join(test_img_path, fname)
        image = tf.keras.preprocessing.image.load_img(img_file, target_size=(299, 299))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = tf.image.per_image_standardization(image)
        image = tf.expand_dims(image, axis=0)
        test_predscore = model(image, training=False)[0].numpy()
        labels_out.append(predict2strings(test_predscore, threshold=threshold))

    sub = pd.DataFrame({"image": sample["image"], "labels": labels_out})
    sub.to_csv("submission.csv", index=False)




## === cell 5
write_csv_kaggle_tags()
print(pd.read_csv("submission.csv").head())
print("Wrote submission.csv with rows:", len(pd.read_csv("submission.csv")))
