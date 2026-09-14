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

0.5444901728004163

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I remove the `tensorflow_addons` dependency that is causing the protobuf `GetPrototype` crash and keep the rest of the pipeline intact. Then I fix the missing pretrained model paths by robustly locating `model_0.h5/model_1.h5/model_2.h5` under `/kaggle/input` (or, if they truly don’t exist, fall back to a simple deterministic baseline that still produces a valid submission). Finally, I make prediction/submission generation deterministic and aligned to `sample_submission.csv` order so the output `submission.csv` is always valid with the required columns and format.'
- What this solution (achieved 0.24507) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by avoiding mixed `keras`/`tensorflow.keras` imports and using only `tf.keras` utilities, which is the common trigger for this error in Kaggle TF environments. I also make model loading more robust by clearing the TF session before loading and enabling safe GPU memory growth to prevent random runtime OOMs. To move your score up toward the target (since 0.245 is far below 0.544), I add a minimal, metric-aligned calibration: per-class thresholds chosen from the training label prevalences (keeps your existing model/ensemble logic intact but reduces empty/over-prediction). Submission generation remains deterministic and strictly aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
import gc
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

tf.keras.utils.set_random_seed(SEED)
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
DATA_ROOT = "/kaggle/input/plant-pathology-2021-fgvc8"
test_dir = os.path.join(DATA_ROOT, "test_images")
train_csv_path = os.path.join(DATA_ROOT, "train.csv")

img_size = (256, 256)

assert os.path.isdir(test_dir), f"Missing test_dir: {test_dir}"
assert os.path.exists(train_csv_path), f"Missing train.csv: {train_csv_path}"



## === cell 3
sample_sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
sample_sub.head()




## === cell 4
def load_images(test_path, image):
    """load image from given path"""
    img = load_img(os.path.join(test_path, image))
    img = img.resize(img_size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img




## === cell 5
label_classes = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]




## === cell 6
def _find_model_path(filename: str):
    """Search /kaggle/input recursively for a given model file."""
    for root, _, files in os.walk("/kaggle/input"):
        if filename in files:
            return os.path.join(root, filename)
    return None


model_paths = {f"model_{i}.h5": _find_model_path(f"model_{i}.h5") for i in range(3)}
model_paths



## === cell 7
train_df = pd.read_csv(train_csv_path)


def _make_multihot(labels_series, classes):
    out = np.zeros((len(labels_series), len(classes)), dtype=np.float32)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for r, s in enumerate(labels_series.fillna("").astype(str).values):
        toks = [t for t in s.split(" ") if t]
        for t in toks:
            if t in class_to_idx:
                out[r, class_to_idx[t]] = 1.0
    return out


y_train = _make_multihot(train_df["labels"], label_classes)
prevalence = y_train.mean(axis=0)

thr_per_class = np.clip(0.15 + 0.35 * (1.0 - prevalence), 0.12, 0.45).astype(np.float32)
thr_per_class



## === cell 8
models = []
missing = [k for k, v in model_paths.items() if v is None]

if len(missing) == 0:
    tf.keras.backend.clear_session()
    for i in range(3):
        p = model_paths[f"model_{i}.h5"]
        models.append(load_model(p, compile=False))
else:
    models = None
    print("WARNING: Missing model files:", missing)
    print(
        "Will create a valid submission using a deterministic baseline (all 'healthy')."
    )




## === cell 9
def get_label(prediction_prob, thresh=0.3, per_class_thresh=None):
    """get label for a class that satisfies given threshold"""
    prediction_prob = prediction_prob[0].astype(np.float32)

    if per_class_thresh is None:
        mask = prediction_prob >= float(thresh)
    else:
        per_class_thresh = np.asarray(per_class_thresh, dtype=np.float32)
        if per_class_thresh.shape[0] != prediction_prob.shape[0]:
            mask = prediction_prob >= float(thresh)
        else:
            mask = prediction_prob >= per_class_thresh

    labels = [label_classes[i] for i, m in enumerate(mask) if bool(m)]
    return " ".join(labels)




## === cell 10
def predict(test_path, threshold):
    """predict on test set, using given threshold"""
    if "image" in sample_sub.columns and len(sample_sub) > 0:
        image_list = sample_sub["image"].tolist()
    else:
        image_list = sorted(os.listdir(test_path))

    images, labels = [], []

    if models is None:
        for image in image_list:
            images.append(image)
            labels.append("healthy")
        return images, labels

    for image in image_list:
        img = load_images(test_path, image)

        pred_prob = (
            (1 / 3 * models[0].predict(img, verbose=0))
            + (1 / 3 * models[1].predict(img, verbose=0))
            + (1 / 3 * models[2].predict(img, verbose=0))
        )

        preds = get_label(pred_prob, thresh=threshold, per_class_thresh=thr_per_class)

        if preds.strip() == "":
            preds = "healthy"

        images.append(image)
        labels.append(preds)

        del img, pred_prob
    gc.collect()
    return images, labels




## === cell 11
image_ids, labels = predict(test_dir, threshold=0.2)



## === cell 12
submission_file = pd.DataFrame({"image": image_ids, "labels": labels})

submission_file = submission_file[["image", "labels"]]

if "image" in sample_sub.columns and len(sample_sub) == len(submission_file):
    submission_file = (
        submission_file.set_index("image").reindex(sample_sub["image"]).reset_index()
    )

submission_file.to_csv("submission.csv", index=False)
submission_file.head()



## === cell 13
assert os.path.exists("submission.csv"), "submission.csv was not created"
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image", "labels"], f"Bad columns: {chk.columns.tolist()}"
assert len(chk) == len(
    sample_sub
), f"Row count mismatch: got {len(chk)} expected {len(sample_sub)}"
chk.tail()
