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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.78649138712602

# 6. Current score

0.62033

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I remove the failing TFRecord parsing and unnecessary torch imports, load the Keras model safely (falling back to dummy predictions if the file is missing), read the test images directly from the provided folder, generate predictions with the loaded model (or a default class), and finally write a correctly‑formatted `submission.csv`. This fixes the runtime errors, ensures a valid CSV is produced, and keeps the core model‑prediction logic intact.'
- What this solution (achieved 0.60949) has done: 'The changes fix the TensorFlow import failure handling, correctly distinguish between a Keras model and the RandomForest fallback (removing the unsupported `verbose` argument), and ensure predictions are generated for every image listed in the official `sample_submission.csv`. By iterating over the sample submission’s image IDs we guarantee the output CSV has the exact required length and order, producing a valid submission file.'
- What this solution (achieved 0.61622) has done: 'The fix expands the fallback feature set (means, stds + color histograms) and trains the RandomForest on the full training set with stronger hyper‑parameters, which raises validation accuracy toward the target while keeping the original logic unchanged. It also restructures the notebook into sequential cells and adds the new `extract_features` helper used both for training and inference, ensuring a correct `submission.csv` is always written.'
- What this solution (achieved 0.61921) has done: 'I added the missing `RandomForestClassifier` import, which fixes the NameError and allows the fallback model to be trained. With the import corrected, the prediction loop runs without raising an exception, so the lengths of `image_ids` and `predictions` match and a proper `submission.csv` is written. No other logic changes were made, preserving the original model‑fallback design while ensuring the script completes successfully.'
- What this solution (achieved 0.62145) has done: 'The fix adds richer HSV‑color histogram features to the handcrafted extractor and strengthens the fallback RandomForest by using more trees, which should boost validation accuracy toward the target while keeping the original workflow intact. No other logic is altered, and the script still writes a correctly‑formatted `submission.csv`.'
- What this solution (achieved 0.61958) has done: 'The changes switch to process‑based parallelism for feature extraction (which is CPU‑bound) and batch‑predict all test images when using the RandomForest fallback, eliminating per‑image prediction overhead. This keeps the exact same preprocessing, model, and prediction logic while substantially reducing total runtime.'
- What this solution (achieved 0.61996) has done: 'Implemented robust TensorFlow handling by forcing a fallback when TF import fails, preventing the protobuf error. Enhanced the fallback RandomForest model with stronger hyper‑parameters (more trees and `max_features='sqrt'`) to boost validation accuracy toward the target. Added explanatory comments and kept the original workflow intact, ensuring a correctly formatted `submission.csv` is always generated.'
- What this solution (achieved 0.62108) has done: 'Implemented a richer feature extractor by adding mean and std for HSV channels (increasing the vector size from 114 → 120) and slightly boosted the fallback RandomForest with more trees (8000). These changes keep the original workflow intact while providing the model with extra discriminative information, nudging validation accuracy toward the target score.'
- What this solution (achieved 0.62145) has done: 'Implemented richer feature extraction by expanding RGB/HSV histograms to 32 bins each, added skewness‑like variability via larger bin resolution, and introduced an ensemble fallback model using both RandomForest and ExtraTrees via a hard‑voting classifier. These changes keep the original workflow while providing a more discriminative feature set and a stronger classifier, aiming to raise validation accuracy toward the target score. The script now reliably writes a correctly‑formatted `submission.csv`.'
- What this solution (achieved 0.62033) has done: 'I change the voting ensemble to use soft voting, which averages class probabilities from the RandomForest and ExtraTrees models instead of hard voting on class votes. This small tweak can improve predictive performance and move the validation score closer to the target while preserving all existing logic.'

# 9. Code solution

## === cell 0
import os
import warnings
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)  # added ExtraTrees and Voting
import concurrent.futures

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception as e:
    tf = None
    warnings.warn(f"TensorFlow import failed: {e}. Falling back to sklearn model.")

    def load_model(path):
        raise FileNotFoundError("TensorFlow not available; cannot load Keras model.")


def second_model_preprocess(img_np):
    """
    Resize image to 224x224 and scale to [0,1] as expected by many Keras models.
    """
    img = Image.fromarray(img_np.astype("uint8"), "RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def extract_features(img_np):
    """
    Produce a compact feature vector from an image:
    - mean, std, min, max, 25th and 75th percentiles per RGB channel (18 values)
    - 32‑bin normalized histogram per RGB channel (96 values)
    - 32‑bin normalized histogram per HSV channel (96 values)
    - mean and std per HSV channel (6 values)
    Total length = 216.
    """
    img = Image.fromarray(img_np.astype("uint8"), "RGB").resize((64, 64))
    arr = np.array(img) / 255.0  # shape (64,64,3)

    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    mins = arr.min(axis=(0, 1))
    maxs = arr.max(axis=(0, 1))
    perc25 = np.percentile(arr, 25, axis=(0, 1))
    perc75 = np.percentile(arr, 75, axis=(0, 1))

    hist_r, _ = np.histogram(arr[:, :, 0], bins=32, range=(0, 1), density=True)
    hist_g, _ = np.histogram(arr[:, :, 1], bins=32, range=(0, 1), density=True)
    hist_b, _ = np.histogram(arr[:, :, 2], bins=32, range=(0, 1), density=True)

    hsv_img = img.convert("HSV")
    hsv_arr = np.array(hsv_img) / 255.0
    hist_h, _ = np.histogram(hsv_arr[:, :, 0], bins=32, range=(0, 1), density=True)
    hist_s, _ = np.histogram(hsv_arr[:, :, 1], bins=32, range=(0, 1), density=True)
    hist_v, _ = np.histogram(hsv_arr[:, :, 2], bins=32, range=(0, 1), density=True)

    mean_hsv = hsv_arr.mean(axis=(0, 1))
    std_hsv = hsv_arr.std(axis=(0, 1))

    return np.concatenate(
        [
            mean,
            std,
            mins,
            maxs,
            perc25,
            perc75,
            hist_r,
            hist_g,
            hist_b,
            hist_h,
            hist_s,
            hist_v,
            mean_hsv,
            std_hsv,
        ]
    )




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
model_path = "/kaggle/input/newmodel60/keras/default/1/newModel60.keras"
model2 = None
try:
    model2 = load_model(model_path)
    print(f"Keras model loaded from {model_path}")
except Exception as e:
    print(f"Warning: could not load model at {model_path}. Reason: {e}")

if model2 is None:
    train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
    train_img_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
    if not os.path.isdir(train_img_dir):
        train_img_dir = "/kaggle/input/train_images"

    train_df = pd.read_csv(train_csv_path)

    tasks = []
    for _, row in train_df.iterrows():
        img_path = os.path.join(train_img_dir, row["image_id"])
        if os.path.isfile(img_path):
            tasks.append((img_path, int(row["label"])))  # store label as int for later

    def process_task(task):
        img_path, label = task
        img_np = np.array(Image.open(img_path).convert("RGB"))
        feat = extract_features(img_np)
        return feat, label

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(executor.map(process_task, tasks))

    feats, labs = zip(*results) if results else ([], [])
    X = np.array(feats)
    y = np.array(labs)

    rf = RandomForestClassifier(
        n_estimators=8000,
        max_features="sqrt",
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    et = ExtraTreesClassifier(
        n_estimators=4000,
        max_features="sqrt",
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    voting = VotingClassifier(
        estimators=[("rf", rf), ("et", et)],
        voting="soft",  # use soft voting to average probabilities
        n_jobs=-1,
    )
    voting.fit(X, y)
    model2 = voting
    print(
        "Fallback ensemble (RandomForest + ExtraTrees) trained with enhanced features on the full dataset."
    )




## === cell 2
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
if not os.path.isfile(sample_sub_path):
    sample_sub_path = "/kaggle/input/sample_submission.csv"

sample_df = pd.read_csv(sample_sub_path)
image_ids = sample_df["image_id"].tolist()

test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
if not os.path.isdir(test_image_dir):
    test_image_dir = "/kaggle/input/test_images"


if (
    tf is not None
    and hasattr(model2, "predict")
    and not isinstance(
        model2, (RandomForestClassifier, ExtraTreesClassifier, VotingClassifier)
    )
):

    def predict_one(img_id):
        img_path = os.path.join(test_image_dir, img_id)
        if not os.path.isfile(img_path):
            return 0  # default class when missing

        img_np = np.array(Image.open(img_path).convert("RGB"))
        img_input = second_model_preprocess(img_np)
        probs = model2.predict(img_input)[0]
        return int(np.argmax(probs))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        predictions = list(executor.map(predict_one, image_ids))
else:

    def extract_one(img_id):
        img_path = os.path.join(test_image_dir, img_id)
        if not os.path.isfile(img_path):
            return np.zeros(216, dtype=np.float32)  # match new feature length
        img_np = np.array(Image.open(img_path).convert("RGB"))
        return extract_features(img_np)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        test_feats = list(executor.map(extract_one, image_ids))

    if test_feats:
        X_test = np.vstack(test_feats)
        predictions = model2.predict(X_test).astype(int).tolist()
    else:
        predictions = [0] * len(image_ids)




## === cell 3
submission = pd.DataFrame({"image_id": image_ids, "label": predictions})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file saved as {submission_path} with {len(submission)} rows.")
