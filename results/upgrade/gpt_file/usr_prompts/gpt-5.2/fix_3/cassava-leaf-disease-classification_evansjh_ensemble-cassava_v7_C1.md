# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import glob
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
np.random.seed(42)



## === cell 1
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

sample_csv = pd.read_csv(sample_path)
train_df = pd.read_csv(train_csv_path)

print("sample_csv shape:", sample_csv.shape)
print("train_df shape:", train_df.shape)
print("test_image_dir exists:", os.path.isdir(test_image_dir))
print("train_image_dir exists:", os.path.isdir(train_image_dir))



## === cell 2
model_path_1 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/BestModel_3454_8937.h5"
)
model_path_2 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/best_model_0.37458707.h5"
)
model_path_3 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/googlenet_inceptionv3.h5"
)
model_path_4 = (
    "/kaggle/input/bestmodel_550_2/tensorflow2/default/1/BestModel_3577_8940.h5"
)
model_path_5 = (
    "/kaggle/input/bestmodel_8878/tensorflow2/default/1/BestModel_8878_0358.h5"
)
model_path_6 = "/kaggle/input/bestmodel_8875/tensorflow2/default/1/BestModel_8875.h5"

models_info = [
    (model_path_1, (550, 550)),
    (model_path_2, (512, 512)),
    (model_path_3, (448, 448)),
    (model_path_6, (512, 512)),
]



## === cell 3
class_labels = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}



def _extract_features_pil(image_path, size):
    """Simple, deterministic feature extractor: resize -> RGB -> [0,1] -> flatten."""
    try:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            im = im.resize(size, resample=Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr.reshape(-1)
    except Exception:
        return None


def train_sklearn_ensemble(
    train_df, train_image_dir, sizes, per_size_max_samples=5000, random_state=42
):
    """
    Train one LogisticRegression per input size (acts as one 'model' in the ensemble).
    Minimal changes, deterministic sampling, and fast enough for Kaggle CPU.
    """
    loaded_models = []

    df = train_df.copy()
    df["path"] = df["image_id"].apply(lambda x: os.path.join(train_image_dir, x))
    df = df[df["path"].apply(os.path.exists)].reset_index(drop=True)

    if len(df) == 0:
        return loaded_models

    for size in sizes:
        if per_size_max_samples is not None and len(df) > per_size_max_samples:
            df_sample, _ = train_test_split(
                df,
                train_size=per_size_max_samples,
                random_state=random_state,
                stratify=df["label"],
            )
        else:
            df_sample = df

        X_list = []
        y_list = []
        for img_id, y, p in zip(
            df_sample["image_id"].values,
            df_sample["label"].values,
            df_sample["path"].values,
        ):
            feat = _extract_features_pil(p, size)
            if feat is None:
                continue
            X_list.append(feat)
            y_list.append(int(y))

        if len(X_list) < 50:
            continue

        X = np.vstack(X_list)
        y = np.asarray(y_list, dtype=np.int64)

        clf = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=200,
            n_jobs=1,
            random_state=random_state,
        )
        clf.fit(X, y)

        loaded_models.append((clf, size, f"sklearn_logreg_{size[0]}x{size[1]}"))
        print(
            f"Trained sklearn model for size={size} on n={len(y)} samples, dim={X.shape[1]}"
        )

    return loaded_models


def predict_one_image_ensemble(image_path, models):
    """
    Core logic preserved: each model predicts argmax; final class by majority vote;
    tie-break by highest average confidence among tied classes.
    """
    model_predictions = []
    confidence_scores = {}

    for model, input_size, _path in models:
        feat = _extract_features_pil(image_path, input_size)
        if feat is None:
            continue

        proba = model.predict_proba(feat.reshape(1, -1))[0]
        predicted_class = int(np.argmax(proba))
        confidence_score = float(proba[predicted_class])

        model_predictions.append(predicted_class)
        confidence_scores.setdefault(predicted_class, []).append(confidence_score)

    if len(model_predictions) == 0:
        return None  # caller will fallback

    class_votes = Counter(model_predictions)
    most_common = class_votes.most_common()
    final_predicted_class = most_common[0][0]

    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        tied_classes = [cls for cls, count in most_common if count == most_common[0][1]]
        final_predicted_class = max(
            tied_classes,
            key=lambda cls: sum(confidence_scores.get(cls, [0.0]))
            / max(1, len(confidence_scores.get(cls, []))),
        )

    return final_predicted_class


sizes = [info[1] for info in models_info]
loaded_models = train_sklearn_ensemble(
    train_df=train_df,
    train_image_dir=train_image_dir,
    sizes=sizes,
    per_size_max_samples=5000,  # speed-safe while improving accuracy vs majority baseline
    random_state=42,
)

print(f"Loaded {len(loaded_models)} models.")
for _, s, p in loaded_models:
    print(" model:", p, "size:", s)



## === cell 4
available_test_images = (
    set(os.listdir(test_image_dir)) if os.path.isdir(test_image_dir) else set()
)
majority_label = int(train_df["label"].value_counts().idxmax())

image_predictions = []

if len(loaded_models) == 0:
    print(
        "No models trained; writing fallback submission with majority label =",
        majority_label,
    )
    submission_df = sample_csv.copy()
    submission_df["label"] = majority_label
else:
    missing = 0
    bad_pred = 0

    for image_id in sample_csv["image_id"].tolist():
        if image_id not in available_test_images:
            missing += 1
            image_predictions.append({"image_id": image_id, "label": majority_label})
            continue

        img_path = os.path.join(test_image_dir, image_id)
        pred = predict_one_image_ensemble(img_path, loaded_models)
        if pred is None:
            bad_pred += 1
            pred = majority_label
        image_predictions.append({"image_id": image_id, "label": int(pred)})

    if missing:
        print(
            f"Warning: {missing} images from sample_submission were not found in test_image_dir."
        )
    if bad_pred:
        print(
            f"Warning: {bad_pred} images failed feature extraction; used majority fallback for those."
        )

    submission_df = pd.DataFrame(image_predictions)

submission_df = submission_df[["image_id", "label"]]
submission_df = sample_csv[["image_id"]].merge(submission_df, on="image_id", how="left")
submission_df["label"] = submission_df["label"].fillna(majority_label).astype(int)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_df.head())
print(
    "Rows:", len(submission_df), " Unique images:", submission_df["image_id"].nunique()
)



## === cell 5
submission_df
