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

0.8893925657298277

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.61099) has done: 'I fix the TensorFlow import crash by avoiding the incompatible TF/Keras stack in this environment and switching to `keras` (Keras 3) inference-only loading. I also fix the missing external model paths by dynamically discovering any `.h5/.keras` model files that actually exist under `/kaggle/input`, then loading what’s available instead of hard-failing. To ensure you always get a valid `submission.csv`, I add a safe fallback that writes a majority-class submission derived from `train.csv` if no models can be loaded. Finally, I keep your core ensemble voting logic intact, but make the test image iteration deterministic and align the output rows to `sample_submission.csv` ordering.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings
from collections import Counter

import numpy as np
import pandas as pd

try:
    from keras.saving import load_model  # Keras 3
    from keras.preprocessing.image import load_img, img_to_array
except Exception as e:
    raise RuntimeError(
        "Failed to import Keras. This notebook expects Keras (Keras 3) to be available."
    ) from e

warnings.filterwarnings("ignore")
np.random.seed(42)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

sample_csv = pd.read_csv(sample_path)
train_df = pd.read_csv(train_csv_path)

print("sample_csv shape:", sample_csv.shape)
print("train_df shape:", train_df.shape)
print("test_image_dir exists:", os.path.isdir(test_image_dir))



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
def discover_models_under_kaggle_input(max_models=6):
    """
    Fix: The provided model paths may not exist. Discover candidate model files
    under /kaggle/input and return a list of paths.
    Priority: .h5 and .keras files.
    """
    candidates = []
    for ext in ("*.h5", "*.keras"):
        candidates.extend(glob.glob(f"/kaggle/input/**/{ext}", recursive=True))
    candidates = sorted(set(candidates))
    if max_models is not None:
        candidates = candidates[:max_models]
    return candidates


def safe_load_models(models_info):
    """
    Fix: Do not hard-fail if some model files are missing or incompatible.
    Load what exists; keep (model, input_size) pairs.
    """
    loaded = []
    for path, input_size in models_info:
        if not os.path.exists(path):
            continue
        try:
            m = load_model(path, compile=False)
            loaded.append((m, input_size, path))
        except Exception as e:
            print(
                f"Skipping model (failed to load): {path}\n  reason: {type(e).__name__}: {e}"
            )
    return loaded


loaded_models = safe_load_models(models_info)

if len(loaded_models) == 0:
    discovered = discover_models_under_kaggle_input(max_models=10)
    print("Discovered model files under /kaggle/input (first 10):")
    for p in discovered[:10]:
        print(" ", p)

    fallback_info = []
    for i, p in enumerate(discovered[: len(models_info)]):
        fallback_info.append((p, models_info[i][1]))
    loaded_models = safe_load_models(fallback_info)

print(f"Loaded {len(loaded_models)} models.")
for _, _, p in loaded_models:
    print(" model:", p)



## === cell 4
class_labels = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}


def predict_one_image_ensemble(image_path, models):
    """
    Core logic preserved: each model predicts argmax; final class by majority vote;
    tie-break by highest average confidence among tied classes.
    """
    model_predictions = []
    confidence_scores = {}

    for model, input_size, _path in models:
        img = load_img(image_path, target_size=input_size)
        img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)

        preds = model.predict(img_array, verbose=0)
        predicted_class = int(np.argmax(preds, axis=1)[0])
        confidence_score = float(preds[0][predicted_class])

        model_predictions.append(predicted_class)
        confidence_scores.setdefault(predicted_class, []).append(confidence_score)

    class_votes = Counter(model_predictions)
    most_common = class_votes.most_common()
    final_predicted_class = most_common[0][0]

    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        tied_classes = [cls for cls, count in most_common if count == most_common[0][1]]
        final_predicted_class = max(
            tied_classes,
            key=lambda cls: sum(confidence_scores[cls]) / len(confidence_scores[cls]),
        )

    return final_predicted_class




## === cell 5

available_test_images = (
    set(os.listdir(test_image_dir)) if os.path.isdir(test_image_dir) else set()
)

image_predictions = []

if len(loaded_models) == 0:
    majority_label = int(train_df["label"].value_counts().idxmax())
    print(
        "No models loaded; writing fallback submission with majority label =",
        majority_label,
    )

    submission_df = sample_csv.copy()
    submission_df["label"] = majority_label

else:
    missing = 0
    for image_id in sample_csv["image_id"].tolist():
        if image_id not in available_test_images:
            missing += 1
            image_predictions.append(
                {
                    "image_id": image_id,
                    "label": int(train_df["label"].value_counts().idxmax()),
                }
            )
            continue

        img_path = os.path.join(test_image_dir, image_id)
        pred = predict_one_image_ensemble(img_path, loaded_models)
        image_predictions.append({"image_id": image_id, "label": int(pred)})

    if missing:
        print(
            f"Warning: {missing} images from sample_submission were not found in test_image_dir."
        )

    submission_df = pd.DataFrame(image_predictions)

submission_df = submission_df[["image_id", "label"]]
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_df.head())
print(
    "Rows:", len(submission_df), " Unique images:", submission_df["image_id"].nunique()
)



## === cell 6
submission_df
