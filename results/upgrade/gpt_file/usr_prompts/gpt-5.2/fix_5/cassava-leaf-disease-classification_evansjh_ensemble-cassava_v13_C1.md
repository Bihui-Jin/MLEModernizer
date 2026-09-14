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

0.8756421879721971

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.72048) has done: 'The timeout is dominated by per-image/per-model Python loops that repeatedly decode JPEGs and call `model.predict()` on a single image at a time, plus expensive PIL image loading/resizing for every model. I keep the exact ensemble voting and tie-break logic, but make inference batch-based: for each unique input size, build a single `tf.data` pipeline that decodes/resizes once and feeds batches through all models of that size, then aggregate predictions exactly as before. This removes thousands of tiny `predict()` calls and redundant image preprocessing while preserving identical evaluation semantics. I also ensure deterministic `tf.data` behavior and avoid extra pandas merges/work that don’t affect results.'

# 9. Code solution

## === cell 0
import os
from collections import Counter

import numpy as np
import pandas as pd

import tf_keras as keras

from PIL import Image

print("Keras (tf_keras):", keras.__version__)

SEED = 42
np.random.seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
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
model_path_7 = "/kaggle/input/googlenet_512/tensorflow2/default/1/GOOG1.h5"

test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_image_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
sample = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

assert os.path.exists(sample), f"Missing sample_submission.csv at {sample}"
assert os.path.isdir(test_image_dir), f"Missing test image dir at {test_image_dir}"
assert os.path.exists(train_csv_path), f"Missing train.csv at {train_csv_path}"
assert os.path.isdir(train_image_dir), f"Missing train image dir at {train_image_dir}"



## === cell 2
sample_csv = pd.read_csv(sample)
print(sample_csv.head())
print("sample_submission rows:", len(sample_csv))
print("sample_submission columns:", list(sample_csv.columns))

train_df = pd.read_csv(train_csv_path)
print(train_df.head())
print("train rows:", len(train_df))
print("train columns:", list(train_df.columns))
print("label distribution:\n", train_df["label"].value_counts().sort_index())



## === cell 3
models_info = [
    (model_path_3, (448, 448)),
    (model_path_7, (512, 512)),
    (model_path_1, (512, 512)),
    (model_path_2, (512, 512)),
    (model_path_4, (512, 512)),
    (model_path_5, (512, 512)),
    (model_path_6, (512, 512)),
]

existing_models_info = [(p, s) for (p, s) in models_info if os.path.exists(p)]
missing_models = [p for (p, _) in models_info if not os.path.exists(p)]

print("Missing model files (will be skipped):")
for p in missing_models:
    print(" -", p)

models = []
for path, input_size in existing_models_info:
    m = keras.models.load_model(path, compile=False)
    models.append((m, input_size))
    print(f"Loaded model: {path} with input size {input_size}")

print("Number of external models loaded:", len(models))



## === cell 4
NUM_CLASSES = 5

if len(models) == 0:
    raise RuntimeError(
        "No external models were found/loaded. This script avoids importing TensorFlow "
        "due to environment protobuf issues, so it cannot train a fallback model here."
    )
else:
    print("Using external models only; fallback training skipped.")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3451466648.py in <cell line: 0>()
      6 
      7 if len(models) == 0:
----> 8     raise RuntimeError(
      9         "No external models were found/loaded. This script avoids importing TensorFlow "
     10         "due to environment protobuf issues, so it cannot train a fallback model here."

RuntimeError: No external models were found/loaded. This script avoids importing TensorFlow due to environment protobuf issues, so it cannot train a fallback model here.

## === cell 5
class_labels = {
    0: "Cassava Bacterial Blight (CBB)",
    1: "Cassava Brown Streak Disease (CBSD)",
    2: "Cassava Green Mottle (CGM)",
    3: "Cassava Mosaic Disease (CMD)",
    4: "Healthy",
}

image_ids = sample_csv["image_id"].tolist()

missing_imgs = [
    img for img in image_ids if not os.path.exists(os.path.join(test_image_dir, img))
]
print("Missing test images referenced by sample_submission:", len(missing_imgs))
if len(missing_imgs) > 0:
    print("Example missing images:", missing_imgs[:5])



## === cell 6


def load_image_batch(image_id_batch, image_dir, input_size):
    batch = np.empty(
        (len(image_id_batch), input_size[0], input_size[1], 3), dtype=np.float32
    )
    for i, img_id in enumerate(image_id_batch):
        p = os.path.join(image_dir, img_id)
        with Image.open(p) as im:
            im = im.convert("RGB")
            im = im.resize(
                (input_size[1], input_size[0]), resample=Image.BILINEAR
            )  # PIL uses (W,H)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        batch[i] = arr
    return batch


id_to_pos = {img_id: i for i, img_id in enumerate(image_ids)}
final_labels = np.full(len(image_ids), 4, dtype=np.int32)

existing_ids = [
    img_id
    for img_id in image_ids
    if os.path.exists(os.path.join(test_image_dir, img_id))
]
existing_positions = np.array([id_to_pos[i] for i in existing_ids], dtype=np.int32)

models_by_size = {}
for m, sz in models:
    models_by_size.setdefault(tuple(sz), []).append(m)

per_image_model_pred = np.empty((len(existing_ids), len(models)), dtype=np.int32)
per_image_model_conf = np.empty((len(existing_ids), len(models)), dtype=np.float32)

BATCH_SIZE = 32

for input_size, model_list in models_by_size.items():
    size_model_indices = [
        i for i, (_, sz) in enumerate(models) if tuple(sz) == input_size
    ]

    for start in range(0, len(existing_ids), BATCH_SIZE):
        end = min(start + BATCH_SIZE, len(existing_ids))
        batch_ids = existing_ids[start:end]
        x = load_image_batch(batch_ids, test_image_dir, input_size=input_size)

        for global_mi in size_model_indices:
            model = models[global_mi][0]
            preds = model.predict(x, verbose=0)  # (B, NUM_CLASSES)

            cls = np.argmax(preds, axis=1).astype(np.int32)
            conf = preds[np.arange(preds.shape[0]), cls].astype(np.float32)

            per_image_model_pred[start:end, global_mi] = cls
            per_image_model_conf[start:end, global_mi] = conf

for row_i, pos in enumerate(existing_positions):
    model_predictions = per_image_model_pred[row_i].tolist()

    confidence_scores = {}
    for mi, cls in enumerate(model_predictions):
        confidence_scores.setdefault(int(cls), []).append(
            float(per_image_model_conf[row_i, mi])
        )

    class_votes = Counter(model_predictions)
    most_common = class_votes.most_common()
    final_predicted_class = most_common[0][0]

    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        tied_classes = [cls for cls, count in most_common if count == most_common[0][1]]
        final_predicted_class = max(
            tied_classes,
            key=lambda cls: sum(confidence_scores[cls]) / len(confidence_scores[cls]),
        )

    final_labels[pos] = int(final_predicted_class)

submission_df = pd.DataFrame({"image_id": image_ids, "label": final_labels.astype(int)})



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/827757679.py in <cell line: 0>()
     76     class_votes = Counter(model_predictions)
     77     most_common = class_votes.most_common()
---> 78     final_predicted_class = most_common[0][0]
     79 
     80     if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:

IndexError: list index out of range

## === cell 7
submission_df = submission_df.merge(
    sample_csv[["image_id"]], on="image_id", how="right"
)
submission_df["label"] = submission_df["label"].fillna(4).astype(int)
submission_df = submission_df[["image_id", "label"]]

print(submission_df.head())
print("submission rows:", len(submission_df), "expected:", len(sample_csv))
assert len(submission_df) == len(sample_csv), "Submission row count mismatch."
assert list(submission_df.columns) == [
    "image_id",
    "label",
], "Submission columns mismatch."

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print("Wrote:", out_path)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/312421194.py in <cell line: 0>()
      1 # Keep the exact sample order/rows; right-join + fill fallback for absolute safety.
----> 2 submission_df = submission_df.merge(
      3     sample_csv[["image_id"]], on="image_id", how="right"
      4 )
      5 submission_df["label"] = submission_df["label"].fillna(4).astype(int)

NameError: name 'submission_df' is not defined

## === cell 8
submission_df

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2788716369.py in <cell line: 0>()
----> 1 submission_df

NameError: name 'submission_df' is not defined
