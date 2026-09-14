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

3.9

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

0.7632215170746449

# 6. Current score

0.12593

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12556) has done: 'I remove/avoid the imports that trigger the protobuf `MessageFactory.GetPrototype` crash (not needed for inference) and also remove the dependency on `efficientnet`/`tensorflow_hub` since the provided environment doesn’t have those offline wheels. Then I robustly load the pre-trained `.h5` model if it exists, and fall back to a minimal EfficientNetB3 classifier only if the weight file is unavailable (so the notebook still runs end-to-end and produces a valid CSV). Finally, I fix the test path to the actual provided dataset location, ensure the generator produces predictions in the exact sample submission order, and always write `submission.csv` with `image_id,label`.'
- What this solution (achieved 0.12593) has done: 'I fix dataset path discovery so it correctly finds `test_images` in this environment (your current `DATA_ROOT` points one level too deep, so all test image paths are wrong). I make `DATA_ROOT` selection validate that both `sample_submission.csv` and `test_images/` exist before accepting a candidate, and add a small fallback search to locate `test_images` if the chosen root is missing it. These changes are score-neutral but unblock end-to-end execution and ensure predictions align with the sample submission order. Everything else (model, preprocessing, batching, argmax labels, and CSV writing) stays the same so you get a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

SEED = 42
DEBUG = False

np.random.seed(SEED)

DATA_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]


def _is_valid_root(root: str) -> bool:
    return (
        os.path.isdir(root)
        and os.path.isdir(os.path.join(root, "test_images"))
        and os.path.isfile(os.path.join(root, "sample_submission.csv"))
    )


DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if _is_valid_root(p):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    sample_candidates = [
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
    found_sample = next((p for p in sample_candidates if os.path.isfile(p)), None)
    if found_sample is not None:
        base = os.path.dirname(found_sample)
        possible_roots = [base] + [
            os.path.join(base, d)
            for d in os.listdir(base)
            if os.path.isdir(os.path.join(base, d))
        ]
        for r in possible_roots:
            if _is_valid_root(r):
                DATA_ROOT = r
                break

if DATA_ROOT is None:
    search_prefixes = ["/kaggle/input", "/kaggle/data"]
    for pref in search_prefixes:
        if not os.path.isdir(pref):
            continue
        for root, dirs, files in os.walk(pref):
            if "sample_submission.csv" in files and "test_images" in dirs:
                DATA_ROOT = root
                break
        if DATA_ROOT is not None:
            break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate a valid cassava-leaf-disease-classification dataset root containing "
        "'test_images/' and 'sample_submission.csv' under /kaggle/input or /kaggle/data."
    )

TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

if not os.path.isdir(TEST_IMG_DIR):
    raise FileNotFoundError(f"Expected test_images directory at: {TEST_IMG_DIR}")
if not os.path.isfile(SAMPLE_SUB_PATH):
    raise FileNotFoundError(f"Expected sample_submission.csv at: {SAMPLE_SUB_PATH}")

print("DATA_ROOT:", DATA_ROOT)
print("TEST_IMG_DIR:", TEST_IMG_DIR)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)



## === cell 1
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Torch device:", device)

my_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
in_features = my_model.classifier[1].in_features
my_model.classifier[1] = nn.Linear(in_features, 5)
my_model = my_model.to(device)
my_model.eval()

preprocess = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

_ = my_model(torch.zeros(1, 3, 224, 224, device=device))
print(
    "Model ready. Output dim:",
    my_model(torch.zeros(1, 3, 224, 224, device=device)).shape[-1],
)



## === cell 2
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if "image_id" not in sample_sub.columns:
    raise ValueError("sample_submission.csv does not contain 'image_id' column")

df_test = pd.DataFrame({"image_id": sample_sub["image_id"].astype(str).values})
df_test["path"] = df_test["image_id"].apply(lambda x: os.path.join(TEST_IMG_DIR, x))

missing = df_test.loc[~df_test["path"].apply(os.path.isfile)]
if len(missing) > 0:
    print("First few missing paths:\n", missing.head())
    raise FileNotFoundError(
        f"{len(missing)} test images referenced in sample_submission are missing on disk."
    )

print("Test rows:", len(df_test))
print(df_test.head())



## === cell 3
batch_size = 64
pred_labels = np.empty(len(df_test), dtype=np.int64)

with torch.inference_mode():
    for start in range(0, len(df_test), batch_size):
        end = min(start + batch_size, len(df_test))
        batch_paths = df_test["path"].iloc[start:end].tolist()

        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(preprocess(img))
        x = torch.stack(imgs, dim=0).to(device)

        logits = my_model(x)
        batch_pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
        pred_labels[start:end] = batch_pred

final_csv = df_test[["image_id"]].copy()
final_csv["label"] = pred_labels

if len(final_csv) != len(sample_sub):
    raise RuntimeError("Submission row count mismatch vs sample_submission.")
if final_csv["label"].isna().any():
    raise RuntimeError("NaNs found in predicted labels.")
if not np.issubdtype(final_csv["label"].dtype, np.integer):
    final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())



## === cell 4
final_csv.head()
