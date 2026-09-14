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

0.8230583257781807

# 6. Current score

0.11024

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.11024) has done: 'I fix the immediate runtime blockers by (1) making the code robust to the absence of the external `.pth` checkpoint dataset and (2) ensuring `model` is always defined so inference can run and a valid `submission.csv` is written. Because your current score is “Not yielded”, the priority is to produce a valid end-to-end pipeline; if the checkpoint is missing, I fall back to a standard torchvision ResNet-50 backbone with a 5-class head so predictions can still be generated. I also fix the checkpoint loading logic to correctly handle either a full saved `nn.Module` or a `state_dict` (with safe, minimal assumptions). These changes keep the inference/preprocessing loop semantics the same and only add a necessary fallback to make the notebook executable in this environment.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms, models



## === cell 1
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

MODEL_PATH = (
    "/kaggle/input/resnet50_70_512x512/pytorch/default/1/Resnet50_70_512x512.pth"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_model_preprocess = transforms.Compose(
    [
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)




## === cell 2
def _find_checkpoint_fallback():
    candidates = glob.glob("/kaggle/input/**/*.pth", recursive=True)
    if not candidates:
        return None

    def score(p):
        s = p.lower()
        sc = 0
        for token, w in [
            ("cassava", 5),
            ("resnet50", 5),
            ("resnet", 2),
            ("512", 2),
            ("leaf", 1),
            ("disease", 1),
        ]:
            if token in s:
                sc += w
        sc -= 0.0001 * len(p)
        return sc

    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0]


def _build_fallback_model(num_classes=5):
    """
    Bug fix: ensure we can still run end-to-end if no external checkpoint exists.
    Core logic preserved: still a ResNet-style classifier producing 5 logits.
    """
    m = models.resnet50(weights=None)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m


if not os.path.exists(MODEL_PATH):
    alt = _find_checkpoint_fallback()
    if alt is not None:
        print(f"MODEL_PATH not found. Using fallback checkpoint: {alt}")
        MODEL_PATH = alt
    else:
        print(
            f"Warning: MODEL_PATH not found: {MODEL_PATH}\n"
            f"Also did not find any .pth under /kaggle/input.\n"
            f"Proceeding with an untrained fallback ResNet50 model so a valid submission.csv is produced."
        )
        MODEL_PATH = None

model = None

if MODEL_PATH is not None:
    ckpt = torch.load(MODEL_PATH, map_location=device)

    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
    elif isinstance(ckpt, dict):
        state_dict = None
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif all(isinstance(k, str) for k in ckpt.keys()):
            state_dict = ckpt

        if state_dict is not None:
            model = _build_fallback_model(num_classes=5)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(
                    f"Warning: missing keys when loading state_dict (showing up to 10): {missing[:10]}"
                )
            if unexpected:
                print(
                    f"Warning: unexpected keys when loading state_dict (showing up to 10): {unexpected[:10]}"
                )
        else:
            print(
                "Warning: Unrecognized checkpoint dict format. Proceeding with an untrained fallback model."
            )
            model = _build_fallback_model(num_classes=5)
    else:
        print(
            "Warning: Unrecognized checkpoint format. Proceeding with an untrained fallback model."
        )
        model = _build_fallback_model(num_classes=5)

if model is None:
    model = _build_fallback_model(num_classes=5)

model.to(device)
model.eval()



## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
image_ids = sample_sub["image_id"].astype(str).tolist()

prediction = []
missing = 0
failed = 0

with torch.no_grad():
    for image_id in image_ids:
        img_path = os.path.join(TEST_IMG_DIR, image_id)

        if not os.path.exists(img_path):
            missing += 1
            prediction.append(0)
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            failed += 1
            prediction.append(0)
            continue

        x = main_model_preprocess(img).unsqueeze(0).to(device)

        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())
        prediction.append(pred)

if missing:
    print(f"Warning: {missing} test images were missing. Filled with class 0.")
if failed:
    print(f"Warning: {failed} test images failed to load. Filled with class 0.")

assert len(prediction) == len(
    image_ids
), "Internal error: prediction and image_ids length mismatch."

submission = pd.DataFrame({"image_id": image_ids, "label": prediction})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print(f"Wrote submission.csv with {len(submission)} rows.")
