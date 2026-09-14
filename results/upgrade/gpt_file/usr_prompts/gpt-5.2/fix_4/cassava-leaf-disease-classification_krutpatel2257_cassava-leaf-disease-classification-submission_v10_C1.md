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

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

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

0.8839528558476881

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'Your code isn’t yielding a score mainly because it likely errors at runtime (missing `saved-model/model.pth`, wrong cell numbering, and `torch.load` device mismatch) and it also uses random augmentations at inference which makes predictions unstable and usually worse. I make the smallest changes needed to (1) reliably load the model if present (with `map_location`), (2) ensure inference runs end-to-end and writes a valid `submission.csv`, and (3) remove randomness from test-time transforms (use a deterministic resize/center-crop + normalization) to push accuracy upward toward your target. I also add a safe fallback that creates a valid submission (using the sample submission labels) if the model file is missing, so you always get a scorable CSV. Core model architecture and basic inference semantics (argmax over logits) remain unchanged.'
- What this solution (achieved 0.11584) has done: 'I fix the Albumentations runtime error by updating the `RandomResizedCrop` call to the v2 API so the notebook runs end-to-end. Since your current score is far below target, I also make inference deterministic by replacing the random heavy augmentation pipeline (which was both broken and harmful at test time) with a simple resize/center-crop + normalization pipeline consistent with ImageNet-pretrained backbones, while keeping the same model architecture and argmax prediction logic. I keep the robust model-loading logic and ensure a valid `submission.csv` is always written with the required columns. These changes are minimal, directly address the crash, and should substantially improve score versus the current unstable/incorrect setup.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is far below the target (0.88395), and the biggest likely cause is that the model is not actually being loaded in this environment (`../input/saved-model/model.pth` doesn’t exist here), so you’re effectively submitting the sample labels (near-random for this test set). The minimal score-improving change is to load weights from a file path that actually exists in this dataset layout (if available), while keeping the exact same architecture and argmax inference. I add a small “search common locations” block to find `model.pth` inside `../input/` and load it with `map_location`, and I keep inference deterministic as it already is (no random test-time aug). The script still always write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

import albumentations  # kept to preserve original imports/environment, though unused in final inference
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms



## === cell 1
model_path = "../input/saved-model/model.pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




## === cell 2
def find_model_path(
    preferred_path: str, filename: str = "model.pth", search_root: str = "../input"
):
    if os.path.exists(preferred_path):
        return preferred_path
    candidates = []
    for root, _, files in os.walk(search_root):
        if filename in files:
            candidates.append(os.path.join(root, filename))
    candidates = sorted(candidates, key=lambda p: (len(p), p))
    return candidates[0] if candidates else preferred_path


resolved_model_path = find_model_path(model_path)
if resolved_model_path != model_path:
    print(f"INFO: Using discovered model path: {resolved_model_path}")
else:
    print(f"INFO: Using model path: {resolved_model_path}")



## === cell 3
model = models.resnext50_32x4d(pretrained=False)
model.fc = nn.Linear(2048, 5)
model.to(device)

model_loaded = False
if os.path.exists(resolved_model_path):
    try:
        state = torch.load(resolved_model_path, map_location=device)

        if (
            isinstance(state, dict)
            and "state_dict" in state
            and isinstance(state["state_dict"], dict)
        ):
            state_dict = state["state_dict"]
        else:
            state_dict = state

        if isinstance(state_dict, dict) and any(
            k.startswith("module.") for k in state_dict.keys()
        ):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model_loaded = True
        print("INFO: Model loaded successfully.")
    except Exception as e:
        print(f"WARNING: Failed to load model from {resolved_model_path}: {e}")
else:
    print(f"WARNING: Model file not found at {resolved_model_path}")



## === cell 4
sample_sub = pd.read_csv(sample_sub_path)

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 5
predictions = []

if not model_loaded:
    sub_df = sample_sub.copy()
    sub_df["label"] = sub_df["label"].astype(int)
    sub_df[["image_id", "label"]].to_csv("submission.csv", index=False)
    print("Saved fallback submission.csv (model not loaded).")
    print(sub_df.head())
else:
    with torch.inference_mode():
        for _, sample_row in sample_sub.iterrows():
            img_path = os.path.join(test_images_path, sample_row.image_id)

            image = Image.open(img_path).convert("RGB")
            image = transform(image).to(device)

            outputs = model(image.unsqueeze(0))
            _, pred_label = torch.max(outputs, 1)

            predictions.append([sample_row.image_id, int(pred_label.item())])

    sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])

    sub_df = sample_sub[["image_id"]].merge(sub_df, on="image_id", how="left")
    sub_df["label"] = sub_df["label"].fillna(0).astype(int)

    sub_df.to_csv("submission.csv", index=False)
    print("Saved submission.csv")
    print(sub_df.head())
