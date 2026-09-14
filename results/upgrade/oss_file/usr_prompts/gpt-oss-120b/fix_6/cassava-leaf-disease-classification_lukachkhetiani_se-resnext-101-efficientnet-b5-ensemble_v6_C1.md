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

0.8656693865216077

# 6. Current score

0.10725

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09716) has done: 'I replace the unavailable efficientnet‑pytorch import with the built‑in torchvision implementation, add a safe fallback for the checkpoint, and ensure the model, preprocessing, and CSV output are correctly defined so the script runs end‑to‑end and produces a valid submission.csv with the proper length.'
- What this solution (achieved 0.13042) has done: 'I make two small but effective adjustments: (1) load the fine‑tuned EfficientNet‑B5 checkpoint with `strict=False` so that any mismatched keys (caused by using torchvision’s implementation) are ignored and the pretrained weights are still applied; (2) add the proper ImageNet normalization after scaling the image to [0, 1] because EfficientNet expects normalized inputs. These changes keep the original architecture and inference loop unchanged while considerably improving the model’s predictions, moving the validation accuracy much closer to the target score.'
- What this solution (achieved 0.2358) has done: 'I add a simple test‑time augmentation by also predicting on the horizontally‑flipped image and averaging the logits before taking the arg‑max. This keeps the original model and preprocessing unchanged while giving the model a second view of each leaf, which should raise the validation accuracy and move the score closer to the target.'
- What this solution (achieved 0.10725) has done: 'I make the checkpoint loading more robust by searching recursively for any `.pth` file in the project (including the typical `input/...` directory) and loading the first one found. This tiny change keeps the model architecture unchanged but greatly raises the chance of actually loading the fine‑tuned weights, which should move the validation accuracy much closer to the target score. No other logic is altered.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import cv2
import tqdm
import numpy as np
import pandas as pd
import torch
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
model = models.efficientnet_b5(pretrained=True)

if hasattr(model, "classifier"):
    if isinstance(model.classifier, torch.nn.Sequential):
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, 5)
    else:
        in_features = model.classifier.in_features
        model.classifier = torch.nn.Linear(in_features, 5)
else:
    in_features = model._fc.in_features
    model._fc = torch.nn.Linear(in_features, 5)

checkpoint_candidates = glob.glob(os.path.join("**", "*.pth"), recursive=True)
checkpoint_path = None
if checkpoint_candidates:
    for cand in checkpoint_candidates:
        if "input" in cand or "working" in cand:
            checkpoint_path = cand
            break
    if checkpoint_path is None:
        checkpoint_path = checkpoint_candidates[0]

if checkpoint_path and os.path.isfile(checkpoint_path):
    try:
        state = torch.load(checkpoint_path, map_location=device)
        load_result = model.load_state_dict(state, strict=False)
        print(f"Loaded fine‑tuned weights from {checkpoint_path}")
        if load_result.missing_keys:
            print(
                f"  Missing keys ({len(load_result.missing_keys)}): {load_result.missing_keys}"
            )
        if load_result.unexpected_keys:
            print(
                f"  Unexpected keys ({len(load_result.unexpected_keys)}): {load_result.unexpected_keys}"
            )
    except Exception as e:
        print(f"Warning: could not load checkpoint ({e}); using ImageNet weights.")
else:
    print("No checkpoint found; using ImageNet pretrained weights.")

model.to(device)
model.eval()



## === cell 2
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)


def process(image):
    """Resize, apply CLAHE, convert to torch tensor, normalize and move to device."""
    img = cv2.resize(image, (456, 456))  # EfficientNet‑B5 default input size
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    tensor = (
        torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32, device=device) / 255.0
    )
    tensor = (tensor - mean) / std
    tensor = tensor.unsqueeze(0)  # add batch dim
    return tensor




## === cell 3
test_path = "../input/cassava-leaf-disease-classification/test_images"
files = sorted(glob.glob(os.path.join(test_path, "*")))
names, labels = [], []

for file in tqdm.tqdm(files, desc="Predicting"):
    img = cv2.imread(file)
    if img is None:
        continue  # skip unreadable files

    tensor_orig = process(img)

    img_flipped = cv2.flip(img, 1)
    tensor_flip = process(img_flipped)

    with torch.no_grad():
        out_orig = model(tensor_orig)
        out_flip = model(tensor_flip)

        avg_logits = (out_orig + out_flip) / 2.0
        pred_label = int(torch.argmax(avg_logits, dim=1).item())

    names.append(os.path.basename(file))
    labels.append(pred_label)



## === cell 4
submission = pd.DataFrame({"image_id": names, "label": labels})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path} with {len(submission)} rows.")
