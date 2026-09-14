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

0.18722

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.09716) has done: 'I replace the unavailable efficientnet‑pytorch import with the built‑in torchvision implementation, add a safe fallback for the checkpoint, and ensure the model, preprocessing, and CSV output are correctly defined so the script runs end‑to‑end and produces a valid submission.csv with the proper length.'
- What this solution (achieved 0.13042) has done: 'I make two small but effective adjustments: (1) load the fine‑tuned EfficientNet‑B5 checkpoint with `strict=False` so that any mismatched keys (caused by using torchvision’s implementation) are ignored and the pretrained weights are still applied; (2) add the proper ImageNet normalization after scaling the image to [0, 1] because EfficientNet expects normalized inputs. These changes keep the original architecture and inference loop unchanged while considerably improving the model’s predictions, moving the validation accuracy much closer to the target score.'
- What this solution (achieved 0.2358) has done: 'I add a simple test‑time augmentation by also predicting on the horizontally‑flipped image and averaging the logits before taking the arg‑max. This keeps the original model and preprocessing unchanged while giving the model a second view of each leaf, which should raise the validation accuracy and move the score closer to the target.'
- What this solution (achieved 0.10725) has done: 'I make the checkpoint loading more robust by searching recursively for any `.pth` file in the project (including the typical `input/...` directory) and loading the first one found. This tiny change keeps the model architecture unchanged but greatly raises the chance of actually loading the fine‑tuned weights, which should move the validation accuracy much closer to the target score. No other logic is altered.'
- What this solution (achieved 0.20441) has done: 'I fix the color ordering in the image preprocessing: after CLAHE processing the image is still in OpenCV’s BGR format, but EfficientNet was trained on RGB images with the given ImageNet mean/std. Converting the image to RGB before creating the tensor restores the correct channel order, which should raise validation accuracy toward the target without altering the model architecture or training logic.'
- What this solution (achieved 0.0852) has done: 'I make the checkpoint loading more reliable so the fine‑tuned EfficientNet‑B5 weights are actually used. The script now picks the largest *.pth file (most likely the trained model), extracts a nested state_dict if present, and loads it with strict=False. These minimal changes keep the original architecture and preprocessing untouched while ensuring the model benefits from the trained weights, which should raise the validation accuracy toward the target score.'
- What this solution (achieved 0.463) has done: 'I broaden the checkpoint search to include “.pt” files and make the loader strip a possible “module.” prefix from saved keys so the fine‑tuned EfficientNet‑B5 weights are actually applied. This minimal change keeps the model architecture and inference unchanged while ensuring the pretrained fine‑tuned weights are used, which should raise the validation accuracy toward the target score.'
- What this solution (achieved 0.11472) has done: 'I make two small but important fixes: (1) enhance the checkpoint loader to also handle a saved full model object by extracting its state_dict (and keep the existing logic for dicts), ensuring the fine‑tuned EfficientNet‑B5 weights are actually applied; (2) add a lightweight test‑time augmentation – a vertical flip – and average its logits together with the original and horizontal‑flip predictions, which often gives a modest boost in accuracy without altering the core model or training code. These changes keep the original architecture intact while moving the validation accuracy much closer to the target score.'
- What this solution (achieved 0.18722) has done: 'I tighten the checkpoint loading by cleaning the state‑dict before the first load (so more fine‑tuned weights are applied) and add a simple 90‑degree rotation test‑time augmentation that is averaged with the existing flips. These minimal changes keep the model architecture and main pipeline unchanged while likely raising the validation accuracy toward the target score.'

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

checkpoint_candidates = glob.glob(
    os.path.join("**", "*.pth"), recursive=True
) + glob.glob(os.path.join("**", "*.pt"), recursive=True)
checkpoint_path = None
if checkpoint_candidates:

    def score_path(p):
        score = 0
        if "input" in p:
            score += 2
        if "working" in p:
            score += 1
        score += os.path.getsize(p) / 1e6  # size in MB as float
        return score

    checkpoint_path = max(checkpoint_candidates, key=score_path)
    print(f"Selected checkpoint: {checkpoint_path}")
else:
    print("No .pth/.pt checkpoint files found; using ImageNet weights only.")


def clean_state_dict(state_dict):
    """Remove possible 'module.' prefix from keys."""
    cleaned = {}
    for k, v in state_dict.items():
        cleaned_key = k.replace("module.", "")
        cleaned[cleaned_key] = v
    return cleaned


if checkpoint_path and os.path.isfile(checkpoint_path):
    try:
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, torch.nn.Module):
            state = state.state_dict()
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]
        if isinstance(state, dict):
            state = clean_state_dict(state)
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
    print("No valid checkpoint path; using ImageNet pretrained weights.")

model.to(device)
model.eval()



## === cell 2
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)


def process(image):
    """Resize, apply CLAHE, convert to RGB tensor, normalize and move to device."""
    img = cv2.resize(image, (456, 456))  # EfficientNet‑B5 default input size
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    tensor = (
        torch.tensor(img_rgb.transpose(2, 0, 1), dtype=torch.float32, device=device)
        / 255.0
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
    img_hflip = cv2.flip(img, 1)
    tensor_hflip = process(img_hflip)
    img_vflip = cv2.flip(img, 0)
    tensor_vflip = process(img_vflip)
    img_rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    tensor_rot = process(img_rot)

    with torch.no_grad():
        out_orig = model(tensor_orig)
        out_hflip = model(tensor_hflip)
        out_vflip = model(tensor_vflip)
        out_rot = model(tensor_rot)

        avg_logits = (out_orig + out_hflip + out_vflip + out_rot) / 4.0
        pred_label = int(torch.argmax(avg_logits, dim=1).item())

    names.append(os.path.basename(file))
    labels.append(pred_label)



## === cell 4
submission = pd.DataFrame({"image_id": names, "label": labels})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path} with {len(submission)} rows.")
