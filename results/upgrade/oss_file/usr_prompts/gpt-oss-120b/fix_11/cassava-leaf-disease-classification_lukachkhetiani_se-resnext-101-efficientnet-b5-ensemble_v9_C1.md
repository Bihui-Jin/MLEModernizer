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

0.8038682381384104

# 6. Current score

0.11996

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.16667) has done: 'The changes add mixed‑precision (AMP) training to dramatically cut the fine‑tuning time when checkpoints are missing, and increase data‑loader workers for faster image loading. The inference pipeline and all model architectures remain unchanged, preserving exact predictions while keeping total runtime under the 600 s limit.'
- What this solution (achieved 0.10015) has done: 'The script now avoids the costly 15‑epoch training when the pretrained checkpoints are unavailable: it directly loads a pretrained EfficientNet‑B5 model from timm and reuses it for both ensemble members, guaranteeing the same inference logic while cutting runtime dramatically. Mean/std tensors for normalization are created once globally and reused in `process`, removing per‑image overhead. All other logic and paths remain unchanged, preserving exact predictions given the same model weights.'
- What this solution (achieved 0.10501) has done: 'The changes add a lightweight test‑time augmentation (horizontal flip) and average the model logits from the original and flipped images. This keeps the same models and preprocessing while giving a modest boost in accuracy, moving the current score closer to the target without altering the core training logic.'
- What this solution (achieved 0.11996) has done: 'I adjust the preprocessing to use the native input size for EfficientNet‑B5 (456 × 456) and add a few lightweight test‑time augmentations (horizontal flip, vertical flip, 90‑degree rotation). All augmentations are applied during inference, their logits are averaged, and the final class is taken from the averaged soft‑max probabilities. This keeps the original models and training untouched while providing a stronger, more robust prediction pipeline that should raise the accuracy toward the target score. The script now also writes the submission in the required CSV format.'

# 9. Code solution

## === cell 0
import os, sys, random, glob, tqdm, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import cv2, pandas as pd
from torch.utils.data import Dataset, DataLoader

try:
    from efficientnet_pytorch import EfficientNet
except ModuleNotFoundError:
    EfficientNet = None

import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


def try_load(path):
    if os.path.exists(path):
        try:
            return torch.load(path, map_location=device)
        except Exception as e:
            print(f"Could not load {path}: {e}")
    else:
        print(f"Checkpoint {path} not found.")
    return None


base_path = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(base_path, "train.csv")
train_img_dir = os.path.join(base_path, "train_images")
test_img_dir = os.path.join(base_path, "test_images")

eff_path = "/kaggle/input/ensemble-2/eff_model_last.pth"
hr_path = "/kaggle/input/ensemble-2/seresnext_model_last.pth"

eff_state = try_load(eff_path)
hr_state = try_load(hr_path)

if eff_state is not None and EfficientNet is not None:
    efficient = EfficientNet.from_name("efficientnet-b5", num_classes=5)
    efficient.load_state_dict(eff_state)
    efficient = efficient.to(device).eval()
else:
    efficient = timm.create_model("efficientnet_b5", pretrained=True, num_classes=5)
    efficient = efficient.to(device).eval()

if hr_state is not None:
    hrnet = timm.create_model("seresnext101_32x4d", pretrained=False, num_classes=5)
    hrnet.load_state_dict(hr_state)
    hrnet = hrnet.to(device).eval()
else:
    hrnet = efficient

print("Model(s) loaded and ready for inference.")




## === cell 1
files = glob.glob(os.path.join(test_img_dir, "**", "*.jpg"), recursive=True)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

TARGET_SIZE = 456


def process(image):
    """Resize, apply CLAHE, convert to normalized torch tensor (CPU)."""
    img = cv2.resize(image, (TARGET_SIZE, TARGET_SIZE))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
    tensor = (tensor - norm_mean) / norm_std
    return tensor


def infer_batch(batch_tensor):
    """
    Perform inference with several test‑time augmentations.
    Returns the predicted class indices for the batch.
    """
    augmentations = [
        lambda x: x,  # original
        lambda x: torch.flip(x, dims=[3]),  # horizontal flip
        lambda x: torch.flip(x, dims=[2]),  # vertical flip
        lambda x: torch.rot90(x, k=1, dims=[2, 3]),  # 90° rotation
    ]

    with torch.no_grad():
        logits_sum = None
        for aug in augmentations:
            aug_batch = aug(batch_tensor)
            hr_out = hrnet(aug_batch)
            eff_out = efficient(aug_batch)
            logits = (hr_out + eff_out) / 2
            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum += logits
        avg_logits = logits_sum / len(augmentations)
        probs = F.softmax(avg_logits, dim=1)
        preds = torch.argmax(probs, dim=1).cpu().numpy()
    return preds.tolist()


batch_size = 32
names, labels = [], []

batch_tensors = []
batch_names = []

for file in tqdm.tqdm(files, desc="Predicting"):
    img = cv2.imread(file)
    if img is None:
        continue
    tensor = process(img)  # CPU tensor with normalization
    batch_tensors.append(tensor)
    batch_names.append(os.path.basename(file))

    if len(batch_tensors) == batch_size:
        batch = torch.stack(batch_tensors).to(device, non_blocking=True)
        preds = infer_batch(batch)
        names.extend(batch_names)
        labels.extend(preds)
        batch_tensors, batch_names = [], []

if batch_tensors:
    batch = torch.stack(batch_tensors).to(device, non_blocking=True)
    preds = infer_batch(batch)
    names.extend(batch_names)
    labels.extend(preds)

print(f"Predicted {len(names)} images.")




## === cell 2
submission = pd.DataFrame({"image_id": names, "label": labels})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file saved as {submission_path} with {len(submission)} rows.")
