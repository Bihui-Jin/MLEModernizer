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

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I remove the notebook-style shell/cd/pip cells that don’t run in a plain Kaggle Python script environment and instead rely on the already-available Kaggle input dataset paths. Then I fix the `ModuleNotFoundError` by providing a small fallback EfficientNet-B5 implementation via `torchvision` (while still loading your `b5.pth` if it’s present), keeping the same single-model inference flow and argmax labeling. I also make the inference robust to missing CUDA and ensure test image file discovery matches the provided dataset structure. Finally, I always write a valid `submission.csv` with the required `image_id,label` columns.'
- What this solution (achieved 0.11584) has done: 'Your current score is near random because the checkpoint is likely not loading (or not matching) and the input normalization does not match EfficientNet’s expected preprocessing. To move the score toward the 0.865 target with minimal changes, I (1) make checkpoint loading robust to common key prefixes (`module.`, `model.`) and enforce strict loading when possible, and (2) switch preprocessing to standard ImageNet mean/std normalization while keeping your same CLAHE + resize + single-image inference + argmax flow. I also ensure we always output predictions for every image in `sample_submission.csv` order (no accidental drops from unreadable images). These are small, directly score-relevant fixes and should substantially improve accuracy if `b5.pth` is a real trained cassava checkpoint.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import tqdm

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
BASE = "/kaggle/input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(BASE, "test_images")

if not os.path.isdir(TEST_DIR):
    alt = "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images"
    if os.path.isdir(alt):
        TEST_DIR = alt

CKPT_CANDIDATES = [
    "b5.pth",
    "/kaggle/working/b5.pth",
    "/kaggle/input/b5v3checkpoint/b5.pth",
]
CKPT_PATH = next((p for p in CKPT_CANDIDATES if os.path.isfile(p)), None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 2
def build_model(num_classes: int = 5) -> nn.Module:
    try:
        m = models.efficientnet_b5(weights=None)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
        return m
    except Exception as e:
        raise RuntimeError(
            "torchvision.models.efficientnet_b5 is not available in this environment."
        ) from e


def _clean_state_dict_keys(state: dict) -> dict:
    if not isinstance(state, dict):
        return state
    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


model = build_model(num_classes=5)

if CKPT_PATH is not None:
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    if (
        isinstance(ckpt, dict)
        and "state_dict" in ckpt
        and isinstance(ckpt["state_dict"], dict)
    ):
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        state = None

    if state is not None:
        state = _clean_state_dict_keys(state)

        try:
            model.load_state_dict(state, strict=True)
            print(f"Loaded checkpoint strictly from: {CKPT_PATH}")
        except Exception as e:
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"Loaded checkpoint non-strictly from: {CKPT_PATH}")
            print(f"  missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
            print(f"  strict load error was: {repr(e)}")
else:
    print("No checkpoint found; running with random weights (will score poorly).")

model = model.to(device).eval()




## === cell 3
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
    1, 3, 1, 1
)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
    1, 3, 1, 1
)


@torch.no_grad()
def process(image_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.resize(image_bgr, (512, 512), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return x.to(device)




## === cell 4
files = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
if len(files) == 0:
    raise FileNotFoundError(f"No test images found in {TEST_DIR}")

names, labels = [], []

sample_path = os.path.join(BASE, "sample_submission.csv")
if not os.path.isfile(sample_path):
    raise FileNotFoundError(f"sample_submission.csv not found at {sample_path}")

sample = pd.read_csv(sample_path)
id_to_path = {os.path.basename(p): p for p in files}

fallback_label = 0  # will be updated after we have some predictions

for image_id in tqdm.tqdm(sample["image_id"].tolist(), total=len(sample)):
    file = id_to_path.get(image_id, None)
    if file is None:
        names.append(image_id)
        labels.append(fallback_label)
        continue

    img = cv2.imread(file)
    if img is None:
        names.append(image_id)
        labels.append(fallback_label)
        continue

    x = process(img)
    out = model(x)
    pred = int(torch.argmax(out, dim=1).detach().cpu().item())

    names.append(image_id)
    labels.append(pred)

    fallback_label = pred  # keep a reasonable fallback if later reads fail




## === cell 5
sub = pd.DataFrame({"image_id": names, "label": labels})

sub = sample[["image_id"]].merge(sub, on="image_id", how="left")
if sub["label"].isna().any():
    fill = int(pd.Series(labels).mode().iloc[0]) if len(labels) else 0
    sub["label"] = sub["label"].fillna(fill).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape {sub.shape}")
print(sub.head())
