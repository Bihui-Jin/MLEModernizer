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

0.8856

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.15396) has done: 'Diagnosis: Cell 3 crashes because `albumentations.Cutout` was removed/relocated in Albumentations v2.x, so accessing it as `albumentations.Cutout` raises `AttributeError`. The rest of the pipeline expects an augmentation object in the Compose list that performs cutout-style masking. Albumentations v2 provides the equivalent functionality via `albumentations.augmentations.dropout.cutout.Cutout` (and related dropout transforms), so we can import it and keep the same semantics.

Patch summary: Modify only cell 3 to use a version-compatible Cutout import. Try the v2 import path first and fall back to the older `albumentations.Cutout` if present, preserving the original augmentation behavior and interface.

Updated cells: Only cell 3 is updated below.

Compatibility notes for cell k+1: `sub_aug` remains an `albumentations.Compose` and `sub_aug(image=image)["image"]` continues to return a normalized HWC numpy array, so cell 4 works unchanged.

Assumptions: The installed Albumentations version is 2.0.8 (as provided), where Cutout exists under the new module path.'
- What this solution (achieved 0.05531) has done: 'Your very low score is most consistent with the pretrained weights not actually being loaded, so the model is effectively random; the current code silently proceeds with random initialization if it can’t find the checkpoint. I make a minimal, execution-safe change that (1) searches specifically for `model(11).pth` first (then any `.pth/.pt`), (2) fails fast if no weights are found (so you don’t unknowingly submit random predictions again), and (3) loads checkpoints robustly whether they were saved as a plain `state_dict` or wrapped under keys like `state_dict`. These changes preserve the same model architecture and TTA logic, but ensure the intended trained weights are actually used, which should move accuracy sharply toward your target.'
- What this solution (achieved 0.05531) has done: 'The crash happens because `model_path` points to a Kaggle dataset (`rn-tta-calr-ft-ofasf`) that is not present in this environment, and the current logic raises `FileNotFoundError` when it can’t find any `.pth/.pt` checkpoint. To unblock execution while preserving the rest of the notebook’s behavior, I keep the checkpoint search but, if nothing is found, fall back to running with randomly initialized weights instead of hard-failing. This keeps the model architecture and inference code unchanged and ensures `model` is still placed on the correct device and set to `eval()` for cell 3+.'

# 9. Code solution

## === cell 0
import os

import albumentations
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms



## === cell 1
model_path = "../input/rn-tta-calr-ft-ofasf/model(11).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"



## === cell 2
model = models.resnext50_32x4d(pretrained=False)
model.fc = nn.Linear(2048, 5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def _find_checkpoint(initial_path: str) -> str:
    if os.path.exists(initial_path):
        return initial_path

    base_dir = os.path.dirname(sample_sub_path)
    candidates = [
        os.path.join(base_dir, "model(11).pth"),
        os.path.join(base_dir, "model.pth"),
        os.path.join(base_dir, "model(11).pt"),
        os.path.join(base_dir, "model.pt"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    preferred = None
    fallback = None
    search_roots = [
        "../input",
        "../kaggle/input",
        "/kaggle/input",
        "/kaggle/data",
        "input",
        "data",
        "/kaggle/working",
    ]
    for root_dir in search_roots:
        if not os.path.isdir(root_dir):
            continue
        for root, _, files in os.walk(root_dir):
            for f in files:
                lf = f.lower()
                if lf == "model(11).pth":
                    preferred = os.path.join(root, f)
                    break
                if (lf.endswith(".pth") or lf.endswith(".pt")) and fallback is None:
                    fallback = os.path.join(root, f)
            if preferred is not None:
                break
        if preferred is not None:
            break

    return preferred if preferred is not None else fallback


def _load_checkpoint_weights(model, ckpt_path: str, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    cleaned = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if len(missing) > 0:
        print(
            f"WARNING: Missing keys when loading checkpoint (showing up to 20): {missing[:20]}"
        )
    if len(unexpected) > 0:
        print(
            f"WARNING: Unexpected keys when loading checkpoint (showing up to 20): {unexpected[:20]}"
        )


model_path_found = _find_checkpoint(model_path)

if model_path_found is None:
    print(
        "WARNING: No .pth/.pt checkpoint found; proceeding with randomly initialized weights. "
        f"Tried configured model_path='{model_path}' and searched common Kaggle input roots."
    )
else:
    model_path = model_path_found
    print(f"Loading model weights from: {model_path}")
    _load_checkpoint_weights(model, model_path, device)

model.eval()


## === cell 3
try:
    from albumentations.augmentations.dropout.cutout import Cutout as AlbCutout
except Exception:
    AlbCutout = getattr(albumentations, "Cutout", None)

sub_aug = albumentations.Compose(
    [
        albumentations.RandomResizedCrop(size=(256, 256)),
        albumentations.Transpose(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.ShiftScaleRotate(p=0.5),
        albumentations.HueSaturationValue(
            hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
        ),
        albumentations.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        albumentations.CoarseDropout(p=0.5),
        (
            AlbCutout(p=0.5)
            if AlbCutout is not None
            else albumentations.CoarseDropout(p=0.5)
        ),
    ],
    p=1.0,
)



## === cell 4
sample_sub = pd.read_csv(sample_sub_path)

predictions = []
for _, sample_row in sample_sub.iterrows():
    image_pred = 0
    for j in range(10):
        image = np.array(
            Image.open(os.path.join(test_images_path, sample_row.image_id))
        )
        image = sub_aug(image=image)["image"]
        image = transforms.ToTensor()(np.array(image))
        image = image.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        outputs = model(image.unsqueeze(0))
        image_pred += outputs
    image_pred /= 10
    _, pred_label = torch.max(image_pred, 1)

    predictions.append([sample_row.image_id, pred_label.item()])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
