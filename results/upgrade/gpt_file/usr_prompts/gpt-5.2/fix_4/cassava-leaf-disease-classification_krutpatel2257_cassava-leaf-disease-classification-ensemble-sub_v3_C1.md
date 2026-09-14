# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
scipy==1.15.3
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

# 5. Code solution

## === cell 0
import os
import glob
import warnings

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax

import torch
import torch.nn as nn
from torchvision import models, transforms

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)



## === cell 1
resnet_model_path = "../input/rn-wc-tta-calr-clahe-cutmix/model(24).pth"
effnet_model_path = "../input/en-b4-tta-calr-clahe-v3/eff_epoch_11.pth"

sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"


def _find_first_existing(path_candidates):
    for p in path_candidates:
        if p and os.path.isfile(p):
            return p
    return None


def _autofind_checkpoint(patterns, search_root="/kaggle/input"):
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(os.path.join(search_root, "**", pat), recursive=True))
    hits = [h for h in hits if os.path.isfile(h)]
    hits.sort()
    return hits[0] if hits else None


_resnet_found = _find_first_existing([resnet_model_path])
_effnet_found = _find_first_existing([effnet_model_path])

if _resnet_found is None:
    _resnet_found = _autofind_checkpoint(
        patterns=[
            "model(24).pth",
            "*resnext*50*.pth",
            "*resnext*.pth",
            "*resnet*.pth",
        ]
    )
if _effnet_found is None:
    _effnet_found = _autofind_checkpoint(
        patterns=[
            "eff_epoch_11.pth",
            "*efficientnet*b4*.pth",
            "*eff*epoch*.pth",
        ]
    )

resnet_model_path = _resnet_found
effnet_model_path = _effnet_found

USE_EXTERNAL_CKPTS = (resnet_model_path is not None) and (effnet_model_path is not None)

print("External resnet checkpoint:", resnet_model_path)
print("External effnet checkpoint:", effnet_model_path)
print("USE_EXTERNAL_CKPTS =", USE_EXTERNAL_CKPTS)




## === cell 2
def load_checkpoint_safely(model, ckpt_path, device=DEVICE):
    if ckpt_path is None or (not os.path.isfile(ckpt_path)):
        raise FileNotFoundError(f"Checkpoint path not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt

    new_state = {}
    for k, v in state.items():
        nk = k[7:] if isinstance(k, str) and k.startswith("module.") else k
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print(
            f"[WARN] Missing keys when loading {os.path.basename(ckpt_path)}: {len(missing)}"
        )
    if unexpected:
        print(
            f"[WARN] Unexpected keys when loading {os.path.basename(ckpt_path)}: {len(unexpected)}"
        )
    return model




## === cell 3
if USE_EXTERNAL_CKPTS:
    resnet_model = models.resnext50_32x4d(weights=None)
    resnet_model.fc = nn.Linear(2048, 5)
    resnet_model.to(DEVICE)
    resnet_model = load_checkpoint_safely(
        resnet_model, resnet_model_path, device=DEVICE
    )
    resnet_model.eval()

    effnet_model = models.efficientnet_b4(weights=None)
    in_features = effnet_model.classifier[1].in_features
    effnet_model.classifier[1] = nn.Linear(in_features, 5)
    effnet_model.to(DEVICE)
    effnet_model = load_checkpoint_safely(
        effnet_model, effnet_model_path, device=DEVICE
    )
    effnet_model.eval()
else:
    resnet_model = models.resnext50_32x4d(
        weights=models.ResNeXt50_32X4D_Weights.DEFAULT
    )
    resnet_model.fc = nn.Linear(2048, 5)  # keep 5-way head as required
    resnet_model.to(DEVICE).eval()

    effnet_model = models.efficientnet_b4(
        weights=models.EfficientNet_B4_Weights.DEFAULT
    )
    in_features = effnet_model.classifier[1].in_features
    effnet_model.classifier[1] = nn.Linear(
        in_features, 5
    )  # keep 5-way head as required
    effnet_model.to(DEVICE).eval()

print("Models ready on:", DEVICE)



## === cell 4
to_tensor = transforms.ToTensor()



## === cell 5
try:
    sub_aug = A.Compose(
        [
            A.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0)),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.8),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    )
except Exception as e:
    print(
        "[WARN] Failed to build albumentations TTA pipeline, falling back to resize+normalize. Error:"
    )
    print(str(e))
    sub_aug = A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ],
        p=1.0,
    )



## === cell 6
if not os.path.isfile(sample_sub_path):
    raise FileNotFoundError(f"sample_submission.csv not found at: {sample_sub_path}")
if not os.path.isdir(test_images_path):
    raise FileNotFoundError(f"test_images directory not found at: {test_images_path}")

sample_sub = pd.read_csv(sample_sub_path)
assert "image_id" in sample_sub.columns and "label" in sample_sub.columns
print("Test rows:", len(sample_sub))



## === cell 7
tta_count = 10

predictions = []
resnet_model.eval()
effnet_model.eval()

with torch.no_grad():
    for _, sample_row in sample_sub.iterrows():
        img_path = os.path.join(test_images_path, sample_row.image_id)
        base_img = np.array(Image.open(img_path).convert("RGB"))

        image_pred = 0.0
        for _ in range(tta_count):
            aug_img = sub_aug(image=base_img)["image"]  # HWC float32 normalized

            if not isinstance(aug_img, np.ndarray):
                aug_img = np.array(aug_img)
            aug_img = np.ascontiguousarray(aug_img, dtype=np.float32)

            img_tensor = to_tensor(aug_img).to(DEVICE, dtype=torch.float32)

            outputs = resnet_model(img_tensor.unsqueeze(0))
            image_pred += outputs.detach().cpu().numpy()

            outputs = effnet_model(img_tensor.unsqueeze(0))
            image_pred += outputs.detach().cpu().numpy()

        image_pred /= tta_count * 2
        predictions.extend(image_pred)

predictions = np.asarray(predictions)
pred_labels = softmax(predictions, axis=1).argmax(axis=1)

sub_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": pred_labels.astype(int)}
)
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("Saved to:", os.path.abspath("submission.csv"))
