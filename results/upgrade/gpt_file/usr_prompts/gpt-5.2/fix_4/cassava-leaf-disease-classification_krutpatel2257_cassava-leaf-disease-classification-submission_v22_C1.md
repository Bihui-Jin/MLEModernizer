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

0.8649138712601995

# 6. Current score

0.10164

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.15919) has done: 'I fix the missing `efficientnet_pytorch` dependency by removing the notebook-only `pip install` cell and switching to `torchvision.models.efficientnet_b4`, which is available in your environment. I keep the same inference-only approach and ensure the classifier head outputs 5 classes, then load the provided `.pth` weights in a robust way (handling common key-prefix issues and map_location). I also fix device handling and wrap inference in `torch.no_grad()` to avoid unnecessary memory use and runtime issues. Finally, I ensure a valid `submission.csv` is always produced with the required `image_id,label` columns aligned to `sample_submission.csv`.'
- What this solution (achieved 0.15845) has done: 'I fix the missing model checkpoint path by automatically locating a `.pth` file under `../input/` (or falling back to the originally specified path if it exists), so the notebook runs end-to-end without manual dataset dependency changes. To move the accuracy score up toward your target (your current score is far below target), I keep the same EfficientNet-B4 inference-only approach but ensure inference preprocessing matches standard EfficientNet training (resize to 380 and use ImageNet normalization) and correct a subtle tensor conversion issue (avoid double-converting already-normalized arrays). I also make the image iteration robust and fast by iterating over `image_id` values directly and using `torch.inference_mode()`. Finally, I always write a valid `submission.csv` with `image_id,label` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.10164) has done: 'I fix the immediate runtime failure by removing the hard dependency on an external `.pth` checkpoint (none exists under `../input/` in this environment) and ensuring the pipeline still runs end-to-end. To move accuracy up substantially toward your target without changing the core “EfficientNet-B4 inference over images” approach, I switch to using torchvision’s built-in pretrained EfficientNet-B4 weights (ImageNet) and keep the same 5-class head (randomly initialized) as in your logic. I also ensure preprocessing matches EfficientNet’s expected inference transforms by using the official `EfficientNet_B4_Weights` transform (includes correct resize/crop/normalization), which is a minimal but high-impact fix for score compared to the current ad-hoc resize. Finally, I keep the submission aligned to `sample_submission.csv` and always write `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob

import albumentations
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



## === cell 1
model_path = "../input/en-b4-tta-calr-15-v2/model(13).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

assert os.path.exists(sample_sub_path), f"Missing sample submission: {sample_sub_path}"
assert os.path.isdir(test_images_path), f"Missing test images dir: {test_images_path}"

ckpt_path = None
if os.path.exists(model_path):
    ckpt_path = model_path
else:
    candidates = sorted(glob.glob("../input/**/*.pth", recursive=True))
    if len(candidates) > 0:
        ckpt_path = candidates[0]

print("Checkpoint found?", ckpt_path is not None)
if ckpt_path is not None:
    print("Using ckpt_path:", ckpt_path)
else:
    print(
        "No .pth checkpoint found under ../input/. Will use torchvision pretrained EfficientNet-B4 weights."
    )



## === cell 2
from torchvision.models import EfficientNet_B4_Weights

weights = EfficientNet_B4_Weights.DEFAULT if ckpt_path is None else None
model = models.efficientnet_b4(weights=weights)

in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, 5)

model.to(device)

if ckpt_path is not None:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    clean_sd = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        clean_sd[nk] = v

    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    print(
        f"Loaded weights. Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}"
    )
    if len(missing) and len(missing) < 25:
        print("Missing keys (first 20):", missing[:20])
    if len(unexpected) and len(unexpected) < 25:
        print("Unexpected keys (first 20):", unexpected[:20])

model.eval()

if weights is not None:
    preprocess = weights.transforms()
else:
    preprocess = None



## === cell 3
sub_aug = albumentations.Compose(
    [
        albumentations.Resize(380, 380),
        albumentations.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)


def chw_tensor_from_hwc_float(img_hwc_float: np.ndarray) -> torch.Tensor:
    if img_hwc_float.dtype != np.float32:
        img_hwc_float = img_hwc_float.astype(np.float32)
    t = torch.from_numpy(img_hwc_float).permute(2, 0, 1).contiguous()
    return t




## === cell 4
sample_sub = pd.read_csv(sample_sub_path)

tta_count = 1  # keep identical behavior (no extra TTA loops)

pred_labels = []

with torch.inference_mode():
    for image_id in sample_sub["image_id"].values:
        img_path = os.path.join(test_images_path, image_id)
        pil_img = Image.open(img_path).convert("RGB")

        image_pred = None
        for _ in range(tta_count):
            if preprocess is not None:
                image_t = preprocess(pil_img).to(device)
            else:
                image_np = np.array(pil_img)
                aug_img = sub_aug(image=image_np)["image"]  # float32 normalized HWC
                image_t = chw_tensor_from_hwc_float(aug_img).to(device)

            outputs = model(image_t.unsqueeze(0))
            image_pred = outputs if image_pred is None else (image_pred + outputs)

        image_pred = image_pred / float(tta_count)
        pred_label = int(torch.argmax(image_pred, dim=1).item())
        pred_labels.append(pred_label)

sub_df = pd.DataFrame({"image_id": sample_sub["image_id"].values, "label": pred_labels})
sub_df["label"] = sub_df["label"].astype(int)

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
assert sub_df.shape[0] == sample_sub.shape[0]
assert list(sub_df.columns) == ["image_id", "label"]
