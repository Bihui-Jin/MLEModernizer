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

0.866424901783016

# 6. Current score

0.08931

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11136) has done: 'I fix the missing `efficientnet_pytorch` import by removing the unavailable wheel install and switching to the built-in `torchvision` EfficientNet-B4, then load your `.pth` safely (handling common key-prefix issues). I also update the Albumentations transforms to the v2 API (e.g., `RandomResizedCrop(size=(h,w))` and replace removed `Cutout` with `CoarseDropout`) so augmentation runs. Finally, I correct the TTA averaging bug (`/=10` should match the number of passes) and make the inference loop device/dtype-correct, producing a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.11024) has done: 'I fix the failure caused by the missing external checkpoint by making the script automatically fall back to a reasonable “no-checkpoint” mode when the `.pth` file isn’t present, while keeping your EfficientNet-B4 + TTA inference core logic intact. To improve the very low score (0.11136) toward the target, I also remove clearly-train-time augmentations (crop/dropout/color jitter) from test-time TTA and replace them with standard, label-preserving test transforms (resize/center-crop + flips/transpose), which typically yields a large accuracy gain without changing the modeling approach. I correct the Albumentations→Tensor conversion to avoid double-normalization pitfalls by explicitly converting HWC float32 to CHW tensor. The script still write a valid `submission.csv` with the required columns and row alignment with `sample_submission.csv`.'
- What this solution (achieved 0.08931) has done: 'Your score is extremely low because the checkpoint path points to a dataset that isn’t present in your environment, so the model is running with random weights; the smallest change that moves accuracy toward the target is to load a real pretrained EfficientNet-B4 baseline when the external `.pth` is missing. This preserves your core architecture and inference/TTA logic, but replaces “random model” with “ImageNet-initialized model,” which typically yields a large jump in accuracy on cassava. I also add a small safety fix to ensure inputs are `float32` and on the right device, avoiding silent dtype/device mismatches. The script still runs end-to-end and writes a valid `submission.csv` with the required columns aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



## === cell 1
model_path = "../input/en-b4-tta-calr-40/model_40.pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

assert os.path.exists(sample_sub_path), f"Missing sample submission: {sample_sub_path}"
assert os.path.isdir(test_images_path), f"Missing test_images dir: {test_images_path}"

has_ckpt = os.path.exists(model_path)
print("Checkpoint found:", has_ckpt, "| path:", model_path)



## === cell 2
if has_ckpt:
    model = models.efficientnet_b4(weights=None)
else:
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 5)
model = model.to(device)


def _strip_prefix(sd, prefix):
    if (
        isinstance(sd, dict)
        and len(sd) > 0
        and all(k.startswith(prefix) for k in sd.keys())
    ):
        return {k[len(prefix) :]: v for k, v in sd.items()}
    return sd


if has_ckpt:
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )
    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "model.")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(
        f"Loaded checkpoint. Missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
    )
else:
    print(
        "INFO: No cassava checkpoint found; using ImageNet pretrained backbone weights."
    )

model.eval()



## === cell 3
sub_aug = A.Compose(
    [
        A.Resize(512, 512, p=1.0),
        A.CenterCrop(380, 380, p=1.0),  # EfficientNet-B4 default-ish crop size
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Transpose(p=0.2),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)


def to_tensor_hwc_float01_or_norm(img_hwc: np.ndarray) -> torch.Tensor:
    if img_hwc.ndim != 3 or img_hwc.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got shape {img_hwc.shape}")
    x = (
        torch.from_numpy(img_hwc.astype(np.float32, copy=False))
        .permute(2, 0, 1)
        .contiguous()
    )
    return x




## === cell 4
sample_sub = pd.read_csv(sample_sub_path)

predictions = []
tta_passes = 5  # keep original intent

with torch.no_grad():
    for _, sample_row in sample_sub.iterrows():
        img_path = os.path.join(test_images_path, sample_row.image_id)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        logits_sum = None
        for _ in range(tta_passes):
            aug_img = sub_aug(image=img_np)["image"]  # float32 HWC normalized
            x = (
                to_tensor_hwc_float01_or_norm(aug_img)
                .unsqueeze(0)
                .to(device, non_blocking=True)
            )

            out = model(x)
            logits_sum = out if logits_sum is None else (logits_sum + out)

        logits_avg = logits_sum / float(tta_passes)
        pred_label = int(torch.argmax(logits_avg, dim=1).item())
        predictions.append([sample_row.image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
