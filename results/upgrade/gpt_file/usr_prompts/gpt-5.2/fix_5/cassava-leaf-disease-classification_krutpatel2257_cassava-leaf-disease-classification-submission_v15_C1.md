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

0.19768

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24178) has done: 'I first fix the missing pretrained weights issue by loading a locally available ResNeXt50 ImageNet backbone (instead of a non-existent `.pth` path) so the notebook runs end-to-end and produces a valid `submission.csv`. Next, I update the Albumentations augmentation definitions to the v2 API (the current `RandomResizedCrop(256, 256)` signature is invalid) and replace removed transforms (Cutout) with their supported equivalents. Finally, I fix inference correctness/stability issues (proper RGB conversion, `torch.no_grad()`, device handling, and ensuring transforms output is used correctly) so predictions are generated for all test images and written in the required format.'
- What this solution (achieved 0.23468) has done: 'Your score is far below the target, and the main reason is that you’re running a fresh ImageNet ResNeXt with a randomly initialized 5-class head (and you never load the provided fine-tuned checkpoint), so predictions are essentially garbage. The smallest change that should move accuracy sharply upward toward your target is to correctly load the local `model(11).pth` weights into the model (handling common checkpoint key formats) and then run the same TTA inference as you already do. I also ensure the Albumentations output is converted to a float tensor correctly (right now `ToTensor()` can mis-handle already-normalized float arrays), without changing the model or overall approach. The rest of your pipeline (TTA loop, argmax, submission formatting) stays the same.'
- What this solution (achieved 0.24215) has done: 'Your score is far below the target because the fine-tuned checkpoint still isn’t being applied to the model’s 5-class head (most likely the checkpoint stores a different head name/shape), so you’re effectively running an ImageNet backbone with a randomly initialized classifier. I make the smallest change that directly fixes this: load the checkpoint in a “compatible” way by (1) mapping common key prefixes and (2) skipping only the final-layer weights when shapes don’t match, while keeping everything else identical. This preserves your architecture and TTA inference loop, but ensures you actually use the learned cassava features, which should move accuracy sharply upward toward your target band. I also keep preprocessing identical, only ensuring tensor dtype is float for safety.'
- What this solution (achieved 0.19768) has done: 'Your score is far below the target because you’re still doing “train-time style” augmentations at inference (RandomResizedCrop/color jitter/dropout), which destroys accuracy for classification; the smallest change that should move accuracy sharply upward is to switch to a deterministic test-time transform (resize/center-crop + normalize) while keeping your exact model, checkpoint-loading logic, and TTA averaging loop intact. I also set `model.eval()` right before inference (to be safe) and make the checkpoint load slightly stricter for the classifier head by explicitly remapping common `fc`/`classifier` key variants so the fine-tuned 5-class head is more likely to load instead of being skipped. These are minimal changes that preserve core semantics (argmax over averaged logits) but should improve correctness and move accuracy toward your 0.8856 target. The script still runs end-to-end and writes a valid `submission.csv` with the required columns.'

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



## === cell 1
model_path = "../input/rn-tta-calr-ft-ofasf/model(11).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 2
try:
    weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
except Exception:
    weights = None

model = models.resnext50_32x4d(weights=weights)
model.fc = nn.Linear(2048, 5)
model = model.to(device)


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "network"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        if any(torch.is_tensor(v) for v in ckpt_obj.values()):
            return ckpt_obj
    return ckpt_obj


def _strip_known_prefixes(state_dict):
    for prefix in ("module.", "model.", "net.", "network."):
        state_dict = {
            (k[len(prefix) :] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()
        }
    return state_dict


def _remap_classifier_keys(sd):
    remapped = {}
    for k, v in sd.items():
        if k.startswith("classifier."):
            remapped["fc." + k[len("classifier.") :]] = v
        elif k.startswith("head."):
            remapped["fc." + k[len("head.") :]] = v
        else:
            remapped[k] = v

    extra = {}
    for k, v in remapped.items():
        if k.startswith("fc."):
            alt = "classifier." + k[len("fc.") :]
            if alt not in remapped:
                extra[alt] = v
    remapped.update(extra)
    return remapped


def _load_ckpt_compat(model, state_dict):
    model_sd = model.state_dict()
    filtered = {}
    skipped = []
    for k, v in state_dict.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            if k in model_sd:
                skipped.append((k, tuple(v.shape), tuple(model_sd[k].shape)))
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return missing, unexpected, skipped


if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    sd = _extract_state_dict(ckpt)
    if isinstance(sd, dict):
        sd = _strip_known_prefixes(sd)
        sd = _remap_classifier_keys(sd)

        missing, unexpected, skipped = _load_ckpt_compat(model, sd)
        print("Loaded checkpoint (compat):", model_path)
        print(
            "Missing keys:",
            len(missing),
            "Unexpected keys:",
            len(unexpected),
            "Skipped (shape-mismatch):",
            len(skipped),
        )
        if len(skipped) > 0:
            print("Example skipped keys:", skipped[:5])
    else:
        print(
            "WARNING: checkpoint format not understood; running without fine-tuned weights."
        )
else:
    print(
        "WARNING: model_path does not exist, running without fine-tuned weights:",
        model_path,
    )

model.eval()



## === cell 3
from albumentations.pytorch import ToTensorV2

sub_aug = A.Compose(
    [
        A.SmallestMaxSize(max_size=256, p=1.0),
        A.CenterCrop(height=256, width=256, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ],
    p=1.0,
)

sub_aug_with_tensor = sub_aug



## === cell 4
sample_sub = pd.read_csv(sample_sub_path)

predictions = []
tta_n = 10

model.eval()
with torch.no_grad():
    for _, sample_row in sample_sub.iterrows():
        image_path = os.path.join(test_images_path, sample_row.image_id)

        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        image_pred = None
        for _ in range(tta_n):
            aug_out = sub_aug_with_tensor(image=image_np)
            x = aug_out["image"].to(device).float()  # torch.FloatTensor CHW

            outputs = model(x.unsqueeze(0))
            if image_pred is None:
                image_pred = outputs
            else:
                image_pred += outputs

        image_pred /= tta_n
        pred_label = int(torch.argmax(image_pred, dim=1).item())
        predictions.append([sample_row.image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with", len(sub_df), "rows")
