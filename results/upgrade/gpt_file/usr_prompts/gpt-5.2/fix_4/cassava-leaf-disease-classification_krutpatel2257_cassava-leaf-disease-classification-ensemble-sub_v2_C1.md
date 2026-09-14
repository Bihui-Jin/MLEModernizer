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

# 5. Target score

0.8931701420368692

# 6. Current score

0.08408

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.08408) has done: 'I fix the Albumentations v2 API break that prevents `sub_aug` from being created (the root cause of the cascading `NameError`s), by switching `RandomResizedCrop(height, width, ...)` to the new `RandomResizedCrop(size=(h,w), ...)` signature. I also keep the rest of the inference/ensemble logic identical, only adding a safe CPU fallback for the augmentation if Albumentations ever fails at runtime so the notebook always produces a valid `submission.csv`. Finally, I make the file/path resolution a bit more robust while keeping the same weights-loading behavior, so it runs end-to-end in this Kaggle environment and writes the correct submission format.'

# 9. Code solution

## === cell 0
import os
import glob
import re

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax

import torch
import torch.nn as nn
from torchvision import models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)



## === cell 1
resnet_model_path = "../input/rn-wc-tta-calr-clahe-cutmix/model(24).pth"
effnet_model_path = "../input/en-b4-tta-calr-clahe-v2-8/model(15).pth"
effnet_model_folds_path = [
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_0_11.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_1_10.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_2_13.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_3_12.pth",
    "../input/en-b4-tta-calr-clahe-v3-5-folds/EFFICIENT_NET_B4_4_11.pth",
]

sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"




## === cell 2
def _find_pth_by_regex(pattern: str, root: str = "/kaggle/input"):
    cand = []
    for p in glob.glob(os.path.join(root, "**", "*.pth"), recursive=True):
        if re.search(pattern, os.path.basename(p), flags=re.IGNORECASE):
            cand.append(p)
    return sorted(cand)[0] if cand else None


def _maybe_resolve_path(path: str, fallback_pattern: str):
    if path and os.path.exists(path):
        return path
    found = _find_pth_by_regex(fallback_pattern)
    return found  # may be None


resnet_model_path = _maybe_resolve_path(
    resnet_model_path, r"(resn|resnext|rn|next50|32x4d).*\.pth|model.*\.pth"
)

resolved_folds = [p for p in effnet_model_folds_path if os.path.exists(p)]
if len(resolved_folds) == 0:
    b4_pths = sorted(glob.glob("/kaggle/input/**/*.pth", recursive=True))
    b4_pths = [
        p
        for p in b4_pths
        if re.search(r"(b4|efficient.*b4|eff.*b4)", os.path.basename(p), re.IGNORECASE)
    ]
    resolved_folds = b4_pths[:5] if len(b4_pths) else []

effnet_model_folds_path = resolved_folds

print("Resolved resnet weights:", resnet_model_path)
print("Resolved effnet fold weights:", effnet_model_folds_path)




## === cell 3
def _clean_state_dict(sd):
    if (
        isinstance(sd, dict)
        and "state_dict" in sd
        and isinstance(sd["state_dict"], dict)
    ):
        sd = sd["state_dict"]
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_sd[nk] = v
    return new_sd


resnet_model = models.resnext50_32x4d(
    weights=models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2
)
resnet_model.fc = nn.Linear(2048, 5)
resnet_model.to(DEVICE).eval()

if resnet_model_path is not None and os.path.exists(resnet_model_path):
    resnet_sd = torch.load(resnet_model_path, map_location="cpu")
    resnet_sd = _clean_state_dict(resnet_sd)
    resnet_model.load_state_dict(resnet_sd, strict=False)
    print("Loaded ResNeXt custom weights.")
else:
    print(
        "WARNING: ResNeXt custom weights not found; using ImageNet backbone + random 5-class head."
    )

effnet_model = models.efficientnet_b4(
    weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
)
if isinstance(effnet_model.classifier, nn.Sequential):
    in_features = effnet_model.classifier[-1].in_features
    effnet_model.classifier[-1] = nn.Linear(in_features, 5)
else:
    in_features = effnet_model.classifier.in_features
    effnet_model.classifier = nn.Linear(in_features, 5)
effnet_model.to(DEVICE).eval()




## === cell 4
def load_image_np(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.array(im)




## === cell 5
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


def _img_to_tensor_chw_normalized(img_hwc: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(img_hwc).permute(2, 0, 1).contiguous().float()
    return x




## === cell 6
def predict_model_tta(
    model: torch.nn.Module, image_paths: list, tta_count: int, batch_size: int = 16
) -> np.ndarray:
    n = len(image_paths)
    preds = np.zeros((n, 5), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        batch_paths = image_paths[start:end]
        bs = len(batch_paths)

        imgs0 = [load_image_np(p) for p in batch_paths]

        logits_sum = torch.zeros((bs, 5), device=DEVICE, dtype=torch.float32)
        for _ in range(tta_count):
            batch_imgs = []
            for img0 in imgs0:
                try:
                    img = sub_aug(image=img0)["image"]
                except Exception:
                    img = A.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                        max_pixel_value=255.0,
                        p=1.0,
                    )(image=img0)["image"]
                batch_imgs.append(_img_to_tensor_chw_normalized(img))

            x = torch.stack(batch_imgs, dim=0).to(DEVICE, non_blocking=True)
            out = model(x).detach().float()
            logits_sum += out

        logits_avg = (logits_sum / float(tta_count)).cpu().numpy()
        preds[start:end] = logits_avg

    return preds




## === cell 7
sample_sub = pd.read_csv(sample_sub_path)
image_paths = [
    os.path.join(test_images_path, iid) for iid in sample_sub["image_id"].tolist()
]

tta_count = 5
resnet_predictions = predict_model_tta(
    resnet_model, image_paths, tta_count=tta_count, batch_size=16
)
print("ResNeXt preds shape:", resnet_predictions.shape)



## === cell 8
tta_count = 1
effnet_fold_preds = []

if len(effnet_model_folds_path) > 0:
    for model_path in effnet_model_folds_path:
        if not os.path.exists(model_path):
            continue
        sd = torch.load(model_path, map_location="cpu")
        sd = _clean_state_dict(sd)
        effnet_model.load_state_dict(sd, strict=False)
        effnet_model.to(DEVICE).eval()
        fold_preds = predict_model_tta(
            effnet_model, image_paths, tta_count=tta_count, batch_size=16
        )
        effnet_fold_preds.append(fold_preds)

if len(effnet_fold_preds) == 0:
    print(
        "WARNING: EfficientNet fold weights not found; using ImageNet backbone + random 5-class head."
    )
    effnet_predictions = predict_model_tta(
        effnet_model, image_paths, tta_count=tta_count, batch_size=16
    )
else:
    effnet_predictions = np.mean(np.stack(effnet_fold_preds, axis=0), axis=0)

print("EffNet preds shape:", effnet_predictions.shape)



## === cell 9
combine_logits = (effnet_predictions * 0.5) + (resnet_predictions * 0.5)
combine_preds = softmax(combine_logits, axis=1).argmax(axis=1).astype(int)

sub_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": combine_preds}
)
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head())
print("Saved submission.csv with shape:", sub_df.shape)
print(
    "submission.csv exists:",
    os.path.exists("submission.csv"),
    "size:",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)
