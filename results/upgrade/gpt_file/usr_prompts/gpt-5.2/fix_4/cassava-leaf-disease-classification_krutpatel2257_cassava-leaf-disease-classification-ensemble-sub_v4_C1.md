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

0.8952855847688124

# 6. Current score

0.10688

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the unavailable `efficientnet_pytorch` dependency and replace it with `torchvision.models.efficientnet_b4`, which preserves the intended EfficientNet-B4 backbone while fixing the import error. I also make the model weight paths robust by searching under `/kaggle/input` for the referenced `.pth` files (or gracefully falling back to untrained weights if they truly aren’t present), ensuring the notebook runs end-to-end and always writes `submission.csv`. I update the Albumentations `RandomResizedCrop` call to the v2 API (`size=(H, W)`) to fix the validation error, and I ensure consistent device placement and `torch.no_grad()` during inference to avoid runtime issues. Finally, I keep the original inference logic (TTA, weighting, softmax+argmax) intact so the score behavior is unchanged aside from necessary compatibility fixes.'
- What this solution (achieved 0.61061) has done: 'Your current 0.61099 is far below the 0.8953 target, so we should make a small change that plausibly increases accuracy without changing the core model/ensemble/TTA logic. The biggest issue is that you’re doing *random* augmentations at test time (RandomResizedCrop + rotations), which can distort leaves and hurt accuracy; we switch to a standard deterministic validation-style resize/center-crop normalization for TTA (and keep flips/transpose as the only stochastic TTA). We also batch the 10 TTA forward passes per image into a single tensor to reduce overhead and keep runtime safely under the limit, without changing the inference semantics. Everything else (two backbones, weight loading, 0.6/0.4 weighting, softmax+argmax, submission format/pathing) stays the same.'
- What this solution (achieved 0.10688) has done: 'Your current score (0.61061) is far below the target (0.89529), so we should make a small, high-impact correction that improves accuracy without changing the ensemble/TTA/training semantics. The main likely issue is an incorrect input normalization: Albumentations `Normalize` already outputs float tensors in [0,1] normalized by mean/std, but you then apply `transforms.ToTensor()` which divides by 255 again, badly scaling inputs and crushing accuracy. I replace `ToTensor()` with a small converter that turns the already-normalized HWC float image into a CHW torch tensor without rescaling. Everything else (models, weights loading, 0.6/0.4 blending, 10x TTA, softmax+argmax, submission format) remains unchanged.'

# 9. Code solution

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
print("DEVICE:", DEVICE)



## === cell 1
resnet_model_path = "../input/rn-wc-tta-calr-clahe-cutmix/model(24).pth"
effnet_model_path = "../input/en-b4-tta-calr-clahe-v3/eff_epoch_11.pth"

sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"




## === cell 2
def resolve_path(preferred_path: str):
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    base = os.path.basename(preferred_path) if preferred_path else ""
    if not base:
        return None

    candidates = glob.glob(f"/kaggle/input/**/{base}", recursive=True)
    if candidates:
        candidates = sorted(candidates, key=lambda p: (len(p), p))
        return candidates[0]
    return None


resnet_model_path_resolved = resolve_path(resnet_model_path)
effnet_model_path_resolved = resolve_path(effnet_model_path)

print("Resolved resnet weights:", resnet_model_path_resolved)
print("Resolved effnet weights:", effnet_model_path_resolved)

if not os.path.exists(sample_sub_path):
    sample_sub_path_alt = resolve_path(sample_sub_path)
    if sample_sub_path_alt:
        sample_sub_path = sample_sub_path_alt

if not os.path.exists(test_images_path):
    dirs = glob.glob("/kaggle/input/**/test_images", recursive=True)
    if dirs:
        test_images_path = sorted(dirs, key=lambda p: (len(p), p))[0]

print("sample_sub_path:", sample_sub_path)
print("test_images_path:", test_images_path)
assert os.path.exists(
    sample_sub_path
), f"sample_submission.csv not found at {sample_sub_path}"
assert os.path.isdir(
    test_images_path
), f"test_images directory not found at {test_images_path}"



## === cell 3
resnet_model = models.resnext50_32x4d(weights=None)
resnet_model.fc = nn.Linear(2048, 5)
resnet_model.to(DEVICE)

effnet_model = models.efficientnet_b4(weights=None)
in_features = effnet_model.classifier[-1].in_features
effnet_model.classifier[-1] = nn.Linear(in_features, 5)
effnet_model.to(DEVICE)


def load_weights_safely(model, path):
    """
    Keep behavior: load provided checkpoints if present; otherwise run with random init
    (still producing a valid submission). Handle common wrappers like 'state_dict' and 'module.'.
    """
    if not path or not os.path.exists(path):
        print(
            f"WARNING: weights not found for {model.__class__.__name__}; using random initialization."
        )
        return

    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt

    if isinstance(state_dict, dict) and any(
        k.startswith("module.") for k in state_dict.keys()
    ):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(
            f"NOTE: Missing keys while loading {os.path.basename(path)}: {len(missing)}"
        )
    if unexpected:
        print(
            f"NOTE: Unexpected keys while loading {os.path.basename(path)}: {len(unexpected)}"
        )
    print(f"Loaded weights from: {path}")


load_weights_safely(resnet_model, resnet_model_path_resolved)
load_weights_safely(effnet_model, effnet_model_path_resolved)

resnet_model.eval()
effnet_model.eval()




## === cell 4
def albumentations_image_to_tensor(img: np.ndarray) -> torch.Tensor:
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)




## === cell 5
sub_aug = A.Compose(
    [
        A.Resize(height=512, width=512),
        A.CenterCrop(height=512, width=512),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
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
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 7
sample_sub = pd.read_csv(sample_sub_path)
tta_count = 10

predictions = []

with torch.no_grad():
    for _, sample_row in sample_sub.iterrows():
        img_path = os.path.join(test_images_path, sample_row.image_id)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        tta_batch = []
        for _ in range(tta_count):
            aug_img = sub_aug(image=img_np)["image"]
            tta_batch.append(albumentations_image_to_tensor(aug_img))
        tta_batch = torch.stack(tta_batch, dim=0).to(DEVICE)  # [T,3,H,W]

        res_out = resnet_model(tta_batch).detach().cpu().numpy()  # [T,5]
        eff_out = effnet_model(tta_batch).detach().cpu().numpy()  # [T,5]

        image_pred = (0.6 * res_out + 0.4 * eff_out).mean(
            axis=0, keepdims=True
        )  # [1,5]
        predictions.extend(image_pred)

predictions = softmax(np.asarray(predictions), axis=1).argmax(axis=1)

sub_df = pd.DataFrame(
    {"image_id": sample_sub.image_id, "label": predictions.astype(int)}
)
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
