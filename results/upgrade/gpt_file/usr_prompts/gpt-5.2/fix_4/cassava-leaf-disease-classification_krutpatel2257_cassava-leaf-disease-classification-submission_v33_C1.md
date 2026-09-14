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

0.8913569054094893

# 6. Current score

0.05643

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'I fix the run-blocking issues while preserving the same ResNeXt50 inference + TTA core logic. First, I make the model checkpoint path robust by searching common `/kaggle/input/**` locations and loading on CPU/GPU correctly (your current path points to a dataset that isn’t present). Next, I update the Albumentations `RandomResizedCrop` call to the v2 API (expects `size=(H,W)`), which currently raises a validation error and prevents `sub_aug` from being defined. Finally, I ensure inference uses `torch.no_grad()` and proper normalization/tensor shapes, and that we always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.1009) has done: 'I fix the run-blocking checkpoint issue by falling back to a standard torchvision pretrained ResNeXt50 if no external `.pth/.pt` file exists in the Kaggle environment, which preserves the same model architecture and inference flow while avoiding the FileNotFoundError. I also make loading robust to minor key mismatches by using `strict=False` only when a checkpoint is found and keys don’t exactly match, so inference still runs. Finally, I keep your Albumentations v2 `RandomResizedCrop(size=...)` and the same TTA loop, ensuring a valid `submission.csv` is always produced with the required columns.'
- What this solution (achieved 0.05643) has done: 'Your low score is primarily because the code is almost certainly running without the intended cassava-trained checkpoint (so the 5-class head is random), and because the Albumentations pipeline is currently mixing `Normalize` with `torchvision.transforms.ToTensor()` (which re-scales floats again and breaks normalization). I make two minimal, directly score-relevant fixes while preserving the same ResNeXt50 + logits-averaging TTA core logic: (1) robustly locate a real cassava `.pth/.pt` by preferring checkpoints that look like cassava/leaf/model and (2) fix the image-to-tensor conversion so normalized float HWC becomes CHW without extra scaling. These changes should move accuracy sharply upward toward your target without altering the model architecture or the TTA averaging semantics. The script still run end-to-end and write a valid `submission.csv`.'

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



## === cell 1
model_path = "../input/rn-wc-tta-calr-clahe-cutmix/model(24).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"




## === cell 2
def resolve_checkpoint_path(primary_path: str) -> str:
    if os.path.exists(primary_path):
        return primary_path

    candidates = []
    for root in ("/kaggle/input", "../input", "./input"):
        if os.path.isdir(root):
            candidates.extend(
                glob.glob(os.path.join(root, "**", "*.pth"), recursive=True)
            )
            candidates.extend(
                glob.glob(os.path.join(root, "**", "*.pt"), recursive=True)
            )

    def score_path(p: str) -> int:
        pl = p.lower()
        base = os.path.basename(pl)
        s = 0
        for kw, w in [
            ("cassava", 50),
            ("leaf", 40),
            ("disease", 40),
            ("classification", 20),
            ("model", 20),
            ("best", 15),
            ("fold", 10),
            ("resnext", 10),
            ("resne", 10),
            ("rn", 5),
        ]:
            if kw in pl:
                s += w
        try:
            size = os.path.getsize(p)
        except OSError:
            size = 0
        if size < 5_000_000:  # <5MB unlikely to be a full model state_dict
            s -= 100
        else:
            s += min(50, size // 50_000_000)  # mild preference for larger checkpoints
        for ch in ("epoch", "iter", "step"):
            if ch in base:
                s += 5
        return s

    if candidates:
        candidates.sort(key=lambda p: (score_path(p), os.path.getsize(p)), reverse=True)
        return candidates[0]

    return primary_path


resolved_model_path = resolve_checkpoint_path(model_path)
print("Resolved model checkpoint:", resolved_model_path)
print("Checkpoint exists:", os.path.exists(resolved_model_path))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



## === cell 3
has_ckpt = os.path.exists(resolved_model_path)

if has_ckpt:
    model = models.resnext50_32x4d(weights=None)
else:
    try:
        weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
        model = models.resnext50_32x4d(weights=weights)
    except Exception:
        model = models.resnext50_32x4d(pretrained=True)

model.fc = nn.Linear(2048, 5)
model.to(device)

if has_ckpt:
    ckpt = torch.load(resolved_model_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_state[nk] = v
        state = new_state

    try:
        model.load_state_dict(state, strict=True)
        print("Loaded checkpoint with strict=True")
    except RuntimeError as e:
        print(
            "Warning: strict=True load failed, retrying strict=False. Error:",
            str(e)[:300],
        )
        model.load_state_dict(state, strict=False)
        print("Loaded checkpoint with strict=False")
else:
    print(
        "No checkpoint found; using torchvision ImageNet-pretrained backbone with randomly initialized 5-class head."
    )

model.eval()



## === cell 4
sub_aug = albumentations.Compose(
    [
        albumentations.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0)),
        albumentations.Transpose(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)



## === cell 5
if not os.path.exists(sample_sub_path):
    alt = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
    if os.path.exists(alt):
        sample_sub_path = alt

if not os.path.isdir(test_images_path):
    alt_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
    if os.path.isdir(alt_dir):
        test_images_path = alt_dir

print("sample_sub_path:", sample_sub_path, "exists:", os.path.exists(sample_sub_path))
print("test_images_path:", test_images_path, "exists:", os.path.isdir(test_images_path))



## === cell 6
sample_sub = pd.read_csv(sample_sub_path)
tta_count = 10

predictions = []


def hwc_float_to_chw_tensor(x_hwc: np.ndarray) -> torch.Tensor:
    x = np.ascontiguousarray(x_hwc.transpose(2, 0, 1))  # CHW
    return torch.from_numpy(x).float()


with torch.no_grad():
    for _, sample_row in sample_sub.iterrows():
        image_pred = None
        img_path = os.path.join(test_images_path, sample_row.image_id)

        image_np = np.array(Image.open(img_path).convert("RGB"))

        for _ in range(tta_count):
            aug_img = sub_aug(image=image_np)["image"]  # HWC float32 normalized
            image_t = hwc_float_to_chw_tensor(aug_img).to(device)

            outputs = model(image_t.unsqueeze(0))  # [1,5]
            if image_pred is None:
                image_pred = outputs
            else:
                image_pred = image_pred + outputs

        image_pred = image_pred / tta_count
        pred_label = int(torch.argmax(image_pred, dim=1).item())
        predictions.append([sample_row.image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with rows:", len(sub_df))
print("submission.csv exists:", os.path.exists("submission.csv"))
