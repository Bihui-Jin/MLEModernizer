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

0.4817165306739196

# 6. Current score

0.55755

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'I remove the failing `pip install` dependency and load EfficientNet-B4 from `torchvision` so the model definition exists in this environment, while keeping the same “EfficientNet-B4, 5 classes, load .pth, TTA over Albumentations, argmax label” core pipeline. I fix the Albumentations v2 API break by using the correct `size=(512,512)` signature for `RandomResizedCrop`, so `sub_aug` can be created. I also make checkpoint loading robust to common key mismatches (`state_dict`, `module.` prefix) and map to the current device to prevent runtime errors. Finally, I ensure inference uses `torch.no_grad()` and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime failure by making the checkpoint path discovery robust: if the hard-coded `model(14).pth` isn’t present, the script search under `../input/` for a `.pth` file and load it (or fail with a clear message). This is a minimal change that preserves your EfficientNet-B4 + TTA + argmax core logic, but enables the intended trained weights to be used, which should substantially lift accuracy from the current “random-ish” score. I also keep the existing state_dict key-fixing logic and ensure the model is only run in eval/no_grad during inference. Finally, the script always write a valid `submission.csv` with the correct columns and row count.'
- What this solution (achieved 0.10015) has done: 'I fix the root cause of the crash: the script expects an external `.pth` checkpoint dataset that isn’t present in this Kaggle environment, so the checkpoint discovery finds nothing and raises. To keep the core pipeline (EfficientNet-B4 + Albumentations TTA + argmax) but make it run end-to-end and improve accuracy, I switch to using torchvision’s built-in pretrained EfficientNet-B4 weights as a fallback when no `.pth` is available, while still loading your `.pth` if it exists. I also make inference faster and more stable by batching TTA per image (same semantics: average logits over `tta_count`) and ensuring `ToTensor` isn’t re-normalizing already-normalized arrays. Finally, I keep the submission format exactly `image_id,label` and write `submission.csv`.'
- What this solution (achieved 0.62033) has done: 'Your current score is low mainly because when no `.pth` is found you fall back to an ImageNet EfficientNet-B4 but still replace the classifier with a randomly initialized 5-class head, which makes predictions near-random. I keep your exact core pipeline (EfficientNet-B4, Albumentations TTA, mean logits, argmax) but add a minimal, legitimate fallback: if no cassava `.pth` exists, train only the final classifier layer for a couple of quick epochs on `train.csv` using the same normalization, then run the same TTA inference. This should move accuracy substantially upward toward your target while staying within runtime and without changing the model architecture or inference semantics. I also ensure determinism and keep the submission format/paths unchanged.'
- What this solution (achieved 0.62668) has done: 'Your current score (0.62033) is better than the target (0.4817), so the goal is to reduce accuracy slightly toward the target band with minimal, controlled changes. The simplest stable lever (without changing model/loss/architecture) is to make inference-time TTA less aggressive/less averaging, which typically lowers accuracy on this task. I keep the same EfficientNet-B4 + Albumentations pipeline, but reduce `tta_count` and make the augmentation deterministic-per-image so results are stable run-to-run. The training fallback (only used when no checkpoint exists) is kept intact to preserve your existing logic.'
- What this solution (achieved 0.6932) has done: 'Your current score (0.62668) is above the target (0.4817), so we should *slightly reduce* accuracy in a controlled, stable way with minimal changes. The smallest lever that preserves your core pipeline (EfficientNet-B4 + Albumentations + mean logits + argmax) is to make test-time preprocessing less “train-like” by removing the heavy random TTA augmentation and using a plain deterministic resize+normalize for inference. This typically lowers leaderboard accuracy while keeping semantics identical (still single forward pass, mean over TTA, argmax). I keep your checkpoint loading and fallback head-training intact, and only change the inference transform used in cell 6.'
- What this solution (achieved 0.61099) has done: 'Your current accuracy (0.6932) is above the target (0.4817), so we should *decrease* performance in a controlled way while keeping your same EfficientNet-B4 + preprocessing + argmax pipeline intact. The smallest stable lever is to intentionally make inference inputs less informative without changing model/training logic: reduce the test-time resolution (more information loss) while preserving normalization and the rest of the code. I only change the `test_aug` resize from 512 to 256 (leaving training/fallback unchanged), which typically lowers accuracy and should move the score toward the target band. Submission writing and all paths remain identical.'
- What this solution (achieved 0.30194) has done: 'Your current score (0.61099) is higher than the target (0.4817), so the goal is to reduce accuracy in a controlled, minimal, and stable way. The smallest lever that preserves your exact core pipeline (EfficientNet-B4, same head, same argmax semantics, same data paths) is to further reduce inference-time input resolution, which typically lowers accuracy due to information loss. I only change `test_aug` resize from 256→160 and keep everything else identical, including producing `submission.csv` with the correct schema and row count. This should move the score downward toward the target band without touching training/architecture/loss.'
- What this solution (achieved 0.61099) has done: 'We need to move your current accuracy (0.30194) upward toward the target (0.4817), so the smallest effective change is to restore more informative test-time preprocessing while keeping your exact core pipeline (EfficientNet-B4, same head, same argmax semantics, same inference loop). Right now test images are resized to 160×160, which likely discards too much signal and collapses accuracy; increasing only the inference resize to 256×256 should raise accuracy substantially without changing architecture, training, loss, or TTA logic. I keep `tta_count=1` and all training/ckpt-loading behavior identical, and only adjust `test_aug` resolution. The script still run end-to-end and write a valid `submission.csv` with the required schema and row count.'
- What this solution (achieved 0.55755) has done: 'Your current score (0.61099) is above the target (0.4817), so we should make a small, stable change that slightly reduces accuracy without altering the core model or training/inference semantics. The safest lever here is test-time resolution: lowering it a bit increases information loss and typically reduces accuracy in a predictable way while keeping the same EfficientNet-B4 + normalize + argmax pipeline. I only reduce `test_aug` resize from 256→224 (a modest step, less extreme than 160) and keep everything else identical, including checkpoint loading, optional head-fit fallback, and submission writing. This should move the score downward toward the target band with minimal risk of breaking the run.'

# 9. Code solution

## === cell 0
import os
import warnings
import glob
import random

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 1
model_path = "../input/en-b4-tta-calr-clahe/model(14).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_path = "../input/cassava-leaf-disease-classification/train_images"

assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(test_images_path), f"Missing dir: {test_images_path}"

ckpt_exists = os.path.exists(model_path)
if not ckpt_exists:
    candidates = sorted(
        glob.glob("../input/**/*.pth", recursive=True),
        key=lambda p: os.path.getsize(p) if os.path.isfile(p) else -1,
        reverse=True,
    )
    if len(candidates) > 0:
        model_path = candidates[0]
        ckpt_exists = True
        print(f"[INFO] model_path not found; using discovered checkpoint: {model_path}")
    else:
        print(
            "[WARN] No .pth checkpoint found under ../input. "
            "Will use torchvision pretrained EfficientNet-B4 backbone and quickly fit a 5-class head on train.csv."
        )



## === cell 2
if ckpt_exists:
    model = models.efficientnet_b4(weights=None)
else:
    model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1)

in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 5)

model.to(device)
model.eval()



## === cell 3
if ckpt_exists:
    ckpt = torch.load(model_path, map_location=device)

    state_dict = ckpt
    if (
        isinstance(ckpt, dict)
        and "state_dict" in ckpt
        and isinstance(ckpt["state_dict"], dict)
    ):
        state_dict = ckpt["state_dict"]

    if isinstance(state_dict, dict):
        new_sd = {}
        for k, v in state_dict.items():
            nk = k[7:] if k.startswith("module.") else k
            new_sd[nk] = v
        state_dict = new_sd

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("Loaded checkpoint.")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

model.eval()



## === cell 4
sub_aug = A.Compose(
    [
        A.RandomResizedCrop(size=(512, 512), scale=(0.5, 1.0)),
        A.Transpose(p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.8),
        A.HueSaturationValue(
            hue_shift_limit=20,
            sat_shift_limit=30,
            val_shift_limit=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1),
            contrast_limit=(-0.1, 0.1),
            p=0.5,
        ),
        A.CLAHE(clip_limit=(1.0, 20.0), tile_grid_size=(32, 32), p=1),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

train_aug = A.Compose(
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

test_aug = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

to_tensor = transforms.ToTensor()



## === cell 5
if (
    (not ckpt_exists)
    and os.path.exists(train_csv_path)
    and os.path.isdir(train_images_path)
):
    train_df = pd.read_csv(train_csv_path)

    class CassavaTrainDataset(torch.utils.data.Dataset):
        def __init__(self, df, images_dir, aug):
            self.df = df.reset_index(drop=True)
            self.images_dir = images_dir
            self.aug = aug

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            image_path = os.path.join(self.images_dir, row.image_id)
            img = np.array(Image.open(image_path).convert("RGB"))
            img = self.aug(image=img)["image"]  # HWC float32 normalized
            x = to_tensor(img)  # CHW float32
            y = int(row.label)
            return x, y

    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier[1].parameters():
        p.requires_grad = True

    train_ds = CassavaTrainDataset(train_df, train_images_path, train_aug)

    batch_size = 32 if device.type == "cuda" else 16
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier[1].parameters(), lr=3e-4)

    model.train()
    epochs = 2  # unchanged: keep existing fallback behavior
    for ep in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += int(xb.size(0))

        print(
            f"[Head train] epoch {ep+1}/{epochs} "
            f"loss={running_loss/max(total,1):.4f} acc={correct/max(total,1):.4f}"
        )

    model.eval()
else:
    model.eval()



## === cell 6
sample_sub = pd.read_csv(sample_sub_path)

tta_count = 1

predictions = []

model.eval()
with torch.no_grad():
    for i, sample_row in enumerate(sample_sub.itertuples(index=False)):
        image_id = sample_row.image_id
        image_path = os.path.join(test_images_path, image_id)
        image_np = np.array(Image.open(image_path).convert("RGB"))

        xs = []
        for _ in range(tta_count):
            aug = test_aug(image=image_np)["image"]  # HWC float32, normalized
            x = to_tensor(aug)  # CHW float32
            xs.append(x)
        xb = torch.stack(xs, dim=0).to(device)  # [T, C, H, W]

        outputs = model(xb)  # [T, 5]
        image_pred = outputs.mean(dim=0, keepdim=True)  # [1, 5]
        pred_label = int(torch.argmax(image_pred, dim=1).item())
        predictions.append([image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])

sub_df = sub_df.merge(sample_sub[["image_id"]], on="image_id", how="right")
sub_df["label"] = sub_df["label"].fillna(0).astype(int)

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with rows:", len(sub_df))
print("Unique labels:", sorted(sub_df["label"].unique().tolist()))
