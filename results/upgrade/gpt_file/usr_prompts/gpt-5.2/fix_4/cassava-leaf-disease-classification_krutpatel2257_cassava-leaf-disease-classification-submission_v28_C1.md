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

0.8884859474161378

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I remove the failing external `efficientnet_pytorch` dependency and keep the existing RESNEXT50 inference path, since your config already uses `RESNET_50`. I also make model weight loading robust by automatically locating a `.pth` inside the Kaggle input directory (or cleanly proceeding with random weights if none exist), so the notebook always runs end-to-end and writes `submission.csv`. Albumentations v2 changed `RandomResizedCrop`’s signature, so I update it to the new `size=(H,W)` form and replace deprecated/removed transforms (`Cutout`) with safe equivalents to prevent runtime errors. Finally, I fix inference to use `torch.no_grad()`, correct device/dtype handling, and ensure the submission columns match `sample_submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your very low score is mainly because the inference pipeline applies heavy *training-style* random augmentations (crop/rotate/color/dropout) to the test set, which destroys signal and makes predictions unstable; for accuracy metric you typically want deterministic resizing/normalization and, if using TTA, only mild test-time flips. I keep your model and checkpoint-loading logic intact, but change the test transform to a deterministic `Resize`+`CenterCrop` (and optional mild flip TTA) so predictions match the distribution the model likely expects. I also switch the per-image loop to a DataLoader with a fixed order to reduce overhead and ensure correct alignment with `sample_submission.csv` without changing evaluation semantics. These are minimal changes aimed specifically at increasing accuracy toward your 0.888 target without altering the core model logic.'
- What this solution (achieved 0.61099) has done: 'Your score is far below the target, so we should increase accuracy with the smallest changes that don’t alter the core model or training loop (there is no training here). The main likely cause is a preprocessing mismatch: the checkpoint was probably trained with ImageNet-style resizing to a smaller input (commonly 224/256/384), while the current inference uses 512 + center crop, which can significantly hurt accuracy. I keep the same model, checkpoint-loading, and TTA logic, but change the test preprocessing to `Resize(256) + CenterCrop(224)` (standard for ResNe(X)t ImageNet normalization) to better match typical training, and I also ensure deterministic CPU threading and DataLoader worker seeding for stable results. These are minimal, directly score-relevant adjustments that should move accuracy substantially toward your 0.888 target.'

# 9. Code solution

## === cell 0
import os
import glob
import random

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models



## === cell 1
config = {
    "DATA": {
        "IMAGES": "train_images",
        "LABELS": "train.csv",
        "SUB_IMAGES": "test_images",
        "SUB_LABELS": "sample_submission.csv",
        "SUB_OUTPUT": "submission.csv",
    },
    "DEVICE": "cuda",
    "NUM_GPU": torch.cuda.device_count(),
    "TRAIN_BATCH_SIZE": 32,
    "VAL_BATCH_SIZE": 16,
    "CLASSES": 5,
    "CV_FOLDS": 5,
    "NUM_EPOCHS": 15,
    "MODEL_PATH": "model.pth",
    "SGD": {"LR": 0.0005, "MOMENTUM": 0.9, "WEIGHT_DECAY": 0.001},
    "COS_ANN_LR": {"ETA_MIN": 0.00001},
    "MODEL_TYPE": "RESNET_50",
}



## === cell 2
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)



## === cell 3
model_path = "../input/en-b4-tta-calr-clahe-v2-12-14/model(18).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

if not os.path.exists(sample_sub_path):
    sample_sub_path = (
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
    )
if not os.path.exists(test_images_path):
    test_images_path = "/kaggle/input/cassava-leaf-disease-classification/test_images"


def find_any_pth(start_dir="/kaggle/input"):
    candidates = glob.glob(os.path.join(start_dir, "**", "*.pth"), recursive=True)
    candidates_sorted = sorted(
        candidates,
        key=lambda p: (("model" not in os.path.basename(p).lower()), len(p)),
    )
    return candidates_sorted[0] if candidates_sorted else None


resolved_model_path = (
    model_path if os.path.exists(model_path) else find_any_pth("/kaggle/input")
)
print("Resolved model path:", resolved_model_path)



## === cell 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



## === cell 5
if config["MODEL_TYPE"] == "RESNET_50":
    model = models.resnext50_32x4d(weights=None)
    model.fc = nn.Linear(2048, config["CLASSES"])
else:
    model = models.resnext50_32x4d(weights=None)
    model.fc = nn.Linear(2048, config["CLASSES"])

model.to(device)

if resolved_model_path is not None and os.path.exists(resolved_model_path):
    ckpt = torch.load(resolved_model_path, map_location=device)
    if (
        isinstance(ckpt, dict)
        and "state_dict" in ckpt
        and isinstance(ckpt["state_dict"], dict)
    ):
        state = ckpt["state_dict"]
        state = {
            k.replace("model.", "").replace("module.", ""): v for k, v in state.items()
        }
        model.load_state_dict(state, strict=False)
        print(
            "Loaded checkpoint (state_dict) with strict=False from:",
            resolved_model_path,
        )
    elif isinstance(ckpt, dict):
        state = {
            k.replace("model.", "").replace("module.", ""): v for k, v in ckpt.items()
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        print("Loaded checkpoint (dict) strict=False from:", resolved_model_path)
        if missing:
            print("Missing keys (first 10):", missing[:10])
        if unexpected:
            print("Unexpected keys (first 10):", unexpected[:10])
    else:
        print("Checkpoint format not recognized; proceeding without loading weights.")
else:
    print(
        "No .pth weights found; proceeding with random initialized weights (submission will still be generated)."
    )

model.eval()



## === cell 6
CROP_SIZE = 224
RESIZE_SHORT = 256

sub_aug_base = A.Compose(
    [
        A.SmallestMaxSize(max_size=RESIZE_SHORT, interpolation=1, p=1.0),
        A.CenterCrop(height=CROP_SIZE, width=CROP_SIZE, p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)

sub_aug_hflip = A.Compose(
    [
        A.SmallestMaxSize(max_size=RESIZE_SHORT, interpolation=1, p=1.0),
        A.CenterCrop(height=CROP_SIZE, width=CROP_SIZE, p=1.0),
        A.HorizontalFlip(p=1.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)


def chw_tensor_from_normalized_hwc(img_hwc_float32: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.transpose(img_hwc_float32, (2, 0, 1))).float()




## === cell 7
assert os.path.exists(
    sample_sub_path
), f"sample_submission.csv not found at {sample_sub_path}"
assert os.path.isdir(
    test_images_path
), f"test_images dir not found at {test_images_path}"

sample_sub = pd.read_csv(sample_sub_path)
assert (
    "image_id" in sample_sub.columns and "label" in sample_sub.columns
), "Submission must have columns image_id,label"
print(sample_sub.head())
print("Test images:", len(sample_sub))




## === cell 8
class CassavaTestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: str):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        image_id = self.df.loc[idx, "image_id"]
        img_path = os.path.join(self.image_dir, image_id)
        image = np.array(Image.open(img_path).convert("RGB"))

        img0 = sub_aug_base(image=image)["image"]
        img1 = sub_aug_hflip(image=image)["image"]

        x0 = chw_tensor_from_normalized_hwc(img0)
        x1 = chw_tensor_from_normalized_hwc(img1)
        return image_id, x0, x1


test_ds = CassavaTestDataset(sample_sub, test_images_path)
test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
    worker_init_fn=seed_worker,
    generator=g,
)



## === cell 9
predictions = []
model.eval()

with torch.no_grad():
    for image_ids, x0, x1 in test_loader:
        x0 = x0.to(device, non_blocking=True)
        x1 = x1.to(device, non_blocking=True)

        out0 = model(x0)
        out1 = model(x1)
        out = (out0 + out1) / 2.0

        pred = torch.argmax(out, dim=1).detach().cpu().numpy().astype(int).tolist()
        predictions.extend(list(zip(image_ids, pred)))

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])

sub_df = sample_sub[["image_id"]].merge(sub_df, on="image_id", how="left")
assert sub_df["label"].isna().sum() == 0, "Some predictions are missing after merge."



## === cell 10
out_path = config["DATA"]["SUB_OUTPUT"]
sub_df.to_csv(out_path, index=False)

print("Saved:", out_path)
print(sub_df.head())
print("Rows:", len(sub_df), "Cols:", list(sub_df.columns))
assert out_path.endswith(".csv") and os.path.exists(out_path)
assert len(sub_df) == len(sample_sub)
assert list(sub_df.columns) == ["image_id", "label"]
