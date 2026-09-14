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

0.15658

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10501) has done: 'I fix the missing `efficientnet_pytorch` dependency by removing the broken pip-install cell and switching to `torchvision`’s built-in `efficientnet_b4` while keeping the same core “EfficientNet-B4, 5 classes, load weights, TTA loop, argmax label” approach. I also update the Albumentations API usage to the v2.x signature (tuple `size`) and replace removed transforms (`Cutout`) with `CoarseDropout` so augmentation construction no longer crashes. Finally, I fix inference bugs that prevented correct TTA averaging (divide-by-10 typo) and ensure tensors are normalized and shaped correctly, so the notebook runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.30605) has done: 'I fix the immediate runtime error by making the checkpoint path robust: the code search `../input` for a `.pth`/`.pt` file and load it if the originally hardcoded dataset path isn’t present. To move accuracy toward your target, I also fix a core normalization bug: after Albumentations `Normalize`, the image is already float CHW-ready and must not be passed through `ToTensor()` (which rescales again and destroys calibration), so I convert HWC float to a Torch tensor directly. Finally, I keep your same EfficientNet-B4 + TTA + argmax logic, but apply `softmax` before averaging across TTA (probability averaging is standard for accuracy and is a minimal post-processing calibration fix), and ensure the submission is written as `submission.csv` with correct columns.'
- What this solution (achieved 0.28438) has done: 'The run is failing because the script cannot find any trained checkpoint under `../input`, so `torch.load()` never gets a valid path. I make checkpoint discovery robust across the provided filesystem (`/kaggle/data`, `/kaggle/input`, `../input`) and, if no checkpoint exists, fall back to torchvision’s ImageNet pretrained EfficientNet-B4 weights (same architecture/head) so inference can still run end-to-end and the score moves upward toward your target. I also fix the Python version mismatch issue by avoiding `os.path.exists("../input/...")` assumptions and by ensuring the chosen checkpoint path is actually loadable before calling `torch.load()`. Core logic (EfficientNet-B4, 5-class head, Albumentations TTA loop, softmax-then-average, argmax) is preserved.'
- What this solution (achieved 0.15658) has done: 'Your current score is low because you’re effectively doing “random-head” inference when no cassava checkpoint is found, and the TTA pipeline uses heavy train-time augmentations that can hurt test-time accuracy. I keep your EfficientNet‑B4 + 5‑class head + TTA + softmax-mean + argmax core logic, but (1) make checkpoint discovery also look inside the cassava dataset directory for any provided `.pth/.pt` and (2) switch TTA transforms to a standard *test-time* set (resize/center-crop + mild flips/transpose), which typically moves accuracy up substantially without changing the model/training logic. I also speed up and stabilize inference by using a DataLoader (same semantics, just batched I/O) and enabling cuDNN benchmark for fixed-size inputs. These are minimal, directly score-relevant changes aimed at moving you closer to the 0.866 target.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings

import albumentations as A
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

torch.backends.cudnn.benchmark = True




## === cell 1
model_path = "../input/en-b4-tta-calr-8/model(12).pth"

DATA_ROOT_CANDIDATES = [
    "../input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/input",
    "/kaggle/data",
]


def first_existing_dir(paths):
    for p in paths:
        if p and os.path.isdir(p):
            return p
    return None


cassava_root = first_existing_dir(DATA_ROOT_CANDIDATES)
if cassava_root is None:
    raise FileNotFoundError(
        "Could not find cassava dataset directory under expected locations: "
        + ", ".join(DATA_ROOT_CANDIDATES)
    )

sample_sub_path = os.path.join(cassava_root, "sample_submission.csv")
test_images_path = os.path.join(cassava_root, "test_images")

assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(test_images_path), f"Missing dir: {test_images_path}"


def discover_checkpoint(preferred_path: str, extra_search_dirs=None):
    """
    Change is score-relevant: many low scores come from failing to load the intended cassava-trained weights.
    We broaden search to include the dataset directory itself and typical Kaggle mounts.
    """
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    search_roots = ["../input", "/kaggle/input", "/kaggle/data", "/kaggle/working"]
    if extra_search_dirs:
        for d in extra_search_dirs:
            if d and d not in search_roots:
                search_roots.append(d)

    candidates = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for ext in ("*.pth", "*.pt", "*.bin", "*.ckpt"):
            candidates.extend(glob.glob(os.path.join(root, "**", ext), recursive=True))

    preferred = []
    for p in candidates:
        bn = os.path.basename(p).lower()
        if any(
            tok in bn
            for tok in [
                "efficientnet",
                "enb4",
                "en-b4",
                "cassava",
                "leaf",
                "model",
                "checkpoint",
                "ckpt",
            ]
        ):
            preferred.append(p)

    candidates = preferred if preferred else candidates
    candidates = sorted(set(candidates))

    filtered = []
    for p in candidates:
        try:
            if os.path.getsize(p) >= 1_000_000:  # >= ~1MB
                filtered.append(p)
        except OSError:
            continue

    return filtered[0] if filtered else None


found_ckpt = discover_checkpoint(model_path, extra_search_dirs=[cassava_root])
if found_ckpt is None:
    print(
        "No checkpoint found under Kaggle mounts. Will use torchvision pretrained EfficientNet-B4 weights."
    )
else:
    model_path = found_ckpt
    print("Using model checkpoint:", model_path)




## === cell 2
use_pretrained_backbone = found_ckpt is None

model = models.efficientnet_b4(
    weights=(
        models.EfficientNet_B4_Weights.IMAGENET1K_V1
        if use_pretrained_backbone
        else None
    )
)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, 5)
model = model.to(DEVICE)

if found_ckpt is not None:
    state = torch.load(model_path, map_location=DEVICE)
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k.replace("module.", "")
            new_state[nk] = v
        state = new_state

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(
        "Loaded checkpoint. Missing keys:",
        len(missing),
        "Unexpected keys:",
        len(unexpected),
    )
else:
    print(
        "Initialized model with ImageNet pretrained weights (head is randomly initialized for 5 classes)."
    )

model.eval()




## === cell 3
IMG_SIZE = 380  # EfficientNet-B4 default resolution

sub_aug = A.Compose(
    [
        A.LongestMaxSize(max_size=IMG_SIZE, p=1.0),
        A.PadIfNeeded(
            min_height=IMG_SIZE, min_width=IMG_SIZE, border_mode=0, value=0, p=1.0
        ),
        A.CenterCrop(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)




## === cell 4
def load_rgb_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def albumentations_to_torch_chw(img_hwc: np.ndarray) -> torch.Tensor:
    if img_hwc.dtype != np.float32:
        img_hwc = img_hwc.astype(np.float32)
    img_chw = np.transpose(img_hwc, (2, 0, 1))
    return torch.from_numpy(img_chw)




## === cell 5
from torch.utils.data import Dataset, DataLoader

sample_sub = pd.read_csv(sample_sub_path)
assert set(sample_sub.columns) >= {
    "image_id",
    "label",
}, "Unexpected sample submission columns"


class CassavaTestDataset(Dataset):
    def __init__(self, image_ids, images_dir):
        self.image_ids = list(image_ids)
        self.images_dir = images_dir

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, image_id)
        image_np = load_rgb_image(img_path)
        return image_id, image_np


def collate_keep_numpy(batch):
    image_ids = [b[0] for b in batch]
    images = [b[1] for b in batch]  # list of HWC uint8
    return image_ids, images


batch_size = 16 if DEVICE.type == "cuda" else 4
loader = DataLoader(
    CassavaTestDataset(sample_sub["image_id"].values, test_images_path),
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=(DEVICE.type == "cuda"),
    collate_fn=collate_keep_numpy,
)

pred_rows = []
tta_n = 5  # preserve original loop count

with torch.no_grad():
    for image_ids, images_hwc in loader:
        prob_sum = None

        for _ in range(tta_n):
            batch_t = []
            for img in images_hwc:
                aug_np = sub_aug(image=img)["image"]  # float32 HWC normalized
                batch_t.append(albumentations_to_torch_chw(aug_np))
            batch_t = torch.stack(batch_t, dim=0).to(DEVICE)  # [B,3,H,W]

            logits = model(batch_t)  # [B,5]
            probs = torch.softmax(logits, dim=1)
            prob_sum = probs if prob_sum is None else (prob_sum + probs)

        prob_mean = prob_sum / float(tta_n)
        pred_labels = (
            torch.argmax(prob_mean, dim=1).detach().cpu().numpy().astype(int).tolist()
        )

        pred_rows.extend(list(zip(image_ids, pred_labels)))

sub_df = pd.DataFrame(pred_rows, columns=["image_id", "label"])
sub_df = sample_sub[["image_id"]].merge(sub_df, on="image_id", how="left")
assert sub_df["label"].isna().sum() == 0, "Some predictions are missing after merge."

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("Label distribution:\n", sub_df["label"].value_counts().sort_index())
