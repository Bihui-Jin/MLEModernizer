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

geopandas==0.14.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.8700513750377757

# 6. Current score

0.61211

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'Your notebook fails because it depends on missing external packages/configs (`utils`, `model`, custom config module, and `build_transforms`) that are not available in this Kaggle environment. To make it run end-to-end with minimal semantic changes (still: load images → apply standard val transforms → run a CNN → argmax to labels), I replace those missing pieces with a torchvision ResNet50 inference pipeline and ensure test images are read from the correct available path. I also fix the submission-length error by generating predictions for *exactly* the `image_id`s in `sample_submission.csv` (correct order, no missing/extra rows). Finally, the script always writes a valid `submission.csv` in the working directory.'
- What this solution (achieved 0.1065) has done: 'Your current score is near-random because the model is a randomly initialized ResNet50 (`weights=None`) with no learned weights. The smallest legitimate change that preserves your core logic (single ResNet50 forward pass + argmax) is to switch to ImageNet-pretrained ResNet50 weights and keep the same normalization/center-crop inference pipeline. This should move accuracy substantially toward your target without changing architecture, loss, training loops (none), or submission semantics. I also keep the submission aligned exactly to `sample_submission.csv` order and write `submission.csv` as before.'
- What this solution (achieved 0.40882) has done: 'Your score is near-random because the model’s final classification head is randomly initialized (you replace the pretrained ImageNet `fc` with a new 5-class `Linear`), so even with ImageNet weights the logits are essentially random for the 5 labels. The smallest change that preserves the core inference logic (single ResNet50 forward pass + argmax) is to keep the pretrained backbone but swap the head to a zero-shot ImageNet-class mapping: use the original 1000-way `fc`, then map ImageNet predictions to the 5 cassava labels using a fixed, documented class-to-label mapping. This is still pure inference (no training loops added) and should move accuracy substantially toward your target. I also keep submission ordering exactly aligned with `sample_submission.csv` and continue writing `./submission.csv`.'
- What this solution (achieved 0.61958) has done: 'Your current gap to the target is large (0.40882 → 0.87005), and the main issue is that the “ImageNet class → cassava label” mapping is essentially arbitrary, so predictions are mostly noise. With minimal changes and keeping your core “single pretrained ResNet50 forward pass + argmax + deterministic mapping + CSV” logic, I replace the tiny hardcoded mapping with a frequency-calibrated mapping learned from `train.csv` using the same pretrained ResNet50 (no training, no loss/optimizer). This uses the training set only to estimate a stable mapping from ImageNet-argmax IDs to cassava labels (majority vote with sensible fallbacks), which should move accuracy substantially toward your target. I also keep submission ordering exactly aligned to `sample_submission.csv` and continue writing `./submission.csv`.'
- What this solution (achieved 0.61136) has done: 'Your current approach is capped because it forces cassava labels to be a deterministic function of the ImageNet argmax class; we can improve toward the target without changing the core “single pretrained ResNet50 forward pass + mapping + argmax + CSV” logic by (1) learning the mapping from a larger and more representative slice of `train.csv`, (2) using confidence-gated fallbacks so low-confidence ImageNet predictions don’t inject noise, and (3) making the mapping itself confidence-aware (only trust ImageNet IDs that are consistently mapped). These changes keep the same model/architecture and still do no training, but they should move accuracy upward from 0.61958 toward your 0.87005 target while staying within the runtime budget. I also keep submission ordering exactly matching `sample_submission.csv` and continue writing `./submission.csv`.'
- What this solution (achieved 0.61211) has done: 'Your current score (0.61136) is far below the target (0.87005), and the main limiter is that the ImageNet→cassava mapping is learned from only 12k train images with a fairly strict confidence gate, which leaves many ImageNet IDs unmapped (defaulting to the global prior) and reduces useful signal. To move accuracy upward while preserving the exact core logic (pretrained ResNet50 forward pass → ImageNet argmax → deterministic mapping → cassava label / fallback), I (1) learn the mapping from the full training set (still no training, just forward passes), (2) use a top-1/top-2 “margin” gate in addition to probability to reduce noisy votes without throwing away too much data, and (3) use a slightly lower fallback threshold at inference to rely on the mapping more often. These are minimal, metric-aligned changes that should improve the mapping quality and push the score toward your target without changing architecture, loss, or adding any training loop. The submission generation/order remains exactly aligned to `sample_submission.csv` and writes `./submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import copy
import datetime
import random

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

import cv2
import torchvision
import torchvision.transforms as T


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

assert os.path.exists(TEST_IMG_DIR), f"Missing test_images dir at {TEST_IMG_DIR}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv at {SAMPLE_SUB_PATH}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing train_images dir at {TRAIN_IMG_DIR}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv at {TRAIN_CSV_PATH}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
model = torchvision.models.resnet50(weights=weights)

model = model.to(device)
model.eval()

weights_path = None
if isinstance(weights_path, str) and os.path.exists(weights_path):
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print(
        "Loaded weights. Missing keys:",
        len(missing),
        "Unexpected keys:",
        len(unexpected),
    )

FALLBACK_LABEL = 3



## === cell 2
val_tfms = T.Compose(
    [
        T.ToPILImage(),
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class InferSet(Dataset):
    def __init__(self, img_dir: str, image_ids, transforms, labels=None):
        self.img_dir = img_dir
        self.image_ids = list(image_ids)
        self.transforms = transforms
        self.labels = None if labels is None else list(labels)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        fn = self.image_ids[idx]
        path = os.path.join(self.img_dir, fn)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(img)
        if self.labels is None:
            return img, fn
        return img, fn, int(self.labels[idx])


def build_imagenet_to_cassava_mapping(
    model,
    train_img_dir: str,
    train_csv_path: str,
    transforms,
    device,
    batch_size: int = 64,
    num_workers: int = 2,
    max_items: int = None,
    min_prob_for_count: float = 0.25,
    min_margin_for_count: float = 0.05,
    min_purity: float = 0.52,
    min_support: int = 3,
):
    """
    Estimate a robust mapping from ImageNet argmax class IDs -> cassava labels using the training set.
    No training is performed; we only run forward passes.

    Changes to move score toward target (still same core logic: ResNet50 -> ImageNet argmax -> mapping):
      - Use (by default) the full training set for a more representative mapping table (max_items=None).
      - Add a small top1-top2 margin gate to reduce noisy votes without discarding too much data.
      - Slightly relax the counting thresholds so more ImageNet IDs get usable evidence (less global-prior fallback).
    """
    df = pd.read_csv(train_csv_path)
    df = df.sort_values("image_id").reset_index(drop=True)
    if max_items is not None and len(df) > max_items:
        df = df.iloc[:max_items].copy()

    ds = InferSet(
        train_img_dir, df["image_id"].tolist(), transforms, labels=df["label"].tolist()
    )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    counts = np.zeros((1000, 5), dtype=np.int64)

    with torch.no_grad():
        model.eval()
        for images, fns, y in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)  # [B, 1000]
            probs = torch.softmax(logits, dim=1)

            top2_prob, top2_idx = probs.topk(2, dim=1)  # [B,2]
            conf = top2_prob[:, 0]
            margin = top2_prob[:, 0] - top2_prob[:, 1]
            pred_imnet = top2_idx[:, 0]

            pred_imnet = pred_imnet.detach().cpu().numpy().astype(int)
            conf = conf.detach().cpu().numpy().astype(np.float32)
            margin = margin.detach().cpu().numpy().astype(np.float32)
            y = np.asarray(y, dtype=int)

            for k, c, m, lbl in zip(pred_imnet, conf, margin, y):
                if c < float(min_prob_for_count):
                    continue
                if m < float(min_margin_for_count):
                    continue
                if 0 <= k < 1000 and 0 <= lbl < 5:
                    counts[k, lbl] += 1

    global_prior = int(df["label"].value_counts().idxmax())

    mapping = {k: global_prior for k in range(1000)}
    observed = counts.sum(axis=1)

    for k in range(1000):
        total = int(observed[k])
        if total <= 0:
            mapping[k] = global_prior
            continue

        best_lbl = int(np.argmax(counts[k]))
        best_cnt = int(counts[k, best_lbl])
        purity = best_cnt / max(1, total)

        if total < int(min_support) or purity < float(min_purity):
            mapping[k] = global_prior
        else:
            mapping[k] = best_lbl

    return mapping, global_prior


sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_ids = sample_sub["image_id"].tolist()

IMAGENET_TO_CASSAVA, FALLBACK_LABEL = build_imagenet_to_cassava_mapping(
    model=model,
    train_img_dir=TRAIN_IMG_DIR,
    train_csv_path=TRAIN_CSV_PATH,
    transforms=val_tfms,
    device=device,
    batch_size=64,
    num_workers=2,
    max_items=None,
    min_prob_for_count=0.25,
    min_margin_for_count=0.05,
    min_purity=0.52,
    min_support=3,
)

infer_ds = InferSet(TEST_IMG_DIR, test_ids, val_tfms)
infer_loader = DataLoader(
    infer_ds,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

len(infer_ds), infer_loader.__iter__().__next__()[0].shape



## === cell 3
MIN_PROB_FOR_PRED = 0.25
MIN_MARGIN_FOR_PRED = 0.03

pred_map = {}

with torch.no_grad():
    model.eval()
    for images, fns in infer_loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)  # [B, 1000]
        probs = torch.softmax(logits, dim=1)

        top2_prob, top2_idx = probs.topk(2, dim=1)
        conf = top2_prob[:, 0]
        margin = top2_prob[:, 0] - top2_prob[:, 1]
        imagenet_preds = top2_idx[:, 0]

        imagenet_preds = imagenet_preds.detach().cpu().numpy().astype(int)
        conf = conf.detach().cpu().numpy().astype(np.float32)
        margin = margin.detach().cpu().numpy().astype(np.float32)

        mapped = []
        for k, c, m in zip(imagenet_preds, conf, margin):
            if float(c) < float(MIN_PROB_FOR_PRED) or float(m) < float(
                MIN_MARGIN_FOR_PRED
            ):
                mapped.append(int(FALLBACK_LABEL))
            else:
                mapped.append(int(IMAGENET_TO_CASSAVA.get(int(k), int(FALLBACK_LABEL))))

        for fn, p in zip(fns, mapped):
            pred_map[fn] = int(p)

preds_ordered = [pred_map[fn] for fn in test_ids]
sub = pd.DataFrame({"image_id": test_ids, "label": preds_ordered})

out_path = "./submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote:", out_path)
print("Rows:", len(sub), "Expected:", len(sample_sub))
print(sub.head())



## === cell 4
assert os.path.exists("./submission.csv"), "submission.csv was not created"
check = pd.read_csv("./submission.csv")
assert list(check.columns) == [
    "image_id",
    "label",
], f"Bad columns: {check.columns.tolist()}"
assert len(check) == len(sample_sub), f"Bad length: {len(check)} vs {len(sample_sub)}"
assert check["label"].between(0, 4).all(), "Labels must be in [0,4]"
check.tail()
