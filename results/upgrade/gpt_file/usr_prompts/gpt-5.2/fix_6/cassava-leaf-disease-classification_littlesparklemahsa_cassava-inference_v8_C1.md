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

3.14

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
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
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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

0.8909035962526443

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the crash by making device selection automatic (CPU fallback when no NVIDIA driver is available) and by only using CUDA autocast when CUDA is present. I also make model weight loading more robust by handling common checkpoint formats (`state_dict`, `model`, `module.` prefixes) so it can run end-to-end and actually write `submission.csv`. Finally, I add a safe fallback to generate a valid submission from `sample_submission.csv` if the external model files aren’t available in this environment, ensuring “Not yielded” becomes a valid `.csv` output without changing the intended inference logic when weights exist.'
- What this solution (achieved 0.11584) has done: 'Your very low score is consistent with the “no weights found → write sample_submission” fallback, which effectively submits all-4s and lands around ~0.1 accuracy. To move the score toward your target with minimal core-logic change, I (1) make the code auto-discover any `.pth` checkpoints available anywhere under `/kaggle/input` (while keeping your existing `CFG.model_paths` priority), and (2) add a safe CPU/GPU-agnostic test-time augmentation that’s already consistent with your current logic (horizontal flip) without changing the model/training approach. This keeps the same ConvNeXt-Tiny inference pipeline, but greatly increases the chance that real weights are loaded in this environment instead of falling back to the sample submission. The submission writing and schema remain unchanged.'
- What this solution (achieved 0.11584) has done: 'Your low score is consistent with either (1) still not actually loading the intended pretrained weights (so predictions are effectively random), or (2) a label mapping mismatch between the checkpoint’s class order and the competition’s `0..4` label IDs. I keep your exact ConvNeXt-Tiny inference + flip-TTA + fold-ensemble logic, but make checkpoint discovery prioritize likely Cassava solutions, then add a tiny “label permutation calibration” step that uses `train.csv` to infer the best class-index→label-id mapping from each checkpoint (no training, just a quick pass over a small subset). This is a minimal, metric-aligned fix that often turns ~0.1 accuracy into a much higher score when the model’s internal class order differs. The submission format, paths, and core inference remain the same; we only (a) ensure we load the right weights and (b) map predicted indices to the correct label IDs.'
- What this solution (achieved 0.11584) has done: 'Your score (0.11584) suggests the pipeline is still effectively producing near-random predictions, most commonly because the loaded checkpoint(s) don’t actually match `convnext_tiny` (so most weights stay randomly initialized under `strict=False`) and/or because inference is being done with mismatched preprocessing (e.g., wrong input size for the checkpoint). To move toward the target with minimal core-logic change, I (1) filter auto-discovered checkpoints by compatibility by measuring how many parameters actually load, and only ensemble “good” ones (otherwise you ensemble noise), (2) automatically infer the checkpoint’s expected input resolution from common metadata (and fall back to 384), and (3) keep your exact model + flip-TTA + permutation-calibration approach unchanged aside from these safety gates. This should increase accuracy by ensuring you only use real, matching Cassava weights and apply the right resize, without changing the intended inference semantics. The script still always writes a valid `submission.csv` in the required format.'
- What this solution (achieved 0.11584) has done: 'Your current score strongly indicates you’re still falling back to `sample_submission.csv` (or using incompatible checkpoints that mostly don’t load), so the smallest meaningful improvement is to (1) make checkpoint discovery actually find Cassava checkpoints in this dataset environment by scanning `/kaggle/input` and `/kaggle/data` and (2) ensure we only accept checkpoints that load into your ConvNeXt-Tiny head correctly (including common key name mismatches like `head.fc` vs `head`). I keep your exact inference pipeline (ConvNeXt-Tiny, flip-TTA, optional permutation calibration, ensemble averaging) and only tighten weight-loading compatibility so real weights are used when present. I also make the test/train image root resolution more robust (some environments place them under different but existing folders), which prevents silent “no images”/fallback behavior. These minimal changes should move your accuracy substantially upward toward the target without changing the model or training logic.'

# 9. Code solution

## === cell 0
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import permutations

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

import albumentations as A
from albumentations.pytorch import ToTensorV2

from tqdm.auto import tqdm
import timm


class CFG:
    img_size = 384
    batch_size = 64
    num_workers = 4

    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
    train_csv = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

    model_paths = [
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold0.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold1.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold2.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold3.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold4.pth",
    ]

    calibrate_label_permutation = True
    calib_max_images = 512  # small + deterministic; keeps runtime within limits
    calib_seed = 123
    calib_batch_size = 64

    min_loaded_param_ratio = 0.80

    allow_ckpt_img_size_override = True

    ckpt_search_roots = ["/kaggle/input", "/kaggle/data"]

    allow_classifier_key_remap = True


def build_test_tfms(img_size: int):
    return A.Compose(
        [
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


test_tfms = build_test_tfms(CFG.img_size)




## === cell 1
class TestDataset(Dataset):
    def __init__(self, folder, tfms):
        self.paths = sorted([str(p) for p in Path(folder).glob("*.jpg")])
        self.tfms = tfms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tfms(image=img)["image"]
        return img, os.path.basename(img_path)


class TrainSubsetDataset(Dataset):
    def __init__(self, image_paths, labels, tfms):
        self.image_paths = image_paths
        self.labels = labels
        self.tfms = tfms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tfms(image=img)["image"]
        return img, int(self.labels[idx])


class CassavaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=False, num_classes=5
        )

    def forward(self, x):
        return self.backbone(x)




## === cell 2
def _extract_state_dict(ckpt):
    """
    BUGFIX: checkpoints can be raw state_dict, or dict with keys like 'state_dict'/'model'.
    Also handles DataParallel 'module.' prefix.
    """
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state_dict", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                ckpt = ckpt[k]
                break

    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt)}")

    if any(key.startswith("module.") for key in ckpt.keys()):
        ckpt = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    return ckpt


def _maybe_remap_classifier_keys(state: dict) -> dict:
    """
    SCORE FIX (minimal): many timm checkpoints store the classifier as head.fc.* while the current
    timm convnext uses head.* (or vice versa). Remapping these keys can turn a low-load-ratio
    checkpoint into a compatible one without changing the model.
    """
    if not CFG.allow_classifier_key_remap:
        return state

    if not isinstance(state, dict) or len(state) == 0:
        return state

    remaps = [
        ("head.fc.weight", "head.weight"),
        ("head.fc.bias", "head.bias"),
        ("head.weight", "head.fc.weight"),
        ("head.bias", "head.fc.bias"),
    ]

    state2 = dict(state)
    for src, dst in remaps:
        if src in state2 and dst not in state2:
            state2[dst] = state2[src]
    return state2


def _find_candidate_checkpoints(roots, max_files=30):
    """
    SCORE FIX (minimal): prioritize cassava-related checkpoints first to avoid picking unrelated .pth files.
    Expanded to search both /kaggle/input and /kaggle/data to reduce chance of "no weights found".
    """
    if isinstance(roots, (str, Path)):
        roots = [roots]
    roots = [Path(r) for r in roots if Path(r).exists()]

    if not roots:
        return []

    priority_globs = [
        "**/*cassava*/*best*.pth",
        "**/*cassava*/*fold*.pth",
        "**/*cassava*/*.pth",
        "**/*best*.pth",
        "**/*fold*.pth",
        "**/*.pth",
    ]

    seen = set()
    out = []
    for root in roots:
        for pat in priority_globs:
            for p in root.glob(pat):
                ps = str(p)
                if ps in seen:
                    continue
                seen.add(ps)
                out.append(ps)
                if len(out) >= max_files:
                    return out
    return out


def _infer_img_size_from_ckpt(ckpt, default_size: int) -> int:
    """
    SCORE FIX (minimal): try to infer expected input size from common training checkpoint metadata.
    If not found, keep the existing default (384) to preserve behavior.
    """
    if not CFG.allow_ckpt_img_size_override:
        return default_size

    if not isinstance(ckpt, dict):
        return default_size

    candidates = []

    for k in ("img_size", "image_size", "input_size"):
        if k in ckpt:
            candidates.append(ckpt[k])

    for k in ("cfg", "config", "hparams", "args"):
        if k in ckpt and isinstance(ckpt[k], dict):
            for kk in ("img_size", "image_size", "input_size"):
                if kk in ckpt[k]:
                    candidates.append(ckpt[k][kk])

    for v in candidates:
        try:
            if isinstance(v, int):
                s = v
            elif isinstance(v, (tuple, list)) and len(v) >= 2:
                s = int(v[-1])
            elif isinstance(v, str):
                import re

                nums = re.findall(r"\d+", v)
                s = int(nums[-1]) if nums else default_size
            else:
                continue
            if 128 <= s <= 768:
                return s
        except Exception:
            pass

    return default_size


def _loaded_param_ratio(model: nn.Module, state: dict) -> float:
    """
    SCORE FIX (minimal): quantify how much of the model is actually being initialized from checkpoint.
    If ratio is low, predictions are near-random; better to skip that checkpoint.
    """
    model_sd = model.state_dict()
    total = 0
    matched = 0
    for k, v in model_sd.items():
        total += v.numel()
        if (
            k in state
            and isinstance(state[k], torch.Tensor)
            and tuple(state[k].shape) == tuple(v.shape)
        ):
            matched += v.numel()
    return float(matched) / float(total) if total > 0 else 0.0


@torch.no_grad()
def _predict_proba(model, loader, device):
    model.eval()
    out = []
    for imgs, _ in loader:
        imgs = imgs.to(device, non_blocking=(device == "cuda"))
        if device == "cuda":
            with autocast(device_type="cuda"):
                logits = model(imgs)
        else:
            logits = model(imgs)
        out.append(torch.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def _resolve_train_image_root():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification/train_images",
        "/kaggle/data/cassava-leaf-disease-classification/train_images",
        "/kaggle/input/train_images",
        "/kaggle/data/train_images",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]


@torch.no_grad()
def _calibrate_label_permutation(model, device, tfms):
    """
    SCORE FIX (minimal, metric-aligned): find best mapping from model's class index -> competition label id.
    This addresses common issue where checkpoint was trained with different class ordering.
    Uses a small deterministic subset of train.csv; no training, just evaluation.
    """
    if (not CFG.calibrate_label_permutation) or (not os.path.exists(CFG.train_csv)):
        return None

    train_df = pd.read_csv(CFG.train_csv)
    img_root = _resolve_train_image_root()

    rs = np.random.RandomState(CFG.calib_seed)
    idx = rs.choice(
        len(train_df), size=min(CFG.calib_max_images, len(train_df)), replace=False
    )
    sub = train_df.iloc[idx].copy().reset_index(drop=True)

    paths = [os.path.join(img_root, x) for x in sub["image_id"].tolist()]
    labels = sub["label"].astype(int).values

    ds = TrainSubsetDataset(paths, labels, tfms=tfms)
    loader = DataLoader(
        ds,
        batch_size=CFG.calib_batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=(device == "cuda"),
    )

    probs = _predict_proba(model, loader, device)
    pred_idx = probs.argmax(axis=1)

    best_perm = tuple(range(5))
    best_acc = -1.0
    for perm in permutations(range(5)):
        mapped = np.array([perm[i] for i in pred_idx], dtype=np.int64)
        acc = (mapped == labels).mean()
        if acc > best_acc:
            best_acc = acc
            best_perm = perm

    print(
        f"Calibrated label permutation (class_index->label_id): {best_perm} (subset acc={best_acc:.4f})"
    )
    return np.array(best_perm, dtype=np.int64)




## === cell 3
@torch.no_grad()
def inference():
    if not os.path.exists(CFG.test_dir):
        alt = "/kaggle/data/cassava-leaf-disease-classification/test_images"
        if os.path.exists(alt):
            CFG.test_dir = alt

    available_model_paths = [p for p in CFG.model_paths if os.path.exists(p)]
    if len(available_model_paths) == 0:
        discovered = _find_candidate_checkpoints(CFG.ckpt_search_roots, max_files=25)
        available_model_paths = [p for p in discovered if os.path.exists(p)]
        if len(available_model_paths) > 0:
            print("Discovered checkpoints (candidates for ensembling):")
            for p in available_model_paths:
                print("  ", p)

    if len(available_model_paths) == 0:
        sample_path = (
            "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
        )
        if not os.path.exists(sample_path):
            sample_path = "/kaggle/input/sample_submission.csv"
        if not os.path.exists(sample_path):
            sample_path = "/kaggle/data/sample_submission.csv"
        sub = pd.read_csv(sample_path)
        sub.to_csv("submission.csv", index=False)
        print(
            "WARNING: No model weights found under configured paths or search roots; wrote sample_submission as submission.csv."
        )
        print(sub.head())
        return sub

    inferred_img_size = CFG.img_size
    for p in available_model_paths:
        try:
            ckpt0 = torch.load(p, map_location="cpu")
            inferred_img_size = _infer_img_size_from_ckpt(ckpt0, CFG.img_size)
            break
        except Exception:
            continue

    tfms = build_test_tfms(inferred_img_size)
    if inferred_img_size != CFG.img_size:
        print(
            f"Using checkpoint-inferred img_size={inferred_img_size} (CFG.img_size was {CFG.img_size})."
        )

    dataset = TestDataset(CFG.test_dir, tfms=tfms)
    if len(dataset) == 0:
        raise RuntimeError(f"No test images found under: {CFG.test_dir}")

    loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=(CFG.device == "cuda"),
    )

    ensemble_probs = None
    used_paths = []

    for fold, path in enumerate(available_model_paths):
        print(f"Loading fold {fold} → {os.path.basename(path)} on device={CFG.device}")
        model = CassavaModel().to(CFG.device)

        ckpt = torch.load(path, map_location=CFG.device)
        state = _extract_state_dict(ckpt)

        state = _maybe_remap_classifier_keys(state)

        ratio = _loaded_param_ratio(model, state)
        print(f"Checkpoint loadable parameter ratio: {ratio:.3f}")
        if ratio < CFG.min_loaded_param_ratio:
            print(
                f"Skipping {os.path.basename(path)} because ratio<{CFG.min_loaded_param_ratio} (likely incompatible / wrong model)."
            )
            del model, ckpt, state
            if CFG.device == "cuda":
                torch.cuda.empty_cache()
            continue

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"Note: load_state_dict strict=False; missing={len(missing)}, unexpected={len(unexpected)}"
            )

        perm = _calibrate_label_permutation(model, CFG.device, tfms=tfms)

        fold_probs = []
        for imgs, _ in tqdm(loader, leave=False, desc=f"Fold {fold} TTA"):
            imgs = imgs.to(CFG.device, non_blocking=(CFG.device == "cuda"))

            if CFG.device == "cuda":
                with autocast(device_type="cuda"):
                    p1 = torch.softmax(model(imgs), dim=1)
                    p2 = torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
            else:
                p1 = torch.softmax(model(imgs), dim=1)
                p2 = torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)

            p = ((p1 + p2) / 2).cpu().numpy()

            if perm is not None:
                p = p[:, np.argsort(perm)]  # label-id columns order

            fold_probs.append(p)

        fold_probs = np.concatenate(fold_probs, axis=0)
        ensemble_probs = (
            fold_probs if ensemble_probs is None else (ensemble_probs + fold_probs)
        )
        used_paths.append(path)

        del model, ckpt, state
        if CFG.device == "cuda":
            torch.cuda.empty_cache()

    if ensemble_probs is None or len(used_paths) == 0:
        sample_path = (
            "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
        )
        if not os.path.exists(sample_path):
            sample_path = "/kaggle/input/sample_submission.csv"
        if not os.path.exists(sample_path):
            sample_path = "/kaggle/data/sample_submission.csv"
        sub = pd.read_csv(sample_path)
        sub.to_csv("submission.csv", index=False)
        print(
            "WARNING: All discovered checkpoints were incompatible with convnext_tiny; wrote sample_submission as submission.csv."
        )
        print(sub.head())
        return sub

    final_labels = np.argmax(ensemble_probs / len(used_paths), axis=1)

    sub = pd.DataFrame(
        {
            "image_id": [os.path.basename(p) for p in dataset.paths],
            "label": final_labels.astype(int),
        }
    )
    sub = sub.sort_values("image_id").reset_index(drop=True)
    sub.to_csv("submission.csv", index=False)

    print(f"\n{len(sub)} predictions written to submission.csv")
    print(f"Used {len(used_paths)} checkpoint(s) in ensemble:")
    for p in used_paths:
        print("  ", p)
    print(sub.head())
    return sub


inference()
