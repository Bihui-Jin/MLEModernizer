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

0.8927168328800241

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the pipeline so it always produces a valid `submission.csv` by resolving the missing checkpoint issue and preventing downstream `KeyError`s. The main change is to make `load_model()` robust: it search for the checkpoint, and if it’s not available in this environment, it fall back to an untrained EfficientNet-B4 (still the same architecture) and continue inference so a submission is generated. I also add a small safety initialization for `test["label"]` and ensure transforms always include a deterministic resize when `RandomResizedCrop` is specified (it was previously ignored), which fixes a silent logic bug and improves stability. All paths remain unchanged and the submission format/order exactly match `sample_submission.csv`.'
- What this solution (achieved 0.05531) has done: 'Your low score is consistent with doing heavy random augmentations (including dropout/cutout) at inference time and averaging raw logits; this makes predictions noisy and harms accuracy. To move toward the target with minimal semantic change, I switch inference-time transforms to deterministic test-time preprocessing (Resize + Normalize only), while keeping your model architecture, checkpoint loading, dataloader, and multi-pass averaging loop intact. I also average in probability space (softmax) instead of logit space across the repeated passes, which is a minimal post-processing change aligned with accuracy and typically stabilizes class argmax. These two changes should substantially increase the score without changing the overall approach or I/O paths.'
- What this solution (achieved 0.05531) has done: 'Your 0.05531 score is consistent with the model running with randomly initialized weights because the checkpoint is not being found/loaded in this environment; fixing checkpoint resolution is the smallest change that can realistically move accuracy toward your 0.8927 target without changing the model or inference logic. I keep your EfficientNet-B4 architecture and the same inference loop, but make `_find_checkpoint_file` also search for any `.pt/.pth` under the available input folders and load the “best match” if `baselinebest.pt` is absent. I also add a strict sanity print to confirm whether weights were loaded (so you can verify you’re not silently submitting random predictions). Paths and the submission format/order remain unchanged.'
- What this solution (achieved 0.05531) has done: 'Your current score strongly suggests the model is still effectively untrained (random weights) because the checkpoint isn’t being found/loaded, and/or the loaded file isn’t a compatible EfficientNet-B4 5-class state dict. To move accuracy toward the 0.8927 target with minimal changes and identical inference semantics, I (1) make checkpoint discovery prefer “cassava/efficientnet/b4/best” candidates and only accept checkpoints whose keys actually match EfficientNet-B4, and (2) load in `eval()` mode and print an explicit “weights loaded OK” signal so you can verify it’s not silently random. Everything else (architecture, transforms, dataloader, multi-pass probability averaging, submission schema/order, and paths) stays the same.'
- What this solution (achieved 0.13453) has done: 'Your score is near-random and far from the 0.8927 target, which strongly suggests you’re still doing inference with randomly initialized EfficientNet-B4 weights (no valid checkpoint found/loaded). The smallest change that should move accuracy sharply upward is to stop silently falling back to random weights: instead, if no compatible EfficientNet-B4 5-class checkpoint is found, we deterministically use a strong ImageNet-pretrained EfficientNet-B4 as a fallback (same architecture) with only the classifier randomly initialized. I also slightly broaden checkpoint compatibility to accept common EfficientNet-B4 key patterns (so you actually load weights when they exist), while keeping the same dataloader, transforms, multi-epoch averaging loop, and submission format unchanged. This should move the score much closer to the target band without changing the overall approach.'
- What this solution (achieved 0.61099) has done: 'Your score (0.13453) is still far below the target (0.8927), which most strongly indicates the checkpoint being loaded is not a cassava-finetuned B4 (or the head weights aren’t being loaded), so the model behaves near-random. I make checkpoint loading stricter: we only accept a checkpoint if it (a) matches EfficientNet-B4 keys and (b) has a 5-class classifier tensor shape; otherwise we fall back to an ImageNet-pretrained backbone *with a deterministic, data-driven classifier initialization* (class-prior bias) to move accuracy upward without changing architecture or inference semantics. I also ensure the model uses the correct EfficientNet-B4 preprocessing by switching `Normalize` to the official torchvision weights’ mean/std (a minimal, metric-aligned change), and keep your same multi-pass averaging loop and submission formatting unchanged. These are small, safe changes that should substantially increase accuracy toward the target while preserving your core pipeline.'

# 9. Code solution

## === cell 0
import os
import gc
import time
import random
from typing import Dict, Optional, List, Tuple

import numpy as np
import pandas as pd
import cv2

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from tqdm import tqdm

from torchvision import models



## === cell 1
image_size = 380



## === cell 2
config = dict(
    seed=22,
    experiment_name="baseline",
    test_location="../input/cassava-leaf-disease-classification/test_images",
    checkpoint_path="../input/saved-models",
    checkpoint="baselinebest.pt",
    model="efficientnet-b4",
    epochs=10,
    batch_size=32,
    workers=8,
    inference_augmentations=[
        dict(
            name="RandomResizedCrop", params=dict(height=image_size, width=image_size)
        ),
        dict(
            name="HorizontalFlip",
            params=dict(
                always_apply=False,
                p=0.5,
            ),
        ),
        dict(name="Transpose", params=dict(p=0.5)),
        dict(name="VerticalFlip", params=dict(always_apply=False, p=0.5)),
        dict(
            name="HueSaturationValue",
            params=dict(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
            ),
        ),
        dict(
            name="RandomBrightnessContrast",
            params=dict(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
        ),
        dict(
            name="Normalize",
            params=dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ),
        dict(name="CoarseDropout", params=dict(p=0.5)),
        dict(name="Cutout", params=dict(p=0.5)),
    ],
)




## === cell 3
def _resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    candidates = [
        path,
        path.replace("../input", "/kaggle/input"),
        path.replace("../input", "/kaggle/data/input"),
        path.replace("../input", "/kaggle/data"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return path


sample_path = _resolve_path(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_path)
test = sample_sub[["image_id"]].copy()

test["label"] = 0

test.head()




## === cell 4
def seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 5
def _strip_prefix_if_present(
    state_dict: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {
        k[len(prefix) :] if k.startswith(prefix) else k: v
        for k, v in state_dict.items()
    }


def _candidate_checkpoint_paths(
    base_dir: Optional[str] = None,
) -> List[str]:
    roots = []
    if base_dir is not None:
        roots.append(_resolve_path(base_dir))
    roots += [
        _resolve_path("../input"),
        "/kaggle/input",
        "/kaggle/data/input",
        "/kaggle/data",
        ".",
    ]
    out = []
    seen = set()
    for r in roots:
        if r and r not in seen:
            out.append(r)
            seen.add(r)
    return out


def _looks_like_efficientnet_b4_state_dict(state: Dict[str, torch.Tensor]) -> bool:
    keys = set(state.keys())

    stem_ok = (
        ("features.0.0.weight" in keys)
        or ("features.0.0.0.weight" in keys)
        or ("features.0.0.0.0.weight" in keys)
    )

    classifier_ok = (
        ("classifier.1.weight" in keys and "classifier.1.bias" in keys)
        or ("classifier.weight" in keys and "classifier.bias" in keys)
        or ("head.weight" in keys and "head.bias" in keys)
        or ("fc.weight" in keys and "fc.bias" in keys)
    )

    return stem_ok and classifier_ok


def _extract_state_dict(obj) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
        if all(isinstance(k, str) for k in obj.keys()):
            return obj
    return None


def _find_checkpoint_file(checkpoint_name: str, base_dir: Optional[str] = None) -> str:
    search_roots = _candidate_checkpoint_paths(base_dir=base_dir)

    for root in search_roots:
        direct = os.path.join(root, checkpoint_name)
        if os.path.exists(direct):
            return direct
        if os.path.exists(root) and os.path.isdir(root):
            for dirpath, _, filenames in os.walk(root):
                if checkpoint_name in filenames:
                    return os.path.join(dirpath, checkpoint_name)

    exts = (".pt", ".pth", ".bin")
    candidates: List[Tuple[str, int]] = []
    for root in search_roots:
        if not (os.path.exists(root) and os.path.isdir(root)):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                lf = fn.lower()
                if not lf.endswith(exts):
                    continue
                full = os.path.join(dirpath, fn)
                lfull = full.lower()
                score = 0
                if "cassava" in lfull:
                    score += 8
                if "efficientnet" in lfull:
                    score += 6
                if "b4" in lfull or "effb4" in lfull:
                    score += 4
                if "best" in lf:
                    score += 10
                if os.path.splitext(checkpoint_name)[0].lower() in lf:
                    score += 5
                if base_dir is not None and _resolve_path(base_dir) in full:
                    score += 3
                candidates.append((full, score))

    if candidates:
        candidates.sort(key=lambda x: (x[1], -len(x[0])), reverse=True)
        chosen = candidates[0][0]
        print(
            f"Warning: exact checkpoint '{checkpoint_name}' not found; "
            f"falling back to candidate checkpoint: {chosen}"
        )
        return chosen

    return os.path.join(_resolve_path(base_dir or ""), checkpoint_name)


def _remap_common_classifier_keys(
    state: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    remapped = dict(state)

    if "head.weight" in remapped and "classifier.1.weight" not in remapped:
        remapped["classifier.1.weight"] = remapped.pop("head.weight")
    if "head.bias" in remapped and "classifier.1.bias" not in remapped:
        remapped["classifier.1.bias"] = remapped.pop("head.bias")

    if "fc.weight" in remapped and "classifier.1.weight" not in remapped:
        remapped["classifier.1.weight"] = remapped.pop("fc.weight")
    if "fc.bias" in remapped and "classifier.1.bias" not in remapped:
        remapped["classifier.1.bias"] = remapped.pop("fc.bias")

    if "classifier.weight" in remapped and "classifier.1.weight" not in remapped:
        remapped["classifier.1.weight"] = remapped.pop("classifier.weight")
    if "classifier.bias" in remapped and "classifier.1.bias" not in remapped:
        remapped["classifier.1.bias"] = remapped.pop("classifier.bias")

    return remapped


def _init_classifier_bias_from_train_priors(model: nn.Module) -> None:
    train_csv = _resolve_path("../input/cassava-leaf-disease-classification/train.csv")
    if not os.path.exists(train_csv):
        train_csv = _resolve_path("../input/train.csv")
    if not os.path.exists(train_csv):
        print("Warning: train.csv not found; leaving classifier bias at default init.")
        return

    df = pd.read_csv(train_csv)
    counts = df["label"].value_counts().sort_index()
    prior = (
        (counts / counts.sum())
        .reindex(range(5), fill_value=1e-6)
        .values.astype(np.float64)
    )
    prior = np.clip(prior, 1e-8, 1.0)
    log_prior = np.log(prior)
    log_prior = log_prior - log_prior.mean()

    with torch.no_grad():
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            head = model.classifier[1]
            if isinstance(head, nn.Linear) and head.out_features == 5:
                head.bias.copy_(torch.tensor(log_prior, dtype=head.bias.dtype))
                print(
                    "Initialized classifier bias from train label priors:",
                    prior.round(4),
                )
                return
    print("Warning: could not locate EfficientNet classifier[1] to set prior bias.")


def load_model():
    if config["model"] != "efficientnet-b4":
        raise ValueError(
            f"Only efficientnet-b4 is supported by this script, got {config['model']}"
        )

    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 5)

    ckpt_path = _find_checkpoint_file(config["checkpoint"], config["checkpoint_path"])
    if not os.path.exists(ckpt_path):
        print(
            "Warning: checkpoint not found. Using ImageNet-pretrained EfficientNet-B4 backbone as fallback.\n"
            f"Looked for: {config['checkpoint']} (starting at {config['checkpoint_path']}).\n"
            f"Final attempted path: {ckpt_path}"
        )
        model = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 5)
        _init_classifier_bias_from_train_priors(model)
        model = model.to(device)
        model.eval()
        return model

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = _extract_state_dict(checkpoint)
    if state is None:
        print(
            f"Warning: checkpoint at {ckpt_path} is not a recognizable state dict container. "
            "Using ImageNet-pretrained EfficientNet-B4 backbone as fallback."
        )
        model = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 5)
        _init_classifier_bias_from_train_priors(model)
        model = model.to(device)
        model.eval()
        return model

    state = _strip_prefix_if_present(state, "module.")
    state = _strip_prefix_if_present(state, "model.")
    state = _remap_common_classifier_keys(state)

    def _classifier_has_5_classes(st: Dict[str, torch.Tensor]) -> bool:
        w = st.get("classifier.1.weight", None)
        b = st.get("classifier.1.bias", None)
        if w is None or b is None:
            return False
        try:
            return (tuple(w.shape)[0] == 5) and (tuple(b.shape)[0] == 5)
        except Exception:
            return False

    if (not _looks_like_efficientnet_b4_state_dict(state)) or (
        not _classifier_has_5_classes(state)
    ):
        print(
            f"Warning: checkpoint found at {ckpt_path} but it doesn't look like a compatible EfficientNet-B4 5-class state_dict. "
            "Using ImageNet-pretrained EfficientNet-B4 backbone as fallback."
        )
        model = models.efficientnet_b4(
            weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
        )
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 5)
        _init_classifier_bias_from_train_priors(model)
        model = model.to(device)
        model.eval()
        return model

    missing, unexpected = model.load_state_dict(state, strict=False)

    print("Loading model from checkpoint:", ckpt_path)
    if isinstance(checkpoint, dict):
        if "epoch" in checkpoint:
            try:
                print("Epoch", int(checkpoint["epoch"]))
            except Exception:
                pass
        if "train_loss" in checkpoint:
            print("Train loss", checkpoint["train_loss"])
        if "val_loss" in checkpoint:
            print("Validation loss", checkpoint["val_loss"])
        if "metrics" in checkpoint:
            print("Accuracy", checkpoint["metrics"])
        if "lr" in checkpoint:
            print("Learning rate", checkpoint["lr"])

    if missing:
        print(
            f"Warning: missing keys when loading checkpoint (showing up to 20): {missing[:20]}"
        )
    if unexpected:
        print(
            f"Warning: unexpected keys when loading checkpoint (showing up to 20): {unexpected[:20]}"
        )

    print(
        "Checkpoint load sanity:",
        "OK" if (len(unexpected) == 0 and len(missing) <= 20) else "PARTIAL",
    )

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: total={num_params:,}, trainable={num_trainable:,}")

    model = model.to(device)
    model.eval()
    return model




## === cell 6
def get_transforms():
    w = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    mean = list(w.transforms().mean)
    std = list(w.transforms().std)
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=mean,
                std=std,
                max_pixel_value=255.0,
                p=1.0,
            ),
        ]
    )




## === cell 7
class CassavaDataset(Dataset):
    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __getitem__(self, n):
        image_path = os.path.join(
            _resolve_path(config["test_location"]), self.images[n]
        )
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        image = np.moveaxis(image, -1, 0)
        image = torch.as_tensor(image, dtype=torch.float32)
        return image

    def __len__(self):
        return len(self.images)




## === cell 8
def get_dataloader():
    transforms = get_transforms()
    test_data = np.array(test["image_id"])

    data = CassavaDataset(test_data, transforms)

    num_workers = int(config["workers"])
    if device.type == "cpu":
        num_workers = min(num_workers, 2)

    dataloader = DataLoader(
        data,
        shuffle=False,
        batch_size=config["batch_size"],
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
    )
    return dataloader




## === cell 9
def infer(model, dataloader):
    print("Running inference...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device, non_blocking=(device.type == "cuda"))
            batch_hat = model(batch)
            batch_prob = torch.softmax(batch_hat, dim=1)
            predictions.append(batch_prob.detach().cpu())

    return torch.cat(predictions, dim=0)




## === cell 10
if torch.cuda.is_available():
    torch.cuda.empty_cache()

dataloader = get_dataloader()
model = load_model()
predictions = None
print("Inferring experiment", config["experiment_name"])

for epoch in range(int(config["epochs"])):
    print("Epoch:", epoch)
    start_time = time.time()

    if epoch == 0:
        predictions = infer(model, dataloader)
    else:
        predictions += infer(model, dataloader)

    print("Time:", time.time() - start_time)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

predictions /= float(config["epochs"])
results = predictions.numpy()
test["label"] = np.argmax(results, axis=-1).astype(int)



## === cell 11
submission = sample_sub[["image_id"]].merge(
    test[["image_id", "label"]], on="image_id", how="left"
)
if submission["label"].isna().any():
    missing_ids = (
        submission.loc[submission["label"].isna(), "image_id"].head(10).tolist()
    )
    raise RuntimeError(
        f"Missing predictions for some test images (showing up to 10): {missing_ids}"
    )

submission = submission[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
