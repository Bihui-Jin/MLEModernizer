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

3.12

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

0.07586

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.42862) has done: 'I fix the TTA batching bug that causes a shape mismatch by iterating the base and TTA loaders in lockstep (same ordering) and only adding TTA probabilities when the batch size matches; this preserves the same core model/augmentation logic while preventing runtime errors on the last smaller batch. I also make the probability accumulation robust by reinitializing the TTA iterator each pass, and I add a small safety check to ensure the test set isn’t empty. Finally, I ensure `ensemble_predictions` is always defined before writing the submission and keep the submission aligned to `sample_submission.csv` ordering.'
- What this solution (achieved 0.43871) has done: 'Your low score is consistent with the code silently falling back to untrained ImageNet ResNet152 heads when the fold checkpoints aren’t found/loaded, which produces near-random predictions. I make the checkpoint discovery more robust by auto-searching common Kaggle input locations for those `.pt` files and correctly loading either a full model or a state_dict (including handling `module.` prefixes). I also ensure the loaded model always has a 5-class head (rebuilding if needed) while keeping the same architecture/inference/TTA logic. These minimal changes should move accuracy substantially upward toward your target by actually using the trained weights.'
- What this solution (achieved 0.43087) has done: 'Your current score strongly suggests you’re still not actually loading the trained fold weights (so inference is essentially random). I make the checkpoint discovery match both filename patterns you use (`best_model_tuned_*` vs `best_model_tuned_resnet152_*`) and search for both the full-model `.pt` and the state_dict `.pt`, then load whichever exists per fold. I also handle the common case where the saved file is a dict wrapper like `{"state_dict": ...}` (or nested under keys like `model`), which your current loader doesn’t unwrap—this is a minimal change that keeps the same architecture/inference/TTA logic but should move accuracy much closer to your target. Everything else (transforms, TTA count, ensemble averaging, submission formatting) stays the same.'
- What this solution (achieved 0.43199) has done: 'Your score is far below the target, and the most likely cause (given your logs/plans) is still that the fold checkpoints are not being found or not being correctly interpreted as state_dicts, so you’re effectively running mostly-untrained models at inference. I make the checkpoint discovery stricter and more exhaustive (searching for all fold files, not just fold_0 patterns) and improve the unwrapping logic to correctly extract a state_dict from common wrappers (including PyTorch Lightning `state_dict` formats). I also enforce that we only accept a “loaded” model if a non-trivial number of keys actually match, otherwise we fall back and warn—this prevents silent partial-loads that behave like random heads. Core model architecture (ResNet152 + linear head), transforms, TTA averaging, and submission formatting remain unchanged.'
- What this solution (achieved 0.07586) has done: 'Your score is far below the target, which strongly indicates inference is still running mostly with untrained weights (or loading the wrong tensors into the wrong places), so the smallest score-improving change is to make checkpoint discovery/loading actually succeed and be verifiably correct. I keep your ResNet152 + (fc→5) architecture, transforms, and TTA/ensemble logic unchanged, but expand checkpoint search to include common Kaggle dataset layouts and filenames, and improve unwrapping so we correctly extract a real `state_dict` even from Lightning-style checkpoints. I also add a hard check that the loaded `fc.*` tensors match shape `(5, ...)` (and skip those checkpoints if not), because a mismatched head load can silently degrade predictions. Finally, I ensure we only proceed with “fallback untrained” if no valid checkpoints are found, since that’s what produces ~0.43 accuracy.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import json
import copy
import pathlib

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet152, ResNet152_Weights
from torch.utils.data import Dataset, DataLoader

import albumentations
from albumentations.pytorch.transforms import ToTensorV2

from PIL import Image

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification/"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True  # faster inference


seed_everything(42)
Image.MAX_IMAGE_PIXELS = None



## === cell 1
test_glob = os.path.join(BASE_PATH, "test_images", "*.jpg")
test_images = sorted(glob.glob(test_glob))
if len(test_images) == 0:
    test_glob_alt = "/kaggle/input/test_images/*.jpg"
    test_images = sorted(glob.glob(test_glob_alt))

print(f"Found test images: {len(test_images)}")
if len(test_images) == 0:
    raise FileNotFoundError(
        f"No test images found. Tried: {test_glob} and {test_glob_alt}"
    )

df_test = pd.DataFrame({"path": test_images})
df_test["label"] = -1



## === cell 2
width = 512
height = 512

test_transforms = albumentations.Compose(
    [
        albumentations.CenterCrop(height=height, width=width, p=1.0),
        albumentations.Resize(height=height, width=width),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

tta_transforms = albumentations.Compose(
    [
        albumentations.RandomResizedCrop(
            size=(height, width), scale=(0.8, 1.0), ratio=(0.75, 1.3333333333), p=1.0
        ),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.Transpose(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.ShiftScaleRotate(p=0.5),
        albumentations.HueSaturationValue(
            hue_shift_limit=0.2,
            sat_shift_limit=0.2,
            val_shift_limit=0.2,
            p=0.5,
        ),
        albumentations.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1),
            contrast_limit=(-0.1, 0.1),
            p=0.5,
        ),
        albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 3
class TestDataset(Dataset):
    def __init__(self, image_ids, labels, transform=None):
        self.transform = transform
        self.image_ids = list(image_ids)
        self.labels = list(labels)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        path = self.image_ids[index]
        with Image.open(path) as im:
            img = im.convert("RGB")
            img = np.array(img)
        label = torch.tensor(self.labels[index], dtype=torch.long)

        if self.transform:
            return self.transform(image=img)["image"], label
        return img, label


test_dataset = TestDataset(
    image_ids=df_test.path, labels=df_test.label, transform=test_transforms
)
tta_dataset = TestDataset(
    image_ids=df_test.path, labels=df_test.label, transform=tta_transforms
)

num_workers = 2
pin_memory = DEVICE == "cuda"
batch_size = 32 if DEVICE == "cuda" else 8

test_dl = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    drop_last=False,
)
tta_dl = DataLoader(
    tta_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    drop_last=False,
)

test_set_size = len(test_dataset)
tta_num = 5

print(f"test_set_size={test_set_size}, tta_num={tta_num}, batch_size={batch_size}")




## === cell 4
def build_resnet152_num_classes(num_classes: int = 5) -> nn.Module:
    model = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def _strip_module_prefix(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(
        isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()
    ):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def _unwrap_state_dict(maybe: object) -> dict | None:
    """
    Change (score-relevant): broaden checkpoint unwrapping to correctly extract the real model state_dict
    (common in Lightning: keys like 'state_dict' and nested wrappers). This prevents untrained inference.
    """
    if not isinstance(maybe, dict):
        return None

    if len(maybe) > 0 and all(isinstance(k, str) for k in maybe.keys()):
        if any(
            k.startswith(
                ("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.", "fc.")
            )
            for k in maybe.keys()
        ):
            return maybe

    for key in ("state_dict", "model_state_dict", "model", "net", "weights"):
        if key in maybe and isinstance(maybe[key], dict):
            inner = maybe[key]
            if len(inner) > 0 and all(isinstance(k, str) for k in inner.keys()):
                if any(
                    k.startswith(
                        (
                            "conv1.",
                            "bn1.",
                            "layer1.",
                            "layer2.",
                            "layer3.",
                            "layer4.",
                            "fc.",
                        )
                    )
                    for k in inner.keys()
                ):
                    return inner
            for key2 in ("state_dict", "model_state_dict", "model", "net", "weights"):
                if key2 in inner and isinstance(inner[key2], dict):
                    inner2 = inner[key2]
                    if len(inner2) > 0 and all(
                        isinstance(k, str) for k in inner2.keys()
                    ):
                        return inner2
            return inner

    for key in ("checkpoint", "ckpt"):
        if key in maybe and isinstance(maybe[key], dict):
            inner = maybe[key]
            for k2 in ("state_dict", "model_state_dict", "model", "net", "weights"):
                if k2 in inner and isinstance(inner[k2], dict):
                    return inner[k2]

    return None


def _safe_torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        return None


def _candidate_checkpoint_files() -> list[str]:
    """
    Change (score-relevant): search more comprehensively under /kaggle/input for any fold checkpoints,
    including datasets that are nested one level deeper. The current ~0.43 suggests checkpoints still not found/used.
    """
    roots = [
        "/kaggle/input",
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
    patterns = [
        "**/best_model_tuned_resnet152_10_epochs_fold_*.pt",
        "**/best_model_tuned_10_epochs_fold_*.pt",
        "**/best_model_tuned_weights_resnet152_10_epochs_fold_*.pt",
        "**/best_model_tuned_weights_10_epochs_fold_*.pt",
        "**/*resnet152*fold_*.pt",
        "**/*fold_*.pt",
    ]
    hits = []
    for r in roots:
        if not os.path.isdir(r):
            continue
        for pat in patterns:
            hits.extend(glob.glob(os.path.join(r, pat), recursive=True))
    hits = sorted(set(hits))
    return hits


def _extract_fold_from_name(path: str) -> int | None:
    base = os.path.basename(path)
    for token in ("fold_", "fold"):
        if token in base:
            idx = base.find(token) + len(token)
            if idx < len(base) and base[idx] == "_":
                idx += 1
            digits = ""
            while idx < len(base) and base[idx].isdigit():
                digits += base[idx]
                idx += 1
            if digits != "":
                try:
                    return int(digits)
                except Exception:
                    return None
    return None


def _group_checkpoints_by_fold(
    files: list[str], expected_folds: int = 5
) -> dict[int, dict[str, str]]:
    """
    Return fold -> {'model': path or None, 'weights': path or None} best-effort.
    """
    out: dict[int, dict[str, str]] = {i: {} for i in range(expected_folds)}
    for p in files:
        f = _extract_fold_from_name(p)
        if f is None or f not in out:
            continue
        name = os.path.basename(p).lower()
        key = "weights" if "weights" in name or "state_dict" in name else "model"
        prev = out[f].get(key)
        if prev is None:
            out[f][key] = p
        else:
            prev_score = (("resnet152" in os.path.basename(prev).lower()) * 10) + (
                len(prev)
            )
            new_score = (("resnet152" in name) * 10) + (len(p))
            if new_score >= prev_score:
                out[f][key] = p
    return out


def try_load_fold_model_from_paths(
    model_path: str | None, weights_path: str | None, fold: int
) -> nn.Module | None:
    loaded_model_obj = _safe_torch_load(model_path) if model_path is not None else None
    loaded_weights_obj = (
        _safe_torch_load(weights_path) if weights_path is not None else None
    )

    if isinstance(loaded_model_obj, nn.Module):
        model = loaded_model_obj
    else:
        model = build_resnet152_num_classes(5)

    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        try:
            in_features = model.fc.in_features
            if getattr(model.fc, "out_features", None) != 5:
                model.fc = nn.Linear(in_features, 5)
        except Exception:
            model = build_resnet152_num_classes(5)

    state_dict = _unwrap_state_dict(loaded_weights_obj)
    if state_dict is None:
        state_dict = _unwrap_state_dict(loaded_model_obj)

    if isinstance(state_dict, dict):
        state_dict = _strip_module_prefix(state_dict)

        if "fc.weight" in state_dict and hasattr(state_dict["fc.weight"], "shape"):
            if tuple(state_dict["fc.weight"].shape)[0] != 5:
                print(
                    f"Warning: fold {fold} checkpoint fc.weight shape {tuple(state_dict['fc.weight'].shape)} != (5, ...). Skipping."
                )
                return None
        if "fc.bias" in state_dict and hasattr(state_dict["fc.bias"], "shape"):
            if tuple(state_dict["fc.bias"].shape)[0] != 5:
                print(
                    f"Warning: fold {fold} checkpoint fc.bias shape {tuple(state_dict['fc.bias'].shape)} != (5,). Skipping."
                )
                return None

        model_keys = set(model.state_dict().keys())
        sd_keys = set(state_dict.keys())
        matched = len(model_keys.intersection(sd_keys))
        if matched < 200:
            print(
                f"Warning: fold {fold} state_dict match too small ({matched} keys). Treating as load failure."
            )
            return None

        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception:
            try:
                model.load_state_dict(state_dict, strict=False)
            except Exception:
                return None
    else:
        if not isinstance(loaded_model_obj, nn.Module):
            return None

    model.eval()
    return model


all_ckpts = _candidate_checkpoint_files()
print(
    f"Checkpoint discovery: found {len(all_ckpts)} candidate .pt files under /kaggle/input"
)
by_fold = _group_checkpoints_by_fold(all_ckpts, expected_folds=5)

models: list[nn.Module] = []
for fold in range(5):
    mp = by_fold[fold].get("model")
    wp = by_fold[fold].get("weights")
    m = try_load_fold_model_from_paths(mp, wp, fold)
    if m is not None:
        print(f"Loaded fold {fold} from model={mp} weights={wp}")
        models.append(m)
    else:
        print(
            f"Warning: could not load fold {fold} from discovered candidates (model={mp}, weights={wp})"
        )

if len(models) == 0:
    print(
        "No fold models could be loaded; using 5x ImageNet-initialized ResNet152 models (untrained heads) as fallback."
    )
    for _ in range(5):
        m = build_resnet152_num_classes(5)
        m.eval()
        models.append(m)
else:
    print(f"Loaded {len(models)} fold models total.")

models = [m.to(DEVICE) for m in models]



## === cell 5
num_models = len(models)
num_classes = 5


def get_model_probabilities(model: nn.Module) -> np.ndarray:
    """
    Keep core inference/TTA logic identical: average base prediction with `tta_num` TTA predictions.
    """
    model.eval()
    final_probabilities = np.zeros((test_set_size, num_classes), dtype=np.float32)

    with torch.no_grad():
        idx = 0

        for base_images, _ in test_dl:
            bsz = base_images.size(0)
            base_images = base_images.to(DEVICE, non_blocking=True)
            logits = model(base_images)
            probs = (
                torch.softmax(logits, dim=1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            final_probabilities[idx : idx + bsz] += probs
            idx += bsz

        for _ in range(tta_num):
            idx = 0
            for tta_images, _ in tta_dl:
                bsz = tta_images.size(0)
                tta_images = tta_images.to(DEVICE, non_blocking=True)
                tta_logits = model(tta_images)
                tta_probs = (
                    torch.softmax(tta_logits, dim=1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                final_probabilities[idx : idx + bsz] += tta_probs
                idx += bsz

    final_probabilities /= float(tta_num + 1)
    return final_probabilities


ensemble_probabilities = np.zeros((test_set_size, num_classes), dtype=np.float32)

for i, model in enumerate(models):
    probs = get_model_probabilities(model)
    ensemble_probabilities += probs
    print(f"Model {i+1}/{num_models} done")

ensemble_probabilities /= float(num_models)
ensemble_predictions = ensemble_probabilities.argmax(axis=1).astype(int)

print("Pred distribution:", np.bincount(ensemble_predictions, minlength=num_classes))



## === cell 6
final_test_submission = df_test.copy()
final_test_submission["image_id"] = final_test_submission["path"].str.split("/").str[-1]
final_test_submission["label"] = ensemble_predictions

final_test_csv = final_test_submission[["image_id", "label"]]

sample_path = os.path.join(BASE_PATH, "sample_submission.csv")
if os.path.exists(sample_path):
    sample = pd.read_csv(sample_path)
    final_test_csv = sample[["image_id"]].merge(
        final_test_csv, on="image_id", how="left"
    )
    final_test_csv["label"] = final_test_csv["label"].fillna(0).astype(int)

final_test_csv.to_csv("submission.csv", index=False)
print("Submission csv file created! -> submission.csv")
print(final_test_csv.head())
print(f"Rows in submission: {len(final_test_csv)}")
