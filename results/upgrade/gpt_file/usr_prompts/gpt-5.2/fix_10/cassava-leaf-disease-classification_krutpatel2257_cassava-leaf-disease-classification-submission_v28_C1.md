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
- What this solution (achieved 0.61099) has done: 'Your current gap to target is large (0.61099 vs 0.88849), and the most likely bottleneck is a mismatch between the checkpoint’s expected input preprocessing and the inference pipeline: ResNeXt checkpoints for Cassava are commonly trained with 512px inputs (often CenterCrop/Resize to 512), while the current code forces ImageNet-style 224 crops. I keep the exact same model (resnext50_32x4d), checkpoint-loading logic, and the same 2-view TTA averaging, but switch the test preprocessing back to a deterministic 512 pipeline (Resize to 512 + CenterCrop 512 + ImageNet normalize). To keep runtime within limits, I also increase DataLoader workers slightly (no semantic change) and keep ordering/merge alignment unchanged so the submission stays valid. These minimal changes are directly score-relevant and should move accuracy upward toward your target.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.88849), so we should increase accuracy with the smallest inference-only changes that reduce the likely preprocessing mismatch. The most probable issue is that your checkpoint was trained with a different input resolution than 512; many cassava ResNe(X)t checkpoints use 384 (very common) or 448 rather than 512, and a wrong resize/crop can strongly degrade accuracy. I keep the exact same model, weight-loading, and 2-view (base + hflip) TTA averaging, but switch the test-time pipeline to a deterministic `Resize -> CenterCrop` at a more likely training size (384) while keeping ImageNet normalization. This is a minimal, directly score-relevant change that should move accuracy substantially toward your target without changing core logic.'
- What this solution (achieved 0.61099) has done: 'Your current score is far below the target, so we should increase accuracy with the smallest inference-only change that reduces a likely train/test preprocessing mismatch. The biggest mismatch here is that your checkpoint path suggests an EfficientNet-B4 TTA model, but the code builds a ResNeXt50; even if weights load “strict=False”, the features won’t match and accuracy collapses. I keep the exact same inference approach (no training, same 2-view TTA averaging, same transforms), but switch the backbone to `torchvision.models.efficientnet_b4` when the checkpoint filename indicates “b4/efficientnet”, and replace the classifier head to 5 classes to match the task. This is minimal, directly score-relevant, and should move you materially closer to the target without changing evaluation semantics.'
- What this solution (achieved 0.61099) has done: 'Your score gap to the target is large, so we should improve accuracy with the smallest inference-only change likely to matter: make the test preprocessing match the backbone implied by the checkpoint. EfficientNet-B4 checkpoints are typically trained at 380px (often resize+center-crop 380), while ResNeXt50 commonly uses 224; using a mismatched 384 pipeline for both can depress accuracy. I keep your model selection, checkpoint-loading, and 2-view (base + hflip) TTA unchanged, but set the crop/resize size conditionally (EfficientNet-B4: 380; ResNeXt50: 224) while keeping the same normalization and deterministic ordering to preserve evaluation semantics. This should move the accuracy upward toward 0.888 without changing the core logic.'
- What this solution (achieved 0.61099) has done: 'Your score is far below the target, so the smallest score-relevant fix is to ensure we load the *correct* weights for the inferred backbone instead of silently proceeding with a partially-mismatched `strict=False` load (which often yields near-random predictions). I keep your backbone inference, transforms, and 2-view TTA exactly as-is, but change checkpoint loading to (1) prefer a checkpoint whose filename matches the inferred backbone (EfficientNet-B4 vs ResNeXt50) and (2) load with `strict=True` after cleaning common key prefixes; if strict loading fails, we fall back to your current `strict=False` behavior (so it still runs). This should materially increase accuracy toward your target without changing the model architecture or inference semantics. I also add a tiny safety to handle Lightning-style keys and log what happened so you can confirm the right weights were used.'
- What this solution (achieved 0.61099) has done: 'Your current score is far below the target, so the smallest safe way to move accuracy upward is to stop “guessing” the checkpoint/backbone and instead ensure the *same backbone and input preprocessing* used by the checkpoint are used at inference. I keep your exact inference approach (single model, 2-view TTA average, argmax) but (1) make backbone inference more reliable by inspecting the checkpoint’s state_dict keys (not just filename), and (2) ensure preprocessing uses the correct interpolation and size defaults for that backbone (ResNeXt: 224; EfficientNet-B4: 380) with consistent Albumentations interpolation constants. This avoids silent partial/mismatched loads (which commonly yields ~0.6 accuracy) while preserving your core logic and producing the same submission format.'

# 9. Code solution

## === cell 0
import os
import glob
import random

import albumentations as A
import cv2
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
def _extract_state_dict(ckpt_obj):
    if (
        isinstance(ckpt_obj, dict)
        and "state_dict" in ckpt_obj
        and isinstance(ckpt_obj["state_dict"], dict)
    ):
        return ckpt_obj["state_dict"]
    if isinstance(ckpt_obj, dict):
        return ckpt_obj
    return None


def _clean_state_dict_keys(state: dict) -> dict:
    cleaned = {}
    for k, v in state.items():
        nk = k
        for pref in ("model.", "module.", "net.", "backbone.", "encoder."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        nk = nk.replace("model.model.", "model.").replace("module.model.", "model.")
        cleaned[nk] = v
    return cleaned


def infer_backbone_from_ckpt_contents(path: str | None) -> str:
    """
    Change is score-relevant: filename-based guessing can pick the wrong backbone,
    causing partial/mismatched loads and low accuracy (~0.6). Inspecting keys is a
    minimal, inference-only way to match the checkpoint's true architecture.
    """
    if not path or (not os.path.exists(path)):
        return "resnext50"

    try:
        ckpt = torch.load(path, map_location="cpu")
        state = _extract_state_dict(ckpt)
        if state is None:
            return "resnext50"
        state = _clean_state_dict_keys(state)
        keys = list(state.keys())

        has_eff_features = any(k.startswith("features.") for k in keys)
        has_eff_classifier = any(k.startswith("classifier.") for k in keys)

        has_res_layers = any(
            k.startswith("layer1.") or k.startswith("layer2.") for k in keys
        )
        has_fc = any(k == "fc.weight" or k.startswith("fc.") for k in keys)

        if has_eff_features and has_eff_classifier and not (has_res_layers and has_fc):
            return "efficientnet_b4"
        if has_res_layers and has_fc and not (has_eff_features and has_eff_classifier):
            return "resnext50"

        name = os.path.basename(path).lower()
        if ("efficientnet" in name) or ("en-b" in name) or ("b4" in name):
            return "efficientnet_b4"
        return "resnext50"
    except Exception as e:
        print(
            "Backbone inference from contents failed; using filename heuristic. Reason:",
            repr(e),
        )
        name = os.path.basename(path).lower() if path else ""
        if ("efficientnet" in name) or ("en-b" in name) or ("b4" in name):
            return "efficientnet_b4"
        return "resnext50"


def find_best_pth_for_backbone(
    backbone: str, start_dir: str = "/kaggle/input"
) -> str | None:
    candidates = glob.glob(os.path.join(start_dir, "**", "*.pth"), recursive=True)
    if not candidates:
        return None

    def score(p: str) -> tuple:
        name = os.path.basename(p).lower()
        full = p.lower()

        if backbone == "efficientnet_b4":
            match = int(
                ("efficientnet" in name)
                or ("en-b" in name)
                or ("b4" in name)
                or ("efficientnet" in full)
            )
        else:
            match = int(("resnext" in name) or ("32x4d" in name) or ("resnext" in full))

        has_model = int("model" in name)
        return (-match, -has_model, len(p))

    return sorted(candidates, key=score)[0]


backbone = infer_backbone_from_ckpt_contents(resolved_model_path)
print("Inferred backbone (from ckpt contents when possible):", backbone)

if resolved_model_path is None or (not os.path.exists(resolved_model_path)):
    resolved_model_path = find_best_pth_for_backbone(backbone, "/kaggle/input")
else:
    candidate = find_best_pth_for_backbone(backbone, "/kaggle/input")
    if candidate is not None:
        cur = os.path.basename(resolved_model_path).lower()
        cand = os.path.basename(candidate).lower()
        if backbone == "efficientnet_b4":
            cur_ok = ("efficientnet" in cur) or ("en-b" in cur) or ("b4" in cur)
            cand_ok = ("efficientnet" in cand) or ("en-b" in cand) or ("b4" in cand)
        else:
            cur_ok = ("resnext" in cur) or ("32x4d" in cur)
            cand_ok = ("resnext" in cand) or ("32x4d" in cand)
        if (not cur_ok) and cand_ok:
            resolved_model_path = candidate

print("Backbone-matched model path:", resolved_model_path)

if backbone == "efficientnet_b4":
    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, config["CLASSES"])
else:
    model = models.resnext50_32x4d(weights=None)
    model.fc = nn.Linear(2048, config["CLASSES"])

model.to(device)


loaded_ok = False
if resolved_model_path is not None and os.path.exists(resolved_model_path):
    ckpt = torch.load(resolved_model_path, map_location="cpu")
    state = _extract_state_dict(ckpt)

    if state is not None:
        state = _clean_state_dict_keys(state)

        try:
            model.load_state_dict(state, strict=True)
            loaded_ok = True
            print("Loaded checkpoint with strict=True from:", resolved_model_path)
        except Exception as e:
            print("Strict load failed; falling back to strict=False. Reason:", repr(e))
            missing, unexpected = model.load_state_dict(state, strict=False)
            print("Loaded checkpoint with strict=False from:", resolved_model_path)
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
if backbone == "efficientnet_b4":
    CROP_SIZE = 380
    RESIZE_SHORT = 380
else:
    CROP_SIZE = 224
    RESIZE_SHORT = 224

print(f"Inference size: RESIZE_SHORT={RESIZE_SHORT}, CROP_SIZE={CROP_SIZE}")

sub_aug_base = A.Compose(
    [
        A.SmallestMaxSize(max_size=RESIZE_SHORT, interpolation=cv2.INTER_LINEAR, p=1.0),
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
        A.SmallestMaxSize(max_size=RESIZE_SHORT, interpolation=cv2.INTER_LINEAR, p=1.0),
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
    batch_size=32,
    shuffle=False,
    num_workers=4,
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

out_path = config["DATA"]["SUB_OUTPUT"]
sub_df.to_csv(out_path, index=False)

print("Saved:", out_path)
print(sub_df.head())
print("Rows:", len(sub_df), "Cols:", list(sub_df.columns))
assert out_path.endswith(".csv") and os.path.exists(out_path)
assert len(sub_df) == len(sample_sub)
assert list(sub_df.columns) == ["image_id", "label"]
