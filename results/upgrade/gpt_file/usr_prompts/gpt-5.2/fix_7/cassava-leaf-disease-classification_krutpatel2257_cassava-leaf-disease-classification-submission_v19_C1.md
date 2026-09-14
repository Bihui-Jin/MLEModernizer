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

0.8689936536718041

# 6. Current score

0.11024

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11061) has done: 'I remove the failing `pip install` dependency and instead build the exact EfficientNet-B4 architecture using `torchvision` so the provided `.pth` can load without `efficientnet_pytorch`. I fix the Albumentations v2 API break by using the correct `size=(H,W)` signature and replacing removed transforms (`Cutout`) with their current equivalents so TTA runs. I also fix inference-time bugs that would silently hurt accuracy (wrong averaging divisor, missing `torch.no_grad()`, and missing softmax/logit handling), while keeping the same core approach: EfficientNet-B4 + 5x TTA over the sample submission list and argmax for labels. Finally, the script always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.11024) has done: 'I fix the immediate runtime failure by making the checkpoint path robust: the current `model_path` points to a dataset that isn’t present, so I search common Kaggle input locations for `model_50.pth` (and fail fast with a clear message if it truly doesn’t exist). I also make the model weight loading tolerant to minor key mismatches (`strict=False`) while keeping the same EfficientNet-B4 architecture, which prevents silent “all-random-weights” behavior and should raise accuracy substantially toward your target. Finally, I fix a subtle inference bug: Albumentations `Normalize` already outputs float images, so applying `torchvision.transforms.ToTensor()` rescales by 1/255 again; I replace it with a correct HWC->CHW tensor conversion to avoid destroying the input scale and hurting score. The rest of the core logic (EfficientNet-B4 + 5x TTA over `sample_submission.csv` and argmax) stays the same.'
- What this solution (achieved 0.11024) has done: 'Your run is failing because the referenced pretrained checkpoint (`model_50.pth`) is not present anywhere in this Kaggle environment, so the code never gets past checkpoint resolution/loading. Since we must keep the same core logic (EfficientNet-B4 + TTA + argmax) but still produce a valid submission, I (1) make the checkpoint resolution robust to this environment by allowing a clean fallback to an untrained model (so the notebook runs end-to-end), and (2) add a clear warning so it’s obvious why the score be low without weights. I also make the file paths robust to both `../input/...` and `/kaggle/data/input/...` layouts so inference can always find `sample_submission.csv` and `test_images`. This fix the runtime errors and always write a valid `submission.csv`; meaningful score improvement require providing the missing checkpoint file in `/kaggle/input`.'
- What this solution (achieved 0.11024) has done: 'Your current score is extremely low because the model is effectively untrained in this environment (the checkpoint is missing), so the smallest change that can legitimately move accuracy toward your target is to load real pretrained weights without changing the architecture or inference semantics. I keep the same EfficientNet-B4 + 5x TTA + argmax core logic, but (1) point `model_path` to a checkpoint that actually exists in this dataset (`final_50.pth` is present in `/kaggle/data/...`), and (2) make weight loading stricter by auto-matching common key prefixes and requiring a high key-match ratio so we don’t silently run random weights. I also switch the TTA crop from `RandomResizedCrop` to deterministic `Resize+CenterCrop` (still 5 passes with the same flips/affine jitter) to reduce harmful randomness at inference; this typically improves accuracy materially when weights are correct and should move you much closer to the target. The script still runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 0.11024) has done: 'Your score is far below the target because the inference-time augmentation pipeline is extremely destructive for test-time prediction: it applies heavy geometric/color jitter plus CoarseDropout during TTA, which typically collapses accuracy even with good weights. To move accuracy upward toward the target while preserving the same core logic (EfficientNet-B4, 5x TTA loop, argmax over averaged logits), I keep TTA=5 but make the TTA transforms inference-safe by removing CoarseDropout and replacing random heavy transforms with mild, realistic TTA (small rotate/scale + flips only) while keeping the same normalization. I also switch to a 380px resize (EfficientNet-B4’s native resolution) with center crop to better match the model’s expected input distribution, without changing the model or loop structure. These are minimal, directly score-relevant changes that should materially increase accuracy toward your target.'
- What this solution (achieved 0.11024) has done: 'Your score is far below the target because the checkpoint you load is almost certainly not aligned with `torchvision.models.efficientnet_b4`’s internal layer naming/structure, so even with a high “key overlap” you can still end up with a badly-initialized network (effectively near-random). The smallest change that preserves your core logic (EfficientNet-B4 + 5x TTA + argmax) but should materially increase accuracy is to switch to the exact same EfficientNet-B4 implementation used by the checkpoint via `efficientnet_pytorch`, and then load the checkpoint strictly so we don’t silently run mismatched weights. I also keep TTA=5 but make it inference-safe/deterministic (no random rotate/scale) so predictions don’t degrade due to unnecessary randomness at test-time. Paths and the submission writing remain identical, and the script still runs end-to-end and always produces `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import glob

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

import albumentations as A



## === cell 1
model_path = "../input/en-b4-tta-calr-50/final_50.pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def resolve_existing_path(p: str, must_be_dir: bool = False) -> str:
    if os.path.isdir(p) if must_be_dir else os.path.exists(p):
        return p

    base = p.replace("\\", "/")
    candidates = []

    for root in ["../input", "/kaggle/input", "/kaggle/data/input", "/kaggle/data"]:
        alt = os.path.join(
            root, os.path.basename(os.path.dirname(base)), os.path.basename(base)
        )
        candidates.append(alt)

    for root in ["/kaggle/input", "/kaggle/data/input", "/kaggle/data"]:
        if base.startswith("../input/"):
            candidates.append(os.path.join(root, base[len("../input/") :]))

    for c in candidates:
        if os.path.isdir(c) if must_be_dir else os.path.exists(c):
            return c

    search_name = os.path.basename(base)
    for root in ["../input", "/kaggle/input", "/kaggle/data/input", "/kaggle/data"]:
        if os.path.isdir(root):
            pattern = os.path.join(root, "**", search_name)
            for hit in glob.glob(pattern, recursive=True):
                if os.path.isdir(hit) if must_be_dir else os.path.exists(hit):
                    return hit

    return p  # return original; caller decides whether to error


def resolve_checkpoint_path_or_none(p: str) -> str | None:
    if os.path.exists(p):
        return p

    fname = os.path.basename(p)
    search_roots = ["../input", "/kaggle/input", "/kaggle/data/input", "/kaggle/data"]
    candidates = []
    for root in search_roots:
        if os.path.isdir(root):
            candidates.extend(
                glob.glob(os.path.join(root, "**", fname), recursive=True)
            )

    preferred = [c for c in candidates if "en-b4-tta-calr-50" in c.replace("\\", "/")]
    if preferred:
        return preferred[0]
    if candidates:
        return candidates[0]
    return None


sample_sub_path = resolve_existing_path(sample_sub_path, must_be_dir=False)
test_images_path = resolve_existing_path(test_images_path, must_be_dir=True)

ckpt_path = resolve_checkpoint_path_or_none(model_path)
if ckpt_path is None:
    print(
        "WARNING: Checkpoint not found. The model will run with random weights, so Kaggle score will be very low.\n"
        "To improve score toward the target, add the checkpoint dataset under /kaggle/input."
    )
else:
    print("Using model checkpoint:", ckpt_path)

print("Using sample_submission:", sample_sub_path)
print("Using test_images dir:", test_images_path)



## === cell 2
try:
    from efficientnet_pytorch import (
        EfficientNet,
    )  # available on Kaggle for this competition in most environments
except Exception as e:
    EfficientNet = None
    effnet_import_error = e

if ckpt_path is not None and EfficientNet is None:
    raise RuntimeError(
        "Checkpoint is present but efficientnet_pytorch is not available in this environment. "
        "This solution expects the checkpoint to match efficientnet_pytorch EfficientNet-B4.\n"
        f"Import error: {effnet_import_error}"
    )

if EfficientNet is not None:
    model = EfficientNet.from_name("efficientnet-b4")
    in_features = model._fc.in_features
    model._fc = nn.Linear(in_features, 5)
else:
    from torchvision import models as tv_models

    model = tv_models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 5)

model = model.to(device)


def _extract_state_dict(ckpt_obj):
    state_dict = ckpt_obj
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                state_dict = ckpt_obj[key]
                break
    return state_dict


def _strip_prefix(state_dict: dict, prefix: str) -> dict:
    out = {}
    for k, v in state_dict.items():
        if isinstance(k, str) and k.startswith(prefix):
            out[k[len(prefix) :]] = v
        else:
            out[k] = v
    return out


if ckpt_path is not None:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = _extract_state_dict(ckpt)
    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "Checkpoint did not contain a state_dict-like mapping; cannot load weights."
        )

    sd = dict(state_dict)
    if any(isinstance(k, str) and k.startswith("module.") for k in sd.keys()):
        sd = _strip_prefix(sd, "module.")
    for pref in ["model.", "net."]:
        if any(isinstance(k, str) and k.startswith(pref) for k in sd.keys()):
            sd = _strip_prefix(sd, pref)

    missing, unexpected = model.load_state_dict(sd, strict=True)
    print(
        f"Loaded checkpoint with strict=True. Missing: {len(missing)} Unexpected: {len(unexpected)}"
    )

model.eval()



## === cell 3
sub_aug = A.Compose(
    [
        A.Resize(380, 380, p=1.0),
        A.CenterCrop(380, 380, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.0),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)


def alb_to_tensor(img_hwc: np.ndarray) -> torch.Tensor:
    if img_hwc.dtype != np.float32:
        img_hwc = img_hwc.astype(np.float32)
    return torch.from_numpy(img_hwc).permute(2, 0, 1).contiguous()




## === cell 4
sample_sub = pd.read_csv(sample_sub_path)

predictions = []
tta_n = 5  # keep identical TTA loop count

with torch.no_grad():
    for _, sample_row in sample_sub.iterrows():
        img_path = os.path.join(test_images_path, sample_row.image_id)
        image_np = np.array(Image.open(img_path).convert("RGB"))

        image_logits = 0.0
        for _ in range(tta_n):
            aug = sub_aug(image=image_np)["image"]  # HWC float32 normalized
            img_t = alb_to_tensor(aug).to(device)  # CHW float32 (already normalized)
            outputs = model(img_t.unsqueeze(0))  # logits
            image_logits = image_logits + outputs

        image_logits = image_logits / float(tta_n)
        pred_label = int(torch.argmax(image_logits, dim=1).item())
        predictions.append([sample_row.image_id, pred_label])

sub_df = pd.DataFrame(predictions, columns=["image_id", "label"])
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("Label distribution:\n", sub_df["label"].value_counts(dropna=False).sort_index())
