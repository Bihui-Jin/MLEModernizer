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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

0.8230583257781807

# 6. Current score

0.13976

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'I fix the immediate runtime blockers by (1) making the code robust to the absence of the external `.pth` checkpoint dataset and (2) ensuring `model` is always defined so inference can run and a valid `submission.csv` is written. Because your current score is “Not yielded”, the priority is to produce a valid end-to-end pipeline; if the checkpoint is missing, I fall back to a standard torchvision ResNet-50 backbone with a 5-class head so predictions can still be generated. I also fix the checkpoint loading logic to correctly handle either a full saved `nn.Module` or a `state_dict` (with safe, minimal assumptions). These changes keep the inference/preprocessing loop semantics the same and only add a necessary fallback to make the notebook executable in this environment.'
- What this solution (achieved 0.11584) has done: 'Your score is far below the target, and the main reason is that the current script likely runs with an untrained random ResNet50 (or partially mismatched checkpoint), plus a preprocessing mismatch (mean/std and resize) that can destroy accuracy for a pretrained cassava model. I make the smallest changes that keep the same core inference loop and model family: (1) use ImageNet normalization (the most common for ResNet checkpoints) and (2) load the checkpoint more robustly by stripping common prefixes (e.g., `module.`) and inferring the correct `num_classes` from the checkpoint’s `fc.weight` when possible. These changes should substantially increase accuracy toward your target without changing the architecture/training approach (still ResNet50 + linear head, single-image loop, argmax). The script still always produce a valid `submission.csv`.'
- What this solution (achieved 0.3722) has done: 'Your current score is far below the target, which strongly suggests the checkpoint isn’t being loaded into the same architecture it was trained with (so you’re effectively running a near-random head). I keep your ResNet50 + argmax inference core unchanged, but make checkpoint loading stricter and more compatible: (1) build the model with the correct `num_classes` inferred from the checkpoint, (2) correctly handle common head key variants (`fc.*`, `classifier.*`, `head.*`) by remapping them to `fc.*`, and (3) if the checkpoint indicates ImageNet pretraining (missing backbone weights), initialize the backbone with ImageNet weights instead of `weights=None`. These are minimal, directly score-relevant changes that should move accuracy substantially upward toward your target while still producing the same `submission.csv` format.'
- What this solution (achieved 0.15695) has done: 'Your current score is far below target, so the most likely issue is a mismatch between the checkpoint’s expected input preprocessing and the model you’re running at inference. I keep your ResNet50+argmax inference exactly the same, but add a minimal, safe test-time augmentation (TTA) path (original + horizontal flip) to improve accuracy without changing training or architecture. I also make the checkpoint head remapping slightly more compatible by handling `*.fc.*` and `*.classifier.*` keys (common in wrappers) while still preferring strict loading when possible. These changes are directly score-relevant and should move you closer to the target band while still producing a valid `submission.csv`.'
- What this solution (achieved 0.13976) has done: 'Your score is far below the target, so we should increase accuracy with minimal, score-relevant fixes while keeping your ResNet50 + argmax inference core unchanged. The biggest likely issue is input preprocessing mismatch: cassava ResNet50 checkpoints commonly use a centered-crop style eval pipeline (resize shorter side then center crop), not a hard `Resize((512,512))` which distorts aspect ratio and can tank accuracy. I change preprocessing to `Resize(512) + CenterCrop(512)` (and the same for the deterministic hflip path) while keeping the same normalization, image size, model, and TTA averaging. I also add an optional, safe softmax-averaging switch (default on) because averaging probabilities is typically better calibrated than averaging logits, without changing the final argmax semantics.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms, models



## === cell 1
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")

MODEL_PATH = (
    "/kaggle/input/resnet50_70_512x512/pytorch/default/1/Resnet50_70_512x512.pth"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_model_preprocess = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

main_model_preprocess_hflip = transforms.Compose(
    [
        transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(512),
        transforms.Lambda(lambda im: transforms.functional.hflip(im)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

USE_SOFTMAX_TTA_AVG = True




## === cell 2
def _find_checkpoint_fallback():
    candidates = glob.glob("/kaggle/input/**/*.pth", recursive=True)
    if not candidates:
        return None

    def score(p):
        s = p.lower()
        sc = 0
        for token, w in [
            ("cassava", 5),
            ("resnet50", 5),
            ("resnet", 2),
            ("512", 2),
            ("leaf", 1),
            ("disease", 1),
        ]:
            if token in s:
                sc += w
        sc -= 0.0001 * len(p)
        return sc

    candidates = sorted(candidates, key=score, reverse=True)
    return candidates[0]


def _strip_state_dict_prefixes(state_dict: dict) -> dict:
    if not state_dict:
        return state_dict

    def strip_prefix(sd, prefix):
        if sd and all(k.startswith(prefix) for k in sd.keys()):
            return {k[len(prefix) :]: v for k, v in sd.items()}
        return sd

    state_dict = strip_prefix(state_dict, "module.")
    state_dict = strip_prefix(state_dict, "model.")
    state_dict = strip_prefix(state_dict, "net.")
    return state_dict


def _remap_common_head_keys_to_fc(state_dict: dict) -> dict:
    """
    Score-relevant compatibility fix:
    some checkpoints save final layer as classifier/head OR with extra nesting like "model.fc.weight".
    We remap only if plain "fc.*" is absent.
    """
    if not state_dict:
        return state_dict

    has_fc = any(k.startswith("fc.") for k in state_dict.keys())
    if has_fc:
        return state_dict

    suffix_candidates = [
        ("fc.weight", "fc.bias"),
        ("classifier.weight", "classifier.bias"),
        ("head.weight", "head.bias"),
        ("last_linear.weight", "last_linear.bias"),
        ("output.weight", "output.bias"),
        ("logits.weight", "logits.bias"),
    ]
    keys = list(state_dict.keys())
    for w_suf, b_suf in suffix_candidates:
        w_keys = [k for k in keys if k.endswith(w_suf)]
        if w_keys:
            w_key = sorted(w_keys, key=len)[0]
            b_key = w_key[: -len(w_suf)] + b_suf
            new_sd = dict(state_dict)
            new_sd["fc.weight"] = new_sd.pop(w_key)
            if b_key in new_sd:
                new_sd["fc.bias"] = new_sd.pop(b_key)
            return new_sd

    for prefix in ("classifier.", "head.", "last_linear.", "output.", "logits."):
        w_key = prefix + "weight"
        b_key = prefix + "bias"
        if w_key in state_dict:
            new_sd = dict(state_dict)
            new_sd["fc.weight"] = new_sd.pop(w_key)
            if b_key in new_sd:
                new_sd["fc.bias"] = new_sd.pop(b_key)
            return new_sd

    return state_dict


def _infer_num_classes_from_state_dict(state_dict: dict, default: int = 5) -> int:
    if not state_dict:
        return default
    for k in (
        "fc.weight",
        "classifier.weight",
        "head.weight",
        "last_linear.weight",
        "output.weight",
        "logits.weight",
    ):
        w = state_dict.get(k, None)
        if isinstance(w, torch.Tensor) and w.ndim == 2 and w.shape[0] >= 2:
            return int(w.shape[0])

    for kk, vv in state_dict.items():
        if (
            isinstance(kk, str)
            and kk.endswith(".fc.weight")
            and isinstance(vv, torch.Tensor)
            and vv.ndim == 2
        ):
            return int(vv.shape[0])

    return default


def _should_use_imagenet_backbone(state_dict: dict) -> bool:
    if not state_dict:
        return True
    return ("conv1.weight" not in state_dict) and (
        "layer1.0.conv1.weight" not in state_dict
    )


def _build_fallback_model(num_classes=5, use_imagenet_backbone=False):
    weights = models.ResNet50_Weights.DEFAULT if use_imagenet_backbone else None
    m = models.resnet50(weights=weights)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m


if not os.path.exists(MODEL_PATH):
    alt = _find_checkpoint_fallback()
    if alt is not None:
        print(f"MODEL_PATH not found. Using fallback checkpoint: {alt}")
        MODEL_PATH = alt
    else:
        print(
            f"Warning: MODEL_PATH not found: {MODEL_PATH}\n"
            f"Also did not find any .pth under /kaggle/input.\n"
            f"Proceeding with an untrained fallback ResNet50 model so a valid submission.csv is produced."
        )
        MODEL_PATH = None

model = None

if MODEL_PATH is not None:
    ckpt = torch.load(MODEL_PATH, map_location=device)

    if isinstance(ckpt, torch.nn.Module):
        model = ckpt
    elif isinstance(ckpt, dict):
        state_dict = None
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            state_dict = ckpt["model"]
        elif all(isinstance(k, str) for k in ckpt.keys()):
            state_dict = ckpt

        if state_dict is not None:
            state_dict = _strip_state_dict_prefixes(state_dict)
            state_dict = _remap_common_head_keys_to_fc(state_dict)

            num_classes = _infer_num_classes_from_state_dict(state_dict, default=5)
            use_imagenet = _should_use_imagenet_backbone(state_dict)

            model = _build_fallback_model(
                num_classes=num_classes, use_imagenet_backbone=use_imagenet
            )

            try:
                model.load_state_dict(state_dict, strict=True)
            except Exception as e:
                print(
                    f"Warning: strict=True load failed ({type(e).__name__}: {e}). Falling back to strict=False."
                )
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    print(
                        f"Warning: missing keys when loading state_dict (showing up to 10): {missing[:10]}"
                    )
                if unexpected:
                    print(
                        f"Warning: unexpected keys when loading state_dict (showing up to 10): {unexpected[:10]}"
                    )
        else:
            print(
                "Warning: Unrecognized checkpoint dict format. Proceeding with an untrained fallback model."
            )
            model = _build_fallback_model(num_classes=5, use_imagenet_backbone=True)
    else:
        print(
            "Warning: Unrecognized checkpoint format. Proceeding with an untrained fallback model."
        )
        model = _build_fallback_model(num_classes=5, use_imagenet_backbone=True)

if model is None:
    model = _build_fallback_model(num_classes=5, use_imagenet_backbone=True)

model.to(device)
model.eval()



## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
image_ids = sample_sub["image_id"].astype(str).tolist()

prediction = []
missing = 0
failed = 0

softmax = torch.nn.Softmax(dim=1)

with torch.no_grad():
    for image_id in image_ids:
        img_path = os.path.join(TEST_IMG_DIR, image_id)

        if not os.path.exists(img_path):
            missing += 1
            prediction.append(0)
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            failed += 1
            prediction.append(0)
            continue

        x1 = main_model_preprocess(img).unsqueeze(0).to(device)
        x2 = main_model_preprocess_hflip(img).unsqueeze(0).to(device)

        logits1 = model(x1)
        logits2 = model(x2)

        if USE_SOFTMAX_TTA_AVG:
            probs = (softmax(logits1) + softmax(logits2)) / 2.0
            pred = int(torch.argmax(probs, dim=1).item())
        else:
            logits = (logits1 + logits2) / 2.0
            pred = int(torch.argmax(logits, dim=1).item())

        prediction.append(pred)

if missing:
    print(f"Warning: {missing} test images were missing. Filled with class 0.")
if failed:
    print(f"Warning: {failed} test images failed to load. Filled with class 0.")

assert len(prediction) == len(
    image_ids
), "Internal error: prediction and image_ids length mismatch."

submission = pd.DataFrame({"image_id": image_ids, "label": prediction})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print(f"Wrote submission.csv with {len(submission)} rows.")
