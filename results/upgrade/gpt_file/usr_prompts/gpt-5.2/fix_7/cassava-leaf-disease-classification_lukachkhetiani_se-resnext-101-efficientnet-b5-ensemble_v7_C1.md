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

0.8656693865216077

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I remove the notebook-style shell/cd/pip cells that don’t run in a plain Kaggle Python script environment and instead rely on the already-available Kaggle input dataset paths. Then I fix the `ModuleNotFoundError` by providing a small fallback EfficientNet-B5 implementation via `torchvision` (while still loading your `b5.pth` if it’s present), keeping the same single-model inference flow and argmax labeling. I also make the inference robust to missing CUDA and ensure test image file discovery matches the provided dataset structure. Finally, I always write a valid `submission.csv` with the required `image_id,label` columns.'
- What this solution (achieved 0.11584) has done: 'Your current score is near random because the checkpoint is likely not loading (or not matching) and the input normalization does not match EfficientNet’s expected preprocessing. To move the score toward the 0.865 target with minimal changes, I (1) make checkpoint loading robust to common key prefixes (`module.`, `model.`) and enforce strict loading when possible, and (2) switch preprocessing to standard ImageNet mean/std normalization while keeping your same CLAHE + resize + single-image inference + argmax flow. I also ensure we always output predictions for every image in `sample_submission.csv` order (no accidental drops from unreadable images). These are small, directly score-relevant fixes and should substantially improve accuracy if `b5.pth` is a real trained cassava checkpoint.'
- What this solution (achieved 0.11584) has done: 'Your current score suggests the model is effectively random, so the most likely issue is a checkpoint architecture mismatch (your code builds `torchvision.efficientnet_b5`, but many Cassava “B5” checkpoints were trained with `efficientnet_pytorch` and won’t load meaningfully). To move accuracy toward the 0.8657 target with minimal disruption, I keep your same single-model argmax inference, but add a robust fallback model builder: try `efficientnet_pytorch` (if available in the Kaggle environment) and otherwise use `torchvision`, then load the checkpoint in the first compatible way. I also make preprocessing match EfficientNet defaults more closely by resizing to 456 (EffNet-B5 default) while keeping your CLAHE + ImageNet normalization semantics. Finally, I keep the submission aligned exactly to `sample_submission.csv` order and still guarantee a valid `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your score is near-random, so the most likely cause is that the checkpoint isn’t actually being applied to the model you’re running (either no checkpoint is found, or the keys don’t match and most weights stay random). To move accuracy upward toward the 0.8657 target with minimal disruption, I (1) expand checkpoint discovery to include common Kaggle dataset locations under `/kaggle/input/**/b5*.pth`, and (2) make checkpoint loading handle the very common “wrapped” formats (e.g., `{"model": {"state_dict": ...}}`, `{"model_state_dict": ...}`) and also handle the case where the state dict corresponds to the EfficientNet backbone without the classifier head (by loading all matching keys and leaving only the head randomly initialized). I also add a single sanity check print showing what fraction of parameters were actually loaded, so you can immediately confirm you’re no longer running random weights; this is directly score-relevant while keeping your same single-model argmax inference and preprocessing flow.'
- What this solution (achieved 0.11584) has done: 'Your current score is near-random, which strongly suggests the checkpoint isn’t actually being used correctly (either not found, or loaded into a non-matching model so most weights remain random). I keep your single-model EfficientNet-B5 argmax inference and preprocessing flow, but make checkpoint discovery and loading more robust to common Kaggle Cassava checkpoint formats (nested dicts and key prefixes like `encoder.`/`backbone.`/`model.module.`), and ensure we only accept a loaded model if a meaningful fraction of parameters match by shape. I also align the final classifier replacement to handle both torchvision-style (`classifier[1]`) and efficientnet_pytorch-style (`_fc`) consistently, without changing the overall architecture choice you already have. These minimal changes should move accuracy up substantially toward your 0.8657 target if a real trained B5 checkpoint exists in `/kaggle/input`.'
- What this solution (achieved 0.11584) has done: 'Your score is near-random, so the smallest likely fix is to ensure we’re actually loading the correct weights into the correct model keys. I keep the same single-model EfficientNet-B5 argmax inference, but (1) make checkpoint key “cleaning” less destructive (don’t strip essential prefixes like `features.`), and (2) add a tiny head-mapping shim for torchvision EfficientNet checkpoints where the classifier is saved as `classifier.weight/bias` (common in many training scripts) so it loads into `classifier.1.*`. These changes are narrowly targeted at getting a real trained checkpoint to load meaningfully, which should move accuracy up toward your target. Everything else (CLAHE + resize + ImageNet normalization + per-image loop + submission format/order) stays the same.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings

warnings.filterwarnings("ignore")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
import tqdm

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
BASE = "/kaggle/input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(BASE, "test_images")

if not os.path.isdir(TEST_DIR):
    alt = "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images"
    if os.path.isdir(alt):
        TEST_DIR = alt

_ckpt_globs = [
    "/kaggle/input/**/b5.pth",
    "/kaggle/input/**/B5.pth",
    "/kaggle/input/**/b5*.pth",
    "/kaggle/input/**/efficientnet*b5*.pth",
    "/kaggle/input/**/*efficientnet*B5*.pth",
    "/kaggle/input/**/*effnet*b5*.pth",
    "/kaggle/input/**/*cassava*b5*.pth",
    "/kaggle/input/**/*.pth",
]
_ckpt_found = []
for pat in _ckpt_globs:
    _ckpt_found.extend(glob.glob(pat, recursive=True))
_ckpt_found = sorted({p for p in _ckpt_found if os.path.isfile(p)})

CKPT_CANDIDATES = [
    "b5.pth",
    "/kaggle/working/b5.pth",
    "/kaggle/input/b5v3checkpoint/b5.pth",
] + _ckpt_found

CKPT_PATH = next((p for p in CKPT_CANDIDATES if os.path.isfile(p)), None)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"TEST_DIR={TEST_DIR}")
print(f"CKPT_PATH={CKPT_PATH}")




## === cell 2
def _clean_state_dict_keys(state: dict) -> dict:
    if not isinstance(state, dict):
        return state
    cleaned = {}
    for k, v in state.items():
        nk = k

        prefixes = [
            "module.",
            "model.",
            "model.module.",
            "net.",
        ]
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if nk.startswith(p):
                    nk = nk[len(p) :]
                    changed = True

        cleaned[nk] = v
    return cleaned


def _extract_state_dict(ckpt_obj):
    if ckpt_obj is None:
        return None

    if isinstance(ckpt_obj, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
            "ema_state_dict",
            "student",
            "teacher",
        ]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                inner = ckpt_obj[k]
                for kk in ["state_dict", "model_state_dict", "model", "net", "weights"]:
                    if kk in inner and isinstance(inner[kk], dict):
                        return inner[kk]
                return inner

        tensor_like = True
        for v in ckpt_obj.values():
            if not (torch.is_tensor(v) or isinstance(v, np.ndarray)):
                tensor_like = False
                break
        if tensor_like:
            return ckpt_obj

    return None


def build_model_torchvision(num_classes: int = 5) -> nn.Module:
    m = models.efficientnet_b5(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, num_classes)
    return m


def build_model_efficientnet_pytorch(num_classes: int = 5) -> nn.Module:
    from efficientnet_pytorch import EfficientNet  # may or may not be available

    m = EfficientNet.from_name("efficientnet-b5")
    in_features = m._fc.in_features
    m._fc = nn.Linear(in_features, num_classes)
    return m


def _count_loaded_params_by_shape(model: nn.Module, loaded_sd: dict):
    model_sd = model.state_dict()
    loaded_elems = 0
    total_elems = 0
    matched_tensors = 0
    total_tensors = 0
    for k, v in model_sd.items():
        total_tensors += 1
        total_elems += v.numel()
        if (
            k in loaded_sd
            and hasattr(loaded_sd[k], "shape")
            and tuple(loaded_sd[k].shape) == tuple(v.shape)
        ):
            matched_tensors += 1
            loaded_elems += int(v.numel())
    return loaded_elems, total_elems, matched_tensors, total_tensors


def _maybe_remap_torchvision_efficientnet_head_keys(state: dict) -> dict:
    if not isinstance(state, dict):
        return state
    if "classifier.weight" in state and "classifier.1.weight" not in state:
        state = dict(state)
        state["classifier.1.weight"] = state.pop("classifier.weight")
    if "classifier.bias" in state and "classifier.1.bias" not in state:
        state = dict(state)
        state["classifier.1.bias"] = state.pop("classifier.bias")
    return state


def try_build_and_load(num_classes: int = 5):
    model_variants = []
    try:
        model_variants.append(
            ("efficientnet_pytorch", build_model_efficientnet_pytorch(num_classes))
        )
    except Exception:
        pass
    model_variants.append(("torchvision", build_model_torchvision(num_classes)))

    load_report = []
    state = None
    if CKPT_PATH is not None:
        ckpt = torch.load(CKPT_PATH, map_location="cpu")
        state = _extract_state_dict(ckpt)
        if state is not None:
            state = _clean_state_dict_keys(state)

    best_model = None
    best_tag = None
    best_missing = None
    best_unexpected = None
    best_strict = False
    best_loaded_frac = -1.0
    best_matched_tensors = -1

    for tag, m in model_variants:
        if state is None:
            best_model, best_tag = m, tag
            load_report.append(
                f"No checkpoint found; using random weights with {tag} (will score poorly)."
            )
            break

        state_for_variant = state
        if tag == "torchvision":
            state_for_variant = _maybe_remap_torchvision_efficientnet_head_keys(
                state_for_variant
            )

        try:
            m.load_state_dict(state_for_variant, strict=True)
            loaded_elems, total_elems, mt, tt = _count_loaded_params_by_shape(
                m, state_for_variant
            )
            best_model, best_tag = m, tag
            best_missing, best_unexpected = [], []
            best_strict = True
            best_loaded_frac = loaded_elems / max(1, total_elems)
            best_matched_tensors = mt
            load_report.append(
                f"Loaded checkpoint strictly using {tag} from: {CKPT_PATH} "
                f"(loaded_frac={best_loaded_frac:.4f}, matched_tensors={mt}/{tt})"
            )
            break
        except Exception as e_strict:
            try:
                incompat = m.load_state_dict(state_for_variant, strict=False)
                missing = list(getattr(incompat, "missing_keys", []))
                unexpected = list(getattr(incompat, "unexpected_keys", []))

                loaded_elems, total_elems, mt, tt = _count_loaded_params_by_shape(
                    m, state_for_variant
                )
                loaded_frac = loaded_elems / max(1, total_elems)

                if (loaded_frac > best_loaded_frac) or (
                    abs(loaded_frac - best_loaded_frac) < 1e-12
                    and mt > best_matched_tensors
                ):
                    best_model, best_tag = m, tag
                    best_missing, best_unexpected = missing, unexpected
                    best_strict = False
                    best_loaded_frac = loaded_frac
                    best_matched_tensors = mt

                load_report.append(
                    f"Tried {tag}: strict failed ({repr(e_strict)}); non-strict loaded_frac={loaded_frac:.4f}, "
                    f"matched_tensors={mt}/{tt}, missing={len(missing)}, unexpected={len(unexpected)}"
                )
            except Exception as e_nonstrict:
                load_report.append(
                    f"Tried {tag}: strict failed ({repr(e_strict)}); non-strict also failed ({repr(e_nonstrict)})"
                )

    return (
        best_model,
        best_tag,
        best_strict,
        best_missing,
        best_unexpected,
        best_loaded_frac,
        best_matched_tensors,
        load_report,
    )


(
    model,
    model_tag,
    strict_loaded,
    missing_keys,
    unexpected_keys,
    loaded_frac,
    matched_tensors,
    report,
) = try_build_and_load(num_classes=5)
for line in report:
    print(line)

if CKPT_PATH is not None:
    print(
        f"Selected model impl: {model_tag} (strict_loaded={strict_loaded}, loaded_frac≈{loaded_frac:.4f}, matched_tensors={matched_tensors})"
    )
    if not strict_loaded and missing_keys is not None and unexpected_keys is not None:
        print(
            f"  missing keys: {len(missing_keys)}, unexpected keys: {len(unexpected_keys)}"
        )

model = model.to(device).eval()



## === cell 3
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
    1, 3, 1, 1
)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
    1, 3, 1, 1
)

IMG_SIZE = 456


@torch.no_grad()
def process(image_bgr: np.ndarray) -> torch.Tensor:
    img = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
    x = (x - _IMAGENET_MEAN) / _IMAGENET_STD
    return x.to(device)




## === cell 4
files = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
if len(files) == 0:
    raise FileNotFoundError(f"No test images found in {TEST_DIR}")

sample_path = os.path.join(BASE, "sample_submission.csv")
if not os.path.isfile(sample_path):
    raise FileNotFoundError(f"sample_submission.csv not found at {sample_path}")

sample = pd.read_csv(sample_path)
id_to_path = {os.path.basename(p): p for p in files}

names, labels = [], []
fallback_label = 0  # updated after first successful prediction

for image_id in tqdm.tqdm(sample["image_id"].tolist(), total=len(sample)):
    file = id_to_path.get(image_id, None)
    if file is None:
        names.append(image_id)
        labels.append(fallback_label)
        continue

    img = cv2.imread(file)
    if img is None:
        names.append(image_id)
        labels.append(fallback_label)
        continue

    x = process(img)
    out = model(x)
    pred = int(torch.argmax(out, dim=1).detach().cpu().item())

    names.append(image_id)
    labels.append(pred)
    fallback_label = pred



## === cell 5
sub = pd.DataFrame({"image_id": names, "label": labels})

sub = sample[["image_id"]].merge(sub, on="image_id", how="left")
if sub["label"].isna().any():
    fill = int(pd.Series(labels).mode().iloc[0]) if len(labels) else 0
    sub["label"] = sub["label"].fillna(fill).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print(
    f"Wrote {out_path} with shape {sub.shape} (model_impl={model_tag}, ckpt={CKPT_PATH}, loaded_frac≈{loaded_frac:.4f}, matched_tensors={matched_tensors})"
)
print(sub.head())
