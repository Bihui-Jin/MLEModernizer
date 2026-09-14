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

0.8757932910244787

# 6. Current score

0.06726

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I remove the notebook-style `cd`/`pip` cells that don’t run in a plain Python Kaggle script and instead rely on already-installed Kaggle packages (torch/torchvision/timm/cv2). To fix the crash, I replace `efficientnet_pytorch` with the equivalent EfficientNet-B5 from `timm` and load your provided checkpoint with `strict=False` so it won’t error if key prefixes differ. I also fix a logic bug in `crop_image` (it referenced `img` instead of the passed `image`) and make the softmax call use an explicit `dim=1` to avoid runtime errors on newer torch. Finally, I ensure the submission is aligned to `sample_submission.csv` order and always writes `submission.csv`.'
- What this solution (achieved 0.17601) has done: 'I fix the immediate runtime blocker by removing the unavailable external checkpoints and ensuring the model objects are always defined, so inference can run end-to-end and write `submission.csv`. Because your current score (0.05531) is far below the target (0.8758), the safest minimal score-improving change is to use ImageNet-pretrained weights for the same backbones (EfficientNet-B5 and SEResNeXt101) rather than random initialization. I also make the MiDaS `torch.hub.load` robust by using the local cache when available (no internet) and falling back to a no-crop path if MiDaS cannot be loaded, so the script never crashes. The rest of your prediction loop and submission alignment to `sample_submission.csv` be preserved.'
- What this solution (achieved 0.05531) has done: 'Your score is far below the target, so the smallest safe way to move accuracy upward without changing the core model/loop is to (1) switch preprocessing to each timm model’s native `resolve_data_config` + `create_transform` (so normalization/resize match what pretrained weights expect) and (2) ensure the input color order is correct (timm expects RGB, while OpenCV loads BGR). I also keep your MiDaS cropping logic intact but apply it on the RGB image to avoid channel-mismatch artifacts that can hurt predictions. Finally, I keep the same ensemble averaging and submission alignment, only changing the input pipeline so the pretrained backbones are used correctly.'
- What this solution (achieved 0.25635) has done: 'I fix the runtime error by ensuring the `timm` transforms receive the expected input type (PIL Image), since the current transform pipeline is a torchvision Compose that doesn’t accept raw numpy arrays in this environment. This is a minimal change confined to the preprocessing function and does not alter the model architecture, ensemble logic, or prediction semantics. I also add a small safety fallback for empty/invalid crops to avoid rare downstream transform failures, while keeping the rest of the inference loop intact. The script then run end-to-end and write a valid `submission.csv` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.06726) has done: 'Your current score (0.25635) is far below the target (0.87579), so we should make a minimal change that legitimately increases accuracy without changing the core ensemble/model logic. The biggest likely issue is a mismatch between checkpoint keys and the `timm` model names (your `strict=False` load may silently skip most weights), so I add a tiny, safe state-dict normalization that strips common prefixes (`module.`, `model.`, etc.) and then re-load to maximize real checkpoint usage. I also make the EfficientNet variant fallback robust (try `tf_efficientnet_b5_ns` first, then `efficientnet_b5`) to better match common training backbones for such checkpoints, while keeping the same architecture family and inference loop. Finally, I add a deterministic test-time augmentation (single horizontal flip averaged) which is a minimal inference-only adjustment that typically improves accuracy without changing training or losses.'
- What this solution (achieved 0.06726) has done: 'Your current score (0.06726) is far below the target (0.87579), so the most likely cause is that the “provided checkpoints” aren’t actually being loaded into the timm models (most keys silently skipped), leaving you effectively with weak/random heads. I keep your exact ensemble + MiDaS-crop + flip-TTA inference logic, but (1) make checkpoint key remapping slightly smarter for common timm/DP variants and (2) explicitly report and enforce a minimum “load coverage” by trying a small set of backbone name aliases that often differ between training and timm naming (without changing the architecture family). If the checkpoints still don’t match, we fall back to pretrained ImageNet (as you already do), but this patch maximizes the chance you’re using the intended trained weights, which should move accuracy sharply upward toward the target. Submission writing and alignment to `sample_submission.csv` remain unchanged.'
- What this solution (achieved 0.06726) has done: 'Your score is extremely far below the target (0.067 vs 0.876), which strongly suggests the inference pipeline is effectively not using meaningful trained weights or is feeding the models inputs in a way that breaks the pretrained/finetuned expectations. To move accuracy upward with minimal semantic changes, I (1) improve checkpoint compatibility by auto-detecting and stripping the common `timm` classifier prefixes (`classifier.*`, `fc.*`, `head.*`) and remapping them to the current model’s classifier layer names, and (2) add a safe “test a few SEResNeXt aliases” fallback similar to what you already do for EfficientNet so more checkpoints actually load. I keep the same models/ensemble/cropping/TTA and only add small, directly relevant weight-loading logic plus a tiny fix to ensure MiDaS receives the expected input type (PIL/normalized tensor) consistently to avoid producing a near-empty crop mask.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from PIL import Image

warnings.filterwarnings("ignore")

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

EFF_CKPT = "/kaggle/input/ensemblev5/eff_best.pth"
SE_CKPT = "/kaggle/input/ensemblev5/seresnext_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.manual_seed(0)
np.random.seed(0)
if device.type == "cuda":
    torch.cuda.manual_seed_all(0)




## === cell 1
midas = None
transform = None
try:
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS", pretrained=True)
    midas.to(device).eval()
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = midas_transforms.default_transform
    print("Loaded MiDaS from torch.hub cache.")
except Exception as e:
    print("MiDaS not available (offline/no cache). Will skip depth-based cropping.")
    print("MiDaS load error:", repr(e))
    midas = None
    transform = None

_eff_candidates = [
    "tf_efficientnet_b5_ns",
    "tf_efficientnet_b5",
    "efficientnet_b5",
]
efficient = None
eff_model_name = None
for _name in _eff_candidates:
    try:
        efficient = timm.create_model(_name, pretrained=True, num_classes=5)
        eff_model_name = _name
        break
    except Exception:
        continue
if efficient is None:
    raise RuntimeError("Could not create an EfficientNet-B5 model from timm.")

_se_candidates = [
    "seresnext101_32x4d",
    "legacy_seresnext101_32x4d",
]
seresnext = None
se_model_name = None
for _name in _se_candidates:
    try:
        seresnext = timm.create_model(_name, pretrained=True, num_classes=5)
        se_model_name = _name
        break
    except Exception:
        continue
if seresnext is None:
    raise RuntimeError("Could not create a SEResNeXt101_32x4d model from timm.")


def _normalize_state_dict_keys(state):
    """
    Change (score-up, minimal): broaden key normalization so more real checkpoint weights load.
    This preserves the exact model/loop logic; it only increases the fraction of matching parameters.
    """
    if not isinstance(state, dict):
        return state

    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if "net" in state and isinstance(state["net"], dict):
        state = state["net"]

    cleaned = {}
    for k, v in state.items():
        nk = k

        for pref in (
            "module.",
            "model.",
            "net.",
            "encoder.",
            "backbone.",
            "student.",
        ):
            if nk.startswith(pref):
                nk = nk[len(pref) :]

        while nk.startswith("model."):
            nk = nk[len("model.") :]

        cleaned[nk] = v
    return cleaned


def _infer_classifier_prefixes(model_sd_keys):
    """
    Minimal helper to identify the classifier layer prefix used by the current timm model.
    """
    candidates = []
    for pref in (
        "classifier.",
        "fc.",
        "head.",
        "head.fc.",
        "model.classifier.",
        "model.fc.",
        "model.head.",
    ):
        for k in model_sd_keys:
            if k.startswith(pref):
                candidates.append(pref)
                break
    for pref in ("classifier.", "fc.", "head."):
        if pref in candidates:
            return pref
    return candidates[0] if candidates else None


def _remap_head_keys_to_model(sd, model):
    """
    Change (score-up, minimal): remap checkpoint head names (classifier/fc/head) to this model's head prefix
    when shapes match. This keeps architecture identical but increases real checkpoint utilization.
    """
    if not isinstance(sd, dict) or len(sd) == 0:
        return sd

    model_sd = model.state_dict()
    model_keys = list(model_sd.keys())
    dst_pref = _infer_classifier_prefixes(model_keys)
    if dst_pref is None:
        return sd

    src_prefs = ("classifier.", "fc.", "head.", "head.fc.")
    remapped = dict(sd)
    for sp in src_prefs:
        for k in list(sd.keys()):
            if not k.startswith(sp):
                continue
            tail = k[len(sp) :]
            newk = dst_pref + tail
            if (
                newk in model_sd
                and hasattr(sd[k], "shape")
                and hasattr(model_sd[newk], "shape")
            ):
                if tuple(sd[k].shape) == tuple(model_sd[newk].shape):
                    remapped[newk] = sd[k]
    return remapped


def _ckpt_coverage(model, sd):
    model_sd = model.state_dict()
    match = 0
    total = 0
    for k, v in model_sd.items():
        total += 1
        if k in sd and hasattr(sd[k], "shape") and sd[k].shape == v.shape:
            match += 1
    return match / max(total, 1)


def _load_ckpt_with_stats(model, ckpt_path, tag):
    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path} (using pretrained {tag} weights)")
        return {"loaded": False, "missing": None, "unexpected": None, "coverage": 0.0}

    raw = torch.load(ckpt_path, map_location="cpu")
    sd = _normalize_state_dict_keys(raw)

    sd = _remap_head_keys_to_model(sd, model)

    coverage = _ckpt_coverage(model, sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"Loaded {tag} checkpoint from {ckpt_path}. "
        f"missing={len(missing)}, unexpected={len(unexpected)}, coverage={coverage:.3f}"
    )
    return {
        "loaded": True,
        "missing": missing,
        "unexpected": unexpected,
        "coverage": coverage,
    }


eff_stats = _load_ckpt_with_stats(
    efficient, EFF_CKPT, f"EfficientNet({eff_model_name})"
)

if eff_stats["loaded"] and eff_stats["coverage"] < 0.20:
    print(
        f"EfficientNet checkpoint coverage is low ({eff_stats['coverage']:.3f}). "
        f"Trying other EfficientNet-B5 aliases for better key alignment..."
    )
    best = (eff_stats["coverage"], efficient, eff_model_name)
    for _name in _eff_candidates:
        if _name == eff_model_name:
            continue
        try:
            m = timm.create_model(_name, pretrained=True, num_classes=5)
        except Exception:
            continue
        st = _load_ckpt_with_stats(m, EFF_CKPT, f"EfficientNet({ _name })")
        if st["loaded"] and st["coverage"] > best[0]:
            best = (st["coverage"], m, _name)
    if best[1] is not efficient:
        efficient = best[1]
        eff_model_name = best[2]
        print(
            f"Using EfficientNet alias '{eff_model_name}' with improved coverage={best[0]:.3f}"
        )

se_stats = _load_ckpt_with_stats(seresnext, SE_CKPT, f"SEResNeXt({se_model_name})")

if se_stats["loaded"] and se_stats["coverage"] < 0.20:
    print(
        f"SEResNeXt checkpoint coverage is low ({se_stats['coverage']:.3f}). "
        f"Trying other SEResNeXt aliases for better key alignment..."
    )
    best = (se_stats["coverage"], seresnext, se_model_name)
    for _name in _se_candidates:
        if _name == se_model_name:
            continue
        try:
            m = timm.create_model(_name, pretrained=True, num_classes=5)
        except Exception:
            continue
        st = _load_ckpt_with_stats(m, SE_CKPT, f"SEResNeXt({ _name })")
        if st["loaded"] and st["coverage"] > best[0]:
            best = (st["coverage"], m, _name)
    if best[1] is not seresnext:
        seresnext = best[1]
        se_model_name = best[2]
        print(
            f"Using SEResNeXt alias '{se_model_name}' with improved coverage={best[0]:.3f}"
        )

efficient.to(device).eval()
seresnext.to(device).eval()

eff_cfg = resolve_data_config({}, model=efficient)
se_cfg = resolve_data_config({}, model=seresnext)
eff_tf = create_transform(**eff_cfg, is_training=False)
se_tf = create_transform(**se_cfg, is_training=False)

print("Models and timm transforms have been loaded...\n")




## === cell 2
def _to_pil_rgb(image_rgb: np.ndarray) -> Image.Image:
    if (
        image_rgb is None
        or not isinstance(image_rgb, np.ndarray)
        or image_rgb.size == 0
    ):
        image_rgb = np.zeros(
            (
                eff_cfg.get("input_size", (3, 456, 456))[1],
                eff_cfg.get("input_size", (3, 456, 456))[2],
                3,
            ),
            dtype=np.uint8,
        )
    if image_rgb.dtype != np.uint8:
        image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)
    return Image.fromarray(image_rgb, mode="RGB")


def processor(image_rgb: np.ndarray) -> torch.Tensor:
    pil = _to_pil_rgb(image_rgb)
    x_eff = eff_tf(pil)  # CHW float tensor
    x_se = se_tf(pil)
    return x_eff.unsqueeze(0).to(device), x_se.unsqueeze(0).to(device)


@torch.no_grad()
def get_depth(img_rgb: np.ndarray) -> np.ndarray:
    if midas is None or transform is None:
        h, w = img_rgb.shape[:2]
        return np.ones((h, w), dtype=bool)

    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb, 0, 255).astype(np.uint8)
    img_rgb = np.ascontiguousarray(img_rgb)

    input_batch = transform(img_rgb).to(device)
    prediction = midas(input_batch)
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
    )
    output = prediction.detach().cpu().numpy()
    img_min = float(np.min(output))
    img_max = float(np.max(output))
    return output > ((img_min + img_max) / 3.0)


def crop_image(image: np.ndarray, depth: np.ndarray) -> np.ndarray:
    depth = depth.astype(np.uint8)
    mask_3d = np.stack((depth, depth, depth), axis=2)
    masked_arr = np.where(mask_3d == 1, image, 0).astype(np.uint8)

    coords = np.where(np.any(masked_arr != 0, axis=2))
    if coords[0].size == 0 or coords[1].size == 0:
        return image

    y_min, y_max = int(coords[0].min()), int(coords[0].max())
    x_min, x_max = int(coords[1].min()), int(coords[1].max())

    y_min = max(0, y_min)
    x_min = max(0, x_min)
    y_max = min(image.shape[0] - 1, y_max)
    x_max = min(image.shape[1] - 1, x_max)

    if y_max <= y_min or x_max <= x_min:
        return image

    cropped = masked_arr[y_min : y_max + 1, x_min : x_max + 1]
    if (
        cropped is None
        or cropped.size == 0
        or cropped.shape[0] < 2
        or cropped.shape[1] < 2
    ):
        return image
    return cropped




## === cell 3
files = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
if len(files) == 0:
    raise FileNotFoundError(f"No test images found in {TEST_IMG_DIR}")

names, labels = [], []

with torch.no_grad():
    for file in files:
        img_bgr = cv2.imread(file)
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        depth = get_depth(img_rgb)
        img_cropped = crop_image(img_rgb, depth)

        eff_t, se_t = processor(img_cropped)
        se_out = seresnext(se_t)
        eff_out = efficient(eff_t)
        total1 = (se_out + eff_out) / 2.0

        img_flip = np.ascontiguousarray(img_cropped[:, ::-1, :])
        eff_t2, se_t2 = processor(img_flip)
        se_out2 = seresnext(se_t2)
        eff_out2 = efficient(eff_t2)
        total2 = (se_out2 + eff_out2) / 2.0

        total = (total1 + total2) / 2.0
        pred = int(torch.argmax(torch.softmax(total, dim=1), dim=1).item())

        names.append(os.path.basename(file))
        labels.append(pred)

print("Predictions:", len(labels))




## === cell 4
sub = pd.read_csv(SAMPLE_SUB_PATH)
pred_df = pd.DataFrame({"image_id": names, "label": labels})

sub = sub[["image_id"]].merge(pred_df, on="image_id", how="left")

if sub["label"].isna().any():
    fallback = int(pred_df["label"].mode().iloc[0]) if len(pred_df) else 0
    sub["label"] = sub["label"].fillna(fallback).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote", out_path)
print(sub.head())
