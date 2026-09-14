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

0.2272

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.38117) has done: 'I make the model checkpoint loading robust so the notebook runs even when the external `/kaggle/input/ensemblev5/*.pth` files are unavailable, by falling back to `pretrained=True` weights (same architectures, same inference logic). I also fix the downstream `NameError` by ensuring the model variables are always defined and moved to the correct device. To avoid internet dependency/timeouts, I disable MiDaS depth-cropping by default (it currently uses `torch.hub` which can fail offline), while keeping the rest of preprocessing and ensemble averaging unchanged. Finally, I keep the submission creation logic but ensure it always writes a valid `submission.csv` with the required columns and row order.'
- What this solution (achieved 0.1136) has done: 'Your current low score is mainly because the “fallback” models are ImageNet-pretrained but not cassava-fine-tuned, so accuracy collapses. To move toward the target with minimal logic changes, I (1) load the official cassava fine-tuned weights from Kaggle’s public `cassava-model` dataset if present, keeping the same two-model ensemble and identical inference flow, and only fall back to ImageNet if those weights truly aren’t available. I also (2) fix a bug in `load_state_dict_forgiving` where non-`module.` keys get mishandled, which can silently break checkpoint loading and hurt accuracy. Everything else (preprocessing, MiDaS disabled by default, averaging logits, submission formatting) stays the same.'
- What this solution (achieved 0.10575) has done: 'Your score is far below the target, so the most likely issue is that the cassava-finetuned checkpoints are still not being found/loaded and you’re effectively submitting ImageNet-pretrained predictions. I make checkpoint discovery robust by searching all plausible Kaggle input locations (including nested competition folder copies) and also accept common filename variants, while keeping the exact same two-model ensemble and inference logic. I also harden checkpoint loading to correctly handle common checkpoint formats (`state_dict`, `model`, `model_state_dict`) and strip prefixes (`module.`, `model.`) without altering any other behavior. These minimal changes should substantially increase accuracy toward your target if the finetuned weights are present anywhere in `/kaggle/input`.'
- What this solution (achieved 0.2272) has done: 'Your current score strongly suggests the finetuned cassava checkpoints still aren’t being loaded (so you’re effectively using ImageNet-pretrained classifiers). I keep the exact same two-model timm ensemble and inference flow, but (1) broaden checkpoint discovery to include common file extensions and names, and (2) make the state_dict loading more robust to additional real-world key prefixes (e.g., `backbone.` / `encoder.`) that can otherwise cause most weights to remain randomly initialized. Finally, I add a small sanity print that reports whether each model ended up using finetuned weights vs fallback, so you can confirm the fix correlates with the score jump toward the target.'

# 9. Code solution

## === cell 0
import os
import glob
import json
import warnings

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import timm

warnings.filterwarnings("ignore")

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(DATA_DIR), f"Missing DATA_DIR: {DATA_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing TEST_IMG_DIR: {TEST_IMG_DIR}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv: {SAMPLE_SUB_PATH}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)




## === cell 1
def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for key in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
    return ckpt_obj


def load_state_dict_forgiving(model: torch.nn.Module, checkpoint_path: str):
    """
    Score fix: ensure checkpoint loading actually maps keys correctly and robustly.
    - Accepts common checkpoint formats (state_dict/model/model_state_dict).
    - Strips common prefixes without corrupting keys.
    This is directly relevant because if prefixes aren't stripped, most weights won't load,
    making accuracy collapse (matching the current very low score).
    """
    if checkpoint_path is None or (not os.path.exists(checkpoint_path)):
        raise FileNotFoundError(checkpoint_path)

    sd = torch.load(checkpoint_path, map_location="cpu")
    sd = _extract_state_dict(sd)

    if not isinstance(sd, dict):
        raise ValueError(f"Checkpoint at {checkpoint_path} is not a state_dict dict.")

    strip_prefixes = [
        "module.",
        "model.",
        "net.",
        "backbone.",
        "encoder.",
    ]

    def _strip(k: str) -> str:
        changed = True
        while changed:
            changed = False
            for p in strip_prefixes:
                if k.startswith(p):
                    k = k[len(p) :]
                    changed = True
        return k

    new_sd = {}
    for k, v in sd.items():
        if not isinstance(k, str):
            continue
        k2 = _strip(k)
        new_sd[k2] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(
        f"Loaded {os.path.basename(checkpoint_path)}: missing={len(missing)}, unexpected={len(unexpected)}"
    )
    return model


def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def expand_ckpt_candidates(candidate_files):
    """
    Score fix: search for finetuned weights across typical Kaggle mount points.
    Minimal change: just broaden filename patterns/extensions so we actually find
    the finetuned cassava weights if present anywhere under /kaggle/input.
    """
    roots = [
        "/kaggle/input",
        "/kaggle/input/cassava-model",
        "/kaggle/input/ensemblev5",
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
    out = []
    out.extend(candidate_files)

    exts = [".pth", ".pt", ".bin"]

    basenames = list({os.path.basename(p) for p in candidate_files if p})
    for r in roots:
        for bn in basenames:
            out.append(os.path.join(r, bn))
            out.append(os.path.join(r, "**", bn))

            base, ext = os.path.splitext(bn)
            if ext.lower() in exts:
                for e in exts:
                    out.append(os.path.join(r, base + e))
                    out.append(os.path.join(r, "**", base + e))

    alt_names = [
        "eff_best.pth",
        "eff_best.pt",
        "eff_best.bin",
        "efficientnet_b5.pth",
        "tf_efficientnet_b5.pth",
        "tf_efficientnet_b5_ns.pth",
        "efficientnetb5.pth",
        "eff.pth",
        "seresnext_best.pth",
        "seresnext_best.pt",
        "seresnext_best.bin",
        "seresnext101_32x4d.pth",
        "seresnext101.pth",
        "se_resnext101_32x4d.pth",
        "seresnext.pth",
    ]
    for r in roots:
        for bn in alt_names:
            out.append(os.path.join(r, bn))
            out.append(os.path.join(r, "**", bn))

    expanded = []
    for p in out:
        if "*" in p:
            expanded.extend(sorted(glob.glob(p, recursive=True)))
        else:
            expanded.append(p)

    seen = set()
    uniq = []
    for p in expanded:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


eff_ckpt_candidates = expand_ckpt_candidates(
    [
        "/kaggle/input/ensemblev5/eff_best.pth",
        "/kaggle/input/cassava-model/eff_best.pth",
        "/kaggle/input/cassava-model/tf_efficientnet_b5.pth",
        "/kaggle/input/cassava-model/efficientnet_b5.pth",
    ]
)
se_ckpt_candidates = expand_ckpt_candidates(
    [
        "/kaggle/input/ensemblev5/seresnext_best.pth",
        "/kaggle/input/cassava-model/seresnext_best.pth",
        "/kaggle/input/cassava-model/seresnext101_32x4d.pth",
        "/kaggle/input/cassava-model/seresnext101.pth",
    ]
)

eff_ckpt = find_first_existing(eff_ckpt_candidates)
se_ckpt = find_first_existing(se_ckpt_candidates)

print("EfficientNet ckpt:", eff_ckpt)
print("SE-ResNeXt ckpt:", se_ckpt)

efficient = timm.create_model("tf_efficientnet_b5", pretrained=False, num_classes=5)
seresnext = timm.create_model("seresnext101_32x4d", pretrained=False, num_classes=5)

efficient_loaded_finetuned = False
seresnext_loaded_finetuned = False

if eff_ckpt is not None:
    try:
        efficient = load_state_dict_forgiving(efficient, eff_ckpt)
        efficient_loaded_finetuned = True
        print("EfficientNet checkpoint loaded.")
    except Exception as e:
        print(
            "EfficientNet checkpoint failed; falling back to pretrained=True. Error:",
            repr(e),
        )
        efficient = timm.create_model(
            "tf_efficientnet_b5", pretrained=True, num_classes=5
        )
else:
    print("EfficientNet checkpoint not found; using pretrained=True fallback.")
    efficient = timm.create_model("tf_efficientnet_b5", pretrained=True, num_classes=5)

if se_ckpt is not None:
    try:
        seresnext = load_state_dict_forgiving(seresnext, se_ckpt)
        seresnext_loaded_finetuned = True
        print("SE-ResNeXt checkpoint loaded.")
    except Exception as e:
        print(
            "SE-ResNeXt checkpoint failed; falling back to pretrained=True. Error:",
            repr(e),
        )
        seresnext = timm.create_model(
            "seresnext101_32x4d", pretrained=True, num_classes=5
        )
else:
    print("SE-ResNeXt checkpoint not found; using pretrained=True fallback.")
    seresnext = timm.create_model("seresnext101_32x4d", pretrained=True, num_classes=5)

efficient.eval().to(device)
seresnext.eval().to(device)

print("Models are ready.")
print(
    f"Finetuned loaded? efficient={efficient_loaded_finetuned}, seresnext={seresnext_loaded_finetuned}\n"
)



## === cell 2
USE_MIDAS = False
midas = None
transform = None

if USE_MIDAS:
    try:
        midas = torch.hub.load("intel-isl/MiDaS", "MiDaS", pretrained=True)
        midas.to(device).eval()
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = midas_transforms.default_transform
        print("MiDaS loaded.")
    except Exception as e:
        USE_MIDAS = False
        print(
            "MiDaS could not be loaded; proceeding without depth-based cropping.\nError:",
            repr(e),
        )


def processor(image_bgr: np.ndarray) -> torch.Tensor:
    img = (
        cv2.resize(image_bgr, (512, 512), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        / 255.0
    )
    img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225], dtype=np.float32
    )
    image = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    return image


def get_depth(img_bgr: np.ndarray) -> np.ndarray:
    if (not USE_MIDAS) or (midas is None) or (transform is None):
        h, w = img_bgr.shape[:2]
        return np.ones((h, w), dtype=bool)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = (
            F.interpolate(
                prediction.unsqueeze(1),
                size=img_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze(0)
            .squeeze(0)
        )
    output = prediction.detach().float().cpu().numpy()
    img_min = float(np.min(output))
    img_max = float(np.max(output))
    thr = (img_min + img_max) / 3.0
    return output > thr


def crop_image(image_bgr: np.ndarray, depth_mask: np.ndarray) -> np.ndarray:
    depth = depth_mask.astype(np.uint8)
    mask_3d = np.stack((depth, depth, depth), axis=2)
    masked_arr = np.where(mask_3d == 1, image_bgr, 0).astype(np.uint8)

    c = np.where(masked_arr != 0)
    if c[0].size == 0 or c[1].size == 0:
        return image_bgr

    x_max = int(np.max(c[1]))
    x_min = int(np.min(c[1]))
    y_max = int(np.max(c[0]))
    y_min = int(np.min(c[0]))

    h, w = image_bgr.shape[:2]
    x_min = max(0, min(x_min, w - 1))
    x_max = max(0, min(x_max, w - 1))
    y_min = max(0, min(y_min, h - 1))
    y_max = max(0, min(y_max, h - 1))
    if x_max <= x_min or y_max <= y_min:
        return image_bgr

    return masked_arr[y_min:y_max, x_min:x_max]




## === cell 3
sample_df = pd.read_csv(SAMPLE_SUB_PATH)
test_image_ids = sample_df["image_id"].tolist()

test_paths = [os.path.join(TEST_IMG_DIR, iid) for iid in test_image_ids]
missing_files = [p for p in test_paths if not os.path.exists(p)]
assert len(missing_files) == 0, f"Missing test images (first 5): {missing_files[:5]}"

names, labels = [], []

with torch.no_grad():
    for file_path, image_id in zip(test_paths, test_image_ids):
        img = cv2.imread(file_path)
        if img is None:
            names.append(image_id)
            labels.append(0)
            continue

        depth = get_depth(img)
        img_c = crop_image(img, depth)
        x = processor(img_c)

        se_out = seresnext(x)
        eff_out = efficient(x)
        total = (se_out + eff_out) / 2.0

        pred = int(torch.argmax(F.softmax(total, dim=1), dim=1).detach().cpu().item())

        names.append(image_id)
        labels.append(pred)

sub_df = pd.DataFrame({"image_id": names, "label": labels})
sub_df = sub_df.set_index("image_id").loc[test_image_ids].reset_index()

out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub_df.head())
print("rows:", len(sub_df))
assert out_path.endswith(".csv") and os.path.exists(out_path)
assert list(sub_df.columns) == ["image_id", "label"]
assert len(sub_df) == len(sample_df)
