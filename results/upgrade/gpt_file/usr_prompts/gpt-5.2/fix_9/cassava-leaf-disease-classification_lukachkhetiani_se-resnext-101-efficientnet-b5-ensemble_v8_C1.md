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

0.7234814143245694

# 6. Current score

0.09903

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11136) has done: 'I fix the pipeline so it always runs end-to-end and writes a valid `submission.csv` with exactly the same `image_id` ordering/length as `sample_submission.csv`. The main blocker is missing external weight files (`/kaggle/input/ensemble/...`), so I add a safe fallback: use `timm` pretrained backbones (still the same two-model ensemble) when those weights aren’t present, and print which path was used. I also make inference robust to any unreadable/missing images by falling back to the sample row’s label (keeps submission length valid) instead of crashing, which fixes the length assertion and Kaggle “same length” error. These changes preserve the core logic (two timm models + average logits + argmax) while ensuring a valid submission is always produced.'
- What this solution (achieved 0.11136) has done: 'Your current score (0.11136) is far below the target (0.72348), so we should improve performance rather than just ensuring a valid CSV. The biggest issue is that when your private ensemble weights are missing, the fallback uses generic ImageNet-pretrained classifiers with a randomly initialized 5-class head, which produces near-random predictions and explains the ~0.11 accuracy. With minimal change to the existing two-model average-logits + argmax core logic, I switch the fallback to load publicly available Cassava-finetuned weights shipped inside the competition dataset (`/kaggle/input/.../*.pth`) when your external `/kaggle/input/ensemble/...` files aren’t present. This keeps the same architecture and inference flow, but replaces random heads with task-trained heads, which should move accuracy sharply upward toward (and likely within) the target band.'
- What this solution (achieved 0.11136) has done: 'Your score is far below the target, and the main reason is that the current “fallback” almost certainly never finds Cassava-finetuned weights inside the competition dataset, leaving you with ImageNet backbones plus a randomly initialized 5-class head (near-random ~0.11 accuracy). I keep your exact core logic (two timm models, average logits, argmax) but make the weight fallback actually discover and load any `.pth/.pt/.bin` checkpoints that exist under `/kaggle/input` (including the dataset folder), preferring ones that look like efficientnet/hrnet and then loading them with the same `_clean_state_dict` routine. I also make the weight loader slightly more robust by explicitly reinitializing the classifier head to 5 classes after loading when a checkpoint has a mismatched head, so the model doesn’t silently keep an incompatible classifier. This should move accuracy sharply upward toward your target while preserving the architecture/inference semantics.'
- What this solution (achieved 0.11136) has done: 'Your score is far below the target, and the likely root cause is still that no Cassava-finetuned weights are being found/loaded, leaving the 5-class heads effectively random. I keep your exact two-model timm ensemble + average logits + argmax pipeline, but make the checkpoint discovery both more targeted (avoid scanning all of `/kaggle/input`, which is slow/noisy) and more likely to match real cassava finetuned checkpoint names by searching the dataset folder plus the optional `/kaggle/input/ensemble` folder and ranking candidates. I also harden weight loading to handle common checkpoint formats (state_dict nested, EMA keys) and ensure classifier head compatibility without changing inference semantics. These minimal changes should move accuracy sharply upward toward your target while still producing the same valid `submission.csv` format and order.'
- What this solution (achieved 0.11136) has done: 'Your current gap to the target is large (0.11136 vs 0.72348, higher-is-better), and the most likely cause is still that no Cassava-finetuned checkpoints are being loaded, leaving the 5-class heads effectively random. I keep your exact core logic (two timm models, average logits, argmax) but make the checkpoint discovery reliably find real finetuned weights by searching `/kaggle/input` for plausible Cassava checkpoints while staying fast via filename filtering and a capped candidate set. I also harden loading by (a) inferring the correct architecture from the checkpoint’s tensor shapes (EffNet-B5 vs HRNet-W48) to avoid mismatched loads and (b) always recreating the model with the checkpoint’s `num_classes` (usually 5) so the head loads instead of being skipped. These are minimal, inference-only changes and should move accuracy sharply upward toward the target band without changing submission semantics.'
- What this solution (achieved 0.11136) has done: 'Your score is far below the target (0.11136 vs 0.72348), and the most likely reason is that your fallback still isn’t loading any Cassava-finetuned checkpoints, so you’re effectively running ImageNet backbones with a random 5-class head. Keeping your exact core logic (two timm models, average logits, argmax), I make checkpoint discovery both faster and much more likely to find real finetuned weights by (1) scanning only `/kaggle/input/ensemble` and `/kaggle/input` with stronger filename filters, and (2) validating candidates by actually attempting a partial `load_state_dict` into the expected architecture and selecting the one with the best key-match ratio. I also fix a subtle but important issue: you currently reset the classifier to 5 *after* loading when `ncls != 5`, which discards loaded classifier weights; instead we always build with `num_classes=5` for Cassava and only ignore head keys when mismatched. These are inference-only changes, preserve evaluation semantics, and should move accuracy sharply upward toward the target band if any finetuned checkpoints exist in the environment.'
- What this solution (achieved 0.09903) has done: 'Your current score (0.11136) is far below the target (0.72348), so we should improve accuracy rather than further hardening the CSV-writing. The biggest likely issue is that your fallback still ends up using ImageNet-pretrained backbones with a randomly initialized 5-class head (because `timm.create_model(..., pretrained=True, num_classes=5)` does not load a trained 5-class head), which yields near-random predictions. With minimal changes and keeping the exact two-model ensemble + average logits + argmax core logic, I (1) always construct fallback models with `num_classes=1000`, then replace the classifier layer to 5 classes and load any matching 5-class checkpoint if found; and (2) expand checkpoint discovery to explicitly include common filenames and allow slightly smaller checkpoints, increasing the chance of finding usable Cassava-finetuned weights shipped in attached inputs. These changes keep your inference semantics but make it much more likely that the head is trained (or at least not purely random), moving accuracy toward the target band.'

# 9. Code solution

## === cell 0
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import timm
import tqdm
import glob
from pathlib import Path
import re

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)


def _clean_state_dict(sd):
    """Robustly extract a usable state_dict from common checkpoint formats."""
    if not isinstance(sd, dict):
        return sd

    for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
        if k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break

    for k in ["state_dict_ema", "model_ema", "ema"]:
        if k in sd and isinstance(sd[k], dict):
            sd = sd[k]
            break

    if isinstance(sd, dict) and sd and any(isinstance(k, str) for k in sd.keys()):
        if any(str(k).startswith("module.") for k in sd.keys()):
            sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        if any(str(k).startswith("model.") for k in sd.keys()):
            sd = {k.replace("model.", "", 1): v for k, v in sd.items()}

    return sd


def _infer_arch_from_sd(sd):
    keys = list(sd.keys())
    if any(("stages." in k or "transition" in k or "hrnet" in k) for k in keys):
        return "hrnet_w48"
    if any(("conv_stem" in k or "blocks." in k) for k in keys):
        return "tf_efficientnet_b5"
    if "conv_stem.weight" in sd:
        return "tf_efficientnet_b5"
    if any(k.startswith("stem.") for k in keys) and any(
        k.startswith("stages.") for k in keys
    ):
        return "hrnet_w48"
    return None


def load_weights_if_exists(model, weight_path: str):
    """
    Robustness: handle nested checkpoint dicts + common key prefixes; tolerate head mismatch with strict=False.
    """
    if weight_path and os.path.exists(weight_path):
        ckpt = torch.load(weight_path, map_location="cpu")
        sd = _clean_state_dict(ckpt)

        missing, unexpected = model.load_state_dict(sd, strict=False)

        if missing or unexpected:
            print(
                f"Warning loading {os.path.basename(weight_path)}: missing={len(missing)} unexpected={len(unexpected)}"
            )
        print(f"Loaded weights: {weight_path}")
    else:
        print(f"Weight file not found, using model as-initialized: {weight_path}")
    return model


def process(image_bgr):
    img = cv2.resize(image_bgr, (384, 384), interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    x = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(device) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x




## === cell 1
eff_weight_path = "/kaggle/input/ensemble/eff_model_last_adam.pth"
hr_weight_path = "/kaggle/input/ensemble/hr_model_last_adam.pth"


def _fast_find_candidate_checkpoints(
    search_roots, exts=(".pth", ".pt", ".bin"), max_files=300
):
    """
    Score fix: increase probability of finding cassava-finetuned checkpoints by:
    - searching common roots (still bounded),
    - widening filename patterns slightly,
    - allowing somewhat smaller checkpoints (some finetuned heads-only or compact formats).
    """
    out = []
    pat = re.compile(
        r"(cassava|leaf|disease|cbsd|cbb|cgm|cmd|healthy|cls|classif|efficientnet|effnet|tf_efficientnet|hrnet|w48|b5|fold|best|final|epoch|checkpoint|ckpt|model)",
        re.IGNORECASE,
    )
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        root = str(root)
        for ext in exts:
            for p in glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True):
                bn = os.path.basename(p)
                if not pat.search(bn):
                    continue
                try:
                    sz = os.path.getsize(p)
                    if sz < 500_000:
                        continue
                except Exception:
                    pass
                out.append(p)
                if len(out) >= max_files:
                    break
            if len(out) >= max_files:
                break
        if len(out) >= max_files:
            break

    seen, dedup = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup


def _try_score_ckpt_for_arch(arch, ckpt_path, num_classes=5):
    """
    Score fix: validate candidates by key/shape match ratio for backbone weights,
    ignoring head keys (common mismatch across checkpoints).
    """
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = _clean_state_dict(ckpt)
        if not isinstance(sd, dict) or len(sd) == 0:
            return -1.0

        m = timm.create_model(arch, pretrained=False, num_classes=num_classes)
        msd = m.state_dict()

        ignore_prefixes = ("classifier.", "head.", "fc.", "global_pool.")
        sd_keys = [
            k
            for k in sd.keys()
            if isinstance(k, str) and not k.startswith(ignore_prefixes)
        ]
        if not sd_keys:
            return -1.0

        matched = 0
        for k in sd_keys:
            v = sd.get(k, None)
            if (
                k in msd
                and torch.is_tensor(v)
                and torch.is_tensor(msd[k])
                and tuple(v.shape) == tuple(msd[k].shape)
            ):
                matched += 1
        return matched / max(1, len(sd_keys))
    except Exception:
        return -1.0


def _pick_best_ckpt_by_loadability(candidates, arch, min_ratio=0.15):
    best_p, best_s = None, -1.0
    for p in candidates:
        s = _try_score_ckpt_for_arch(arch, p, num_classes=5)
        if s > best_s:
            best_p, best_s = p, s
    if best_s >= min_ratio:
        return best_p, best_s
    return None, best_s


all_ckpts = _fast_find_candidate_checkpoints(
    search_roots=[
        Path("/kaggle/input/ensemble"),
        Path("/kaggle/input/cassava-leaf-disease-classification"),
        Path("/kaggle/input"),
    ],
    max_files=300,
)

eff_to_load = eff_weight_path if os.path.exists(eff_weight_path) else None
hr_to_load = hr_weight_path if os.path.exists(hr_weight_path) else None

if eff_to_load is None:
    eff_to_load, eff_ratio = _pick_best_ckpt_by_loadability(
        all_ckpts, "tf_efficientnet_b5", min_ratio=0.15
    )
else:
    eff_ratio = None

if hr_to_load is None:
    hr_to_load, hr_ratio = _pick_best_ckpt_by_loadability(
        all_ckpts, "hrnet_w48", min_ratio=0.15
    )
else:
    hr_ratio = None


def _reset_classifier_to_num_classes(model, num_classes=5):
    if hasattr(model, "reset_classifier"):
        model.reset_classifier(num_classes=num_classes)
        return model
    if hasattr(model, "classifier") and isinstance(model.classifier, torch.nn.Module):
        in_features = getattr(model.classifier, "in_features", None)
        if in_features is not None:
            model.classifier = torch.nn.Linear(in_features, num_classes)
            return model
    if hasattr(model, "fc") and isinstance(model.fc, torch.nn.Module):
        in_features = getattr(model.fc, "in_features", None)
        if in_features is not None:
            model.fc = torch.nn.Linear(in_features, num_classes)
            return model
    return model


def _build_model_for_ckpt(default_arch, ckpt_path):
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = _clean_state_dict(ckpt)
        arch = _infer_arch_from_sd(sd) or default_arch
        m = timm.create_model(arch, pretrained=False, num_classes=5)
        m = load_weights_if_exists(m, ckpt_path)
        return m.to(device).eval()

    m = timm.create_model(default_arch, pretrained=True, num_classes=1000)
    m = _reset_classifier_to_num_classes(m, num_classes=5)
    return m.to(device).eval()


efficient = _build_model_for_ckpt("tf_efficientnet_b5", eff_to_load)
hrnet = _build_model_for_ckpt("hrnet_w48", hr_to_load)

print("Models have been loaded...\n")
print(f"Effective eff weights used: {eff_to_load} (match_ratio={eff_ratio})")
print(f"Effective hr weights used: {hr_to_load} (match_ratio={hr_ratio})")




## === cell 2
sample = pd.read_csv(SAMPLE_SUB_PATH)
test_names = sample["image_id"].tolist()

names, labels = [], []

with torch.no_grad():
    for image_id in tqdm.tqdm(test_names):
        file_path = os.path.join(TEST_IMG_DIR, image_id)
        img = cv2.imread(file_path)

        if img is None:
            pred = int(sample.loc[sample["image_id"] == image_id, "label"].iloc[0])
            names.append(image_id)
            labels.append(pred)
            continue

        topred = process(img)
        hr_out = hrnet(topred)
        eff_out = efficient(topred)

        total = (hr_out + eff_out) / 2.0
        pred = int(torch.argmax(F.softmax(total, dim=1), dim=1).item())

        names.append(image_id)
        labels.append(pred)




## === cell 3
df = pd.DataFrame({"image_id": names, "label": labels})

out_path = "submission.csv"
df.to_csv(out_path, index=False)

print(df.head())
print(f"\nSaved submission to: {out_path} (rows={len(df)})")

assert out_path.endswith(".csv") and os.path.exists(out_path)
assert list(df.columns) == ["image_id", "label"]
assert len(df) == len(pd.read_csv(SAMPLE_SUB_PATH))
assert df["image_id"].tolist() == pd.read_csv(SAMPLE_SUB_PATH)["image_id"].tolist()
