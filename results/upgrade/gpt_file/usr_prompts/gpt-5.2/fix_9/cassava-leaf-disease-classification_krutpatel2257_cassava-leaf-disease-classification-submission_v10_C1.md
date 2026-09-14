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

0.8839528558476881

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'Your code isn’t yielding a score mainly because it likely errors at runtime (missing `saved-model/model.pth`, wrong cell numbering, and `torch.load` device mismatch) and it also uses random augmentations at inference which makes predictions unstable and usually worse. I make the smallest changes needed to (1) reliably load the model if present (with `map_location`), (2) ensure inference runs end-to-end and writes a valid `submission.csv`, and (3) remove randomness from test-time transforms (use a deterministic resize/center-crop + normalization) to push accuracy upward toward your target. I also add a safe fallback that creates a valid submission (using the sample submission labels) if the model file is missing, so you always get a scorable CSV. Core model architecture and basic inference semantics (argmax over logits) remain unchanged.'
- What this solution (achieved 0.11584) has done: 'I fix the Albumentations runtime error by updating the `RandomResizedCrop` call to the v2 API so the notebook runs end-to-end. Since your current score is far below target, I also make inference deterministic by replacing the random heavy augmentation pipeline (which was both broken and harmful at test time) with a simple resize/center-crop + normalization pipeline consistent with ImageNet-pretrained backbones, while keeping the same model architecture and argmax prediction logic. I keep the robust model-loading logic and ensure a valid `submission.csv` is always written with the required columns. These changes are minimal, directly address the crash, and should substantially improve score versus the current unstable/incorrect setup.'
- What this solution (achieved 0.11584) has done: 'Your current score (0.11584) is far below the target (0.88395), and the biggest likely cause is that the model is not actually being loaded in this environment (`../input/saved-model/model.pth` doesn’t exist here), so you’re effectively submitting the sample labels (near-random for this test set). The minimal score-improving change is to load weights from a file path that actually exists in this dataset layout (if available), while keeping the exact same architecture and argmax inference. I add a small “search common locations” block to find `model.pth` inside `../input/` and load it with `map_location`, and I keep inference deterministic as it already is (no random test-time aug). The script still always write a valid `submission.csv`.'
- What this solution (achieved 0.11584) has done: 'Your current score is extremely low because the model weights are almost certainly not being loaded, so you’re effectively submitting a near-random/fallback file. To move the score sharply upward toward the target while keeping the same architecture and argmax inference semantics, I (1) broaden the model weight search to include common weight filenames/locations in `../input`, and (2) make the loader robust to common checkpoint formats by safely handling key prefixes and mismatched `fc` shapes (while still keeping `strict=True` for all matching layers). I also ensure the model is put in `eval()` regardless and switch to a DataLoader-based inference loop (no change in predictions, just reliability/speed) so it finishes under the time limit and always writes a valid `submission.csv`. If no weights are found, it still write a valid fallback submission, but the primary aim is to actually load the provided weights if they exist anywhere under the Kaggle input tree.'
- What this solution (achieved 0.11584) has done: 'Your score is still near-random because the script is almost certainly falling back to `sample_submission.csv` labels when it can’t find/load real weights, so the most direct way to move toward the target is to reliably locate and correctly load the checkpoint. I keep your exact model (ResNeXt50 + `fc=5`) and argmax inference, but make the checkpoint search cover more realistic Kaggle weight filenames and handle both full-model (`torch.save(model)`) and state-dict checkpoints safely. I also make the state-dict key cleanup a bit more robust (`module.` and common Lightning `_orig_mod.` prefixes) and keep inference deterministic as you already do. If no weights exist anywhere under `../input`, it still produce a valid `submission.csv`, but the primary improvement is ensuring weights actually get loaded when present.'
- What this solution (achieved 0.11584) has done: 'Your current score is near-random, which strongly suggests the model weights still aren’t being loaded correctly (or you’re loading an unrelated checkpoint), so the most direct way to move toward the 0.8839 target is to (1) search for the checkpoint in more realistic locations (including `../input/**` and `../kaggle/input/**`) and (2) correctly handle common checkpoint formats (plain `state_dict`, Lightning `state_dict` with `model.` prefix, etc.) while keeping the exact same ResNeXt50+fc(5) architecture and argmax inference. I also ensure the submission strictly follows the `sample_submission.csv` order (already mostly done) and add deterministic settings so results don’t vary run-to-run. No training, no architecture changes, and no metric/prediction-semantic changes—just more robust weight discovery/loading so you stop falling back to the sample labels.'
- What this solution (achieved 0.11584) has done: 'Your score is near-random because the notebook is still almost certainly not loading the intended trained weights, so it falls back to the sample submission labels. The smallest change that should move accuracy sharply upward toward your 0.8839 target is to (1) look for checkpoint files in more realistic locations/filenames (including `.ckpt` and common “fold”/“best” patterns) and (2) correctly extract/clean the `state_dict` from more checkpoint formats (Lightning-style `state_dict` with `model.`/`net.`/`module.` prefixes). I keep your exact model (ResNeXt50 + `fc=5`), your argmax inference semantics, and the deterministic test-time transform, only improving robustness of weight discovery/loading. The script still always writes a valid `submission.csv` even if no weights are found.'
- What this solution (achieved 0.11584) has done: 'Your score is near-random (0.11584), which strongly suggests you’re still not actually using trained weights (either not found, not loaded, or loaded in an incompatible format). To move sharply upward toward the 0.88395 target without changing the model/inference semantics, I make the checkpoint discovery more robust (include any `.pth/.pt/.ckpt` under `/kaggle/input` and prefer “best/fold” files) and make loading handle common Lightning/EMA wrappers and key prefixes while still preserving your ResNeXt50+argmax pipeline. I also ensure the model is always moved to the correct device *after* loading (a common silent pitfall) and keep deterministic test transforms unchanged. The script still always write a valid `submission.csv`, but the main improvement is dramatically increasing the chance the intended weights actually get loaded and used.'

# 9. Code solution

## === cell 0
import os

import albumentations  # kept to preserve original imports/environment, though unused in final inference
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms



## === cell 1
model_path = "../input/saved-model/model.pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 2
def find_model_path(
    preferred_path: str,
    filenames=(
        "model.pth",
        "best.pth",
        "checkpoint.pth",
        "weights.pth",
        "model.pt",
        "best.pt",
        "checkpoint.pt",
        "pytorch_model.bin",
        "last.ckpt",
        "best.ckpt",
        "model.ckpt",
        "checkpoint.ckpt",
        "resnext50.pth",
        "resnext50_32x4d.pth",
        "cassava.pth",
        "cassava_model.pth",
        "fold0.pth",
        "fold1.pth",
        "fold2.pth",
        "fold3.pth",
        "fold4.pth",
        "fold0.ckpt",
        "fold1.ckpt",
        "fold2.ckpt",
        "fold3.ckpt",
        "fold4.ckpt",
    ),
    search_roots: tuple = ("../input", "../kaggle/input", "/kaggle/input"),
):
    if os.path.exists(preferred_path):
        return preferred_path

    candidates = []

    for search_root in search_roots:
        if not os.path.exists(search_root):
            continue
        for root, _, files in os.walk(search_root):
            file_set = set(files)
            for fn in filenames:
                if fn in file_set:
                    candidates.append(os.path.join(root, fn))

            for f in files:
                lf = f.lower()
                if lf.endswith((".pth", ".pt", ".ckpt")):
                    candidates.append(os.path.join(root, f))

    def _rank(p: str):
        lp = p.lower()
        score = 0
        if "best" in lp:
            score -= 20
        if "fold" in lp:
            score -= 10
        if "final" in lp:
            score -= 5
        if "last" in lp:
            score += 5
        score += len(p)
        return (score, p)

    candidates = sorted(list(set(candidates)), key=_rank)
    return candidates[0] if candidates else preferred_path


resolved_model_path = find_model_path(model_path)
print(f"INFO: Resolved model path: {resolved_model_path}")



## === cell 3
model = models.resnext50_32x4d(pretrained=False)
model.fc = nn.Linear(2048, 5)

model_loaded = False


def _extract_state_dict(ckpt):
    if isinstance(ckpt, nn.Module):
        return ckpt.state_dict()

    if isinstance(ckpt, dict):
        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "model_dict",
            "ema_state_dict",
            "ema",
            "model_ema",
            "model_ema_state_dict",
        ]:
            if key in ckpt and isinstance(ckpt[key], dict):
                inner = ckpt[key]
                if "state_dict" in inner and isinstance(inner["state_dict"], dict):
                    return inner["state_dict"]
                return inner
            if key in ckpt and isinstance(ckpt[key], nn.Module):
                return ckpt[key].state_dict()

        if all(isinstance(k, str) for k in ckpt.keys()):
            tensor_like = sum([hasattr(v, "shape") for v in ckpt.values()])
            if tensor_like >= max(1, int(0.5 * len(ckpt))):
                return ckpt

    return None


def _clean_state_dict_keys(state_dict):
    sd = dict(state_dict)

    prefixes = (
        "module.",
        "_orig_mod.",
        "model.",
        "net.",
        "network.",
        "encoder.",
        "backbone.",
        "student.",
        "teacher.",
    )

    changed = True
    while changed:
        changed = False
        for p in prefixes:
            if any(k.startswith(p) for k in sd.keys()):
                sd = {k[len(p) :]: v for k, v in sd.items()}
                changed = True

    for p in ("model.", "net.", "network."):
        if any(k.startswith(p) for k in sd.keys()):
            sd = {k[len(p) :]: v for k, v in sd.items()}

    return sd


def _maybe_drop_classifier(sd, model_obj):
    sd = dict(sd)
    if "fc.weight" in sd and hasattr(model_obj, "fc"):
        if sd["fc.weight"].shape != model_obj.fc.weight.shape:
            print(
                "WARNING: Dropping fc.* from checkpoint due to shape mismatch: "
                f"{tuple(sd['fc.weight'].shape)} vs {tuple(model_obj.fc.weight.shape)}"
            )
            sd.pop("fc.weight", None)
            sd.pop("fc.bias", None)
    return sd


if os.path.exists(resolved_model_path):
    try:
        ckpt = torch.load(resolved_model_path, map_location="cpu")

        if isinstance(ckpt, nn.Module):
            try:
                model.load_state_dict(ckpt.state_dict(), strict=True)
                model_loaded = True
                print(
                    "INFO: Loaded weights from full nn.Module checkpoint (strict=True)."
                )
            except Exception as e:
                print(f"WARNING: Full-model strict load failed: {e}")

        if not model_loaded:
            state_dict = _extract_state_dict(ckpt)
            if state_dict is None:
                raise ValueError(
                    "Checkpoint format not recognized (no state_dict-like object found)."
                )

            state_dict = _clean_state_dict_keys(state_dict)
            state_dict = _maybe_drop_classifier(state_dict, model)

            try:
                model.load_state_dict(state_dict, strict=True)
                model_loaded = True
                print("INFO: Model loaded (strict=True).")
            except Exception as e_strict:
                incompatible = model.load_state_dict(state_dict, strict=False)
                model_loaded = True
                print(
                    f"INFO: Model loaded (strict=False) after strict=True failed: {e_strict}"
                )
                try:
                    missing = list(incompatible.missing_keys)
                    unexpected = list(incompatible.unexpected_keys)
                    if missing:
                        print(f"INFO: Missing keys (first 20): {missing[:20]}")
                    if unexpected:
                        print(f"INFO: Unexpected keys (first 20): {unexpected[:20]}")
                except Exception:
                    pass

        model.to(device)
        model.eval()

    except Exception as e:
        print(f"WARNING: Failed to load model from {resolved_model_path}: {e}")
        model.to(device)
        model.eval()
else:
    print(f"WARNING: Model file not found at {resolved_model_path}")
    model.to(device)
    model.eval()



## === cell 4
sample_sub = pd.read_csv(sample_sub_path)

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 5
from torch.utils.data import Dataset, DataLoader


class TestDataset(Dataset):
    def __init__(self, df, images_root, transform):
        self.image_ids = df["image_id"].tolist()
        self.images_root = images_root
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.images_root, image_id)
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img)
        return image_id, x


predictions = []

if not model_loaded:
    sub_df = sample_sub.copy()
    sub_df["label"] = sub_df["label"].astype(int)
    sub_df[["image_id", "label"]].to_csv("submission.csv", index=False)
    print("Saved fallback submission.csv (model not loaded).")
    print(sub_df.head())
else:
    ds = TestDataset(sample_sub, test_images_path, transform)
    dl = DataLoader(
        ds,
        batch_size=64 if device.type == "cuda" else 16,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    all_image_ids = []
    all_preds = []

    model.eval()
    with torch.inference_mode():
        for image_ids, x in dl:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            pred = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(int)
            all_image_ids.extend(list(image_ids))
            all_preds.extend(pred.tolist())

    sub_df = pd.DataFrame({"image_id": all_image_ids, "label": all_preds})

    sub_df = sample_sub[["image_id"]].merge(sub_df, on="image_id", how="left")
    sub_df["label"] = sub_df["label"].fillna(0).astype(int)

    sub_df.to_csv("submission.csv", index=False)
    print("Saved submission.csv")
    print(sub_df.head())
    print("INFO: model_loaded=True, predictions generated from model.")
