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

3.11

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
pillow==11.3.0
protobuf==6.33.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
timm==1.0.19
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
transformers==4.53.3

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

0.8791175581746751

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime errors by removing an incompatible TensorFlow import (it triggers the protobuf `MessageFactory.GetPrototype` crash in this environment) and by correcting all `pd.DataFrame(columns={...})` usages to use lists instead of sets. Then I make the pipeline robust to missing external model files by conditionally skipping a model if its checkpoint directory/file isn’t present, so the notebook always runs end-to-end and still writes a valid `submission.csv`. Finally, I ensure `image_id` ordering matches `sample_submission.csv` (stable submission alignment) and keep the original ensemble logic unchanged so scoring behavior remains consistent when weights are available.'
- What this solution (achieved 0.05531) has done: 'I fix the protobuf/transformers import crash by removing the unused `ViTForImageClassification` top-level import and instead importing it lazily only inside the ViT block, which avoids triggering the `MessageFactory.GetPrototype` issue at startup. Then, to move accuracy up toward the target (your current score is far below), I stop silently skipping the ensemble by switching the model artifact paths to the real Kaggle dataset location under `/kaggle/input/...` (the current `../input/...` paths point to non-existent datasets here, causing nearly-all-zero labels). Finally, I keep the existing ensemble logic unchanged but make the `image_id` alignment explicitly follow `sample_submission.csv` order so predictions line up correctly in the submission.'
- What this solution (achieved 0.05531) has done: 'Your score is extremely low because in this environment none of the referenced model artifacts actually exist under `/kaggle/input/cassava-leaf-disease-classification/`, so the code skips all models and submits all-zero labels (near-random for 5 classes). To move accuracy up toward the target while preserving the same ensemble logic, I (1) auto-discover the correct competition dataset root under `/kaggle/input/` and (2) point the model paths to that discovered root, falling back to the current behavior if no artifacts are found. I also make `torch.load` more robust to common checkpoint formats (`state_dict` vs `model`) without changing architectures, so available weights actually get used. Finally, submission ordering remains locked to `sample_submission.csv` to avoid accidental misalignment.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is far below the target (0.8791), and the most likely reason is that none (or only a tiny subset) of the model artifacts are actually being found/loaded, causing near-constant predictions. I make a minimal, core-logic-preserving fix by (1) broadening model artifact discovery to search common locations under the detected dataset root (including `working/` and nested folders) and (2) making checkpoint loading tolerant to common key prefixes (e.g., `module.`) and typical dict wrappers, without changing any architecture or inference flow. I also ensure `available_models` correctly reflects “actually loadable” artifacts (e.g., ViT directory must contain config/weights), so we don’t silently “use” a broken model. These changes should move accuracy sharply upward toward the target while keeping ensemble semantics unchanged.'
- What this solution (achieved 0.05531) has done: 'Your current score is near-random, and in this environment that most likely comes from either (1) no real model artifacts being found (so everything defaults to label=0) or (2) artifacts exist but aren’t actually being loaded due to strict key mismatches, so predictions stay effectively constant. I keep your ensemble logic and architectures identical, but make checkpoint discovery more permissive (search common filename patterns under the dataset/working roots) and make state_dict loading tolerant to common wrappers and key-prefix differences while still staying “strict” on real shape mismatches. I also ensure the MobileNet head matches 5 classes (so a 1000-class random-initialized head can’t slip through) without changing the network’s core blocks. These minimal robustness fixes are the most direct way to move accuracy sharply upward toward your 0.879 target without changing evaluation semantics.'
- What this solution (achieved 0.05531) has done: 'Your current score is near-random because the ensemble is effectively producing constant/invalid predictions: in this environment you likely have no real model artifacts, and even when artifacts exist the DataFrame currently stores probability vectors as `object` arrays that can turn into `NaN` after merges, causing the fallback label=0. I keep your exact architectures and ensemble logic, but make two minimal, score-relevant fixes: (1) broaden artifact discovery to also search under `/kaggle/input/**` (not just the detected dataset root) so weights are actually found, and (2) store per-image probabilities as stable Python lists and use a safer, vectorized ensemble aggregation to prevent `NaN`/dtype issues after merges. This should move accuracy sharply upward toward the target without changing evaluation semantics or training/inference behavior. The script still always writes a valid `submission.csv` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import sys
import os
import random
import json
import gc
import math

import cv2
import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image

from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2

import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Dataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)




## === cell 1
def _find_dataset_root():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
    base = "/kaggle/input"
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            d = os.path.join(base, name)
            if not os.path.isdir(d):
                continue
            if os.path.exists(
                os.path.join(d, "sample_submission.csv")
            ) and os.path.isdir(os.path.join(d, "test_images")):
                candidates.append(d)
            nested = os.path.join(d, "cassava-leaf-disease-classification")
            if os.path.exists(
                os.path.join(nested, "sample_submission.csv")
            ) and os.path.isdir(os.path.join(nested, "test_images")):
                candidates.append(nested)

    for c in candidates:
        if os.path.exists(os.path.join(c, "sample_submission.csv")) and os.path.isdir(
            os.path.join(c, "test_images")
        ):
            return c
    return "/kaggle/input/cassava-leaf-disease-classification"


path = _find_dataset_root()
image_path = os.path.join(path, "test_images") + "/"

sample_path = os.path.join(path, "sample_submission.csv")
sample_submission = pd.read_csv(sample_path)

submission_df = sample_submission.copy()
submission_df["label"] = 0

existing = set(os.listdir(image_path)) if os.path.isdir(image_path) else set()
missing = [x for x in submission_df["image_id"].tolist() if x not in existing]
if len(missing) > 0:
    print(
        f"Warning: {len(missing)} sample_submission images not found in {image_path}. They will remain with label=0."
    )

print("Using dataset root:", path)
print("Test images dir exists:", os.path.isdir(image_path), "num files:", len(existing))



## === cell 2
MODEL_BASE = path


def _glob_first(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _glob_all(paths):
    return [p for p in paths if p and os.path.exists(p)]


def _candidate_roots(base_root):
    roots = []
    if base_root:
        roots.append(base_root)
        roots.append(os.path.join(base_root, "cassava-leaf-disease-classification"))
    roots.append("/kaggle/working")
    roots.append("/kaggle/working/cassava-leaf-disease-classification")
    roots.append(
        "/kaggle/working/cassava-leaf-disease-classification/cassava-leaf-disease-classification"
    )
    roots.append("/kaggle/data/cassava-leaf-disease-classification")
    roots.append(
        "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification"
    )

    base = "/kaggle/input"
    if os.path.isdir(base):
        for name in sorted(os.listdir(base)):
            d = os.path.join(base, name)
            if os.path.isdir(d):
                roots.append(d)
                roots.append(os.path.join(d, "cassava-leaf-disease-classification"))

    seen = set()
    out = []
    for r in roots:
        if r and r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _find_resnext_ckpts():
    rels = [
        "models/resnext50_32x4d_fold1_best.pth",
        "models/resnext50_32x4d*.pth",
        "models/*resnext*.pth",
        "resnext50_32x4d_fold1_best.pth",
        "*resnext*.pth",
        "*resnext*.pt",
        "*resnext*.bin",
    ]
    found = []
    import glob

    for root in _candidate_roots(MODEL_BASE):
        for rel in rels:
            for p in glob.glob(os.path.join(root, rel)):
                if os.path.isfile(p):
                    found.append(p)
    found = sorted(list(dict.fromkeys(found)))
    return found


def _find_vit_dir():
    rels = [
        "model-vit/original_save_pretrained",
        "models/model-vit/original_save_pretrained",
        "vit/original_save_pretrained",
        "original_save_pretrained",
        "model-vit",
        "models/model-vit",
        "vit",
    ]
    cands = []
    for root in _candidate_roots(MODEL_BASE):
        for rel in rels:
            cands.append(os.path.join(root, rel))
    for d in cands:
        if os.path.isdir(d):
            has_config = os.path.exists(os.path.join(d, "config.json"))
            has_weights = any(
                os.path.exists(os.path.join(d, w))
                for w in ["pytorch_model.bin", "model.safetensors"]
            )
            if has_config and has_weights:
                return d
            nested = os.path.join(d, "original_save_pretrained")
            if (
                os.path.isdir(nested)
                and os.path.exists(os.path.join(nested, "config.json"))
                and any(
                    os.path.exists(os.path.join(nested, w))
                    for w in ["pytorch_model.bin", "model.safetensors"]
                )
            ):
                return nested
    return None


def _find_mobilenet_ckpt():
    rels = [
        "model-mobilenet/mn3_bt20_ep5_lr1.pth",
        "models/model-mobilenet/mn3_bt20_ep5_lr1.pth",
        "mobilenet/mn3_bt20_ep5_lr1.pth",
        "mn3_bt20_ep5_lr1.pth",
        "models/*mobilenet*.pth",
        "*mobilenet*.pth",
        "*mn3*.pth",
        "*mobile*.pth",
        "*mobilenet*.pt",
    ]
    import glob

    cands = []
    for root in _candidate_roots(MODEL_BASE):
        for rel in rels:
            cands.extend(glob.glob(os.path.join(root, rel)))
    cands = [p for p in sorted(list(dict.fromkeys(cands))) if os.path.isfile(p)]
    return _glob_first(cands)


used_models_pytorch = {
    "resnext": _find_resnext_ckpts(),  # list of ckpt paths
    "vit": _find_vit_dir(),  # directory or None
    "mobilenet": _find_mobilenet_ckpt(),  # file or None
}


def _exists_model_artifact(name, obj):
    if name == "resnext":
        return (
            isinstance(obj, (list, tuple))
            and len(obj) > 0
            and all(os.path.exists(p) for p in obj)
        )
    if name == "vit":
        return (
            isinstance(obj, str)
            and os.path.isdir(obj)
            and os.path.exists(os.path.join(obj, "config.json"))
            and (
                os.path.exists(os.path.join(obj, "pytorch_model.bin"))
                or os.path.exists(os.path.join(obj, "model.safetensors"))
            )
        )
    if name == "mobilenet":
        return isinstance(obj, str) and os.path.exists(obj)
    return False


available_models = {
    k: v for k, v in used_models_pytorch.items() if _exists_model_artifact(k, v)
}
missing_models = [k for k in used_models_pytorch.keys() if k not in available_models]
if missing_models:
    print("Warning: Missing model artifacts for:", missing_models)
    print("They will be skipped so a valid submission is still produced.")
print("Available models:", list(available_models.keys()))
print(
    "Resolved artifact paths/dirs:", {k: available_models[k] for k in available_models}
)




## === cell 3
class CustomResNext(nn.Module):
    def __init__(self, model_name="resnext50_32x4d", pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.fc.in_features
        self.model.fc = nn.Linear(n_features, 5)

    def forward(self, x):
        return self.model(x)


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df["image_path_id"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image = cv2.imread(file_name)
        if image is None:
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image




## === cell 4
if "resnext" in available_models:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_transforms():
        return Compose(
            [
                Resize(512, 512),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def _extract_state_dict(obj):
        if isinstance(obj, dict):
            for k in [
                "state_dict",
                "model",
                "model_state_dict",
                "net",
                "weights",
                "ema_state_dict",
                "ema",
            ]:
                if k in obj and isinstance(obj[k], dict):
                    obj = obj[k]
                    break
        return obj

    def _strip_module_prefix(state_dict):
        if not isinstance(state_dict, dict):
            return state_dict
        keys = list(state_dict.keys())
        if not keys or not all(isinstance(k, str) for k in keys):
            return state_dict
        if all(k.startswith("module.") for k in keys):
            return {k[len("module.") :]: v for k, v in state_dict.items()}
        return state_dict

    def _safe_load_state_dict_strictish(model, sd):
        model_sd = model.state_dict()
        if not isinstance(sd, dict):
            raise ValueError("Checkpoint state is not a dict.")
        if set(sd.keys()) == set(model_sd.keys()):
            model.load_state_dict(sd, strict=True)
            return

        filtered = {}
        for k, v in sd.items():
            if k in model_sd and hasattr(v, "shape") and hasattr(model_sd[k], "shape"):
                if tuple(v.shape) == tuple(model_sd[k].shape):
                    filtered[k] = v

        if len(filtered) < max(50, int(0.5 * len(model_sd))):
            model.load_state_dict(sd, strict=True)
            return

        missing = [k for k in model_sd.keys() if k not in filtered]
        unexpected = [k for k in sd.keys() if k not in model_sd]
        if missing:
            print(
                f"ResNeXt: filtered load with {len(filtered)}/{len(model_sd)} tensors; missing={len(missing)}, unexpected={len(unexpected)}"
            )
        model.load_state_dict(filtered, strict=False)

    def inference(model, states, test_loader, device):
        model.to(device)
        probabilities = []
        for images in tqdm(test_loader, desc="ResNeXt inference"):
            images = images.to(device)
            avg_preds = []
            for state in states:
                sd = _strip_module_prefix(_extract_state_dict(state))
                _safe_load_state_dict_strictish(model, sd)
                model.eval()
                with torch.no_grad():
                    y_preds = model(images)
                avg_preds.append(y_preds.softmax(1).to("cpu").numpy())
            avg_preds = np.mean(avg_preds, axis=0)
            probabilities.append(avg_preds)
        return np.concatenate(probabilities, axis=0)

    predictions_resnext = pd.DataFrame(columns=["image_id"])
    predictions_resnext["image_id"] = submission_df["image_id"].values
    predictions_resnext["image_path_id"] = image_path + predictions_resnext[
        "image_id"
    ].astype(str)

    model = CustomResNext("resnext50_32x4d", pretrained=False)
    states = [torch.load(f, map_location="cpu") for f in available_models["resnext"]]

    test_dataset = TestDataset(predictions_resnext, transform=get_transforms())
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    predictions = inference(model, states, test_loader, device)

    predictions_resnext["resnext"] = [
        p.astype(np.float32).tolist() for p in predictions
    ]
    predictions_resnext = predictions_resnext.drop(["image_path_id"], axis=1)

    torch.cuda.empty_cache()
    del model, states, test_dataset, test_loader, predictions
    gc.collect()



## === cell 5
if "vit" in available_models:
    from transformers import ViTForImageClassification

    IMG_SIZE = 224
    BATCH_SIZE = 16
    num_classes = 5

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    class LeafDatasetViT(torch.utils.data.Dataset):
        def __init__(self, df, data_path, mode="train", transforms=None):
            super().__init__()
            self.df_data = df.values
            self.data_path = data_path
            self.transforms = transforms
            self.data_dir = "train_images" if mode == "train" else "test_images"

        def __len__(self):
            return len(self.df_data)

        def __getitem__(self, index):
            img_name = self.df_data[index][0]
            img_path = os.path.join(self.data_path, self.data_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            if self.transforms is not None:
                img = self.transforms(img)
            return img

    transforms_val_vit = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    def predict_vit(model, test_dataset, device):
        preds = []
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )
        model.eval()
        for test_images in tqdm(test_dataloader, desc="ViT inference"):
            test_images = test_images.to(device)
            with torch.no_grad():
                output = model(test_images)
            preds.extend(output.logits.data.softmax(1).cpu().numpy())
        return preds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions_vit = pd.DataFrame(columns=["image_id"])
    predictions_vit["image_id"] = submission_df["image_id"].values

    model = ViTForImageClassification.from_pretrained(
        available_models["vit"], num_labels=num_classes
    )
    model.to(device)

    test_dataset = LeafDatasetViT(
        df=predictions_vit, data_path=path, mode="test", transforms=transforms_val_vit
    )

    predictions_raw_vit = predict_vit(model, test_dataset, device)
    predictions_vit["vit"] = [
        np.asarray(p, dtype=np.float32).tolist() for p in predictions_raw_vit
    ]

    torch.cuda.empty_cache()
    del model, test_dataset, predictions_raw_vit
    gc.collect()



## === cell 6
if "mobilenet" in available_models:
    __all__ = ["mobilenetv3_large", "mobilenetv3_small"]

    def _make_divisible(v, divisor, min_value=None):
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    class h_sigmoid(nn.Module):
        def __init__(self, inplace=True):
            super(h_sigmoid, self).__init__()
            self.relu = nn.ReLU6(inplace=inplace)

        def forward(self, x):
            return self.relu(x + 3) / 6

    class h_swish(nn.Module):
        def __init__(self, inplace=True):
            super(h_swish, self).__init__()
            self.sigmoid = h_sigmoid(inplace=inplace)

        def forward(self, x):
            return x * self.sigmoid(x)

    class SELayer(nn.Module):
        def __init__(self, channel, reduction=4):
            super(SELayer, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Linear(channel, _make_divisible(channel // reduction, 8)),
                nn.ReLU(inplace=True),
                nn.Linear(_make_divisible(channel // reduction, 8), channel),
                h_sigmoid(),
            )

        def forward(self, x):
            b, c, _, _ = x.size()
            y = self.avg_pool(x).view(b, c)
            y = self.fc(y).view(b, c, 1, 1)
            return x * y

    def conv_3x3_bn(inp, oup, stride):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            h_swish(),
        )

    def conv_1x1_bn(inp, oup):
        return nn.Sequential(
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup), h_swish()
        )

    class InvertedResidual(nn.Module):
        def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
            super(InvertedResidual, self).__init__()
            assert stride in [1, 2]
            self.identity = stride == 1 and inp == oup

            if inp == hidden_dim:
                self.conv = nn.Sequential(
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size,
                        stride,
                        (kernel_size - 1) // 2,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    SELayer(hidden_dim) if use_se else nn.Identity(),
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    nn.Conv2d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size,
                        stride,
                        (kernel_size - 1) // 2,
                        groups=hidden_dim,
                        bias=False,
                    ),
                    nn.BatchNorm2d(hidden_dim),
                    SELayer(hidden_dim) if use_se else nn.Identity(),
                    h_swish() if use_hs else nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )

        def forward(self, x):
            if self.identity:
                return x + self.conv(x)
            else:
                return self.conv(x)

    class MobileNetV3(nn.Module):
        def __init__(self, cfgs, mode, num_classes=1000, width_mult=1.0):
            super(MobileNetV3, self).__init__()
            self.cfgs = cfgs
            assert mode in ["large", "small"]

            input_channel = _make_divisible(16 * width_mult, 8)
            layers = [conv_3x3_bn(3, input_channel, 2)]
            block = InvertedResidual
            for k, t, c, use_se, use_hs, s in self.cfgs:
                output_channel = _make_divisible(c * width_mult, 8)
                exp_size = _make_divisible(input_channel * t, 8)
                layers.append(
                    block(input_channel, exp_size, output_channel, k, s, use_se, use_hs)
                )
                input_channel = output_channel
            self.features = nn.Sequential(*layers)
            self.conv = conv_1x1_bn(input_channel, exp_size)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            output_channel = {"large": 1280, "small": 1024}
            output_channel = (
                _make_divisible(output_channel[mode] * width_mult, 8)
                if width_mult > 1.0
                else output_channel[mode]
            )
            self.classifier = nn.Sequential(
                nn.Linear(exp_size, output_channel),
                h_swish(),
                nn.Dropout(0.2),
                nn.Linear(output_channel, num_classes),
            )
            self._initialize_weights()

        def forward(self, x):
            x = self.features(x)
            x = self.conv(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

        def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2.0 / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    def mobilenetv3_large(**kwargs):
        cfgs = [
            [3, 1, 16, 0, 0, 1],
            [3, 4, 24, 0, 0, 2],
            [3, 3, 24, 0, 0, 1],
            [5, 3, 40, 1, 0, 2],
            [5, 3, 40, 1, 0, 1],
            [5, 3, 40, 1, 0, 1],
            [3, 6, 80, 0, 1, 2],
            [3, 2.5, 80, 0, 1, 1],
            [3, 2.3, 80, 0, 1, 1],
            [3, 2.3, 80, 0, 1, 1],
            [3, 6, 112, 1, 1, 1],
            [3, 6, 112, 1, 1, 1],
            [5, 6, 160, 1, 1, 2],
            [5, 6, 160, 1, 1, 1],
            [5, 6, 160, 1, 1, 1],
        ]
        return MobileNetV3(cfgs, mode="large", **kwargs)

    class LeafDatasetMobile(torch.utils.data.Dataset):
        def __init__(self, df, data_path, mode="train", transforms=None):
            super().__init__()
            self.df_data = df.values
            self.data_path = data_path
            self.transforms = transforms
            self.data_dir = "train_images" if mode == "train" else "test_images"

        def __len__(self):
            return len(self.df_data)

        def __getitem__(self, index):
            img_name = self.df_data[index][0]
            img_path = os.path.join(self.data_path, self.data_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            if self.transforms is not None:
                img = self.transforms(img)
            return img

    def predict_mobilenet(model, test_dataset, device):
        preds = []
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset, batch_size=20, shuffle=False, num_workers=2
        )
        model.eval()
        for test_images in tqdm(test_dataloader, desc="MobileNet inference"):
            test_images = test_images.to(device)
            with torch.no_grad():
                output = model(test_images)
            preds.extend(output.softmax(1).cpu().numpy())
        return preds

    transforms_val_mobile = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predictions_mobilenet = pd.DataFrame(columns=["image_id"])
    predictions_mobilenet["image_id"] = submission_df["image_id"].values

    model = mobilenetv3_large(num_classes=5)
    model.to(device)

    ckpt = torch.load(available_models["mobilenet"], map_location="cpu")
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                sd = ckpt[k]
                break
        else:
            sd = ckpt
    else:
        sd = ckpt

    if isinstance(sd, dict) and all(
        isinstance(k, str) and k.startswith("module.") for k in sd.keys()
    ):
        sd = {k[len("module.") :]: v for k, v in sd.items()}

    def _safe_load_state_dict_strictish(model, sd):
        model_sd = model.state_dict()
        if set(sd.keys()) == set(model_sd.keys()):
            model.load_state_dict(sd, strict=True)
            return
        filtered = {}
        for k, v in sd.items():
            if k in model_sd and hasattr(v, "shape") and hasattr(model_sd[k], "shape"):
                if tuple(v.shape) == tuple(model_sd[k].shape):
                    filtered[k] = v
        if len(filtered) < max(50, int(0.5 * len(model_sd))):
            model.load_state_dict(sd, strict=True)
            return
        print(f"MobileNet: filtered load with {len(filtered)}/{len(model_sd)} tensors")
        model.load_state_dict(filtered, strict=False)

    _safe_load_state_dict_strictish(model, sd)

    test_dataset = LeafDatasetMobile(
        df=predictions_mobilenet,
        data_path=path,
        mode="test",
        transforms=transforms_val_mobile,
    )
    predictions_raw_mobilenet = predict_mobilenet(model, test_dataset, device)

    predictions_mobilenet["mobilenet"] = [
        np.asarray(p, dtype=np.float32).tolist() for p in predictions_raw_mobilenet
    ]

    torch.cuda.empty_cache()
    del model, test_dataset, predictions_raw_mobilenet, ckpt, sd
    gc.collect()



## === cell 7
submission_df = sample_submission.copy()
submission_df["label"] = 0

if "resnext" in available_models:
    submission_df = submission_df.merge(predictions_resnext, on="image_id", how="left")

if "mobilenet" in available_models:
    submission_df = submission_df.merge(
        predictions_mobilenet, on="image_id", how="left"
    )

if "vit" in available_models:
    submission_df = submission_df.merge(predictions_vit, on="image_id", how="left")

model_keys = list(available_models.keys())



## === cell 8
if len(model_keys) > 0:
    prob_sum = np.zeros((len(submission_df), 5), dtype=np.float32)
    for mk in model_keys:
        col = submission_df[mk].values
        for i, p in enumerate(col):
            if isinstance(p, (list, tuple, np.ndarray)):
                a = np.asarray(p, dtype=np.float32).reshape(-1)
                if a.shape[0] == 5 and np.all(np.isfinite(a)):
                    prob_sum[i] += a
    submission_df["label"] = prob_sum.argmax(axis=1).astype(int)
else:
    submission_df["label"] = submission_df["label"].astype(int)

submission_df = sample_submission[["image_id"]].merge(
    submission_df[["image_id", "label"]], on="image_id", how="left"
)
submission_df["label"] = submission_df["label"].fillna(0).astype(int)

out_path = "submission.csv"
submission_df[["image_id", "label"]].to_csv(out_path, index=False)

print(submission_df.head())
print(f"Wrote {out_path} with shape {submission_df[['image_id','label']].shape}")
print("Models used:", model_keys)
print(
    "Label value counts:\n",
    submission_df["label"].value_counts(dropna=False).sort_index(),
)
