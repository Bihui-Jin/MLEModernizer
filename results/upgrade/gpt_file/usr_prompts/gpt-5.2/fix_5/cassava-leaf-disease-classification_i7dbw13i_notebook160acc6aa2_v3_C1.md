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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
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

0.8783620429132668

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime errors by replacing invalid `set` usage for pandas `columns` with lists, and I ensure `submission_df` is always created from `sample_submission.csv` so it’s correctly ordered and stable. I also make model checkpoint loading robust to common formats (`state_dict`, `model`, or raw state dict) and to CPU/GPU differences via `map_location`, which prevents failures in Kaggle. Finally, I keep the existing ResNeXt inference/softmax averaging logic intact and guarantee a valid `submission.csv` is written even if no model files are present (it fall back to label 0 rather than crashing).'
- What this solution (achieved 0.05531) has done: 'Your current score (0.05531) is far below the target (0.87836), and the biggest likely cause is that your ResNeXt checkpoint path points to a non-existent dataset (`../input/models/...`), so inference is skipped and you submit all-zeros. I make a minimal, score-relevant fix by (1) auto-discovering `.pth` checkpoints anywhere under `/kaggle/input/` when the configured path is missing, and (2) ensuring the discovered checkpoints are used for the existing ResNeXt averaging logic without changing the model or transforms. I also make the ResNeXt head replacement more robust across timm variants (some models use `classifier` instead of `fc`) to avoid silent mis-loads. These changes should move accuracy substantially toward your target by actually using the trained weights rather than the fallback labels.'
- What this solution (achieved 0.05531) has done: 'Your score is extremely low because the code is still effectively falling back to constant predictions in most runs: even when a checkpoint is found, `strict=True` plus key-mangling can silently prevent correct weight loading (or crash), and the current state-dict extraction may strip needed prefixes incorrectly for timm models. I make a minimal, score-relevant change to load checkpoints robustly (handle `model.*` vs `module.*` without over-stripping, allow `strict=False` while reporting missing/unexpected keys) so inference actually uses the trained weights. I also make checkpoint discovery slightly broader (still restricted to ResNeXt) and ensure we only average states that successfully load, which should move accuracy sharply upward toward your target without changing the model, transforms, or ensembling logic. The submission writing and format remain unchanged.'
- What this solution (achieved 0.05531) has done: 'Your current score is near-random, which strongly suggests the pipeline is still not actually using a compatible trained checkpoint (so it effectively submits constant/garbage labels). I make the smallest score-relevant changes to (1) filter checkpoint discovery to Cassava-related directories to avoid loading unrelated ResNeXt weights, (2) load each checkpoint once into the model (instead of re-loading per batch) so weights are applied consistently and faster, and (3) validate that loaded checkpoints truly match the 5-class head by requiring the classifier weights to load (otherwise skip). This keeps your ResNeXt architecture, transforms, and “average softmax across checkpoints” inference semantics intact, but makes it far more likely that you’re averaging the intended Cassava fold weights, moving accuracy toward your target. The script still always write a valid `submission.csv` in the required format.'

# 9. Code solution

## === cell 0
import sys
import os
import random
import json
import gc
import cv2
import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score
from functools import partial
from albumentations import (
    Compose,
    OneOf,
    Normalize,
    Resize,
    RandomResizedCrop,
    RandomCrop,
    CenterCrop,
    HorizontalFlip,
    VerticalFlip,
    Rotate,
    ShiftScaleRotate,
    Transpose,
)
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)



## === cell 1
path = "/kaggle/input/cassava-leaf-disease-classification/"
image_path = os.path.join(path, "test_images") + "/"

IMAGE_SIZE = (512, 512)

sample_path = os.path.join(path, "sample_submission.csv")
submission_df = pd.read_csv(sample_path)
submission_df["label"] = 0  # placeholder; will be overwritten if we can infer

available = set(os.listdir(image_path))
missing = [img for img in submission_df["image_id"].tolist() if img not in available]
if len(missing) > 0:
    print(
        f"Warning: {len(missing)} images from sample_submission not found in {image_path}. Example: {missing[:3]}"
    )



## === cell 2
onlykeras = False

used_models_pytorch = {
    "resnext": [f"../input/models/resnext50_32x4d_fold{fold}_best.pth" for fold in [1]]
}
used_models_keras = {}

stacked_mean = False




## === cell 3
class CustomResNext(nn.Module):
    def __init__(self, model_name="resnext50_32x4d", pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)

        if hasattr(self.model, "fc") and isinstance(self.model.fc, nn.Module):
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, 5)
        elif hasattr(self.model, "classifier") and isinstance(
            self.model.classifier, nn.Module
        ):
            n_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(n_features, 5)
        else:
            if hasattr(self.model, "reset_classifier"):
                self.model.reset_classifier(5)
            else:
                raise AttributeError(
                    "Could not locate classifier head to reset to 5 classes."
                )

    def forward(self, x):
        x = self.model(x)
        return x


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
            raise FileNotFoundError(f"Could not read image: {file_name}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            sd = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            sd = ckpt["model"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_sd[nk] = v
    return new_sd


def _discover_resnext_checkpoints(search_root="/kaggle/input", limit=16):
    """
    Change (score-relevant): restrict discovery to Cassava-related directories
    to avoid accidentally loading unrelated 'resnext' checkpoints from other datasets,
    which can produce near-random accuracy.
    """
    hits = []
    allow_hint = ("cassava", "leaf", "disease", "cgm")
    for root, _, files in os.walk(search_root):
        rlow = root.lower()
        if not any(h in rlow for h in allow_hint):
            continue
        for fn in files:
            lfn = fn.lower()
            if not (lfn.endswith(".pth") or lfn.endswith(".pt")):
                continue
            if "resnext" in lfn:
                hits.append(os.path.join(root, fn))
    hits = sorted(
        hits,
        key=lambda p: (
            ("best" not in os.path.basename(p).lower()),
            ("fold" not in os.path.basename(p).lower()),
            p,
        ),
    )
    return hits[:limit]


def _try_load_state_dict_into_model(model, state_dict):
    """
    Load robustly so we actually use weights. Attempt:
      1) direct keys
      2) keys with 'model.' stripped (common)
    """
    res1 = model.load_state_dict(state_dict, strict=False)
    missing1 = getattr(res1, "missing_keys", [])
    unexpected1 = getattr(res1, "unexpected_keys", [])

    if len(missing1) > 0 and any(k.startswith("model.") for k in state_dict.keys()):
        stripped = {}
        for k, v in state_dict.items():
            nk = k[len("model.") :] if k.startswith("model.") else k
            stripped[nk] = v
        res2 = model.load_state_dict(stripped, strict=False)
        return res2, stripped

    return res1, state_dict


def _classifier_keys_for_model(model):
    keys = set()
    sd = model.state_dict()
    for k in sd.keys():
        if k.endswith("fc.weight") or k.endswith("fc.bias"):
            keys.add(k)
        if k.endswith("classifier.weight") or k.endswith("classifier.bias"):
            keys.add(k)
    return keys




## === cell 4
if "resnext" in used_models_pytorch:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_transforms():
        return Compose(
            [
                Resize(512, 512),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def inference(model, state_dicts, test_loader, device):
        """
        Change (score-relevant & minimal): load each checkpoint ONCE and run full
        inference, then average probabilities. This preserves the original ensemble
        semantics (mean of softmaxes) but avoids repeatedly re-loading weights per batch,
        which can be slow and can mask issues.
        """
        model.to(device)
        model.eval()

        all_probs_per_ckpt = []
        for sd in state_dicts:
            model.load_state_dict(sd, strict=False)
            probs = []
            for images in tqdm(test_loader, desc="Infer resnext", leave=False):
                images = images.to(device, non_blocking=True)
                with torch.no_grad():
                    y_preds = model(images)
                probs.append(y_preds.softmax(1).to("cpu").numpy())
            all_probs_per_ckpt.append(np.concatenate(probs, axis=0))

        return np.mean(all_probs_per_ckpt, axis=0)

    predictions_resnext = pd.DataFrame(columns=["image_id"])
    predictions_resnext["image_id"] = submission_df["image_id"].values
    predictions_resnext["image_path_id"] = image_path + predictions_resnext[
        "image_id"
    ].astype(str)

    model = CustomResNext("resnext50_32x4d", pretrained=False)

    ckpt_paths = used_models_pytorch["resnext"]
    existing_paths = [p for p in ckpt_paths if os.path.exists(p)]

    if len(existing_paths) == 0:
        discovered = _discover_resnext_checkpoints("/kaggle/input", limit=16)
        if len(discovered) > 0:
            print("Configured ResNeXt checkpoints not found; discovered candidates:")
            for p in discovered[:16]:
                print(" -", p)
            existing_paths = discovered
        else:
            print(
                "Warning: no ResNeXt checkpoints found under /kaggle/input; ResNeXt inference will be skipped."
            )

    if len(existing_paths) > 0:
        loaded_states = []
        head_keys = _classifier_keys_for_model(model)

        for f in existing_paths:
            ckpt = torch.load(f, map_location="cpu")
            raw_sd = _extract_state_dict(ckpt)

            tmp_model = CustomResNext("resnext50_32x4d", pretrained=False)
            load_res, used_sd = _try_load_state_dict_into_model(tmp_model, raw_sd)

            missing = set(getattr(load_res, "missing_keys", []))
            unexpected = getattr(load_res, "unexpected_keys", [])

            if len(head_keys) > 0 and any(k in missing for k in head_keys):
                print(
                    f"Skipping checkpoint (head did not load -> likely incompatible): {f}"
                )
                print(f"  missing_head_keys={[k for k in head_keys if k in missing]}")
                del tmp_model
                gc.collect()
                continue

            if len(missing) > 2000:
                print(f"Skipping checkpoint (seems incompatible): {f}")
                print(
                    f"  missing_keys={len(missing)} unexpected_keys={len(unexpected)}"
                )
                del tmp_model
                gc.collect()
                continue

            print(
                f"Loaded checkpoint: {os.path.basename(f)} (missing={len(missing)}, unexpected={len(unexpected)})"
            )
            loaded_states.append(used_sd)

            del tmp_model
            gc.collect()

        states = loaded_states

        if len(states) == 0:
            print(
                "Warning: all discovered checkpoints were incompatible; ResNeXt inference will be skipped."
            )
        else:
            test_dataset = TestDataset(predictions_resnext, transform=get_transforms())
            test_loader = DataLoader(
                test_dataset,
                batch_size=16,
                shuffle=False,
                num_workers=2,
                pin_memory=torch.cuda.is_available(),
            )
            predictions = inference(model, states, test_loader, device)

            predictions_resnext["resnext"] = [np.squeeze(p) for p in predictions]
            predictions_resnext = predictions_resnext.drop(["image_path_id"], axis=1)

    torch.cuda.empty_cache()
    try:
        del model
    except Exception:
        pass
    gc.collect()



## === cell 5
submission_df["label"] = 0

if (
    "resnext" in used_models_pytorch
    and "predictions_resnext" in globals()
    and "resnext" in predictions_resnext.columns
):
    submission_df = submission_df.merge(predictions_resnext, on="image_id", how="left")

if "efficientnetb3" in used_models_pytorch and "predictions_cutmix" in globals():
    submission_df = submission_df.merge(predictions_cutmix, on="image_id", how="left")

if "vit2020" in used_models_pytorch and "predictions_vit" in globals():
    submission_df = submission_df.merge(predictions_vit, on="image_id", how="left")

if "vit2019" in used_models_pytorch and "predictions_vit2019" in globals():
    submission_df = submission_df.merge(predictions_vit2019, on="image_id", how="left")



## === cell 6
model_cols = list(used_models_pytorch.keys()) + list(used_models_keras.keys())
model_cols = [
    c for c in model_cols if c in submission_df.columns
]  # only those actually present

if stacked_mean and all(
    c in submission_df.columns
    for c in ["vit2020", "resnext", "mobilenet", "efficientnetb4"]
):
    submission_df["stage_1"] = submission_df.apply(
        lambda row: [np.mean(e) for e in zip(row["vit2020"], row["resnext"])], axis=1
    )
    submission_df["label"] = submission_df.apply(
        lambda row: int(
            np.argmax(
                [
                    np.sum(e)
                    for e in zip(
                        row["mobilenet"], row["stage_1"], row["efficientnetb4"]
                    )
                ]
            )
        ),
        axis=1,
    )
elif len(model_cols) > 0:
    submission_df["label"] = submission_df.apply(
        lambda row: int(
            np.argmax([np.sum(e) for e in zip(*[row[m] for m in model_cols])])
        ),
        axis=1,
    )
else:
    submission_df["label"] = 0



## === cell 7
print(submission_df.head(1))



## === cell 8
submission_path = "submission.csv"
submission_df[["image_id", "label"]].to_csv(submission_path, index=False)
print(f"Wrote {submission_path} with shape {submission_df[['image_id','label']].shape}")
print(pd.read_csv(submission_path).head())
