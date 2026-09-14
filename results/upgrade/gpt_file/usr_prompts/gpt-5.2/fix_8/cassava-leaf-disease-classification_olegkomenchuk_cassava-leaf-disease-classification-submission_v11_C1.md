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

3.10

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

0.885766092475068

# 6. Current score

0.40209

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the import/path/runtime issues so the notebook runs end-to-end and always writes a valid `submission.csv`. Specifically, I remove the failing external wheel install and switch the EfficientNet backbone to `torchvision.models.efficientnet_b0` (keeps the same core EfficientNet-B0 idea while using an available package). I also fix cell numbering/order, ensure `Path`, `Dataset`, `DataLoader`, and albumentations imports are defined before use, and make the test dataframe use the correct `image_id` column alignment (not index) so predictions map to the right files. Finally, I make checkpoint loading optional (since the referenced path doesn’t exist here) so inference can still run and produce a submission even without the external weights.'
- What this solution (achieved 0.61099) has done: 'Your current score suggests the model is running with random weights because the checkpoint path doesn’t exist, so the smallest meaningful improvement is to load a real pretrained checkpoint from the dataset’s provided EfficientNet-B0 Fold-3 weights. I change `model_path` to a path that exists under `/kaggle/input/cassava-leaf-disease-classification/` and make the loader robust to common key prefixes (`module.`) and different checkpoint dict formats, without changing the model architecture or inference logic. This should substantially increase accuracy toward the target while keeping the rest of your pipeline identical. I also keep the submission alignment logic intact to avoid mis-ordering predictions.'
- What this solution (achieved 0.61099) has done: 'Your current score strongly suggests inference is still using randomly initialized weights (or a mismatched checkpoint) even though the code tries to load a file, so the smallest effective change is to (1) automatically discover an existing `.pth/.pt/.torch` under the dataset directory and (2) load it robustly by accepting common key names and allowing `strict=False` if the head shape differs, while keeping the same EfficientNet-B0 model definition. I also keep your exact preprocessing and prediction logic, but I add a safety print of missing/unexpected keys so you can confirm the checkpoint actually applied. Finally, I keep submission alignment identical (merge on `image_id`) to avoid any ordering mistakes while ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.61099) is far below the target (0.8858), and the most likely cause is that you are still not actually loading the intended trained weights (the referenced `effnetb0f3/...` path probably doesn’t exist in this dataset layout, so you end up effectively using random weights or a mismatched checkpoint). I make the smallest change that directly impacts accuracy: robustly locate an EfficientNet-B0 cassava checkpoint under `/kaggle/input` and *require* a successful, shape-compatible load (use `strict=True` when possible; only fall back to `strict=False` if the head is incompatible). I also add the missing resize normalization detail: EfficientNet-B0 on ImageNet typically expects 224x224; keeping 512 can work but often hurts if the checkpoint was trained at 224, so I switch `image_size` to `(224, 224)` (minimal config-only change, same model/inference loop). These two changes should move the score substantially toward the target without changing the core architecture or inference semantics.'
- What this solution (achieved 0.40209) has done: 'Your current score is far below the target, and the most likely reason is that you’re still not actually loading a compatible trained checkpoint (so inference is effectively random/near-ImageNet). I make the smallest changes that directly impact accuracy while preserving your model/inference logic: (1) instantiate EfficientNet-B0 with ImageNet pretrained backbone weights (head still replaced to 5 classes), and (2) make checkpoint loading stricter by preferring checkpoints that clearly match your 5-class head and only accepting non-strict loads when they still provide a valid 5-class classifier. This should move accuracy substantially upward toward the target without changing the data pipeline, augmentations, batching, or prediction method. The script still always write a valid `submission.csv`.'
- What this solution (achieved 0.40209) has done: 'Your score is far below the target, and with this code the most likely reason is that no real cassava-trained checkpoint is being loaded (so you’re effectively doing ImageNet features + random 5-class head). I make the smallest change that directly improves accuracy: (1) automatically pick the *best-matching* cassava EfficientNet-B0 checkpoint by verifying it contains a 5-class `classifier.1.*` head, and (2) require a strict load for that best candidate (fall back only if needed, but still reject checkpoints that don’t provide the 5-class head). This keeps your model architecture, preprocessing, and inference loop the same, while greatly increasing the chance you’re using the intended trained weights. The script still run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 0.40209) has done: 'Your score (0.402) is far below the target, and with this pipeline the most likely cause is still that no cassava-trained checkpoint is being loaded and/or the loaded weights don’t actually match the model (so you’re effectively using an ImageNet backbone with a random 5-class head). I make the smallest changes that directly increase the chance of loading a compatible cassava EfficientNet-B0 checkpoint: first, search only in the competition input directory (to avoid irrelevant `.pth` files), then require the checkpoint to contain a 5-class `classifier.1.*` head and *load that head strictly*. If the backbone keys don’t match exactly, I still allow a non-strict load for the backbone, but only after confirming the 5-class head weights were loaded, so accuracy should move up toward your target while keeping the same model/inference logic. I also print which checkpoint was used and whether the classifier head was loaded to make the result auditable.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import random

import pandas as pd
import cv2 as cv

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from albumentations import Compose, Normalize
from albumentations.pytorch import ToTensorV2

import torchvision




## === cell 1
def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)




## === cell 2
class Config:
    cfg = {
        "batch_size": 32,
        "num_workers": 2,
        "image_size": (224, 224),
        "num_classes": 5,
        "model_path": "/kaggle/input/cassava-leaf-disease-classification/effnetb0f3/3_fold_model_effnet_b0_best.torch",
    }




## === cell 3
base_dir = Path("/kaggle/input/cassava-leaf-disease-classification")
test_img_dir = base_dir / "test_images"
test_df = pd.read_csv(base_dir / "sample_submission.csv")  # columns: image_id,label

assert "image_id" in test_df.columns, "sample_submission.csv must contain image_id"
print(test_df.head(), test_df.shape)




## === cell 4
class CassavaDataset(Dataset):
    def __init__(self, df: pd.DataFrame, image_dir: Path, image_size, augments=None):
        self.image_ids = df["image_id"].tolist()
        self.image_dir = Path(image_dir)
        self.image_size = image_size
        self.augments = augments

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = self.image_dir / image_id

        image = cv.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image = cv.resize(image, self.image_size, interpolation=cv.INTER_AREA)

        if self.augments:
            image = self.augments(image=image)["image"]

        return {"X": image, "image_id": image_id}

    def __len__(self):
        return len(self.image_ids)




## === cell 5
class Augments:
    test_augments = Compose(
        [
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )




## === cell 6
test_dataset = CassavaDataset(
    df=test_df,
    image_dir=test_img_dir,
    image_size=Config.cfg["image_size"],
    augments=Augments.test_augments,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=Config.cfg["batch_size"],
    shuffle=False,
    num_workers=Config.cfg["num_workers"],
    pin_memory=torch.cuda.is_available(),
)

print("test size:", len(test_dataset))
print("batch X shape:", next(iter(test_dataloader))["X"].shape)




## === cell 7
def efficientnet_b0(num_classes: int):
    try:
        weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
    except Exception:
        weights = "IMAGENET1K_V1"
    model = torchvision.models.efficientnet_b0(weights=weights)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(
        in_features=in_features, out_features=num_classes, bias=True
    )
    return model


model = efficientnet_b0(Config.cfg["num_classes"]).to(device)




## === cell 8
def _strip_prefix_from_state_dict(state_dict, prefix: str):
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items()}


def _find_checkpoint_paths(search_root: Path):
    exts = (".pth", ".pt", ".torch")
    hits = []
    if search_root.exists():
        for ext in exts:
            hits.extend([str(p) for p in search_root.rglob(f"*{ext}")])
    return sorted(set(hits))


def _extract_state_dict(checkpoint_obj):
    state_dict = None
    if isinstance(checkpoint_obj, dict):
        for k in ["model_state_dict", "state_dict", "model", "net", "weights"]:
            if k in checkpoint_obj and isinstance(checkpoint_obj[k], dict):
                state_dict = checkpoint_obj[k]
                break
        if state_dict is None and all(
            isinstance(k, str) for k in checkpoint_obj.keys()
        ):
            state_dict = checkpoint_obj
    else:
        state_dict = checkpoint_obj
    return state_dict


def _normalize_state_dict_keys(sd: dict) -> dict:
    sd = _strip_prefix_from_state_dict(sd, "module.")
    sd = _strip_prefix_from_state_dict(sd, "model.")
    sd = _strip_prefix_from_state_dict(sd, "net.")
    return sd


def _get_head_out_features(sd: dict, head_key: str = "classifier.1.weight"):
    w = sd.get(head_key, None)
    if w is None or not hasattr(w, "shape") or len(w.shape) != 2:
        return None
    return int(w.shape[0])


def _ckpt_score(path: str, sd: dict) -> tuple:
    """
    Change rationale (score improvement): aggressively prefer checkpoints that clearly
    contain a 5-class EfficientNet-B0 classifier head, so we avoid accidentally loading
    unrelated checkpoints (which yields low accuracy).
    Higher tuple is better.
    """
    p = path.lower()
    has_eff = ("eff" in p) or ("efficientnet" in p)
    has_b0 = "b0" in p
    has_fold = "fold" in p
    out_feats = _get_head_out_features(sd)
    head_is_5 = out_feats == Config.cfg["num_classes"]
    head_exists = out_feats is not None
    under_comp_dir = str(Path(p)).startswith(str(base_dir).lower())
    return (
        int(head_is_5),
        int(under_comp_dir),
        int(has_eff and has_b0),
        int(has_fold),
        int(head_exists),
    )


ckpt_path = Config.cfg["model_path"]
checkpoint_loaded = False
ckpt_path_found = None
classifier_head_loaded = False

search_roots = [base_dir]

discovered = []
for root in search_roots:
    discovered.extend(_find_checkpoint_paths(root))
discovered = sorted(set(discovered))

candidates = []
if ckpt_path and os.path.exists(ckpt_path):
    candidates.append(ckpt_path)
for p in discovered:
    if p not in candidates:
        candidates.append(p)

ranked = []
last_error = None
for p in candidates:
    try:
        checkpoint = torch.load(p, map_location="cpu")
        sd = _extract_state_dict(checkpoint)
        if sd is None or not isinstance(sd, dict):
            continue
        sd = _normalize_state_dict_keys(sd)
        ranked.append((_ckpt_score(p, sd), p, sd))
    except Exception as e:
        last_error = e
        continue

ranked.sort(key=lambda x: x[0], reverse=True)

best = None
for score_tuple, p, sd in ranked:
    if _get_head_out_features(sd) == Config.cfg["num_classes"]:
        best = (p, sd, score_tuple)
        break

if best is not None:
    p, state_dict, score_tuple = best

    head_w_key = "classifier.1.weight"
    head_b_key = "classifier.1.bias"
    has_head = (head_w_key in state_dict) and (head_b_key in state_dict)
    head_ok = False
    if has_head:
        try:
            expected_w = model.state_dict()[head_w_key].shape
            expected_b = model.state_dict()[head_b_key].shape
            head_ok = (tuple(state_dict[head_w_key].shape) == tuple(expected_w)) and (
                tuple(state_dict[head_b_key].shape) == tuple(expected_b)
            )
        except Exception:
            head_ok = False

    if not head_ok:
        last_error = RuntimeError(
            f"Rejected checkpoint without a compatible 5-class head: {p}"
        )
    else:
        try:
            model.load_state_dict(state_dict, strict=True)
            ckpt_path_found = p
            checkpoint_loaded = True
            classifier_head_loaded = True
        except RuntimeError as e_strict:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if (head_w_key in missing) or (head_b_key in missing):
                last_error = RuntimeError(
                    f"Rejected non-strict load: head missing from {p}"
                )
            else:
                ckpt_path_found = p
                checkpoint_loaded = True
                classifier_head_loaded = True
                print(
                    f"Loaded checkpoint (non-strict backbone fallback): {ckpt_path_found}"
                )
                print("Missing keys (first 20):", missing[:20])
                print("Unexpected keys (first 20):", unexpected[:20])

model.to(device)

if checkpoint_loaded:
    print(f"Checkpoint successfully loaded from: {ckpt_path_found}")
    print("classifier_head_loaded:", classifier_head_loaded)
else:
    print(
        "WARNING: No usable cassava 5-class checkpoint was loaded; using ImageNet-pretrained EfficientNet-B0 backbone with a randomly initialized 5-class head."
    )
    if last_error is not None:
        print("Last checkpoint load error (for debugging):", repr(last_error))
    if len(ranked) > 0:
        print("Top 5 discovered checkpoint candidates by score (score_tuple, path):")
        for tup, p, _sd in ranked[:5]:
            print(tup, p)



## === cell 9
model.eval()

y_prediction = []
image_ids_out = []

with torch.no_grad():
    for batch in test_dataloader:
        X_test = batch["X"].to(device, non_blocking=True)
        logits = model(X_test)
        preds = logits.argmax(dim=-1).detach().cpu().numpy().tolist()

        y_prediction.extend(preds)
        image_ids_out.extend(batch["image_id"])

print("num predictions:", len(y_prediction), "num image_ids:", len(image_ids_out))
print(
    "checkpoint_loaded:",
    checkpoint_loaded,
    "classifier_head_loaded:",
    classifier_head_loaded,
)



## === cell 10
pred_df = pd.DataFrame({"image_id": image_ids_out, "label": y_prediction})
sub = test_df[["image_id"]].merge(pred_df, on="image_id", how="left")

sub["label"] = sub["label"].fillna(0).astype(int)

assert sub.shape[0] == test_df.shape[0]
assert list(sub.columns) == ["image_id", "label"]

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
