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

0.8952855847688124

# 6. Current score

0.08707

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0852) has done: 'I fix the run-blocking checkpoint path issue by making the script automatically locate an available `.pt/.pth` checkpoint under `../input/` (or fall back to `weights=IMAGENET1K_V1` if none exists) so inference always runs end-to-end. I also ensure `test["label"]` is always created (even if inference fails) so the submission merge cannot KeyError. Finally, I correct the seeding function to avoid the contradictory `deterministic=True` with `benchmark=True` (stability-only) and keep all core inference logic (TTA loop, model, transforms, argmax) unchanged.'
- What this solution (achieved 0.08707) has done: 'Your current score (0.0852) is far below the target (0.8953), and the most likely reason is that inference is effectively random because the code is not loading the intended cassava-trained checkpoint (it falls back to ImageNet weights or a mismatched checkpoint with many missing/unexpected keys). I make the smallest change that materially improves accuracy: prefer loading a checkpoint whose classifier head matches 5 classes and fail fast (instead of silently running with a bad/mismatched state_dict). I also fix the inference augmentation pipeline to use a deterministic center-crop/resize-only transform for test-time inference (your current pipeline uses strong random augmentations and coarse dropout during inference, which can severely hurt accuracy even when averaging). These changes keep the same model architecture, same inference averaging loop, same argmax post-processing, and still produce `submission.csv`.'
- What this solution (achieved 0.08707) has done: 'Your score is far below the target, so the most likely issue is still that you’re not actually loading cassava-trained weights (and/or you’re not finding the real test images directory), which makes predictions near-random. I make the smallest changes that (1) robustly locate the correct `test_images/` folder from the known Kaggle paths, and (2) robustly locate a compatible 5-class EfficientNet-B4 checkpoint by scanning only plausible cassava locations (instead of all `../input`), preferring filenames that look like cassava models and verifying the state_dict keys. I also make the checkpoint loader accept both `net.*` and plain torchvision `classifier.*` key formats (common mismatch), without changing the architecture or inference averaging logic. These changes are directly aimed at turning “random inference” into “using the intended trained weights”, which is the main lever to move accuracy toward ~0.89.'
- What this solution (achieved 0.08707) has done: 'Your score is far below target, so the primary issue is still that inference is running with wrong/untrained weights (effectively near-random). I make the smallest change that materially improves accuracy: load the official cassava-trained EfficientNet-B4 checkpoint from the competition dataset (`../input/cassava-leaf-disease-classification/`), which matches your model and 5-class head. To keep the same architecture and inference loop, I only adjust the checkpoint search order and make the loader remap common key prefixes (e.g., `model.` / `net.`) so the correct weights actually load. This should move accuracy toward the target without altering your TTA/averaging or post-processing.'
- What this solution (achieved 0.08707) has done: 'Your current score is far below target, so the highest-impact minimal fix is to ensure we are actually using cassava-trained weights rather than an incompatible/empty checkpoint fallback. I keep your model, transforms, and inference averaging loop unchanged, but make checkpoint discovery prioritize only files that can be verified as EfficientNet-B4 5-class weights (by checking multiple possible classifier key patterns) and correctly remap common key prefixes so good checkpoints don’t get rejected. I also reduce false “incompatible” rejections by validating compatibility more directly (matching tensor shapes) instead of counting missing/unexpected keys, which can be high for harmless metadata but still load correctly. This should move predictions from near-random toward the target accuracy while still producing the same `submission.csv` format end-to-end.'

# 9. Code solution

## === cell 0
import os
import gc
import time
import random
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from tqdm import tqdm
from torchvision import models

warnings.filterwarnings("ignore")




## === cell 1
image_size = 380




## === cell 2
config = dict(
    seed=22,
    experiment_name="modified",
    test_location="../input/cassava-leaf-disease-classification/test_images",
    checkpoint_path="../input/saved-models",
    checkpoint="baselineepoch20.pt",
    model="efficientnet-b4",  # kept for compatibility with existing config
    epochs=10,
    batch_size=16,
    workers=8,
    inference_augmentations=[
        dict(name="LongestMaxSize", params=dict(max_size=image_size, p=1.0)),
        dict(
            name="PadIfNeeded",
            params=dict(
                min_height=image_size,
                min_width=image_size,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0,
            ),
        ),
        dict(
            name="CenterCrop", params=dict(height=image_size, width=image_size, p=1.0)
        ),
        dict(
            name="Normalize",
            params=dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ),
    ],
)




## === cell 3
sample_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
sample = pd.read_csv(sample_path)
test = sample[["image_id"]].copy()
test.head()




## === cell 4
def seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False




## === cell 5
seed(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def resolve_test_location():
    candidates = [
        Path(config["test_location"]),
        Path("../input/test_images"),
        Path("../input/cassava-leaf-disease-classification/test_images"),
        Path(
            "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images"
        ),
        Path("/kaggle/input/cassava-leaf-disease-classification/test_images"),
        Path(
            "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images"
        ),
        Path("/kaggle/data/cassava-leaf-disease-classification/test_images"),
    ]
    for p in candidates:
        try:
            if p.exists():
                jpgs = list(p.glob("*.jpg"))
                if len(jpgs) > 0:
                    return str(p)
        except Exception:
            continue
    return config["test_location"]


config["test_location"] = resolve_test_location()
print("Resolved test_location:", config["test_location"])




## === cell 6
class EfficientNetB4Like(nn.Module):
    """
    Replace efficientnet_pytorch dependency with torchvision EfficientNet-B4.
    Keep the same 'num_classes=5' head.
    """

    def __init__(self, num_classes=5, weights=None):
        super().__init__()
        self.net = models.efficientnet_b4(weights=weights)
        in_features = self.net.classifier[1].in_features
        self.net.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)


def _strip_module_prefix(state_dict):
    if not state_dict:
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def _extract_state_dict(checkpoint):
    state_dict = None
    meta = {}
    if isinstance(checkpoint, dict):
        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        else:
            tensor_values = [v for v in checkpoint.values() if torch.is_tensor(v)]
            if len(tensor_values) == len(checkpoint):
                state_dict = checkpoint
        meta = checkpoint
    else:
        state_dict = checkpoint
    return state_dict, meta


def _remap_state_dict_for_wrapper(sd):
    if not isinstance(sd, dict) or len(sd) == 0:
        return sd

    if any(k.startswith("net.") for k in sd.keys()):
        return sd

    if any(k.startswith("model.") for k in sd.keys()):
        return {
            ("net." + k[len("model.") :]) if k.startswith("model.") else k: v
            for k, v in sd.items()
        }

    if any(k.startswith("features.") or k.startswith("classifier.") for k in sd.keys()):
        return {f"net.{k}": v for k, v in sd.items()}

    return sd


def _classifier_out_features_from_state_dict(sd):
    prefixes = [
        "net.classifier.1",
        "classifier.1",
        "model.classifier.1",
        "net.fc",
        "fc",
        "model.fc",
        "net.head",
        "head",
        "model.head",
    ]
    for prefix in prefixes:
        w = sd.get(f"{prefix}.weight", None)
        b = sd.get(f"{prefix}.bias", None)
        if torch.is_tensor(w) and w.ndim == 2:
            return int(w.shape[0])
        if torch.is_tensor(b) and b.ndim == 1:
            return int(b.shape[0])
    return None


def _shape_compatibility_ratio(model, sd):
    msd = model.state_dict()
    overlap = 0
    good = 0
    for k, v in sd.items():
        if k in msd and torch.is_tensor(v) and torch.is_tensor(msd[k]):
            overlap += 1
            if tuple(v.shape) == tuple(msd[k].shape):
                good += 1
    if overlap == 0:
        return 0.0, 0
    return good / overlap, overlap


def _likely_ckpt_dirs():
    return [
        Path("../input/cassava-leaf-disease-classification"),
        Path(
            "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification"
        ),
        Path("../input/saved-models"),
        Path("../input"),
        Path("/kaggle/input/cassava-leaf-disease-classification"),
        Path("/kaggle/input"),
    ]


def _find_best_checkpoint():
    preferred = Path(config["checkpoint_path"]) / config["checkpoint"]
    candidates = []
    if preferred.exists():
        candidates.append(preferred)

    patterns = ["*.pt", "*.pth", "*.bin"]
    for root in _likely_ckpt_dirs():
        try:
            if root.exists():
                for pat in patterns:
                    candidates.extend(root.rglob(pat))
        except Exception:
            continue

    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        return None

    scored = []
    for p in candidates:
        name = p.name.lower()
        kw = 0
        for k in [
            "cassava",
            "leaf",
            "disease",
            "efficientnet",
            "eff",
            "b4",
            "baseline",
            "epoch",
            "best",
            "fold",
        ]:
            if k in name:
                kw += 1

        out5 = 0
        readable = 0
        try:
            ckpt = torch.load(str(p), map_location="cpu")
            sd, _ = _extract_state_dict(ckpt)
            if isinstance(sd, dict):
                readable = 1
                sd = _strip_module_prefix(sd)
                sd = _remap_state_dict_for_wrapper(sd)
                of = _classifier_out_features_from_state_dict(sd)
                if of == 5:
                    out5 = 1
        except Exception:
            readable = 0
            out5 = 0

        try:
            sz = p.stat().st_size
        except Exception:
            sz = 0

        is_preferred = 1 if p == preferred else 0
        in_comp_folder = (
            1 if "cassava-leaf-disease-classification" in str(p).lower() else 0
        )

        scored.append((is_preferred, out5, in_comp_folder, kw, readable, sz, p))

    scored.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4], x[5]), reverse=True)
    best = scored[0][-1]
    return str(best)


def load_model():
    ckpt_path = _find_best_checkpoint()
    if ckpt_path is None:
        print(
            "WARNING: No checkpoint found under expected inputs. Falling back to ImageNet pretrained EfficientNet-B4 weights."
        )
        weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
        model = EfficientNetB4Like(num_classes=5, weights=weights).to(device)
        return model

    print(f"Loading checkpoint: {ckpt_path}")
    model = EfficientNetB4Like(num_classes=5, weights=None)

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict, meta = _extract_state_dict(checkpoint)

    if state_dict is None or not isinstance(state_dict, dict):
        raise ValueError(
            "Could not find model weights in checkpoint (expected a dict or keys: 'model' or 'state_dict')."
        )

    state_dict = _strip_module_prefix(state_dict)
    state_dict = _remap_state_dict_for_wrapper(state_dict)

    out_features = _classifier_out_features_from_state_dict(state_dict)
    if out_features is not None and out_features != 5:
        raise ValueError(
            f"Checkpoint head out_features={out_features} but expected 5 classes. "
            f"Refusing to run with mismatched weights: {ckpt_path}"
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    ratio, overlap = _shape_compatibility_ratio(model, state_dict)

    if overlap < 50 or ratio < 0.90:
        raise ValueError(
            f"Checkpoint appears incompatible by shape check (overlap={overlap}, shape_match_ratio={ratio:.3f}). "
            f"Refusing to run to avoid near-random predictions. Path: {ckpt_path}"
        )

    if isinstance(meta, dict):
        for k in ["epoch", "train_loss", "val_loss", "metrics", "lr"]:
            if k in meta:
                print(f"{k}: {meta[k]}")
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
    print(f"Shape match ratio over overlap keys: {ratio:.3f} (overlap={overlap})")

    model.to(device)
    return model




## === cell 7
def get_transforms():
    transforms = []
    for item in config["inference_augmentations"]:
        name = item["name"]
        params = item["params"]
        if not hasattr(A, name):
            continue
        transforms.append(getattr(A, name)(**params))
    return A.Compose(transforms)




## === cell 8
class CassavaDataset(Dataset):
    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __getitem__(self, n):
        img_name = self.images[n]
        img_path = os.path.join(config["test_location"], img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        image = np.moveaxis(image, -1, 0)
        image = torch.tensor(image, dtype=torch.float32)
        return image

    def __len__(self):
        return len(self.images)




## === cell 9
def get_dataloader():
    transforms = get_transforms()
    test_data = np.array(test["image_id"].values)

    data = CassavaDataset(test_data, transforms)
    dataloader = DataLoader(
        data,
        shuffle=False,
        batch_size=config["batch_size"],
        pin_memory=torch.cuda.is_available(),
        num_workers=min(config["workers"], os.cpu_count() or 1),
        drop_last=False,
    )
    return dataloader




## === cell 10
def infer(model, dataloader):
    print("Running inference...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device, non_blocking=torch.cuda.is_available())
            batch_hat = model(batch)
            predictions.append(batch_hat.detach().cpu())

    return torch.cat(predictions, dim=0)




## === cell 11
if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    dataloader = get_dataloader()
    model = load_model()
    predictions = None
    print("Inferring experiment", config["experiment_name"])

    for epoch in range(config["epochs"]):
        print("Epoch:", epoch)
        start_time = time.time()

        if epoch == 0:
            predictions = infer(model, dataloader)
        else:
            predictions += infer(model, dataloader)

        print("Time:", time.time() - start_time)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    predictions /= config["epochs"]
    results = predictions.numpy()

    test["label"] = np.argmax(results, axis=-1).astype(int)




## === cell 12
sub = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")
if "label" not in test.columns:
    test["label"] = 0

sub = sub.drop(columns=["label"], errors="ignore")
sub = sub.merge(test[["image_id", "label"]], on="image_id", how="left")

sub["label"] = sub["label"].fillna(0).astype(int)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with rows:", len(sub))
