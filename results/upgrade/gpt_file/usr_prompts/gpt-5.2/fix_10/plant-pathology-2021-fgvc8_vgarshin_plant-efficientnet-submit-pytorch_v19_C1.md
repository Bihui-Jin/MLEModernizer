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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7627146814404437

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I remove the broken `pip install` step and make `efficientnet_pytorch` optional by switching to a torchvision EfficientNet implementation when it’s not available, keeping the same model head and inference semantics. I also harden the environment setup so `KAGGLE`, paths, and `DEVICE` are always defined, even if CUDA is unavailable, and ensure the code can run without external model packs by falling back to a safe “healthy” submission if weights/params aren’t found. Finally, I fix the TTA/model ensembling aggregation bug that produced `numpy.float64 object is not iterable`, replacing it with a correct average over models and TTAs and writing a valid `submission.csv` with the required `image,labels` columns.'
- What this solution (achieved 0.24507) has done: 'Your current low score is mainly because the notebook is usually falling back to an all-“healthy” submission when it can’t find `params.json`/`ths.json` and (more importantly) when it can’t find/load the pretrained weights in `MDLS_PATH`. I make the path resolution for `MDLS_PATH` robust by searching common Kaggle input locations for the `plant-models-{VER}` dataset, so the intended trained weights/metadata are actually found and used. I also add a safe fallback to derive `LABELS_`/`LABELS` from `train.csv` when `params.json` is missing, so inference still produces multi-label outputs instead of being forced into “healthy”. These are minimal changes that preserve your model/inference logic but should move the score upward toward the target by ensuring the real ensemble predictions are used.'
- What this solution (achieved 0.24507) has done: 'Your current gap to the target is large (0.245 → 0.763), and the most likely cause is that inference is using randomly initialized backbones when `weights=None`, even though your saved heads were trained on pretrained backbones. I keep your architecture and inference loop identical, but load the correct ImageNet pretrained weights for the torchvision EfficientNet/ResNeXt fallbacks so the checkpoint weights align with the expected feature extractor. I also make the state-dict loading tolerant to the common “module.” prefix mismatch (caused by DataParallel) without changing model structure, so more checkpoints actually load instead of being silently skipped. These two minimal fixes should materially increase F1 toward your target while preserving core logic and output format.'
- What this solution (achieved 0.24507) has done: 'Your current score is far below target, and the most likely cause (given your safeguards) is that the EfficientNet fallback is not producing the same feature tensor shape as your trained checkpoints expect: in torchvision EfficientNet, calling the model returns class logits, not pooled features, so your custom `myfc` is being fed the wrong representation and predictions collapse toward “healthy”. I keep your architecture and inference loop intact, but fix `EffNet.extract()` for the torchvision path to return pooled features (using `features -> avgpool -> flatten`), matching what `efficientnet_pytorch` does after removing `_fc`. I also make `get_labels()` handle the “healthy + diseases” case correctly by removing `healthy` when other labels are present (this aligns better with the competition’s multi-label semantics and typically increases mean F1 without changing the model). These are minimal, metric-aligned fixes intended to move your score upward toward the target while preserving everything else and still producing a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'Your current score is far below target, so the smallest high-impact fix is to ensure the submission rows align 1:1 with the actual `test_images` directory rather than relying on `sample_submission.csv` (which can mismatch in notebooks and silently ruin F1). I keep your model/weights/ensemble logic unchanged, but rebuild `df_sub` from the filenames found in `IMGS_PATH`, then sort them for deterministic ordering. I also make label-index mapping consistent when `params.json` is missing by always using a stable `index->label` dict derived from sorted unique labels in `train.csv` (this avoids thresholding against the wrong class order). These are minimal changes focused on correctness/alignment, which should move your score upward toward the target without changing architecture or inference semantics.'
- What this solution (achieved 0.24507) has done: 'Your current score (0.245) is far below the target (0.763), and the most common “silent killer” here is label-order mismatch: when `params.json`/`LABELS` isn’t available (or differs from what the checkpoints were trained with), your thresholding decodes the wrong classes and collapses predictions toward “healthy”. I keep your model and inference unchanged, but ensure `LABELS_`/`LABELS` are always derived in the exact same order as the competition’s official class set (from `train.csv`, in a stable order) and that the mapping is consistent with the model output dimension. I also add a strict sanity check that the loaded model head output size matches `len(LABELS_)` so we don’t generate mis-decoded submissions. These are minimal, correctness-focused changes that should move the score upward toward your target without changing architecture or evaluation semantics.'
- What this solution (achieved 0.24507) has done: 'Your current score is far below the target, so the smallest high-impact change is to fix input normalization: your models were almost certainly trained with ImageNet normalization, but inference currently feeds only `[0,1]` scaled pixels, which commonly collapses predictions and tanks mean F1. I keep your dataset/model/ensemble logic identical, but apply the correct EfficientNet/ResNeXt ImageNet mean/std normalization in `PlantDataset` for inference. I also make sure TTA is applied before normalization (so flips don’t mix normalized channels incorrectly) and keep the submission writing unchanged.'
- What this solution (achieved 0.24507) has done: 'Your current score is far below target, so the smallest likely high-impact change is to fix a subtle but critical TTA/normalization mismatch: in your test path you normalize after applying TTA, but in the train path you never normalize at all (and the TTA is applied in a different order), which can cause predictions to collapse and hurt mean F1. I make the inference preprocessing consistent and metric-aligned by applying TTA on the uint8 image first, then scaling to [0,1], then ImageNet-normalizing (matching typical EfficientNet/ResNeXt training), without changing your model architecture, ensembling, thresholds, or decoding logic. I also ensure the same normalization is used in the (rarely used) training-label branch to keep semantics consistent and avoid silent divergence if you later toggle `TEST=False`. Everything else (paths, weights loading, ensemble averaging, submission format) stays the same.'
- What this solution (achieved 0.24507) has done: 'Your score is far below target, so the smallest likely high-impact fix is to make inference use the exact same class order and thresholds that the checkpoints were trained with; right now, if `params.json`/`ths.json` aren’t found (or mismatch), decoding can be catastrophically wrong even though the model runs. I make `MDLS_PATH` resolution stricter by also searching for `model_best_*.pth` under common Kaggle input dirs and then loading `params.json`/`ths.json` from the same located directory, ensuring labels/thresholds align with the weights. I also add a minimal safety fallback: if thresholds are missing, compute per-class thresholds from training label frequencies (kept conservative) rather than defaulting all to 0.5, which tends to under-predict rare diseases and hurts mean F1. Core model, preprocessing, TTA/ensembling, and submission formatting remain unchanged.'

# 9. Code solution

## === cell 0
import os
import sys
import gc
import json
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torch.utils.data.sampler import SequentialSampler

warnings.filterwarnings("ignore")

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)



## === cell 1
pass



## === cell 2
try:
    from efficientnet_pytorch import model as enet  # original dependency

    HAS_EFFNET_PYTORCH = True
    print("Using efficientnet_pytorch")
except Exception as e:
    HAS_EFFNET_PYTORCH = False
    enet = None
    print(
        "efficientnet_pytorch not available; will use torchvision EfficientNet fallback.",
        repr(e),
    )



## === cell 3
TEST = True
VER = "v103"

if KAGGLE:
    if Path("../input/plant-pathology-2021-fgvc8").exists():
        DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    elif Path("/kaggle/input/plant-pathology-2021-fgvc8").exists():
        DATA_PATH = "/kaggle/input/plant-pathology-2021-fgvc8"
    elif Path("/kaggle/data/plant-pathology-2021-fgvc8").exists():
        DATA_PATH = "/kaggle/data/plant-pathology-2021-fgvc8"
    else:
        DATA_PATH = "../input/plant-pathology-2021-fgvc8"

    def _find_models_dir(ver: str) -> str:
        candidate_names = [
            f"plant-models-{ver}",
            f"plant_models_{ver}",
            f"models_{ver}",
            f"plant-models-{ver}".replace("-", "_"),
        ]
        base_dirs = [Path("../input"), Path("/kaggle/input"), Path("/kaggle/data")]
        direct_candidates = []
        for base in base_dirs:
            if not base.exists():
                continue
            for nm in candidate_names:
                p = base / nm
                direct_candidates.append(p)
            try:
                for child in base.iterdir():
                    if child.is_dir():
                        for nm in candidate_names:
                            direct_candidates.append(child / nm)
            except Exception:
                pass

        for p in direct_candidates:
            try:
                if (
                    p.exists()
                    and p.is_dir()
                    and len(list(p.glob("model_best_*.pth"))) > 0
                ):
                    return str(p)
            except Exception:
                pass

        for p in direct_candidates:
            if p.exists() and p.is_dir():
                return str(p)

        for base in base_dirs:
            if not base.exists():
                continue
            try:
                for pth in base.rglob("model_best_*.pth"):
                    return str(pth.parent)
            except Exception:
                pass

        return f"../input/plant-models-{ver}"

    MDLS_PATH = _find_models_dir(VER)
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TTAS = [0, 1, 2]
FOLDS = [0]
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()

print("DATA_PATH:", DATA_PATH)
print("IMGS_PATH:", IMGS_PATH)
print("MDLS_PATH:", MDLS_PATH)



## === cell 4
params_path = Path(MDLS_PATH) / "params.json"
ths_path = Path(MDLS_PATH) / "ths.json"


def _resolve_train_csv(data_path: str) -> Path:
    candidates = [
        Path(data_path) / "train.csv",
        Path("/kaggle/input/plant-pathology-2021-fgvc8/train.csv"),
        Path("../input/plant-pathology-2021-fgvc8/train.csv"),
        Path("/kaggle/data/plant-pathology-2021-fgvc8/train.csv"),
        Path("/kaggle/data/train.csv"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path(data_path) / "train.csv"


def _build_labels_from_train_csv(data_path: str):
    """
    Derive labels in a stable, competition-consistent order directly from train.csv.
    """
    train_path = _resolve_train_csv(data_path)
    if not train_path.exists():
        return [], {}, {}

    df_train = pd.read_csv(train_path)
    all_labs = set()
    for s in df_train["labels"].astype(str).values:
        for t in s.split():
            t = t.strip()
            if t:
                all_labs.add(t)

    labels_ = sorted(list(all_labs))
    label_to_idx = {lab: i for i, lab in enumerate(labels_)}
    idx_to_label = {str(i): lab for i, lab in enumerate(labels_)}
    return labels_, label_to_idx, idx_to_label


def _build_freq_thresholds_from_train_csv(data_path: str, labels_: list):
    """
    Change (score-relevant, minimal): if ths.json is missing, use conservative per-class
    thresholds based on label prevalence. This usually improves mean F1 vs a flat 0.5
    in long-tailed multi-label problems, without changing model logic.
    """
    train_path = _resolve_train_csv(data_path)
    if (not train_path.exists()) or len(labels_) == 0:
        return {}

    df_train = pd.read_csv(train_path)
    counts = dict.fromkeys(labels_, 0)
    n = len(df_train)
    for s in df_train["labels"].astype(str).values:
        present = set([t.strip() for t in s.split() if t.strip()])
        for t in present:
            if t in counts:
                counts[t] += 1

    ths = {}
    for i, lab in enumerate(labels_):
        p = counts[lab] / max(n, 1)
        th = 0.5 - 0.25 * (1.0 - min(1.0, p / 0.15))  # p>=0.15 -> ~0.5 ; p<< -> ~0.25
        th = float(np.clip(th, 0.25, 0.5))
        ths[str(i)] = th
    return ths


if params_path.exists():
    with open(params_path) as file:
        params = json.load(file)
    LABELS_ = params["labels_"]
    LABELS = params["labels"]
    WORKERS = 2 if KAGGLE else params.get("workers", 2)
    print(
        "loaded params:",
        {k: params.get(k) for k in ["img_size", "batch_size", "dropout", "backbone"]},
    )
    print("num labels:", len(LABELS_))
else:
    LABELS_, _lbl_to_idx, _idx_to_lbl = _build_labels_from_train_csv(DATA_PATH)
    if len(LABELS_) > 0:
        LABELS = _idx_to_lbl  # get_labels expects index->name mapping as strings
    else:
        LABELS_, LABELS = [], {}

    params = {
        "img_size": 512,
        "batch_size": 8,
        "dropout": 0.2,
        "backbone": "efficientnet-b0",
    }
    WORKERS = 2
    print("WARNING: params.json not found; using fallback params:", params)
    print("Derived num labels from train.csv:", len(LABELS_))

if ths_path.exists():
    with open(ths_path) as file:
        ths = json.load(file)
    print("thresholds loaded:", list(ths.items())[:3], "...")
else:
    ths = {}
    if len(LABELS_) > 0:
        ths = _build_freq_thresholds_from_train_csv(DATA_PATH, LABELS_)
        print(
            "WARNING: ths.json not found; using prevalence-based thresholds. Example:",
            list(ths.items())[:3],
            "...",
        )
    else:
        print(
            "WARNING: ths.json not found and labels unknown; thresholds dict is empty (defaults to 0.5)."
        )

if len(LABELS_) > 0:
    if not isinstance(LABELS, dict) or len(LABELS) != len(LABELS_):
        LABELS = {str(i): lab for i, lab in enumerate(LABELS_)}
        print(
            "WARNING: Rebuilt LABELS mapping from LABELS_ to enforce consistent class order."
        )



## === cell 5
test_dir = Path(IMGS_PATH)
if not test_dir.exists():
    raise FileNotFoundError(f"IMGS_PATH does not exist: {IMGS_PATH}")

test_images = sorted([p.name for p in test_dir.glob("*.jpg")])
if len(test_images) == 0:
    raise RuntimeError(f"No .jpg files found in IMGS_PATH: {IMGS_PATH}")

df_sub = pd.DataFrame({"image": test_images, "labels": "healthy"})
print("df_sub shape (from test_images):", df_sub.shape)
print(df_sub.head())




## === cell 6
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :, :]
    elif axis == 2:
        return img[:, ::-1, :]
    elif axis == 3:
        return img[::-1, ::-1, :]
    else:
        return img


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PlantDataset(data.Dataset):
    def __init__(self, df, size, labels, transform=None, tta=0):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.labels = labels
        self.transform = transform
        self.tta = tta

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = f"{IMGS_PATH}/{img_name}"
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))

        img = flip(img, axis=self.tta)
        img = img.astype(np.float32) / 255.0

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        img = img.transpose(2, 0, 1)

        if self.labels:
            label = np.zeros(len(self.labels)).astype(np.float32)
            for lbl in row.labels.split():
                if lbl in self.labels:
                    label[self.labels[lbl]] = 1
            return torch.tensor(img.copy()), torch.tensor(label)
        else:
            return torch.tensor(img.copy())


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        backbone = params["backbone"]

        if HAS_EFFNET_PYTORCH:
            self.enet = enet.EfficientNet.from_name(backbone)
            nc = self.enet._fc.in_features
            self.enet._fc = nn.Identity()
            self._extract = "effnet_pytorch"
        else:
            tv_map = {
                "efficientnet-b0": (
                    torchvision.models.efficientnet_b0,
                    torchvision.models.EfficientNet_B0_Weights.DEFAULT,
                ),
                "efficientnet-b1": (
                    torchvision.models.efficientnet_b1,
                    torchvision.models.EfficientNet_B1_Weights.DEFAULT,
                ),
                "efficientnet-b2": (
                    torchvision.models.efficientnet_b2,
                    torchvision.models.EfficientNet_B2_Weights.DEFAULT,
                ),
                "efficientnet-b3": (
                    torchvision.models.efficientnet_b3,
                    torchvision.models.EfficientNet_B3_Weights.DEFAULT,
                ),
                "efficientnet-b4": (
                    torchvision.models.efficientnet_b4,
                    torchvision.models.EfficientNet_B4_Weights.DEFAULT,
                ),
                "efficientnet-b5": (
                    torchvision.models.efficientnet_b5,
                    torchvision.models.EfficientNet_B5_Weights.DEFAULT,
                ),
                "efficientnet-b6": (
                    torchvision.models.efficientnet_b6,
                    torchvision.models.EfficientNet_B6_Weights.DEFAULT,
                ),
                "efficientnet-b7": (
                    torchvision.models.efficientnet_b7,
                    torchvision.models.EfficientNet_B7_Weights.DEFAULT,
                ),
            }
            if backbone not in tv_map:
                backbone = "efficientnet-b0"
            ctor, wts = tv_map[backbone]
            self.enet = ctor(weights=wts)
            nc = self.enet.classifier[-1].in_features
            self.enet.classifier = nn.Identity()
            self._extract = "torchvision"

        self.myfc = nn.Sequential(
            nn.Dropout(params["dropout"]),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        if self._extract == "torchvision":
            x = self.enet.features(x)
            x = self.enet.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x


class ResNext(nn.Module):
    def __init__(self, params, out_dim):
        super(ResNext, self).__init__()
        self.rsnxt = torchvision.models.resnext50_32x4d(
            weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT
        )
        nc = self.rsnxt.fc.in_features
        self.rsnxt.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nc, int(nc / 4)),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(nc / 4), out_dim),
        )
        self.rsnxt = nn.DataParallel(self.rsnxt)

    def forward(self, x):
        return self.rsnxt(x)




## === cell 7
def _maybe_strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    keys = list(state_dict.keys())
    if len(keys) == 0:
        return state_dict
    has_module = all(k.startswith("module.") for k in keys)
    if has_module:
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


models_list = []
loaded_any = False

for n_fold in FOLDS:
    if len(LABELS_) == 0:
        break

    if params["backbone"] == "resnext":
        model = ResNext(params=params, out_dim=len(LABELS_))
    else:
        model = EffNet(params=params, out_dim=len(LABELS_))

    path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
    if not Path(path).exists():
        print("WARNING: model weights not found:", path)
        continue

    state_dict = torch.load(path, map_location=torch.device("cpu"))

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e1:
        try:
            model.load_state_dict(_maybe_strip_module_prefix(state_dict), strict=True)
            print("loaded with stripped 'module.' prefix:", path)
        except Exception as e2:
            print("WARNING: failed to load weights:", path)
            print(" - raw load error:", repr(e1))
            print(" - stripped load error:", repr(e2))
            continue

    with torch.no_grad():
        dummy = torch.zeros(
            1, 3, params["img_size"], params["img_size"], dtype=torch.float32
        )
        out = model(dummy)
        if out.shape[-1] != len(LABELS_):
            raise RuntimeError(
                f"Model output dim ({out.shape[-1]}) != len(LABELS_) ({len(LABELS_)}). "
                "This would mis-decode labels and destroy F1; fix LABELS_/checkpoint mismatch."
            )

    model.float()
    model.eval()
    model.to(DEVICE)
    models_list.append(model)
    loaded_any = True
    print("loaded:", path)

del_state = locals().get("state_dict", None)
if del_state is not None:
    del state_dict
gc.collect()



## === cell 8
datasets, loaders = [], []
if loaded_any:
    for tta in TTAS:
        dataset = PlantDataset(
            df=df_sub,
            size=params["img_size"],
            labels=None,
            transform=None,
            tta=tta,
        )
        datasets.append(dataset)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=params["batch_size"],
            sampler=SequentialSampler(dataset),
            num_workers=WORKERS,
            pin_memory=(DEVICE.type == "cuda"),
        )
        loaders.append(loader)
else:
    print("No models loaded; will write all-healthy submission.")




## === cell 9
def get_labels(probs_row, labels, ths):
    idxs = []
    for i, x in enumerate(probs_row):
        th = ths.get(str(i), 0.5)
        if x > th:
            idxs.append(i)
    row_labels = [labels.get(str(i), None) for i in idxs]
    row_labels = [x for x in row_labels if x is not None]

    if len(row_labels) == 0:
        return "healthy"
    if "healthy" in row_labels and len(row_labels) > 1:
        row_labels = [x for x in row_labels if x != "healthy"]
        if len(row_labels) == 0:
            return "healthy"
    return " ".join(row_labels)


if loaded_any:
    all_model_tta = []
    with torch.no_grad():
        for mi, model in enumerate(models_list):
            for tj, loader in enumerate(loaders):
                preds_batches = []
                for img_data in loader:
                    img_data = img_data.to(DEVICE, non_blocking=True)
                    preds = model(img_data).sigmoid().detach().cpu().numpy()
                    preds_batches.append(preds)
                preds_full = np.vstack(preds_batches)
                all_model_tta.append(preds_full)
                print(f"model {mi} | tta {tj} -> done; shape={preds_full.shape}")

    logits = np.mean(np.stack(all_model_tta, axis=0), axis=0)
    df_sub["labels"] = [get_labels(x, LABELS, ths) for x in logits]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 10
print("value counts:")
print(df_sub["labels"].value_counts().head(20))
print(df_sub.head())



## === cell 11
df_sub = df_sub[["image", "labels"]]
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print("submission.csv preview:")
print(pd.read_csv("submission.csv").head())
