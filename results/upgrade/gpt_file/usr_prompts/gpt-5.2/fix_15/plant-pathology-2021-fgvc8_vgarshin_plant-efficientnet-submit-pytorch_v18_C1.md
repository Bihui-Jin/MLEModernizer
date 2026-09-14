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

0.7949030470914142

# 6. Current score

0.272

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.06777) has done: 'I remove the failing `pip install` and instead use the already-installed `torchvision`/`torch` stack, and patch the EfficientNet import so the code runs whether `efficientnet_pytorch` exists or not. I also fix the `KAGGLE`/path variables so downstream cells don’t crash, add safe CPU fallback if CUDA isn’t available, and make the model-weight loading robust (if the external model folder isn’t present, it still produce a valid submission). Finally, I fix the TTA/model ensembling aggregation bug that caused `np.vstack` to receive a scalar, ensuring logits are stacked consistently and labels are generated correctly for the submission CSV.'
- What this solution (achieved 0.24507) has done: 'Your score is extremely low mainly because the notebook is likely not loading the intended trained weights (it silently falls back to randomly initialized models), and because the label-index-to-name mapping inside `get_labels()` can be wrong if `LABELS` is `{name: index}` (your current code tries to index it as `{index: name}`). I make two minimal, directly score-relevant fixes: (1) robustly locate `MDLS_PATH` by searching the input directory for a matching `model_best_*.pth` folder so weights actually load, and (2) fix `get_labels()` to always use an explicit `idx->name` inverse map, preserving your exact thresholding and “healthy” fallback semantics. These changes preserve the same model architectures, inference loop, TTA, and submission formatting, but should move the score sharply upward toward your target.'
- What this solution (achieved 0.06777) has done: 'The current score gap is large, so the most likely score-killer is that your inference normalization does not match what the saved weights were trained with (you currently feed raw `[0,1]` RGB without ImageNet mean/std). I make a minimal, inference-only fix: apply torchvision ImageNet normalization inside the test dataset right before converting to tensor, preserving your exact model architectures, TTA, thresholding, and ensembling logic. I also make the test-image filtering use the sample submission’s order (no change in content) to avoid any accidental ordering mismatch, while still producing a valid `submission.csv`. These changes are directly score-relevant and should move performance upward toward your target without changing the core approach.'
- What this solution (achieved 0.23793) has done: 'Your current score is far below the target, which strongly suggests the inference pipeline is still not using the intended trained weights and/or is using thresholds that don’t match the loaded model. I make two minimal, score-relevant fixes: (1) make the model-folder auto-discovery prefer directories that contain both `model_best_*.pth` and `params.json` (and `ths.json` if available), so we reliably load the correct trained artifacts instead of some unrelated folder; and (2) ensure the class index order used at inference is consistent with `LABELS_` by rebuilding `IDX2NAME` from `LABELS_` when possible (this prevents subtle label-order mismatches that can destroy mean F1). These changes keep your exact models, TTA, thresholding logic, and submission formatting, but should move the score substantially upward toward your target.'
- What this solution (achieved 0.28071) has done: 'Your score is still far below the target, so we should assume inference-time preprocessing is mismatched with what the saved weights expect. The most common killer here is missing ImageNet normalization (already fixed) *and* missing resizing interpolation / center-crop style (but we won’t change that), plus accidentally using the wrong backbone/label order/thresholds when loading `params.json`. The smallest score-relevant change that preserves your core logic is: (1) ensure we always use the class order from the model artifacts when available (support both `labels_` as names or indices), (2) ensure thresholds are aligned to that same order by reindexing `ths` if it’s stored as a list, and (3) load weights with a more robust “strip/add module.” procedure while keeping strict loading when it matches. These changes don’t alter architecture, loops, TTA, or loss—just make the loaded artifacts actually match the inference mapping.'
- What this solution (achieved 0.25234) has done: 'Your current score is far below the target, so we should assume the main remaining score-killer is a mismatch between how the model outputs are converted into the required “space-delimited list” labels and how thresholds are applied, especially around the special `healthy` class. I keep the exact same models, weights loading, TTA loop, and sigmoid probabilities, but change the label decoding to (a) never output `healthy` together with other diseases and (b) treat `healthy` as a fallback only when no other class is predicted above threshold. I also make the submission use the full sample_submission order (no filtering by “existing files”), because Kaggle’s hidden test set expects exactly the sample_submission rows; filtering can silently produce an invalid/low-scoring submission due to missing rows. These are minimal inference/post-processing fixes that directly affect mean F1 without altering the core modeling approach.'
- What this solution (achieved 0.20128) has done: 'Your current score is far below the target, so the most likely remaining issue is that inference preprocessing still doesn’t match what the saved model weights expect. I make a minimal, inference-only change: use the exact torchvision ImageNet normalization *and* resize interpolation that the pretrained resnext backbone typically expects (bilinear with `INTER_LINEAR`), while keeping the same dataset structure, model(s), TTA loop, sigmoid, thresholds, and label decoding semantics. I also ensure the input tensor is contiguous `float32` to avoid subtle dtype/layout differences that can hurt throughput/consistency. These changes are directly score-relevant and should move mean F1 upward toward your target without changing the core modeling approach.'
- What this solution (achieved 0.12239) has done: 'Your current score is far below the target, so we should focus on the most likely inference-time mismatch that can destroy mean F1 without changing your model or training logic: the input normalization for ResNeXt. Right now you apply ImageNet mean/std but you do not scale by `1/255` for test inference inside the “labels=None” branch, so the model receives inputs in a very wrong range; we fix this by applying the exact same `/255.0` scaling before normalization for test. We also ensure the same normalization is applied in the training-label branch (for consistency and to avoid future accidental reuse issues), but this doesn’t change your architecture, thresholds, TTA, or ensembling logic. These are minimal, directly score-relevant changes and should move the score substantially upward toward your target band.'
- What this solution (achieved 0.14854) has done: 'Your current score (0.12239) is far below the target (0.7949), so we should focus on a likely remaining inference-time mismatch that can heavily suppress mean F1 without changing your model/loops: class-order and threshold alignment with the saved model artifacts. I minimally (1) rebuild the inference class order from `params.json` more robustly (supporting both `labels_` being names or indices, and sorting indices when `labels` dict is available), and (2) reindex/realign `ths` to the final class order when possible so thresholds apply to the correct classes. This keeps the same architecture, TTA, sigmoid averaging, and label decoding semantics, but should substantially increase score by fixing label/threshold misalignment. The submission remains in the exact `image,labels` format and row order from `sample_submission.csv`.'
- What this solution (achieved 0.272) has done: 'Your current gap to the target is large, so we should focus on the highest-impact inference mismatch without changing your model or loops: the prediction-to-label decoding and thresholds likely don’t match the competition’s “healthy vs diseases” semantics. I keep your exact models, TTA/ensembling, sigmoid probabilities, and submission format, but (1) ensure `healthy` is never predicted via thresholding (only as a fallback when no disease is selected) and (2) tune a single global decision threshold on a small internal validation split made from `train.csv` using the same inference pipeline (no training, just calibration). This is minimal, runs fast, and usually moves mean-F1 dramatically upward because it directly aligns post-processing with the metric. The final submission still uses the sample submission’s row order and writes `submission.csv`.'
- What this solution (achieved 0.272) has done: 'Your score is still far below the target, so we should focus on the smallest inference-only fixes that commonly destroy mean F1 without changing your model or loops. I (1) ensure the loaded `ths.json` is actually used during decoding (your code currently calibrates and then ignores per-class thresholds), (2) calibrate a single scalar multiplier applied to the loaded per-class thresholds (so we keep the same thresholding “shape” but align it to the metric), and (3) compute calibration F1 using the competition-style sample-wise F1 (not micro-F1), which better matches “Mean F1-Score”. These are minimal post-processing changes only: same models, same TTA, same sigmoid probabilities, same submission format and ordering.'
- What this solution (achieved 0.272) has done: 'I make two minimal, score-relevant fixes that keep your exact model(s), TTA loop, sigmoid probabilities, and thresholding/decoding semantics intact. First, I ensure the inference pipeline uses the exact same class order as the loaded weights by reading a `classes.json`/`class_names.json`/`idx2label.json`-style artifact if present in the model folder (many PP2021 solutions save one), otherwise falling back to your current logic; this prevents silent class-permutation that can cap mean-F1 around your current level. Second, I calibrate the threshold-scale using the *competition-style decoded labels* (healthy fallback, never output healthy with diseases) rather than raw multi-hot over all classes, so the chosen scale actually optimizes the same mean-F1 the leaderboard uses (still no training, just post-processing calibration). These are small post-processing/alignment changes only and should move your score upward toward the target without altering the core approach.'
- What this solution (achieved 0.272) has done: 'Your current score is far below the target, so the smallest high-impact change is to ensure the inference model exactly matches the training-time backbone that the saved weights expect; right now `params["backbone"]` defaults to `"resnext"` and may silently build the wrong architecture even if `params.json` indicates an EfficientNet. I minimally (1) add a robust backbone resolver that maps common aliases (e.g., `resnext50_32x4d`, `tf_efficientnet_b4_ns`) into the constructors you already have, and (2) add an auto-fallback: if strict weight loading fails for the chosen backbone, rebuild the other backbone type once and retry loading so we don’t run random/wrong-head weights. This preserves your same models, logits->sigmoid, TTA, thresholding, calibration, and submission formatting, but greatly increases the chance you’re actually using the intended trained weights, moving the score upward toward your target. The submission path/name and CSV schema remain unchanged.'
- What this solution (achieved 0.272) has done: 'The biggest remaining gap to your target score is consistent with a still-mismatched class order between (a) the model’s output logits and (b) your `IDX2NAME`/threshold vector; if those are permuted, mean F1 stay very low no matter how you calibrate thresholds. I make a minimal, inference-only fix: derive the class order directly from the model’s own classifier head output dimension and the loaded artifacts, and then *align* `IDX2NAME`, `name2idx`, and `thr_base` to a single consistent index space (no architecture or loop changes). I also make threshold decoding deterministic by iterating indices in sorted order and ensure `complex`/disease labels are never dropped due to missing keys. These changes keep your same models, TTA, sigmoid averaging, calibration approach, and submission format, but should move the score upward toward the target.'

# 9. Code solution

## === cell 0
import os
import gc
import sys
import json
import time
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torch.utils.data.sampler import SequentialSampler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KAGGLE = True

try:
    from efficientnet_pytorch import model as enet  # type: ignore
except Exception:
    from torchvision.models import (
        efficientnet_b0,
        efficientnet_b1,
        efficientnet_b2,
        efficientnet_b3,
        efficientnet_b4,
        efficientnet_b5,
        efficientnet_b6,
        efficientnet_b7,
    )

    class _TorchvisionEfficientNetWrapper(nn.Module):
        def __init__(self, name: str):
            super().__init__()
            name = name.lower()
            mapping = {
                "efficientnet-b0": efficientnet_b0,
                "efficientnet-b1": efficientnet_b1,
                "efficientnet-b2": efficientnet_b2,
                "efficientnet-b3": efficientnet_b3,
                "efficientnet-b4": efficientnet_b4,
                "efficientnet-b5": efficientnet_b5,
                "efficientnet-b6": efficientnet_b6,
                "efficientnet-b7": efficientnet_b7,
            }
            if name not in mapping:
                raise ValueError(
                    f"Unsupported EfficientNet backbone without efficientnet_pytorch: {name}"
                )
            self.net = mapping[name](weights=None)
            in_features = self.net.classifier[-1].in_features
            self._fc = nn.Linear(in_features, 1000)
            self.net.classifier = nn.Identity()

        def forward(self, x):
            return self.net(x)

    class enet:  # noqa: N801
        class EfficientNet:
            @staticmethod
            def from_name(name: str):
                return _TorchvisionEfficientNetWrapper(name)




## === cell 1
TEST = True
VER = "v102"

if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TTAS = [0, 1, 2]
FOLDS = [0]
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()




## === cell 2
default_labels = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]
default_LABELS_ = list(range(len(default_labels)))
default_LABELS = {lbl: i for i, lbl in enumerate(default_labels)}

params = {
    "labels_": default_LABELS_,
    "labels": default_LABELS,
    "workers": 2,
    "img_size": 512,
    "batch_size": 16,
    "dropout": 0.2,
    "backbone": "resnext",  # will work without extra packages
}
ths = {str(i): 0.5 for i in range(len(default_LABELS_))}


def _find_models_path(initial_path: str, folds):
    def folder_score(root: str, files: set):
        hits = 0
        for f in folds:
            if f"model_best_{f}.pth" in files:
                hits += 1
        if hits == 0:
            return -1

        score = hits * 10
        if "params.json" in files:
            score += 3
        if "ths.json" in files:
            score += 2
        return score

    if os.path.isdir(initial_path):
        files = set(os.listdir(initial_path))
        if folder_score(initial_path, files) > 0:
            return initial_path

    if not KAGGLE:
        return initial_path

    base = "../input"
    if not os.path.isdir(base):
        return initial_path

    best_path = None
    best_score = -1
    for root, dirs, files in os.walk(base):
        files_set = set(files)
        sc = folder_score(root, files_set)
        if sc > best_score:
            best_score = sc
            best_path = root
            if sc >= (len(folds) * 10 + 5):  # all folds + params + ths
                break

    return best_path if (best_path is not None and best_score > 0) else initial_path


MDLS_PATH = _find_models_path(MDLS_PATH, FOLDS)

if os.path.exists(f"{MDLS_PATH}/params.json"):
    with open(f"{MDLS_PATH}/params.json") as file:
        params.update(json.load(file))

LABELS_raw = params.get("labels", default_LABELS)
LABELS__raw = params.get("labels_", default_LABELS_)


def _normalize_labels_and_order(labels_dict, labels_):
    if (
        isinstance(labels_, (list, tuple))
        and len(labels_) > 0
        and all(isinstance(x, str) for x in labels_)
    ):
        ordered_names = [str(x) for x in labels_]
        labels_map = {n: i for i, n in enumerate(ordered_names)}
        return ordered_names, labels_map

    if (
        isinstance(labels_dict, dict)
        and len(labels_dict) > 0
        and all(isinstance(v, int) for v in labels_dict.values())
    ):
        inv = {int(v): str(k) for k, v in labels_dict.items()}
        ordered_idx = sorted(inv.keys())
        ordered_names = [inv[i] for i in ordered_idx]
        labels_map = {n: j for j, n in enumerate(ordered_names)}
        return ordered_names, labels_map

    if isinstance(labels_, (list, tuple)) and len(labels_) > 0:
        try:
            ordered = list(labels_)
        except Exception:
            ordered = list(range(len(default_LABELS_)))
    else:
        ordered = list(range(len(default_LABELS_)))
    return ordered, labels_dict if isinstance(labels_dict, dict) else default_LABELS


LABELS_, LABELS = _normalize_labels_and_order(LABELS_raw, LABELS__raw)

WORKERS = 2 if KAGGLE else int(params.get("workers", 2))

if os.path.exists(f"{MDLS_PATH}/ths.json"):
    with open(f"{MDLS_PATH}/ths.json") as file:
        ths = json.load(file)


def _normalize_ths(ths_obj, idx2name):
    if isinstance(ths_obj, (list, tuple, np.ndarray)):
        base = {str(i): float(v) for i, v in enumerate(list(ths_obj))}
    else:
        base = {str(k): float(v) for k, v in dict(ths_obj).items()}

    name2idx = {v: k for k, v in idx2name.items()}
    if any((isinstance(k, str) and (k in name2idx)) for k in base.keys()):
        remap = {}
        for k, v in base.items():
            if k in name2idx:
                remap[str(name2idx[k])] = float(v)
        if len(remap) >= max(1, len(idx2name) // 2):
            return remap
    return base


def _try_load_idx2name_from_artifacts(mdls_path: str):
    candidates = [
        "idx2name.json",
        "idx2label.json",
        "class_names.json",
        "classes.json",
        "labels.json",
    ]
    for fn in candidates:
        p = os.path.join(mdls_path, fn)
        if not os.path.exists(p):
            continue
        try:
            with open(p, "r") as f:
                obj = json.load(f)
            if isinstance(obj, dict) and len(obj) > 0:
                if all(str(k).isdigit() for k in obj.keys()):
                    idx2name = {int(k): str(v) for k, v in obj.items()}
                    if len(idx2name) >= 2:
                        return idx2name
                if all(isinstance(v, int) for v in obj.values()):
                    idx2name = {int(v): str(k) for k, v in obj.items()}
                    if len(idx2name) >= 2:
                        return idx2name
            if (
                isinstance(obj, list)
                and len(obj) > 0
                and all(isinstance(x, str) for x in obj)
            ):
                return {i: str(n) for i, n in enumerate(obj)}
        except Exception:
            pass
    return None


print("DEVICE:", DEVICE)
print("DATA_PATH:", DATA_PATH)
print("IMGS_PATH:", IMGS_PATH)
print("MDLS_PATH:", MDLS_PATH)
print("loaded params keys:", list(params.keys()))
print("num labels (params-derived):", len(LABELS_))




## === cell 3
sample_path = f"{DATA_PATH}/sample_submission.csv"
df_sub = pd.read_csv(sample_path)

df_sub["labels"] = "healthy"
print(df_sub.head(), "rows:", len(df_sub))




## === cell 4
def flip(img, axis=0):
    if axis == 1:
        return img[
            ::-1,
            :,
        ]
    elif axis == 2:
        return img[
            :,
            ::-1,
        ]
    elif axis == 3:
        return img[
            ::-1,
            ::-1,
        ]
    else:
        return img


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PlantDataset(data.Dataset):
    def __init__(
        self, df, size, labels, transform=None, tta=0, normalize=True, imgs_path=None
    ):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.labels = labels
        self.transform = transform
        self.tta = tta
        self.normalize = bool(normalize)
        self.imgs_path = imgs_path if imgs_path is not None else IMGS_PATH

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = f"{self.imgs_path}/{img_name}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"no img file read: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)

        img = img.astype(np.float32) / 255.0

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        if self.labels:
            if self.normalize:
                img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
            img = img.transpose(2, 0, 1)
            label = np.zeros(len(self.labels)).astype(np.float32)
            for lbl in row.labels.split():
                label[self.labels[lbl]] = 1
            return torch.tensor(img), torch.tensor(label)
        else:
            img = flip(img, axis=self.tta)
            if self.normalize:
                img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
            img = img.transpose(2, 0, 1)

            img = np.ascontiguousarray(img, dtype=np.float32)
            return torch.from_numpy(img)


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        self.enet = enet.EfficientNet.from_name(params["backbone"])
        nc = self.enet._fc.in_features
        self.enet._fc = nn.Identity()
        self.myfc = nn.Sequential(
            nn.Dropout(params["dropout"]),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x


class ResNext(nn.Module):
    def __init__(self, params, out_dim):
        super(ResNext, self).__init__()
        self.rsnxt = torchvision.models.resnext50_32x4d(weights=None)
        nc = self.rsnxt.fc.in_features
        self.rsnxt.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nc, int(nc / 4)),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(nc / 4), out_dim),
        )
        if torch.cuda.is_available():
            self.rsnxt = nn.DataParallel(self.rsnxt)

    def forward(self, x):
        return self.rsnxt(x)




## === cell 5
models = []
weights_missing = False


def _load_state_dict_robust(model: nn.Module, state_dict: dict):
    try:
        model.load_state_dict(state_dict, strict=True)
        return True
    except RuntimeError:
        pass

    stripped = {}
    if any(k.startswith("module.") for k in state_dict.keys()):
        for k, v in state_dict.items():
            stripped[k.replace("module.", "", 1)] = v
        try:
            model.load_state_dict(stripped, strict=True)
            return True
        except RuntimeError:
            pass

    added = {}
    if not any(k.startswith("module.") for k in state_dict.keys()):
        for k, v in state_dict.items():
            added["module." + k] = v
        try:
            model.load_state_dict(added, strict=True)
            return True
        except RuntimeError:
            pass

    model.load_state_dict(stripped if len(stripped) else state_dict, strict=False)
    return False


def _resolve_backbone_name(raw: str) -> str:
    s = str(raw).lower().strip()
    if s in ["resnext", "resnext50", "resnext50_32x4d", "resnext50-32x4d"]:
        return "resnext"
    if "efficientnet" in s:
        for b in ["b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7"]:
            if f"b{b[1:]}" in s or f"_{b}_" in s or s.endswith(b):
                return f"efficientnet-{b}"
        return s
    return s


def _build_model_from_params(params_local, out_dim: int) -> nn.Module:
    bb = _resolve_backbone_name(params_local.get("backbone", "resnext"))
    if bb == "resnext":
        return ResNext(params=params_local, out_dim=out_dim)
    p2 = dict(params_local)
    p2["backbone"] = bb
    return EffNet(params=p2, out_dim=out_dim)


for n_fold in FOLDS:
    model = _build_model_from_params(params, out_dim=len(LABELS_))

    path = "{}/model_best_{}.pth".format(MDLS_PATH, n_fold)
    strict_loaded = False
    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu")

        strict_loaded = _load_state_dict_robust(model, state_dict)

        if not strict_loaded:
            bb = _resolve_backbone_name(params.get("backbone", "resnext"))
            alt_params = dict(params)
            alt_params["backbone"] = "efficientnet-b4" if bb == "resnext" else "resnext"
            alt_model = _build_model_from_params(alt_params, out_dim=len(LABELS_))
            alt_strict = _load_state_dict_robust(alt_model, state_dict)
            if alt_strict:
                print(
                    f"loaded with alternate backbone after strict-fail: {bb} -> {alt_params['backbone']}"
                )
                model = alt_model
                params["backbone"] = alt_params["backbone"]
                strict_loaded = True

        print("loaded:", path, "| strict:", strict_loaded)
        del state_dict
    else:
        weights_missing = True
        print("WARNING: model weights not found:", path)

    model.float()
    model.eval()
    model.to(DEVICE)
    models.append(model)

gc.collect()




## === cell 6
artifact_idx2name = _try_load_idx2name_from_artifacts(MDLS_PATH)

if artifact_idx2name is not None:
    IDX2NAME = dict(sorted(artifact_idx2name.items(), key=lambda x: x[0]))
else:
    if (
        isinstance(LABELS_, (list, tuple))
        and len(LABELS_) > 0
        and all(isinstance(x, str) for x in LABELS_)
    ):
        IDX2NAME = {i: str(name) for i, name in enumerate(LABELS_)}
    else:
        if (
            isinstance(LABELS, dict)
            and len(LABELS) > 0
            and all(isinstance(v, int) for v in LABELS.values())
        ):
            IDX2NAME = {int(v): str(k) for k, v in LABELS.items()}
        else:
            IDX2NAME = {i: str(i) for i in range(len(LABELS_))}

_sorted_items = sorted(IDX2NAME.items(), key=lambda kv: int(kv[0]))
_dense_idx2name = {
    new_i: str(name) for new_i, (_old_i, name) in enumerate(_sorted_items)
}
IDX2NAME = _dense_idx2name

ths = _normalize_ths(ths, IDX2NAME)
print("thresholds loaded/aligned:", len(ths))
print(
    "first idx2name:",
    {k: IDX2NAME[k] for k in sorted(IDX2NAME)[: min(6, len(IDX2NAME))]},
)

HEALTHY_IDX = None
for i, n in IDX2NAME.items():
    if n == "healthy":
        HEALTHY_IDX = int(i)
        break
print("HEALTHY_IDX:", HEALTHY_IDX)
print("final backbone used:", params.get("backbone"))

with torch.no_grad():
    dummy = torch.zeros(
        (1, 3, int(params["img_size"]), int(params["img_size"])), device=DEVICE
    )
    out = models[0](dummy)
    model_C = int(out.shape[1])
del dummy, out
gc.collect()

if model_C != len(IDX2NAME):
    print(
        f"WARNING: model outputs C={model_C} but IDX2NAME has {len(IDX2NAME)}; remapping to safe default."
    )
    if model_C == len(default_labels):
        IDX2NAME = {i: default_labels[i] for i in range(model_C)}
    else:
        names = [IDX2NAME.get(i, str(i)) for i in range(len(IDX2NAME))]
        if len(names) < model_C:
            names = names + [f"class_{i}" for i in range(len(names), model_C)]
        else:
            names = names[:model_C]
        IDX2NAME = {i: str(names[i]) for i in range(model_C)}
    HEALTHY_IDX = None
    for i, n in IDX2NAME.items():
        if n == "healthy":
            HEALTHY_IDX = int(i)
            break
    ths = _normalize_ths(ths, IDX2NAME)

print("final C:", len(IDX2NAME), "| HEALTHY_IDX:", HEALTHY_IDX)




## === cell 7
def predict_proba_for_df(df_images, imgs_path):
    datasets, loaders = [], []
    for tta in TTAS:
        dataset = PlantDataset(
            df=df_images,
            size=params["img_size"],
            labels=None,
            transform=None,
            tta=tta,
            normalize=True,
            imgs_path=imgs_path,
        )
        datasets.append(dataset)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(params["batch_size"]),
            sampler=SequentialSampler(dataset),
            num_workers=int(WORKERS),
            pin_memory=torch.cuda.is_available(),
        )
        loaders.append(loader)

    all_preds_local = []
    with torch.no_grad():
        for i, model in enumerate(models):
            tta_preds = []
            for j, loader in enumerate(loaders):
                batch_preds = []
                for img_data in loader:
                    img_data = img_data.to(DEVICE, non_blocking=True)
                    preds = model(img_data).sigmoid().detach().cpu().numpy()  # (bs, C)
                    batch_preds.append(preds)
                tta_pred = np.vstack(batch_preds)  # (N, C)
                tta_preds.append(tta_pred)
            tta_preds = np.mean(np.stack(tta_preds, axis=0), axis=0)  # (N, C)
            all_preds_local.append(tta_preds)

    probs = np.mean(np.stack(all_preds_local, axis=0), axis=0)  # (N, C)
    return probs




## === cell 8
def f1_per_sample_mean(y_true, y_pred):
    y_true = y_true.astype(np.int8)
    y_pred = y_pred.astype(np.int8)
    tp = (y_true & y_pred).sum(axis=1)
    fp = ((1 - y_true) & y_pred).sum(axis=1)
    fn = (y_true & (1 - y_pred)).sum(axis=1)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom == 0, 1.0, (2.0 * tp) / denom)  # if both empty -> F1=1
    return float(f1.mean())


def decode_with_scaled_class_thresholds(probs_row, thr_vec):
    pred_names = []
    for i in range(len(probs_row)):
        p = float(probs_row[i])
        if HEALTHY_IDX is not None and i == HEALTHY_IDX:
            continue
        if p > float(thr_vec[i]):
            n = IDX2NAME.get(i, None)
            if n is not None and n != "healthy":
                pred_names.append(n)
    if len(pred_names) == 0:
        return "healthy"
    return " ".join(pred_names)


C = len(IDX2NAME)
thr_base = np.full((C,), 0.5, dtype=np.float32)
for i in range(C):
    if str(i) in ths:
        thr_base[i] = float(ths[str(i)])

if HEALTHY_IDX is not None and 0 <= HEALTHY_IDX < C:
    thr_base[HEALTHY_IDX] = 1.0

train_csv_path = f"{DATA_PATH}/train.csv"
df_train = pd.read_csv(train_csv_path)

name2idx = {v: k for k, v in IDX2NAME.items()}


def make_y(df):
    y = np.zeros((len(df), C), dtype=np.int8)
    for r, labels_str in enumerate(df["labels"].values):
        labs = str(labels_str).split()
        for lab in labs:
            if lab in name2idx:
                y[r, int(name2idx[lab])] = 1
    return y


def _decoded_to_multihot(decoded_labels_list, C, name2idx):
    y = np.zeros((len(decoded_labels_list), C), dtype=np.int8)
    for i, s in enumerate(decoded_labels_list):
        labs = str(s).split()
        for lab in labs:
            if lab in name2idx:
                y[i, int(name2idx[lab])] = 1
    return y


rng = np.random.RandomState(123)
idx = np.arange(len(df_train))
rng.shuffle(idx)
cal_n = min(2048, len(df_train))
cal_idx = idx[:cal_n]
df_cal = df_train.iloc[cal_idx].reset_index(drop=True)

cal_imgs_path = f"{DATA_PATH}/train_images"
cal_probs = predict_proba_for_df(df_cal[["image"]], imgs_path=cal_imgs_path)
y_true = make_y(df_cal)

scales = np.linspace(0.70, 1.30, 13)
best_scale = 1.0
best_f1 = -1.0
for s in scales:
    thr_vec = np.clip(thr_base * float(s), 0.01, 0.99)
    if HEALTHY_IDX is not None and 0 <= HEALTHY_IDX < C:
        thr_vec[HEALTHY_IDX] = 1.0

    decoded = [decode_with_scaled_class_thresholds(row, thr_vec) for row in cal_probs]
    y_pred = _decoded_to_multihot(decoded, C, name2idx)
    f1 = f1_per_sample_mean(y_true, y_pred)

    if f1 > best_f1:
        best_f1 = f1
        best_scale = float(s)

print(
    f"calibration: best threshold scale={best_scale:.3f} | mean sample-F1(decoded semantics)={best_f1:.4f}"
)




## === cell 9
test_imgs_path = f"{DATA_PATH}/test_images"
test_probs = predict_proba_for_df(df_sub[["image"]], imgs_path=test_imgs_path)

thr_final = np.clip(thr_base * best_scale, 0.01, 0.99).astype(np.float32)
if HEALTHY_IDX is not None and 0 <= HEALTHY_IDX < C:
    thr_final[HEALTHY_IDX] = 1.0

df_sub["labels"] = [
    decode_with_scaled_class_thresholds(x, thr_final) for x in test_probs
]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")




## === cell 10
print("value counts:")
print(df_sub.labels.value_counts().head(20))
print(df_sub.head())




## === cell 11
out_path = "submission.csv"
df_sub[["image", "labels"]].to_csv(out_path, index=False)
print("wrote:", out_path, "rows:", len(df_sub))

if weights_missing:
    print(
        "WARNING: One or more model weight files were missing; submission is valid but likely low-scoring."
    )
else:
    print("All requested fold weights found and loaded.")
