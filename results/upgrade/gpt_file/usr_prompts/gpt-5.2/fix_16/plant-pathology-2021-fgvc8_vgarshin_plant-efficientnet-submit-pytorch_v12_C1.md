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

0.8185964912280713

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I remove the broken dependency on `efficientnet_pytorch` (it isn’t available in your environment) by switching to `torchvision`’s EfficientNet while keeping the same overall inference pipeline (load params → build model → load weights → multi-TTA sigmoid → threshold → space-delimited labels). I also make the code robust to missing external model/params by falling back to `sample_submission.csv` with all `healthy` labels so it always produces a valid `submission.csv`. Finally, I fix the TTA aggregation bug that caused `np.vstack` to receive scalars by collecting per-TTA predictions correctly and averaging across TTAs and folds.'
- What this solution (achieved 0.24507) has done: 'Your current score suggests the notebook is effectively submitting mostly/only `healthy` because inference is being disabled when `params.json`/weights aren’t found at `../input/plant-models-v5`. I make a minimal, competition-legal fix to actually load the provided trained artifacts by auto-discovering the correct models directory under `../input` (without changing the model, TTA, thresholding, or overall pipeline). I also add a small compatibility fix for torchvision EfficientNet feature extraction (ensuring the backbone outputs pooled features, not logits), which otherwise can silently degrade predictions even when weights load. These changes should move the score upward toward your target by enabling real inference with the intended checkpoint(s).'
- What this solution (achieved 0.24507) has done: 'Your current score is far below the target, so we should make the smallest changes that restore the intended inference behavior and match the training-time preprocessing, without changing the model architecture or thresholding logic. The biggest likely issue is that test-time images are not being normalized the same way as in training (your `transform=None` path only scales to [0,1]), which can collapse predictions toward “healthy” and tank mean F1. I add a minimal ImageNet normalization in the inference dataset (no new augmentations, same TTA flips), and I make checkpoint loading robust to common saved formats (`state_dict`, `model`, `net`) while keeping `strict=True` whenever possible. These changes should materially increase the score toward your target while preserving the core pipeline (EfficientNet backbone → myfc head → sigmoid → mean over TTA/folds → threshold → space-delimited labels).'
- What this solution (achieved 0.24507) has done: 'Your score gap is large (0.245 → 0.819), so the smallest high-impact fixes are to ensure the model actually runs with the same preprocessing it was trained with and that label post-processing matches multi-label F1 expectations. I keep the same model, checkpoints, TTA averaging, sigmoid, and single global threshold, but (1) apply flips before normalization (so TTA behaves like typical training/inference), (2) prevent the `"healthy"` class from suppressing other predicted diseases (a common cause of very low mean-F1), and (3) make label mapping robust when `LABELS` is either `{"0":"healthy"...}` or `{"healthy":0...}`. These changes are directly tied to improving mean F1 while preserving your pipeline and still producing a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'Your current score is far below the target, so the smallest high-impact change is to fix the label decoding: your model outputs are in `labels_` order, but the code decodes indices using `labels`, which can scramble classes and collapse mean-F1. I switch the `idx→label` mapping to come from `LABELS_` (the ordered list), and keep `LABELS` only for training-time one-hot encoding. I also apply the common PP2021 rule that `healthy` should not co-occur with other labels (remove `healthy` if any disease is predicted), which typically boosts F1 without changing the model. Everything else (EfficientNet backbone, sigmoid, TTA averaging, single threshold, CSV format/path) remains the same.'
- What this solution (achieved 0.24507) has done: 'Your current score is far below target, so the most likely minimal high-impact issue is mismatched inference preprocessing vs. what the model was trained on (normalization and resize interpolation), plus a fragile label index→name mapping when `labels_` is stored as a dict. I keep the same model, checkpoints, sigmoid, TTA averaging, and single threshold, but (1) apply ImageNet normalization in the same order as typical training (flip on uint8/float image before normalization, then normalize), (2) use `INTER_AREA` downsampling for more stable resizing, and (3) make `IDX2LBL` robust for both list and dict `labels_` by sorting numeric keys so class order can’t scramble predictions. These are small changes that should move mean F1 upward toward your target without changing the core inference pipeline or producing invalid submissions. The script still always writes a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'Your gap to target is large (0.245 → 0.819, higher-is-better), so we need a small but high-impact fix that preserves the same model/TTA/threshold pipeline. The most likely remaining failure mode is that TTA flips are currently not applied for inference because the code only flips when `self.labels` is falsy, but `self.labels` is a dict during training and `None` during test—this part is fine; the real issue is the flip function’s axis mapping is inconsistent with typical horizontal/vertical flips (it’s currently inverted), which can meaningfully hurt TTA averaging. I minimally fix `flip()` so TTA=1/2/3 correspond to vertical/horizontal/both in the standard way, keeping everything else identical. This should improve test predictions without changing architecture, loss, thresholding, or training logic, and it still write a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'Your current score (0.245) is far below the target (0.819), so the smallest high-impact fixes are to (1) ensure the label index→name mapping is always correct (even when `labels_` is stored as a dict mapping label→index), and (2) make the submission rows align exactly to the competition’s `sample_submission.csv` order (to avoid silent row-order mismatches that can crush F1). I keep the same model, checkpoints, sigmoid, TTA averaging, and single threshold, but I robustly build `IDX2LBL` from either `labels_` or `labels` and then reindex predictions to the sample submission’s `image` ordering before writing. These changes preserve the core inference pipeline while directly addressing the most common causes of “mostly wrong labels” and “misaligned predictions” low scores. The script still always write a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'Your current score is far below the target, so the most likely minimal high-impact fix is that the inference image preprocessing still doesn’t match what the trained checkpoint expects (common cause: missing/incorrect center crop + normalization order), which collapses probabilities and yields near-random/healthy-heavy outputs. I keep the same model, weights, sigmoid, TTA averaging, and thresholding, but I (1) add a standard EfficientNet/ImageNet-style resize+center-crop pipeline before normalization (while keeping your existing flips as TTA), and (2) ensure the submission row order exactly matches `sample_submission.csv` to avoid any silent misalignment. These are small, inference-only changes that typically move mean F1 sharply upward when the model was trained with that preprocessing. The script still runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'We need to move the score up (0.245 → 0.819), so the smallest high-impact fix is to make inference preprocessing match what the trained EfficientNet checkpoint most likely expects: ImageNet normalization plus the same resize/crop geometry and interpolation. Your pipeline already adds resize+center-crop and normalization, but it always uses a fixed 0.875 crop factor and applies flips after cropping; many PP2021 EfficientNet solutions used `Resize(512)->CenterCrop(448)`-style behavior (i.e., resize to `img_size` then center crop) or simply `cv2.resize` without the 0.875 heuristic, and a mismatch can collapse probabilities toward “healthy”. I keep the exact model, sigmoid, TTA averaging, and thresholding, but (1) switch to a simpler, more standard inference resize to `img_size` (no 0.875 heuristic) to better match common training pipelines, and (2) ensure TTA flips happen on the resized image before normalization (still as you do), while keeping everything else unchanged. This is a minimal, inference-only change that should materially increase mean F1 if the checkpoint was trained with plain resizing rather than the 0.875 crop scheme.'
- What this solution (achieved 0.24507) has done: 'Your current score is far below the target, so we need a small, high-impact fix that keeps your exact model/TTA/threshold pipeline but restores the *intended* input distribution for the EfficientNet checkpoint. The most likely remaining issue is preprocessing mismatch: your inference currently does only resize+ImageNet normalize, but many PP2021 EfficientNet checkpoints were trained with a **center-crop after resize** (or resize-shorter-side then center-crop), and a geometry mismatch can collapse predictions toward “healthy”. I add a minimal, deterministic **resize → center-crop** step (before TTA flips and normalization) while keeping everything else identical, and I keep submission row order aligned to `sample_submission.csv`. This should move mean F1 upward toward your target without changing architecture, loss, training loops, TTA set, or threshold semantics.'
- What this solution (achieved 0.24507) has done: 'We need to move the score up (0.245 → 0.819), so the smallest high-impact change is to fix a likely preprocessing mismatch: your inference currently uses resize-short-side + center-crop, which often does not match how PP2021 EfficientNet checkpoints were trained (many used simple square resize). I keep the same model, weights loading, sigmoid, TTA set, averaging, and thresholding, but switch test preprocessing to a deterministic square resize to `img_size` (using stable interpolation) to better match typical training and avoid collapsing predictions toward “healthy”. I also make sure the class head input feature size always matches the exact EfficientNet variant by reading it from the backbone (avoids silent shape mismatches that can degrade outputs if a different backbone than b0 is used). Everything else remains unchanged, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'Your score is far below the target, so the most likely minimal high-impact fix is that inference is still running with a mismatched input distribution for the trained checkpoint. I keep the exact same model/TTA/threshold pipeline, but (1) switch test-time normalization to use the checkpoint’s own `params["mean"]`/`params["std"]` when present (falling back to ImageNet), and (2) force deterministic, training-like resize interpolation by reading `params["interpolation"]`/`params["resize_interp"]` if available (falling back to the current behavior). These are inference-only, low-risk changes that often recover a large chunk of mean-F1 when a pretrained model is otherwise correct. The script still run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'We need to move the score up (0.245 → 0.819), so the smallest high-impact fix is to make sure the predicted label strings exactly match the competition’s expected class names and formatting. Your current pipeline can silently output wrong/unknown label tokens (e.g., due to case differences, underscores vs spaces, or “scab ”-like artifacts), which the evaluator treats as incorrect and crushes mean F1 even if probabilities are good. I add a minimal label canonicalization step using the label vocabulary found in `train.csv`, and ensure `get_labels()` only emits valid known labels (otherwise falling back to `healthy`). This preserves the exact model, weights, TTA, sigmoid, averaging, and threshold semantics while making the submission interpretable by the metric.'
- What this solution (achieved 0.24507) has done: 'Your current gap to target is large (0.245 → 0.819, higher-is-better), so we need a minimal, high-impact fix without changing the model/TTA/threshold semantics. The biggest likely remaining issue is that inference currently uses `labels=None`, but `PlantDataset.__getitem__` checks `if not self.labels:` to apply flips; when labels is a non-empty dict/list this logic breaks, and it also makes the “train vs test” path ambiguous. I make this explicit by adding a `is_test` flag and using it to control TTA flips and return type, keeping everything else (resize, normalization, EfficientNet, sigmoid, averaging, threshold) unchanged. I also ensure inference always uses `params["labels"]`/`labels_` only for mapping (not to control branching), preventing accidental suppression of TTA or wrong tensor formatting that can collapse predictions toward “healthy”.'

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
from torchvision import transforms
from torch.utils.data.sampler import SequentialSampler

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "torch:",
    torch.__version__,
    "torchvision:",
    torchvision.__version__,
    "device:",
    DEVICE,
)



## === cell 1
try:
    import efficientnet_pytorch  # noqa: F401

    print("efficientnet_pytorch is available.")
except Exception as e:
    print(
        "efficientnet_pytorch not available; will use torchvision EfficientNet instead.",
        repr(e),
    )



## === cell 2
TEST = True
VER = "v5"

if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TH = 0.4
TTAS = [0, 1, 2, 3]
FOLDS = [0]
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()

print("DATA_PATH:", DATA_PATH)
print("IMGS_PATH:", IMGS_PATH)
print("MDLS_PATH (initial):", MDLS_PATH)




## === cell 3
def _find_models_dir(preferred_path: str, ver: str) -> str:
    if os.path.isdir(preferred_path):
        return preferred_path

    base = "../input"
    candidates = []
    if os.path.isdir(base):
        for name in os.listdir(base):
            p = os.path.join(base, name)
            if not os.path.isdir(p):
                continue
            lname = name.lower()
            if ("plant" in lname and "model" in lname) or (ver.lower() in lname):
                candidates.append(p)

    ranked = []
    for p in candidates:
        has_params = os.path.exists(os.path.join(p, "params.json"))
        try:
            has_pth = any(fn.endswith(".pth") for fn in os.listdir(p))
        except Exception:
            has_pth = False
        ranked.append((has_params, has_pth, p))
    ranked.sort(reverse=True)

    if ranked:
        best = ranked[0][2]
        print(f"MDLS_PATH auto-discovered: {best} (fallback from {preferred_path})")
        return best

    print(f"MDLS_PATH not found; keeping: {preferred_path}")
    return preferred_path


MDLS_PATH = _find_models_dir(MDLS_PATH, VER)
print("MDLS_PATH (final):", MDLS_PATH)



## === cell 4
params_path = f"{MDLS_PATH}/params.json"
params = None
LABELS_ = None
LABELS = None

if os.path.exists(params_path):
    with open(params_path) as file:
        params = json.load(file)
    LABELS_ = params["labels_"]
    LABELS = params["labels"]
    WORKERS = 2 if KAGGLE else int(params.get("workers", 2))
    print("loaded params:", params)
else:
    print(
        f"WARNING: params.json not found at {params_path}. Will create a valid submission using sample_submission.csv (all healthy)."
    )
    WORKERS = 2



## === cell 5
sample_path_1 = f"{DATA_PATH}/sample_submission.csv"
sample_path_2 = "../input/sample_submission.csv"

if os.path.exists(sample_path_1):
    df_sub = pd.read_csv(sample_path_1)
elif os.path.exists(sample_path_2):
    df_sub = pd.read_csv(sample_path_2)
else:
    df_sub = pd.DataFrame({"image": sorted(os.listdir(IMGS_PATH))})
    df_sub["labels"] = "healthy"

if "labels" not in df_sub.columns:
    df_sub["labels"] = "healthy"
df_sub = df_sub[["image", "labels"]].copy()
print("df_sub shape:", df_sub.shape)
df_sub.head()



## === cell 6
train_csv_path = f"{DATA_PATH}/train.csv"
KNOWN_LABELS = None
if os.path.exists(train_csv_path):
    _df_tr = pd.read_csv(train_csv_path)
    vocab = set()
    for s in _df_tr["labels"].astype(str).tolist():
        for t in s.split():
            t = t.strip()
            if t:
                vocab.add(t)
    KNOWN_LABELS = sorted(vocab)
    del _df_tr
    print("Known labels from train.csv:", KNOWN_LABELS)
else:
    print("WARNING: train.csv not found for label vocabulary; will not canonicalize.")
    KNOWN_LABELS = None


def _canonicalize_token(tok: str) -> str:
    tok = str(tok).strip()
    if tok == "":
        return tok
    tok2 = tok.lower().replace(" ", "_")
    tok2 = tok2.replace("-", "_")
    return tok2


CANON2OFFICIAL = None
if KNOWN_LABELS is not None:
    CANON2OFFICIAL = {}
    for official in KNOWN_LABELS:
        CANON2OFFICIAL[_canonicalize_token(official)] = official
    if "healthy" not in CANON2OFFICIAL:
        CANON2OFFICIAL["healthy"] = "healthy"


def _to_official_label(tok: str) -> str:
    c = _canonicalize_token(tok)
    if CANON2OFFICIAL is None:
        return c
    return CANON2OFFICIAL.get(c, "")  # unknown -> drop




## === cell 7
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :, :]
    elif axis == 2:
        return img[:, ::-1, :]
    elif axis == 3:
        return img[::-1, ::-1, :]
    else:
        return img


def _get_norm_from_params(params_dict):
    if not isinstance(params_dict, dict):
        return None, None
    mean = params_dict.get("mean", None)
    std = params_dict.get("std", None)
    try:
        if mean is not None and std is not None:
            mean = np.array(mean, dtype=np.float32).reshape(
                3,
            )
            std = np.array(std, dtype=np.float32).reshape(
                3,
            )
            if (
                np.all(np.isfinite(mean))
                and np.all(np.isfinite(std))
                and np.all(std > 0)
            ):
                return mean, std
    except Exception:
        pass
    return None, None


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_PARAM_MEAN, _PARAM_STD = _get_norm_from_params(params)
_NORM_MEAN = _PARAM_MEAN if _PARAM_MEAN is not None else _IMAGENET_MEAN
_NORM_STD = _PARAM_STD if _PARAM_STD is not None else _IMAGENET_STD
print("Using normalization mean/std:", _NORM_MEAN.tolist(), _NORM_STD.tolist())


def _get_interp_from_params(params_dict):
    if not isinstance(params_dict, dict):
        return None
    v = params_dict.get("interpolation", params_dict.get("resize_interp", None))
    if v is None:
        return None
    s = str(v).lower()
    if "area" in s:
        return cv2.INTER_AREA
    if "cubic" in s:
        return cv2.INTER_CUBIC
    if "lanczos" in s:
        return cv2.INTER_LANCZOS4
    if "nearest" in s:
        return cv2.INTER_NEAREST
    if "linear" in s or "bilinear" in s:
        return cv2.INTER_LINEAR
    return None


_FORCED_INTERP = _get_interp_from_params(params)
if _FORCED_INTERP is not None:
    print("Using params-specified resize interpolation:", _FORCED_INTERP)
else:
    print("Using default resize interpolation rule (AREA down / LINEAR up).")


def _resize_to_square(img: np.ndarray, out_size: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((out_size, out_size, 3), dtype=np.uint8)
    if _FORCED_INTERP is not None:
        interp = _FORCED_INTERP
    else:
        interp = cv2.INTER_AREA if (h > out_size or w > out_size) else cv2.INTER_LINEAR
    return cv2.resize(img, (out_size, out_size), interpolation=interp)


class PlantDataset(data.Dataset):
    def __init__(
        self, df, size, labels, transform=None, tta=0, normalize=True, is_test=False
    ):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.labels = labels
        self.transform = transform
        self.tta = int(tta)
        self.normalize = bool(normalize)
        self.is_test = bool(is_test)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = f"{IMGS_PATH}/{img_name}"
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.size, self.size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = _resize_to_square(img, self.size)

        if self.is_test:
            img = flip(img, axis=self.tta)

        img = img.astype(np.float32) / 255.0

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        if not self.is_test:
            img = img.transpose(2, 0, 1)
            label = np.zeros(len(self.labels), dtype=np.float32)
            for lbl in str(row.labels).split():
                if lbl in self.labels:
                    label[self.labels[lbl]] = 1.0
            return torch.tensor(img), torch.tensor(label)
        else:
            if self.normalize:
                img = (img - _NORM_MEAN) / _NORM_STD
            img = img.transpose(2, 0, 1)
            return torch.tensor(img.copy())




## === cell 8
def _make_torchvision_efficientnet(backbone_name: str):
    backbone_name = str(backbone_name).lower()
    if backbone_name in ("efficientnet-b0", "efficientnet_b0", "b0"):
        return torchvision.models.efficientnet_b0(weights=None)
    if backbone_name in ("efficientnet-b1", "efficientnet_b1", "b1"):
        return torchvision.models.efficientnet_b1(weights=None)
    if backbone_name in ("efficientnet-b2", "efficientnet_b2", "b2"):
        return torchvision.models.efficientnet_b2(weights=None)
    if backbone_name in ("efficientnet-b3", "efficientnet_b3", "b3"):
        return torchvision.models.efficientnet_b3(weights=None)
    if backbone_name in ("efficientnet-b4", "efficientnet_b4", "b4"):
        return torchvision.models.efficientnet_b4(weights=None)
    if backbone_name in ("efficientnet-b5", "efficientnet_b5", "b5"):
        return torchvision.models.efficientnet_b5(weights=None)
    if backbone_name in ("efficientnet-b6", "efficientnet_b6", "b6"):
        return torchvision.models.efficientnet_b6(weights=None)
    if backbone_name in ("efficientnet-b7", "efficientnet_b7", "b7"):
        return torchvision.models.efficientnet_b7(weights=None)
    return torchvision.models.efficientnet_b0(weights=None)


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        backbone = (
            params.get("backbone", "efficientnet-b0")
            if params is not None
            else "efficientnet-b0"
        )
        self.enet = _make_torchvision_efficientnet(backbone)

        if hasattr(self.enet, "classifier") and isinstance(
            self.enet.classifier, nn.Sequential
        ):
            nc = int(self.enet.classifier[-1].in_features)
            self.enet.classifier = nn.Identity()
        else:
            nc = 1280

        dropout = float(params.get("dropout", 0.2)) if params is not None else 0.2
        self.myfc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(dropout),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        if hasattr(self.enet, "features") and hasattr(self.enet, "avgpool"):
            x = self.enet.features(x)
            x = self.enet.avgpool(x)
            x = torch.flatten(x, 1)
            return x
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 9
def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "net", "weights"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


models_list = []
can_infer = (
    params is not None
    and LABELS_ is not None
    and LABELS is not None
    and os.path.isdir(MDLS_PATH)
)

if can_infer:
    for n_fold in FOLDS:
        model = EffNet(params, out_dim=len(LABELS_))
        path = "{}/model_best_{}.pth".format(MDLS_PATH, n_fold)
        if not os.path.exists(path):
            print(
                f"WARNING: model weights not found: {path}. Disabling inference and will submit all healthy."
            )
            can_infer = False
            break

        ckpt = torch.load(path, map_location="cpu")
        state_dict = _unwrap_state_dict(ckpt)

        try:
            model.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(
                f"WARNING: strict load failed ({repr(e)}). Trying to strip 'module.' prefixes."
            )
            if isinstance(state_dict, dict):
                new_sd = {}
                for k, v in state_dict.items():
                    nk = k
                    if nk.startswith("module."):
                        nk = nk[len("module.") :]
                    new_sd[nk] = v
                model.load_state_dict(new_sd, strict=True)
            else:
                raise

        model.float()
        model.eval()
        model.to(DEVICE)
        models_list.append(model)
        print("loaded:", path)

    if "ckpt" in locals():
        del ckpt
    if "state_dict" in locals():
        del state_dict
    gc.collect()
else:
    print("Model/params not available; will submit all healthy.")



## === cell 10
loaders = []
if can_infer:
    for tta in TTAS:
        dataset = PlantDataset(
            df=df_sub,
            size=params["img_size"],
            labels=None,
            transform=None,
            tta=tta,
            normalize=True,
            is_test=True,  # Change: explicit test mode so TTA is guaranteed active
        )
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(params["batch_size"]),
            sampler=SequentialSampler(dataset),
            num_workers=int(WORKERS),
            pin_memory=(DEVICE.type == "cuda"),
        )
        loaders.append(loader)




## === cell 11
def _build_idx_to_label(labels_, labels):
    if isinstance(labels_, (list, tuple)):
        return {int(i): str(lbl) for i, lbl in enumerate(labels_)}

    if isinstance(labels_, dict) and len(labels_) > 0:
        keys = list(labels_.keys())

        if all(str(k).isdigit() for k in keys):
            items = sorted(((int(k), labels_[k]) for k in keys), key=lambda x: x[0])
            return {int(k): str(v) for k, v in items}

        vals = list(labels_.values())
        if all(isinstance(v, (int, np.integer)) for v in vals):
            return {int(v): str(k) for k, v in labels_.items()}

    if isinstance(labels, dict) and len(labels) > 0:
        return {int(v): str(k) for k, v in labels.items()}

    return None


IDX2LBL = _build_idx_to_label(LABELS_, LABELS) if can_infer else None
if can_infer and (IDX2LBL is None or len(IDX2LBL) == 0):
    print(
        "WARNING: Could not build IDX2LBL; disabling inference and will submit all healthy."
    )
    can_infer = False
    IDX2LBL = None


if can_infer and IDX2LBL is not None:
    _new = {}
    dropped = 0
    for i, lbl in IDX2LBL.items():
        off = _to_official_label(lbl)
        if off == "":
            dropped += 1
            continue
        _new[int(i)] = off
    if len(_new) > 0:
        IDX2LBL = _new
    print(f"IDX2LBL after canonicalization: {len(IDX2LBL)} classes (dropped {dropped})")


def get_labels(row_probs, idx2lbl, th):
    idxs = [i for i, x in enumerate(row_probs) if float(x) > float(th)]
    row = [idx2lbl[i] for i in idxs if i in idx2lbl]

    if KNOWN_LABELS is not None:
        row = [r for r in row if r in set(KNOWN_LABELS)]

    if len(row) == 0:
        return "healthy"
    if "healthy" in row and len(row) > 1:
        row = [x for x in row if x != "healthy"]
    return " ".join(row)


if can_infer:
    all_preds = []
    with torch.no_grad():
        for i, model in enumerate(models_list):
            for j, loader in enumerate(loaders):
                preds_batches = []
                for img_data in loader:
                    img_data = img_data.to(DEVICE, non_blocking=True).float()
                    preds = model(img_data).sigmoid().detach().cpu().numpy()
                    preds_batches.append(preds)
                preds_tta = np.vstack(preds_batches)  # [N, C] in df_sub order
                all_preds.append(preds_tta)
                print(f"model {i} | tta_loader {j} -> done, shape {preds_tta.shape}")

    logits = np.mean(np.stack(all_preds, axis=0), axis=0)  # [N, C]
    pred_labels = [get_labels(x, IDX2LBL, TH) for x in logits]
    pred_df = pd.DataFrame({"image": df_sub["image"].values, "labels": pred_labels})

    df_sub = df_sub.drop(columns=["labels"], errors="ignore").merge(
        pred_df, on="image", how="left", validate="one_to_one"
    )
    df_sub["labels"] = df_sub["labels"].fillna("healthy")
    df_sub = (
        df_sub.set_index("image")
        .reindex(
            pd.read_csv(
                sample_path_1 if os.path.exists(sample_path_1) else sample_path_2
            )["image"]
        )
        .reset_index()
    )
else:
    df_sub["labels"] = "healthy"

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 12
print("value counts:")
print(df_sub.labels.value_counts().head(20))
df_sub.head()



## === cell 13
df_sub = df_sub[["image", "labels"]].copy()
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())
