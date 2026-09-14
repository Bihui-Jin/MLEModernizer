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

0.8273499538319488

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I cut the timeout by eliminating repeated disk I/O and Python overhead during inference: cache and pre-process each image once (resize/normalize) and reuse it across all TTAs and all folds. I also speed up DataLoader transfers with `persistent_workers`, `prefetch_factor`, `pin_memory`, and an explicit `collate_fn` to stack tensors efficiently, while keeping the same model forward pass, sigmoid, ensembling, and thresholding semantics. Finally, I avoid building large intermediate Python lists/`vstack` arrays by accumulating probabilities directly into the final numpy buffer in correct sample order. These changes are provably equivalent in outputs (same pixels, same flips, same averaging), just much less redundant work.'
- What this solution (achieved 0.24507) has done: 'Your current low score is consistent with running mostly untrained models because `MDLS_PATH` points to a dataset (`../input/plant-models-v4`) that likely isn’t attached, so the code silently falls back to random weights and predicts near-noise. I make a minimal, score-relevant change: automatically locate the correct folder containing the fold checkpoints (`model_best_*.pth`) under `../input/` (or fall back to your current path if found) so the exact same architecture/inference logic runs with the intended trained weights. I also expand `FOLDS` to load all available fold checkpoints found (still ensembling the same way) to move performance upward toward your target. No changes to the model, loss, TTA semantics, thresholding, or submission format.'
- What this solution (achieved 0.24507) has done: 'Your score is far below the target, and the most likely cause is still that inference is running with untrained (or mismatched) weights and/or mismatched input normalization compared to how the checkpoints were trained. I make two minimal, score-relevant fixes while preserving your model, inference loop, TTAs, thresholding, and submission format: (1) make checkpoint discovery stricter and refuse to ensemble missing folds (so you don’t dilute predictions with random-weight models), and (2) apply the standard EfficientNet ImageNet normalization at inference (a common requirement for EfficientNet checkpoints). These changes keep the same architecture and semantics (sigmoid + mean ensemble + threshold), but should move the score substantially upward toward your target.'
- What this solution (achieved 0.24507) has done: 'Your current score is far below the target, so we should improve it with minimal, score-relevant fixes while keeping the same model/inference/thresholding logic. The biggest likely issue is label-index mapping: your inference uses `LABELS` as an index→name dict, but your dataset/training convention often uses name→index; this can silently break checkpoint alignment and output decoding, crushing F1. I add a small, robust mapping normalization step that guarantees a correct `name_to_index` and `index_to_name` are derived from `labels_`/`labels` regardless of how `params.json` stores them, and then use the correct mapping in `get_labels` and (only if ever used) training-label parsing. I also make the inference preprocessing consistent across all TTAs by applying the same flip-before-normalize ordering (kept) but ensure contiguity/float type consistently to avoid subtle channel/stride issues.'
- What this solution (achieved 0.24507) has done: 'Your score is far below the target, so the most likely issue is still that inference is being diluted by missing/mismatched checkpoints and/or wrong class order relative to the saved weights. I (1) make checkpoint discovery strict and fail fast if no checkpoints are found (so you don’t accidentally submit random-weight predictions), and (2) enforce the checkpoint’s class order by rebuilding `LABELS_ / INDEX_TO_NAME / NAME_TO_INDEX` from `params.json` in a way that prefers an explicit index→name mapping when present. Finally, I keep your exact inference core (EffNet, sigmoid, mean over folds+TTAs, same threshold), but I make `get_labels` respect the competition’s multi-label semantics by not forcing `healthy` to override other predicted diseases (this single line can heavily affect mean F1 while preserving the same thresholding logic). These are minimal, score-relevant changes that should move the score upward toward the target without changing the model architecture or inference loop structure.'
- What this solution (achieved 0.24507) has done: 'I fix the immediate runtime failure by making checkpoint discovery robust across common Kaggle input layouts and filenames, so the code actually loads the intended trained weights instead of crashing or silently using none. I keep your model, TTA, averaging, sigmoid, and thresholding logic unchanged, and only adjust the model-loading path logic plus support for common checkpoint key wrappers (e.g., `state_dict`). Finally, I ensure the script always writes a valid `submission.csv` with the required `image,labels` columns, using the sample submission as the definitive row order.'
- What this solution (achieved 0.24507) has done: 'I fix the runtime failure by making the checkpoint discovery search deeper under `../input` and only accept directories that actually contain `.pth/.pt` files, including common nested layouts; this ensures the intended trained weights are loaded instead of raising `FileNotFoundError`. I also add a strict but informative fallback that prints the top candidate directories and a clear error if none contain checkpoints, so you don’t accidentally submit random-weight outputs. These changes preserve your model architecture, TTA/ensemble logic, sigmoid/thresholding, and submission formatting, but should move the score substantially upward because the main issue is currently “no weights loaded”. Finally, I keep the submission row order aligned to `sample_submission.csv` and always write `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'I fix the runtime failure by making checkpoint discovery broader and tolerant of common Kaggle layouts, and by falling back to a “no-checkpoint” safe path that still produces a valid `submission.csv` (so you can submit even if the model dataset isn’t attached). If checkpoints are found anywhere under `../input`, the exact same model architecture, sigmoid/mean ensembling, TTA, and thresholding logic run using those weights (which should substantially raise your score toward the target, since the current score indicates random/untrained inference). I also ensure we keep the sample submission row order (no filtering to only existing JPGs), because Kaggle expects predictions for every row in `sample_submission.csv`. All other logic (EffNet definition, TTAs, averaging, thresholding, output formatting) is preserved.'
- What this solution (achieved 0.24507) has done: 'Your score is far below the target, so we should improve it with the smallest changes that address likely correctness issues rather than tuning. The biggest score killer here is that the notebook currently *fails* if any `sample_submission.csv` image is not physically present in `/test_images` (common because the real hidden test set isn’t shipped locally), which prevents proper Kaggle inference; we make image loading robust by falling back to a neutral gray image when a file is missing so the script always completes and produces a valid submission. Next, we ensure checkpoint loading is *strictly non-dilutive*: we only ensemble folds that actually loaded weights (already mostly true) and we also correct a subtle TTA bug: `flip()`’s axis mapping is wrong for H/W flips, which changes TTA semantics and hurts accuracy; we fix it without changing the overall “3 TTAs + average” approach. These minimal fixes keep your model, sigmoid+mean ensembling, thresholding, and submission formatting intact, but should move F1 substantially upward toward your target.'
- What this solution (achieved 0.24507) has done: 'Your score is far below the target, so the most likely remaining correctness issue is that the TTA definitions don’t match what the checkpoints expect (and what you intended): your `flip()` maps TTA codes to different geometric transforms than the common convention (0=no flip, 1=horizontal, 2=vertical), which can heavily degrade an ensemble without changing the model. I make the minimal fix: redefine `flip()` so `axis=1` is horizontal (left-right), `axis=2` is vertical (top-bottom), while keeping the same TTAs `[0,1,2]`, same averaging, sigmoid, threshold, and submission formatting. This preserves core logic but aligns TTA semantics with typical training/inference usage and should move mean F1 substantially upward toward your target. Everything else (checkpoint discovery/loading, ImageNet normalization, caching, DataLoader settings, decoding) is left intact.'
- What this solution (achieved 0.24507) has done: 'Your current score is far below the target, so the smallest score-relevant fix is to ensure inference is actually using the correct trained checkpoints and not silently running with random weights or mismatched class order. I make checkpoint discovery stricter: it prefer directories that contain a `params.json` plus fold-like checkpoint filenames, and it only ensemble checkpoints whose output layer size matches `len(LABELS_)` (skipping incompatible weights instead of diluting predictions). I also make label-map construction prefer the checkpoint’s explicit index→name mapping when present, to avoid class-order mismatches that can destroy mean F1. The model, TTAs, sigmoid+mean ensembling, thresholding, and submission format are preserved.'
- What this solution (achieved 0.24507) has done: 'The current score is far below the target, so the highest-probability minimal fix is to ensure we’re actually using the correct trained checkpoints and their exact class order/normalization; otherwise inference behaves like random/noisy predictions. I (1) tighten checkpoint directory selection to strongly prefer a folder that contains `params.json` and fold-like `model_best_*.pth` files, and (2) when a checkpoint is loaded, prefer its embedded label mapping (if present) to avoid class-order mismatch that can collapse mean F1. I also make preprocessing follow the common EfficientNet inference convention by applying ImageNet normalization after TTA flip (kept) but additionally converting to torch tensor with explicit dtype to avoid any silent type/stride issues. Core model architecture, sigmoid+mean ensembling, TTAs, thresholding, and submission formatting remain unchanged.'

# 9. Code solution

## === cell 0
import os
import gc
import json
import time
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.sampler import SequentialSampler
import torchvision
from torchvision import models

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "torch:",
    torch.__version__,
    "| torchvision:",
    torchvision.__version__,
    "| device:",
    DEVICE,
)

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
TEST = True
VER = "v4"

if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TH = 0.3
TTAS = [0, 1, 2]
FOLDS = [0, 1, 2, 3, 4]

IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"
start_time = time.time()

if not os.path.isdir(IMGS_PATH):
    alt = "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/test_images"
    if TEST and os.path.isdir(alt):
        IMGS_PATH = alt
    else:
        alt2 = "../input/plant-pathology-2021-fgvcvc8/test_images"
        if os.path.isdir(alt2):
            IMGS_PATH = alt2

if not os.path.isdir(IMGS_PATH):
    alt3 = "../input/plant-pathology-2021-fgvc8/test_images"
    if os.path.isdir(alt3):
        IMGS_PATH = alt3

print("IMGS_PATH:", IMGS_PATH)



## === cell 2
default_params = {
    "backbone": "efficientnet_b0",
    "img_size": 512,
    "batch_size": 16,
    "dropout": 0.5,
    "workers": 2,
    "labels_": [
        "healthy",
        "scab",
        "frog_eye_leaf_spot",
        "rust",
        "complex",
        "powdery_mildew",
    ],
    "labels": {
        "0": "healthy",
        "1": "scab",
        "2": "frog_eye_leaf_spot",
        "3": "rust",
        "4": "complex",
        "5": "powdery_mildew",
    },
}


def _list_ckpts_recursive(root_dir: str):
    ckpts = []
    if not os.path.isdir(root_dir):
        return ckpts
    for r, _, files in os.walk(root_dir):
        for f in files:
            lf = f.lower()
            if lf.endswith(".pth") or lf.endswith(".pt"):
                ckpts.append(os.path.join(r, f))
    return ckpts


def _score_candidate_dir(d: str, ver: str):
    base = os.path.basename(d).lower()
    sc = 0
    if f"plant-models-{ver}" in base:
        sc += 2000
    if os.path.isfile(os.path.join(d, "params.json")):
        sc += 1500
    if "model" in base or "ckpt" in base or "checkpoint" in base:
        sc += 50

    try:
        ckpts = _list_ckpts_recursive(d)
        sc += 2 * len(ckpts)
        foldish = 0
        for p in ckpts:
            bn = os.path.basename(p).lower()
            if ("model_best_" in bn) or ("best_model_" in bn) or ("fold" in bn):
                foldish += 1
        sc += 50 * foldish
    except Exception:
        pass
    return sc


def _find_models_dir_and_ckpts(preferred_dir: str, ver: str, max_depth: int = 10):
    preferred_ckpts = (
        _list_ckpts_recursive(preferred_dir) if os.path.isdir(preferred_dir) else []
    )
    if len(preferred_ckpts) > 0:
        return preferred_dir, preferred_ckpts

    root = "../input" if KAGGLE else "."
    all_ckpts = []

    if os.path.isdir(root):
        queue = [(root, 0)]
        seen = set()
        while queue:
            d, depth = queue.pop(0)
            if d in seen:
                continue
            seen.add(d)
            if depth > max_depth:
                continue

            try:
                with os.scandir(d) as it:
                    subdirs = []
                    for entry in it:
                        if entry.is_file():
                            lf = entry.name.lower()
                            if lf.endswith(".pth") or lf.endswith(".pt"):
                                all_ckpts.append(entry.path)
                        elif entry.is_dir():
                            subdirs.append(entry.path)
                for sd in subdirs:
                    queue.append((sd, depth + 1))
            except Exception:
                continue

    if len(all_ckpts) == 0:
        return preferred_dir, []

    cand_dirs = set()
    for p in all_ckpts:
        d = os.path.dirname(p)
        cand_dirs.add(d)
        cand_dirs.add(os.path.dirname(d))

    cand_dirs = [d for d in cand_dirs if os.path.isdir(d)]
    cand_dirs = sorted(
        cand_dirs, key=lambda d: _score_candidate_dir(d, ver), reverse=True
    )
    best_dir = cand_dirs[0] if cand_dirs else os.path.dirname(all_ckpts[0])

    best_ckpts = [p for p in all_ckpts if os.path.commonpath([best_dir, p]) == best_dir]

    print("Top checkpoint-containing dirs (up to 10):")
    for c in cand_dirs[:10]:
        print(
            "  ",
            c,
            "| ckpts:",
            len(_list_ckpts_recursive(c)),
            "| has_params:",
            os.path.isfile(os.path.join(c, "params.json")),
        )
    print("Total ckpt files found under ../input:", len(all_ckpts))
    print("Selected best_dir:", best_dir, "| ckpts under it:", len(best_ckpts))

    return best_dir, best_ckpts


MDLS_PATH, _FOUND_CKPTS = _find_models_dir_and_ckpts(MDLS_PATH, VER)
print("MDLS_PATH selected:", MDLS_PATH)
print("ckpts found:", len(_FOUND_CKPTS))

params_path = os.path.join(MDLS_PATH, "params.json")
if os.path.isfile(params_path):
    with open(params_path) as file:
        params = json.load(file)
else:
    params = default_params

_raw_labels = params.get("labels", default_params["labels"])
_labels_list = params.get("labels_", default_params["labels_"])


def _build_label_maps(labels_list, labels_obj):
    name_to_index = {name: i for i, name in enumerate(labels_list)}
    index_to_name = {i: name for i, name in enumerate(labels_list)}

    if isinstance(labels_obj, dict) and len(labels_obj) > 0:

        def _is_intlike(x):
            if isinstance(x, int):
                return True
            if isinstance(x, str) and x.isdigit():
                return True
            return False

        keys = list(labels_obj.keys())
        vals = list(labels_obj.values())
        keys_intlike = all(_is_intlike(k) for k in keys)
        vals_intlike = all(_is_intlike(v) for v in vals)

        if keys_intlike and not vals_intlike:
            idx2name = {int(k): str(v) for k, v in labels_obj.items()}
            c = len(idx2name)
            if set(idx2name.keys()) == set(range(c)):
                index_to_name = dict(sorted(idx2name.items(), key=lambda x: x[0]))
                name_to_index = {v: k for k, v in index_to_name.items()}
                labels_list = [index_to_name[i] for i in range(c)]
        elif (not keys_intlike) and vals_intlike:
            n2idx = {str(k): int(v) for k, v in labels_obj.items()}
            c = len(n2idx)
            if set(n2idx.values()) == set(range(c)):
                name_to_index = n2idx
                index_to_name = {v: k for k, v in name_to_index.items()}
                labels_list = [index_to_name[i] for i in range(c)]

    return labels_list, name_to_index, index_to_name


LABELS_, NAME_TO_INDEX, INDEX_TO_NAME = _build_label_maps(_labels_list, _raw_labels)

WORKERS = 2 if KAGGLE else int(params.get("workers", 2))
print("loaded params:", params)
print("num labels:", len(LABELS_))
print("label order (LABELS_ used):", LABELS_)
print("INDEX_TO_NAME:", INDEX_TO_NAME)
print("NAME_TO_INDEX:", NAME_TO_INDEX)



## === cell 3
sub_path = f"{DATA_PATH}/sample_submission.csv"
if not os.path.isfile(sub_path):
    sub_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"

df_sub = pd.read_csv(sub_path)
df_sub = df_sub[["image", "labels"]].copy()

print("df_sub shape (from sample_submission):", df_sub.shape)
print(df_sub.head())




## === cell 4
def flip(img, axis=0):
    if axis == 1:  # horizontal flip (flip columns / width)
        return img[:, ::-1, :]
    elif axis == 2:  # vertical flip (flip rows / height)
        return img[::-1, :, :]
    elif axis == 3:  # both
        return img[::-1, ::-1, :]
    else:
        return img


_IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMNET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_imagenet(img_rgb_01: np.ndarray) -> np.ndarray:
    return (img_rgb_01 - _IMNET_MEAN) / _IMNET_STD


class PlantDataset(data.Dataset):
    def __init__(self, df, size, labels, transform=None, tta=0, base_cache=None):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.labels = labels
        self.transform = transform
        self.tta = tta
        self.base_cache = base_cache

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        if self.base_cache is not None:
            img = self.base_cache[index]
        else:
            row = self.df.iloc[index]
            img_name = row.image
            img_path = f"{IMGS_PATH}/{img_name}"
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found/readable: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.size, self.size))
            img = img.astype(np.float32) / 255.0

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        if self.labels:
            img = img.transpose(2, 0, 1)
            row = self.df.iloc[index]
            label = np.zeros(len(self.labels)).astype(np.float32)
            for lbl in row.labels.split():
                label[self.labels[lbl]] = 1
            return torch.tensor(img), torch.tensor(label)
        else:
            img = flip(img, axis=self.tta)
            img = normalize_imagenet(img).astype(np.float32, copy=False)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            return torch.from_numpy(img)


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        backbone = params.get("backbone", "efficientnet_b0")
        if backbone == "efficientnet_b0":
            base = models.efficientnet_b0(weights=None)
            feat_dim = 1280
        elif backbone == "efficientnet_b1":
            base = models.efficientnet_b1(weights=None)
            feat_dim = 1280
        elif backbone == "efficientnet_b2":
            base = models.efficientnet_b2(weights=None)
            feat_dim = 1408
        elif backbone == "efficientnet_b3":
            base = models.efficientnet_b3(weights=None)
            feat_dim = 1536
        else:
            base = models.efficientnet_b0(weights=None)
            feat_dim = 1280

        self.enet = base
        self.enet.classifier = nn.Identity()

        drop = float(params.get("dropout", 0.5))
        self.myfc = nn.Sequential(
            nn.Dropout(drop),
            nn.Linear(feat_dim, int(feat_dim / 4)),
            nn.Dropout(drop),
            nn.Linear(int(feat_dim / 4), out_dim),
        )

    def extract(self, x):
        x = self.enet.features(x)
        x = self.enet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 5
def _extract_fold_id(fname: str):
    base = os.path.basename(fname)
    stem = os.path.splitext(base)[0]
    for tok in ["model_best_", "best_model_", "fold_", "fold"]:
        if tok in stem:
            try:
                tail = stem.split(tok, 1)[1]
                digits = ""
                for ch in tail:
                    if ch.isdigit():
                        digits += ch
                    else:
                        break
                if digits != "":
                    return int(digits)
            except Exception:
                pass
    if stem.isdigit():
        return int(stem)
    return None


def _collect_fold_ckpts_from_list(ckpt_files):
    fold_to_path = {}
    for p in ckpt_files:
        fid = _extract_fold_id(p)
        if fid is None:
            continue
        cur = fold_to_path.get(fid)
        if cur is None:
            fold_to_path[fid] = p
        else:
            pbase = os.path.basename(p).lower()
            cbase = os.path.basename(cur).lower()

            def _pref(b):
                s = 0
                if "model_best" in b:
                    s += 100
                if "best" in b:
                    s += 10
                if b.endswith(".pth"):
                    s += 1
                return s

            if _pref(pbase) > _pref(cbase):
                fold_to_path[fid] = p
    return dict(sorted(fold_to_path.items(), key=lambda x: x[0]))


fold_ckpts = _collect_fold_ckpts_from_list(
    _FOUND_CKPTS if len(_FOUND_CKPTS) > 0 else _list_ckpts_recursive(MDLS_PATH)
)
print("Found fold checkpoints:", fold_ckpts)

if len(fold_ckpts) > 0:
    FOLDS = list(fold_ckpts.keys())
print("FOLDS selected:", FOLDS)


def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
    return obj


def _state_dict_out_dim(sd: dict):
    if not isinstance(sd, dict):
        return None
    for key in ["myfc.3.weight", "myfc.3.bias"]:
        if key in sd and hasattr(sd[key], "shape"):
            if key.endswith("weight") and len(sd[key].shape) == 2:
                return int(sd[key].shape[0])
            if key.endswith("bias") and len(sd[key].shape) == 1:
                return int(sd[key].shape[0])
    return None


def _maybe_update_label_maps_from_ckpt_obj(ckpt_obj):
    global LABELS_, NAME_TO_INDEX, INDEX_TO_NAME
    if not isinstance(ckpt_obj, dict):
        return
    cand = None
    if "params" in ckpt_obj and isinstance(ckpt_obj["params"], dict):
        cand = ckpt_obj["params"]
    elif "hparams" in ckpt_obj and isinstance(ckpt_obj["hparams"], dict):
        cand = ckpt_obj["hparams"]
    if cand is None:
        return
    lbls_obj = cand.get("labels", None)
    lbls_list = cand.get("labels_", None)
    if lbls_obj is None and lbls_list is None:
        return
    new_list = lbls_list if lbls_list is not None else LABELS_
    new_obj = (
        lbls_obj
        if lbls_obj is not None
        else {str(i): n for i, n in enumerate(new_list)}
    )
    new_LABELS_, new_N2I, new_I2N = _build_label_maps(new_list, new_obj)
    if len(new_LABELS_) == len(LABELS_):
        LABELS_, NAME_TO_INDEX, INDEX_TO_NAME = new_LABELS_, new_N2I, new_I2N
        print("Updated label maps from checkpoint metadata. LABELS_:", LABELS_)


models_list = []
for n_fold in FOLDS:
    model = EffNet(params, out_dim=len(LABELS_))
    path = fold_ckpts.get(n_fold, os.path.join(MDLS_PATH, f"model_best_{n_fold}.pth"))
    if os.path.isfile(path):
        state_obj = torch.load(path, map_location="cpu")

        _maybe_update_label_maps_from_ckpt_obj(state_obj)

        state_dict = _unwrap_state_dict(state_obj)

        out_dim_sd = _state_dict_out_dim(state_dict)
        if out_dim_sd is not None and out_dim_sd != len(LABELS_):
            print(
                "checkpoint incompatible, SKIPPING fold:",
                n_fold,
                "| path:",
                path,
                "| ckpt_out_dim:",
                out_dim_sd,
                "| expected:",
                len(LABELS_),
            )
            continue

        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            new_sd = {}
            for k, v in state_dict.items():
                nk = k.replace("module.", "")
                new_sd[nk] = v
            model.load_state_dict(new_sd, strict=False)

        print("loaded:", path)
        model.float()
        model.eval()
        model.to(DEVICE)
        models_list.append(model)
    else:
        print("checkpoint missing, SKIPPING fold:", n_fold, "| expected:", path)

gc.collect()

if not models_list:
    print("WARNING: No compatible checkpoints loaded from MDLS_PATH:", MDLS_PATH)
    print(
        "Proceeding with default 'healthy' predictions to ensure a valid submission is produced."
    )
else:
    print("num models:", len(models_list))



## === cell 6
img_size = int(params["img_size"])

_missing = 0
base_cache = []
for img_name in df_sub["image"].values:
    img_path = f"{IMGS_PATH}/{img_name}"
    img = cv2.imread(img_path)
    if img is None:
        _missing += 1
        img = np.full((img_size, img_size, 3), 127, dtype=np.uint8)  # neutral gray
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    base_cache.append(img)
print(
    "cached base images:",
    len(base_cache),
    "| size:",
    img_size,
    "| missing filled:",
    _missing,
)


def fast_collate(batch):
    return torch.stack(batch, dim=0)


datasets, loaders = [], []
bs = int(params["batch_size"])
use_cuda = torch.cuda.is_available()
for tta in TTAS:
    dataset = PlantDataset(
        df=df_sub,
        size=img_size,
        labels=None,
        transform=None,
        tta=tta,
        base_cache=base_cache,
    )
    datasets.append(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=bs,
        sampler=SequentialSampler(dataset),
        num_workers=WORKERS,
        pin_memory=use_cuda,
        persistent_workers=(WORKERS > 0),
        prefetch_factor=4 if WORKERS > 0 else None,
        collate_fn=fast_collate,
    )
    loaders.append(loader)

print("num loaders (TTA):", len(loaders))




## === cell 7
def get_labels(row, index_to_name, th):
    idx = [i for i, x in enumerate(row) if x > th]
    out = [index_to_name.get(i, str(i)) for i in idx]

    if len(out) == 0:
        return "healthy"
    if "healthy" in out and len(out) > 1:
        out = [x for x in out if x != "healthy"]
        if len(out) == 0:
            return "healthy"
    return " ".join(out)


if not models_list:
    df_sub["labels"] = "healthy"
else:
    num_imgs = len(df_sub)
    num_classes = len(LABELS_)
    probs_accum = np.zeros((num_imgs, num_classes), dtype=np.float32)

    with torch.no_grad():
        for mi, model in enumerate(models_list):
            for ti, loader in enumerate(loaders):
                offset = 0
                for img_data in loader:
                    img_data = img_data.to(DEVICE, non_blocking=True).float()
                    preds = model(img_data).sigmoid()
                    bsz = preds.shape[0]
                    probs_accum[offset : offset + bsz] += (
                        preds.detach().cpu().numpy().astype(np.float32)
                    )
                    offset += bsz
                print(f"model {mi} | tta {ti} -> done, N={offset}")

    probs = probs_accum / float(len(models_list) * len(loaders))
    df_sub["labels"] = [get_labels(p, INDEX_TO_NAME, TH) for p in probs]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 8
print("value counts:")
print(df_sub.labels.value_counts().head(20))
print(df_sub.head())



## === cell 9
df_sub = df_sub[["image", "labels"]]
df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
print(df_sub.head())
