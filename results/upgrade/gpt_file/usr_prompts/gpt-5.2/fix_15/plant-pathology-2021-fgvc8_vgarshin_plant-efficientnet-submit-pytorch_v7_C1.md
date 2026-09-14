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

0.8039335180055411

# 6. Current score

0.272

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28044) has done: 'The timeout is dominated by repeated disk I/O and image decoding: the script reads and preprocesses the entire test set separately for each TTA and for each fold (FOLDS×TTAS full passes). To preserve identical model logic and identical TTA semantics while cutting runtime, I cache the preprocessed base image tensors (after resize/normalize/CHW) once per image and then apply flips on the cached tensor for each TTA, eliminating redundant cv2.imread/cvtColor/resize work. I also restructure inference to iterate each loader once and run all fold-models on the same batch (same computations, just reordered), avoiding repeated DataLoader iteration overhead and letting the GPU stay busier. Finally, I speed up label decoding with a vectorized thresholding approach that preserves the exact rule about dropping “healthy” when other labels are present.'
- What this solution (achieved 0.272) has done: 'Your current score (0.28044) is far below the target (0.80393), so we should make a small change that legitimately increases F1 without changing the model or training. The biggest issue is the label decoding: using a fixed TH=0.5 for all classes is usually too strict for this competition and can collapse predictions toward “healthy”, tanking recall and mean F1. I keep the exact same ensemble/TTA/model code, but replace the single global threshold with lightweight, per-class thresholds computed from the training label frequencies (more frequent classes get slightly higher thresholds; rarer classes get lower thresholds), and keep the exact same “drop healthy if other labels present” rule. This preserves evaluation semantics (multi-label thresholding) while typically moving the score substantially upward toward your target.'
- What this solution (achieved 0.272) has done: 'Your score gap to the target is large (0.272 vs 0.8039), and the most likely cause (without changing the model/training) is that the current post-processing thresholds are badly miscalibrated for mean F1 on this dataset. I keep the exact same model, weights loading, ensemble, and TTA inference, but replace the frequency-based thresholds with per-class thresholds selected by maximizing F1 on a held-out split of the training labels using the already-computed label priors (no model training needed). This is a minimal semantic change (still sigmoid + threshold + “drop healthy if others present”), but it usually increases recall/precision balance substantially vs heuristic thresholds. I also ensure label-index mapping is consistent and deterministic, and keep the submission formatting identical.'
- What this solution (achieved 0.272) has done: 'Your score is far below the target, and the biggest “minimal-change” issue is that your threshold selection is currently independent of the model outputs (it uses label priors only), which can easily collapse predictions and tank mean F1. I keep the exact same model, weights loading, TTA, and ensemble averaging, but compute per-class thresholds by maximizing F1 on a held-out split using the model’s own out-of-fold (OOF) predicted probabilities from the training images (same sigmoid+threshold semantics). To stay within the runtime limit, this OOF threshold fitting uses only TTA=0 and the same folds you already load, and it caches decoded/normalized tensors similarly to your test pipeline. Finally, the submission formatting and “drop healthy if other labels present” rule are preserved exactly.'
- What this solution (achieved 0.28044) has done: 'Your gap to the target is large (0.272 → 0.8039), and the most likely “minimal-change” issue is post-processing: your per-class thresholds are optimized independently per class, but the competition metric is *mean samplewise F1*, where the optimal decision rule depends on the *set* of labels predicted per image. I keep the exact same model/weights, ensembling, and TTA averaging, but fit a single global threshold on the held-out OOF probabilities by directly maximizing mean F1 (same sigmoid+threshold semantics). Then I apply that global threshold to test predictions while preserving your existing “if any other label predicted, drop healthy” rule and submission formatting.'
- What this solution (achieved 0.272) has done: 'The timeout is dominated by repeated JPEG decode/resize work: you currently build 4 separate DataLoaders for TTA (reading the same images 4×) and then you also run an extra full-pass OOF inference over ~15k train images to tune a threshold. I keep the exact model forward + sigmoid + ensembling + threshold-search semantics, but eliminate redundant image I/O by caching decoded/resized tensors once per image and reusing them for all TTAs and for OOF; this is provably equivalent because the transforms are deterministic and identical to the original pipeline. I also avoid allocating full `(N,C)` TTA buffers and instead accumulate directly into the final sum, and I precompute label split indices once for the `Y` matrix. Finally, I tune DataLoader settings for deterministic throughput (workers, persistent workers, pinned memory) without changing batch size, model, or thresholds.'
- What this solution (achieved 0.272) has done: 'Your score is far below the target (0.272 vs 0.8039), and the most likely minimal-change cause is that inference is effectively using untrained/random weights because `MDLS_PATH` points to a dataset (`/kaggle/input/plant-models-v1`) that may not exist in your environment. I add a tiny, safe auto-discovery fallback: if `params.json` or model weight files aren’t found in the requested folder, the code search `/kaggle/input/*` for a folder containing the expected `model_best_0.pth`/`model_best_1.pth` and use it. This preserves your exact model, TTA, ensembling, and threshold-fitting logic, but ensures you actually load the trained weights, which should move the score strongly toward your target. I also make label index mapping robust by building `idx_to_name` from `LABELS_` (the true class order) to avoid silent class-order mismatches if `LABELS` is incomplete/misaligned.'
- What this solution (achieved 0.272) has done: 'Your score is far below the target (0.272 vs 0.8039), and the most likely minimal-change cause is that the model outputs are being fed with the wrong input normalization (your dataset currently only scales to [0,1] but EfficientNet models are typically trained with ImageNet mean/std normalization). I keep the exact same model, folds, TTA flips, ensembling, and OOF global-threshold fitting, and only add the missing per-channel normalization in the cached image tensor creation so inference matches training preprocessing. I also make the “count” divisor correct for multiple TTAs (currently it only counts models, not TTAs), which is a pure averaging bug fix that can materially affect thresholding behavior while preserving semantics. These two changes are minimal, directly relevant to mean F1, and should move the score substantially upward toward your target without altering the core approach.'
- What this solution (achieved 0.272) has done: 'Your score is far below the target, so we should move it upward with minimal semantic changes. The two most likely “silent killers” here are (1) a wrong averaging divisor (`count` ignores TTAs, scaling probabilities down) and (2) a fold/model mismatch in OOF threshold fitting (you’re predicting fold-0 validation with fold-0 model instead of the model that didn’t train on that fold). I fix both while keeping the exact same model, weights, caching, TTA flips, sigmoid+thresholding, and “drop healthy if others present” logic. These are correctness fixes that typically yield a large F1 jump without changing the core approach, and still produce a valid `submission.csv`.'

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
from torch.utils.data.sampler import SequentialSampler
from torchvision import models

KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
TEST = True
VER = "v1"

if KAGGLE:
    DATA_PATH = "/kaggle/input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"/kaggle/input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TH = 0.5
TTAS = [0, 1, 2, 3]
FOLDS = [0, 1]

_primary_imgs = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"
_nested_imgs = (
    f"{DATA_PATH}/plant-pathology-2021-fgvc8/test_images"
    if TEST
    else f"{DATA_PATH}/plant-pathology-2021-fgvcvc8/train_images"
)
if os.path.isdir(_primary_imgs):
    IMGS_PATH = _primary_imgs
elif os.path.isdir(_nested_imgs):
    IMGS_PATH = _nested_imgs
else:
    IMGS_PATH = _primary_imgs  # keep original for error visibility

start_time = time.time()
print("DATA_PATH:", DATA_PATH)
print("IMGS_PATH:", IMGS_PATH)
print("MDLS_PATH (requested):", MDLS_PATH)




## === cell 2
def autodiscover_models_dir(requested_dir: str, folds):
    if os.path.isdir(requested_dir):
        ok = True
        for f in folds:
            if not os.path.exists(os.path.join(requested_dir, f"model_best_{f}.pth")):
                ok = False
                break
        if ok:
            return requested_dir

    base = "/kaggle/input"
    if not os.path.isdir(base):
        return requested_dir

    best = None
    try:
        for name in os.listdir(base):
            cand = os.path.join(base, name)
            if not os.path.isdir(cand):
                continue
            if not os.path.exists(os.path.join(cand, f"model_best_{folds[0]}.pth")):
                continue
            all_ok = True
            for f in folds:
                if not os.path.exists(os.path.join(cand, f"model_best_{f}.pth")):
                    all_ok = False
                    break
            if all_ok:
                has_params = os.path.exists(os.path.join(cand, "params.json"))
                best = (
                    (has_params, cand)
                    if best is None
                    else max(best, (has_params, cand))
                )
    except Exception as e:
        print("WARNING: autodiscover failed with:", repr(e))
        return requested_dir

    if best is not None:
        return best[1]
    return requested_dir


MDLS_PATH = autodiscover_models_dir(MDLS_PATH, FOLDS)
print("MDLS_PATH (resolved):", MDLS_PATH)

params_path = f"{MDLS_PATH}/params.json"
if os.path.exists(params_path):
    with open(params_path) as file:
        params = json.load(file)
    LABELS_ = params["labels_"]
    LABELS = params["labels"]
    if KAGGLE:
        WORKERS = max(2, min(8, (os.cpu_count() or 4) // 2))
    else:
        WORKERS = params.get("workers", 2)
    print("loaded params.json from:", params_path, "WORKERS:", WORKERS)
else:
    default_labels = [
        "scab",
        "frog_eye_leaf_spot",
        "rust",
        "complex",
        "powdery_mildew",
        "healthy",
    ]
    LABELS_ = default_labels
    LABELS = {str(i): lab for i, lab in enumerate(default_labels)}
    params = {
        "backbone": "efficientnet_b0",
        "img_size": 224,
        "batch_size": 32,
        "dropout": 0.2,
        "workers": 2,
    }
    if KAGGLE:
        WORKERS = max(2, min(8, (os.cpu_count() or 4) // 2))
    else:
        WORKERS = 2
    print(
        f"WARNING: {params_path} not found. Using fallback params/labels: {params} WORKERS: {WORKERS}"
    )




## === cell 3
sub_path_primary = f"{DATA_PATH}/sample_submission.csv"
sub_path_nested = f"{DATA_PATH}/plant-pathology-2021-fgvc8/sample_submission.csv"
if os.path.exists(sub_path_primary):
    sub_path = sub_path_primary
elif os.path.exists(sub_path_nested):
    sub_path = sub_path_nested
else:
    sub_path = sub_path_primary

df_sub = pd.read_csv(sub_path)
if "labels" not in df_sub.columns:
    df_sub["labels"] = "healthy"
else:
    df_sub["labels"] = "healthy"
print("df_sub shape:", df_sub.shape)
print(df_sub.head())




## === cell 4
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class PlantDatasetCached(data.Dataset):
    def __init__(
        self,
        df,
        size,
        labels,
        tta=0,
        cache=None,
        imgs_path=None,
    ):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.labels = labels  # None for inference
        self.tta = int(tta)
        self.cache = cache if cache is not None else {}
        self.imgs_path = imgs_path if imgs_path is not None else IMGS_PATH

        self._y = None
        if self.labels:
            y = np.zeros((len(self.df), len(self.labels)), dtype=np.float32)
            for i, s in enumerate(self.df["labels"].astype(str).tolist()):
                for lbl in s.split():
                    y[i, self.labels[lbl]] = 1.0
            self._y = y

    def __len__(self):
        return self.df.shape[0]

    def _load_base(self, img_name: str):
        t = self.cache.get(img_name)
        if t is not None:
            return t
        img_path = f"{self.imgs_path}/{img_name}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not readable: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # CHW
        base = torch.from_numpy(img)  # float32 CPU tensor in [0,1]
        base = (base - IMAGENET_MEAN) / IMAGENET_STD  # normalized float32 CPU tensor
        self.cache[img_name] = base
        return base

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        base = self._load_base(img_name)

        if self.labels:
            return base, torch.from_numpy(self._y[index])

        if self.tta == 0:
            return base
        elif self.tta == 1:
            return torch.flip(base, dims=(1,))  # vertical flip (H)
        elif self.tta == 2:
            return torch.flip(base, dims=(2,))  # horizontal flip (W)
        else:
            return torch.flip(base, dims=(1, 2))  # both


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        self.enet = models.efficientnet_b0(weights=None)
        nc = self.enet.classifier[1].in_features
        self.enet.classifier = nn.Identity()
        self.myfc = nn.Sequential(
            nn.Dropout(params["dropout"]),
            nn.Linear(nc, int(nc / 4)),
            nn.ELU(),
            nn.BatchNorm1d(int(nc / 4)),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 5
models_ens = []
missing_any = False
for n_fold in FOLDS:
    model = EffNet(params, out_dim=len(LABELS_))
    path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("loaded:", path)
        del state_dict
    else:
        missing_any = True
        print(f"WARNING: missing weights {path}. Using untrained model for this fold.")
    model.float()
    model.eval()
    model.to(DEVICE)
    models_ens.append(model)

if missing_any:
    print(
        "WARNING: One or more folds are missing weight files. "
        "Score will be very low if running with untrained weights."
    )
gc.collect()




## === cell 6
test_cache = {}

datasets, loaders = [], []
for tta in TTAS:
    dataset = PlantDatasetCached(
        df=df_sub,
        size=params["img_size"],
        labels=None,
        tta=tta,
        cache=test_cache,
        imgs_path=IMGS_PATH,
    )
    datasets.append(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params["batch_size"],
        sampler=SequentialSampler(dataset),
        num_workers=WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(WORKERS > 0),
        prefetch_factor=4 if WORKERS > 0 else None,
        drop_last=False,
    )
    loaders.append(loader)
print("Prepared loaders:", len(loaders), "Shared test cache dict:", "enabled")




## === cell 7
@torch.no_grad()
def predict_sum_over_models(xb, models_list):
    preds = []
    for m in models_list:
        preds.append(m(xb).sigmoid())
    return torch.stack(preds, dim=0).sum(dim=0)


with torch.no_grad():
    N = len(df_sub)
    C = len(LABELS_)
    all_preds_sum = np.zeros((N, C), dtype=np.float32)

    count = 0  # number of (model, TTA) predictions accumulated per sample

    for j, loader in enumerate(loaders):
        ofs = 0
        for img_data in loader:
            bs = img_data.shape[0]
            img_data = img_data.to(DEVICE, non_blocking=True)
            batch_sum = predict_sum_over_models(img_data, models_ens)
            all_preds_sum[ofs : ofs + bs] += batch_sum.detach().cpu().numpy()
            ofs += bs

        count += len(models_ens)  # models for this TTA
        print(f"loader {j} -> done, summed over {len(models_ens)} model(s)")

logits = all_preds_sum / float(count)

idx_to_name = list(LABELS_)
healthy_idx = idx_to_name.index("healthy") if "healthy" in idx_to_name else None

train_csv_primary = f"{DATA_PATH}/train.csv"
train_csv_nested = f"{DATA_PATH}/plant-pathology-2021-fgvc8/train.csv"
train_csv_path = (
    train_csv_primary if os.path.exists(train_csv_primary) else train_csv_nested
)
df_train = pd.read_csv(train_csv_path)

name_to_idx = {name: i for i, name in enumerate(idx_to_name)}

Y = np.zeros((len(df_train), len(LABELS_)), dtype=np.uint8)
labels_list = df_train["labels"].astype(str).str.split().tolist()
for i, labs in enumerate(labels_list):
    for lab in labs:
        j = name_to_idx.get(lab)
        if j is not None:
            Y[i, j] = 1

_primary_train_imgs = f"{DATA_PATH}/train_images"
_nested_train_imgs = f"{DATA_PATH}/plant-pathology-2021-fgvc8/train_images"
if os.path.isdir(_primary_train_imgs):
    TRAIN_IMGS_PATH = _primary_train_imgs
elif os.path.isdir(_nested_train_imgs):
    TRAIN_IMGS_PATH = _nested_train_imgs
else:
    TRAIN_IMGS_PATH = _primary_train_imgs


def sample_f1_mean(y_true_bin, y_pred_bin):
    tp = (y_true_bin & y_pred_bin).sum(axis=1).astype(np.float32)
    fp = ((1 - y_true_bin) & y_pred_bin).sum(axis=1).astype(np.float32)
    fn = (y_true_bin & (1 - y_pred_bin)).sum(axis=1).astype(np.float32)
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, (2 * tp) / denom, 0.0)
    return float(f1.mean())


def make_folds_by_labelcount(df, n_splits=2, seed=0):
    rng = np.random.RandomState(seed)
    label_counts = df["labels"].astype(str).apply(lambda x: len(x.split())).values
    order = np.lexsort((rng.rand(len(df)), label_counts))
    folds = np.zeros(len(df), dtype=np.int64)
    for k, idx in enumerate(order):
        folds[idx] = k % n_splits
    return folds


_old_imgs_path = IMGS_PATH
IMGS_PATH = TRAIN_IMGS_PATH

fold_ids = make_folds_by_labelcount(df_train, n_splits=max(2, len(FOLDS)), seed=0)

train_cache = {}

oof_pred = np.zeros((len(df_train), len(LABELS_)), dtype=np.float32)
oof_mask = np.zeros((len(df_train),), dtype=bool)

for fold_pos, fold_id in enumerate(FOLDS):
    if fold_id >= fold_ids.max() + 1:
        continue
    val_idx = np.where(fold_ids == fold_id)[0]
    if val_idx.size == 0:
        continue

    df_val = df_train.iloc[val_idx].reset_index(drop=True)
    val_ds = PlantDatasetCached(
        df=df_val,
        size=params["img_size"],
        labels=None,
        tta=0,  # unchanged
        cache=train_cache,
        imgs_path=IMGS_PATH,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=params["batch_size"],
        sampler=SequentialSampler(val_ds),
        num_workers=WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(WORKERS > 0),
        prefetch_factor=4 if WORKERS > 0 else None,
        drop_last=False,
    )

    if len(models_ens) >= 2:
        model = models_ens[1 - fold_pos]
    else:
        model = models_ens[0]

    with torch.no_grad():
        ofs = 0
        Pv_fold = np.zeros((len(df_val), len(LABELS_)), dtype=np.float32)
        for xb in val_loader:
            bs = xb.shape[0]
            xb = xb.to(DEVICE, non_blocking=True)
            p = model(xb).sigmoid()
            Pv_fold[ofs : ofs + bs] = p.detach().cpu().numpy()
            ofs += bs

    oof_pred[val_idx] = Pv_fold
    oof_mask[val_idx] = True
    print(f"OOF preds collected for fold {fold_id}: {val_idx.size} rows")

IMGS_PATH = _old_imgs_path

Pv = oof_pred[oof_mask]
Yv = Y[oof_mask].astype(np.uint8)

grid = np.linspace(0.05, 0.95, 37, dtype=np.float32)
best_t, best_score = 0.5, -1.0
for t in grid:
    pred = (Pv > float(t)).astype(np.uint8)
    if healthy_idx is not None:
        other_any = (pred.sum(axis=1) > 1) & (pred[:, healthy_idx] == 1)
        if np.any(other_any):
            pred[other_any, healthy_idx] = 0
    score = sample_f1_mean(Yv, pred)
    if score > best_score:
        best_score, best_t = score, float(t)

thr = np.full(len(LABELS_), best_t, dtype=np.float32)
if healthy_idx is not None:
    thr[healthy_idx] = max(float(thr[healthy_idx]), 0.50)

print(
    "Global threshold (OOF-by-fold fitted for mean samplewise F1):",
    best_t,
    "OOF mean F1:",
    best_score,
)
print("Applied thresholds:", {idx_to_name[i]: float(thr[i]) for i in range(len(thr))})

mask = logits > thr[None, :]
if healthy_idx is not None:
    other_present = (mask.sum(axis=1) > 1) & (mask[:, healthy_idx])
    if np.any(other_present):
        mask[other_present, healthy_idx] = False

out_labels = []
for r in range(mask.shape[0]):
    idxs = np.flatnonzero(mask[r])
    if idxs.size == 0:
        out_labels.append("healthy")
    else:
        out_labels.append(" ".join(idx_to_name[i] for i in idxs))
df_sub["labels"] = out_labels

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")




## === cell 8
print("value counts:")
print(df_sub.labels.value_counts().head(20))
print(df_sub.head())




## === cell 9
out = df_sub[["image", "labels"]].copy()
out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())
