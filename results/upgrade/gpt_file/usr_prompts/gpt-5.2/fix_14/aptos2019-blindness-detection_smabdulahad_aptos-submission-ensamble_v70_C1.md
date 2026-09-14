# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.12

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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import timm
from tqdm import tqdm




## === cell 1
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_path(*candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return candidates[0]


BASE_INPUT = resolve_path(
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data/input/aptos2019-blindness-detection",
)

TRAIN_CSV = resolve_path(
    os.path.join(BASE_INPUT, "train.csv"),
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
)
TEST_CSV = resolve_path(
    os.path.join(BASE_INPUT, "test.csv"),
    "/kaggle/input/test.csv",
    "/kaggle/data/test.csv",
)

TRAIN_IMG_DIR = resolve_path(
    os.path.join(BASE_INPUT, "train_images"),
    "/kaggle/input/train_images",
    "/kaggle/data/train_images",
)
TEST_IMG_DIR = resolve_path(
    os.path.join(BASE_INPUT, "test_images"),
    "/kaggle/input/test_images",
    "/kaggle/data/test_images",
)

assert os.path.exists(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing test.csv at {TEST_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing train_images dir at {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing test_images dir at {TEST_IMG_DIR}"




## === cell 2
class BlindnessDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, test=False):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations.iloc[idx, 0] + ".png")
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            label = int(self.annotations.iloc[idx, 1])
            return image, label




## === cell 3
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 4
test_dataset = BlindnessDataset(TEST_CSV, TEST_IMG_DIR, transform=transform, test=True)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)



## === cell 5
model_name = "efficientnet_b0"
imagenet_model = timm.create_model(
    model_name, pretrained=True
)  # keep pretrained head intact
imagenet_model.to(device)
imagenet_model.eval()

print(f"Using model: {model_name} (pretrained=True, ImageNet head), device={device}")



## === cell 6
train_df = pd.read_csv(TRAIN_CSV)
train_dataset = BlindnessDataset(
    TRAIN_CSV, TRAIN_IMG_DIR, transform=transform, test=False
)
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

num_imagenet = 1000
sum_sev = torch.zeros(num_imagenet, device=device)
count_sev = torch.zeros(num_imagenet, device=device)

with torch.no_grad():
    for images, labels in tqdm(
        train_loader, desc="Calibrating from train (ImageNet probs -> severity)"
    ):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).float()

        logits1000 = imagenet_model(images)  # [B,1000]
        probs1000 = nn.functional.softmax(logits1000, dim=1)  # [B,1000]

        sum_sev += (probs1000 * labels[:, None]).sum(dim=0)
        count_sev += probs1000.sum(dim=0)

mean_sev = sum_sev / count_sev.clamp_min(1e-12)  # [1000]
mean_sev = mean_sev.clamp(0.0, 4.0)




## === cell 7
def quadratic_weighted_kappa(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5
) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape
    N = n_classes

    O = np.zeros((N, N), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < N and 0 <= b < N:
            O[a, b] += 1.0

    hist_true = O.sum(axis=1)
    hist_pred = O.sum(axis=0)
    E = np.outer(hist_true, hist_pred)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            W[i, j] = ((i - j) ** 2) / ((N - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (W * O).sum() / denom


def apply_thresholds(exp_sev: np.ndarray, th: np.ndarray) -> np.ndarray:
    exp_sev = np.asarray(exp_sev, dtype=np.float64)
    th = np.asarray(th, dtype=np.float64)
    pred = np.digitize(exp_sev, bins=th, right=False).astype(np.int64)
    return np.clip(pred, 0, 4)


def enforce_strictly_increasing(th: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    th = np.asarray(th, dtype=np.float64).copy()
    th.sort()
    for i in range(1, len(th)):
        if th[i] <= th[i - 1] + eps:
            th[i] = th[i - 1] + eps
    return th


train_exp_sev = []
train_labels = []

with torch.no_grad():
    for images, labels in tqdm(train_loader, desc="Compute train expected severity"):
        images = images.to(device, non_blocking=True)
        logits1000 = imagenet_model(images)
        probs1000 = nn.functional.softmax(logits1000, dim=1)
        exp_sev = (probs1000 * mean_sev[None, :]).sum(dim=1)  # [B]
        train_exp_sev.append(exp_sev.cpu().numpy())
        train_labels.append(labels.numpy())

train_exp_sev = np.concatenate(train_exp_sev, axis=0)
train_labels = np.concatenate(train_labels, axis=0).astype(np.int64)


def fit_thresholds_qwk_from_candidates(
    exp_sev: np.ndarray,
    y_true: np.ndarray,
    candidates: np.ndarray,
    init_th: np.ndarray | None = None,
) -> np.ndarray:
    exp_sev = np.asarray(exp_sev, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.float64)

    if init_th is None:
        init = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float64)
        th = np.array(
            [candidates[np.argmin(np.abs(candidates - t))] for t in init],
            dtype=np.float64,
        )
        th = np.sort(th)
    else:
        th = np.sort(np.asarray(init_th, dtype=np.float64))

    th = enforce_strictly_increasing(th)

    best_th = th.copy()
    best_k = quadratic_weighted_kappa(
        y_true, apply_thresholds(exp_sev, best_th), n_classes=5
    )

    for _ in range(6):
        improved = False
        for i in range(4):
            lo = candidates.min() if i == 0 else best_th[i - 1] + 1e-6
            hi = candidates.max() if i == 3 else best_th[i + 1] - 1e-6
            feasible = candidates[(candidates > lo) & (candidates < hi)]
            if feasible.size == 0:
                continue

            local_best = best_th[i]
            local_best_k = best_k
            for v in feasible:
                trial = best_th.copy()
                trial[i] = float(v)
                trial = enforce_strictly_increasing(trial)
                k = quadratic_weighted_kappa(
                    y_true, apply_thresholds(exp_sev, trial), n_classes=5
                )
                if k > local_best_k:
                    local_best_k = k
                    local_best = float(v)

            if local_best != best_th[i]:
                improved = True
            best_th[i] = local_best
            best_th = enforce_strictly_increasing(best_th)
            best_k = local_best_k

        if not improved:
            break

    return enforce_strictly_increasing(np.sort(best_th))


def stratified_kfold_indices(y: np.ndarray, n_splits: int = 5, seed: int = 42):
    y = np.asarray(y, dtype=np.int64)
    rng = np.random.RandomState(seed)
    idx_by_class = [np.where(y == c)[0] for c in range(5)]
    for arr in idx_by_class:
        rng.shuffle(arr)

    folds = [[] for _ in range(n_splits)]
    for c in range(5):
        cls_idx = idx_by_class[c]
        for i, ix in enumerate(cls_idx):
            folds[i % n_splits].append(int(ix))

    fold_indices = []
    all_idx = set(range(len(y)))
    for f in range(n_splits):
        val_idx = np.array(sorted(folds[f]), dtype=np.int64)
        train_idx = np.array(
            sorted(list(all_idx - set(val_idx.tolist()))), dtype=np.int64
        )
        fold_indices.append((train_idx, val_idx))
    return fold_indices


def rank_to_uniform(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.int64)
    ranks[order] = np.arange(len(x), dtype=np.int64)
    u = (ranks.astype(np.float64) + 0.5) / float(len(x))
    return u


def map_uniform_thresholds_to_expsev(uniform_th: np.ndarray, exp_sev_ref: np.ndarray):
    uniform_th = np.asarray(uniform_th, dtype=np.float64)
    exp_sev_ref = np.asarray(exp_sev_ref, dtype=np.float64)
    mapped = np.quantile(exp_sev_ref, uniform_th)
    return enforce_strictly_increasing(np.asarray(mapped, dtype=np.float64))


def fit_thresholds_qwk_continuous(
    exp_sev: np.ndarray,
    y_true: np.ndarray,
    init_th: np.ndarray,
    n_outer: int = 2,
    n_iter_golden: int = 24,
    eps: float = 1e-6,
) -> np.ndarray:
    exp_sev = np.asarray(exp_sev, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    th = enforce_strictly_increasing(np.asarray(init_th, dtype=np.float64))

    lo_global = float(np.quantile(exp_sev, 0.001))
    hi_global = float(np.quantile(exp_sev, 0.999))
    if (
        not np.isfinite(lo_global)
        or not np.isfinite(hi_global)
        or hi_global <= lo_global
    ):
        lo_global, hi_global = float(exp_sev.min()), float(exp_sev.max())

    phi = 0.6180339887498949  # golden ratio conjugate

    def score(th_vec: np.ndarray) -> float:
        return quadratic_weighted_kappa(
            y_true, apply_thresholds(exp_sev, th_vec), n_classes=5
        )

    best_k = score(th)

    for _ in range(n_outer):
        improved_any = False
        for i in range(4):
            lo = lo_global if i == 0 else th[i - 1] + eps
            hi = hi_global if i == 3 else th[i + 1] - eps
            if hi <= lo:
                continue

            a, b = lo, hi
            c = b - phi * (b - a)
            d = a + phi * (b - a)

            th_c = th.copy()
            th_c[i] = c
            th_c = enforce_strictly_increasing(th_c)
            f_c = score(th_c)

            th_d = th.copy()
            th_d[i] = d
            th_d = enforce_strictly_increasing(th_d)
            f_d = score(th_d)

            for _it in range(n_iter_golden):
                if b - a < 1e-5:
                    break
                if f_c < f_d:
                    a = c
                    c = d
                    f_c = f_d
                    d = a + phi * (b - a)

                    th_d = th.copy()
                    th_d[i] = d
                    th_d = enforce_strictly_increasing(th_d)
                    f_d = score(th_d)
                else:
                    b = d
                    d = c
                    f_d = f_c
                    c = b - phi * (b - a)

                    th_c = th.copy()
                    th_c[i] = c
                    th_c = enforce_strictly_increasing(th_c)
                    f_c = score(th_c)

            if f_c >= f_d:
                if f_c > best_k:
                    th[i] = c
                    th = enforce_strictly_increasing(th)
                    best_k = f_c
                    improved_any = True
            else:
                if f_d > best_k:
                    th[i] = d
                    th = enforce_strictly_increasing(th)
                    best_k = f_d
                    improved_any = True

        if not improved_any:
            break

    return enforce_strictly_increasing(th)


qs = np.linspace(0.01, 0.99, 81)
folds = stratified_kfold_indices(train_labels, n_splits=5, seed=42)

oof_exp = np.zeros_like(train_exp_sev, dtype=np.float64)
oof_counts = np.zeros_like(train_exp_sev, dtype=np.float64)

probs1000_all = []
with torch.no_grad():
    for images, _labels in tqdm(
        train_loader, desc="Cache train ImageNet probs (for OOF mapping)"
    ):
        images = images.to(device, non_blocking=True)
        logits1000 = imagenet_model(images)
        probs1000 = nn.functional.softmax(logits1000, dim=1).cpu().numpy()
        probs1000_all.append(probs1000)
probs1000_all = np.concatenate(probs1000_all, axis=0).astype(np.float64)  # [N,1000]

for fi, (tr_idx, va_idx) in enumerate(folds, start=1):
    sum_sev_f = np.zeros((num_imagenet,), dtype=np.float64)
    cnt_sev_f = np.zeros((num_imagenet,), dtype=np.float64)
    y_tr = train_labels[tr_idx].astype(np.float64)

    p_tr = probs1000_all[tr_idx]  # [n_tr,1000]
    sum_sev_f += (p_tr * y_tr[:, None]).sum(axis=0)
    cnt_sev_f += p_tr.sum(axis=0)
    mean_sev_f = sum_sev_f / np.clip(cnt_sev_f, 1e-12, None)
    mean_sev_f = np.clip(mean_sev_f, 0.0, 4.0)

    p_va = probs1000_all[va_idx]  # [n_va,1000]
    exp_va = (p_va * mean_sev_f[None, :]).sum(axis=1)  # [n_va]
    oof_exp[va_idx] += exp_va
    oof_counts[va_idx] += 1.0

    exp_tr = (p_tr * mean_sev_f[None, :]).sum(axis=1)

    th_f = fit_thresholds_qwk_from_candidates(
        exp_tr,
        train_labels[tr_idx],
        candidates=np.quantile(exp_tr, qs),
        init_th=None,
    )
    pred_va = apply_thresholds(exp_va, th_f)
    k_va = quadratic_weighted_kappa(train_labels[va_idx], pred_va, n_classes=5)
    print(f"Fold {fi} val QWK (OOF mapping): {k_va:.5f}, th={th_f}")

oof_exp = oof_exp / np.clip(oof_counts, 1e-12, None)

shift_grid = np.array(
    [-0.10, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10],
    dtype=np.float64,
)

oof_u = rank_to_uniform(oof_exp)
candidates_u = np.quantile(oof_u, qs)

oof_th_u = fit_thresholds_qwk_from_candidates(
    oof_u,
    train_labels,
    candidates=candidates_u,
    init_th=np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float64),
)

oof_pred_u = apply_thresholds(oof_u, oof_th_u)
oof_kappa_u = quadratic_weighted_kappa(train_labels, oof_pred_u, n_classes=5)
print(f"OOF thresholds in rank-uniform space: {oof_th_u}")
print(f"OOF QWK using rank-uniform thresholds (diagnostic): {oof_kappa_u:.5f}")

thresholds_from_u = map_uniform_thresholds_to_expsev(oof_th_u, train_exp_sev)

oof_th_direct = fit_thresholds_qwk_from_candidates(
    oof_exp,
    train_labels,
    candidates=np.quantile(oof_exp, qs),
    init_th=None,
)

alpha_grid = np.linspace(0.25, 0.95, 15)
best = {"k": -1.0, "alpha": None, "shift": None, "th": None}

for a in alpha_grid:
    base_th = enforce_strictly_increasing(
        a * thresholds_from_u + (1.0 - a) * oof_th_direct
    )

    local_candidates = np.unique(
        np.concatenate(
            [
                np.linspace(float(t - 0.12), float(t + 0.12), 73, dtype=np.float64)
                for t in base_th
            ],
            axis=0,
        )
    )

    for sh in shift_grid:
        exp_use = oof_exp + float(sh)

        th_ref = fit_thresholds_qwk_from_candidates(
            exp_use,
            train_labels,
            candidates=local_candidates,
            init_th=base_th,
        )
        th_ref = fit_thresholds_qwk_continuous(
            exp_use, train_labels, init_th=th_ref, n_outer=1, n_iter_golden=18
        )

        pred = apply_thresholds(exp_use, th_ref)
        k = quadratic_weighted_kappa(train_labels, pred, n_classes=5)

        if k > best["k"]:
            best = {
                "k": float(k),
                "alpha": float(a),
                "shift": float(sh),
                "th": th_ref.copy(),
            }

print(
    f"Chosen via OOF: alpha={best['alpha']:.3f}, shift={best['shift']:.3f}, OOF QWK={best['k']:.5f}"
)

alpha = best["alpha"]
shift = best["shift"]

thresholds = enforce_strictly_increasing(best["th"])


def make_local_candidates(
    exp_sev: np.ndarray, base_th: np.ndarray, radius: float = 0.10, n_points: int = 81
):
    exp_sev = np.asarray(exp_sev, dtype=np.float64)
    base_th = np.asarray(base_th, dtype=np.float64)
    lo_global = float(np.quantile(exp_sev, 0.005))
    hi_global = float(np.quantile(exp_sev, 0.995))
    grid = []
    for t in base_th:
        lo = max(lo_global, float(t - radius))
        hi = min(hi_global, float(t + radius))
        if hi <= lo:
            lo, hi = lo_global, hi_global
        grid.append(np.linspace(lo, hi, n_points, dtype=np.float64))
    return np.unique(np.concatenate(grid, axis=0))


exp_oof_shifted = oof_exp + float(shift)
local_candidates_oof = make_local_candidates(
    exp_oof_shifted, thresholds, radius=0.10, n_points=81
)

thresholds = fit_thresholds_qwk_from_candidates(
    exp_oof_shifted,
    train_labels,
    candidates=local_candidates_oof,
    init_th=thresholds,
)
thresholds = fit_thresholds_qwk_continuous(
    exp_oof_shifted,
    train_labels,
    init_th=thresholds,
    n_outer=2,
    n_iter_golden=24,
)

train_pred_diag = apply_thresholds(train_exp_sev + float(shift), thresholds)
train_kappa_diag = quadratic_weighted_kappa(train_labels, train_pred_diag, n_classes=5)
oof_pred_diag = apply_thresholds(exp_oof_shifted, thresholds)
oof_kappa_diag = quadratic_weighted_kappa(train_labels, oof_pred_diag, n_classes=5)

print(f"Final learned thresholds (OOF-optimized, shift-aware): {thresholds}")
print(f"Using global shift on exp_sev: {shift:.4f}")
print(f"Diagnostic Train QWK (+shift): {train_kappa_diag:.5f}")
print(f"Diagnostic OOF QWK (+shift): {oof_kappa_diag:.5f}")



## === cell 8
all_exp_sev = []

with torch.no_grad():
    for images in tqdm(test_loader, desc="Inference"):
        images = images.to(device, non_blocking=True)

        logits1000 = imagenet_model(images)  # [B,1000]
        probs1000 = nn.functional.softmax(logits1000, dim=1)  # [B,1000]

        exp_sev = (probs1000 * mean_sev[None, :]).sum(dim=1)  # [B]
        all_exp_sev.append(exp_sev.cpu().numpy())

all_exp_sev = np.concatenate(all_exp_sev, axis=0)

final_predictions = apply_thresholds(all_exp_sev + float(shift), thresholds).astype(int)



## === cell 9
test_ids = pd.read_csv(TEST_CSV)["id_code"].values
assert len(test_ids) == len(
    final_predictions
), "Prediction length mismatch with test ids."

submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

print(submission_df.head())
print(
    f"Wrote submission to: {os.path.abspath(submission_path)} with shape {submission_df.shape}"
)
