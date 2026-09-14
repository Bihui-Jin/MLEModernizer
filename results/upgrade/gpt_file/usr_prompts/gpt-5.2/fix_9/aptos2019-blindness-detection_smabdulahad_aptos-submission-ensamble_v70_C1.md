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

# 5. Target score

0.8355883073049606

# 6. Current score

0.68197

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.38294) has done: 'Your code likely didn’t yield a Kaggle score because it depends on an external `.pth` file path that isn’t present in your environment, and the fallback training is too slow/heavy for the 600s constraint and also isn’t aligned to the QWK metric. I make two minimal, score-relevant changes: (1) remove the dependency on missing external weights by always running a fast, deterministic inference-only baseline using a pretrained `efficientnet_b0`, and (2) replace argmax with an expected-value-to-ordinal mapping (a common, minimal post-processing for ordinal targets) which typically improves QWK without changing the model architecture or loss. The rest of your pipeline (dataset, transforms, model family, inference loop, submission writing) stays intact, and it always write a valid `submission.csv`.'
- What this solution (achieved 0.02751) has done: 'Your current score gap to the target is large (0.38294 vs 0.83559), so we need a meaningful but still minimal change that keeps your core inference-only pretrained EfficientNet pipeline intact. The biggest score limiter is that the model head is randomly initialized because `num_classes=5` breaks loading ImageNet pretrained classifier weights, so predictions are essentially random; we keep EfficientNet-B0 but load it as pretrained with its default head and then map it to 5 classes using a fixed, deterministic projection built from ImageNet class indices (no training). To better match the ordinal QWK metric without changing your approach, we also switch the post-processing from “expected class index” on 5-way probabilities to an ordinal mapping using the expected severity computed from the projected 5-way distribution. These changes are fast, deterministic, and should move the score substantially upward toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.02815) has done: 'Your current score is far below the target, and the main reason is that the “ImageNet→5 bins” projection makes predictions essentially unrelated to DR severity. I keep your exact pretrained EfficientNet-B0 inference-only setup, but replace the projection with a minimal, label-aware calibration: compute a per-class (0–4) expected severity from the 1000-way probabilities using training-set mean severity for each ImageNet class (computed by running the same pretrained model once on the train images). Then we map the expected severity to {0..4} (same rounding/clipping semantics), which typically boosts QWK a lot while staying fast and preserving your core approach. The rest (paths, dataset, transforms, inference loop, submission writing) stays intact and still produces `submission.csv`.'
- What this solution (achieved 0.66462) has done: 'Your current gap to the target is large, and the biggest score limiter is the final “round expected severity” step, which is not calibrated to maximize QWK (it tends to collapse predictions toward the majority class). I keep your exact pretrained EfficientNet inference + train-set calibration logic, but replace the naive rounding with a minimal, metric-aligned post-processing: learn 4 thresholds on the training set that map the continuous expected severity to 5 ordinal classes by directly maximizing quadratic weighted kappa. This doesn’t change the model, images, transforms, or inference loop; it only changes how the already-computed `exp_sev` is discretized. The thresholds are fit quickly (grid search) on the cached training expected severities and then applied to test, producing a valid `submission.csv` as before.'
- What this solution (achieved 0.67533) has done: 'I keep your exact pretrained EfficientNet-B0 inference + “ImageNet-class-to-mean-severity” calibration intact, but fix the main score limiter: fitting thresholds on the same data you evaluate them on (overfitting) and using a coarse quantile grid. I fit the 4 thresholds on an out-of-fold (OOF) split of the training expected severities (no change to model/transform), and then refit thresholds once on the full training set using that OOF-initialized solution. I also slightly densify the candidate threshold grid to improve discretization for QWK while staying fast. These changes only affect the post-processing discretization step and should move your public score upward toward the 0.8356 target.'
- What this solution (achieved 0.67832) has done: 'We keep your exact pretrained EfficientNet-B0 + ImageNet→severity calibration pipeline, and only make threshold fitting more metric-aligned and less fold-sensitive to move QWK upward toward the target. Specifically, we (1) fit thresholds on out-of-fold predictions (true OOF) instead of on in-fold expectations, (2) expand the candidate threshold grid slightly (still quantiles, still fast) to reduce discretization error, and (3) use a more robust aggregation of per-fold thresholds (median instead of mean) for initialization before the final full-train refinement. These are minimal post-processing changes only; the model, transforms, calibration, and inference loop remain unchanged and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.68197) has done: 'Your current score (0.67832) is far below the target (0.83559), so we should improve QWK with the smallest change that doesn’t alter your model/inference core. The biggest controllable lever in your pipeline is the discretization step: we fit thresholds in a way that directly optimizes QWK but generalizes better by using OOF expected severities and then doing a tighter local coordinate search around those OOF thresholds on the full train. This keeps the same EfficientNet/ImageNet-prob calibration and the same thresholding approach, but reduces discretization error from a coarse candidate grid. We also ensure thresholds remain strictly increasing and add a deterministic fallback if any threshold degeneracy occurs.'

# 9. Code solution

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


qs = np.linspace(0.01, 0.99, 81)
candidates_all = np.quantile(train_exp_sev, qs)

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

oof_th_init = fit_thresholds_qwk_from_candidates(
    oof_exp,
    train_labels,
    candidates=np.quantile(oof_exp, qs),
    init_th=None,
)

oof_pred = apply_thresholds(oof_exp, oof_th_init)
oof_kappa = quadratic_weighted_kappa(train_labels, oof_pred, n_classes=5)
print(f"OOF-fitted init thresholds: {oof_th_init}")
print(f"OOF QWK using OOF-fitted thresholds (diagnostic): {oof_kappa:.5f}")

thresholds = fit_thresholds_qwk_from_candidates(
    train_exp_sev,
    train_labels,
    candidates=candidates_all,
    init_th=oof_th_init,
)


def make_local_candidates(
    exp_sev: np.ndarray, base_th: np.ndarray, radius: float = 0.18, n_points: int = 61
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


local_candidates = make_local_candidates(
    train_exp_sev, thresholds, radius=0.18, n_points=61
)

thresholds = fit_thresholds_qwk_from_candidates(
    train_exp_sev,
    train_labels,
    candidates=local_candidates,
    init_th=thresholds,
)

train_pred = apply_thresholds(train_exp_sev, thresholds)
train_kappa = quadratic_weighted_kappa(train_labels, train_pred, n_classes=5)
print(f"Final learned thresholds: {thresholds}")
print(f"Train QWK (using thresholds on full train): {train_kappa:.5f}")



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

final_predictions = apply_thresholds(all_exp_sev, thresholds).astype(int)



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
