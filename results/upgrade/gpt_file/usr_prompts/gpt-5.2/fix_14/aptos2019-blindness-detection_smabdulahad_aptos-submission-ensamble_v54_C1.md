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

0.8486945958951383

# 6. Current score

0.38096

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.12364) has done: 'I fix the immediate runtime blocker by removing the dependency on missing external pretrained weights and instead load each model with `pretrained=True` from `timm`, keeping the same ensemble + softmax-weighted averaging logic. I also make the dataset image loading robust (force RGB and close files) to avoid PIL mode issues, and fix the empty-model-list case that caused `torch.cat()` to fail. Finally, I ensure the submission is created with the required columns and correct length, and written to `submission.csv` in the working directory.'
- What this solution (achieved 0.17746) has done: 'Your score is far below the target, and the main issue is that you’re using ImageNet-pretrained classifiers with random (untrained) 5-class heads (`num_classes=5`), so predictions are essentially noise for this task. To move the score toward the target without changing the core “ensemble + softmax weighted averaging” inference logic, the minimal effective fix is to use each model’s pretrained ImageNet head (keep `num_classes` default) and map the 1000-class probabilities down to 5 DR classes via a fixed, deterministic binning of the expected severity (computed from class index). This keeps the same inference semantics (softmax → weighted average → argmax), but makes the base probabilities non-random and usually much more structured than an untrained head. I also keep the current weights logic, but make it robust if a key is missing, and ensure the submission remains aligned and valid.'
- What this solution (achieved 0.18972) has done: 'Your current low score is mainly driven by the ImageNet→5-class “hard binning” (one-hot) mapping, which destroys most ranking information that quadratic weighted kappa benefits from. I keep your exact ensemble + softmax-weighted averaging core logic, but change the mapping to a deterministic *soft* 5-bin aggregation over the 1000 ImageNet probabilities (summing probability mass into 5 contiguous bins) so the output remains a true probability distribution. Then I convert the final 5-class probabilities into an ordinal prediction using the expected severity (and rounding), which is a minimal post-processing aligned with the ordinal nature of the metric. These are small, inference-only changes that should move the score up toward your target without changing the overall approach.'
- What this solution (achieved 0.18605) has done: 'Your current approach is bottlenecked by an arbitrary ImageNet→5-class binning that doesn’t correspond to DR severity, so even with soft bins the model outputs remain largely misaligned with the ordinal target, keeping QWK low. To move the score upward toward 0.848 with minimal change and without altering your ensemble/inference structure, I keep the same “softmax → per-model weighting → average” logic but calibrate the final discrete prediction using tuned ordinal thresholds (a standard, metric-aligned post-processing for QWK). I also (safely) compute thresholds from your provided per-model validation scores to avoid hardcoding random values, and keep a fallback default if anything is missing. This keeps everything inference-only, fast, and produces the same required `submission.csv`.'
- What this solution (achieved 0.29885) has done: 'Your current gap to the target is large, and the biggest lever that doesn’t change your core “pretrained timm ensemble + softmax-weighted averaging” logic is to replace the arbitrary ImageNet→5 mapping with a deterministic ordinal mapping based on each ImageNet class index. Concretely, instead of summing contiguous 200-class bins, we compute an expected ImageNet class index (0–999) from the 1000-way probabilities and then map that scalar to DR labels using fixed quantile thresholds; this preserves more rank information, which QWK rewards. I keep your per-model weighting and your thresholding-to-label mechanism, but compute the thresholds from the 1000-class expectation quantiles (with a safe fallback), and ensure the submission remains aligned and valid. Changes are inference-only, fast, and keep the same overall semantics (softmax → weighted average → ordinal discretization).'
- What this solution (achieved 0.35722) has done: 'Your current pipeline is inference-only and can’t realistically reach the target because it never trains on the DR labels; however, we can still make a minimal, metric-aligned improvement by calibrating the ordinal thresholds more sensibly. Instead of deriving thresholds from *test* severity quantiles (which can yield an arbitrary class distribution), we compute thresholds from the *train* label distribution (same 0–4 ordinal target) by mapping those label quantiles into the predicted severity scale using the predicted severity’s own quantile function. This keeps your exact core logic (ImageNet pretrained models → softmax → expected index → weighted ensemble → thresholding) but makes the final discretization better matched to the competition’s label balance, which usually improves QWK versus naive test-quantile binning. We also keep a safe fallback to your previous method if anything unexpected happens.'
- What this solution (achieved 0.36385) has done: 'Your current score is far below the target because the pipeline never uses the DR labels to learn a mapping from ImageNet “severity index” to DR classes; the biggest minimal lever that preserves your core inference logic is to compute the 4 ordinal thresholds using the *training labels* in a way that maximizes quadratic weighted kappa on out-of-fold (OOF) train predictions, then apply those thresholds to test. This keeps the same core semantics (ImageNet-pretrained timm models → softmax → expected index → weighted ensemble → thresholding), but replaces the heuristic threshold selection with a metric-aligned, label-informed one. To avoid leakage, thresholds are fit on OOF predictions using a simple KFold, and the same thresholds are used for test. Changes are inference-only (no training of model weights), deterministic, and should move the score upward toward your target band.'
- What this solution (achieved 0.35207) has done: 'Your gap to the target is still large (>30%), so the smallest change with the biggest expected QWK gain is to keep your exact “ImageNet pretrained timm ensemble → expected index severity → ordinal thresholds” pipeline, but make the threshold fitting much closer to the evaluation metric by optimizing thresholds on OOF predictions with a continuous optimizer (Nelder–Mead) instead of a coarse coordinate/grid search. This does not change the model(s), transforms, or inference semantics; it only improves the 4 cutpoints that convert your scalar severity into {0,1,2,3,4}, which is exactly what QWK is sensitive to. I also fix a small correctness bug in `_fit_thresholds_qwk` where `sev` and `y_true` could be filtered to different lengths (currently it can silently misalign), which can hurt threshold fitting stability. Everything still runs end-to-end within the same paths and writes a valid `submission.csv`.'
- What this solution (achieved 0.36342) has done: 'Your current gap to the target is still large, so the smallest high-impact change (without changing your ensemble/models/transforms or “expected index → thresholds” core logic) is to make the threshold fitting *more robust and metric-aligned* by (1) using stratified folds instead of random splits (reduces instability in OOF threshold learning) and (2) fitting thresholds with a true QWK objective but also matching the *train label distribution* as a soft regularizer to avoid overfitting noisy OOF severities. I also add deterministic seeding/CUDA determinism to stabilize the fitted thresholds across runs (stability helps score). Finally, I keep all paths and the submission format identical, still writing `submission.csv` in the working directory.'
- What this solution (achieved 0.29637) has done: 'Your score is far below the target, and the main bottleneck is that the model never learns anything from the DR labels; the smallest high-impact change that keeps your exact inference core (ImageNet pretrained models → softmax → expected index → weighted ensemble → thresholds) is to fit the 4 thresholds more directly to maximize QWK on OOF predictions. I keep your stratified OOF setup and ensemble exactly as-is, but replace the threshold fitter with a fast dynamic-programming optimizer that finds the globally best monotone thresholds (on the OOF severities) for QWK, rather than a local Nelder–Mead search plus a distribution penalty. This is deterministic, runs fast (N≈3295), preserves your evaluation semantics, and should move QWK upward toward your target. Everything else (paths, transforms, models, submission writing) stays unchanged.'
- What this solution (achieved 0.38096) has done: 'The timeout is dominated by (1) the extremely expensive brute-force 4-cut threshold search (nested loops over ~3k items) and (2) unnecessary softmax + expected-index computation for every model/batch. I keep the exact same inference and “severity → thresholds → labels” pipeline, but make it compute-identical faster by (a) replacing softmax with a mathematically equivalent log-sum-exp formulation to get the same expected index without materializing 1000-class probabilities, and (b) replacing the QWK threshold brute-force with a provably equivalent dynamic-programming maximization of the same objective (same QWK, same search space of cut points), reducing it from O(n^4) to O(n^2). I also cache constant weight matrices/vectors, enable `torch.inference_mode()`, and use faster vectorized histogram construction in `_qwk` without changing semantics. These changes target the bottlenecks directly and preserve the model architectures, data pipeline, and evaluation logic.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm

import cv2




## === cell 1
def seed_everything(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(123)




## === cell 2
class BlindnessDataset(Dataset):
    def __init__(
        self,
        csv_file=None,
        root_dir=None,
        transform=None,
        test=False,
        id_codes=None,
        labels=None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.test = test

        if id_codes is not None:
            self.id_codes = np.asarray(id_codes)
            self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)
        else:
            annotations = pd.read_csv(csv_file)
            self.id_codes = annotations.iloc[:, 0].values
            self.labels = (
                None if test else annotations.iloc[:, 1].values.astype(np.int64)
            )

    def __len__(self):
        return len(self.id_codes)

    def __getitem__(self, idx):
        img_id = self.id_codes[idx]
        img_name = os.path.join(self.root_dir, f"{img_id}.png")

        img_bgr = cv2.imread(img_name, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"Could not read image: {img_name}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img_rgb)

        if self.transform:
            image = self.transform(image)

        if self.test:
            return image
        else:
            return image, int(self.labels[idx])




## === cell 3
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)



## === cell 4
test_csv_file = "/kaggle/input/aptos2019-blindness-detection/test.csv"
test_root_dir = "/kaggle/input/aptos2019-blindness-detection/test_images"
test_dataset = BlindnessDataset(
    csv_file=test_csv_file, root_dir=test_root_dir, transform=transform, test=True
)

_num_workers = min(8, (os.cpu_count() or 2))
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)



## === cell 5
model_paths = {"resnet18": None, "seresnext50_32x4d": None, "seresnext101_32x4d": None}

model_names = {
    "resnet18": "resnet18",
    "efficientnet_b5": "efficientnet_b5",
    "inception_resnet_v2": "inception_resnet_v2",
    "inception_v4": "inception_v4",
    "seresnext50_32x4d": "seresnext50_32x4d",
    "seresnext101_32x4d": "seresnext101_32x4d",
}



## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models_list = []
loaded_model_keys = []

for model_key in model_paths.keys():
    model_name = model_names[model_key]

    model = timm.create_model(model_name, pretrained=True)  # keep pretrained head
    model.to(device)
    model.eval()
    models_list.append(model)
    loaded_model_keys.append(model_key)

if len(models_list) == 0:
    raise RuntimeError("No models were loaded; cannot run inference.")



## === cell 7
validation_scores = {
    "resnet18": 0.887,
    "seresnext50_32x4d": 0.777,  # 0.709,
    "seresnext101_32x4d": 0.9697,  # 0.951
}

validation_scores = {
    k: v for k, v in validation_scores.items() if k in loaded_model_keys
}

if len(validation_scores) == 0:
    weights = {k: 1.0 / len(loaded_model_keys) for k in loaded_model_keys}
else:
    total_score = sum(validation_scores.values())
    weights = {k: v / total_score for k, v in validation_scores.items()}



## === cell 8
_imagenet_idx = torch.arange(1000, dtype=torch.float32, device=device)[
    None, :
]  # [1,1000]

_NUM_CLASSES = 5
_W_QWK = (
    (np.arange(_NUM_CLASSES)[:, None] - np.arange(_NUM_CLASSES)[None, :]) ** 2
).astype(np.float64) / ((_NUM_CLASSES - 1) ** 2)


def imagenet_logits_to_expected_index(logits_1000: torch.Tensor) -> torch.Tensor:
    b, c = logits_1000.shape
    if c != 1000:
        raise RuntimeError(f"Expected 1000-class logits, got shape {logits_1000.shape}")
    m = logits_1000.max(dim=1, keepdim=True).values
    ex = torch.exp(logits_1000 - m)  # [B,1000]
    num = (ex * _imagenet_idx).sum(dim=1)  # [B]
    den = ex.sum(dim=1)  # [B]
    return num / den


def severity_to_label_with_thresholds(
    sev: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    t0, t1, t2, t3 = thresholds.tolist()
    return np.digitize(sev, bins=[t0, t1, t2, t3]).astype(np.int64)


def _qwk(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 5) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    y_true = np.clip(y_true, 0, num_classes - 1)
    y_pred = np.clip(y_pred, 0, num_classes - 1)

    O = (
        np.bincount(y_true * num_classes + y_pred, minlength=num_classes * num_classes)
        .astype(np.float64)
        .reshape(num_classes, num_classes)
    )

    act_hist = np.bincount(y_true, minlength=num_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=num_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() == 0:
        return 0.0
    E = E / E.sum() * O.sum()

    if num_classes == 5:
        W = _W_QWK
    else:
        W = np.zeros((num_classes, num_classes), dtype=np.float64)
        for i in range(num_classes):
            for j in range(num_classes):
                W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    den = (W * E).sum()
    if den <= 0:
        return 0.0
    return 1.0 - (W * O).sum() / den


def _fit_thresholds_qwk(sev: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    sev = np.asarray(sev, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)

    mask = np.isfinite(sev) & (y_true >= 0) & (y_true <= 4)
    sev = sev[mask]
    y_true = y_true[mask]
    n = len(sev)
    if n == 0:
        return np.array([200.0, 400.0, 600.0, 800.0], dtype=np.float32)

    order = np.argsort(sev, kind="mergesort")
    y = y_true[order]
    sev_sorted = sev[order]

    pref = np.zeros((5, n + 1), dtype=np.int32)
    for k in range(5):
        pref[k, 1:] = np.cumsum((y == k).astype(np.int32))

    act_hist = pref[:, n].astype(np.float64)
    N = float(n)

    W = _W_QWK  # [5,5] float64
    A = (W.T @ act_hist).astype(np.float64)  # [5]

    wpref = W.T @ pref.astype(np.float64)  # [5, n+1]
    best_qwk = -1e18
    best_cuts = None

    dp1 = (wpref[0, 1:] - wpref[0, 0]).astype(np.float64)  # t=1..n

    dp2 = np.full(n + 1, np.inf, dtype=np.float64)
    arg2 = np.full(n + 1, -1, dtype=np.int32)
    dp3 = np.full(n + 1, np.inf, dtype=np.float64)
    arg3 = np.full(n + 1, -1, dtype=np.int32)
    dp4 = np.full(n + 1, np.inf, dtype=np.float64)
    arg4 = np.full(n + 1, -1, dtype=np.int32)

    for t in range(2, n):  # end index t
        best_val = np.inf
        best_s = -1
        wt = wpref[1, t]
        for s in range(1, t):
            val = dp1[s - 1] + (wt - wpref[1, s])
            if val < best_val:
                best_val = val
                best_s = s
        dp2[t] = best_val
        arg2[t] = best_s

    for t in range(3, n):
        best_val = np.inf
        best_s = -1
        wt = wpref[2, t]
        for s in range(2, t):
            val = dp2[s] + (wt - wpref[2, s])
            if val < best_val:
                best_val = val
                best_s = s
        dp3[t] = best_val
        arg3[t] = best_s

    for t in range(4, n):
        best_val = np.inf
        best_s = -1
        wt = wpref[3, t]
        for s in range(3, t):
            val = dp3[s] + (wt - wpref[3, s])
            if val < best_val:
                best_val = val
                best_s = s
        dp4[t] = best_val
        arg4[t] = best_s

    for c4 in range(4, n):
        num = dp4[c4] + (wpref[4, n] - wpref[4, c4])

        c3 = int(arg4[c4])
        if c3 < 3:
            continue
        c2 = int(arg3[c3])
        if c2 < 2:
            continue
        c1 = int(arg2[c2])
        if c1 < 1:
            continue

        pred_hist = np.array([c1, c2 - c1, c3 - c2, c4 - c3, n - c4], dtype=np.float64)
        den = (pred_hist * A).sum() / N
        if den <= 0:
            continue
        qwk = 1.0 - num / den
        if qwk > best_qwk:
            best_qwk = qwk
            best_cuts = (c1, c2, c3, c4)

    if best_cuts is None:
        qs = np.quantile(sev_sorted, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
        return np.maximum.accumulate(
            qs + np.array([0.0, 1e-3, 2e-3, 3e-3], dtype=np.float32)
        )

    c1, c2, c3, c4 = best_cuts

    def cut_to_thr(ci: int) -> float:
        if ci <= 0:
            return float(sev_sorted[0])
        if ci >= n:
            return float(sev_sorted[-1])
        return float(0.5 * (sev_sorted[ci - 1] + sev_sorted[ci]))

    thr = np.array(
        [cut_to_thr(c1), cut_to_thr(c2), cut_to_thr(c3), cut_to_thr(c4)],
        dtype=np.float64,
    )
    thr = np.clip(thr, 0.0, 999.0)
    thr = np.maximum.accumulate(
        thr + np.array([0.0, 1e-3, 2e-3, 3e-3], dtype=np.float64)
    )
    return thr.astype(np.float32)


def predict_severity_for_loader(loader: DataLoader) -> np.ndarray:
    all_sev = []
    with torch.inference_mode():
        for batch in tqdm(loader, leave=False):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                images = batch[0]
            else:
                images = batch
            images = images.to(device, non_blocking=True)

            sev_ens = None
            for model_key, model in zip(loaded_model_keys, models_list):
                logits_1000 = model(images)  # [B, 1000]
                sev = imagenet_logits_to_expected_index(logits_1000)  # [B]
                w = weights.get(model_key, 1.0 / len(loaded_model_keys))
                sev_w = w * sev
                sev_ens = sev_w if sev_ens is None else (sev_ens + sev_w)

            if sev_ens is None:
                raise RuntimeError("No model outputs produced for this batch.")

            all_sev.append(sev_ens.cpu().numpy())
    return np.concatenate(all_sev, axis=0)


train_csv_file = "/kaggle/input/aptos2019-blindness-detection/train.csv"
train_root_dir = "/kaggle/input/aptos2019-blindness-detection/train_images"
train_df = pd.read_csv(train_csv_file)
train_labels_full = train_df["diagnosis"].values.astype(np.int64)
train_ids_full = train_df["id_code"].values


def make_stratified_folds(y: np.ndarray, n_splits: int = 3, seed: int = 123):
    y = np.asarray(y, dtype=np.int64)
    rng = np.random.RandomState(seed)
    folds = [[] for _ in range(n_splits)]
    for cls in range(5):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        parts = np.array_split(cls_idx, n_splits)
        for i in range(n_splits):
            folds[i].extend(parts[i].tolist())
    folds = [np.array(sorted(f), dtype=np.int64) for f in folds]
    return folds


n_splits = 3
folds = make_stratified_folds(train_labels_full, n_splits=n_splits, seed=123)

train_ds_full = BlindnessDataset(
    root_dir=train_root_dir,
    transform=transform,
    test=False,
    id_codes=train_ids_full,
    labels=train_labels_full,
)
train_loader_full = DataLoader(
    train_ds_full,
    batch_size=16,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)

train_sev_full = predict_severity_for_loader(train_loader_full).astype(np.float32)
if len(train_sev_full) != len(train_df):
    raise RuntimeError("Train severity length mismatch.")

oof_sev = np.zeros(len(train_df), dtype=np.float32)
for fold_i in range(n_splits):
    val_idx = folds[fold_i]
    oof_sev[val_idx] = train_sev_full[val_idx]

thresholds = _fit_thresholds_qwk(oof_sev, train_labels_full)

severity = predict_severity_for_loader(test_loader)
final_predictions = (
    severity_to_label_with_thresholds(severity, thresholds).clip(0, 4).astype(int)
)

print("Fitted thresholds (OOF-QWK):", thresholds)



## === cell 9
test_ids = pd.read_csv(test_csv_file)["id_code"].values
if len(final_predictions) != len(test_ids):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(final_predictions)} preds for {len(test_ids)} ids"
    )

submission_df = pd.DataFrame({"id_code": test_ids, "diagnosis": final_predictions})
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Thresholds used:", thresholds)
print("Wrote submission.csv with shape:", submission_df.shape)
