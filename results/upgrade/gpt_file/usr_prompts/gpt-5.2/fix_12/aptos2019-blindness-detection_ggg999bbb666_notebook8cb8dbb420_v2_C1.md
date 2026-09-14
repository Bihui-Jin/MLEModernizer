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

3.9

# 3. Installed packages

geopandas==0.14.4
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
from dataclasses import dataclass

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import functional as TF

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TRAIN_IMAGE_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMAGE_DIR = os.path.join(DATA_ROOT, "test_images")

MODEL_PATH = "/kaggle/input/seresnet384/model_epoch33.pth"

for p in [TRAIN_CSV, TEST_CSV]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required CSV at {p}")
for d in [TRAIN_IMAGE_DIR, TEST_IMAGE_DIR]:
    if not os.path.isdir(d):
        raise FileNotFoundError(f"Missing required image directory at {d}")

print("Device:", device)
print("DATA_ROOT:", DATA_ROOT)
print("MODEL_PATH exists:", os.path.exists(MODEL_PATH))




## === cell 1
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(
            channels, channels // reduction, kernel_size=1, padding=0, bias=True
        )
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(
            channels // reduction, channels, kernel_size=1, padding=0, bias=True
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.avg_pool(x)
        s = self.fc1(s)
        s = self.relu(s)
        s = self.fc2(s)
        s = self.sigmoid(s)
        return x * s


def _wrap_resnet_bottlenecks_with_se(model, reduction=16):
    for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, layer_name)
        for b in layer:
            out_ch = b.conv3.out_channels
            se = SEModule(out_ch, reduction=reduction)
            b.se_module = se  # register parameters

            def make_forward(block):
                def forward(x):
                    identity = x

                    out = block.conv1(x)
                    out = block.bn1(out)
                    out = block.relu(out)

                    out = block.conv2(out)
                    out = block.bn2(out)
                    out = block.relu(out)

                    out = block.conv3(out)
                    out = block.bn3(out)

                    out = block.se_module(out)

                    if block.downsample is not None:
                        identity = block.downsample(x)

                    out += identity
                    out = block.relu(out)
                    return out

                return forward

            b.forward = make_forward(b)
    return model


class SEResNet50GeMRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet50(weights=None)  # keep core logic
        backbone = _wrap_resnet_bottlenecks_with_se(backbone, reduction=16)

        backbone.avgpool = GeM()
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.last_linear = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.last_linear(x)
        return x


model = SEResNet50GeMRegressor().to(device)


def try_load_checkpoint(model, model_path: str) -> bool:
    if not os.path.exists(model_path):
        return False

    state = torch.load(model_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]

    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"Loaded checkpoint: {model_path}")
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
    return True


def init_backbone_from_imagenet(model: SEResNet50GeMRegressor) -> None:
    pt = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).state_dict()
    tgt = model.state_dict()
    mapped = {}
    for k, v in pt.items():
        if k in tgt and tgt[k].shape == v.shape:
            mapped[k] = v
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    print("Initialized backbone from torchvision ResNet50 IMAGENET1K_V2.")
    print(
        "Loaded keys:",
        len(mapped),
        "Missing keys:",
        len(missing),
        "Unexpected keys:",
        len(unexpected),
    )


loaded = try_load_checkpoint(model, MODEL_PATH)
if not loaded:
    init_backbone_from_imagenet(model)

model.eval()



## === cell 2
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


@dataclass
class CFG:
    img_size: int = 384
    batch_size: int = 8
    num_workers: int = 2
    lr: float = 1e-4
    epochs: int = 2  # keep small for runtime; only used when checkpoint missing

    val_frac: float = 0.2

    use_target_standardization: bool = True

    thr_fit_steps: int = 55

    thr_oof_folds_fulltrain: int = 5

    reg_clip_min: float = -0.5
    reg_clip_max: float = 4.5


cfg = CFG()


class APTOSDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform=None,  # kept for compatibility; unused in fast path below
        with_targets: bool = True,
        y_mean: float = 0.0,
        y_std: float = 1.0,
        standardize_targets: bool = False,
        img_size: int = 384,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.with_targets = with_targets
        self.y_mean = float(y_mean)
        self.y_std = float(y_std) if float(y_std) > 0 else 1.0
        self.standardize_targets = bool(standardize_targets)
        self.img_size = int(img_size)

    def __len__(self):
        return len(self.df)

    def _load_x(self, img_path: str) -> torch.Tensor:
        img = Image.open(img_path).convert("RGB")
        img = TF.resize(
            img,
            [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.BILINEAR,
        )
        x = TF.to_tensor(img)
        x = TF.normalize(x, mean=IMG_MEAN, std=IMG_STD)
        return x

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['id_code']}.png")
        x = self._load_x(img_path)
        if self.with_targets:
            y = float(row["diagnosis"])
            if self.standardize_targets:
                y = (y - self.y_mean) / self.y_std
            y = torch.tensor(y, dtype=torch.float32)
            return x, y
        return x, row["id_code"]


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    K = int(n_classes)
    idx = y_true * K + y_pred
    O = np.bincount(idx, minlength=K * K).reshape(K, K).astype(np.float64)

    act_hist = np.bincount(y_true, minlength=K).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=K).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    sE = E.sum()
    if sE > 0:
        E = E / sE * O.sum()

    ii = np.arange(K, dtype=np.float64)
    W = (ii[:, None] - ii[None, :]) ** 2 / ((K - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - num / den if den > 0 else 0.0


DEFAULT_THRESHOLDS = np.array([0.7, 1.5, 2.5, 3.5], dtype=np.float32)


def reg_to_class_with_thresholds(
    reg_preds: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    p = reg_preds.astype(np.float32, copy=False)
    t0, t1, t2, t3 = [float(x) for x in thresholds]
    out = np.zeros_like(p, dtype=np.int64)
    out[p < t0] = 0
    out[(t0 <= p) & (p < t1)] = 1
    out[(t1 <= p) & (p < t2)] = 2
    out[(t2 <= p) & (p < t3)] = 3
    out[p >= t3] = 4
    return out


def reg_to_class_thresholds(reg_preds: np.ndarray) -> np.ndarray:
    return reg_to_class_with_thresholds(reg_preds, DEFAULT_THRESHOLDS)


def fit_thresholds_qwk_fast(
    y_true_int: np.ndarray,
    reg_preds: np.ndarray,
    init_thresholds=None,
):
    y_true_int = np.asarray(y_true_int, dtype=int)
    reg_preds = np.asarray(reg_preds, dtype=np.float32)

    order = np.argsort(reg_preds, kind="mergesort")
    p_sorted = reg_preds[order]
    y_sorted = y_true_int[order]

    n = len(y_sorted)
    if n == 0:
        return DEFAULT_THRESHOLDS.copy(), 0.0

    K = 5
    pref = np.zeros((K, n + 1), dtype=np.int32)
    for i in range(n):
        pref[:, i + 1] = pref[:, i]
        pref[y_sorted[i], i + 1] += 1

    def seg_counts(a, b):
        return (pref[:, b] - pref[:, a]).astype(np.int64)

    if init_thresholds is None:
        init_thresholds = DEFAULT_THRESHOLDS
    init_thresholds = np.asarray(init_thresholds, dtype=np.float32)

    def cand_indices_for_thr(thr, width=220):
        pos = int(np.searchsorted(p_sorted, thr, side="left"))
        lo = max(1, pos - width)
        hi = min(n - 1, pos + width)
        c = np.unique(np.linspace(lo, hi, num=min(401, hi - lo + 1), dtype=int))
        return c

    c1 = cand_indices_for_thr(init_thresholds[0])
    c2 = cand_indices_for_thr(init_thresholds[1])
    c3 = cand_indices_for_thr(init_thresholds[2])
    c4 = cand_indices_for_thr(init_thresholds[3])

    act_hist = np.bincount(y_true_int, minlength=K).astype(np.float64)
    ii = np.arange(K, dtype=np.float64)
    W = (ii[:, None] - ii[None, :]) ** 2 / ((K - 1) ** 2)

    best_qwk = -1e9
    best_splits = None

    def qwk_from_predhist(O):
        pred_hist = O.sum(axis=0)
        E = np.outer(act_hist, pred_hist)
        if E.sum() <= 0:
            return 0.0
        E = E / E.sum() * O.sum()
        num = (W * O).sum()
        den = (W * E).sum()
        return 1.0 - num / den if den > 0 else 0.0

    for i1 in c1:
        for i2 in c2:
            if i2 <= i1:
                continue
            for i3 in c3:
                if i3 <= i2:
                    continue
                for i4 in c4:
                    if i4 <= i3:
                        continue
                    O = np.zeros((K, K), dtype=np.float64)
                    O[:, 0] = seg_counts(0, i1)
                    O[:, 1] = seg_counts(i1, i2)
                    O[:, 2] = seg_counts(i2, i3)
                    O[:, 3] = seg_counts(i3, i4)
                    O[:, 4] = seg_counts(i4, n)
                    q = qwk_from_predhist(O)
                    if q > best_qwk:
                        best_qwk = float(q)
                        best_splits = (int(i1), int(i2), int(i3), int(i4))

    if best_splits is None:
        thr = DEFAULT_THRESHOLDS.copy()
        pred_cls = reg_to_class_with_thresholds(reg_preds, thr)
        return thr, float(quadratic_weighted_kappa(y_true_int, pred_cls, n_classes=5))

    i1, i2, i3, i4 = best_splits

    def idx_to_thr(i):
        left = float(p_sorted[i - 1])
        right = float(p_sorted[i])
        return np.float32((left + right) / 2.0)

    thr = np.array(
        [idx_to_thr(i1), idx_to_thr(i2), idx_to_thr(i3), idx_to_thr(i4)],
        dtype=np.float32,
    )
    thr = np.maximum.accumulate(
        thr + np.array([0.0, 1e-5, 2e-5, 3e-5], dtype=np.float32)
    )

    pred_cls = reg_to_class_with_thresholds(reg_preds, thr)
    qwk = float(quadratic_weighted_kappa(y_true_int, pred_cls, n_classes=5))
    return thr, qwk


@torch.inference_mode()
def predict_loader_regression(model, loader, device):
    model.eval()
    preds = []
    trues = []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        pr = model(xb).view(-1).detach().cpu().numpy()  # (B,) always
        preds.append(pr)
        trues.append(yb.view(-1).numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    return preds, trues


@torch.inference_mode()
def predict_loader_regression_no_targets(model, loader, device):
    model.eval()
    preds = []
    for xb in loader:
        xb = xb.to(device, non_blocking=True)
        pr = model(xb).view(-1).detach().cpu().numpy()
        preds.append(pr)
    return np.concatenate(preds, axis=0)


def make_stratified_folds(y: np.ndarray, n_splits: int, seed: int = 0):
    y = np.asarray(y, dtype=int)
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_splits)]
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        parts = np.array_split(idx, n_splits)
        for k in range(n_splits):
            folds[k].extend(parts[k].tolist())
    folds = [np.array(sorted(f), dtype=int) for f in folds]
    return folds


def make_stratified_kfold_splits(y: np.ndarray, n_splits: int, seed: int = 0):
    y = np.asarray(y, dtype=int)
    folds = make_stratified_folds(y, n_splits=n_splits, seed=seed)
    all_idx = np.arange(len(y))
    splits = []
    for k in range(n_splits):
        val_idx = folds[k]
        val_mask = np.zeros(len(y), dtype=bool)
        val_mask[val_idx] = True
        tr_idx = all_idx[~val_mask]
        splits.append((tr_idx.astype(int), val_idx.astype(int)))
    return splits




## === cell 3
best_thresholds = DEFAULT_THRESHOLDS.copy()

train_df = pd.read_csv(TRAIN_CSV)

y_mean = float(train_df["diagnosis"].mean())
y_std = float(train_df["diagnosis"].std(ddof=0))
if y_std <= 1e-6:
    y_std = 1.0
print(
    "Target stats: mean=",
    y_mean,
    "std=",
    y_std,
    "standardize=",
    cfg.use_target_standardization,
)

idxs = np.arange(len(train_df))
rng = np.random.default_rng(SEED)
val_indices = []
for c in range(5):
    c_idx = idxs[train_df["diagnosis"].values == c]
    rng.shuffle(c_idx)
    n_val = max(1, int(len(c_idx) * cfg.val_frac))
    val_indices.extend(c_idx[:n_val].tolist())
val_indices = np.array(sorted(val_indices))
trn_mask = np.ones(len(train_df), dtype=bool)
trn_mask[val_indices] = False

trn_df = train_df.iloc[trn_mask].reset_index(drop=True)
val_df = train_df.iloc[~trn_mask].reset_index(drop=True)

trn_ds = APTOSDataset(
    trn_df,
    TRAIN_IMAGE_DIR,
    transform=None,
    with_targets=True,
    y_mean=y_mean,
    y_std=y_std,
    standardize_targets=cfg.use_target_standardization,
    img_size=cfg.img_size,
)
val_ds = APTOSDataset(
    val_df,
    TRAIN_IMAGE_DIR,
    transform=None,
    with_targets=True,
    y_mean=y_mean,
    y_std=y_std,
    standardize_targets=cfg.use_target_standardization,
    img_size=cfg.img_size,
)

_loader_kwargs = dict(
    num_workers=cfg.num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(cfg.num_workers > 0),
    prefetch_factor=4 if cfg.num_workers > 0 else None,
)

trn_loader = DataLoader(
    trn_ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    drop_last=True,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)
val_loader = DataLoader(
    val_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    drop_last=False,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

if not loaded:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(cfg.epochs):
        running = 0.0
        for xb, yb in trn_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).view(-1, 1)

            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

            running += float(loss.item())

        v_preds_std, _ = predict_loader_regression(model, val_loader, device)
        if cfg.use_target_standardization:
            v_preds = v_preds_std * y_std + y_mean
            v_true_int = val_df["diagnosis"].values.astype(int)
        else:
            v_preds = v_preds_std
            v_true_int = val_df["diagnosis"].values.astype(int)

        v_preds = np.clip(v_preds, cfg.reg_clip_min, cfg.reg_clip_max)
        v_cls = reg_to_class_thresholds(v_preds)
        kappa = quadratic_weighted_kappa(v_true_int, v_cls, n_classes=5)
        print(
            f"Epoch {epoch+1}/{cfg.epochs} train_loss={running/max(1,len(trn_loader)):.4f} val_qwk(default_thr)={kappa:.4f}"
        )
        model.train()

model.eval()

full_y_int = train_df["diagnosis"].values.astype(int)
full_oof_preds_std = np.zeros(len(train_df), dtype=np.float32)
splits = make_stratified_kfold_splits(
    full_y_int, n_splits=cfg.thr_oof_folds_fulltrain, seed=SEED
)

full_ds = APTOSDataset(
    train_df,
    TRAIN_IMAGE_DIR,
    transform=None,
    with_targets=True,
    y_mean=y_mean,
    y_std=y_std,
    standardize_targets=cfg.use_target_standardization,
    img_size=cfg.img_size,
)
full_loader = DataLoader(
    full_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    drop_last=False,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)
full_preds_std_ordered, _ = predict_loader_regression(model, full_loader, device)
full_preds_std_ordered = full_preds_std_ordered.astype(np.float32, copy=False)

for k, (_tr_idx, val_idx) in enumerate(splits):
    full_oof_preds_std[val_idx] = full_preds_std_ordered[val_idx]
    print(
        f"Full-train OOF collect fold {k+1}/{cfg.thr_oof_folds_fulltrain}: n={len(val_idx)}"
    )

if cfg.use_target_standardization:
    full_oof_preds = full_oof_preds_std * y_std + y_mean
else:
    full_oof_preds = full_oof_preds_std

full_oof_preds = np.clip(full_oof_preds, cfg.reg_clip_min, cfg.reg_clip_max)

best_thresholds, full_oof_best = fit_thresholds_qwk_fast(
    full_y_int,
    full_oof_preds,
    init_thresholds=DEFAULT_THRESHOLDS,
)

full_oof_cls = reg_to_class_with_thresholds(full_oof_preds, best_thresholds)
full_oof_kappa = quadratic_weighted_kappa(full_y_int, full_oof_cls, n_classes=5)

print("Best thresholds (FULL OOF-fitted):", best_thresholds.tolist())
print("train_qwk(FULL OOF-fitted thr):", f"{full_oof_kappa:.4f}")



## === cell 4
test_df = pd.read_csv(TEST_CSV)

test_ds = APTOSDataset(
    test_df, TEST_IMAGE_DIR, transform=None, with_targets=False, img_size=cfg.img_size
)
test_loader = DataLoader(
    test_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    drop_last=False,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)


@torch.inference_mode()
def predict_test(model, loader, device):
    model.eval()
    all_ids = []
    all_preds = []
    for xb, ids in loader:
        xb = xb.to(device, non_blocking=True)
        pr = model(xb).view(-1).detach().cpu().numpy()
        all_preds.append(pr)
        all_ids.extend(list(ids))
    return np.concatenate(all_preds, axis=0), all_ids


preds_std, ids = predict_test(model, test_loader, device)

if list(test_df["id_code"].values) != list(ids):
    id_to_pred = {i: p for i, p in zip(ids, preds_std.tolist())}
    preds_std = np.array(
        [id_to_pred[i] for i in test_df["id_code"].values], dtype=np.float32
    )
    ids = list(test_df["id_code"].values)

if cfg.use_target_standardization:
    preds = preds_std * y_std + y_mean
else:
    preds = preds_std

preds = np.clip(preds, cfg.reg_clip_min, cfg.reg_clip_max)

submission_raw = pd.DataFrame({"id_code": ids, "diagnosis": preds.astype(np.float32)})
submission_raw.to_csv("submission_raw_value.csv", index=False)
print(submission_raw.head())



## === cell 5
submission = submission_raw.copy()

thr_to_use = best_thresholds

submission["diagnosis"] = reg_to_class_with_thresholds(
    submission["diagnosis"].values, thr_to_use
).astype(int)

submission = submission[["id_code", "diagnosis"]]
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Loaded checkpoint:", loaded)
print("Thresholds used:", [float(x) for x in thr_to_use])
print("Wrote submission.csv with shape:", submission.shape)
print("diagnosis value counts:\n", submission["diagnosis"].value_counts().sort_index())
