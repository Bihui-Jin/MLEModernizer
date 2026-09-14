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

# 5. Target score

0.8925204558744121

# 6. Current score

0.65769

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.69312) has done: 'I remove the hard dependency on the missing external checkpoint by falling back to training the same SEResNet50+GeM regressor on the provided `train.csv`/images when the checkpoint isn’t available, so the notebook runs end-to-end and still produces `submission.csv`. To keep core logic intact, the model architecture, loss (MSE regression), and inference/post-processing thresholds are preserved; only the weight-loading path is made robust. I also add a simple train/val split and report a validation quadratic-weighted-kappa during training to ensure the pipeline is learning rather than outputting random classes. Finally, I ensure the submission has the correct columns, integer labels 0–4, and is written with a `.csv` suffix.'
- What this solution (achieved 0.64463) has done: 'You’re currently below the target (you mentioned ~0.693 vs target ~0.893), and the main “minimal-change” lever for QWK in this regression+fixed-threshold setup is calibrating the regression-to-class thresholds on a validation split instead of using hardcoded cutpoints. I keep your exact model, loss (MSE), training loop structure, and inference pipeline, but when training is required (checkpoint missing), I fit 4 thresholds on the validation set by maximizing QWK (coordinate-wise search) and then apply those thresholds to test predictions. If the external checkpoint is present and loads, we keep your original fixed thresholds to avoid unexpected drift (since we can’t validate the checkpoint’s calibration here). This should improve score toward the target while staying within your solution’s core logic and still writing a valid `submission.csv`.'
- What this solution (achieved 0.64463) has done: 'Your current score (0.64463) is far below the target (0.89252), so we should make a small, metric-aligned change that improves QWK without changing your model, loss, or training loop. The biggest issue is that when a checkpoint *is* loaded, you keep fixed thresholds that may be poorly calibrated; we instead fit thresholds on a validation split in both cases (loaded or not) using the same QWK-maximizing search you already use. To keep the core logic intact, we won’t touch the architecture, transforms, MSE regression, or inference; we only add a lightweight validation prediction pass + threshold fitting when `loaded=True`. This typically improves QWK substantially because QWK is very sensitive to the regression-to-class discretization.'
- What this solution (achieved 0.65769) has done: 'Your current score (0.64463) is well below the target (0.89252), so we should make small, metric-aligned fixes without changing the model or training approach. The biggest likely issue is a mismatch between training targets (0–4) and the regression head scale: with MSE regression, it helps a lot to standardize targets during training and then un-standardize predictions before threshold fitting and test inference, while keeping the exact same architecture/loss/loop. Additionally, your current val-threshold fitting is limited by a very small, stratified val set; we keep the same split logic but increase `val_frac` modestly and fit thresholds on out-of-fold-style predictions by using the whole val set after training (no extra training tricks), which usually improves QWK calibration. Finally, we ensure inference uses a proper DataLoader (instead of per-batch PIL loop) for consistent transforms/performance and to reduce accidental ordering/IO issues, while keeping prediction semantics identical.'
- What this solution (achieved 0.64728) has done: 'Your current score (0.65769) is far below the target (0.89252), so the most “minimal but metric-aligned” improvement is to keep your exact regressor+MSE setup but improve how thresholds are calibrated. I (1) fix a subtle bug in `predict_loader_regression` that can mis-handle the last batch shape and corrupt predictions/threshold fitting, and (2) replace the single holdout threshold fit with a small stratified K-fold out-of-fold (OOF) threshold fit (no model/loop changes), then average fold thresholds for a more stable discretization that typically raises QWK. This keeps architecture, transforms, loss, and training/inference semantics intact; it only improves the regression→class mapping calibration that QWK is highly sensitive to. The script still run end-to-end within time and write a valid `submission.csv`.'
- What this solution (achieved 0.65769) has done: 'Your current score (0.64728) is well below the target (0.89252), so we should make a small metric-aligned improvement without changing your model, loss, or training loop. The biggest issue is that your “K-fold” thresholding is not true out-of-fold calibration (it fits thresholds on subsets, then averages), which is noisy and can hurt QWK; we instead compute OOF predictions for the entire validation split and fit one set of thresholds on those OOF preds. This keeps the exact same regressor outputs and only changes the regression→class discretization calibration step (the most sensitive lever for QWK here). We also add a tiny safety fix to ensure validation targets are always treated as integers (no rounding artifacts) during threshold fitting/evaluation.'

# 9. Code solution

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
tfm = transforms.Compose(
    [
        transforms.Resize(
            (384, 384), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


@dataclass
class CFG:
    img_size: int = 384
    batch_size: int = 8
    num_workers: int = 2
    lr: float = 1e-4
    epochs: int = 2  # keep small for runtime; only used when checkpoint missing

    val_frac: float = 0.2

    use_target_standardization: bool = True

    thr_k_folds: int = 5
    thr_fit_steps: int = 55


cfg = CFG()


class APTOSDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        transform=None,
        with_targets: bool = True,
        y_mean: float = 0.0,
        y_std: float = 1.0,
        standardize_targets: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.with_targets = with_targets
        self.y_mean = float(y_mean)
        self.y_std = float(y_std) if float(y_std) > 0 else 1.0
        self.standardize_targets = bool(standardize_targets)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.image_dir, f"{row['id_code']}.png")
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img) if self.transform is not None else img
        if self.with_targets:
            y = float(row["diagnosis"])
            if self.standardize_targets:
                y = (y - self.y_mean) / self.y_std
            y = torch.tensor(y, dtype=torch.float32)
            return x, y
        return x, row["id_code"]


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        O[a, b] += 1.0
    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    E = E / E.sum() * O.sum()

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

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


def fit_thresholds_qwk(
    y_true_int: np.ndarray, reg_preds: np.ndarray, init_thresholds=None, n_steps=50
):
    y_true_int = np.asarray(y_true_int, dtype=int)
    reg_preds = np.asarray(reg_preds, dtype=np.float32)

    if init_thresholds is None:
        thr = DEFAULT_THRESHOLDS.copy()
    else:
        thr = np.array(init_thresholds, dtype=np.float32).copy()

    ranges = [
        (max(-0.5, thr[0] - 1.25), thr[0] + 1.25),
        (thr[1] - 1.25, thr[1] + 1.25),
        (thr[2] - 1.25, thr[2] + 1.25),
        (thr[3] - 1.25, thr[3] + 1.25),
    ]

    def score(thresholds):
        pred_cls = reg_to_class_with_thresholds(reg_preds, thresholds)
        return quadratic_weighted_kappa(y_true_int, pred_cls, n_classes=5)

    best = score(thr)

    for _ in range(4):
        for i in range(4):
            lo, hi = ranges[i]
            grid = np.linspace(lo, hi, n_steps, dtype=np.float32)
            best_i = thr[i]
            for v in grid:
                cand = thr.copy()
                cand[i] = float(v)
                if not (cand[0] < cand[1] < cand[2] < cand[3]):
                    continue
                s = score(cand)
                if s > best:
                    best = s
                    best_i = float(v)
                    thr = cand
            thr[i] = best_i

        ranges = [
            (thr[0] - 0.4, thr[0] + 0.4),
            (thr[1] - 0.4, thr[1] + 0.4),
            (thr[2] - 0.4, thr[2] + 0.4),
            (thr[3] - 0.4, thr[3] + 0.4),
        ]

    return thr.astype(np.float32), float(best)


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
    transform=tfm,
    with_targets=True,
    y_mean=y_mean,
    y_std=y_std,
    standardize_targets=cfg.use_target_standardization,
)
val_ds = APTOSDataset(
    val_df,
    TRAIN_IMAGE_DIR,
    transform=tfm,
    with_targets=True,
    y_mean=y_mean,
    y_std=y_std,
    standardize_targets=cfg.use_target_standardization,
)

trn_loader = DataLoader(
    trn_ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
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

        v_preds_std, v_true_std = predict_loader_regression(model, val_loader, device)
        if cfg.use_target_standardization:
            v_preds = v_preds_std * y_std + y_mean
            v_true_int = val_df["diagnosis"].values.astype(int)
        else:
            v_preds = v_preds_std
            v_true_int = val_df["diagnosis"].values.astype(int)

        v_cls = reg_to_class_thresholds(v_preds)
        kappa = quadratic_weighted_kappa(v_true_int, v_cls, n_classes=5)
        print(
            f"Epoch {epoch+1}/{cfg.epochs} train_loss={running/max(1,len(trn_loader)):.4f} val_qwk(default_thr)={kappa:.4f}"
        )
        model.train()

model.eval()

val_y_int = val_df["diagnosis"].values.astype(int)
folds = make_stratified_folds(val_y_int, n_splits=cfg.thr_k_folds, seed=SEED)

oof_preds_std = np.zeros(len(val_df), dtype=np.float32)

for k, fold_idx in enumerate(folds):
    fold_sub_df = val_df.iloc[fold_idx].reset_index(drop=True)
    fold_ds = APTOSDataset(
        fold_sub_df,
        TRAIN_IMAGE_DIR,
        transform=tfm,
        with_targets=True,
        y_mean=y_mean,
        y_std=y_std,
        standardize_targets=cfg.use_target_standardization,
    )
    fold_loader = DataLoader(
        fold_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    f_preds_std, _ = predict_loader_regression(model, fold_loader, device)
    oof_preds_std[fold_idx] = f_preds_std.astype(np.float32, copy=False)
    print(f"OOF collect fold {k+1}/{cfg.thr_k_folds}: n={len(fold_idx)}")

if cfg.use_target_standardization:
    oof_preds = oof_preds_std * y_std + y_mean
else:
    oof_preds = oof_preds_std

best_thresholds, oof_best = fit_thresholds_qwk(
    val_y_int,
    oof_preds,
    init_thresholds=DEFAULT_THRESHOLDS,
    n_steps=cfg.thr_fit_steps,
)

best_thresholds = np.maximum.accumulate(
    best_thresholds + np.array([0.0, 1e-5, 2e-5, 3e-5], dtype=np.float32)
)

oof_cls = reg_to_class_with_thresholds(oof_preds, best_thresholds)
oof_kappa = quadratic_weighted_kappa(val_y_int, oof_cls, n_classes=5)

print("Best thresholds (OOF-fitted):", best_thresholds.tolist())
print("val_qwk(OOF-fitted thr):", f"{oof_kappa:.4f}")




## === cell 4
test_df = pd.read_csv(TEST_CSV)

test_ds = APTOSDataset(test_df, TEST_IMAGE_DIR, transform=tfm, with_targets=False)
test_loader = DataLoader(
    test_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)


@torch.inference_mode()
def predict_test(model, loader, device):
    model.eval()
    all_ids = []
    all_preds = []
    for xb, ids in loader:
        xb = xb.to(device, non_blocking=True)
        pr = model(xb).view(-1).detach().cpu().numpy()  # keep shape stable
        all_preds.append(pr)
        all_ids.extend(list(ids))
    return np.concatenate(all_preds, axis=0), all_ids


preds_std, ids = predict_test(model, test_loader, device)

if cfg.use_target_standardization:
    preds = preds_std * y_std + y_mean
else:
    preds = preds_std

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
