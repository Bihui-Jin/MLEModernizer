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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
from PIL import Image, ImageChops
import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def trim(im):
    """
    Keep: always return an image; prevents transforms failing on None.
    """
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -10)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

try:
    torch.use_deterministic_algorithms(True)
except Exception:
    torch.use_deterministic_algorithms(False)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False




## === cell 1
def regress2class(out, threshold):
    """
    Convert continuous regression output to ordinal classes using thresholds.
    threshold: list/tuple/np.array of 4 increasing floats.
    """
    if isinstance(out, torch.Tensor):
        if out.dim() == 2 and out.size(1) == 1:
            out = out.squeeze(1)
        out = out.detach().float().cpu().numpy()
    out = np.asarray(out).reshape(-1)

    thr = np.asarray(threshold, dtype=np.float32).reshape(-1)
    pred = np.zeros(out.shape[0], dtype=np.int64)
    for t in thr:
        pred += (out >= t).astype(np.int64)
    return pred


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - (num / den)


def optimize_thresholds(
    y_true, y_continuous, init_thresholds=None, n_iter=2, grid=None
):
    """
    Coordinate descent to maximize QWK on provided labels/predictions.
    Minimal, fast, deterministic; does not change model, only post-processing.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_continuous = np.asarray(y_continuous, dtype=np.float32)

    if init_thresholds is None:
        qs = [0.2, 0.4, 0.6, 0.8]
        init_thresholds = [float(np.quantile(y_continuous, q)) for q in qs]

    thr = np.array(init_thresholds, dtype=np.float32)
    thr.sort()

    best_thr = thr.copy()
    best_score = quadratic_weighted_kappa(y_true, regress2class(y_continuous, best_thr))

    if grid is None:
        grid = np.quantile(y_continuous, np.linspace(0.005, 0.995, 199)).astype(
            np.float32
        )
    else:
        grid = np.asarray(grid, dtype=np.float32).reshape(-1)

    for _ in range(n_iter):
        for k in range(4):
            local_best_t = best_thr[k]
            local_best_score = best_score

            low = -np.inf if k == 0 else (best_thr[k - 1] + 1e-4)
            high = np.inf if k == 3 else (best_thr[k + 1] - 1e-4)

            candidates = grid[(grid > low) & (grid < high)]
            if candidates.size == 0:
                continue

            for t in candidates:
                thr_try = best_thr.copy()
                thr_try[k] = t
                thr_try.sort()
                score = quadratic_weighted_kappa(
                    y_true, regress2class(y_continuous, thr_try)
                )
                if score > local_best_score:
                    local_best_score = score
                    local_best_t = t

            best_thr[k] = local_best_t
            best_thr.sort()
            best_score = local_best_score

    return best_thr.tolist(), float(best_score)




## === cell 2
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class Regressor(nn.Module):
    def __init__(self):
        super(Regressor, self).__init__()

        self.backbone = timm.models.tf_efficientnet_b5_ns(pretrained=False)
        if hasattr(self.backbone, "reset_classifier"):
            self.backbone.reset_classifier(0)
        self.backbone.global_pool = GeM(flatten=True)
        in_features = getattr(self.backbone, "num_features", 1000)
        self.regressor = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None, pretrained_backbone=False):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.models.tf_efficientnet_b4_ns(
            pretrained=pretrained_backbone
        )
        if hasattr(self.backbone, "reset_classifier"):
            self.backbone.reset_classifier(0)
        self.backbone.global_pool = GeM(flatten=True)

        in_features = getattr(self.backbone, "num_features", 1000)

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )

        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )

        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features, 500),
            nn.SiLU(),
            nn.Linear(500, 4),
        )

        self.final_regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x, final=False):
        x = self.backbone(x)

        c_out = self.classifier(x)
        r_out = self.regressor(x)
        o_out = self.ordinal(x)

        if final:
            out = torch.cat((c_out, r_out, o_out), 1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 4.5
            return out
        else:
            r_out = torch.sigmoid(r_out) * 4.5
            o_out = torch.sigmoid(o_out)
            return c_out, r_out, o_out


def _clean_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        if nk.startswith("net."):
            nk = nk[len("net.") :]
        cleaned[nk] = v
    return cleaned




## === cell 3
DATA_DIR = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

train_df = pd.read_csv(TRAIN_CSV)
train_ids = train_df["id_code"].astype(str).values
train_y = train_df["diagnosis"].astype(int).values

WEIGHTS_CANDIDATES = [
    "../input/weights/B4_3stage_75epoch.pkl",
    "../input/weights/B4_3stage_75epoch.pth",
    "../input/aptos2019-blindness-detection/B4_3stage_75epoch.pkl",
    "../input/aptos2019-blindness-detection/B4_3stage_75epoch.pth",
]

net = ThreeStage_Model(pretrained_backbone=False)
loaded = False
loaded_path = None

for wpath in WEIGHTS_CANDIDATES:
    if os.path.exists(wpath):
        state = torch.load(wpath, map_location="cpu")
        state = _clean_state_dict_keys(state)
        net.load_state_dict(state, strict=False)
        loaded = True
        loaded_path = wpath
        break

if not loaded:
    net = ThreeStage_Model(pretrained_backbone=True)

net = net.to(device)
net.eval()

try:
    from timm.data import resolve_data_config, create_transform

    _cfg = resolve_data_config({}, model=net.backbone)
    transform = create_transform(**_cfg, is_training=False)
    input_size = int(_cfg["input_size"][-1])
except Exception:
    input_size = 300
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
        ]
    )

print("Device:", device)
print("Weights loaded:", loaded, "| path:", loaded_path)
print("Transform input_size:", input_size)




## === cell 4
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

train_cont_oof = np.zeros(len(train_ids), dtype=np.float32)
missing_train_images = 0


def _predict_continuous_for_indices(indices):
    preds = np.zeros(len(indices), dtype=np.float32)
    miss = 0
    with torch.inference_mode():
        for j, i in enumerate(indices):
            idx = train_ids[i]
            image_name = os.path.join(TRAIN_IMG_DIR, f"{idx}.png")
            if not os.path.exists(image_name):
                miss += 1
                preds[j] = 0.0
                continue
            img = Image.open(image_name).convert("RGB")
            img = trim(img)
            img_t = transform(img).unsqueeze(0).to(device)
            out = net(img_t, final=True)  # (1,1) in [0,4.5]
            preds[j] = float(out.squeeze().detach().float().cpu().item())
    return preds, miss


for fold, (tr_idx, va_idx) in enumerate(skf.split(train_ids, train_y), start=1):
    va_preds, miss = _predict_continuous_for_indices(va_idx)
    missing_train_images += miss
    train_cont_oof[va_idx] = va_preds
    print(f"Fold {fold}: computed OOF preds for {len(va_idx)} samples | missing={miss}")

init_thresholds = [0.7, 1.5, 2.5, 3.5]
fold_thresholds = []
fold_scores = []

opt_thr = init_thresholds

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_ids, train_y), start=1):
    y_tr_cont = train_cont_oof[tr_idx]
    grid_tr = np.quantile(y_tr_cont, np.linspace(0.005, 0.995, 399)).astype(np.float32)

    thr_f, _ = optimize_thresholds(
        train_y[tr_idx],
        y_tr_cont,
        init_thresholds=init_thresholds,
        n_iter=6,
        grid=grid_tr,
    )
    qwk_f = quadratic_weighted_kappa(
        train_y[va_idx], regress2class(train_cont_oof[va_idx], thr_f)
    )
    fold_thresholds.append(thr_f)
    fold_scores.append(qwk_f)
    print(f"Fold {fold}: thr_fit_on_train={thr_f} | val_qwk={qwk_f:.6f}")

opt_thr = np.mean(np.array(fold_thresholds, dtype=np.float32), axis=0).tolist()
opt_thr = sorted(opt_thr)
oof_qwk = quadratic_weighted_kappa(train_y, regress2class(train_cont_oof, opt_thr))

print("Missing train images encountered:", missing_train_images)
print("Averaged (leakage-safer) thresholds:", opt_thr)
print("OOF QWK using averaged thresholds:", oof_qwk)
print("Mean fold VAL QWK:", float(np.mean(fold_scores)))




## === cell 5
submission_rows = []
missing_test_images = 0

with torch.inference_mode():
    for idx in test_ids:
        image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
        if not os.path.exists(image_name):
            missing_test_images += 1
            submission_rows.append([idx, 0])
            continue

        img = Image.open(image_name).convert("RGB")
        img = trim(img)
        img_t = transform(img).unsqueeze(0).to(device)

        out = net(img_t, final=True)  # (1,1) continuous in [0,4.5]
        pred = int(regress2class(out, opt_thr)[0])
        pred = max(0, min(4, pred))
        submission_rows.append([idx, pred])

submission = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])
print("Missing test images encountered:", missing_test_images)
print("Built submission dataframe:", submission.shape)

submission["id_code"] = submission["id_code"].astype(str)
submission["diagnosis"] = submission["diagnosis"].astype(int)
submission.to_csv("submission.csv", index=False)

print("Saved submission.csv with shape:", submission.shape)
print(submission.head())
print("diagnosis value counts:\n", submission["diagnosis"].value_counts().sort_index())
