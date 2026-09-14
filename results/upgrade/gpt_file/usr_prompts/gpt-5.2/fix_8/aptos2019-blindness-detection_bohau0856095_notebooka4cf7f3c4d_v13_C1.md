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

# 5. Target score

0.9103041346388836

# 6. Current score

0.1571

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02258) has done: 'I fix the missing weights issue by loading the model checkpoint only if it exists; otherwise, the script fall back to using the defined `Model` with ImageNet pretrained backbone weights so it can run end-to-end. I also fix runtime bugs (`Parameter` not imported, hard-coded `cuda:0` when GPU may be unavailable, and unsafe `.data` usage / scalar conversion) and ensure `test_ids` is read as a list of strings. Finally, I make the image path resolution robust to either `../input/...` or the provided `/kaggle/data/...` layout, and guarantee a valid `submission.csv` with the exact required columns is written.'
- What this solution (achieved -0.06179) has done: 'The main runtime error is a feature-dimension mismatch: `timm`’s `tf_efficientnet_b4_ns` returns 1000-d logits by default, but your `regressor` expects 1792 features. I fix this by creating the backbone with `num_classes=0` so it outputs pooled features, keeping your GeM pooling and regressor logic intact. I also make inference robust so a single unreadable image can’t abort the loop (which caused the empty submission), and I guarantee the written CSV is non-empty and aligned to `test.csv`’s `id_code` order. These changes are correctness/stability focused and should also improve score versus the accidental “linear on logits” behavior.'
- What this solution (achieved 0.10929) has done: 'Your current score is far below the target, so the most likely issue is not the backbone anymore but the *post-processing to discrete classes* for QWK. I keep your model exactly as-is and only (1) ensure EfficientNet runs in inference mode deterministically, (2) collect the raw regression outputs on the test set, and (3) replace the fixed hard-coded thresholds with *distribution-calibrated thresholds* derived from the training label distribution (a common minimal fix for this competition that usually improves QWK). This preserves your “regression -> thresholds -> class” core logic, but aligns the predicted class frequencies to the dataset’s expected ordinal distribution, which typically moves QWK up substantially. The script still run end-to-end and write a valid `submission.csv` with the required columns in `test.csv` order.'
- What this solution (achieved 0.03801) has done: 'Your score is far below the target, so we should improve toward it without changing the model/training core logic (there is no training here). The biggest lever consistent with your current “regression → thresholds → class” pipeline is to compute thresholds in a way that better matches the QWK/ordinal nature: instead of using test-prediction quantiles (which can be unstable), we fit thresholds on the training set by minimizing negative quadratic weighted kappa (classic APTOS fix) using the model’s own train predictions. This keeps the architecture, preprocessing, and regression head identical, but makes the discretization step metric-aligned and usually yields a large QWK gain. We also make inference more efficient/stable by batching and ensuring strict `id_code` order in the written submission.'
- What this solution (achieved 0.21984) has done: 'Your current score is far below the target, so the most likely remaining issue is that the “fit thresholds on train predictions” step is too weak/unstable, leaving your discrete class mapping poorly aligned with QWK. I keep your exact model and inference logic, but (1) make threshold fitting more robust by optimizing thresholds with a simple coordinate-descent over QWK using a better initialization (train label distribution + raw prediction quantiles) and more iterations, and (2) compute thresholds on out-of-fold (OOF) train predictions via a minimal K-fold split to avoid overly-optimistic/overfit thresholds. This preserves your “regression → thresholds → class” core approach and should materially improve QWK without changing architecture, loss, or training loops. The script still run end-to-end and write a valid `submission.csv` in `test.csv` order.'
- What this solution (achieved 0.1571) has done: 'Your gap to the target is large (0.21984 → 0.9103), so the biggest “minimal-change” lever is the discretization step for QWK. I keep your model and inference exactly the same, but change threshold fitting to be (1) truly out-of-fold (each fold’s predictions come from a separately initialized model so they vary, instead of the same fixed ImageNet model repeated 5x), and (2) optimized with a small grid-search refinement around the best coordinate-descent thresholds to better align with QWK. This avoids changing architecture/training loops (there is still no training), but makes the threshold calibration meaningful and typically yields a large QWK jump for APTOS. The script still run end-to-end and write a valid `submission.csv` with `id_code,diagnosis` in `test.csv` order.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import timm
from torch.nn import Parameter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CANDIDATE_ROOTS = [
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data/input/aptos2019-blindness-detection",
]
DATA_ROOT = next((p for p in CANDIDATE_ROOTS if os.path.exists(p)), None)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find aptos2019-blindness-detection directory in expected locations."
    )

TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).tolist()

train_df = pd.read_csv(TRAIN_CSV)
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["diagnosis"] = train_df["diagnosis"].astype(int)

transforms_eval = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)
tranforms = transforms_eval  # keep backward-compatible name used below


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
            + "{:.4f}".format(self.p.detach().cpu().item())
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns",
            pretrained=True,
            num_classes=0,  # feature output
            global_pool="",  # we'll override with GeM below
        )
        self.backbone.global_pool = GeM(flatten=True)

        in_features = getattr(self.backbone, "num_features", None)
        if in_features is None:
            raise RuntimeError(
                "Backbone does not expose num_features; cannot build regressor."
            )
        self.regressor = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < num_classes and 0 <= b < num_classes:
            O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=num_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=num_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (W * O).sum() / denom


def thresholds_to_pred(raw, thr):
    thr = np.asarray(thr, dtype=np.float32)
    raw = np.asarray(raw, dtype=np.float32)
    return (raw[:, None] >= thr[None, :]).sum(axis=1).astype(np.int64)


def _init_thresholds_from_label_dist(raw, y, eps=1e-6):
    raw = np.asarray(raw, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    hist = np.bincount(y, minlength=5).astype(np.float64)
    frac = hist / max(hist.sum(), 1.0)
    cum = np.cumsum(frac)  # length 5, last is 1
    q = np.clip(cum[:-1], eps, 1 - eps)  # 4 thresholds

    return np.quantile(raw, q).astype(np.float32)


def fit_thresholds_qwk(raw_train, y_train, init_thr=None, iters=80):
    raw_train = np.asarray(raw_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)

    fixed = np.array([0.7, 1.5, 2.5, 3.5], dtype=np.float32)
    q_init = np.quantile(raw_train, [0.2, 0.4, 0.6, 0.8]).astype(np.float32)
    ld_init = _init_thresholds_from_label_dist(raw_train, y_train).astype(np.float32)

    if init_thr is None:
        init_thr = fixed
    else:
        init_thr = np.asarray(init_thr, dtype=np.float32)

    thr = (0.34 * fixed + 0.33 * q_init + 0.33 * ld_init).astype(np.float32)
    thr = 0.5 * thr + 0.5 * init_thr
    thr = np.sort(thr)

    eps = 1e-3
    best_thr = thr.copy()
    best_kappa = quadratic_weighted_kappa(
        y_train, thresholds_to_pred(raw_train, best_thr)
    )

    step = 0.5
    for _ in range(iters):
        improved = False
        for i in range(4):
            base = best_thr[i]
            for direction in (-1.0, 1.0):
                cand = best_thr.copy()
                cand[i] = base + direction * step

                if i > 0:
                    cand[i] = max(cand[i], cand[i - 1] + eps)
                if i < 3:
                    cand[i] = min(cand[i], cand[i + 1] - eps)

                k = quadratic_weighted_kappa(
                    y_train, thresholds_to_pred(raw_train, cand)
                )
                if k > best_kappa:
                    best_kappa = k
                    best_thr = cand
                    improved = True

        if not improved:
            step *= 0.5
            if step < 5e-3:
                break

    return best_thr.astype(np.float32), float(best_kappa)


def refine_thresholds_local_grid(raw_train, y_train, base_thr, span=0.25, step=0.05):
    raw_train = np.asarray(raw_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    base_thr = np.asarray(base_thr, dtype=np.float32)

    eps = 1e-3
    best_thr = base_thr.copy()
    best_k = quadratic_weighted_kappa(y_train, thresholds_to_pred(raw_train, best_thr))

    deltas = np.arange(-span, span + 1e-9, step, dtype=np.float32)

    for i in range(4):
        cur_best_thr_i = best_thr[i]
        for d in deltas:
            cand = best_thr.copy()
            cand[i] = cur_best_thr_i + float(d)
            if i > 0:
                cand[i] = max(cand[i], cand[i - 1] + eps)
            if i < 3:
                cand[i] = min(cand[i], cand[i + 1] - eps)
            k = quadratic_weighted_kappa(y_train, thresholds_to_pred(raw_train, cand))
            if k > best_k:
                best_k = k
                best_thr = cand

    return best_thr.astype(np.float32), float(best_k)


CANDIDATE_WEIGHT_PATHS = [
    "../input/weights/tf_efficientnet_b4_ns_regress.pth",
    "/kaggle/input/weights/tf_efficientnet_b4_ns_regress.pth",
    "/kaggle/data/input/weights/tf_efficientnet_b4_ns_regress.pth",
]
WEIGHT_PATH = next((p for p in CANDIDATE_WEIGHT_PATHS if os.path.exists(p)), None)


def _load_model():
    net = Model()
    if WEIGHT_PATH is not None:
        ckpt = torch.load(WEIGHT_PATH, map_location=device)
        if isinstance(ckpt, nn.Module):
            net = ckpt
        elif isinstance(ckpt, dict):
            state = ckpt.get("state_dict", ckpt)
            try:
                net.load_state_dict(state, strict=True)
            except RuntimeError:
                stripped = {}
                for k, v in state.items():
                    stripped[
                        k.replace("module.", "", 1) if k.startswith("module.") else k
                    ] = v
                net.load_state_dict(stripped, strict=False)

    net = net.to(device)
    net.eval()
    return net


torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_image_tensor(img_path):
    img = Image.open(img_path).convert("RGB")
    return tranforms(img)


def predict_raw(ids, img_dir, net, batch_size=16):
    raw = np.full((len(ids),), np.nan, dtype=np.float32)
    ok = np.zeros((len(ids),), dtype=bool)

    batch_t = []
    batch_idx = []

    with torch.no_grad():
        for i, idx in enumerate(ids):
            image_name = os.path.join(img_dir, f"{idx}.png")
            try:
                t = _load_image_tensor(image_name)
            except Exception:
                continue

            batch_t.append(t)
            batch_idx.append(i)

            if len(batch_t) >= batch_size:
                x = torch.stack(batch_t, dim=0).to(device, non_blocking=True)
                out = net(x).squeeze(1).detach().cpu().numpy().astype(np.float32)
                raw[np.array(batch_idx, dtype=np.int64)] = out
                ok[np.array(batch_idx, dtype=np.int64)] = True
                batch_t, batch_idx = [], []

        if batch_t:
            x = torch.stack(batch_t, dim=0).to(device, non_blocking=True)
            out = net(x).squeeze(1).detach().cpu().numpy().astype(np.float32)
            raw[np.array(batch_idx, dtype=np.int64)] = out
            ok[np.array(batch_idx, dtype=np.int64)] = True

    return raw, ok


def make_oof_raw_predictions(train_df, img_dir, n_splits=5, seed=1337, batch_size=16):
    ids = train_df["id_code"].tolist()
    y = train_df["diagnosis"].values.astype(np.int64)

    rng = np.random.RandomState(seed)
    idx = np.arange(len(ids))
    rng.shuffle(idx)

    folds = np.full(len(ids), -1, dtype=np.int64)
    for i, j in enumerate(idx):
        folds[j] = i % n_splits

    raw_oof = np.full((len(ids),), np.nan, dtype=np.float32)
    ok_oof = np.zeros((len(ids),), dtype=bool)

    for f in range(n_splits):
        torch.manual_seed(seed + f)
        np.random.seed(seed + f)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + f)

        net_f = _load_model()

        sel = np.where(folds == f)[0]
        fold_ids = [ids[i] for i in sel]
        raw_f, ok_f = predict_raw(fold_ids, img_dir, net=net_f, batch_size=batch_size)
        raw_oof[sel] = raw_f
        ok_oof[sel] = ok_f

        del net_f
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return raw_oof, ok_oof, y


raw_oof, ok_oof, y_train = make_oof_raw_predictions(
    train_df, TRAIN_IMG_DIR, n_splits=5, seed=1337, batch_size=16
)

valid_oof = np.isfinite(raw_oof) & ok_oof
if valid_oof.sum() < 200:
    calibrated_thr = np.array([0.7, 1.5, 2.5, 3.5], dtype=np.float32)
    oof_kappa = None
else:
    calibrated_thr, oof_kappa = fit_thresholds_qwk(
        raw_oof[valid_oof],
        y_train[valid_oof],
        init_thr=(0.7, 1.5, 2.5, 3.5),
        iters=80,
    )
    calibrated_thr, oof_kappa = refine_thresholds_local_grid(
        raw_oof[valid_oof],
        y_train[valid_oof],
        base_thr=calibrated_thr,
        span=0.25,
        step=0.05,
    )

net = _load_model()
raw_test, ok_test = predict_raw(test_ids, TEST_IMG_DIR, net=net, batch_size=16)

test_pred = np.zeros((len(test_ids),), dtype=np.int64)
finite_test = np.isfinite(raw_test)
if finite_test.any():
    test_pred[finite_test] = thresholds_to_pred(raw_test[finite_test], calibrated_thr)
test_pred = np.clip(test_pred, 0, 4).astype(np.int64)

print("Calibrated thresholds:", [float(x) for x in calibrated_thr])
if oof_kappa is not None:
    print("OOF Train QWK using calibrated thresholds (on valid OOF preds):", oof_kappa)
print("Test finite preds:", int(finite_test.sum()), "out of", len(test_ids))



## === cell 1
df = pd.DataFrame({"id_code": test_ids, "diagnosis": test_pred.astype(int)})
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

if df.empty:
    raise RuntimeError("Submission DataFrame is empty; inference produced no rows.")

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print(
    "diagnosis value counts:\n", df["diagnosis"].value_counts(dropna=False).sort_index()
)
