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

0.8857740987904832

# 6. Current score

0.10302

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00825) has done: 'I fix the blocking issue where the expected checkpoint file `128best.pth` is not present by making the script robust: it use the checkpoint if found, otherwise it fall back to an ImageNet-pretrained backbone and still produce a valid `submission.csv`. I also fix the cascading `NameError` issues by ensuring `device` and `model` are always defined and that inference runs even in the fallback path. These changes keep the core model/inference/post-processing logic intact (same architecture shape and same thresholding), while guaranteeing end-to-end execution and a correctly formatted CSV submission.'
- What this solution (achieved 0.06854) has done: 'Your current score is extremely low because the script is almost certainly running in the ImageNet-fallback path (no competition-trained checkpoint), so the raw regression outputs are essentially uncalibrated for DR severity and the fixed thresholds produce near-random ordinal labels. To move the score substantially toward the target without changing the model or training loop, I (1) reliably locate and load a provided checkpoint if it exists in common Kaggle input locations, and (2) if we still must fall back, replace the hard-coded thresholds with a rank-based mapping that matches the public test label distribution estimated from `train.csv` (a standard, label-leak-free calibration for ordinal kappa). This preserves the same inference semantics (single scalar -> 5 ordinal classes) while making the post-processing robust and much closer to a sensible classifier for QWK. The output format and paths remain unchanged and a valid `submission.csv` is always written.'
- What this solution (achieved 0.08343) has done: 'Your score is far below the target, so we should cautiously increase performance without changing the core model/inference/training semantics. The biggest likely issue is that the ImageNet-fallback path produces nearly uninformative raw scores, and your current “match train distribution” mapping is too crude for QWK; we can improve calibration by learning optimal ordinal thresholds on a proper validation split (using the same scalar output and the same thresholding idea) and then applying those thresholds to test. This keeps the architecture identical and does not introduce any training loop changes (we only add a lightweight post-hoc threshold fit on train predictions), but it typically improves QWK a lot versus fixed or rank-only bins. We also make inference faster within the 600s limit by batching images in a DataLoader (same transforms, same model outputs), which doesn’t change results beyond negligible floating-point differences.'
- What this solution (achieved 0.07902) has done: 'Your score is far below the target, so we need a careful improvement that keeps your model and training approach intact while making the post-processing (thresholding) much better aligned with QWK. The biggest likely issue is that you are fitting thresholds on a validation split but then applying them directly to test without correcting for a scale/shift mismatch between train and test model outputs; for a single-output regression head this mismatch can be large and destroys ordinal thresholding. I add a minimal “distribution alignment” step: fit thresholds on validation predictions, but before applying them to test, linearly align test raw predictions to the validation prediction distribution (robust mean/std using percentiles) and then apply the same thresholds. This preserves your core logic (same model, same scalar output, same thresholding) and typically increases QWK substantially versus unaligned thresholds when using a non-perfect checkpoint (or ImageNet fallback).'
- What this solution (achieved 0.10542) has done: 'Your current score (0.079) is far below the target (0.886), so we need a genuine lift while keeping your model and training loop unchanged. The biggest issue is that you’re fitting thresholds on only a small validation subset, then applying them to test; for QWK this is unstable, and your linear “alignment” is too weak when the raw score distribution shifts. I keep the same scalar-output model and the same “apply thresholds” semantics, but (1) fit thresholds on **out-of-fold (OOF)** predictions over the whole training set (no leakage), and (2) apply a minimal **monotonic calibration** by mapping test predictions through the OOF CDF before thresholding. This is still just post-processing of the same raw scalar outputs, but it typically boosts ordinal QWK a lot versus single-split threshold fitting.'
- What this solution (achieved 0.10302) has done: 'Your current score is far below the target, so we should improve performance with the smallest changes that don’t alter your model or training loop. The biggest likely issue is a bug in your test→OOF ECDF mapping (you’re effectively mapping to itself), which can destroy the calibration and make thresholds behave poorly. I fix ECDF mapping to correctly map test predictions onto the OOF prediction distribution (monotonic quantile mapping), and I also ensure the OOF predictions are produced with the model in eval mode (BatchNorm/Dropout stability) before fitting thresholds. These are purely post-processing/inference-stability fixes that preserve the same scalar-output + thresholding semantics but should move QWK materially upward.'

# 9. Code solution

## === cell 0
import os
from glob import glob

import numpy as np
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms, models

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

try:
    os.makedirs("/kaggle/working", exist_ok=True)
    os.chdir("/kaggle/working")
except Exception:
    pass

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        p = self.p.data.tolist()[0]
        return f"{self.__class__.__name__}(p={p:.4f}, eps={self.eps})"


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class SEResNet50Like(nn.Module):
    """
    Torchvision ResNet50 backbone with GeM pooling and a single output head.
    Preserves core inference semantics (CNN -> pooling -> linear -> scalar).
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet50(weights=weights)

        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        self.avg_pool = GeM()
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

        x = self.avg_pool(x)  # (B, 2048, 1, 1)
        x = torch.flatten(x, 1)  # (B, 2048)
        x = self.last_linear(x)  # (B, 1)
        return x


def get_se_resnet50_gem(pretrain):
    if pretrain == "imagenet":
        return SEResNet50Like(pretrained=True)
    return SEResNet50Like(pretrained=False)




## === cell 2
def find_checkpoint():
    explicit = [
        "/kaggle/input/128best/128best.pth",
        "../input/128best/128best.pth",
        "/kaggle/input/aptos2019-blindness-detection/128best/128best.pth",
    ]
    for p in explicit:
        if os.path.exists(p):
            return p

    for p in glob("/kaggle/input/**/128best.pth", recursive=True):
        if os.path.exists(p):
            return p
    for p in glob("/kaggle/input/**/*.pth", recursive=True):
        if os.path.basename(p) == "128best.pth":
            return p

    return None


MODEL_PATH = find_checkpoint()

CANDIDATE_TEST_IMG_DIRS = [
    "/kaggle/input/aptos2019-blindness-detection/test_images",
    "/kaggle/data/aptos2019-blindness-detection/test_images",
    "../input/aptos2019-blindness-detection/test_images",
    "/kaggle/input/test_images",
    "/kaggle/data/test_images",
]
TEST_IMAGE_PATH = next((p for p in CANDIDATE_TEST_IMG_DIRS if os.path.isdir(p)), None)

CANDIDATE_TRAIN_IMG_DIRS = [
    "/kaggle/input/aptos2019-blindness-detection/train_images",
    "/kaggle/data/aptos2019-blindness-detection/train_images",
    "../input/aptos2019-blindness-detection/train_images",
    "/kaggle/input/train_images",
    "/kaggle/data/train_images",
]
TRAIN_IMAGE_PATH = next((p for p in CANDIDATE_TRAIN_IMG_DIRS if os.path.isdir(p)), None)

CANDIDATE_TEST_CSVS = [
    "/kaggle/input/aptos2019-blindness-detection/test.csv",
    "/kaggle/data/aptos2019-blindness-detection/test.csv",
    "../input/aptos2019-blindness-detection/test.csv",
    "/kaggle/input/test.csv",
    "/kaggle/data/test.csv",
]
TEST_CSV_PATH = next((p for p in CANDIDATE_TEST_CSVS if os.path.exists(p)), None)

CANDIDATE_TRAIN_CSVS = [
    "/kaggle/input/aptos2019-blindness-detection/train.csv",
    "/kaggle/data/aptos2019-blindness-detection/train.csv",
    "../input/aptos2019-blindness-detection/train.csv",
    "/kaggle/input/train.csv",
    "/kaggle/data/train.csv",
]
TRAIN_CSV_PATH = next((p for p in CANDIDATE_TRAIN_CSVS if os.path.exists(p)), None)

if TEST_IMAGE_PATH is None:
    raise FileNotFoundError(
        "Could not find test_images directory in expected Kaggle input paths."
    )
if TEST_CSV_PATH is None:
    raise FileNotFoundError("Could not find test.csv in expected Kaggle input paths.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
print("TEST_IMAGE_PATH:", TEST_IMAGE_PATH)
print("TRAIN_IMAGE_PATH:", TRAIN_IMAGE_PATH)
print("TEST_CSV_PATH:", TEST_CSV_PATH)
print("TRAIN_CSV_PATH:", TRAIN_CSV_PATH)
print("MODEL_PATH:", MODEL_PATH)




## === cell 3
use_imagenet_fallback = MODEL_PATH is None

model = get_se_resnet50_gem(pretrain="imagenet" if use_imagenet_fallback else None).to(
    device
)

if not use_imagenet_fallback:
    ckpt = torch.load(MODEL_PATH, map_location="cpu")

    if (
        isinstance(ckpt, dict)
        and "state_dict" in ckpt
        and isinstance(ckpt["state_dict"], dict)
    ):
        state_dict = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        state_dict = ckpt
    else:
        state_dict = ckpt

    new_state = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print("Loaded checkpoint. missing:", len(missing), "unexpected:", len(unexpected))
else:
    print(
        "WARNING: model checkpoint not found; using ImageNet-pretrained ResNet50 backbone fallback."
    )

model.eval()




## === cell 4
infer_tfms = transforms.Compose(
    [
        transforms.Resize(
            (128, 128), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


class FundusDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, tfms, has_label: bool):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfms = tfms
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_code = str(row["id_code"])
        im_path = os.path.join(self.img_dir, f"{id_code}.png")
        if not os.path.exists(im_path):
            candidates = glob(os.path.join(self.img_dir, f"{id_code}.*"))
            if len(candidates) == 0:
                raise FileNotFoundError(f"Missing image for id_code={id_code}")
            im_path = candidates[0]
        image = Image.open(im_path).convert("RGB")
        x = self.tfms(image)
        if self.has_label:
            y = int(row["diagnosis"])
            return id_code, x, y
        return id_code, x


def predict_df(df: pd.DataFrame, img_dir: str, batch_size: int = 16):
    model.eval()

    ds = FundusDataset(
        df, img_dir=img_dir, tfms=infer_tfms, has_label=("diagnosis" in df.columns)
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    ids = []
    preds = []
    ys = [] if "diagnosis" in df.columns else None
    with torch.no_grad():
        for batch in dl:
            if "diagnosis" in df.columns:
                b_ids, x, y = batch
                ys.extend(y.numpy().tolist())
            else:
                b_ids, x = batch
            x = x.to(device, non_blocking=True)
            out = model(x).squeeze(1).detach().cpu().numpy()
            ids.extend(list(b_ids))
            preds.extend(out.tolist())
    if ys is None:
        return pd.DataFrame({"id_code": ids, "pred": preds})
    return pd.DataFrame({"id_code": ids, "pred": preds, "diagnosis": ys})


test_df = pd.read_csv(TEST_CSV_PATH)
raw_test = predict_df(test_df, img_dir=TEST_IMAGE_PATH, batch_size=16)
raw_pred = raw_test.rename(columns={"pred": "diagnosis"})[["id_code", "diagnosis"]]
raw_pred.to_csv("submission_raw_value.csv", index=False)
raw_pred.head()




## === cell 5
def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape
    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0
    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    denom = (W * E).sum()
    if denom == 0:
        return 0.0
    return 1.0 - (W * O).sum() / denom


def apply_thresholds(pred, thr):
    p = np.asarray(pred, dtype=np.float64)
    t0, t1, t2, t3 = thr
    out = np.zeros(len(p), dtype=int)
    out[p >= t0] = 1
    out[p >= t1] = 2
    out[p >= t2] = 3
    out[p >= t3] = 4
    return out


def fit_thresholds_coordinate_descent(pred, y, init_thr=None, n_iter=20):
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=int)

    if init_thr is None:
        qs = [0.20, 0.45, 0.70, 0.85]
        init_thr = np.quantile(pred, qs).tolist()

    thr = np.array(init_thr, dtype=np.float64)

    grid = np.unique(np.quantile(pred, np.linspace(0.02, 0.98, 97))).astype(np.float64)
    best = quadratic_weighted_kappa(y, apply_thresholds(pred, thr))

    for _ in range(n_iter):
        improved = False
        for k in range(4):
            lo = -np.inf if k == 0 else thr[k - 1] + 1e-6
            hi = np.inf if k == 3 else thr[k + 1] - 1e-6
            candidates = grid[(grid > lo) & (grid < hi)]
            if len(candidates) == 0:
                continue
            local_best = best
            local_t = thr[k]
            for v in candidates:
                cand = thr.copy()
                cand[k] = v
                score = quadratic_weighted_kappa(y, apply_thresholds(pred, cand))
                if score > local_best + 1e-12:
                    local_best = score
                    local_t = v
            if local_best > best + 1e-12:
                thr[k] = local_t
                best = local_best
                improved = True
        if not improved:
            break
    return thr.tolist(), best


def make_stratified_folds(df, n_folds=5, seed=42):
    rng = np.random.RandomState(seed)
    df = df.copy()
    df["_rand"] = rng.uniform(size=len(df))
    fold = np.full(len(df), -1, dtype=int)

    for cls in [0, 1, 2, 3, 4]:
        idx = df.index[df["diagnosis"] == cls].to_numpy()
        idx = idx[np.argsort(df.loc[idx, "_rand"].to_numpy())]
        for i, j in enumerate(idx):
            fold[j] = i % n_folds

    if (fold < 0).any():
        remaining = np.where(fold < 0)[0]
        for i, j in enumerate(remaining):
            fold[j] = i % n_folds

    df["fold"] = fold
    df = df.drop(columns=["_rand"])
    return df


def ecdf_map_to_reference(test_pred, ref_pred):
    test_pred = np.asarray(test_pred, dtype=np.float64)
    ref_pred = np.asarray(ref_pred, dtype=np.float64)

    test_sorted = np.sort(test_pred)
    ref_sorted = np.sort(ref_pred)

    n_test = len(test_sorted)
    n_ref = len(ref_sorted)
    if n_test == 0 or n_ref == 0:
        return test_pred.copy()

    ranks = np.searchsorted(test_sorted, test_pred, side="left")
    q = ranks / max(1, n_test)  # [0,1)
    mapped = np.quantile(ref_sorted, np.clip(q, 0.0, 1.0))
    return mapped


submission = raw_pred.copy()

if (TRAIN_CSV_PATH is not None) and (TRAIN_IMAGE_PATH is not None):
    train_df = pd.read_csv(TRAIN_CSV_PATH)[["id_code", "diagnosis"]].copy()
    train_df = make_stratified_folds(train_df, n_folds=5, seed=42)

    oof_pred = np.zeros(len(train_df), dtype=np.float64)
    y_true = train_df["diagnosis"].values.astype(int)

    for f in range(5):
        fold_df = train_df.loc[
            train_df["fold"] == f, ["id_code", "diagnosis"]
        ].reset_index(drop=True)
        pred_df = predict_df(fold_df, img_dir=TRAIN_IMAGE_PATH, batch_size=16)

        fold_idx = train_df.index[train_df["fold"] == f].to_numpy()
        oof_pred[fold_idx] = pred_df["pred"].values.astype(np.float64)

        thr_f, qwk_f = fit_thresholds_coordinate_descent(
            pred_df["pred"].values,
            pred_df["diagnosis"].values,
            init_thr=None,
            n_iter=25,
        )
        print(f"Fold {f}: val_qwk={qwk_f:.5f} thr={thr_f}")

    thr, oof_qwk = fit_thresholds_coordinate_descent(
        oof_pred, y_true, init_thr=None, n_iter=35
    )
    print("Fitted thresholds (OOF):", thr, "oof_qwk:", oof_qwk)

    mapped_test_pred = ecdf_map_to_reference(
        submission["diagnosis"].values.astype(np.float64), oof_pred
    )
    submission["diagnosis"] = apply_thresholds(mapped_test_pred, thr).astype(int)

else:
    if not use_imagenet_fallback:
        submission.loc[submission.diagnosis < 0.7, "diagnosis"] = 0
        submission.loc[
            (0.7 <= submission.diagnosis) & (submission.diagnosis < 1.5), "diagnosis"
        ] = 1
        submission.loc[
            (1.5 <= submission.diagnosis) & (submission.diagnosis < 2.5), "diagnosis"
        ] = 2
        submission.loc[
            (2.5 <= submission.diagnosis) & (submission.diagnosis < 3.5), "diagnosis"
        ] = 3
        submission.loc[3.5 <= submission.diagnosis, "diagnosis"] = 4
        submission["diagnosis"] = submission["diagnosis"].astype(int)
    else:
        if TRAIN_CSV_PATH is None:
            submission.loc[submission.diagnosis < 0.7, "diagnosis"] = 0
            submission.loc[
                (0.7 <= submission.diagnosis) & (submission.diagnosis < 1.5),
                "diagnosis",
            ] = 1
            submission.loc[
                (1.5 <= submission.diagnosis) & (submission.diagnosis < 2.5),
                "diagnosis",
            ] = 2
            submission.loc[
                (2.5 <= submission.diagnosis) & (submission.diagnosis < 3.5),
                "diagnosis",
            ] = 3
            submission.loc[3.5 <= submission.diagnosis, "diagnosis"] = 4
            submission["diagnosis"] = submission["diagnosis"].astype(int)
        else:
            train_df2 = pd.read_csv(TRAIN_CSV_PATH)
            counts = (
                train_df2["diagnosis"]
                .value_counts()
                .reindex([0, 1, 2, 3, 4], fill_value=0)
                .astype(int)
            )
            n = len(submission)
            proportions = (counts / counts.sum()).values
            desired = (proportions * n).round().astype(int)
            diff = n - desired.sum()
            if diff != 0:
                desired[int(desired.argmax())] += diff

            order = submission["diagnosis"].values.argsort()
            labels = pd.Series(index=range(n), dtype=int)
            start = 0
            for cls, k in enumerate(desired):
                if k <= 0:
                    continue
                idx = order[start : start + k]
                labels.iloc[idx] = cls
                start += k
            labels = labels.fillna(4).astype(int)
            submission["diagnosis"] = labels.values

submission["diagnosis"] = submission["diagnosis"].astype(int)
submission = submission[["id_code", "diagnosis"]]
submission.to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:",
    submission.shape,
    "| fallback:",
    use_imagenet_fallback,
)
submission.head()
