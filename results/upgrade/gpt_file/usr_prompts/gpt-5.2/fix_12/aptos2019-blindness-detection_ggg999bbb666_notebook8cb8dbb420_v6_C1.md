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

0.9030739204737794

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make minimal changes to ensure you always generate a valid submission and improve the expected kappa by fixing two common inference-time issues: checkpoint loading compatibility (handling `fc.*` shape mismatches and other non-critical key mismatches safely) and applying test-time augmentation (horizontal flip) while keeping the same model and thresholds. I also switch preprocessing to use the same normalization but run inference in batches to reduce overhead and help complete within the time limit (without changing model logic). Finally, I keep your exact thresholding scheme (evaluation semantics) so any score changes come from more reliable/robust predictions rather than a different mapping.'
- What this solution (achieved -0.06214) has done: 'I make the code reliably find and load your checkpoints from `/kaggle/input` (your current resolver can pick unrelated `.pth` files, which often leads to near-random predictions and ~0 QWK). I also remove the “distribution matching” thresholding and instead use a fixed, deterministic quantile-based mapping (using only your test predictions) to avoid train/test distribution mismatch and extreme class collapse; this preserves the same core semantics (continuous → 5 ordinal bins) without changing the model or loss. Finally, I ensure the submission strictly follows the sample submission row order and format and always writes `submission.csv`. These are minimal inference/calibration fixes aimed at moving score upward from “no valid score / likely ~0” toward the target band.'
- What this solution (achieved -0.00868) has done: 'Your current negative QWK is most consistent with a prediction-to-class mapping issue rather than the ensemble itself, so I keep your exact models/inference but replace the test-quantile binning with a small, valid “optimize thresholds on a train validation split” step that directly targets quadratic weighted kappa. This preserves your core semantics (continuous regression → 5 ordinal bins) while making the bin edges much more likely to align with the metric. I also add a safety fallback to your old quantile thresholds if the threshold search fails for any reason, and keep the submission ordering/format identical. No architecture, loss, or training loop is introduced; this is purely post-processing calibration toward the competition metric.'
- What this solution (achieved -0.00868) has done: 'Your negative QWK strongly suggests the ordinal mapping (thresholds) is still poorly calibrated to the model’s raw outputs, not that inference is broken. I keep your exact ensemble models, resizing, preprocessing, and TTA, but make the threshold fitting more metric-aligned and stable by (1) using out-of-fold (OOF) calibration predictions over a larger, stratified subset and (2) optimizing thresholds with a short coordinate-descent + ternary-search refinement directly on QWK. This preserves the same continuous→5-bin semantics (only cutpoints change) while greatly reducing the chance of pathological class collapse that drives QWK below zero. I also ensure checkpoint loading is safe across common wrappers and keep the submission order exactly matching `test.csv`, writing `submission.csv` as before.'
- What this solution (achieved -0.00868) has done: 'Your negative QWK strongly suggests the calibration step is still unstable (and/or misaligned to the true label distribution), so I keep your exact models, preprocessing, TTA, and continuous→5-bin semantics, but make threshold fitting both more robust and more representative. Specifically, I (1) fit thresholds on **out-of-fold predictions built from the full train set** (still using your existing checkpoints, no training) instead of a small 1600-sample subset, and (2) add a **monotonic, bounded optimizer** (coordinate descent over sorted unique prediction values) that can’t produce invalid threshold orderings. This is a minimal post-processing change that typically moves QWK upward substantially when the model is reasonable, without changing architecture or inference. The script still writes `submission.csv` in the required format.'
- What this solution (achieved -0.00868) has done: 'I make two minimal, score-relevant fixes that commonly cause negative/near-zero QWK in this competition even when the model weights are reasonable: (1) correct the “OOF calibration” bug (your current code predicts the same model on the same images for every fold, so the thresholds are fit on leaky/in-sample predictions and can become badly miscalibrated), and (2) ensure we calibrate thresholds on predictions that are produced under the same transform pipeline as test (same resize/TTA) but without data leakage. Core model, inference, TTA, and the “continuous → 5 bins via 4 thresholds” semantics remain unchanged; only the fold-prediction routing is fixed. This should move the score upward from “no valid / negative QWK” toward the target band by making the threshold fitting actually reflect generalization.'
- What this solution (achieved 0.0) has done: 'Your negative QWK is most consistent with a broken calibration/OOF step rather than the CNN inference itself. I keep your exact model(s), preprocessing, and TTA, but fix the OOF construction so each fold is predicted by a model that did not “see” that fold (using your existing multiple checkpoints as the folds), which prevents leaky/in-sample threshold fitting. Then I fit the 4 thresholds on these true OOF predictions to maximize QWK (same continuous→5-bin semantics), and use those thresholds for test predictions. If fewer than 5 usable checkpoints exist, it safely falls back to using as many folds as available (or the prior behavior), while still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
from glob import glob

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet, Bottleneck

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_ROOT = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_IMAGE_PATH = os.path.join(DATA_ROOT, "test_images")
TRAIN_IMAGE_PATH = os.path.join(DATA_ROOT, "train_images")

MODEL_PATH_list = [
    "../input/224best/224best.pth",
    "../input/128best/128best.pth",
    "../input/seresnet384/model_epoch33.pth",
]
resize_list = [224, 128, 384]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _resolve_ckpt_path(p: str) -> str:
    """
    Score-relevant fix: avoid accidentally resolving to an unrelated .pth from another dataset.
    Only search inside /kaggle/input and require that the candidate path contains the intended
    immediate parent folder name (e.g., '224best', '128best', 'seresnet384') when possible.
    """
    if os.path.isfile(p):
        return p

    base = os.path.basename(p)
    parent = os.path.basename(os.path.dirname(p))

    candidates = glob(f"/kaggle/input/**/{base}", recursive=True)
    candidates = [c for c in candidates if os.path.isfile(c)]

    if parent:
        preferred = [c for c in candidates if f"/{parent}/" in c.replace("\\", "/")]
        if len(preferred) > 0:
            candidates = preferred

    if len(candidates) > 0:
        candidates = sorted(candidates, key=lambda x: (len(x), x))
        return candidates[0]

    return p


MODEL_PATH_list = [_resolve_ckpt_path(p) for p in MODEL_PATH_list]
print("Resolved model paths:")
for p in MODEL_PATH_list:
    print(" -", p, "| exists:", os.path.isfile(p))

if not os.path.isfile(TEST_CSV):
    alt_root = "/kaggle/input/aptos2019-blindness-detection"
    alt_csv = os.path.join(alt_root, "test.csv")
    alt_train = os.path.join(alt_root, "train.csv")
    alt_img = os.path.join(alt_root, "test_images")
    alt_train_img = os.path.join(alt_root, "train_images")
    if os.path.isfile(alt_csv):
        DATA_ROOT = alt_root
        TEST_CSV = alt_csv
        TRAIN_CSV = alt_train
        TEST_IMAGE_PATH = alt_img
        TRAIN_IMAGE_PATH = alt_train_img

if os.path.isdir(os.path.join(TEST_IMAGE_PATH, "test_images")):
    TEST_IMAGE_PATH = os.path.join(TEST_IMAGE_PATH, "test_images")
if os.path.isdir(os.path.join(TRAIN_IMAGE_PATH, "train_images")):
    TRAIN_IMAGE_PATH = os.path.join(TRAIN_IMAGE_PATH, "train_images")

print("DATA_ROOT:", DATA_ROOT)
print("TEST_CSV exists:", os.path.isfile(TEST_CSV))
print("TRAIN_CSV exists:", os.path.isfile(TRAIN_CSV))
print("TEST_IMAGE_PATH exists:", os.path.isdir(TEST_IMAGE_PATH))
print("TRAIN_IMAGE_PATH exists:", os.path.isdir(TRAIN_IMAGE_PATH))




## === cell 1
class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        s = self.avg_pool(x)
        s = self.fc1(s)
        s = self.relu(s)
        s = self.fc2(s)
        s = self.sigmoid(s)
        return x * s


class SEBottleneck(Bottleneck):
    def __init__(self, *args, reduction=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.se_module = SEModule(self.conv3.out_channels, reduction=reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se_module(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        return x.pow(1.0 / self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


def get_se_resnet50_gem(pretrain=None):
    model = ResNet(block=SEBottleneck, layers=[3, 4, 6, 3], num_classes=1000)

    if pretrain == "imagenet":
        base = resnet50(weights="DEFAULT")
        model.load_state_dict(base.state_dict(), strict=False)

    model.avgpool = GeM()
    model.fc = nn.Linear(2048, 1)
    return model


def _clean_state_dict_for_model(state, model):
    """
    Score-relevant robustness: handle common checkpoint wrappers and safely drop
    incompatible keys (e.g., different fc head naming/shape).
    """
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]

    if (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        state = state["model"]

    if not isinstance(state, dict):
        raise ValueError("Loaded checkpoint is not a state_dict-like dict")

    if len(state) > 0:
        k0 = next(iter(state.keys()))
        if isinstance(k0, str) and k0.startswith("module."):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model_sd = model.state_dict()
    filtered = {}
    dropped = 0
    for k, v in state.items():
        if k in model_sd and hasattr(v, "shape") and model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            dropped += 1

    if dropped > 0:
        print(f"Checkpoint keys dropped due to mismatch/unexpected: {dropped}")

    return filtered




## === cell 2
test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).tolist()
test_images = [os.path.join(TEST_IMAGE_PATH, f"{iid}.png") for iid in test_ids]

train_df = pd.read_csv(TRAIN_CSV)
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["diagnosis"] = train_df["diagnosis"].astype(int)

calib_df = train_df.copy().reset_index(drop=True)
calib_y = calib_df["diagnosis"].to_numpy(dtype=np.int64)
print(
    "Calibration set (full train):",
    calib_df.shape,
    "label counts:",
    np.bincount(calib_y, minlength=5).tolist(),
)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def load_image_tensor(path, size, hflip=False):
    try:
        img = Image.open(path).convert("RGB")
    except Exception:
        img = Image.new("RGB", (size, size), (0, 0, 0))
    img = img.resize((size, size), resample=Image.BILINEAR)
    if hflip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    t = transforms.ToTensor()(img)
    t = normalize(t)
    return t


def predict_model(model, image_paths, size, batch_size=16):
    model.eval()
    out = np.zeros(len(image_paths), dtype=np.float32)

    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for start in range(0, len(image_paths), batch_size):
            end = min(len(image_paths), start + batch_size)
            batch_paths = image_paths[start:end]

            x1 = torch.stack(
                [load_image_tensor(p, size, hflip=False) for p in batch_paths], dim=0
            )
            x2 = torch.stack(
                [load_image_tensor(p, size, hflip=True) for p in batch_paths], dim=0
            )

            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                y1 = model(x1).view(-1).float()
                y2 = model(x2).view(-1).float()
                y = 0.5 * (y1 + y2)

            out[start:end] = y.detach().cpu().numpy()

            if start % (batch_size * 40) == 0:
                print(f"  infer {start}/{len(image_paths)}")
    return out


def _make_stratified_folds(
    y: np.ndarray, n_splits: int = 5, seed: int = 42
) -> np.ndarray:
    rs_local = np.random.RandomState(seed)
    y = np.asarray(y, dtype=np.int64)
    fold = np.full(len(y), -1, dtype=np.int64)
    for c in range(5):
        idx = np.where(y == c)[0]
        rs_local.shuffle(idx)
        for i, j in enumerate(idx):
            fold[j] = i % n_splits
    missing = np.where(fold < 0)[0]
    if len(missing):
        for k, j in enumerate(missing):
            fold[j] = k % n_splits
    return fold


train_image_paths = [
    os.path.join(TRAIN_IMAGE_PATH, f"{iid}.png") for iid in calib_df["id_code"].tolist()
]

ckpts = [
    (p, sz)
    for p, sz in zip(MODEL_PATH_list, resize_list)
    if (p is None or os.path.isfile(p))
]
if len(ckpts) == 0:
    print(
        "WARNING: No checkpoints found; using ImageNet weights for a single model to produce a valid submission."
    )
    ckpts = [(None, 224)]

n_folds = min(5, len(ckpts))
fold_ids = _make_stratified_folds(calib_y, n_splits=n_folds, seed=42)

preds_test = np.zeros(len(test_images), dtype=np.float32)
preds_calib_oof = np.zeros(len(calib_df), dtype=np.float32)

for idx, (ckpt_path, sz) in enumerate(ckpts):
    use_imagenet = ckpt_path is None
    model = get_se_resnet50_gem(pretrain="imagenet" if use_imagenet else None).to(
        device
    )

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu")
        state = _clean_state_dict_for_model(state, model)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 0:
            print(f"Missing keys (after filtering): {len(missing)}")
        if len(unexpected) > 0:
            print(f"Unexpected keys (after filtering): {len(unexpected)}")

    print(f"Predicting TEST with model {idx+1}/{len(ckpts)} at size {sz} ...")
    model_pred_test = predict_model(model, test_images, sz, batch_size=16)
    preds_test += model_pred_test / len(ckpts)

for f in range(n_folds):
    ckpt_path, sz = ckpts[f]
    use_imagenet = ckpt_path is None
    model = get_se_resnet50_gem(pretrain="imagenet" if use_imagenet else None).to(
        device
    )

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location="cpu")
        state = _clean_state_dict_for_model(state, model)
        model.load_state_dict(state, strict=False)

    mask = fold_ids == f  # held-out fold f is predicted by model f
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        continue
    paths_f = [train_image_paths[i] for i in idxs]
    print(
        f"Predicting CALIB-OOF fold {f+1}/{n_folds} with model {f+1}/{len(ckpts)} at size {sz} (n={len(paths_f)}) ..."
    )
    pred_f = predict_model(model, paths_f, sz, batch_size=16)
    preds_calib_oof[mask] = pred_f

raw_submission = pd.DataFrame({"id_code": test_ids, "diagnosis": preds_test})
raw_submission.to_csv("submission_raw_value.csv", index=False)




## === cell 3
def _threshold_by_fixed_quantiles(raw_pred: np.ndarray) -> np.ndarray:
    thr = np.quantile(raw_pred, [0.20, 0.50, 0.75, 0.90]).astype(np.float32)
    print("Fallback: fixed-quantile thresholds (on test preds):", thr.tolist())
    y = np.zeros_like(raw_pred, dtype=np.int64)
    y[raw_pred >= thr[0]] = 1
    y[raw_pred >= thr[1]] = 2
    y[raw_pred >= thr[2]] = 3
    y[raw_pred >= thr[3]] = 4
    return y


def _qwk(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> float:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    w = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            w[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = np.bincount(y_true.clip(0, n_classes - 1), minlength=n_classes).astype(
        np.float64
    )
    pred_hist = np.bincount(y_pred.clip(0, n_classes - 1), minlength=n_classes).astype(
        np.float64
    )
    E = np.outer(act_hist, pred_hist)
    E = E / (E.sum() + 1e-12) * O.sum()

    num = (w * O).sum()
    den = (w * E).sum() + 1e-12
    return 1.0 - num / den


def _apply_thresholds(raw_pred: np.ndarray, thr: np.ndarray) -> np.ndarray:
    thr = np.asarray(thr, dtype=np.float32)
    y = np.zeros_like(raw_pred, dtype=np.int64)
    y[raw_pred >= thr[0]] = 1
    y[raw_pred >= thr[1]] = 2
    y[raw_pred >= thr[2]] = 3
    y[raw_pred >= thr[3]] = 4
    return y


def _rank_gauss(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    u = (ranks + 0.5) / len(x)
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow

    z = np.empty_like(u, dtype=np.float64)
    mask = u < plow
    if np.any(mask):
        q = np.sqrt(-2 * np.log(u[mask]))
        z[mask] = (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        z[mask] = -z[mask]
    mask = (u >= plow) & (u <= phigh)
    if np.any(mask):
        q = u[mask] - 0.5
        r = q * q
        z[mask] = (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
        )
    mask = u > phigh
    if np.any(mask):
        q = np.sqrt(-2 * np.log(1 - u[mask]))
        z[mask] = (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    return z.astype(np.float32)


def _fit_thresholds_for_qwk_rank_stable(
    raw_pred: np.ndarray, y_true: np.ndarray
) -> np.ndarray:
    raw_pred = raw_pred.astype(np.float32)
    y_true = y_true.astype(np.int64).clip(0, 4)

    z = _rank_gauss(raw_pred)

    values = np.unique(np.sort(z))
    if len(values) < 200:
        return np.quantile(raw_pred, [0.20, 0.50, 0.75, 0.90]).astype(np.float32)

    thr_z = np.quantile(z, [0.20, 0.50, 0.75, 0.90]).astype(np.float32)

    def score(thz):
        y = np.zeros_like(z, dtype=np.int64)
        y[z >= thz[0]] = 1
        y[z >= thz[1]] = 2
        y[z >= thz[2]] = 3
        y[z >= thz[3]] = 4
        return _qwk(y_true, y)

    best = thr_z.copy()
    best_s = float(score(best))

    n = len(values)
    idx_init = [int(np.searchsorted(values, t, side="left")) for t in best]
    windows = []
    for ii in idx_init:
        lo = max(0, ii - 600)
        hi = min(n - 1, ii + 600)
        if hi - lo + 1 > 601:
            idxs = np.linspace(lo, hi, 601).round().astype(int)
            idxs = np.unique(np.clip(idxs, 0, n - 1))
        else:
            idxs = np.arange(lo, hi + 1, dtype=int)
        windows.append(values[idxs])

    for _ in range(8):
        improved = False
        for t in range(4):
            lower = -np.inf if t == 0 else best[t - 1] + 1e-4
            upper = np.inf if t == 3 else best[t + 1] - 1e-4
            cur = best.copy()
            local_best_v = float(best[t])
            local_best_s = best_s

            for v in windows[t]:
                if not (lower < float(v) < upper):
                    continue
                cur[t] = float(v)
                s = float(score(cur))
                if s > local_best_s:
                    local_best_s = s
                    local_best_v = float(v)

            if local_best_s > best_s:
                best_s = local_best_s
                best[t] = local_best_v
                improved = True
        if not improved:
            break

    q = [float((z < best[i]).mean()) for i in range(4)]
    q = np.clip(q, 1e-4, 1 - 1e-4)
    thr_raw = np.quantile(raw_pred, q).astype(np.float32)

    print(
        "Fitted thresholds (rank-stable): z=",
        best.tolist(),
        "raw=",
        thr_raw.tolist(),
        "OOF calib QWK:",
        float(best_s),
    )
    return thr_raw


use_fitted = (
    preds_calib_oof is not None
    and len(calib_y) >= 500
    and np.isfinite(preds_calib_oof).all()
)
if use_fitted:
    try:
        fitted_thr = _fit_thresholds_for_qwk_rank_stable(preds_calib_oof, calib_y)
        pred_labels = _apply_thresholds(preds_test, fitted_thr)
    except Exception as e:
        print("Threshold fitting failed, using fallback quantiles. Error:", repr(e))
        pred_labels = _threshold_by_fixed_quantiles(preds_test)
else:
    print("No sufficient calibration available; using fallback quantiles.")
    pred_labels = _threshold_by_fixed_quantiles(preds_test)

submission = pd.DataFrame({"id_code": test_ids, "diagnosis": pred_labels})
submission = submission.set_index("id_code").loc[test_ids].reset_index()
submission["diagnosis"] = submission["diagnosis"].astype(np.int64).clip(0, 4)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote: submission.csv")
print("submission.csv shape:", submission.shape)
print("Unique diagnosis counts:\n", submission["diagnosis"].value_counts().sort_index())
