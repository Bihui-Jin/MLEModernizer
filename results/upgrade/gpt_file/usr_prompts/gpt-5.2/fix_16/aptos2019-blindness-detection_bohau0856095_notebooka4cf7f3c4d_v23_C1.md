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

# 5. Target score

0.8960656533846909

# 6. Current score

0.25067

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the script robust to the Kaggle environment by removing the failing `pip install` and adding a safe weights-loading fallback (so missing `../input/weights/...` doesn’t crash). I also fix device selection so it runs on CPU when no GPU is available, and ensure `trim()` always returns an image (it currently can return `None`). Finally, I make inference write a non-empty `submission.csv` in the exact required format by iterating over `test.csv` and producing integer class predictions (using the existing regression-to-rounded-class logic), even if pretrained weights are unavailable.'
- What this solution (achieved -0.03795) has done: 'I fix the shape-mismatch crash by ensuring the EfficientNet backbone outputs pooled feature vectors (not 1000-class logits) by resetting its classifier head to identity (`reset_classifier(0)`) and keeping your GeM global pooling, which preserves the intended architecture semantics. I also correct the `transforms` variable name (currently `tranforms`) to avoid subtle bugs, and make the inference call robust by using the regressor output as before. Finally, I ensure `submission.csv` is always written by guarding against earlier failures and by building the submission strictly from `test.csv` order with the required column names and integer classes 0–4.'
- What this solution (achieved -0.06325) has done: 'Your current score is far below the target, so we should improve performance without changing the model or training (there is no training here). The biggest issue is that you never use the intended `final=True` head (which produces the calibrated regression output used in your codebase), and you also quantize by simple rounding instead of using your provided thresholding function. I switch inference to `net(img_t, final=True)` and convert the continuous output to classes via `regress2class`, which better matches QWK behavior for ordinal labels while keeping the same architecture and semantics. I also ensure `regress2class` returns an integer tensor on CPU to avoid dtype/device quirks, and keep the submission order exactly as `test.csv`.'
- What this solution (achieved 0.40026) has done: 'Your current score is far below the target, so we should improve predictions without changing the model/training by fixing two inference mismatches that can collapse QWK: (1) your image preprocessing is likely not what the EfficientNet TF-* models expect, and (2) your fixed thresholds are almost certainly not calibrated to the actual output distribution, causing near-random class assignments. I keep your exact model and `final=True` head, but switch preprocessing to `timm.data.resolve_data_config/create_transform` for tf_efficientnet_b4_ns (more correct normalization/interp) and compute optimal class thresholds on the training set using out-of-fold-style inference (single pass) to directly maximize QWK on train predictions (no training, just calibration). Then I use those calibrated thresholds for test predictions and write `submission.csv` in the required format.'
- What this solution (achieved 0.21368) has done: 'Your current score (0.40026) is far below the target (0.8961), so we should improve toward the target with minimal risk while keeping your model and inference logic intact. The biggest safe gain is to stop calibrating thresholds on the same data you evaluate them on (train-set overfitting hurts generalization/QWK), so I compute thresholds using out-of-fold (OOF) predictions with a simple StratifiedKFold and then average the fold thresholds. I keep your exact network, `final=True` head, and the same `optimize_thresholds` routine—only changing how thresholds are estimated. I also ensure the `timm` transform uses the model’s resolved config (including correct mean/std/interpolation) without forcing an arbitrary input size, to avoid a preprocessing mismatch that can depress performance.'
- What this solution (achieved 0.21075) has done: 'Your current score (0.21368) is far below the target (0.8961), so we should improve generalization while keeping the same model and inference semantics. The main issue is that threshold calibration is currently unstable: it uses a single model pass over all train images and then “OOF” threshold fitting only on the validation slice outputs, but those outputs were produced by a model that may be in pretrained fallback mode and are not stratified-calibrated robustly. I keep your exact network (`final=True`) and threshold optimizer, but (1) compute *true* OOF continuous predictions by running inference fold-by-fold (model stays fixed; this just prevents any subtle ordering/transform issues and makes fold calibration consistent), and (2) increase the threshold optimizer iterations slightly (still deterministic coordinate descent) to reduce underfitting of thresholds. Finally, I ensure the timm transform is resolved from the exact backbone architecture used (`tf_efficientnet_b4_ns`) and explicitly force `img = img.resize` only through that transform path to avoid accidental size mismatch impacting predictions.'
- What this solution (achieved 0.20979) has done: 'Your current score (0.21075) is far below the target (0.8961), so we should improve generalization without changing the model architecture or adding training. The biggest minimal, metric-aligned win is to calibrate thresholds in a leakage-safe way: fit thresholds on *training-fold* predictions and evaluate on *validation-fold* (rather than fitting directly on the validation fold itself). I keep your exact model, `final=True` inference, preprocessing, and the same coordinate-descent threshold optimizer, but compute per-fold thresholds from `tr_idx` and then average them; this typically stabilizes QWK on test. I also keep submission ordering exactly from `test.csv` and still write a valid `submission.csv`.'
- What this solution (achieved 0.2332) has done: 'The timeout is dominated by per-image Python overhead (opening/transforming images one-by-one, CPU→GPU transfers) and by running the same backbone 3295 times for OOF with batch size 1. I keep the exact model and threshold-optimization logic, but make inference batched via a DataLoader (pinned memory + multiple workers) and cache preprocessed tensors to disk once so the 5-fold OOF and test inference reuse the same tensors without re-decoding PNGs. I also vectorize the QWK computation to remove Python loops inside the threshold search (same math, much faster), and ensure deterministic DataLoader behavior. These changes preserve evaluation semantics and predictions up to negligible floating-point differences while cutting wall time drastically.'
- What this solution (achieved 0.2332) has done: 'Your score is far below the target, so we make minimal, metric-aligned improvements without changing the model or any training (there is none). The main fix is that your “OOF” predictions are not truly out-of-fold (they’re just the same full-train pass copied into each fold), which makes threshold calibration unstable and harms generalization; we compute real OOF predictions by running inference fold-by-fold. Then we fit thresholds on each fold’s train split and evaluate on its validation split, and finally average thresholds as you already do. This keeps the same network (`final=True`), same preprocessing, same threshold optimizer, and still writes a valid `submission.csv`.'
- What this solution (achieved 0.2364) has done: 'Your current score (0.2332) is far below the target (0.8961), so we should improve calibration/generalization while keeping the exact same model and “predict continuous then threshold to classes” core logic. The smallest high-impact fix is to correct the preprocessing mismatch: your `timm.create_transform` path does not guarantee the same resize/crop pipeline used for EfficientNet inference, so we switch to `timm.data.create_transform` driven by the model’s full data config and explicitly set the final input resolution from that config. Next, we make threshold fitting more robust by (a) producing OOF continuous predictions with the model in `eval()` and (b) fitting thresholds on train-fold predictions using a stable distribution-based initializer from the training-fold quantiles (instead of fixed `[0.7,1.5,2.5,3.5]`), which usually improves QWK without changing the model. Finally, we keep the same averaged-threshold scheme and submission writing, ensuring order matches `test.csv` and output is valid.'
- What this solution (achieved 0.25067) has done: 'Your score is far below the target, so we should improve toward it without changing the model architecture or adding training by fixing the biggest generalization/calibration issue: thresholds are currently averaged across folds, which can blur a strong per-fold fit into a weaker global threshold set for test. I keep your exact OOF prediction pipeline and the same threshold optimizer, but instead select a single “best” threshold set using OOF QWK (evaluate each fold’s fitted thresholds on full OOF, then pick the best) and also blend it with the averaged thresholds only if it improves OOF QWK. I also ensure missing-image predictions are excluded from threshold fitting (they currently get forced to 0.0, which can distort quantiles/grid), while keeping submission semantics identical. These are minimal post-processing/calibration changes that typically move QWK up materially without touching the network itself.'
- What this solution (achieved 0.25067) has done: 'You’re far below the target (0.25067 vs 0.8961), so the smallest legitimate move toward the target is to make sure you’re actually using the provided competition-trained weights rather than falling back to a generic pretrained backbone. Right now your `WEIGHTS_CANDIDATES` paths likely never exist in this environment, which effectively makes predictions near-random even with threshold tuning. I minimally expand weight discovery to search under `../input/` for any matching `B4_3stage_75epoch*.pth|pkl` and prefer an exact-name match; if found, we load it (same model, same inference semantics). Everything else (model architecture, `final=True` continuous prediction, OOF-based threshold calibration, submission writing) stays the same.'

# 9. Code solution

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
    out = np.asarray(out, dtype=np.float32).reshape(-1)

    thr = np.asarray(threshold, dtype=np.float32).reshape(-1)
    return (out[:, None] >= thr[None, :]).sum(axis=1).astype(np.int64)


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    assert y_true.shape == y_pred.shape

    mask = (y_true >= 0) & (y_true < n_classes) & (y_pred >= 0) & (y_pred < n_classes)
    yt = y_true[mask]
    yp = y_pred[mask]
    O = np.bincount(yt * n_classes + yp, minlength=n_classes * n_classes).astype(
        np.float64
    )
    O = O.reshape(n_classes, n_classes)

    act_hist = np.bincount(yt, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(yp, minlength=n_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    i = np.arange(n_classes, dtype=np.float64)
    W = ((i[:, None] - i[None, :]) ** 2) / ((n_classes - 1) ** 2)

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


def _find_weight_candidates():
    explicit = [
        "../input/weights/B4_3stage_75epoch.pkl",
        "../input/weights/B4_3stage_75epoch.pth",
        "../input/aptos2019-blindness-detection/B4_3stage_75epoch.pkl",
        "../input/aptos2019-blindness-detection/B4_3stage_75epoch.pth",
    ]
    found = [p for p in explicit if os.path.exists(p)]

    roots = ["../input"]
    patterns = ("B4_3stage_75epoch", "b4_3stage_75epoch")
    exts = (".pth", ".pkl")

    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                lfn = fn.lower()
                if not lfn.endswith(exts):
                    continue
                if any(pat in lfn for pat in patterns):
                    found.append(os.path.join(dirpath, fn))

    def _rank(p):
        base = os.path.basename(p).lower()
        exact = base in ("b4_3stage_75epoch.pth", "b4_3stage_75epoch.pkl")
        return (0 if exact else 1, len(p), p)

    found = sorted(list(dict.fromkeys(found)), key=_rank)
    return found


WEIGHTS_CANDIDATES = _find_weight_candidates()

net = ThreeStage_Model(pretrained_backbone=False)
loaded = False
loaded_path = None

for wpath in WEIGHTS_CANDIDATES:
    try:
        if os.path.exists(wpath):
            state = torch.load(wpath, map_location="cpu")
            state = _clean_state_dict_keys(state)
            net.load_state_dict(state, strict=False)
            loaded = True
            loaded_path = wpath
            break
    except Exception:
        continue

if not loaded:
    net = ThreeStage_Model(pretrained_backbone=True)

net = net.to(device)
net.eval()

try:
    from timm.data import resolve_model_data_config, create_transform

    _cfg = resolve_model_data_config(net.backbone)
    transform = create_transform(**_cfg, is_training=False)
    input_size = int(_cfg["input_size"][-1])
except Exception:
    input_size = 380  # tf_efficientnet_b4 default-ish
    transform = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
        ]
    )

print("Device:", device)
print("Weights loaded:", loaded, "| path:", loaded_path)
print("Num weight candidates found:", len(WEIGHTS_CANDIDATES))
print("First 5 candidates:", WEIGHTS_CANDIDATES[:5])
print("Transform input_size:", input_size)



## === cell 4
from torch.utils.data import Dataset, DataLoader

CACHE_DIR = "../kaggle/working/cache_aptos_tensors"
os.makedirs(CACHE_DIR, exist_ok=True)


def _safe_tensor_cache_name(image_id: str, split: str):
    return os.path.join(CACHE_DIR, f"{split}__{image_id}.pt")


class APTOSCachedTensorDataset(Dataset):
    def __init__(
        self, ids, img_dir, split_name: str, transform, cache_missing_value=0.0
    ):
        self.ids = [str(x) for x in ids]
        self.img_dir = img_dir
        self.split_name = split_name
        self.transform = transform
        self.cache_missing_value = cache_missing_value

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        cache_path = _safe_tensor_cache_name(image_id, self.split_name)

        if os.path.exists(cache_path):
            x = torch.load(cache_path, map_location="cpu")
            return x, image_id, True  # True means "not missing"
        image_path = os.path.join(self.img_dir, f"{image_id}.png")
        if not os.path.exists(image_path):
            x = torch.zeros(3, input_size, input_size, dtype=torch.float32)
            return x, image_id, False

        img = Image.open(image_path).convert("RGB")
        img = trim(img)
        x = self.transform(img)  # float32 tensor, normalized
        torch.save(x, cache_path)
        return x, image_id, True


def _seed_worker(worker_id):
    seed = 42 + worker_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def predict_continuous_for_ids(ids, img_dir, split_name: str, batch_size=16):
    ds = APTOSCachedTensorDataset(ids, img_dir, split_name, transform)
    num_workers = min(4, os.cpu_count() or 1)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=_seed_worker,
        persistent_workers=(num_workers > 0),
    )

    preds = np.zeros(len(ids), dtype=np.float32)
    ok_mask = np.zeros(len(ids), dtype=bool)

    net.eval()
    with torch.inference_mode():
        offset = 0
        for xb, image_ids, ok in loader:
            bsz = xb.size(0)
            ok_np = np.asarray(ok, dtype=bool)

            xb = xb.to(device, non_blocking=True)
            out = net(xb, final=True).squeeze(1)  # (B,)
            out = out.detach().float().cpu().numpy().astype(np.float32)

            preds[offset : offset + bsz] = out
            ok_mask[offset : offset + bsz] = ok_np
            offset += bsz

    missing = int((~ok_mask).sum())
    return preds, ok_mask, missing




## === cell 5
from sklearn.model_selection import StratifiedKFold

N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

train_cont_oof = np.zeros(len(train_ids), dtype=np.float32)
train_ok_oof = np.zeros(len(train_ids), dtype=bool)
missing_train_images_total = 0

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_ids, train_y), start=1):
    va_ids = train_ids[va_idx]
    va_cont, va_ok, missing_va = predict_continuous_for_ids(
        va_ids, TRAIN_IMG_DIR, split_name=f"train_fold{fold}_va", batch_size=16
    )
    train_cont_oof[va_idx] = va_cont
    train_ok_oof[va_idx] = va_ok
    missing_train_images_total += int(missing_va)
    print(
        f"Fold {fold}: predicted OOF for {len(va_idx)} samples | missing={missing_va}"
    )

fold_thresholds = []
fold_scores = []
fold_oof_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_ids, train_y), start=1):
    tr_idx_ok = tr_idx[train_ok_oof[tr_idx]]
    va_idx_ok = va_idx[train_ok_oof[va_idx]]

    y_tr_cont = train_cont_oof[tr_idx_ok]
    y_tr = train_y[tr_idx_ok]

    init_thresholds = [float(np.quantile(y_tr_cont, q)) for q in (0.2, 0.4, 0.6, 0.8)]
    init_thresholds = sorted(init_thresholds)
    grid_tr = np.quantile(y_tr_cont, np.linspace(0.005, 0.995, 399)).astype(np.float32)

    thr_f, _ = optimize_thresholds(
        y_tr,
        y_tr_cont,
        init_thresholds=init_thresholds,
        n_iter=6,
        grid=grid_tr,
    )

    qwk_f = quadratic_weighted_kappa(
        train_y[va_idx_ok], regress2class(train_cont_oof[va_idx_ok], thr_f)
    )

    qwk_oof_f = quadratic_weighted_kappa(
        train_y[train_ok_oof], regress2class(train_cont_oof[train_ok_oof], thr_f)
    )

    fold_thresholds.append(thr_f)
    fold_scores.append(qwk_f)
    fold_oof_scores.append(qwk_oof_f)
    print(
        f"Fold {fold}: thr_fit_on_train={thr_f} | val_qwk(ok_only)={qwk_f:.6f} | oof_qwk(ok_only)={qwk_oof_f:.6f}"
    )

avg_thr = np.mean(np.array(fold_thresholds, dtype=np.float32), axis=0).tolist()
avg_thr = sorted(avg_thr)

oof_qwk_avg = quadratic_weighted_kappa(
    train_y[train_ok_oof], regress2class(train_cont_oof[train_ok_oof], avg_thr)
)

best_i = int(np.argmax(np.asarray(fold_oof_scores, dtype=np.float32)))
best_thr = sorted([float(x) for x in fold_thresholds[best_i]])
oof_qwk_best = quadratic_weighted_kappa(
    train_y[train_ok_oof], regress2class(train_cont_oof[train_ok_oof], best_thr)
)

opt_thr = best_thr
opt_thr_name = f"best_single_fold_by_oof(fold={best_i+1})"

blend_thr = sorted(
    np.mean(
        np.stack(
            [np.asarray(avg_thr, np.float32), np.asarray(best_thr, np.float32)], 0
        ),
        0,
    ).tolist()
)
oof_qwk_blend = quadratic_weighted_kappa(
    train_y[train_ok_oof], regress2class(train_cont_oof[train_ok_oof], blend_thr)
)

if oof_qwk_avg > oof_qwk_best and oof_qwk_avg >= oof_qwk_blend:
    opt_thr = avg_thr
    opt_thr_name = "avg_over_folds"
elif oof_qwk_blend > oof_qwk_best and oof_qwk_blend >= oof_qwk_avg:
    opt_thr = blend_thr
    opt_thr_name = "blend(avg,best)"

oof_qwk_opt = quadratic_weighted_kappa(
    train_y[train_ok_oof], regress2class(train_cont_oof[train_ok_oof], opt_thr)
)

print("Missing train images encountered (OOF total):", missing_train_images_total)
print("OOF QWK (avg thresholds, ok-only):", float(oof_qwk_avg), "| thr:", avg_thr)
print("OOF QWK (best thresholds, ok-only):", float(oof_qwk_best), "| thr:", best_thr)
print("OOF QWK (blend thresholds, ok-only):", float(oof_qwk_blend), "| thr:", blend_thr)
print("Selected thresholds:", opt_thr, "| scheme:", opt_thr_name)
print("OOF QWK using selected thresholds (ok-only):", float(oof_qwk_opt))
print("Mean fold VAL QWK (ok-only):", float(np.mean(fold_scores)))



## === cell 6
test_cont, test_ok, missing_test_images = predict_continuous_for_ids(
    test_ids, TEST_IMG_DIR, split_name="test", batch_size=16
)

test_pred = regress2class(test_cont, opt_thr)
test_pred = np.clip(test_pred, 0, 4).astype(int)

if not np.all(test_ok):
    test_pred = test_pred.copy()
    test_pred[~test_ok] = 0

submission = pd.DataFrame(
    {"id_code": test_ids.astype(str), "diagnosis": test_pred.astype(int)}
)

print("Missing test images encountered:", missing_test_images)
print("Built submission dataframe:", submission.shape)

submission.to_csv("submission.csv", index=False)

print("Saved submission.csv with shape:", submission.shape)
print(submission.head())
print("diagnosis value counts:\n", submission["diagnosis"].value_counts().sort_index())
