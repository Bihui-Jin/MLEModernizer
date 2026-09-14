# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the script robust to the Kaggle environment by removing the failing `pip install` and adding a safe weights-loading fallback (so missing `../input/weights/...` doesn’t crash). I also fix device selection so it runs on CPU when no GPU is available, and ensure `trim()` always returns an image (it currently can return `None`). Finally, I make inference write a non-empty `submission.csv` in the exact required format by iterating over `test.csv` and producing integer class predictions (using the existing regression-to-rounded-class logic), even if pretrained weights are unavailable.'
- What this solution (achieved -0.03795) has done: 'I fix the shape-mismatch crash by ensuring the EfficientNet backbone outputs pooled feature vectors (not 1000-class logits) by resetting its classifier head to identity (`reset_classifier(0)`) and keeping your GeM global pooling, which preserves the intended architecture semantics. I also correct the `transforms` variable name (currently `tranforms`) to avoid subtle bugs, and make the inference call robust by using the regressor output as before. Finally, I ensure `submission.csv` is always written by guarding against earlier failures and by building the submission strictly from `test.csv` order with the required column names and integer classes 0–4.'
- What this solution (achieved -0.06325) has done: 'Your current score is far below the target, so we should improve performance without changing the model or training (there is no training here). The biggest issue is that you never use the intended `final=True` head (which produces the calibrated regression output used in your codebase), and you also quantize by simple rounding instead of using your provided thresholding function. I switch inference to `net(img_t, final=True)` and convert the continuous output to classes via `regress2class`, which better matches QWK behavior for ordinal labels while keeping the same architecture and semantics. I also ensure `regress2class` returns an integer tensor on CPU to avoid dtype/device quirks, and keep the submission order exactly as `test.csv`.'
- What this solution (achieved 0.40026) has done: 'Your current score is far below the target, so we should improve predictions without changing the model/training by fixing two inference mismatches that can collapse QWK: (1) your image preprocessing is likely not what the EfficientNet TF-* models expect, and (2) your fixed thresholds are almost certainly not calibrated to the actual output distribution, causing near-random class assignments. I keep your exact model and `final=True` head, but switch preprocessing to `timm.data.resolve_data_config/create_transform` for tf_efficientnet_b4_ns (more correct normalization/interp) and compute optimal class thresholds on the training set using out-of-fold-style inference (single pass) to directly maximize QWK on train predictions (no training, just calibration). Then I use those calibrated thresholds for test predictions and write `submission.csv` in the required format.'
- What this solution (achieved 0.21368) has done: 'Your current score (0.40026) is far below the target (0.8961), so we should improve toward the target with minimal risk while keeping your model and inference logic intact. The biggest safe gain is to stop calibrating thresholds on the same data you evaluate them on (train-set overfitting hurts generalization/QWK), so I compute thresholds using out-of-fold (OOF) predictions with a simple StratifiedKFold and then average the fold thresholds. I keep your exact network, `final=True` head, and the same `optimize_thresholds` routine—only changing how thresholds are estimated. I also ensure the `timm` transform uses the model’s resolved config (including correct mean/std/interpolation) without forcing an arbitrary input size, to avoid a preprocessing mismatch that can depress performance.'
- What this solution (achieved 0.21075) has done: 'Your current score (0.21368) is far below the target (0.8961), so we should improve generalization while keeping the same model and inference semantics. The main issue is that threshold calibration is currently unstable: it uses a single model pass over all train images and then “OOF” threshold fitting only on the validation slice outputs, but those outputs were produced by a model that may be in pretrained fallback mode and are not stratified-calibrated robustly. I keep your exact network (`final=True`) and threshold optimizer, but (1) compute *true* OOF continuous predictions by running inference fold-by-fold (model stays fixed; this just prevents any subtle ordering/transform issues and makes fold calibration consistent), and (2) increase the threshold optimizer iterations slightly (still deterministic coordinate descent) to reduce underfitting of thresholds. Finally, I ensure the timm transform is resolved from the exact backbone architecture used (`tf_efficientnet_b4_ns`) and explicitly force `img = img.resize` only through that transform path to avoid accidental size mismatch impacting predictions.'
- What this solution (achieved 0.20979) has done: 'Your current score (0.21075) is far below the target (0.8961), so we should improve generalization without changing the model architecture or adding training. The biggest minimal, metric-aligned win is to calibrate thresholds in a leakage-safe way: fit thresholds on *training-fold* predictions and evaluate on *validation-fold* (rather than fitting directly on the validation fold itself). I keep your exact model, `final=True` inference, preprocessing, and the same coordinate-descent threshold optimizer, but compute per-fold thresholds from `tr_idx` and then average them; this typically stabilizes QWK on test. I also keep submission ordering exactly from `test.csv` and still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
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
    pass
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

    Change (score-relevant): allow passing a fold-specific candidate grid derived from
    the training fold predictions to improve threshold fit stability/generalization.
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



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/178325688.py in <cell line: 0>()
     28 
     29 for fold, (tr_idx, va_idx) in enumerate(skf.split(train_ids, train_y), start=1):
---> 30     va_preds, miss = _predict_continuous_for_indices(va_idx)
     31     missing_train_images += miss
     32     train_cont_oof[va_idx] = va_preds

/tmp/ipykernel_55/178325688.py in _predict_continuous_for_indices(indices)
     22             img = trim(img)
     23             img_t = transform(img).unsqueeze(0).to(device)
---> 24             out = net(img_t, final=True)  # (1,1) in [0,4.5]
     25             preds[j] = float(out.squeeze().detach().float().cpu().item())
     26     return preds, miss

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/1744043663.py in forward(self, x, final)
     90 
     91         c_out = self.classifier(x)
---> 92         r_out = self.regressor(x)
     93         o_out = self.ordinal(x)
     94 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

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

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/758791799.py in <cell line: 0>()
     14         img_t = transform(img).unsqueeze(0).to(device)
     15 
---> 16         out = net(img_t, final=True)  # (1,1) continuous in [0,4.5]
     17         pred = int(regress2class(out, opt_thr)[0])
     18         pred = max(0, min(4, pred))

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/1744043663.py in forward(self, x, final)
     90 
     91         c_out = self.classifier(x)
---> 92         r_out = self.regressor(x)
     93         o_out = self.ordinal(x)
     94 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
