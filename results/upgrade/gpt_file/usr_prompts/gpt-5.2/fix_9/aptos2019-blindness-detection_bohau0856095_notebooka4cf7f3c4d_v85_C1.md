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

0.9229751770428916

# 6. Current score

0.6809

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.00596) has done: 'I remove the hard dependency on an external checkpoint (which isn’t present under `/kaggle/input` here) by falling back to a deterministic, lightweight baseline when no weights are found, so the notebook always runs end-to-end and writes a valid `submission.csv`. I also fix the downstream `NameError`/`AttributeError` by ensuring `net` and `submission` are always defined with the expected types, and by producing predictions for every `id_code` (no skipped rows). The baseline use simple image brightness/contrast/colorfulness heuristics mapped to 0–4 so it produces a non-empty, valid submission without changing the intended inference flow more than necessary given missing weights. If a compatible checkpoint is later attached, the code automatically load it and use the original model path unchanged.'
- What this solution (achieved 0.3989) has done: 'Your current score is near random because the fallback heuristic isn’t aligned with the DR severity signal, so I keep your exact inference flow but replace the heuristic with a small, deterministic, label-informed baseline trained from `train.csv` using only simple image statistics (no architecture/training loop changes). Concretely: extract a few robust per-image features (brightness/contrast/colorfulness/edge density) from a subset of training images, fit a regularized ordinal regression via `sklearn` with cross-validated threshold selection to directly maximize quadratic weighted kappa on a validation split, then apply the same mapping to test features. This preserves the “no checkpoint -> baseline” behavior, but makes the baseline actually correlate with DR grades, which should move the score strongly upward toward your 0.923 target while remaining lightweight and under the 600s limit. If the original checkpoint is present, the model path is unchanged and still used.'
- What this solution (achieved 0.44957) has done: 'Your current score (0.3989) is far below the target (0.9230), so we should improve the fallback path (used when the checkpoint is missing) without changing the core model/inference flow. The smallest high-impact change is to make the fallback baseline better aligned with QWK by (1) using all training images (not a 1200 subsample) and (2) fitting thresholds via a fast coordinate-descent search rather than a tiny local grid, while keeping the same simple image-stat features and Ridge regressor. This should materially increase correlation between predictions and labels and move the score upward toward the target. The model-checkpoint path remains unchanged; if the checkpoint exists it still be used exactly as before.'
- What this solution (achieved 0.44714) has done: 'Your current score (0.44957) is far below the target (0.92298), so we should improve only the fallback path (used when the EfficientNet checkpoint is missing) while leaving the original model/inference flow untouched. The smallest high-impact change is to make the fallback baseline learn a better *ordinal mapping* by fitting thresholds on out-of-fold predictions (so thresholds generalize better to test and better match QWK), instead of tuning thresholds on a single split. We keep the same simple image-stat feature extractor and the same Ridge regressor, but we (1) generate out-of-fold continuous predictions with StratifiedKFold, (2) optimize thresholds on those OOF preds to directly improve QWK, and (3) train the Ridge on all training data afterwards and apply the learned thresholds to test. This should move the score upward toward your target without changing the core model architecture/training loop and still runs within the time limit.'
- What this solution (achieved 0.6841) has done: 'Your current score (0.447) is far below the target (0.923, higher-is-better), so we should improve only the fallback path (used when the missing checkpoint forces heuristic predictions) while keeping the original model/inference flow unchanged. The smallest high-impact fix is to make the fallback baseline use image features that are more DR-relevant (vessel/lesion-like signals) by adding a fast fundus “field-of-view” crop and a couple of inexpensive texture/CLAHE-based features, without changing the overall approach (still: extract simple per-image features → Ridge regressor → threshold optimization for QWK). This should increase correlation with true grades and push QWK upward toward the target. We also fit the scaler+ridge via the same pipeline during cross-validation (so OOF predictions are correctly standardized) and keep runtime bounded by extracting features at low resolution.'
- What this solution (achieved 0.6841) has done: 'We keep your overall fallback pipeline (simple image features → Ridge regression → threshold optimization for QWK → apply to test) exactly the same, but make two small score-relevant fixes that typically lift QWK materially for this competition. First, we preserve and reuse the exact same `StandardScaler` learned during fitting when extracting features at inference time (your current `heuristic_predict_from_pil` bypasses scaling, which can badly miscalibrate predictions and thresholds). Second, we fit the thresholds on OOF predictions produced by the same pipeline object type used at inference (still Ridge+scaler), and we ensure the thresholds are applied to *scaled-consistent* predictions only—no change to model family or training loop, just fixing a mismatch. These changes should move your 0.6841 upward toward the 0.923 target without altering the core logic or requiring any external checkpoint.'
- What this solution (achieved 0.6809) has done: 'We keep your exact pipeline (feature extraction → scaler+ridge → OOF threshold tuning → predict test) but fix two score-relevant calibration issues that typically suppress QWK: (1) use out-of-fold predictions from the *same* model object trained on each fold to also generate per-fold test predictions, then average them (reduces distribution shift vs a single “fit on all” model), and (2) apply a monotonic “rank-to-label” mapping using the tuned thresholds on the averaged test predictions (no new model family, just more consistent inference with how thresholds were tuned). These are minimal changes confined to the fallback baseline path (used when the checkpoint is missing) and preserve the original EfficientNet inference flow unchanged. The script still runs end-to-end, stays deterministic, and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(float(out[i])))
            l2 = int(math.ceil(float(out[i])))
            pred_prob[i][l1] = 1 - (out[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob




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
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=False)
        self.backbone.global_pool = GeM(flatten=True)

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )

        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )

        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
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




## === cell 3
class photometric_distort(object):
    def __call__(self, image):
        distortions = [
            FT.adjust_brightness,
            FT.adjust_contrast,
            FT.adjust_saturation,
            FT.adjust_hue,
        ]
        random.shuffle(distortions)
        for d in distortions:
            if random.random() < 0.5:
                if d.__name__ == "adjust_hue":
                    adjust_factor = random.uniform(-16 / 255.0, 16 / 255.0)
                else:
                    adjust_factor = random.uniform(0.7, 1.3)
                image = d(image, adjust_factor)
        return image


class cropTo4_3(object):
    def __call__(self, image):
        w, h = image.size
        if (w / h) >= (4 / 3):
            new_h = h
            new_w = int(h * 4 / 3)
        else:
            new_h = int(w * 3 / 4)
            new_w = w

        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = left + new_w
        bottom = top + new_h
        return image.crop((left, top, right, bottom))


class trim(object):
    def __call__(self, image):
        bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -10)
        bbox = diff.getbbox()
        if bbox:
            return image.crop(bbox)
        return image




## === cell 4
TEST_CSV = "../input/aptos2019-blindness-detection/test.csv"
TEST_IMG_DIR = "../input/aptos2019-blindness-detection/test_images"
TRAIN_CSV = "../input/aptos2019-blindness-detection/train.csv"
TRAIN_IMG_DIR = "../input/aptos2019-blindness-detection/train_images"

test_df = pd.read_csv(TEST_CSV)
test_ids = np.squeeze(test_df["id_code"].values)

input_size = 512
transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)


def find_checkpoint():
    candidates = []
    candidates += glob.glob(
        "/kaggle/input/**/B4_3stage_11epoch_finetune2_512.pkl", recursive=True
    )
    candidates += glob.glob("/kaggle/input/**/*.pkl", recursive=True)
    candidates += glob.glob("/kaggle/input/**/*.pth", recursive=True)
    candidates += glob.glob("/kaggle/input/**/*.pt", recursive=True)

    for p in candidates:
        if os.path.basename(p) == "B4_3stage_11epoch_finetune2_512.pkl":
            return p
    return None


ckpt_path = find_checkpoint()
use_model = ckpt_path is not None

net = None
if use_model:
    print("Loading checkpoint:", ckpt_path)
    net = ThreeStage_Model()
    state = torch.load(ckpt_path, map_location=device)
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            new_state[nk] = v
        state = new_state
    net.load_state_dict(state, strict=True)
    net = net.to(device)
    net.eval()
else:
    print(
        "WARNING: Model checkpoint not found under /kaggle/input. "
        "Proceeding with a lightweight, deterministic, label-informed baseline to generate a valid submission."
    )


def _safe_imread_rgb(path):
    try:
        img = Image.open(path).convert("RGB")
        return np.asarray(img)
    except Exception:
        return None


def _fundus_fov_crop(arr_rgb: np.ndarray) -> np.ndarray:
    """
    Score-relevant for baseline: stabilizes image stats by removing black borders.
    """
    h, w = arr_rgb.shape[:2]
    small = cv2.resize(arr_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    mask = (gray > 10).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, 5)
    ys, xs = np.where(mask > 0)
    if len(xs) < 50 or len(ys) < 50:
        return arr_rgb
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = int(x0 * (w / 256.0))
    x1 = int(x1 * (w / 256.0))
    y0 = int(y0 * (h / 256.0))
    y1 = int(y1 * (h / 256.0))
    pad_x = int(0.03 * (x1 - x0 + 1))
    pad_y = int(0.03 * (y1 - y0 + 1))
    x0 = max(0, x0 - pad_x)
    x1 = min(w - 1, x1 + pad_x)
    y0 = max(0, y0 - pad_y)
    y1 = min(h - 1, y1 + pad_y)
    if x1 <= x0 or y1 <= y0:
        return arr_rgb
    return arr_rgb[y0 : y1 + 1, x0 : x1 + 1]


def extract_simple_features_rgb(arr_rgb: np.ndarray) -> np.ndarray:
    """
    Simple image-stat features (kept same core approach) for fallback Ridge + thresholds.
    """
    arr_rgb = _fundus_fov_crop(arr_rgb)

    arr = cv2.resize(arr_rgb, (256, 256), interpolation=cv2.INTER_AREA)
    gray_u8 = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    gray = gray_u8.astype(np.float32) / 255.0

    mean = float(gray.mean())
    std = float(gray.std())

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    edge_mean = float(grad.mean())
    edge_p95 = float(np.percentile(grad, 95))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray_u8).astype(np.float32) / 255.0
    clahe_mean = float(cl.mean())
    clahe_std = float(cl.std())

    lap = cv2.Laplacian(cl, cv2.CV_32F, ksize=3)
    lap_mean_abs = float(np.mean(np.abs(lap)))
    lap_p95_abs = float(np.percentile(np.abs(lap), 95))

    arrf = arr.astype(np.float32)
    r, g, b = arrf[:, :, 0], arrf[:, :, 1], arrf[:, :, 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    colorfulness = (
        float(
            np.sqrt(rg.var() + yb.var())
            + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
        )
        / 255.0
    )

    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
    sat_mean = float((hsv[:, :, 1] / 255.0).mean())

    return np.array(
        [
            mean,
            std,
            edge_mean,
            edge_p95,
            colorfulness,
            sat_mean,
            clahe_mean,
            clahe_std,
            lap_mean_abs,
            lap_p95_abs,
        ],
        dtype=np.float32,
    )


def fit_fallback_baseline(test_df_for_oof_blend: pd.DataFrame):
    """
    Minimal score-relevant upgrade (same model family/logic):
    - We already compute OOF preds to tune thresholds for QWK.
    - Change: also compute per-fold test predictions and average them.
      This keeps inference calibrated similarly to the OOF distribution used for threshold tuning,
      usually improving QWK materially vs a single fit-on-all model when using simple features.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge

    train_df = pd.read_csv(TRAIN_CSV)

    X_feats = []
    y = []
    missing_train = 0
    t0 = time.time()
    for _id, diag in zip(train_df["id_code"].values, train_df["diagnosis"].values):
        p = f"{TRAIN_IMG_DIR}/{_id}.png"
        arr = _safe_imread_rgb(p)
        if arr is None:
            missing_train += 1
            continue
        X_feats.append(extract_simple_features_rgb(arr))
        y.append(int(diag))

    if len(y) < 50:
        print(
            "Fallback baseline: not enough readable train images; reverting to constant 0 predictions."
        )
        return None, threshold, None  # (model, thresholds, blended_test_pred)

    X = np.vstack(X_feats)
    y = np.asarray(y, dtype=np.int64)
    print(
        f"Fallback baseline: using {len(y)} train images (missing/unreadable skipped: {missing_train}). "
        f"Feature time: {time.time()-t0:.1f}s"
    )

    test_feats = []
    missing_test = 0
    t1 = time.time()
    for _id in test_df_for_oof_blend["id_code"].values:
        p = f"{TEST_IMG_DIR}/{_id}.png"
        arr = _safe_imread_rgb(p)
        if arr is None:
            missing_test += 1
            test_feats.append(np.zeros((10,), dtype=np.float32))
            continue
        test_feats.append(extract_simple_features_rgb(arr))
    X_test = np.vstack(test_feats)
    print(
        f"Fallback baseline: extracted test features for {len(test_feats)} images "
        f"(missing/unreadable: {missing_test}). Time: {time.time()-t1:.1f}s"
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float32)
    test_pred_folds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        m = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=3.0, random_state=42)),
            ]
        )
        m.fit(X[tr_idx], y[tr_idx].astype(np.float32))
        oof[va_idx] = m.predict(X[va_idx]).astype(np.float32)

        test_pred_folds.append(m.predict(X_test).astype(np.float32))

    oof = np.clip(oof, 0.0, 4.0)
    test_pred_mean = np.mean(np.stack(test_pred_folds, axis=0), axis=0)
    test_pred_mean = np.clip(test_pred_mean, 0.0, 4.0)

    def apply_thr(v, thr):
        thr = np.asarray(thr, dtype=np.float32)
        return (
            (v >= thr[0]).astype(np.int32)
            + (v >= thr[1]).astype(np.int32)
            + (v >= thr[2]).astype(np.int32)
            + (v >= thr[3]).astype(np.int32)
        )

    base = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    thr = base.copy()

    def score(thr_):
        pred = apply_thr(oof, thr_)
        return float(cohen_kappa_score(y, pred, weights="quadratic"))

    best_kappa = score(thr)

    min_gap = 0.05
    for step in [0.25, 0.15, 0.10, 0.05]:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                cur = thr[i]
                for cand in (cur - step, cur, cur + step):
                    new_thr = thr.copy()
                    new_thr[i] = float(cand)
                    new_thr = np.clip(new_thr, 0.0, 4.0)
                    if not (
                        new_thr[0] + min_gap < new_thr[1]
                        and new_thr[1] + min_gap < new_thr[2]
                        and new_thr[2] + min_gap < new_thr[3]
                    ):
                        continue
                    k = score(new_thr)
                    if k > best_kappa + 1e-8:
                        best_kappa = k
                        thr = new_thr
                        improved = True

    print(
        "Fallback baseline: OOF QWK =", float(best_kappa), "thresholds =", thr.tolist()
    )

    base_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=3.0, random_state=42)),
        ]
    )
    base_model.fit(X, y.astype(np.float32))
    return base_model, thr.tolist(), test_pred_mean


fallback_model = None
fallback_thresholds = threshold
fallback_test_pred_mean = None
if not use_model:
    fallback_model, fallback_thresholds, fallback_test_pred_mean = (
        fit_fallback_baseline(test_df)
    )


def heuristic_predict_from_pil(img_pil: Image.Image) -> int:
    if fallback_model is None:
        return 0
    arr = np.asarray(img_pil.convert("RGB"))
    feats = extract_simple_features_rgb(arr).reshape(1, -1)
    pred = float(fallback_model.predict(feats)[0])
    pred = float(np.clip(pred, 0.0, 4.0))
    thr = fallback_thresholds
    if pred < thr[0]:
        return 0
    elif pred < thr[1]:
        return 1
    elif pred < thr[2]:
        return 2
    elif pred < thr[3]:
        return 3
    else:
        return 4




## === cell 5
submission_rows = []
missing = 0


def _apply_thresholds_continuous_to_class(v: float, thr):
    if v < thr[0]:
        return 0
    elif v < thr[1]:
        return 1
    elif v < thr[2]:
        return 2
    elif v < thr[3]:
        return 3
    else:
        return 4


with torch.no_grad():
    if (not use_model) and (fallback_test_pred_mean is not None):
        for idx, v in zip(test_ids, fallback_test_pred_mean.tolist()):
            pred_int = int(
                _apply_thresholds_continuous_to_class(float(v), fallback_thresholds)
            )
            submission_rows.append([idx, pred_int])
    else:
        for idx in test_ids:
            image_name = f"{TEST_IMG_DIR}/{idx}.png"
            if not os.path.exists(image_name):
                missing += 1
                pred_int = 0
            else:
                img_pil = Image.open(image_name).convert("RGB")
                if use_model:
                    img = transform(img_pil).unsqueeze(0).to(device)
                    _, r_out, _ = net(img)
                    pred = regress2class(r_out.data.squeeze(1))
                    pred_int = int(pred.item())
                else:
                    pred_int = int(heuristic_predict_from_pil(img_pil))

            submission_rows.append([idx, pred_int])

if missing > 0:
    print(
        f"Warning: {missing} test images were missing; filled with diagnosis=0 for those ids."
    )

submission = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])

submission = test_df.merge(submission, on="id_code", how="left")
submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int)

print(submission.head())
print("Submission rows:", len(submission), "Expected:", len(test_df))



## === cell 6
out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Columns:", list(submission.columns))
print("dtypes:", submission.dtypes.to_dict())
print("diagnosis value counts:\n", submission["diagnosis"].value_counts().sort_index())
