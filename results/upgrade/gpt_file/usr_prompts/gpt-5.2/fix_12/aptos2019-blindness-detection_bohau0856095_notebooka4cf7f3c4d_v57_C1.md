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

0.9058477549711728

# 6. Current score

0.04639

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix three execution blockers while keeping the modeling/prediction logic the same: (1) remove the broken `pip install` of an unavailable timm wheel and rely on the already-installed `timm`, (2) make device selection robust so it runs on CPU when no GPU is available, and (3) load the pretrained weights only if present; otherwise it still run end-to-end with randomly initialized weights (score be poor, but it produce a valid submission). I also fix a couple of transform bugs that can silently break preprocessing (`is` vs `==` and `trim()` returning `None`) and speed/stability issues by using `torch.no_grad()` and avoiding per-image print spam. Finally, I ensure the submission is always non-empty and matches the required column names and `.csv` suffix.'
- What this solution (achieved 0.0) has done: 'I fix the EfficientNet feature-dimension mismatch that causes the linear layer shape error by correctly taking the backbone’s feature dimension (using `classifier.in_features` when `num_features` is not reliable) and resetting the backbone classifier head so `forward()` returns feature vectors compatible with your custom heads. I keep the model architecture and inference logic the same, only adjusting how features are extracted so the existing heads receive the expected tensor shape. I also make the test image directory resolution a bit more robust to avoid path issues, without changing I/O locations. This should make the notebook run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved -0.15779) has done: 'Your score is 0.0 mainly because the code often runs with randomly initialized weights (no pretrained checkpoint found), which makes predictions essentially random for QWK. The smallest, score-relevant change is to load a standard ImageNet-pretrained EfficientNet backbone (same architecture) when the competition-specific weights are missing; this improves feature quality without changing your heads, loss, or inference logic. I also make weight loading tolerant to common checkpoint key prefixes so that if the file is present, it actually loads into the right submodules. Everything else (transforms, thresholds, regress2class mapping, and submission format) stays the same, and it still produce `submission.csv`.'
- What this solution (achieved -0.05005) has done: 'Your current score is far below the target, so we should make the smallest score-relevant change that improves QWK without changing the model/training core (and you currently do no training). The biggest issue is that you’re using only the regressor head (`r_out`) with fixed thresholds, while your model also has a 5-class classifier head (`c_out`) that can provide a more stable ordinal signal when weights are missing/mismatched. I keep the exact same model definition and weight-loading logic, but change inference to fuse `classifier` and `regressor` outputs into a single continuous prediction (expected value from class probs blended with regressor), then apply the same thresholding to get 0–4. I also add a tiny, deterministic calibration step that aligns the regressor scale to the classifier expected value on-the-fly (per-image blend only, no data leakage), which typically reduces extreme randomness and should move QWK upward toward your target.'
- What this solution (achieved 0.01979) has done: 'Your current QWK (-0.05005) is far below the target (0.9058), so we need a small change that legitimately improves signal without changing the model/training core (you do inference-only). The biggest score-relevant issue is that your regressor head output is never passed through a sigmoid+scale at inference time (unlike in `forward(final=False)`), so `r_val` is essentially an uncalibrated linear output and your fixed thresholds become meaningless. I keep the exact same model and fusion logic, but apply the same regressor activation used in the model (`sigmoid()*4.5`) before fusing and thresholding. This should move predictions from near-random toward sensible 0–4 ordinal outputs and improve QWK substantially toward the target.'
- What this solution (achieved -0.05005) has done: 'Your current score is far below the target, so the smallest safe improvement is to fix an inference bug that makes your regressor output effectively “double-sigmoided” (you already sigmoid inside `forward`, but you sigmoid again in cell 5). I keep your model, thresholds, fusion logic, and transforms the same, but use the already-calibrated `r_out` directly (just clamp) so the fused continuous prediction lands on a sensible 0–4 scale and improves QWK. I also make a tiny safety change to ensure test ids and output rows preserve the exact order of `test.csv`, preventing any accidental misalignment. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.00338) has done: 'Your current score is extremely far below the target, so the smallest legitimate boost is to make your final discrete label mapping better aligned with QWK without changing the model or weights. I keep your exact model, transforms, and inference outputs, but replace the fixed thresholds with thresholds optimized on a small internal validation split from `train.csv` using the same fused continuous prediction you already compute. This does not change the architecture or training (still inference-only); it only calibrates the continuous-to-ordinal conversion to maximize kappa on held-out data, which typically yields a large jump from near-random. I also keep a safe fallback to your original thresholds if optimization fails, and preserve the `test.csv` row order in the submission.'
- What this solution (achieved -0.01917) has done: 'Your score is far below the target, so we should improve QWK with the smallest change that doesn’t alter your model or training (you do inference-only). The main lever left is better, more robust threshold calibration: your current threshold optimization uses a small capped validation set and a coarse/local grid, which can easily produce weak thresholds and unstable generalization. I keep your fused continuous prediction exactly the same, but (1) optimize thresholds on a larger validation set (still held-out, no leakage) and (2) use a slightly stronger yet still lightweight coordinate-descent search with multiple passes and adaptive span, which usually yields a large QWK lift for ordinal problems. Everything else (architecture, weights loading, transforms, submission format and order) stays unchanged.'
- What this solution (achieved 0.00792) has done: 'Your current score is extremely far below the target, so we should make the smallest change that legitimately improves QWK without altering your model architecture or training approach (still inference-only). The biggest remaining score lever is to replace the unstable single split threshold fitting with deterministic out-of-fold (OOF) threshold calibration: we compute fused continuous predictions once per image on a few stratified folds, then optimize thresholds on the aggregated OOF predictions (no leakage) and apply them to test. This keeps the exact same “continuous → 4 thresholds → {0..4}” semantics and the exact same fused continuous prediction function, but makes threshold estimation much more robust and typically yields a large QWK jump. I also keep runtime bounded by limiting the calibration set size deterministically while ensuring all classes are represented.'
- What this solution (achieved 0.04639) has done: 'I make one score-relevant calibration change while preserving your model, transforms, and inference pipeline: calibrate the class/regressor fusion weight (`alpha`) on out-of-fold (OOF) predictions together with the thresholds, instead of using a fixed heuristic alpha. This keeps the exact same semantics (continuous fused score → 4 thresholds → {0..4}) but makes the fusion better aligned with QWK on held-out data, which should move your score upward toward the target. I implement a lightweight, deterministic grid search over a small set of alpha values using the same OOF sample you already compute, then optimize thresholds for the best alpha and use them for test inference. No training, no architecture changes, and it still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
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


def regress2class(out: torch.Tensor) -> torch.Tensor:
    prediction = torch.zeros(out.size(0), device="cpu")
    out_cpu = out.detach().view(-1).cpu()
    for i in range(4):
        prediction += (out_cpu >= threshold[i]).to(torch.float32)
    return prediction


def ordinal2class_prob(out: torch.Tensor) -> torch.Tensor:
    dev = out.device
    pred_prob = torch.zeros(out.size(0), 5, device=dev)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out: torch.Tensor) -> torch.Tensor:
    dev = out.device
    pred_prob = torch.zeros((out.size(0), 5), device=dev)
    out_flat = out.view(-1)
    for i in range(out_flat.size(0)):
        oi = float(out_flat[i].detach().cpu())
        if oi < 4.0:
            l1 = int(math.floor(oi))
            l2 = int(math.ceil(oi))
            pred_prob[i, l1] = 1 - (oi - l1)
            pred_prob[i, l2] = 1 - (l2 - oi)
        else:
            pred_prob[i, 4] = 1.0
    return pred_prob




## === cell 2
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.models.tf_efficientnet_b5_ns(pretrained=True)
        self.backbone.reset_classifier(0, global_pool="")
        self.backbone.global_pool = GeM(flatten=True)

        in_features = getattr(self.backbone, "num_features", None)
        if in_features is None or in_features == 1000:
            in_features = getattr(
                getattr(self.backbone, "classifier", None), "in_features", in_features
            )
        if in_features is None:
            raise RuntimeError("Could not infer in_features for Regressor backbone.")

        self.regressor = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None, pretrained_backbone: bool = False):
        super().__init__()
        self.backbone = timm.models.tf_efficientnet_b4_ns(
            pretrained=pretrained_backbone
        )
        self.backbone.reset_classifier(0, global_pool="")
        self.backbone.global_pool = GeM(flatten=True)

        in_features = getattr(self.backbone, "num_features", None)
        clf = getattr(self.backbone, "classifier", None)
        if hasattr(clf, "in_features"):
            in_features = clf.in_features
        if in_features is None:
            raise RuntimeError(
                "Could not infer in_features for ThreeStage_Model backbone."
            )

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
        return image.crop(bbox) if bbox else image




## === cell 4
BASE1 = "../input/aptos2019-blindness-detection"
BASE2 = "/kaggle/data/aptos2019-blindness-detection"
BASE3 = "/kaggle/input/aptos2019-blindness-detection"
base = BASE1 if os.path.exists(BASE1) else (BASE2 if os.path.exists(BASE2) else BASE3)

test_csv_path = os.path.join(base, "test.csv")
train_csv_path = os.path.join(base, "train.csv")
test_img_dir = os.path.join(base, "test_images")
train_img_dir = os.path.join(base, "train_images")

if not os.path.isdir(test_img_dir) and os.path.isdir(
    os.path.join(test_img_dir, "test_images")
):
    test_img_dir = os.path.join(test_img_dir, "test_images")
if not os.path.isdir(train_img_dir) and os.path.isdir(
    os.path.join(train_img_dir, "train_images")
):
    train_img_dir = os.path.join(train_img_dir, "train_images")

WEIGHTS_FILENAME = "B4_3stage_56epoch_CLAHE.pkl"
CANDIDATE_WEIGHT_PATHS = [
    "../input/weights/B4_3stage_56epoch_CLAHE.pkl",  # original
    f"../input/{WEIGHTS_FILENAME}",
    f"/kaggle/input/{WEIGHTS_FILENAME}",
]
KAGGLE_INPUT_ROOT = "/kaggle/input"
if os.path.isdir(KAGGLE_INPUT_ROOT):
    for d in os.listdir(KAGGLE_INPUT_ROOT):
        CANDIDATE_WEIGHT_PATHS.append(
            os.path.join(KAGGLE_INPUT_ROOT, d, WEIGHTS_FILENAME)
        )

weights_path = None
for p in CANDIDATE_WEIGHT_PATHS:
    if os.path.exists(p):
        weights_path = p
        break

test_df = pd.read_csv(test_csv_path)
test_ids = test_df["id_code"].astype(str).values.tolist()

train_df = pd.read_csv(train_csv_path)
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["diagnosis"] = train_df["diagnosis"].astype(int)

input_size = 380
transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

use_pretrained_backbone = weights_path is None
net = ThreeStage_Model(pretrained_backbone=use_pretrained_backbone)


def _normalize_state_dict_keys_for_net(state_dict: dict) -> dict:
    keys = list(state_dict.keys())
    if any(k.startswith("module.") for k in keys):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        keys = list(state_dict.keys())

    if any(k.startswith("model.") for k in keys):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        keys = list(state_dict.keys())

    if any(k.startswith("net.") for k in keys):
        state_dict = {k.replace("net.", "", 1): v for k, v in state_dict.items()}
        keys = list(state_dict.keys())

    return state_dict


if weights_path is not None and os.path.exists(weights_path):
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    if isinstance(state, dict):
        state = _normalize_state_dict_keys_for_net(state)

    missing, unexpected = net.load_state_dict(state, strict=False)
    print("Loaded weights:", weights_path)
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
else:
    print("WARNING: weights file not found in candidates:")
    for p in CANDIDATE_WEIGHT_PATHS:
        print(" -", p)
    print(
        "Using ImageNet-pretrained EfficientNet backbone (heads randomly initialized). "
        "This should score better than fully-random weights while still producing a valid submission."
    )

net = net.to(device)
net.eval()



## === cell 5
FUSION_ALPHA = 0.62  # default fallback; will be replaced by OOF tuning if successful


def fused_continuous_from_outputs(
    c_out: torch.Tensor, r_out: torch.Tensor, alpha: float
) -> torch.Tensor:
    c_prob = F.softmax(c_out, dim=1)
    classes = torch.arange(5, device=c_prob.device, dtype=c_prob.dtype).view(1, -1)
    c_ev = (c_prob * classes).sum(dim=1)  # (B,)
    r_val = r_out.view(-1).clamp(
        0.0, 4.0
    )  # r_out already sigmoid-scaled in model forward
    fused = alpha * c_ev + (1.0 - alpha) * r_val
    return fused.view(-1)


def fused_continuous_from_image_tensor(img_tensor: torch.Tensor) -> torch.Tensor:
    c_out, r_out, _ = net(img_tensor)
    return fused_continuous_from_outputs(c_out, r_out, FUSION_ALPHA)


def apply_thresholds_to_continuous(y_cont: np.ndarray, thr: np.ndarray) -> np.ndarray:
    y = np.zeros_like(y_cont, dtype=np.int64)
    for t in thr:
        y += (y_cont >= t).astype(np.int64)
    return y.clip(0, 4)


def optimize_thresholds_qwk(
    y_true: np.ndarray, y_cont: np.ndarray, init_thr: np.ndarray
) -> np.ndarray:
    thr = init_thr.astype(np.float64).copy()
    thr.sort()
    best_thr = thr.copy()

    def score_for(thr_vec: np.ndarray) -> float:
        return cohen_kappa_score(
            y_true, apply_thresholds_to_continuous(y_cont, thr_vec), weights="quadratic"
        )

    best_score = score_for(best_thr)

    spans = [0.60, 0.40, 0.25, 0.15]
    grid_n = 61

    for span in spans:
        improved_any = False
        for _pass in range(3):
            improved = False
            for i in range(4):
                lo = 0.0 if i == 0 else (best_thr[i - 1] + 1e-3)
                hi = 4.0 if i == 3 else (best_thr[i + 1] - 1e-3)
                if hi <= lo:
                    continue

                center = float(best_thr[i])
                lo2 = max(lo, center - span)
                hi2 = min(hi, center + span)
                if hi2 <= lo2:
                    continue

                grid = np.linspace(lo2, hi2, grid_n)
                local_best_score = best_score
                local_best_thr = best_thr[i]

                for cand in grid:
                    thr_cand = best_thr.copy()
                    thr_cand[i] = cand
                    thr_cand.sort()
                    s = score_for(thr_cand)
                    if s > local_best_score + 1e-7:
                        local_best_score = s
                        local_best_thr = cand

                if local_best_score > best_score + 1e-7:
                    best_thr[i] = local_best_thr
                    best_thr.sort()
                    best_score = local_best_score
                    improved = True
                    improved_any = True

            if not improved:
                break

        if not improved_any:
            break

    return best_thr


def _sample_calibration_df(
    train_df: pd.DataFrame, max_n: int, seed: int = 42
) -> pd.DataFrame:
    if len(train_df) <= max_n:
        return train_df.reset_index(drop=True)

    parts = []
    for cls, g in train_df.groupby("diagnosis"):
        take = max(30, int(round(max_n * len(g) / len(train_df))))
        take = min(take, len(g))
        parts.append(g.sample(n=take, random_state=seed))
    out = (
        pd.concat(parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    if len(out) > max_n:
        out = out.iloc[:max_n].reset_index(drop=True)
    return out


def _predict_outputs_for_ids(id_list, img_dir, transform, device, net):
    c_logits = np.zeros((len(id_list), 5), dtype=np.float32)
    r_vals = np.zeros((len(id_list),), dtype=np.float32)
    with torch.no_grad():
        for i, idx in enumerate(id_list):
            image_name = os.path.join(img_dir, f"{idx}.png")
            img = Image.open(image_name).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)
            c_out, r_out, _ = net(img_t)
            c_logits[i] = c_out.detach().cpu().numpy().reshape(-1)
            r_vals[i] = float(r_out.view(-1).detach().cpu().item())
    return c_logits, r_vals


CALIB_MAX = 1400  # keep bounded runtime
calib_df = _sample_calibration_df(train_df, CALIB_MAX, seed=42)

skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
oof_true = calib_df["diagnosis"].values.astype(np.int64)
oof_c_logits = np.zeros((len(calib_df), 5), dtype=np.float32)
oof_r_vals = np.zeros((len(calib_df),), dtype=np.float32)

t0 = time.time()
for fold, (tr_i, va_i) in enumerate(skf.split(calib_df["id_code"].values, oof_true), 1):
    va_ids = calib_df.iloc[va_i]["id_code"].values.tolist()
    c_logits, r_vals = _predict_outputs_for_ids(
        va_ids, train_img_dir, transform, device, net
    )
    oof_c_logits[va_i] = c_logits
    oof_r_vals[va_i] = r_vals
    print(f"OOF fold {fold}/4 done (n_val={len(va_i)})")

base_thr = np.array(threshold, dtype=np.float64)

alpha_grid = np.array([0.45, 0.52, 0.58, 0.62, 0.66, 0.70, 0.75], dtype=np.float64)

best_alpha = float(FUSION_ALPHA)
best_thr = base_thr.copy()
best_kappa = -1e9

oof_c_prob = (
    torch.softmax(torch.from_numpy(oof_c_logits), dim=1).numpy().astype(np.float64)
)
oof_c_ev = (oof_c_prob * np.arange(5, dtype=np.float64).reshape(1, -1)).sum(axis=1)
oof_r = oof_r_vals.astype(np.float64).clip(0.0, 4.0)

for a in alpha_grid:
    y_cont = (a * oof_c_ev + (1.0 - a) * oof_r).astype(np.float64)
    try:
        thr_a = optimize_thresholds_qwk(oof_true, y_cont, base_thr)
        pred_a = apply_thresholds_to_continuous(y_cont, thr_a)
        kappa_a = cohen_kappa_score(oof_true, pred_a, weights="quadratic")
        if np.isfinite(kappa_a) and kappa_a > best_kappa + 1e-9:
            best_kappa = float(kappa_a)
            best_alpha = float(a)
            best_thr = thr_a.copy()
    except Exception:
        continue

try:
    y_cont_base = (
        float(FUSION_ALPHA) * oof_c_ev + (1.0 - float(FUSION_ALPHA)) * oof_r
    ).astype(np.float64)
    thr_base = optimize_thresholds_qwk(oof_true, y_cont_base, base_thr)
    pred_base = apply_thresholds_to_continuous(y_cont_base, thr_base)
    kappa_base = cohen_kappa_score(oof_true, pred_base, weights="quadratic")
except Exception:
    kappa_base = -1e9
    thr_base = base_thr.copy()

if np.isfinite(best_kappa) and (best_kappa >= kappa_base - 1e-6):
    FUSION_ALPHA = best_alpha
    threshold = best_thr.tolist()
    print(f"OOF tuned alpha={FUSION_ALPHA:.3f}, thresholds={threshold}")
    print(
        f"OOF QWK base(alpha={float(FUSION_ALPHA):.3f} tuned)={best_kappa:.5f} vs baseline(alpha_default)={kappa_base:.5f} "
        f"(n_calib={len(calib_df)}) in {time.time()-t0:.1f}s"
    )
else:
    threshold = thr_base.tolist()
    print(
        f"Keeping default alpha={FUSION_ALPHA:.3f}; using baseline optimized thresholds={threshold}"
    )
    print(
        f"OOF QWK tuned={best_kappa:.5f} baseline={kappa_base:.5f} (n_calib={len(calib_df)}) in {time.time()-t0:.1f}s"
    )



## === cell 6
submission = []

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        image_name = os.path.join(test_img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        fused = fused_continuous_from_image_tensor(img)
        pred = regress2class(fused)
        submission.append([idx, int(pred.item())])

df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
if df.empty:
    raise RuntimeError("Submission DataFrame is empty; inference produced no rows.")
df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

df = test_df[["id_code"]].merge(df, on="id_code", how="left")
if df["diagnosis"].isna().any():
    raise RuntimeError(
        "Some test ids have missing predictions after merge; check id alignment."
    )
df["diagnosis"] = df["diagnosis"].astype(int)

if len(df) != len(test_df):
    raise RuntimeError(f"Row count mismatch: submission={len(df)} test={len(test_df)}")

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print("Final FUSION_ALPHA used:", FUSION_ALPHA)
print("Final thresholds used:", threshold)
print(df.head())
