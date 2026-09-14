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

0.9092099815223468

# 6. Current score

-0.00618

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.03395) has done: 'I make the weight-loading logic robust so the notebook can run even when an external weights dataset isn’t attached, by falling back to a timm pretrained EfficientNet backbone (same architecture) and skipping the hard failure. I also fix the CUDA/CPU dtype mismatch by ensuring the loaded state_dict is applied before moving the model to GPU, and by safely handling checkpoints that store `state_dict` under different keys. Finally, I ensure inference always produces a `submission.csv` with exactly the required columns and correct row alignment to `sample_submission.csv`, so a valid Kaggle submission is always written.'
- What this solution (achieved 0.01733) has done: 'Your current score is far below the target, so we should improve real predictive signal with minimal, low-risk changes that preserve your model and inference logic. The biggest issue is that you’re not using the model’s intended “final” head (which combines classifier + regressor + ordinal), and you’re using fixed regression thresholds that are unlikely to match this checkpoint’s calibration—both can tank QWK. I (1) run the model in `final=True` mode and convert the resulting continuous score to classes by rounding (a standard, robust post-process for QWK), while keeping your transforms, architecture, and no-training approach intact. I also add deterministic/test-time safety (no behavior change) and keep the submission alignment logic exactly as you already fixed.'
- What this solution (achieved 0.15288) has done: 'Your score is far below the target, so we should improve real predictive signal with the smallest changes that keep your model and inference pipeline intact. The biggest likely issue is that the model output is being rounded with fixed implicit cutpoints (0.5 steps), which is rarely optimal for QWK; instead, we can fit 4 optimal thresholds on the training set using your model’s own continuous outputs and then apply them to test. This keeps the same architecture, weights, transforms, and “final=True” inference, but replaces only the post-processing from continuous→class in a metric-aligned way. We also keep the submission alignment logic unchanged and ensure everything runs within time by computing train predictions once and optimizing thresholds with a lightweight coordinate-descent search.'
- What this solution (achieved 0.15211) has done: 'Your current QWK (0.15288) is far below the target (0.9092), so we should improve real predictive signal with minimal changes that preserve your architecture/inference. The biggest issue is that you are fitting thresholds on *in-sample (train-on-train) predictions*, which overfits and tends to produce thresholds that generalize poorly to the test set for QWK. I keep your exact model, transforms, and “final=True” continuous inference, but fit thresholds on an out-of-fold (OOF) split so the cutpoints are calibrated on held-out predictions (same post-processing idea, just made valid). I also add a tiny monotonicity safeguard and use more stable initialization (quantiles from OOF predictions) while keeping runtime under the 600s limit by predicting train once.'
- What this solution (achieved 0.15211) has done: 'Your current score is far below the target, so we should increase real predictive signal with the smallest safe changes that keep your model and pipeline intact. The largest likely cause of poor generalization is that the OOF “threshold fitting” is not truly out-of-fold: you’re predicting each validation fold using a model that has seen those images (no fold-specific training), which makes the fitted thresholds essentially in-sample and can generalize poorly. I keep the exact same model, weights, transforms, and inference, but fit thresholds on a held-out calibration split (one stratified split) so the cutpoints are derived from genuinely unseen predictions. I also remove the time-guard “early break” (which can silently corrupt threshold fitting) while staying within 600s by calibrating on a subset split rather than 5x passes.'
- What this solution (achieved 0.08065) has done: 'We keep your exact model and preprocessing, but fix the biggest likely cause of the very low QWK: your fallback path silently uses mismatched weights (B4 backbone pretrained, but the head layers are random), which produces near-random predictions. Instead, if the external finetuned checkpoint is missing, we fall back to a fully pretrained EfficientNet-B4 and derive a continuous score from its 5-class logits (expected value), preserving the same continuous→thresholded class post-processing and submission semantics. Additionally, we fit thresholds on a slightly larger, stratified calibration split and use the *true label quantiles* as a safer initialization so threshold fitting doesn’t collapse when predictions are poorly spread. These are minimal, score-relevant changes and should move the score materially toward your target without changing architecture/training loops when the real weights are present.'
- What this solution (achieved 0.09174) has done: 'Your current QWK is far below target, so we should improve predictive signal without changing your model or preprocessing. The biggest low-risk issue is that your fallback path uses an ImageNet-pretrained 5-class head (not DR-trained), so its outputs are nearly random for this competition; instead we make the fallback use a DR-pretrained checkpoint available in the environment (the `timm` “aptos2019” pretrained weights for EfficientNet-B4), keeping the same “continuous expected value → thresholded classes” semantics. Additionally, we calibrate thresholds using a smaller, stratified calibration subset to keep runtime under 600s while reducing noise from a very large 50% split. These changes keep architecture/training/inference logic intact and should move score meaningfully upward when external finetuned weights aren’t present.'
- What this solution (achieved 0.0718) has done: 'Your current score is far below the target, so we should improve real signal while keeping your model and pipeline intact. The biggest low-risk win is to switch inference (and calibration) to standard test-time augmentation via simple horizontal flip and average the continuous outputs, which usually improves QWK without changing training or architecture. I also make calibration thresholds fit on the *same* TTA-averaged continuous predictions (so thresholds match inference), and keep the rest of your submission alignment logic unchanged. All changes are minimal, deterministic, and should run within the 600s limit.'
- What this solution (achieved 0.07853) has done: 'Your current QWK (0.0718) is far below the target (0.9092), so we should increase real signal with the smallest, metric-aligned changes while keeping your model and preprocessing intact. The biggest likely issue is that you’re always calibrating thresholds on only 20% of train, which makes the cutpoints noisy/unstable (especially with TTA) and can seriously hurt generalization; we can safely use a larger stratified calibration split without changing architecture or any training. I increase the calibration fraction and make the threshold search slightly more expressive (more grid points) so it can find better cutpoints for QWK, while keeping the same coordinate-descent method and TTA inference. Everything else (paths, transforms, model, and submission alignment) stays the same.'
- What this solution (achieved -0.00154) has done: 'Your current score (0.07853) is far below the target (0.90921), so we should improve real predictive signal with minimal changes that keep your model and preprocessing intact. The largest likely issue is that your test-time normalization stats don’t match the timm EfficientNet defaults (and especially don’t match timm pretrained weights), which can severely degrade predictions; I switch the Normalize mean/std to timm’s recommended values for the active model while keeping the same resize/crop/trim pipeline. To preserve evaluation semantics and avoid extra compute, I keep your model, TTA, and threshold-fitting approach unchanged, but I also fit thresholds using the same normalization that inference uses (automatically, since transforms are shared). This is a small, metric-relevant change that typically yields a big jump when using timm pretrained backbones and should move you materially toward the target.'
- What this solution (achieved -0.01191) has done: 'Your score is far below the target, so we should make small, metric-aligned fixes that improve real signal without changing the model architecture or training approach. The largest likely issue is a mismatch between the calibration subset (`cal_df`) and the images actually being predicted (missing files / unreadable images), which can silently break threshold fitting and produce near-random class cutpoints; we filter the calibration split to only rows with existing images and use a deterministic, safe image loader. Next, we make threshold fitting more stable by fitting on a slightly larger but still bounded calibration set (capped) and using a stratified split that preserves label distribution, keeping the same coordinate-descent logic. Finally, we ensure the test dataloader order exactly matches `sample_submission.csv` (not just `test.csv`) so predictions can’t be misaligned, which can destroy QWK even if the model is good.'
- What this solution (achieved -0.00618) has done: 'Your current QWK is far below the target, so we should improve real predictive signal while keeping your exact model/transform/inference structure intact. The most likely score-killer here is that the external finetuned weights are either missing or only partially loading (your `strict=False` can leave random head layers), which yields near-random predictions even though the code “runs.” I (1) add a minimal sanity check to detect partial weight loads and automatically fall back to timm’s DR-pretrained (`pretrained="aptos2019"`) EfficientNet when the three-stage checkpoint isn’t fully usable, and (2) make threshold calibration use a smaller but label-stratified calibration set (still deterministic) to stay within the time limit while reducing noisy threshold fits. This preserves your architecture and evaluation semantics; it only prevents accidental random-head inference and stabilizes the metric-aligned post-processing.'

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
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit
import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(max(1, os.cpu_count() // 2))

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor) -> torch.Tensor:
    prediction = torch.zeros(out.size(0), device="cpu")
    out_cpu = out.detach().view(-1).cpu()
    for i in range(4):
        prediction += (out_cpu >= threshold[i]).to(torch.float32)
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
    out = out.detach().view(-1)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(float(out[i])))
            l2 = int(math.ceil(float(out[i])))
            pred_prob[i][l1] = 1 - (out[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob


def continuous_to_class_round(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().view(-1)
    return torch.clamp(torch.round(x), 0, 4).to(torch.int64)


def continuous_to_class_thresholds(x: np.ndarray, thr) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    t0, t1, t2, t3 = map(float, thr)
    return (
        (x >= t0).astype(np.int32)
        + (x >= t1).astype(np.int32)
        + (x >= t2).astype(np.int32)
        + (x >= t3).astype(np.int32)
    )


def qwk(y_true, y_pred) -> float:
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def fit_thresholds_coordinate_descent(
    y_true: np.ndarray,
    y_cont: np.ndarray,
    init_thr=None,
    n_passes: int = 6,
    grid_size: int = 60,
    margin: float = 1e-3,
):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_cont = np.asarray(y_cont, dtype=np.float32)

    if init_thr is None:
        qs = [0.2, 0.4, 0.6, 0.8]
        init_thr = [float(np.quantile(y_cont, q)) for q in qs]
        init_thr = sorted(init_thr)
        for i in range(1, 4):
            if init_thr[i] <= init_thr[i - 1] + margin:
                init_thr[i] = init_thr[i - 1] + margin

    thr = list(map(float, init_thr))
    thr = sorted(thr)
    for i in range(1, 4):
        if thr[i] <= thr[i - 1] + margin:
            thr[i] = thr[i - 1] + margin

    best_pred = continuous_to_class_thresholds(y_cont, thr)
    best_score = qwk(y_true, best_pred)

    ymin, ymax = float(np.min(y_cont)), float(np.max(y_cont))
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymin == ymax:
        return thr, best_score

    for _ in range(n_passes):
        improved = False
        for k in range(4):
            lo = ymin if k == 0 else thr[k - 1] + margin
            hi = ymax if k == 3 else thr[k + 1] - margin
            if hi <= lo:
                continue

            cand = np.linspace(lo, hi, grid_size, dtype=np.float32)
            local_best_t = thr[k]
            local_best_score = best_score

            for t in cand:
                test_thr = thr.copy()
                test_thr[k] = float(t)
                pred = continuous_to_class_thresholds(y_cont, test_thr)
                score = qwk(y_true, pred)
                if score > local_best_score + 1e-12:
                    local_best_score = score
                    local_best_t = float(t)

            if local_best_score > best_score + 1e-12:
                thr[k] = local_best_t
                best_score = local_best_score
                improved = True

        if not improved:
            break

    return thr, best_score




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
BASE_INPUT = "../input/aptos2019-blindness-detection"
if not os.path.exists(BASE_INPUT):
    if os.path.exists("/kaggle/input/aptos2019-blindness-detection"):
        BASE_INPUT = "/kaggle/input/aptos2019-blindness-detection"
    elif os.path.exists("/kaggle/data/aptos2019-blindness-detection"):
        BASE_INPUT = "/kaggle/data/aptos2019-blindness-detection"

TEST_CSV = os.path.join(BASE_INPUT, "test.csv")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")

TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")

SAMPLE_CSV = os.path.join(BASE_INPUT, "sample_submission.csv")
sample_df = pd.read_csv(SAMPLE_CSV)
test_ids = sample_df["id_code"].astype(str).values

input_size = 380


def find_weight_file():
    candidates = [
        "../input/weights/B4_3stage_58epoch_CLAHE.pkl",
        "../input/weights/B4_3stage_58epoch_CLAHE.pth",
        "../input/weights/B4_3stage_58epoch_CLAHE.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    patterns = [
        "../input/**/B4_3stage_58epoch_CLAHE.pkl",
        "../input/**/B4_3stage_58epoch_CLAHE.pth",
        "../input/**/B4_3stage_58epoch_CLAHE.pt",
    ]
    for pat in patterns:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return hits[0]
    return None


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            return ckpt["state_dict"]
        if "model" in ckpt and isinstance(ckpt["model"], dict):
            return ckpt["model"]
    return ckpt


class PretrainedClassifierAsContinuous(nn.Module):
    def __init__(self, model_name: str, pretrained: bool, num_classes: int = 5):
        super().__init__()
        self.m = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x, final=True):
        logits = self.m(x)  # [B,5]
        probs = torch.softmax(logits, dim=1)
        cls = torch.arange(5, device=probs.device, dtype=probs.dtype).view(1, 5)
        cont = (probs * cls).sum(dim=1, keepdim=True)  # [B,1] in [0,4]
        return cont


def _load_fallback_aptos_model():
    aptos_model_name = "tf_efficientnet_b4_ns"
    try:
        m = PretrainedClassifierAsContinuous(
            aptos_model_name, pretrained="aptos2019", num_classes=5
        )
        print(
            f"Falling back to timm pretrained weights: {aptos_model_name} (pretrained='aptos2019')"
        )
        cfg_local = timm.data.resolve_model_data_config(m.m)
        return m, cfg_local["mean"], cfg_local["std"], False
    except Exception as e:
        print(
            f"Warning: could not load pretrained='aptos2019' for {aptos_model_name}: {e}"
        )
        print("Falling back to ImageNet-pretrained weights (pretrained=True).")
        m = PretrainedClassifierAsContinuous(
            aptos_model_name, pretrained=True, num_classes=5
        )
        cfg_local = timm.data.resolve_model_data_config(m.m)
        return m, cfg_local["mean"], cfg_local["std"], False


weight_path = find_weight_file()
using_finetuned_three_stage = False

if weight_path is not None:
    net = ThreeStage_Model()
    ckpt = torch.load(weight_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)

    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights from: {weight_path}")
    if len(missing) > 0:
        print(f"Warning: missing keys (showing up to 12): {missing[:12]}")
    if len(unexpected) > 0:
        print(f"Warning: unexpected keys (showing up to 12): {unexpected[:12]}")

    critical_prefixes = ("classifier.", "regressor.", "ordinal.", "final_regressor.")
    critical_missing = [k for k in missing if k.startswith(critical_prefixes)]
    if len(critical_missing) > 0:
        print(
            f"Warning: checkpoint appears partial (missing {len(critical_missing)} head keys). "
            "Switching to aptos2019-pretrained fallback to avoid random-head predictions."
        )
        net, mean, std, using_finetuned_three_stage = _load_fallback_aptos_model()
    else:
        using_finetuned_three_stage = True
        cfg = timm.data.resolve_model_data_config(net.backbone)
        mean = cfg["mean"]
        std = cfg["std"]
else:
    print("Warning: Could not find external finetuned weights under ../input/.")
    net, mean, std, using_finetuned_three_stage = _load_fallback_aptos_model()

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

net = net.to(device)
net.eval()

print(f"Using device={device}")
print(f"Normalization mean={mean}, std={std}")
print(
    f"Found {len(test_ids)} test ids, images dir exists={os.path.exists(TEST_IMG_DIR)}"
)
print(f"Using finetuned three-stage checkpoint: {using_finetuned_three_stage}")




## === cell 5
def _safe_open_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform=None):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_code = self.ids[idx]
        image_name = os.path.join(self.img_dir, f"{id_code}.png")
        img = _safe_open_rgb(image_name)
        if self.transform is not None:
            img = self.transform(img)
        return id_code, img


class TrainDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        id_code = str(row["id_code"])
        y = int(row["diagnosis"])
        image_name = os.path.join(self.img_dir, f"{id_code}.png")
        img = _safe_open_rgb(image_name)
        if self.transform is not None:
            img = self.transform(img)
        return y, img


def predict_continuous(dataloader, use_tta: bool = True):
    outs = []
    with torch.no_grad():
        for batch in dataloader:
            batch_imgs = batch[1].to(device, non_blocking=True)
            out1 = net(batch_imgs, final=True).squeeze(1)

            if use_tta:
                batch_imgs_flip = torch.flip(batch_imgs, dims=[3])
                out2 = net(batch_imgs_flip, final=True).squeeze(1)
                final_out = 0.5 * (out1 + out2)
            else:
                final_out = out1

            outs.append(final_out.detach().float().cpu().numpy())
    return np.concatenate(outs, axis=0)


thr_to_use = None
if os.path.exists(TRAIN_CSV) and os.path.exists(TRAIN_IMG_DIR):
    train_df = pd.read_csv(TRAIN_CSV)
    train_df["id_code"] = train_df["id_code"].astype(str)

    img_paths = train_df["id_code"].apply(
        lambda s: os.path.join(TRAIN_IMG_DIR, f"{s}.png")
    )
    exists_mask = img_paths.apply(os.path.exists).values
    if not np.all(exists_mask):
        print(
            f"Warning: filtering out {(~exists_mask).sum()} train rows with missing images."
        )
    train_df = train_df.loc[exists_mask].reset_index(drop=True)

    y_true = train_df["diagnosis"].astype(int).values

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
    cal_idx, _ = next(splitter.split(np.zeros(len(y_true)), y_true))
    cal_df = train_df.iloc[cal_idx].reset_index(drop=True)

    max_cal = 2200
    if len(cal_df) > max_cal:
        cal_df = (
            cal_df.groupby("diagnosis", group_keys=False)
            .apply(
                lambda g: g.sample(
                    n=max(1, int(round(max_cal * len(g) / len(cal_df)))),
                    random_state=42,
                )
            )
            .reset_index(drop=True)
        )

    cal_y = cal_df["diagnosis"].astype(int).values

    cal_ds = TrainDataset(cal_df, TRAIN_IMG_DIR, transform=transform)
    cal_dl = DataLoader(
        cal_ds,
        batch_size=8 if device.startswith("cuda") else 4,
        shuffle=False,
        num_workers=2,
        pin_memory=device.startswith("cuda"),
    )

    t0 = time.time()
    cal_cont = predict_continuous(cal_dl, use_tta=True)
    print(
        f"Calibration prediction done in {time.time()-t0:.1f}s on {len(cal_df)} images"
    )

    init_thr = []
    for k in [0, 1, 2, 3]:
        left = cal_cont[cal_y == k]
        right = cal_cont[cal_y == (k + 1)]
        if len(left) == 0 or len(right) == 0:
            init_thr.append(float(np.quantile(cal_cont, (k + 1) / 5.0)))
        else:
            init_thr.append(float(0.5 * (np.median(left) + np.median(right))))
    init_thr = sorted(init_thr)
    for i in range(1, 4):
        if init_thr[i] <= init_thr[i - 1] + 1e-3:
            init_thr[i] = init_thr[i - 1] + 1e-3

    thr_to_use, cal_qwk = fit_thresholds_coordinate_descent(
        cal_y, cal_cont, init_thr=init_thr, n_passes=6, grid_size=120
    )
    cal_pred_qwk = qwk(cal_y, continuous_to_class_thresholds(cal_cont, thr_to_use))
    print(f"Init thresholds (label-informed): {init_thr}")
    print(
        f"Fitted thresholds (cal-optimized): {thr_to_use} | cal QWK={cal_qwk:.5f} | recomputed={cal_pred_qwk:.5f}"
    )
else:
    thr_to_use = [0.5, 1.5, 2.5, 3.5]
    print("Warning: train files not found; using default thresholds:", thr_to_use)

ds = TestDataset(test_ids, TEST_IMG_DIR, transform=transform)
dl = DataLoader(
    ds,
    batch_size=8 if device.startswith("cuda") else 4,
    shuffle=False,
    num_workers=2,
    pin_memory=device.startswith("cuda"),
)

test_cont = predict_continuous(dl, use_tta=True)
test_preds = continuous_to_class_thresholds(test_cont, thr_to_use)

submission_rows = [
    [str(id_code), int(pred)] for id_code, pred in zip(test_ids, test_preds)
]
submission = np.array(submission_rows, dtype=object)
print("Predictions generated:", submission.shape)



## === cell 6
df = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

sample_path = os.path.join(BASE_INPUT, "sample_submission.csv")
sample = pd.read_csv(sample_path)
df = sample[["id_code"]].merge(df, on="id_code", how="left")

if df["diagnosis"].isna().any():
    missing = df.loc[df["diagnosis"].isna(), "id_code"].head(5).tolist()
    raise RuntimeError(
        f"Some test ids did not get predictions (missing images or id mismatch). Examples: {missing}"
    )

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", df.shape)
print(df.head())
