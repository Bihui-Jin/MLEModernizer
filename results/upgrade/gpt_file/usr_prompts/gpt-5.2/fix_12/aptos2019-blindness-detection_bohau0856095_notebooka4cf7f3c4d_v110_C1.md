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

0.923776140726992

# 6. Current score

0.28762

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the code robust to the Kaggle environment by (1) removing the hard dependency on a missing `../input/weights/` directory and instead running with the same model architecture but without loading external weights, and (2) selecting `cuda` only if it’s actually available (otherwise run on CPU) to fix the “no NVIDIA driver” crash. I also fix a couple of small runtime hazards in the transforms (`is` vs `==`, and `trim()` sometimes returning `None`) that can silently break preprocessing. Finally, I change inference to use a DataLoader (same logic, just batched) so it completes within the time limit and always writes a non-empty `submission.csv` with the correct columns.'
- What this solution (achieved -0.03279) has done: 'Your current 0.0 score is consistent with an “all-same-class” style submission caused by not loading the intended pretrained weights; without weights, this EfficientNet-based model outputs near-constant values after the sigmoid, and fixed thresholds then collapse predictions. To move your score toward the 0.9237 target with minimal core-logic change, I (1) reliably locate the weights file across Kaggle’s possible input paths (instead of a single hardcoded path), and (2) if weights still aren’t present, fall back to timm’s ImageNet pretrained backbone (same architecture) so predictions become non-degenerate. I also fix a small but important inference mismatch: when using `ThreeStage_Model`, you should use `final=True` to get the model’s combined regressor output (otherwise you’re discarding the final stage entirely), which is a minimal semantic correction aligned with how the model was designed. These changes preserve the architecture and inference approach, but should lift you off 0.0 and move substantially closer to the target.'
- What this solution (achieved -0.03279) has done: 'Your current score is far below target, so we should make a small change that improves prediction validity without changing the model/training logic. The biggest likely issue is that the weight file isn’t being found reliably, so you’re often running an ImageNet-only backbone with randomly initialized heads, which produces near-constant/poor predictions. I replace the fragile fixed-path weight lookup with a fast recursive search under `/kaggle/input` (and your provided dataset root) for the exact filename, then load it when present. This keeps the same architecture and inference, but should move QWK substantially toward the target by using the intended trained parameters.'
- What this solution (achieved -0.03279) has done: 'Your current score is far below the target, so the most likely issue is still that you’re not actually loading the intended trained weights (or they load with a key-prefix mismatch), leading to essentially untrained heads and near-random/degenerate predictions. I keep the same model and inference logic, but make weight loading robust by (1) searching for any `.pkl/.pth/.pt` file matching the expected stem, and (2) handling common checkpoint formats (`state_dict`, `model`, `net`) plus `module.` prefixes, while allowing strict loading when possible. If weights still aren’t found, we keep your ImageNet-backbone fallback exactly as before (so it always produces a valid submission). This should move QWK substantially upward toward the target by ensuring the correct trained parameters are actually used when present.'
- What this solution (achieved -0.03279) has done: 'Your score is far below the target, so the smallest meaningful improvement is to ensure inference actually uses trained weights rather than an ImageNet-only backbone with randomly initialized heads (which tends to yield near-random QWK). I keep the same model architecture and inference flow, but make weight discovery/loading more reliable by (1) explicitly scanning `/kaggle/input` for any checkpoint file that contains the expected stem (not only exact filename matches), and (2) if multiple matches exist, prefer the largest file (usually the real trained checkpoint over tiny artifacts). I also make the weight loader handle the common case where the checkpoint is a full dict that contains a nested `state_dict` with prefixes like `model.` in addition to `module.`, still loading strictly when possible. These are minimal changes that should move predictions from “degenerate” to “reasonable”, pushing QWK substantially upward toward your target.'
- What this solution (achieved -0.03279) has done: 'Your score is far below the target, so we should make the smallest change that improves prediction quality without altering the model design or inference semantics. The most likely cause of the negative QWK is that the intended trained checkpoint is still not being loaded (so your heads are effectively untrained), so I make weight discovery/loading more robust by (1) preferring exact filename matches first, then stem matches, and (2) adding one more common prefix-stripping case (`"encoder."`) and a safer strict→non-strict fallback. To further avoid degenerate “all-one-class” outputs when weights aren’t found, I keep your ImageNet-backbone fallback but additionally load backbone weights in the non-weights path exactly as before (no architecture change). Finally, I keep `final=True` inference and the same thresholds, only adding a defensive clamp to ensure predictions stay in [0,4] (this doesn’t change valid outputs, just prevents rare out-of-range artifacts).'
- What this solution (achieved 0.10579) has done: 'I fix the shape-mismatch crash by ensuring the EfficientNet backbone outputs the correct feature dimension (1792) for the B4-based heads, instead of accidentally using the classifier logits (1000) due to `timm`’s default forward behavior. This is done by setting `num_classes=0` (features-only) and keeping the existing GeM pooling so the downstream classifier/regressor/ordinal layers remain unchanged. I apply the same fix for the optional ImageNet-pretrained fallback backbone so both paths behave identically. Finally, I make submission writing robust so `submission.csv` is always produced once inference completes.'
- What this solution (achieved 0.00254) has done: 'Your current score is far below the target, so we should make the smallest change that can materially increase QWK without altering the model design: calibrate the fixed regression→class thresholds on the training labels distribution. Since QWK is sensitive to class imbalance, a simple distribution-matching (quantile) threshold calibration using `train.csv` often improves over hardcoded thresholds while keeping the same inference semantics (regression output binned into 5 classes). I keep the exact same model, weights-loading logic, transforms, and inference path (`final=True`), only replacing the static thresholds with data-driven ones computed once from `train.csv`. I also keep a safe fallback to the original thresholds if reading `train.csv` fails for any reason, and still write a valid `submission.csv`.'
- What this solution (achieved 0.31719) has done: 'Your current score is extremely far below the target, so the most likely remaining issue is not “model strength” but a prediction-to-class mapping that’s badly mismatched to the model’s output distribution. I keep your exact model/inference core (same network, same `final=True`, same transforms) and only change the threshold calibration to be *data-driven from the model’s own train-set predictions* (optimize thresholds for QWK on train), instead of using label-quantiles that ignore the model output scale. This is a minimal semantic adjustment (still regression binned into 5 classes) but usually produces a large QWK jump when thresholds were the bottleneck. If train-image inference is too slow, it safely fall back to your current default thresholds and still write a valid `submission.csv`.'
- What this solution (achieved 0.28762) has done: 'Your current score (0.31719) is far below the target (0.92378), so we should improve prediction correctness rather than tweak architecture. The biggest likely issue is that threshold calibration is capped to the *first* 1200 training images (not representative and not shuffled), which can produce badly biased thresholds and low QWK on the test set. I keep the exact same model/inference logic, but make the calibration subset deterministic-and-representative by using a seeded stratified sample across labels (still capped at 1200 for time). I also ensure calibration uses the same ordering/IDs without any leakage and keep the same submission writing.'

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
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(42)



## === cell 1
DEFAULT_THRESHOLD = [0.75, 1.5, 2.5, 3.5]
threshold = DEFAULT_THRESHOLD.copy()


def regress2class(out):
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5).to(out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5)).to(out.device)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(float(out[i])))
            l2 = int(math.ceil(float(out[i])))
            pred_prob[i][l1] = 1 - (out[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob


def _apply_thresholds_np(preds_reg, thr_list):
    thr = np.array(thr_list, dtype=np.float64).reshape(1, -1)
    preds_reg = np.asarray(preds_reg, dtype=np.float64).reshape(-1, 1)
    cls = (preds_reg >= thr).sum(axis=1).astype(np.int64)
    return np.clip(cls, 0, 4)


def _qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def calibrate_thresholds_by_qwk(y_true, preds_reg, default_thr):
    try:
        y_true = np.asarray(y_true, dtype=np.int64)
        preds_reg = np.asarray(preds_reg, dtype=np.float64)
        if len(y_true) == 0 or len(preds_reg) != len(y_true):
            return default_thr

        qs = np.quantile(preds_reg, [0.2, 0.4, 0.6, 0.8]).astype(np.float64)
        thr = np.clip(qs, 0.0, 4.5)

        best_thr = thr.copy()
        best_score = _qwk(y_true, _apply_thresholds_np(preds_reg, best_thr.tolist()))

        step = 0.05
        radius = 0.60  # +/- range around each threshold
        grid_offsets = np.arange(-radius, radius + 1e-12, step, dtype=np.float64)

        for _ in range(3):  # a few passes is enough; keep runtime bounded
            improved = False
            for i in range(4):
                local_best = best_thr[i]
                local_best_score = best_score

                for off in grid_offsets:
                    cand = best_thr.copy()
                    cand[i] = np.clip(best_thr[i] + off, 0.0, 4.5)
                    eps = 1e-3
                    if i > 0:
                        cand[i] = max(cand[i], cand[i - 1] + eps)
                    if i < 3:
                        cand[i] = min(cand[i], cand[i + 1] - eps)
                    if not (cand[0] < cand[1] < cand[2] < cand[3]):
                        continue

                    score = _qwk(y_true, _apply_thresholds_np(preds_reg, cand.tolist()))
                    if score > local_best_score + 1e-12:
                        local_best_score = score
                        local_best = cand[i]

                if local_best != best_thr[i]:
                    best_thr[i] = local_best
                    best_score = local_best_score
                    improved = True

            if not improved:
                break

        out = best_thr.tolist()
        if not (out[0] < out[1] < out[2] < out[3]):
            return default_thr
        return [float(x) for x in out]
    except Exception:
        return default_thr




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
        self.backbone = timm.models.tf_efficientnet_b5_ns(
            pretrained=False, num_classes=0
        )
        self.backbone.global_pool = GeM(flatten=True)

        in_features = getattr(self.backbone, "num_features", 1000)
        self.regressor = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None):
        super(ThreeStage_Model, self).__init__()

        if backbone is None:
            self.backbone = timm.models.tf_efficientnet_b4_ns(
                pretrained=False, num_classes=0
            )
        else:
            self.backbone = backbone

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
            nn.Linear(5 + 1 + 4, 1),
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
DATA_ROOT = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

input_size = 384

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)


def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def find_weights_exact_or_stem(
    filename, stem, roots, exts=(".pkl", ".pth", ".pt"), max_hits=200
):
    exact_hits = []
    stem_hits = []
    filename_low = filename.lower()
    stem_low = stem.lower()

    for root in roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                base, ext = os.path.splitext(fn)
                if ext.lower() not in exts:
                    continue
                full = os.path.join(dirpath, fn)
                fn_low = fn.lower()
                base_low = base.lower()

                if fn_low == filename_low:
                    exact_hits.append(full)
                elif stem_low in fn_low or stem_low in base_low:
                    stem_hits.append(full)

                if len(exact_hits) + len(stem_hits) >= max_hits:
                    break
            if len(exact_hits) + len(stem_hits) >= max_hits:
                break

    def _sort_key(p):
        return os.path.getsize(p) if os.path.exists(p) else -1

    if len(exact_hits) > 0:
        exact_hits = sorted(exact_hits, key=_sort_key, reverse=True)
        return exact_hits
    stem_hits = sorted(stem_hits, key=_sort_key, reverse=True)
    return stem_hits


def _strip_known_prefixes(state_dict):
    if not isinstance(state_dict, dict) or len(state_dict) == 0:
        return state_dict
    prefixes = ["module.", "model.", "net.", "backbone.", "encoder."]
    out = state_dict
    for pref in prefixes:
        if any(k.startswith(pref) for k in out.keys()):
            out = {
                k[len(pref) :] if k.startswith(pref) else k: v for k, v in out.items()
            }
    return out


def load_checkpoint_flex(path, model):
    ckpt = torch.load(path, map_location="cpu")

    candidate = None
    if isinstance(ckpt, dict):
        for key in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                candidate = ckpt[key]
                break
        if candidate is None and all(isinstance(k, str) for k in ckpt.keys()):
            candidate = ckpt
    else:
        candidate = ckpt

    candidate = _strip_known_prefixes(candidate)

    try:
        model.load_state_dict(candidate, strict=True)
        return True, "strict"
    except Exception:
        missing, unexpected = model.load_state_dict(candidate, strict=False)
        return (
            True,
            f"non-strict (missing={len(missing)}, unexpected={len(unexpected)})",
        )


WEIGHTS_STEM = "B4_3stage_18epoch_320finetune"
WEIGHTS_FILENAME = WEIGHTS_STEM + ".pkl"

candidate_weight_paths = [
    "../input/weights/" + WEIGHTS_FILENAME,
    "../input/aptos2019-blindness-detection/weights/" + WEIGHTS_FILENAME,
    "/kaggle/input/weights/" + WEIGHTS_FILENAME,
    "/kaggle/input/aptos2019-blindness-detection/weights/" + WEIGHTS_FILENAME,
]

weights_path = find_first_existing(candidate_weight_paths)
if weights_path is None:
    search_roots = ["/kaggle/input", "../input", DATA_ROOT]
    hits = find_weights_exact_or_stem(
        WEIGHTS_FILENAME, WEIGHTS_STEM, search_roots, max_hits=200
    )
    if len(hits) > 0:
        weights_path = hits[0]

if weights_path is None:
    backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True, num_classes=0)
    net = ThreeStage_Model(backbone=backbone)
    load_msg = "None (using ImageNet pretrained backbone)"
else:
    net = ThreeStage_Model()
    ok, mode = load_checkpoint_flex(weights_path, net)
    load_msg = f"{weights_path} [{mode}]" if ok else f"{weights_path} [failed]"

net = net.to(device)
net.eval()

print("Device:", device)
print("Weights loaded:", load_msg)

threshold = DEFAULT_THRESHOLD.copy()
try:
    tr_df = pd.read_csv(TRAIN_CSV)
    if (
        "id_code" in tr_df.columns
        and "diagnosis" in tr_df.columns
        and os.path.isdir(TRAIN_IMG_DIR)
    ):
        tr_ids = tr_df["id_code"].astype(str).values
        tr_y = tr_df["diagnosis"].astype(int).values

        class TrainDataset(Dataset):
            def __init__(self, ids, y, img_dir, transform=None):
                self.ids = list(ids)
                self.y = np.asarray(y, dtype=np.int64)
                self.img_dir = img_dir
                self.transform = transform

            def __len__(self):
                return len(self.ids)

            def __getitem__(self, idx):
                id_code = self.ids[idx]
                image_name = os.path.join(self.img_dir, f"{id_code}.png")
                img = Image.open(image_name).convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
                return self.y[idx], img

        max_train_calib = 1200
        rng = np.random.RandomState(42)
        if len(tr_ids) > max_train_calib:
            calib_indices = []
            per_class = max_train_calib // 5
            for cls in range(5):
                idx = np.where(tr_y == cls)[0]
                if len(idx) == 0:
                    continue
                take = min(per_class, len(idx))
                calib_indices.append(rng.choice(idx, size=take, replace=False))
            calib_indices = (
                np.concatenate(calib_indices)
                if len(calib_indices)
                else np.arange(len(tr_ids))
            )

            if len(calib_indices) < max_train_calib:
                remaining = np.setdiff1d(
                    np.arange(len(tr_ids)), calib_indices, assume_unique=False
                )
                need = max_train_calib - len(calib_indices)
                if len(remaining) > 0:
                    extra = rng.choice(
                        remaining, size=min(need, len(remaining)), replace=False
                    )
                    calib_indices = np.concatenate([calib_indices, extra])

            calib_indices = rng.permutation(calib_indices)[:max_train_calib]
            tr_ids_cal = tr_ids[calib_indices]
            tr_y_cal = tr_y[calib_indices]
        else:
            tr_ids_cal = tr_ids
            tr_y_cal = tr_y

        calib_bs = 12 if device.type == "cuda" else 6
        calib_loader = DataLoader(
            TrainDataset(tr_ids_cal, tr_y_cal, TRAIN_IMG_DIR, transform=transform),
            batch_size=calib_bs,
            shuffle=False,
            num_workers=2,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        preds_reg = []
        y_true = []
        t0 = time.time()
        with torch.no_grad():
            for yb, xb in calib_loader:
                xb = xb.to(device, non_blocking=True)
                out = (
                    net(xb, final=True)
                    .clamp(0.0, 4.5)
                    .squeeze(1)
                    .detach()
                    .cpu()
                    .numpy()
                )
                preds_reg.append(out)
                y_true.append(yb.numpy())
                if time.time() - t0 > 220:
                    raise TimeoutError("Calibration exceeded time budget")
        preds_reg = np.concatenate(preds_reg, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        threshold = calibrate_thresholds_by_qwk(y_true, preds_reg, DEFAULT_THRESHOLD)
        calib_qwk = _qwk(y_true, _apply_thresholds_np(preds_reg, threshold))
        print(
            f"Calibrated thresholds (QWK-optimized on {len(y_true)} train imgs): {threshold} | train-QWK={calib_qwk:.5f}"
        )
    else:
        print(
            "Threshold calibration skipped (missing train columns/dir); using default thresholds."
        )
except Exception as e:
    print("Threshold calibration failed; using default thresholds. Reason:", repr(e))
    threshold = DEFAULT_THRESHOLD.copy()

print("Using thresholds:", threshold)




## === cell 5
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
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return id_code, img


batch_size = 8 if device.type == "cuda" else 4
num_workers = 2

test_ds = TestDataset(test_ids, TEST_IMG_DIR, transform=transform)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    drop_last=False,
)

submission_rows = []
with torch.no_grad():
    for batch_ids, batch_imgs in test_loader:
        batch_imgs = batch_imgs.to(device, non_blocking=True)

        out = net(batch_imgs, final=True)  # shape [B,1], scaled to [0,4.5]
        out = out.clamp_(0.0, 4.5)

        preds = regress2class(out.data.squeeze(1)).numpy().astype(int)
        preds = np.clip(preds, 0, 4)

        for i in range(len(batch_ids)):
            submission_rows.append([str(batch_ids[i]), int(preds[i])])

submission = np.array(submission_rows, dtype=object)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

out_path = "submission.csv"
df.to_csv(out_path, index=False)

print(df.head())
print("Pred distribution:\n", df["diagnosis"].value_counts(dropna=False).sort_index())
print(f"Wrote {len(df)} rows to {out_path}")
