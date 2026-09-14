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

0.9127199468129416

# 6. Current score

-0.00091

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing weights path issue by making the script robust to the absence of `../input/weights/*` and fall back to using the model with random initialization (still producing a valid submission CSV end-to-end). I also fix the hard-coded CUDA device selection so it runs on Kaggle CPU-only environments by automatically choosing `cuda` only if available, and ensure all tensors are created on the correct device (removing `.cuda()` calls inside helper functions). Finally, I make the dataset loop resilient to transform returning `None` (from `trim`) and to missing/corrupt images, so the submission is never empty and always matches `test.csv` ordering/length.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with running the model with random initialization (because the weights path doesn’t exist), so the minimal way to move toward the target is to ensure the script actually loads a compatible pretrained checkpoint if one is available in the provided dataset folders. I add a robust checkpoint search over `/kaggle/input/**` for your expected filename (and a few common variants), load it with `strict=False` to tolerate minor key mismatches, and keep everything else (model, transforms, inference, thresholds) unchanged. If no checkpoint is found, the script still fall back to random weights and produce a valid submission CSV (same as now). This should increase the score substantially toward your target when the checkpoint is present, without changing core logic.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is consistent with producing near-constant/garbage predictions due to random weights or using the wrong output head for inference. To move the score toward the target with minimal logic change, I (1) make checkpoint discovery actually find any plausible ThreeStage/B4 weights under `/kaggle/input` by matching common substrings (not just one exact filename), and (2) keep your same model but switch inference to the model’s `final=True` path, which is explicitly the intended single-regression output when a three-stage checkpoint is used. I also keep your original regression-to-class conversion and thresholds unchanged to preserve evaluation semantics. The script still run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with either (a) running without the intended checkpoint (random weights) or (b) failing to find the right weights file under `/kaggle/input`, so predictions are effectively garbage/constant. To move toward your target with minimal change and without touching model/metric logic, I (1) expand and harden checkpoint discovery to also catch common Kaggle dataset layouts/filenames (including `.ckpt`) and (2) make the state-dict key normalization robust to common wrappers (`model.`, `net.`, `module.`), while keeping `strict=False` and the same inference path `final=True`. If no checkpoint exists, the script still run end-to-end and produce a valid `submission.csv` exactly as before.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 strongly suggests the model is still effectively untrained at inference time (either no checkpoint is found/loaded, or the loaded file isn’t actually a matching state_dict), so the smallest score-moving change is to make checkpoint discovery and loading more reliable without touching the model or prediction logic. I expand the checkpoint search to also consider the common Kaggle “dataset contains weights under its own folder” layout and relax the filename heuristics so a real ThreeStage/B4 checkpoint is actually found. I also improve checkpoint parsing by handling cases where the loaded object is a full training checkpoint (nested dict) and by selecting the tensor dict that best matches your model’s keys (still using `strict=False`). Everything else (architecture, transforms, `final=True` inference path, and thresholds) stays the same to preserve core evaluation semantics while moving the score upward toward your target.'
- What this solution (achieved -0.03935) has done: 'Your score is 0.0 because the model is almost certainly still running with random weights (no real checkpoint is being found/loaded under your available `/kaggle/input` tree), so the minimal score-moving change is to ensure we load a meaningful pretrained backbone that is guaranteed to exist (timm pretrained weights) without changing your model structure. I keep your ThreeStage model, transforms, and `final=True` inference path unchanged, but switch `pretrained=True` for the EfficientNet backbones only when no external checkpoint is successfully loaded. I also tighten the checkpoint search to prefer files that look like actual model weights (and de-prioritize unrelated `.pth/.ckpt`), which reduces the chance of “loading something wrong” and silently getting garbage predictions. This should move QWK up substantially toward your target while keeping changes minimal and preserving evaluation semantics.'
- What this solution (achieved -0.02112) has done: 'Your current score is far below the target, so we should improve predictive signal with minimal, metric-relevant changes while preserving your model and inference semantics. The biggest issue is that when no compatible checkpoint is found, only the backbone is pretrained but all heads (including `final_regressor`) are random, which typically yields near-random QWK; we instead fall back to a backbone whose classifier is pretrained and use its class logits directly only in the “no checkpoint” case. When a real ThreeStage checkpoint is found, we keep your exact current path (`final=True` + `regress2class` + thresholds) unchanged. This keeps core logic intact, still writes a valid `submission.csv`, and should move the score substantially upward toward the target. We also avoid any calibration/threshold changes to preserve evaluation semantics.'
- What this solution (achieved 0.0) has done: 'Your current negative score is driven by the “no checkpoint” fallback producing essentially random/garbage labels (ImageNet class % 5). To move the score upward toward your target with minimal semantic change, I keep your ThreeStage model and the same `final=True -> regress2class(thresholds)` path when a checkpoint is found, but replace the fallback with a lightweight training of your existing `Regressor` head on the provided `train.csv` images (using the same normalization/resize pipeline) and then use that regressor at inference. This preserves your overall approach (regression + fixed thresholds) and avoids changing thresholds or the ThreeStage architecture, while providing real signal when no external weights exist. I also keep runtime bounded (small input size and capped epochs) and ensure a valid `submission.csv` is always written in `test.csv` order.'
- What this solution (achieved 0.09837) has done: 'The timeout is dominated by repeated per-image PIL decode + transform + single-image GPU calls, plus an expensive checkpoint search (`os.walk` over all of `/kaggle/input`). I keep the exact same model and transforms, but make inference batched via a `Dataset`/`DataLoader` with multiple workers, pinned memory, and non-blocking H2D transfers to cut overhead drastically. I also make checkpoint discovery constant-time by checking the known preferred paths first and only doing a shallow, bounded search if needed (same semantics: load the best matching checkpoint if present). Finally, the fallback training path is preserved but sped up by using a DataLoader and caching feature extraction per batch (same computation, fewer Python loops).'
- What this solution (achieved 0.0) has done: 'Your score is far below the target, so we should add predictive signal with the smallest change that preserves your current architecture and regression→thresholded-class semantics. The main issue is that when no external checkpoint is found, you only train `final_regressor` while `classifier/regressor/ordinal` remain random, so the `final_regressor` is learning on essentially noise features and won’t generalize well. I keep your exact ThreeStage model, transforms, loss, and inference path (`final=True` then `regress2class` with the same thresholds), but in the fallback path I additionally train `classifier`, `regressor`, and `ordinal` (keeping the EfficientNet backbone frozen) so the features used by `final_regressor` become meaningful. This should move QWK substantially upward toward the target without changing evaluation semantics or relying on external weights, and it still writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current score (0.0) is far below the target, so the smallest reliable way to move it upward is to make the fallback training (used when no external checkpoint is found) learn a signal that matches the quadratic-weighted-kappa objective better, without changing your model or regression→thresholded-class semantics. I keep your exact ThreeStage architecture and inference path (`final=True` then `regress2class` with the same thresholds), but change the fallback training loss to an ordinal-friendly classification loss computed from the same `final_regressor` output by converting it to a 5-class probability with your existing `regress2class_prob`. I also remove the hard “ok==False -> label 0” override at test-time (still safe for missing/corrupt images) because it can force many false zeros and depress kappa. These are minimal, metric-aligned changes that preserve the core logic and should push the score substantially toward your target.'
- What this solution (achieved -0.00896) has done: 'Your current 0.0 QWK is most consistent with a submission that’s effectively untrained/poorly trained at inference time, because your fallback only trains the heads while the backbone features are fixed ImageNet features and the training setup is noisy. To move the score upward toward the 0.9127 target with minimal semantic change, I keep your exact ThreeStage architecture and the same inference path (`final=True` then `regress2class` with the same thresholds), but I make the fallback training actually learn the *final* head in a more stable way: train only `final_regressor` on *cached* 10-dim features computed from the current model’s three heads (no gradients through backbone/heads), so optimization is well-conditioned and much faster. I also fix a small but important bug in `regress2class_prob` where probabilities can fail to sum to 1 when `out` has fractional parts (this makes your NLL loss inconsistent), while keeping its intent identical. These changes are limited to the “no checkpoint” fallback path and should substantially increase the score versus 0.0 without changing your evaluation semantics when a real checkpoint is present.'
- What this solution (achieved -0.00091) has done: 'I fix the fallback training crash by ensuring the cached-feature training step keeps gradients enabled for the `final_regressor` (the current `regress2class_prob` path builds a non-differentiable CPU tensor, so the loss has no grad_fn). To preserve your core model/inference semantics, I keep the same `final=True` regression output and the same fixed thresholds for converting regression to classes at test time, and only change the fallback loss computation to a differentiable ordinal-style loss directly from the regression output. I also keep the checkpoint-loading logic intact and make no changes when a real checkpoint is found (score-neutral in that case). The script then run end-to-end and write a valid `submission.csv`.'

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
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = (
    True  # fixed input size -> faster convolutions, negligible FP diffs
)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor):
    out = out.view(-1)
    prediction = torch.zeros(out.size(0), device=out.device, dtype=torch.float32)
    for i in range(4):
        prediction += (out >= threshold[i]).to(torch.float32)
    return prediction


def ordinal2class_prob(out: torch.Tensor):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out: torch.Tensor):
    out = out.view(-1)
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)
    for i in range(out.size(0)):
        oi = float(out[i].item())
        if oi <= 0.0:
            pred_prob[i, 0] = 1.0
        elif oi >= 4.0:
            pred_prob[i, 4] = 1.0
        else:
            l1 = int(math.floor(oi))
            l2 = l1 + 1
            frac = oi - l1
            pred_prob[i, l1] = 1.0 - frac
            pred_prob[i, l2] = frac
    return pred_prob


def regress2class_prob_soft(out: torch.Tensor, eps: float = 1e-6):
    """
    out: [B] or [B,1] regression outputs (roughly 0..4.5)
    returns: [B,5] probabilities that sum to 1 (differentiable w.r.t out)
    """
    x = out.view(-1).clamp(0.0, 4.0 - eps)  # ensure we have l2 within 0..4
    l1 = torch.floor(x).long().clamp(0, 3)  # 0..3
    frac = (x - l1.float()).clamp(0.0, 1.0)

    prob = torch.zeros(x.size(0), 5, device=out.device, dtype=out.dtype)
    prob.scatter_(1, l1.unsqueeze(1), (1.0 - frac).unsqueeze(1))
    prob.scatter_(1, (l1 + 1).unsqueeze(1), frac.unsqueeze(1))

    ge4 = out.view(-1) >= 4.0
    if ge4.any():
        prob[ge4] = 0.0
        prob[ge4, 4] = 1.0

    le0 = out.view(-1) <= 0.0
    if le0.any():
        prob[le0] = 0.0
        prob[le0, 0] = 1.0

    return prob




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
    def __init__(self, pretrained_backbone: bool = False):
        super(Regressor, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b5_ns(
            pretrained=pretrained_backbone
        )
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None, pretrained_backbone: bool = False):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.models.tf_efficientnet_b4_ns(
            pretrained=pretrained_backbone
        )
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
DATA_ROOT = "../input/aptos2019-blindness-detection"
test_csv = os.path.join(DATA_ROOT, "test.csv")
test_img_dir = os.path.join(DATA_ROOT, "test_images")
train_csv = os.path.join(DATA_ROOT, "train.csv")
train_img_dir = os.path.join(DATA_ROOT, "train_images")

if not os.path.exists(test_csv):
    alt_root = "../input"
    if os.path.exists(os.path.join(alt_root, "test.csv")):
        DATA_ROOT = alt_root
        test_csv = os.path.join(DATA_ROOT, "test.csv")
        test_img_dir = os.path.join(DATA_ROOT, "test_images")
        train_csv = os.path.join(DATA_ROOT, "train.csv")
        train_img_dir = os.path.join(DATA_ROOT, "train_images")

test_ids_df = pd.read_csv(test_csv)
test_ids = np.squeeze(test_ids_df["id_code"].values)

train_df = pd.read_csv(train_csv)

input_size = 256

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model(pretrained_backbone=True)
fallback_regressor = None


def _find_checkpoint():
    preferred = [
        "../input/weights/B4_3stage_25epoch_320.pkl",
        "../input/weights/B4_3stage_25epoch_320.pth",
        "../input/weights/B4_3stage_25epoch_320.pt",
        "../input/weights/B4_3stage_25epoch_320.bin",
        "../input/weights/B4_3stage_25epoch_320.ckpt",
    ]
    for p in preferred:
        if os.path.exists(p):
            return p

    search_roots = ["../input", "/kaggle/input"]
    exts = (".pth", ".pt", ".pkl", ".bin", ".ckpt")

    good_substrings = [
        "3stage",
        "three",
        "three_stage",
        "threestage",
        "final_regressor",
        "efficientnet",
        "tf_efficientnet",
        "b4",
        "aptos",
        "blindness",
        "retina",
        "retinopathy",
        "dr",
        "kappa",
        "checkpoint",
        "weights",
        "model",
        "state_dict",
    ]

    best = None
    best_score = -1

    def score_path(path: str) -> int:
        fn_l = os.path.basename(path).lower()
        sc = sum(1 for s in good_substrings if s in fn_l)
        if "b4_3stage_25epoch_320" in fn_l:
            sc += 200
        if any(
            bad in fn_l
            for bad in ["optimizer", "sched", "scheduler", "adam", "ema", "fold"]
        ):
            sc -= 3
        dir_l = os.path.dirname(path).lower()
        if "weights" in dir_l or "checkpoint" in dir_l:
            sc += 2
        if "aptos" in dir_l or "blind" in dir_l or "retina" in dir_l:
            sc += 2
        return sc

    for root in search_roots:
        if not os.path.exists(root):
            continue

        try:
            lvl1 = [os.path.join(root, d) for d in os.listdir(root)]
        except Exception:
            lvl1 = []
        candidates_dirs = [root] + [d for d in lvl1 if os.path.isdir(d)]

        lvl2 = []
        for d1 in candidates_dirs:
            try:
                for d in os.listdir(d1):
                    p = os.path.join(d1, d)
                    if os.path.isdir(p):
                        lvl2.append(p)
            except Exception:
                continue
        candidates_dirs.extend(lvl2)

        for dirpath in candidates_dirs:
            try:
                for fn in os.listdir(dirpath):
                    fn_l = fn.lower()
                    if not fn_l.endswith(exts):
                        continue
                    full = os.path.join(dirpath, fn)
                    sc = score_path(full)
                    if sc > best_score:
                        best_score = sc
                        best = full
            except Exception:
                continue

    if best_score < 3:
        return None
    return best


def _unwrap_state_dict(state):
    if isinstance(state, dict):
        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "weights",
        ]:
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def _normalize_state_keys(sd):
    if not isinstance(sd, dict):
        return sd
    new_sd = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("module.", "model.", "net.", "network."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        new_sd[nk] = v
    return new_sd


def _select_best_tensor_dict(obj, model_state_keys):
    candidates = []

    def add_candidate(d, name):
        if isinstance(d, dict) and any(torch.is_tensor(v) for v in d.values()):
            sd = _normalize_state_keys(d)
            overlap = len(set(sd.keys()) & model_state_keys)
            candidates.append((overlap, name, sd))

    if isinstance(obj, dict):
        add_candidate(obj, "root")
        for k, v in obj.items():
            if isinstance(v, dict):
                add_candidate(v, f"root[{k}]")
                for k2, v2 in v.items():
                    if isinstance(v2, dict):
                        add_candidate(v2, f"root[{k}][{k2}]")

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][2], candidates[0][1]


weights_path = _find_checkpoint()
loaded_weights = False
if weights_path is not None and os.path.exists(weights_path):
    try:
        raw = torch.load(weights_path, map_location="cpu")
        model_keys = set(net.state_dict().keys())

        state = _unwrap_state_dict(raw)
        state = _normalize_state_keys(state)

        if not (
            isinstance(state, dict) and any(torch.is_tensor(v) for v in state.values())
        ):
            state, picked_from = _select_best_tensor_dict(raw, model_keys)
        else:
            picked_from = "unwrap_state_dict"

        if state is None:
            raise ValueError(
                "No compatible tensor state_dict found inside checkpoint object"
            )

        missing, unexpected = net.load_state_dict(state, strict=False)
        loaded_weights = True
        print(f"Loaded checkpoint from: {weights_path} (picked: {picked_from})")
        if len(missing) or len(unexpected):
            print(
                f"Note: load_state_dict strict=False, missing={len(missing)}, unexpected={len(unexpected)}"
            )
    except Exception as e:
        print(f"Failed to load checkpoint at {weights_path}: {repr(e)}")
        loaded_weights = False
        weights_path = None
else:
    print("No checkpoint found for ThreeStage model.")

net = net.to(device)
net.eval()

from torch.utils.data import Dataset, DataLoader


class PNGDataset(Dataset):
    def __init__(self, ids, img_dir, transform, labels=None):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = None if labels is None else np.asarray(labels, dtype=np.float32)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        path = os.path.join(self.img_dir, f"{idx}.png")
        try:
            with Image.open(path) as im:
                img = im.convert("RGB")
            x = self.transform(img)
            ok = True
        except Exception:
            x = torch.zeros(3, input_size * 3 // 4, input_size, dtype=torch.float32)
            ok = False
        if self.labels is None:
            return idx, x, ok
        y = self.labels[i]
        return idx, x, y, ok


def _quick_train_fallback_final_regressor(
    train_df: pd.DataFrame, model: ThreeStage_Model
):
    start_t = time.time()

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.final_regressor.parameters():
        p.requires_grad = True
    model.final_regressor.train()

    df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    max_train = min(len(df), 2800)
    df = df.iloc[:max_train].copy()

    bs = 32 if torch.cuda.is_available() else 16
    num_workers = min(4, (os.cpu_count() or 2))

    dataset = PNGDataset(
        df["id_code"].values, train_img_dir, transform, labels=df["diagnosis"].values
    )
    loader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,  # deterministic cache order
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )

    feat_chunks = []
    y_chunks = []

    with torch.no_grad():
        for _, xb, yb, ok in loader:
            ok_mask = (
                ok.bool()
                if isinstance(ok, torch.Tensor)
                else torch.tensor(ok, dtype=torch.bool)
            )
            if ok_mask.sum().item() == 0:
                continue
            xb = xb[ok_mask].to(device, non_blocking=True)
            yb = yb[ok_mask].to(device, non_blocking=True).long().view(-1)

            feat1000 = model.backbone(xb)
            c_out = model.classifier(feat1000)  # [B,5] logits
            r_raw = model.regressor(feat1000)  # [B,1] raw
            o_raw = model.ordinal(feat1000)  # [B,4] raw
            r_out = torch.sigmoid(r_raw) * 4.5
            o_out = torch.sigmoid(o_raw)

            feat10 = torch.cat((c_out, r_out, o_out), dim=1)  # [B,10]
            feat_chunks.append(feat10.detach().cpu())
            y_chunks.append(yb.detach().cpu())

            if time.time() - start_t > 240:
                break

    if not feat_chunks:
        model.eval()
        return model

    X = torch.cat(feat_chunks, dim=0)  # CPU
    Y = torch.cat(y_chunks, dim=0)  # CPU

    class FeatDataset(Dataset):
        def __init__(self, X, Y):
            self.X = X
            self.Y = Y

        def __len__(self):
            return self.X.size(0)

        def __getitem__(self, i):
            return self.X[i], self.Y[i]

    train_loader = DataLoader(
        FeatDataset(X, Y),
        batch_size=256 if torch.cuda.is_available() else 128,
        shuffle=True,
        num_workers=0,
    )

    opt = torch.optim.Adam(model.final_regressor.parameters(), lr=2e-3)
    ce_loss = nn.NLLLoss()

    epochs = 12
    for ep in range(epochs):
        running = 0.0
        n = 0
        for xb10, yb in train_loader:
            xb10 = xb10.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            out = model.final_regressor(xb10)
            out = torch.sigmoid(out) * 4.5  # [B,1]

            prob5 = regress2class_prob_soft(out.view(-1))
            logp5 = torch.log(prob5.clamp_min(1e-8))
            loss = ce_loss(logp5, yb)

            loss.backward()
            opt.step()

            running += float(loss.item()) * xb10.size(0)
            n += xb10.size(0)

        print(
            f"[fallback_final_regressor_cached] epoch {ep+1}/{epochs} loss={running/max(n,1):.4f}"
        )

        if time.time() - start_t > 540:
            break

    model.eval()
    return model


if not loaded_weights:
    net = _quick_train_fallback_final_regressor(train_df, net)

print(f"Device: {device}")
print(f"Test images dir: {test_img_dir}")
print(f"Loaded weights: {loaded_weights} ({weights_path})")
print(f"Fallback trained heads (no checkpoint): {not loaded_weights}")



## === cell 5
submission_rows = []

test_dataset = PNGDataset(test_ids, test_img_dir, transform, labels=None)
num_workers = min(4, (os.cpu_count() or 2))
infer_bs = 32 if torch.cuda.is_available() else 8

test_loader = DataLoader(
    test_dataset,
    batch_size=infer_bs,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
)

net.eval()
with torch.no_grad():
    seen = 0
    for batch in test_loader:
        ids_b, xb, ok = batch
        seen += len(ids_b)
        if (seen - len(ids_b)) % 50 == 0:
            print(f"Inferencing {seen - len(ids_b)}/{len(test_ids)}")

        xb = xb.to(device, non_blocking=True)
        r_out = net(xb, final=True)  # [B,1] in 0..4.5

        pred = regress2class(r_out.squeeze(1)).to(torch.int64).cpu().numpy()
        ok_np = ok.numpy().astype(bool)

        for idx, pi, ok_i in zip(ids_b, pred, ok_np):
            submission_rows.append([idx, int(pi)])

submission = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])
submission["diagnosis"] = submission["diagnosis"].astype(int)
assert len(submission) == len(test_ids_df), "Submission row count must match test.csv"
assert list(submission.columns) == ["id_code", "diagnosis"]



## === cell 6
out_path = "submission.csv"
submission.to_csv(out_path, index=False)
print(
    f"Wrote {out_path} with shape {submission.shape} and columns {list(submission.columns)}"
)
print(submission.head())
