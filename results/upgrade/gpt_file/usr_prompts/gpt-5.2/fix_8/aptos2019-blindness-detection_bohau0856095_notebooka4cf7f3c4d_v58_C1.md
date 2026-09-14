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

0.908743227595196

# 6. Current score

0.02111

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.03395) has done: 'I (1) remove the failing custom `pip install` and instead rely on the already-installed `timm==1.0.19`, (2) make the script run on both GPU and CPU by selecting `cuda` only if available and removing hard `.cuda()` calls, and (3) fix the missing weights crash by falling back to a reasonable `pretrained=True` backbone when the external weight file is not present. I also fix a couple of transform bugs that can silently break preprocessing (`is` vs `==` for hue, and ensuring `trim()` always returns an image). Finally, I ensure the submission is always written with the required columns and correct row count so Kaggle accepts it.'
- What this solution (achieved 0.01557) has done: 'Your current score is far below the target (higher-is-better), and the biggest likely cause is that inference is using a randomly-initialized “three-stage” head because the external weights file is missing; “pretrained backbone” alone won’t produce meaningful ordinal predictions. To move the score strongly toward the target without changing core architecture or training logic, I (1) load the official timm pretrained weights correctly for this specific EfficientNet variant (so the backbone is actually pretrained, not random), and (2) fall back to the simpler `Regressor` model (same general approach: EfficientNet+GeM+regression+thresholding) when the 3-stage weights are unavailable, because it’s the minimal way to get non-random predictions. I also ensure inference runs under `torch.inference_mode()` for correctness/perf and keep the submission format identical. No changes are made to loss/training/feature extraction; this is purely a robust weight initialization/inference fallback to improve the Kaggle metric.'
- What this solution (achieved 0.02065) has done: 'Your score is far below the target, and the smallest high-impact fix is to use the model output that matches the competition metric better: instead of hard, fixed thresholds on a regression output, we tune those 4 thresholds on a held-out split of the training set to directly maximize Quadratic Weighted Kappa. This keeps your core model and inference loop intact (same EfficientNet+GeM+regression head, same transforms for test), but replaces the static `threshold` list with data-driven thresholds learned from your own training labels. To keep changes minimal and stable, we only sample a small subset of training images for threshold fitting (fast) and we do not add any new training or change the network weights. Finally, we apply the learned thresholds to test predictions and still write a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.04661) has done: 'Your current score is far below the target, so we should make a small change that improves ordinal calibration without changing the model or training. The biggest issue here is that your threshold tuning is done on a random tiny subset (512) and can be unstable/misaligned; we instead tune thresholds on out-of-fold (OOF) predictions from the full training set using a simple 5-fold split, which directly optimizes QWK and typically moves the score substantially upward. This keeps the exact same model, transforms, inference code, and thresholding semantics—only how thresholds are estimated changes. We also slightly tighten correctness by ensuring the tuned thresholds are strictly increasing and by using deterministic folds.'
- What this solution (achieved 0.01295) has done: 'Your current score is far below the target, so we need a small but high-impact change that preserves your model/inference logic while improving the threshold calibration that QWK is very sensitive to. Right now thresholds are tuned using OOF predictions from a model that was trained on all data (not fold-specific), so the “OOF” predictions are effectively in-sample and can overfit the thresholds, hurting generalization to test. I keep the exact same network, transforms, and thresholding semantics, but change the threshold fitting to use a single stratified holdout split (true out-of-sample) and a slightly denser, safe coordinate search around each threshold. This typically improves calibration stability and should move the score upward toward your target without changing the model architecture or training.'
- What this solution (achieved 0.02111) has done: 'Your score is far below the target, so the smallest high-impact fix (without changing the model/training core) is to improve the threshold calibration procedure, because QWK is extremely sensitive to the four cutpoints. I keep your exact model(s), transforms, and inference logic, but replace the current coarse coordinate search with a deterministic, stronger 1D threshold optimization (per-threshold golden-section search) on a true holdout split. This still uses only your existing regression outputs (no retraining), but typically yields substantially better ordinal mapping than a small discrete delta grid. I also compute thresholds from *both* the regressor head and the classifier-expected-value head when ThreeStage weights exist, picking whichever gives better holdout QWK (still the same network, just better use of its outputs).'
- What this solution (achieved 0.02111) has done: 'Your current score (0.02111) is far below the target (0.9087), and the most likely reason (without changing your model/training core) is a mismatch between how the float predictions are generated for threshold-tuning vs how they’re generated for test inference when `use_three_stage=True`. I keep your architecture, transforms, and thresholding approach, but make inference use the *same* “best float source” (regressor head vs classifier-expected-value) that you already select during holdout tuning. I also fix `use_three_stage` to be tied to the actually-instantiated model (so it can’t accidentally call the wrong forward signature) and keep everything else identical so changes are minimal and directly aimed at improving QWK.'

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
from sklearn.model_selection import StratifiedShuffleSplit
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out, thr=None):
    thr = threshold if thr is None else thr
    prediction = torch.zeros(out.size(0), device="cpu")
    for i in range(4):
        prediction += (out.data >= thr[i]).squeeze().detach().cpu()
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
            l1 = int(math.floor(float(out[i].item())))
            l2 = int(math.ceil(float(out[i].item())))
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
    def __init__(self, pretrained_backbone=False):
        super(Regressor, self).__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b5_ns", pretrained=pretrained_backbone, num_classes=1000
        )
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None, pretrained_backbone=False):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=pretrained_backbone, num_classes=1000
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
DATA_DIR = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

WEIGHT_PATH = "../input/weights/B4_3stage_57epoch_CLAHE.pkl"

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].values

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

_use_three_stage_weights_exist = os.path.exists(WEIGHT_PATH)
if _use_three_stage_weights_exist:
    net = ThreeStage_Model(pretrained_backbone=False)
    state = torch.load(WEIGHT_PATH, map_location="cpu")
    net.load_state_dict(state)
    use_three_stage = True
else:
    print(
        f"WARNING: Weights not found at {WEIGHT_PATH}. Falling back to a pretrained backbone regressor for non-random predictions."
    )
    net = Regressor(pretrained_backbone=True)
    use_three_stage = False

net = net.to(device)
net.eval()




## === cell 5
def _predict_regression_for_ids(ids, img_dir, batch_size):
    preds = []
    batch = []
    with torch.inference_mode():
        for i, idx in enumerate(ids):
            image_name = os.path.join(img_dir, f"{idx}.png")
            img = Image.open(image_name).convert("RGB")
            batch.append(transform(img))
            if len(batch) == batch_size or i == len(ids) - 1:
                imgs = torch.stack(batch, dim=0).to(device)
                if use_three_stage:
                    _, r_out, _ = net(imgs)
                    r_out = r_out.data.squeeze(1)
                else:
                    r_out = net(imgs).data.squeeze(1)
                preds.extend(r_out.detach().cpu().numpy().astype(np.float32).tolist())
                batch = []
    return np.asarray(preds, dtype=np.float32)


def _predict_float_twohead_for_ids(ids, img_dir, batch_size):
    r_preds = []
    cexp_preds = []
    batch = []
    with torch.inference_mode():
        for i, idx in enumerate(ids):
            image_name = os.path.join(img_dir, f"{idx}.png")
            img = Image.open(image_name).convert("RGB")
            batch.append(transform(img))
            if len(batch) == batch_size or i == len(ids) - 1:
                imgs = torch.stack(batch, dim=0).to(device)
                c_out, r_out, _ = net(imgs)
                r_out = r_out.data.squeeze(1)
                probs = F.softmax(c_out, dim=1)
                ev = (
                    probs * torch.arange(5, device=probs.device, dtype=probs.dtype)
                ).sum(dim=1)
                r_preds.extend(r_out.detach().cpu().numpy().astype(np.float32).tolist())
                cexp_preds.extend(ev.detach().cpu().numpy().astype(np.float32).tolist())
                batch = []
    return np.asarray(r_preds, dtype=np.float32), np.asarray(
        cexp_preds, dtype=np.float32
    )


def _apply_thresholds(preds_float, thr):
    out = np.zeros_like(preds_float, dtype=np.int64)
    for t in thr:
        out += (preds_float >= t).astype(np.int64)
    return out.clip(0, 4)


def _sanitize_thresholds(thr):
    thr = np.asarray(thr, dtype=np.float32).copy()
    thr = np.clip(thr, 0.0, 4.5)
    thr = np.sort(thr)
    for j in range(1, 4):
        if thr[j] <= thr[j - 1]:
            thr[j] = min(4.5, thr[j - 1] + 1e-3)
    return thr


def _tune_thresholds_qwk(y_true, y_pred_float, init_thr=None, n_rounds=2):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred_float = np.asarray(y_pred_float, dtype=np.float32)

    thr = np.array(
        init_thr if init_thr is not None else [0.75, 1.5, 2.5, 3.5], dtype=np.float32
    )
    thr = _sanitize_thresholds(thr)

    def score_for(thr_vec):
        y_pred = _apply_thresholds(y_pred_float, thr_vec)
        return cohen_kappa_score(y_true, y_pred, weights="quadratic")

    def golden_search_on_k(thr_vec, k, lo, hi, iters=18):
        phi = (1 + 5**0.5) / 2
        invphi = 1 / phi
        invphi2 = invphi * invphi

        a, b = float(lo), float(hi)
        if b - a < 1e-6:
            return thr_vec, score_for(thr_vec)

        h = b - a
        c = a + invphi2 * h
        d = a + invphi * h

        def eval_at(x):
            cand = thr_vec.copy()
            cand[k] = x
            cand = _sanitize_thresholds(cand)
            return cand, score_for(cand)

        cand_c, sc_c = eval_at(c)
        cand_d, sc_d = eval_at(d)

        for _ in range(iters):
            if sc_c < sc_d:
                a = c
                c = d
                sc_c = sc_d
                cand_c = cand_d
                h = b - a
                d = a + invphi * h
                cand_d, sc_d = eval_at(d)
            else:
                b = d
                d = c
                sc_d = sc_c
                cand_d = cand_c
                h = b - a
                c = a + invphi2 * h
                cand_c, sc_c = eval_at(c)

        if sc_c >= sc_d:
            return cand_c, float(sc_c)
        return cand_d, float(sc_d)

    best = float(score_for(thr))

    for _ in range(n_rounds):
        for k in range(4):
            lo = 0.0 if k == 0 else float(thr[k - 1] + 1e-3)
            hi = 4.5 if k == 3 else float(thr[k + 1] - 1e-3)
            if hi <= lo + 1e-4:
                continue
            thr_candidate, sc = golden_search_on_k(thr, k, lo, hi, iters=18)
            if sc >= best:
                thr = thr_candidate
                best = sc

    thr = _sanitize_thresholds(thr)
    return thr.tolist(), float(best)


float_source_for_test = "regressor"  # default for regressor-only model

if os.path.exists(TRAIN_CSV) and os.path.isdir(TRAIN_IMG_DIR):
    train_df = pd.read_csv(TRAIN_CSV)
    y = train_df["diagnosis"].values.astype(int)
    ids = train_df["id_code"].values

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    tr_idx, val_idx = next(splitter.split(ids, y))
    val_ids = ids[val_idx]
    y_val = y[val_idx]

    BATCH_SIZE_FIT = 8 if device == "cuda" else 4

    if use_three_stage:
        val_pred_r, val_pred_cexp = _predict_float_twohead_for_ids(
            val_ids, TRAIN_IMG_DIR, BATCH_SIZE_FIT
        )
        tuned_thr_r, tuned_qwk_r = _tune_thresholds_qwk(
            y_val, val_pred_r, init_thr=threshold, n_rounds=3
        )
        tuned_thr_c, tuned_qwk_c = _tune_thresholds_qwk(
            y_val, val_pred_cexp, init_thr=threshold, n_rounds=3
        )

        if tuned_qwk_c > tuned_qwk_r:
            tuned_threshold = tuned_thr_c
            float_source_for_test = "classifier_expected_value"
            tuned_qwk = tuned_qwk_c
        else:
            tuned_threshold = tuned_thr_r
            float_source_for_test = "regressor_head"
            tuned_qwk = tuned_qwk_r

        print(
            f"Tuned thresholds (holdout, source={float_source_for_test}): {tuned_threshold} (holdout QWK={tuned_qwk:.5f})"
        )
    else:
        val_pred = _predict_regression_for_ids(val_ids, TRAIN_IMG_DIR, BATCH_SIZE_FIT)
        tuned_threshold, tuned_qwk = _tune_thresholds_qwk(
            y_val, val_pred, init_thr=threshold, n_rounds=3
        )
        float_source_for_test = "regressor"
        print(
            f"Tuned thresholds (holdout): {tuned_threshold} (holdout QWK={tuned_qwk:.5f})"
        )
else:
    tuned_threshold = threshold
    print("WARNING: Train data not found; using default thresholds.")
    float_source_for_test = "regressor"



## === cell 6
submission = []
batch = []


def _load_one(idx):
    image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
    img = Image.open(image_name).convert("RGB")
    return transform(img)


BATCH_SIZE = 8 if device == "cuda" else 4

with torch.inference_mode():
    for i, idx in enumerate(test_ids):
        batch.append(_load_one(idx))
        if len(batch) == BATCH_SIZE or i == len(test_ids) - 1:
            imgs = torch.stack(batch, dim=0).to(device)

            if use_three_stage:
                c_out, r_out, _ = net(imgs)
                r_out = r_out.data.squeeze(1)

                if float_source_for_test == "classifier_expected_value":
                    probs = F.softmax(c_out, dim=1)
                    preds_float = (
                        probs * torch.arange(5, device=probs.device, dtype=probs.dtype)
                    ).sum(dim=1)
                else:
                    preds_float = r_out
            else:
                preds_float = net(imgs).data.squeeze(1)

            preds = (
                regress2class(preds_float, thr=tuned_threshold)
                .numpy()
                .astype(int)
                .tolist()
            )

            start = i - (len(batch) - 1)
            batch_ids = test_ids[start : i + 1].tolist()
            submission.extend(list(zip(batch_ids, preds)))
            batch = []

submission_df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

assert len(submission_df) == len(
    test_ids
), f"Submission rows {len(submission_df)} != test rows {len(test_ids)}"
submission_df["diagnosis"] = submission_df["diagnosis"].astype(int).clip(0, 4)



## === cell 7
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Wrote {submission_path} with shape {submission_df.shape}")
print(submission_df.head())
