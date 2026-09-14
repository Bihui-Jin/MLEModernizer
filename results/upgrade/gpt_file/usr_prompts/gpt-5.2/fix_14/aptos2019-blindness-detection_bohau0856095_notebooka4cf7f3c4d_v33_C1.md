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

0.9117575056429484

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.01221) has done: 'I (1) make the script robust to the missing external weights by falling back to a standard timm pretrained EfficientNet backbone, so it can always run end-to-end and create a submission.csv. I also (2) fix the dtype/device mismatch by ensuring the model is moved to the same device as the input (and by loading state_dict onto CPU first, then moving to GPU). Finally, I (3) keep the original inference logic (regression + fixed thresholds) intact, only changing how weights are sourced and loaded so the pipeline produces a valid submission with the correct columns and row count.'
- What this solution (achieved 0.04626) has done: 'Your current score is far below the target, and the biggest likely cause is that inference is running with essentially untrained heads (because only the backbone gets ImageNet weights when the custom checkpoint is missing), which makes the regression output near-random for DR. To move the score toward the target with minimal change, I (1) switch the fallback weights to a full ImageNet-pretrained EfficientNet-B4 model (so the backbone at least matches the forward graph), and (2) replace the fixed, non-optimized thresholds with a tiny, label-based threshold fitting step on the provided `train.csv` using the same regression head output, maximizing QWK on train predictions (no architecture/training changes, just calibration). This keeps your core model and inference flow (regression + thresholds) intact, but makes the discretization far less arbitrary, which is typically decisive for QWK. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so we should improve performance with the smallest changes that keep your architecture/inference (regression + thresholds) intact. The biggest issue is that when the custom checkpoint is missing, only the backbone has meaningful weights while the heads (regressor/ordinal/classifier/final_regressor) are randomly initialized, producing near-random continuous predictions; we initialize those heads deterministically with small weights/biases to output reasonable mid-range values rather than noise. Then we make the threshold fitting less overfit-prone by fitting thresholds on an out-of-fold prediction set (same model outputs, just better calibration), and finally use the same thresholding function consistently in both threshold fitting and test inference to avoid subtle mismatches. These are calibration/initialization fixes only—no training loops, losses, or architecture changes—so runtime stays within the 600s budget.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score strongly suggests the submission is being evaluated as invalid/misaligned rather than merely low-performing. The biggest risk in your script is that `test_df["id_code"]` can become a 2D array via `np.squeeze`, which can silently produce wrong filenames/IDs and lead to a broken submission (all zeros or mismatched ids). I make the test id extraction strictly 1D and string-typed, and I also ensure train/test IDs are consistently strings (no accidental numeric coercion), while keeping your model, transforms, OOF threshold fitting, and inference logic unchanged. Finally, I add a lightweight submission alignment check against `sample_submission.csv` (same ids, same order) to prevent another 0.0 due to format/order issues.'
- What this solution (achieved 0.0) has done: 'Your 0.0 leaderboard score is most consistent with an invalid submission (e.g., duplicate/missing ids, wrong id order, or wrong diagnosis dtype), not just weak model predictions. I make the smallest changes to guarantee (1) predictions cover every test id exactly once, (2) the submission is aligned to `sample_submission.csv` in the exact same order, and (3) `diagnosis` is a clean integer 0–4 column—while keeping your model, transforms, regression inference, and OOF threshold fitting intact. I also fix a subtle indexing bug in OOF assembly (using per-fold indices incorrectly) that can corrupt threshold fitting and thus hurt score, without changing the overall approach. These changes should move the score up from 0.0 toward a meaningful value (and thus toward your target), without altering your architecture/training setup.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an evaluation-alignment issue (IDs/order) rather than just weak predictions, so I make the submission strictly match `sample_submission.csv` IDs and order, and add hard checks that every test ID is predicted exactly once. To move the score upward (toward your target) without changing your model or training/inference approach, I also fix a subtle OOF assembly bug: mapping `id_code -> global_index` using `zip(val_ids, val_idx)` is unsafe if `ids_used` order differs or some images are missing; we instead build a global mapping from all `train_ids` once. Finally, I ensure missing-image cases don’t silently corrupt calibration by fitting thresholds only on the subset of OOF predictions that were actually produced (same grid search, just correct inputs).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the model is producing near-constant predictions (or otherwise extremely miscalibrated), even though the submission format/alignment checks look correct. To move the score upward with minimal change and without altering the model/training logic, I keep your regression-head inference but switch the test-time prediction to use the model’s `final=True` output (the intended “final regressor” fusion of classifier+regressor+ordinal), while keeping the exact same thresholding/csv logic. I also make the OOF threshold fitting use the same `final=True` continuous output, so calibration matches inference (same grid search, same metric). This should increase QWK materially versus using only the intermediate regressor head, while remaining within your existing architecture and workflow.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score indicates the submission is likely being scored as invalid despite the checks, and the most common hidden cause is a wrong dataset root (reading the 367-row subset instead of the full test set) or pointing at empty/mismatched image directories. I make the data root selection deterministic by preferring the standard `/kaggle/input/aptos2019-blindness-detection` and verifying that the chosen directory has the expected train/test image counts; if not, we fall back to the other candidates. This is a minimal change that preserves your model and inference logic but ensures you actually predict on the correct 367 test IDs with existing images. I also add a hard check that every `id_code` in `test.csv` has a corresponding image file and fail early if not, preventing another silent “all zeros” submission that can produce 0.0.'
- What this solution (achieved -0.00714) has done: 'I fix the runtime failure by removing the hard stop when predictions are degenerate, and instead apply a minimal, deterministic fallback that guarantees at least two classes so a valid `submission.csv` is always written. This keeps your core model and inference (three-stage model, final regression output, OOF threshold fitting, threshold-based discretization) unchanged, and only adjusts the last-mile post-processing needed to produce a non-degenerate submission when the model collapses. I also add a very small safety clamp on fitted thresholds to keep them ordered and within a plausible range, avoiding edge cases where discretization collapses to a single class. These changes are directly aimed at end-to-end stability and moving the score from “not yielded” to a valid, scorable submission.'
- What this solution (achieved 0.0) has done: 'Your score is far below the target, so we should improve performance (not degrade it) with the smallest changes that keep your model and inference semantics intact. The biggest likely issue is that your threshold fitting is using out-of-fold predictions from a model that is never trained (and may be using fallback weights), so the fitted thresholds can become effectively arbitrary and hurt QWK; we fit thresholds on a stratified subsample to reduce compute while keeping semantics, and we make the threshold search more stable by optimizing all four thresholds jointly via a small coordinate-descent loop (same discretization, just better calibration). We also remove the “degeneracy fix” that intentionally injects label noise into the submission, because it can push QWK negative when the model is already weak; instead, we only apply a deterministic, non-noisy fallback if predictions are truly invalid (NaN/Inf). These changes are calibration/post-processing only (no architecture, loss, or training loop changes) and still produce a valid `submission.csv` end-to-end within the time budget.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the model is effectively untrained (custom checkpoint missing), and the current OOF threshold-fitting is wasting time and may even crash/overfit while still leaving test predictions near-constant. To move the score up toward your target with minimal semantic change, I keep your exact architecture and regression→threshold discretization, but (1) switch the fallback backbone to the matching **tf_efficientnet_b4_ns(pretrained=True)** with **num_classes=0** so features are correct-dimension and meaningful, and (2) add a tiny, deterministic **single-image TTA (horizontal flip average)** at inference + OOF prediction to stabilize continuous outputs. I also remove the “refuse to fit thresholds” hard stop by fitting thresholds on whatever OOF coverage exists (still using the same coordinate-descent and QWK), which prevents ending up with a degenerate/unusable pipeline that can yield 0.0. These are minimal, metric-aligned changes that should raise you from 0.0 into a non-trivial QWK range without altering the overall approach.'

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

import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True




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
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold


DATA_DIR_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
]


def _count_pngs(d):
    try:
        return sum(1 for f in os.listdir(d) if f.lower().endswith(".png"))
    except Exception:
        return -1


def _score_candidate(d):
    train_csv = os.path.join(d, "train.csv")
    test_csv = os.path.join(d, "test.csv")
    sample_csv = os.path.join(d, "sample_submission.csv")
    train_img = os.path.join(d, "train_images")
    test_img = os.path.join(d, "test_images")
    ok = (
        os.path.exists(train_csv)
        and os.path.exists(test_csv)
        and os.path.exists(sample_csv)
        and os.path.isdir(train_img)
        and os.path.isdir(test_img)
    )
    if not ok:
        return (-1, -1, -1)
    return (_count_pngs(train_img), _count_pngs(test_img), 1)


best = None
DATA_DIR = None
for d in DATA_DIR_CANDIDATES:
    sc = _score_candidate(d)
    if best is None or sc > best:
        best = sc
        DATA_DIR = d

if DATA_DIR is None or best[2] < 0:
    raise FileNotFoundError(
        f"Could not locate a valid competition data dir in candidates: {DATA_DIR_CANDIDATES}"
    )

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df["id_code"] = train_df["id_code"].astype(str)
test_df["id_code"] = test_df["id_code"].astype(str)

test_ids = test_df["id_code"].astype(str).tolist()

missing_test_imgs = [
    i for i in test_ids if not os.path.exists(os.path.join(TEST_IMG_DIR, f"{i}.png"))
]
if len(missing_test_imgs) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing_test_imgs)}/{len(test_ids)} test images under {TEST_IMG_DIR}. "
        f"Example missing id_code: {missing_test_imgs[0]}"
    )

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

net = ThreeStage_Model()

WEIGHTS_CANDIDATES = [
    "../input/weights/B4_3stage_40epoch_CLAHE.pkl",
    "/kaggle/input/weights/B4_3stage_40epoch_CLAHE.pkl",
]
weight_path = None
for p in WEIGHTS_CANDIDATES:
    if os.path.exists(p):
        weight_path = p
        break

if weight_path is None:
    for root, _, files in os.walk("/kaggle/input"):
        if "B4_3stage_40epoch_CLAHE.pkl" in files:
            weight_path = os.path.join(root, "B4_3stage_40epoch_CLAHE.pkl")
            break

if weight_path is not None:
    state = torch.load(weight_path, map_location="cpu")
    net.load_state_dict(state, strict=True)
    print("Loaded custom weights from:", weight_path)
else:
    net.backbone = timm.create_model(
        "tf_efficientnet_b4_ns",
        pretrained=True,
        num_classes=0,
        global_pool="",  # we'll set our GeM pool
    )
    net.backbone.global_pool = GeM(flatten=True)

    def _init_head_stable(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    net.classifier.apply(_init_head_stable)
    net.ordinal.apply(_init_head_stable)
    net.final_regressor.apply(_init_head_stable)
    net.regressor.apply(_init_head_stable)

    with torch.no_grad():
        last = net.regressor[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.weight, 0.0)
            nn.init.constant_(last.bias, 0.0)

    print(
        "WARNING: Custom weights not found; using timm pretrained tf_efficientnet_b4_ns (num_classes=0 features) + stable head init."
    )

net = net.to(device=device, dtype=torch.float32)
net.eval()

print("Data dir:", DATA_DIR)
print("Train csv rows:", len(train_df), "Test csv rows:", len(test_ids))
print("Train images (png) in dir:", _count_pngs(TRAIN_IMG_DIR))
print("Test images (png) in dir:", _count_pngs(TEST_IMG_DIR))




## === cell 5
def _safe_open_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def _predict_continuous_single(img_t, use_final=True):
    out1 = net(img_t, final=True) if use_final else net(img_t)[1]
    out2 = (
        net(torch.flip(img_t, dims=[3]), final=True)
        if use_final
        else net(torch.flip(img_t, dims=[3]))[1]
    )
    out = 0.5 * (out1 + out2)
    return float(out.squeeze(1).item())


def predict_regression_for_ids(id_list, img_dir, max_images=None, use_final=True):
    preds = []
    ids_used = []
    with torch.no_grad():
        for k, idx in enumerate(id_list):
            if max_images is not None and k >= max_images:
                break
            image_name = os.path.join(img_dir, f"{idx}.png")
            try:
                img = _safe_open_rgb(image_name)
            except Exception:
                continue

            img_t = transform(img).unsqueeze(0).to(device=device, dtype=torch.float32)

            try:
                pred = _predict_continuous_single(img_t, use_final=use_final)
            except Exception:
                continue

            preds.append(pred)
            ids_used.append(str(idx))
    return np.array(ids_used, dtype=str), np.array(preds, dtype=np.float32)


def apply_thresholds_continuous(x, thr):
    thr = np.asarray(thr, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return (x[:, None] >= thr[None, :]).sum(axis=1).astype(np.int64)


def _sanitize_thresholds(thr):
    thr = np.asarray(thr, dtype=np.float32).reshape(-1)
    thr = np.clip(thr, -1.0, 5.5)
    eps = 1e-3
    for i in range(1, len(thr)):
        if not (thr[i] > thr[i - 1] + eps):
            thr[i] = thr[i - 1] + eps
    return thr.tolist()


def fit_thresholds_coordinate_descent(
    y_true, y_pred_cont, init_thr, span=1.0, step=0.05, n_rounds=3
):
    y_true = np.asarray(y_true, dtype=np.int64).reshape(-1)
    y_pred_cont = np.asarray(y_pred_cont, dtype=np.float32).reshape(-1)
    base = np.asarray(init_thr, dtype=np.float32).copy()
    thr = base.copy()

    grid = np.arange(-span, span + 1e-9, step, dtype=np.float32)

    def score(th):
        y_hat = apply_thresholds_continuous(y_pred_cont, th)
        return cohen_kappa_score(y_true, y_hat, weights="quadratic")

    best_kappa = score(thr)
    best_thr = thr.copy()

    for _ in range(n_rounds):
        for t in range(4):
            local_best_thr = thr.copy()
            local_best_kappa = best_kappa
            for delta in grid:
                cand = thr.copy()
                cand[t] = thr[t] + float(delta)
                if not (cand[0] < cand[1] < cand[2] < cand[3]):
                    continue
                kappa = score(cand)
                if kappa > local_best_kappa:
                    local_best_kappa = kappa
                    local_best_thr = cand
            thr = local_best_thr
            best_kappa = local_best_kappa
            best_thr = thr.copy()

    return _sanitize_thresholds(best_thr.tolist()), float(best_kappa)


train_ids = train_df["id_code"].astype(str).to_numpy()
y_true = train_df["diagnosis"].to_numpy(dtype=np.int64)

id_to_global = {id_code: i for i, id_code in enumerate(train_ids)}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_pred = np.full(len(train_df), np.nan, dtype=np.float32)

t0 = time.time()
for fold, (_, val_idx) in enumerate(skf.split(train_ids, y_true), 1):
    val_ids = train_ids[val_idx]
    ids_used, y_pred_cont = predict_regression_for_ids(
        val_ids, TRAIN_IMG_DIR, use_final=True
    )

    for id_code, pred in zip(ids_used, y_pred_cont):
        gi = id_to_global.get(str(id_code), None)
        if gi is not None:
            oof_pred[gi] = pred

    print(f"OOF fold {fold}/5: predicted {len(ids_used)}/{len(val_ids)} images")

seen = ~np.isnan(oof_pred)
coverage = float(seen.mean())
print("OOF prediction coverage:", coverage)
print("OOF prediction time (s):", round(time.time() - t0, 2))

if seen.sum() < 500:
    print("WARNING: Very low OOF coverage; using default thresholds without fitting.")
else:
    rng = np.random.RandomState(42)
    max_fit = (
        2500  # small increase for more stable calibration; still within time budget
    )
    idx_seen = np.flatnonzero(seen)
    if idx_seen.size > max_fit:
        chosen = []
        for cls in range(5):
            cls_idx = idx_seen[y_true[idx_seen] == cls]
            if cls_idx.size == 0:
                continue
            take = int(round(max_fit * (cls_idx.size / idx_seen.size)))
            take = max(1, min(take, cls_idx.size))
            chosen.append(rng.choice(cls_idx, size=take, replace=False))
        fit_idx = np.unique(np.concatenate(chosen))
    else:
        fit_idx = idx_seen

    fitted_thr, oof_kappa = fit_thresholds_coordinate_descent(
        y_true[fit_idx], oof_pred[fit_idx], threshold, span=1.0, step=0.05, n_rounds=3
    )
    threshold = fitted_thr
    print("Fitted thresholds (from OOF preds, subsampled):", threshold)
    print("OOF QWK on fit subset (sanity check):", oof_kappa)




## === cell 6
pred_map = {}

with torch.no_grad():
    for idx in test_ids:
        image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
        try:
            img = _safe_open_rgb(image_name)
            img_t = transform(img).unsqueeze(0).to(device=device, dtype=torch.float32)

            pred_cont = _predict_continuous_single(img_t, use_final=True)

            if not np.isfinite(pred_cont):
                raise ValueError("Non-finite prediction")
            pred_cls = int(apply_thresholds_continuous([pred_cont], threshold)[0])
        except Exception:
            pred_cls = 0
        pred_map[str(idx)] = pred_cls

if not os.path.exists(SAMPLE_SUB_CSV):
    raise FileNotFoundError(
        "sample_submission.csv not found; cannot guarantee correct id order."
    )
sample_df = pd.read_csv(SAMPLE_SUB_CSV)
sample_df["id_code"] = sample_df["id_code"].astype(str)

sample_ids = sample_df["id_code"].tolist()
missing_in_pred = [i for i in sample_ids if i not in pred_map]
if len(missing_in_pred) > 0:
    for i in missing_in_pred:
        pred_map[i] = 0

df = sample_df[["id_code"]].copy()
df["diagnosis"] = df["id_code"].map(pred_map).fillna(0).astype(int).clip(0, 4)

if df.shape[0] != sample_df.shape[0]:
    raise RuntimeError(
        f"Submission row count mismatch vs sample: df.shape={df.shape}, sample={sample_df.shape}"
    )
if df["id_code"].duplicated().any():
    raise RuntimeError(
        "Submission has duplicate id_code entries, would likely score 0.0"
    )
if df["diagnosis"].isna().any():
    raise RuntimeError("Submission has NaN diagnosis entries, would likely score 0.0")
if set(df["id_code"]) != set(sample_ids):
    raise RuntimeError(
        "Submission ids set does not match sample_submission ids set (would likely score 0.0)"
    )
if df["id_code"].tolist() != sample_ids:
    raise RuntimeError(
        "Submission id order does not match sample_submission order (would likely score 0.0)"
    )

vc = df["diagnosis"].value_counts().sort_index()
print("Test prediction label distribution:\n", vc.to_string())

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
