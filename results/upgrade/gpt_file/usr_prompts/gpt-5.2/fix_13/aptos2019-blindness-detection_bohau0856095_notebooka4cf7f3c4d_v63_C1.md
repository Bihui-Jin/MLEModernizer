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

0.9141001098370056

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'You’re failing because the notebook expects a local pretrained weights file that isn’t present in the Kaggle dataset, so it crashes before inference and produces no submission. I keep the same model and preprocessing, but add a safe fallback: if the weight file is missing, run inference with the randomly initialized network so a valid `submission.csv` is always written. I also fix the CUDA/CPU dtype mismatch by ensuring the loaded checkpoint tensors (if any) and the model are moved to the same device, and make inference robust so it can’t leave an empty submission. These changes are execution/stability fixes; without the missing weights your score won’t reach the target, but you get a valid submission file end-to-end.'
- What this solution (achieved 0.03852) has done: 'Your current 0.0 score is consistent with effectively-random predictions (likely because the intended pretrained weights aren’t available), so the smallest legitimate move toward the target is to (1) ensure we use pretrained ImageNet weights for the backbone when the competition checkpoint is missing, and (2) calibrate the 4 thresholds on a held-out validation split using quadratic weighted kappa, then reuse those thresholds for test inference. This preserves your exact model architecture and inference semantics (regression output + thresholding), but replaces the fixed hand-chosen thresholds with data-driven ones, which typically improves QWK substantially versus defaults. I also keep the “safe fallback” behavior so a valid `submission.csv` is always written even if no checkpoint exists. All changes are contained to weight initialization, adding a small validation loop, and using calibrated thresholds at inference.'
- What this solution (achieved 0.12958) has done: 'Your current gap to the target is very large (0.0385 vs 0.9141), and the most likely cause is that you’re effectively running an untrained head (and possibly not using the “final” fused regressor at all), so predictions are close to random even with an ImageNet backbone. I keep your model and preprocessing unchanged, but switch inference/calibration to use `forward(final=True)` (the model’s intended final regressor) and calibrate thresholds on that final regression output, which directly aligns with QWK. I also make `regress2class` device-safe and avoid `.data` usage to prevent subtle issues, while keeping identical thresholding semantics. These are minimal changes that should legitimately move the score upward toward the target without changing the training approach or architecture.'
- What this solution (achieved 0.30841) has done: 'The crash comes from a feature-dimension mismatch: EfficientNet-B4/B5 backbones output 1792/2048 features, but the heads are hard-coded for 1000, so linear layers can’t multiply. I fix this by deriving the backbone feature dimension from `timm` (`backbone.num_features`) and using it for all head layers, preserving the same architecture and forward logic. I also keep your checkpoint fallback behavior, but ensure the fallback backbone matches the model’s expected backbone so inference and threshold calibration run end-to-end. Finally, I ensure a submission.csv is always written even if something fails earlier by preventing `submission` from being undefined.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so we should push it up with the smallest changes that keep your exact model/inference approach (final regression output + thresholding). The main issue is that when the competition checkpoint is missing, only the backbone is ImageNet-pretrained while all heads (classifier/regressor/ordinal/final_regressor) remain randomly initialized, which severely hurts QWK. I keep the same architecture and threshold calibration, but (1) switch the fallback to load a full ImageNet-pretrained EfficientNet via `timm` and reuse its classifier weights to sensibly initialize your head layers (a legitimate warm-start, not a new model), and (2) make threshold calibration more reliable by calibrating on out-of-fold predictions (still no training, just better threshold fit), then use the averaged thresholds for test inference. These are minimal, metric-aligned changes that should move QWK substantially upward toward your target while preserving core semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the submission was previously invalid or misaligned; the smallest score-moving fix is to guarantee perfect row/ID alignment with `sample_submission.csv` (the canonical ordering Kaggle expects) and to make inference robust to missing/corrupt images so we always emit exactly 367 predictions. I keep your exact model architecture and regression→thresholding semantics, but I also make threshold calibration deterministic and consistent by freezing seeds in DataLoader workers and ensuring calibration uses the same transform/device behavior as test. Finally, I add a sanity check that `id_code` matches the sample submission set and fill any missing predictions with a safe default class to prevent accidental NaNs/empty outputs (valid CSV always).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the model never loading the intended competition weights and then producing effectively-random outputs (even after threshold calibration), so the smallest legitimate move toward the 0.914 target is to actually find and load the checkpoint if it exists anywhere in `/kaggle/input` (many Kaggle datasets nest files under multiple subfolders). I keep your exact architecture and inference semantics (`forward(final=True)` + regression→thresholding), but broaden the checkpoint search to include common filename variants and also handle `module.`-prefixed keys so a found checkpoint loads cleanly. If no checkpoint is found, the current ImageNet-backbone fallback remains unchanged so the notebook still always produces a valid `submission.csv`. These changes are directly score-relevant (using the intended trained weights) while minimal and safe.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is far below the 0.914 target, so the smallest legitimate move upward is to ensure we actually load a trained competition checkpoint if it exists somewhere under `/kaggle/input` (instead of silently falling back to mostly-random heads). I broaden the checkpoint search patterns to include common filenames (e.g., `*.bin`, `*.ckpt`, generic `*3stage*`, `*efficientnet*b4*`) and add robust state-dict extraction (handles `state_dict`, `model`, nested dicts) plus key-shape filtering so partial-but-correct weights still load rather than failing. If no checkpoint is found, the current ImageNet-backbone fallback remains unchanged to guarantee a valid `submission.csv` is always produced. These changes keep your model architecture, forward path (`final=True`), preprocessing, and threshold calibration semantics intact, but make it much more likely you use the intended trained weights and thus increase QWK toward the target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly indicates an invalid/garbled submission (not just “bad model”): most commonly this comes from non-integer/NaN diagnoses, wrong row alignment, or silently broken inference producing empty/constant outputs. I make two minimal, score-relevant fixes: (1) ensure the test images are read correctly by resolving the *actual* image directory (your `INPUT_ROOT` points at a path that may not contain the extracted folders), and (2) make thresholding consistent by applying your calibrated thresholds during test inference (right now inference uses the global `threshold` but calibration uses numpy; we unify and also clamp predictions to [0,4]). These changes preserve your exact model architecture and “final regression + thresholds” semantics, but make the pipeline reliably produce meaningful, valid class predictions in the correct order, moving QWK upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with an invalid/garbled submission rather than “just a weak model”, and the highest-leverage minimal fix is to ensure test-time class conversion uses the calibrated thresholds (right now `regress2class()` reads a global list, but it’s easy for that to diverge from calibration). I keep your exact model forward path (`final=True`), dataset, transforms, and calibration logic, but make thresholding explicit at inference by applying `apply_thresholds()` on the raw regression outputs using the calibrated `threshold`. I also add a small safety clamp for regression outputs to `[0, 4.5]` before thresholding (doesn’t change semantics when outputs are in-range, but prevents rare NaNs/inf/outliers from producing invalid values). These changes are tightly scoped to prediction post-processing/alignment and should move the score upward from 0.0 by guaranteeing a valid, metric-consistent submission.'

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    out = out.detach()
    thr = torch.tensor(threshold, device=out.device, dtype=out.dtype).view(1, -1)
    pred = (out.view(-1, 1) >= thr).sum(dim=1)
    pred = pred.clamp_(0, 4)
    return pred.to(torch.int64).cpu()


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
        self.backbone = timm.create_model(
            "tf_efficientnet_b5_ns", pretrained=False, num_classes=0, global_pool=""
        )
        self.backbone.global_pool = GeM(flatten=True)
        in_features = getattr(self.backbone, "num_features", None)
        if in_features is None:
            raise RuntimeError(
                "Backbone does not expose num_features; cannot size regressor."
            )
        self.regressor = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None):
        super(ThreeStage_Model, self).__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=False, num_classes=0, global_pool=""
        )
        self.backbone.global_pool = GeM(flatten=True)

        in_features = getattr(self.backbone, "num_features", None)
        if in_features is None:
            raise RuntimeError(
                "Backbone does not expose num_features; cannot size heads."
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
        if bbox:
            return image.crop(bbox)
        return image




## === cell 4
INPUT_ROOT = "/kaggle/input/aptos2019-blindness-detection"
FALLBACK_ROOT = "/kaggle/input"

train_csv_path = os.path.join(INPUT_ROOT, "train.csv")
test_csv_path = os.path.join(INPUT_ROOT, "test.csv")
sample_csv_path = os.path.join(INPUT_ROOT, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
sample_df = pd.read_csv(sample_csv_path)

sample_ids = sample_df["id_code"].astype(str).values
test_ids = test_df["id_code"].astype(str).values

if set(sample_ids) != set(test_ids):
    print("WARNING: sample_submission IDs and test.csv IDs differ.")
    print(
        "sample:",
        len(sample_ids),
        "test:",
        len(test_ids),
        "intersection:",
        len(set(sample_ids) & set(test_ids)),
    )

test_ids_ordered = sample_ids

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


class RetinaDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_id = str(row["id_code"])
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        if "diagnosis" in self.df.columns:
            y = int(row["diagnosis"])
            return img, y
        return img, img_id


def resolve_image_dir(preferred_path, fallback_root, folder_name):
    if (
        os.path.isdir(preferred_path)
        and len(glob.glob(os.path.join(preferred_path, "*.png"))) > 0
    ):
        return preferred_path
    candidates = glob.glob(
        os.path.join(fallback_root, "**", folder_name), recursive=True
    )
    candidates = [
        p
        for p in candidates
        if os.path.isdir(p) and len(glob.glob(os.path.join(p, "*.png"))) > 0
    ]
    if len(candidates) == 0:
        return preferred_path
    candidates = sorted(
        candidates,
        key=lambda p: (0 if "aptos2019-blindness-detection" in p else 1, len(p)),
    )
    return candidates[0]


train_img_dir = resolve_image_dir(
    os.path.join(INPUT_ROOT, "train_images"), FALLBACK_ROOT, "train_images"
)
test_img_dir = resolve_image_dir(
    os.path.join(INPUT_ROOT, "test_images"), FALLBACK_ROOT, "test_images"
)
print("Resolved train_img_dir:", train_img_dir)
print("Resolved test_img_dir :", test_img_dir)




## === cell 5
def _maybe_init_heads_from_imagenet(net: ThreeStage_Model):
    """
    Change rationale (score): when the competition checkpoint is missing, the heads are random.
    We keep the SAME heads/architecture but initialize them from an ImageNet EfficientNet-B4
    classifier as a warm-start (legitimate, no label leakage), which improves predictions and QWK.
    """
    try:
        ref = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=True, num_classes=1000
        )
        ref.eval()
        ref_cls = getattr(ref, "classifier", None)
        if not isinstance(ref_cls, nn.Linear):
            print(
                "WARNING: Could not access ref.classifier as nn.Linear; skipping head init."
            )
            return

        W = ref_cls.weight.detach().cpu()
        b = ref_cls.bias.detach().cpu()

        for head_name in ["classifier", "regressor", "ordinal"]:
            head = getattr(net, head_name, None)
            if head is None:
                continue
            lin1 = head[1]
            if not isinstance(lin1, nn.Linear):
                continue
            if lin1.in_features != W.shape[1]:
                print(f"WARNING: {head_name} in_features mismatch; skipping init.")
                continue

            lin1.weight.data.copy_(W[: lin1.out_features, :])
            lin1.bias.data.copy_(b[: lin1.out_features])

        fr = net.final_regressor[1]
        if isinstance(fr, nn.Linear) and fr.in_features == 10 and fr.out_features == 1:
            with torch.no_grad():
                fr.weight.fill_(0.0)
                fr.bias.fill_(0.0)
    except Exception as e:
        print(
            "WARNING: Head init from ImageNet failed; continuing with default init. Error:",
            repr(e),
        )


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k[len("module.") :]: v for k, v in state_dict.items()}


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if all(isinstance(k, str) for k in obj.keys()) and any(
            torch.is_tensor(v) for v in obj.values()
        ):
            return obj
        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
        ]:
            if key in obj and isinstance(obj[key], dict):
                cand = _extract_state_dict(obj[key])
                if isinstance(cand, dict) and len(cand):
                    return cand
    return None


def _find_checkpoint(root_dir):
    patterns = [
        "**/B4_3stage_2epoch_finetune.pkl",
        "**/B4_3stage_2epoch_finetune.pth",
        "**/B4_3stage_2epoch_finetune.pt",
        "**/*.ckpt",
        "**/*.bin",
        "**/*.pth",
        "**/*.pt",
        "**/*.pkl",
        "**/*3stage*finetune*.*",
        "**/*3stage*.*",
        "**/*efficientnet*b4*.*",
        "**/*tf_efficientnet_b4*.*",
        "**/*aptos*.*",
        "**/*blindness*.*",
    ]
    cands = []
    for pat in patterns:
        cands.extend(glob.glob(os.path.join(root_dir, pat), recursive=True))
    cands = [p for p in cands if os.path.isfile(p)]

    exact = [p for p in cands if os.path.basename(p) == "B4_3stage_2epoch_finetune.pkl"]
    rest = [p for p in cands if p not in exact]

    def _rank(p):
        base = os.path.basename(p).lower()
        score = 0
        if "3stage" in base:
            score -= 10
        if "b4" in base:
            score -= 5
        if "finetune" in base:
            score -= 3
        if base.endswith(".pth") or base.endswith(".pt"):
            score -= 2
        if "aptos2019-blindness-detection" in p.lower():
            score -= 1
        score += len(p) / 2000.0
        return score

    rest_sorted = sorted(rest, key=_rank)

    out = []
    seen = set()
    for p in exact + rest_sorted:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _filter_by_shape(model, state_dict):
    model_sd = model.state_dict()
    filtered = {}
    dropped = 0
    for k, v in state_dict.items():
        if k in model_sd and torch.is_tensor(v) and model_sd[k].shape == v.shape:
            filtered[k] = v
        else:
            dropped += 1
    return filtered, dropped


net = ThreeStage_Model()

weights_path = None
ckpt_candidates = _find_checkpoint(FALLBACK_ROOT)

for cand in ckpt_candidates:
    try:
        obj = torch.load(cand, map_location="cpu")
        sd = _extract_state_dict(obj)
        if sd is None:
            continue
        sd = _strip_module_prefix(sd)
        sd_filt, dropped = _filter_by_shape(net, sd)
        if len(sd_filt) == 0:
            continue
        weights_path = cand
        state_to_load = sd_filt
        print(
            f"Found checkpoint candidate: {cand} (kept {len(sd_filt)} keys, dropped {dropped})"
        )
        break
    except Exception:
        continue

if weights_path is None:
    print(
        "WARNING: Could not find a usable competition checkpoint under", FALLBACK_ROOT
    )
    print(
        "Using ImageNet pretrained EfficientNet-B4 weights as a safe, legitimate fallback."
    )
    net.backbone = timm.create_model(
        "tf_efficientnet_b4_ns", pretrained=True, num_classes=0, global_pool=""
    )
    net.backbone.global_pool = GeM(flatten=True)
    _maybe_init_heads_from_imagenet(net)
else:
    print("Loading weights from:", weights_path)

net = net.to(device)

if weights_path is not None:
    missing, unexpected = net.load_state_dict(state_to_load, strict=False)
    if len(missing) > 0 or len(unexpected) > 0:
        print(
            f"Loaded with strict=False. Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}"
        )

net.eval()




## === cell 6
def predict_final_regression(model, loader):
    model.eval()
    preds = []
    gts = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x, final=True)
            preds.append(out.squeeze(1).detach().cpu().numpy())
            gts.append(np.asarray(y, dtype=np.int64))
    return np.concatenate(preds), np.concatenate(gts)


def apply_thresholds(values, thr):
    values = np.asarray(values, dtype=np.float32)
    thr = list(thr)
    out = np.zeros(values.shape[0], dtype=np.int64)
    for t in thr:
        out += (values >= t).astype(np.int64)
    out = np.clip(out, 0, 4)
    return out


def qwk_for_thresholds(y_true, y_pred_reg, thr):
    y_pred_cls = apply_thresholds(y_pred_reg, thr)
    return cohen_kappa_score(y_true, y_pred_cls, weights="quadratic")


def calibrate_thresholds(y_true, y_pred_reg, init_thr, iters=2):
    thr = np.array(init_thr, dtype=np.float32)
    thr.sort()
    best = qwk_for_thresholds(y_true, y_pred_reg, thr)

    for _ in range(iters):
        for j in range(4):
            lo = 0.0 if j == 0 else float(thr[j - 1] + 1e-3)
            hi = 4.5 if j == 3 else float(thr[j + 1] - 1e-3)
            if hi <= lo:
                continue

            grid = np.linspace(lo, hi, 41, dtype=np.float32)
            local_best = best
            local_t = float(thr[j])
            for t in grid:
                cand = thr.copy()
                cand[j] = t
                score = qwk_for_thresholds(y_true, y_pred_reg, cand)
                if score > local_best:
                    local_best = score
                    local_t = float(t)
            thr[j] = local_t
            best = local_best

    thr = np.clip(thr, 0.0, 4.5)
    thr.sort()
    return thr.tolist(), best


split_seeds = [SEED, SEED + 1]
cal_thrs = []
cal_scores = []

for rs in split_seeds:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=rs)
    _, va_idx = next(
        sss.split(train_df["id_code"].values, train_df["diagnosis"].values)
    )
    val_df = train_df.iloc[va_idx].copy()

    val_ds = RetinaDataset(val_df, train_img_dir, transform)
    val_loader = DataLoader(
        val_ds,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_pred_reg, val_y = predict_final_regression(net, val_loader)
    init_thr = threshold
    thr_i, qwk_i = calibrate_thresholds(val_y, val_pred_reg, init_thr, iters=2)
    cal_thrs.append(thr_i)
    cal_scores.append(float(qwk_i))
    print(
        f"[calibration split seed {rs}] QWK:",
        float(qwk_i),
        "thr:",
        [round(x, 4) for x in thr_i],
    )

cal_thr = np.mean(np.asarray(cal_thrs, dtype=np.float32), axis=0)
cal_thr = np.clip(cal_thr, 0.0, 4.5)
cal_thr.sort()
cal_thr = cal_thr.tolist()

print("Initial thresholds:", init_thr)
print("Averaged calibrated thresholds:", [round(x, 4) for x in cal_thr])
print("Calibration QWKs:", cal_scores, "mean:", float(np.mean(cal_scores)))

threshold = cal_thr




## === cell 7
class TestRetinaDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
        except Exception as e:
            print("WARNING: failed to read/transform", img_path, "error:", repr(e))
            img = torch.zeros(3, input_size * 3 // 4, input_size, dtype=torch.float32)
            img = transforms.Normalize(
                mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]
            )(img)
        return img, img_id


test_ds = TestRetinaDataset(test_ids_ordered, test_img_dir, transform)
test_loader = DataLoader(
    test_ds,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
    worker_init_fn=seed_worker,
    generator=g,
)

pred_by_id = {}
net.eval()

with torch.no_grad():
    seen = 0
    for xb, idb in test_loader:
        xb = xb.to(device, non_blocking=True)
        out = net(xb, final=True).squeeze(1)  # (B,)
        out_np = out.detach().cpu().numpy().astype(np.float32)
        out_np = np.clip(out_np, 0.0, 4.5)
        pred_cls = apply_thresholds(out_np, threshold)  # (B,) int64
        for _id, _p in zip(list(idb), pred_cls.tolist()):
            pred_by_id[str(_id)] = int(_p)
        seen += len(idb)
        if seen % 50 == 0 or seen == len(test_ids_ordered):
            print(seen, "/", len(test_ids_ordered))



## === cell 8
sub = sample_df.copy()
sub["id_code"] = sub["id_code"].astype(str)

missing = 0
diag = []
for _id in sub["id_code"].values:
    if _id in pred_by_id:
        diag.append(pred_by_id[_id])
    else:
        missing += 1
        diag.append(0)
sub["diagnosis"] = np.asarray(diag, dtype=np.int64)

if missing > 0:
    print("WARNING: missing predictions for", missing, "ids; filled with 0.")

assert (
    len(sub) == len(sample_df) and len(sub) > 0
), "Submission is empty or wrong length."
assert list(sub.columns) == ["id_code", "diagnosis"], "Submission columns mismatch."
assert sub["diagnosis"].isna().sum() == 0, "NaNs in diagnosis."
assert sub["diagnosis"].between(0, 4).all(), "Out-of-range diagnosis values."

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("diagnosis value counts:\n", sub["diagnosis"].value_counts().sort_index())
