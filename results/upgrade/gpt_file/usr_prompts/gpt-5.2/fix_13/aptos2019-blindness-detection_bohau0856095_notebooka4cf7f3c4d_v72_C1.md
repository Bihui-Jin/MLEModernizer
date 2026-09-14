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

0.9210361307099028

# 6. Current score

0.71457

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the immediate failure by removing the hard dependency on an external pretrained weights file that doesn’t exist in this environment, and instead run the same model forward pass with deterministic random initialization so the notebook completes and produces a valid `submission.csv`. I also make model loading robust to common checkpoint formats (`state_dict`, `model`, etc.) in case a weights file is present in a different path/name. Finally, I ensure the test image directory is correctly resolved (fallback between common Kaggle paths) and that the submission is aligned to `test.csv` order with the exact required columns and types.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with running the network with random weights (no valid pretrained checkpoint is being loaded), which makes predictions essentially random. The smallest legitimate step toward the 0.921 target is to (1) reliably load a real checkpoint if it exists anywhere under the provided input tree, and (2) ensure inference uses the model’s intended “final” head (the code defines a three-stage model with a `final=True` path but never uses it). I keep the architecture and thresholds unchanged, only fixing weight discovery/loading robustness and switching inference to `net(images, final=True)` when weights are available (fallback to the current regressor output if not). This preserves evaluation semantics while making the submission meaningfully predictive when the checkpoint is present, which should move the score upward toward the target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score indicates the model is still effectively untrained (either no checkpoint is found/loaded, or the loaded keys don’t actually match this model), so the smallest legitimate move toward the 0.921 target is to reliably locate and correctly load a real checkpoint from the available dataset tree. I (1) expand weight discovery to also search under the resolved dataset base (`BASE_INPUT`) and `/kaggle` recursively, (2) make state-dict key cleaning robust to common prefixes (e.g., `model.`, `net.`) so more checkpoints can load correctly, and (3) enforce deterministic inference plus a strict submission order match to `test.csv` without changing the model, transforms, thresholds, or prediction logic. These are minimal, execution-safe changes that should move the score upward if any valid pretrained weights exist in the environment; otherwise behavior remains the same (random weights → low score), but still produces a valid `submission.csv`.'
- What this solution (achieved 0.61936) has done: 'Your 0.0 score strongly suggests the model is still running with random (untrained) weights; the most direct minimal improvement is to actually train the exact same model you already defined on `train.csv` and then run inference. I keep your architecture, transforms, and regression-to-class thresholding intact, and add a small training cell that fits `ThreeStage_Model` using the existing `final=True` regressor head with an MSE loss on the 0–4 labels (matching your regression output scaling). I also add a quick internal validation kappa computation (not used for early stopping) to sanity-check that learning is happening, while ensuring the script still finishes within the time limit by training for a small fixed number of epochs. Finally, inference remains the same and writes a valid `submission.csv` aligned to `test.csv`.'
- What this solution (achieved 0.68149) has done: 'Your current gap to the target is large (0.61936 → 0.9210), so the smallest legitimate move is to better align the existing regression output to the QWK metric without changing the model or training loop. I keep your architecture, transforms, loss (MSE), fixed epoch budget, and inference path (`final=True`) intact, and add only a lightweight post-training calibration step that chooses optimal `threshold` values on your already-created validation split to maximize quadratic weighted kappa. Then inference uses the same `regress2class` function, now driven by tuned thresholds, which typically yields a meaningful QWK lift for ordinal problems. If pretrained weights are found/loaded, the code still run calibration and use those thresholds as well, helping score without altering the learned weights.'
- What this solution (achieved 0.72913) has done: 'The timeout is dominated by CPU-side image preprocessing (PIL open + trim/crop) repeated across train/val/test and by under-utilized dataloader workers. To fix this without changing the model/training logic, I cache transformed tensors to disk on first use (so subsequent epochs/val/test are fast), enable persistent DataLoader workers with a higher worker count, and speed up PIL decoding by using `PIL.ImageFile.LOAD_TRUNCATED_IMAGES` and a more efficient loader pattern. I also vectorize the regression-to-class and regression-to-probability utilities to remove Python loops while keeping identical semantics. These changes preserve the exact transforms, model, loss, and training loop behavior; they only eliminate redundant work and improve input pipeline throughput.'
- What this solution (achieved 0.75263) has done: 'Your current score (0.72913) is well below the target (0.9210), so we should cautiously improve model generalization without changing the architecture or the basic training/inference semantics. The smallest high-impact change here is to fix a caching bug that currently prevents effective tensor caching (it builds cache paths using the image id string but the function parameter is named `idx` and was treated like an integer index), causing repeated expensive preprocessing and reducing the effective training signal under the time budget. With caching actually working, we can safely increase the fixed training epoch count (still no early stopping) while staying within the 600s constraint, which should move QWK upward toward the target. I keep your model, transforms, loss (MSE), regression head usage (`final=True`), and threshold-tuning logic intact.'
- What this solution (achieved 0.68686) has done: 'Your current score (0.75263) is far below the target (0.9210), so we should make a minimal, metric-aligned improvement that doesn’t change the model or training objective. The biggest low-risk lift for QWK in this setup is to use **test-time augmentation (TTA)** at inference (horizontal flip) and average the regressor outputs before applying the already-thresholded discretization; this keeps the exact same model and thresholds semantics, just reduces prediction noise. I also make sure inference is strictly deterministic and keep the submission aligned to `test.csv` order with the required columns. No architecture, loss, transforms, or training loop logic is changed—only inference ensembling with a symmetric augmentation.'
- What this solution (achieved 0.6877) has done: 'Your current score (0.68686) is far below the target (0.9210), so we should make a small, metric-aligned improvement without changing the model, transforms, loss, or training loop. The biggest low-risk gain for QWK in this ordinal regression setup is to replace the simple coordinate-descent threshold tuning with a more reliable optimizer (still only tuning the 4 cutpoints on the fixed validation split) using `scipy.optimize.minimize`, which often yields a sizable kappa lift. I keep your existing tuning as a fallback if SciPy isn’t available, and I keep inference/TTA exactly the same—only the thresholds are better fit to QWK. This preserves the core logic and evaluation semantics while moving the score upward toward the target.'
- What this solution (achieved 0.71457) has done: 'Your current score (0.6877) is far below the target (0.9210), so we should make a small, metric-aligned improvement without changing the model, loss, transforms, or training loop. The biggest low-risk gain for QWK in an ordinal regression setup is to tune thresholds more thoroughly and robustly on the existing validation split, since your final step is discretization and QWK is computed on discrete classes. I keep your existing SciPy optimizer if available, but add a deterministic multi-start + post-refinement (fine grid) to avoid bad local optima and better match QWK. This only changes the 4 cutpoints used by `regress2class` (same evaluation semantics), and keeps inference/TTA and submission formatting identical.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import time
import math
import hashlib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops, ImageFile

from sklearn.metrics import cohen_kappa_score
import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ImageFile.LOAD_TRUNCATED_IMAGES = True




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    out = out.view(-1)
    thr = torch.tensor(threshold, device=out.device, dtype=out.dtype).view(1, -1)
    prediction = (out.view(-1, 1) >= thr).sum(dim=1).to(torch.int64)
    return prediction.cpu()


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    out = out.view(-1)
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)

    lt4 = out < 4.0
    if lt4.any():
        v = out[lt4]
        l1 = torch.floor(v).to(torch.long).clamp(0, 4)
        l2 = torch.ceil(v).to(torch.long).clamp(0, 4)
        w1 = 1.0 - (v - l1.to(v.dtype))
        w2 = 1.0 - (l2.to(v.dtype) - v)
        idx = torch.nonzero(lt4, as_tuple=False).squeeze(1)
        pred_prob[idx, l1] = w1
        pred_prob[idx, l2] = w2

    ge4 = ~lt4
    if ge4.any():
        pred_prob[ge4, 4] = 1.0

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
BASE_INPUT_CANDIDATES = [
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "../input",
    "/kaggle/input",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data",
]


def resolve_base_input():
    for base in BASE_INPUT_CANDIDATES:
        if os.path.isfile(os.path.join(base, "test.csv")) and os.path.isdir(
            os.path.join(base, "test_images")
        ):
            return base
    for base in ["../input", "/kaggle/input", "/kaggle/data"]:
        if os.path.isdir(base):
            hit = glob.glob(
                os.path.join(base, "**", "aptos2019-blindness-detection"),
                recursive=True,
            )
            for h in hit:
                if os.path.isfile(os.path.join(h, "test.csv")) and os.path.isdir(
                    os.path.join(h, "test_images")
                ):
                    return h
    raise FileNotFoundError(
        "Could not resolve dataset base path containing test.csv and test_images/."
    )


BASE_INPUT = resolve_base_input()
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TEST_CSV = os.path.join(BASE_INPUT, "test.csv")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

test_ids = test_df["id_code"].astype(str).values

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

train_transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        photometric_distort(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

CACHE_DIR = "/kaggle/working/aptos_cache_tensors"
os.makedirs(CACHE_DIR, exist_ok=True)


def _transform_fingerprint(transform_obj, input_size_):
    s = f"{repr(transform_obj)}|input_size={input_size_}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


VAL_FP = _transform_fingerprint(transform, input_size)
TRN_FP = _transform_fingerprint(train_transform, input_size)




## === cell 5
class TrainDataset(Dataset):
    def __init__(self, df, img_dir, transform, cache_tag=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.cache_tag = cache_tag  # e.g., "val_<fp>" or "train_<fp>"

    def __len__(self):
        return len(self.df)

    def _cache_path(self, id_code):
        if self.cache_tag is None:
            return None
        return os.path.join(CACHE_DIR, f"{self.cache_tag}__{str(id_code)}.pt")

    def __getitem__(self, i):
        row = self.df.iloc[i]
        id_code = str(row["id_code"])
        y = float(row["diagnosis"])

        cpath = self._cache_path(id_code)
        if cpath is not None and os.path.isfile(cpath):
            img = torch.load(cpath, map_location="cpu")
            return img, torch.tensor([y], dtype=torch.float32), id_code

        image_name = os.path.join(self.img_dir, f"{id_code}.png")
        with Image.open(image_name) as im:
            img = im.convert("RGB")
            img = self.transform(img)

        if cpath is not None:
            tmp = cpath + ".tmp"
            torch.save(img, tmp)
            os.replace(tmp, cpath)

        return img, torch.tensor([y], dtype=torch.float32), id_code


def find_weight_file(base_input):
    explicit = [
        "../input/weights/B4_3stage_5epoch_finetune512.pkl",
        "../input/B4_3stage_5epoch_finetune512.pkl",
        "/kaggle/input/weights/B4_3stage_5epoch_finetune512.pkl",
        "/kaggle/input/B4_3stage_5epoch_finetune512.pkl",
        "/kaggle/data/weights/B4_3stage_5epoch_finetune512.pkl",
        "/kaggle/data/B4_3stage_5epoch_finetune512.pkl",
        os.path.join(base_input, "B4_3stage_5epoch_finetune512.pkl"),
        os.path.join(base_input, "weights", "B4_3stage_5epoch_finetune512.pkl"),
    ]
    for c in explicit:
        if os.path.isfile(c):
            return c

    search_roots = [
        "../input",
        "/kaggle/input",
        "/kaggle/data",
        base_input,
        "/kaggle",
    ]
    patterns_rel = [
        "**/B4_3stage_5epoch_finetune512.pkl",
        "**/*3stage*finetune*512*.pkl",
        "**/*3stage*.pkl",
        "**/*3stage*.pth",
        "**/*3stage*.pt",
        "**/*efficientnet*b4*.pth",
        "**/*efficientnet*b4*.pt",
        "**/*b4*3stage*.pth",
        "**/*b4*3stage*.pt",
    ]

    hits_all = []
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for pr in patterns_rel:
            hits_all.extend(glob.glob(os.path.join(root, pr), recursive=True))

    hits_all = sorted(set(hits_all), key=lambda p: (len(p.split(os.sep)), len(p), p))
    return hits_all[0] if len(hits_all) > 0 else None


def extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def clean_state_dict_keys(state):
    if not isinstance(state, dict) or len(state) == 0:
        return state

    prefixes = [
        "module.",
        "model.",
        "net.",
        "backbone.",
    ]
    model_keys = set(ThreeStage_Model().state_dict().keys())

    def strip_once(sd, pref):
        return {k[len(pref) :] if k.startswith(pref) else k: v for k, v in sd.items()}

    best_state = state
    best_overlap = len(set(state.keys()) & model_keys)

    for pref in prefixes:
        stripped = strip_once(state, pref)
        overlap = len(set(stripped.keys()) & model_keys)
        if overlap > best_overlap:
            best_overlap = overlap
            best_state = stripped

    improved = True
    while improved:
        improved = False
        for pref in prefixes:
            stripped = strip_once(best_state, pref)
            overlap = len(set(stripped.keys()) & model_keys)
            if overlap > best_overlap:
                best_overlap = overlap
                best_state = stripped
                improved = True

    return best_state


def apply_thresholds_np(preds, thr):
    thr = np.asarray(list(thr), dtype=np.float32).reshape(1, -1)
    preds = np.asarray(preds, dtype=np.float32).reshape(-1, 1)
    return (preds >= thr).sum(axis=1).astype(np.int64)


def tune_thresholds_for_qwk(y_true_int, y_pred_float, init_thr, iters=2):
    y_true_int = np.asarray(y_true_int, dtype=np.int64).reshape(-1)
    y_pred_float = np.asarray(y_pred_float, dtype=np.float32).reshape(-1)

    thr = np.array(init_thr, dtype=np.float32)
    thr = np.clip(thr, 0.0, 4.5)
    thr.sort()

    def score(thr_):
        y_hat = apply_thresholds_np(y_pred_float, thr_)
        return cohen_kappa_score(y_true_int, y_hat, weights="quadratic")

    best = score(thr)

    for _ in range(int(iters)):
        for j in range(4):
            lo = 0.0 if j == 0 else float(thr[j - 1] + 1e-3)
            hi = 4.5 if j == 3 else float(thr[j + 1] - 1e-3)
            if hi <= lo:
                continue

            grid = np.linspace(lo, hi, 41, dtype=np.float32)
            local_best = best
            local_thr = float(thr[j])

            for v in grid:
                cand = thr.copy()
                cand[j] = v
                cand.sort()
                s = score(cand)
                if s > local_best:
                    local_best = s
                    local_thr = float(v)

            thr[j] = local_thr
            thr.sort()
            best = local_best

    return thr.tolist(), float(best)


def tune_thresholds_for_qwk_scipy(y_true_int, y_pred_float, init_thr):
    try:
        from scipy.optimize import minimize
    except Exception:
        return None, None

    y_true_int = np.asarray(y_true_int, dtype=np.int64).reshape(-1)
    y_pred_float = np.asarray(y_pred_float, dtype=np.float32).reshape(-1)

    init_thr = np.asarray(init_thr, dtype=np.float64).reshape(-1)
    init_thr = np.clip(init_thr, 0.0, 4.5)
    init_thr.sort()

    def pack(u):
        inc = np.log1p(np.exp(u))  # softplus
        t = np.cumsum(inc)
        t = t / (t[-1] + 1e-12) * 4.5
        eps = 1e-3
        t = np.clip(t, eps, 4.5 - eps)
        t.sort()
        return t

    def objective(u):
        t = pack(u)
        y_hat = apply_thresholds_np(y_pred_float, t)
        k = cohen_kappa_score(y_true_int, y_hat, weights="quadratic")
        return -float(k)

    diffs = np.diff(np.concatenate([[0.0], init_thr]))
    diffs = np.clip(diffs, 1e-3, None)
    u0 = np.log(np.expm1(diffs))  # inverse softplus approx
    if not np.all(np.isfinite(u0)):
        u0 = np.zeros(4, dtype=np.float64)

    res = minimize(
        objective,
        u0,
        method="Nelder-Mead",
        options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-4},
    )
    thr = pack(res.x)
    kappa = -float(res.fun)
    return thr.tolist(), float(kappa)


def tune_thresholds_for_qwk_scipy_multistart(
    y_true_int, y_pred_float, init_thr, seeds=(0, 1, 2, 3, 4)
):
    try:
        from scipy.optimize import minimize
    except Exception:
        return None, None

    y_true_int = np.asarray(y_true_int, dtype=np.int64).reshape(-1)
    y_pred_float = np.asarray(y_pred_float, dtype=np.float32).reshape(-1)

    init_thr = np.asarray(init_thr, dtype=np.float64).reshape(-1)
    init_thr = np.clip(init_thr, 0.0, 4.5)
    init_thr.sort()

    def pack(u):
        inc = np.log1p(np.exp(u))  # softplus => positive increments
        t = np.cumsum(inc)
        t = t / (t[-1] + 1e-12) * 4.5
        eps = 1e-3
        t = np.clip(t, eps, 4.5 - eps)
        t.sort()
        return t

    def objective(u):
        t = pack(u)
        y_hat = apply_thresholds_np(y_pred_float, t)
        k = cohen_kappa_score(y_true_int, y_hat, weights="quadratic")
        return -float(k)

    diffs = np.diff(np.concatenate([[0.0], init_thr]))
    diffs = np.clip(diffs, 1e-3, None)
    u_base = np.log(np.expm1(diffs))
    if not np.all(np.isfinite(u_base)):
        u_base = np.zeros(4, dtype=np.float64)

    best_u = u_base.copy()
    best_fun = objective(best_u)

    for s in seeds:
        rs = np.random.RandomState(int(s))
        u0 = u_base + rs.normal(
            0.0, 0.5, size=u_base.shape
        )  # small deterministic jitter
        res = minimize(
            objective,
            u0,
            method="Nelder-Mead",
            options={"maxiter": 350, "xatol": 1e-4, "fatol": 1e-4},
        )
        if float(res.fun) < float(best_fun):
            best_fun = float(res.fun)
            best_u = np.asarray(res.x, dtype=np.float64)

    thr = pack(best_u)
    kappa = -float(best_fun)
    return thr.tolist(), float(kappa)


weight_path = find_weight_file(BASE_INPUT)

net = ThreeStage_Model().to(device)

loaded = False
load_error = None
if weight_path is not None:
    try:
        ckpt = torch.load(weight_path, map_location="cpu")
        state = extract_state_dict(ckpt)
        state = clean_state_dict_keys(state)
        try:
            net.load_state_dict(state, strict=True)
            loaded = True
        except RuntimeError as e:
            load_error = str(e)
            net.load_state_dict(state, strict=False)
            loaded = True
    except Exception as e:
        load_error = repr(e)
        loaded = False

print(f"Using device: {device}")
print(f"Dataset base: {BASE_INPUT}")
print(f"Num train images: {len(train_df)}")
print(f"Num test images: {len(test_df)}")
print(f"Weights found: {weight_path}")
print(f"Weights loaded: {loaded}")
if load_error is not None:
    print(f"Weight load note: {load_error[:400]}")

perm = np.random.RandomState(42).permutation(len(train_df))
val_n = max(1, int(0.15 * len(train_df)))
val_idx = perm[:val_n]
trn_idx = perm[val_n:]

trn_ds = TrainDataset(
    train_df.iloc[trn_idx], TRAIN_IMG_DIR, train_transform, cache_tag=f"train_{TRN_FP}"
)
val_ds = TrainDataset(
    train_df.iloc[val_idx], TRAIN_IMG_DIR, transform, cache_tag=f"val_{VAL_FP}"
)

_nw = min(8, max(2, (os.cpu_count() or 4) // 2))
_common_dl_kwargs = dict(
    num_workers=_nw,
    pin_memory=(device != "cpu"),
    persistent_workers=(_nw > 0),
    prefetch_factor=4 if _nw > 0 else None,
)

trn_dl = DataLoader(
    trn_ds,
    batch_size=8,
    shuffle=True,
    **{k: v for k, v in _common_dl_kwargs.items() if v is not None},
)
val_dl = DataLoader(
    val_ds,
    batch_size=8,
    shuffle=False,
    **{k: v for k, v in _common_dl_kwargs.items() if v is not None},
)

if not loaded:
    net.train()
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    epochs = 8  # fixed epoch budget (no early stopping)
    start_t = time.time()
    for ep in range(epochs):
        ep_loss = 0.0
        n_seen = 0
        for x, y, _ in trn_dl:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = net(x, final=True)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
            bs = x.size(0)
            ep_loss += float(loss.detach().cpu()) * bs
            n_seen += bs

            if time.time() - start_t > 520:
                break

        net.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for x, y, _ in val_dl:
                x = x.to(device, non_blocking=True)
                pred = net(x, final=True).squeeze(1).detach()
                yhat = regress2class(pred).numpy().astype(int)
                ytrue = y.squeeze(1).numpy().astype(int)
                y_true.append(ytrue)
                y_pred.append(yhat)
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")

        net.train()
        print(
            f"epoch {ep+1}/{epochs} loss={ep_loss/max(1,n_seen):.4f} val_qwk={kappa:.4f}"
        )

        if time.time() - start_t > 520:
            print("Time budget reached; proceeding to inference.")
            break

    net.eval()
else:
    net.eval()

net.eval()
val_preds_float = []
val_true_int = []
with torch.no_grad():
    for x, y, _ in val_dl:
        x = x.to(device, non_blocking=True)
        pred = net(x, final=True).squeeze(1).detach().cpu().numpy()
        val_preds_float.append(pred)
        val_true_int.append(y.squeeze(1).numpy().astype(int))
val_preds_float = np.concatenate(val_preds_float)
val_true_int = np.concatenate(val_true_int)

old_thr = list(threshold)
old_kappa = cohen_kappa_score(
    val_true_int, apply_thresholds_np(val_preds_float, old_thr), weights="quadratic"
)

sc_thr, sc_kappa = tune_thresholds_for_qwk_scipy_multistart(
    val_true_int, val_preds_float, old_thr, seeds=(0, 1, 2, 3, 4, 5, 6, 7)
)

if sc_thr is None:
    sc_thr, sc_kappa = tune_thresholds_for_qwk_scipy(
        val_true_int, val_preds_float, old_thr
    )

if sc_thr is not None:
    cand_thr, cand_kappa = sc_thr, sc_kappa
    print(
        f"SciPy threshold opt (multi-start if available): thr={cand_thr} val_qwk={cand_kappa:.4f}"
    )
else:
    cand_thr, cand_kappa = tune_thresholds_for_qwk(
        val_true_int, val_preds_float, old_thr, iters=2
    )

cand_thr, cand_kappa = tune_thresholds_for_qwk(
    val_true_int, val_preds_float, cand_thr, iters=1
)

if cand_kappa >= old_kappa:
    threshold = cand_thr
    print(
        f"Threshold tuned: old_thr={old_thr} old_qwk={old_kappa:.4f} -> new_thr={threshold} new_qwk={cand_kappa:.4f}"
    )
else:
    print(
        f"Threshold kept: thr={old_thr} val_qwk={old_kappa:.4f} (tuned was {cand_kappa:.4f})"
    )




## === cell 6
class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform, cache_tag=None):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform
        self.cache_tag = cache_tag

    def __len__(self):
        return len(self.ids)

    def _cache_path(self, id_code):
        if self.cache_tag is None:
            return None
        return os.path.join(CACHE_DIR, f"{self.cache_tag}__{str(id_code)}.pt")

    def __getitem__(self, i):
        id_code = str(self.ids[i])
        cpath = self._cache_path(id_code)
        if cpath is not None and os.path.isfile(cpath):
            img = torch.load(cpath, map_location="cpu")
            return img, id_code

        image_name = os.path.join(self.img_dir, f"{id_code}.png")
        with Image.open(image_name) as im:
            img = im.convert("RGB")
            img = self.transform(img)

        if cpath is not None:
            tmp = cpath + ".tmp"
            torch.save(img, tmp)
            os.replace(tmp, cpath)

        return img, id_code


ds = TestDataset(test_ids, TEST_IMG_DIR, transform, cache_tag=f"test_{VAL_FP}")
dl = DataLoader(
    ds,
    batch_size=8,
    shuffle=False,
    **{k: v for k, v in _common_dl_kwargs.items() if v is not None},
)

submission_rows = []
with torch.no_grad():
    for images, ids in dl:
        images = images.to(device, non_blocking=True)

        out1 = net(images, final=True).squeeze(1)

        images_flip = torch.flip(images, dims=[3])  # horizontal flip (W dimension)
        out2 = net(images_flip, final=True).squeeze(1)

        out = 0.5 * (out1 + out2)

        preds = regress2class(out).numpy().astype(int)
        for idx, p in zip(ids, preds):
            submission_rows.append([str(idx), int(p)])

submission = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])




## === cell 7
if submission.shape[0] == 0:
    raise RuntimeError(
        "Submission DataFrame is empty; inference did not produce any rows."
    )

submission["id_code"] = submission["id_code"].astype(str)
submission["diagnosis"] = submission["diagnosis"].astype(int)

submission = test_df[["id_code"]].merge(submission, on="id_code", how="left")
if submission["diagnosis"].isna().any():
    submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print(f"Wrote submission.csv with {len(submission)} rows.")
print(f"Final thresholds used: {threshold}")
