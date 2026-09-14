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

0.9119683744704712

# 6. Current score

-0.00694

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11386) has done: 'I make the checkpoint loading robust so the notebook runs even when the external weights aren’t attached by falling back to a timm pretrained backbone (same architecture) and warning instead of crashing. I also fix the CUDA/CPU dtype mismatch by ensuring the loaded state_dict tensors are moved to the same device as the model and inputs. Finally, I keep the same inference mapping logic but ensure the loop always produces a complete, correctly aligned `submission.csv` with the required columns and order.'
- What this solution (achieved -0.07134) has done: 'Your current score is far below the target, and the biggest likely cause is that inference is not using the model’s intended “final” head (you’re currently discarding the classifier+ordinal outputs and only thresholding the intermediate regressor). I make the smallest change to use `net(img, final=True)` and then map the resulting single regression output to classes with the same `regress2class` thresholds (preserving the submission semantics). I also keep a backward-compatible fallback: if `final=True` isn’t available for some reason, it use your existing path. This should move QWK substantially upward toward your target without changing architecture, training, or introducing approximations.'
- What this solution (achieved -0.01658) has done: 'Your current score is far below the target, so we should improve inference quality without changing the model architecture or training. The biggest low-risk gain is to make test-time augmentation deterministic and enable simple horizontal-flip TTA, then average the model’s *final* regression output (keeping your same `regress2class` thresholds). This preserves your evaluation semantics (still regression→threshold→class) while usually improving QWK on this competition. I also speed up and stabilize inference by batching with a `Dataset/DataLoader` (same preprocessing) to ensure we finish within the time limit and always write a correctly aligned `submission.csv`.'
- What this solution (achieved 0.17048) has done: 'Your current QWK is far below the target, so the most likely issue is prediction calibration/thresholding rather than the model itself. I keep your same model and inference (including flip TTA), but replace the hardcoded thresholds with thresholds optimized on a small validation split from `train.csv` using the same `final=True` regression output you submit. This preserves your evaluation semantics (regression → threshold → class) while making the mapping better aligned to QWK, which should move the score substantially upward toward the target. I also ensure the optimized thresholds are used in `regress2class` during test inference and still always produce a valid `submission.csv`.'
- What this solution (achieved 0.18981) has done: 'Your current score is far below the target, so we should improve inference calibration without changing your model or training loop. The largest likely issue is that threshold optimization is being done on a regression output distribution that doesn’t match the test-time distribution because your validation split is not stratified tightly enough and can be unstable at this dataset size; we keep the same threshold-search logic but make the split more stable and representative. Concretely, we use a fixed stratified K-fold (single fold) instead of a single shuffle split, and we also optimize thresholds on out-of-fold-like predictions from the full train (one fold as “val”) while keeping the exact same inference path (`final=True` with flip TTA and regression→threshold→class). Finally, we ensure thresholds are monotonic and constrained within [0, 4.5] to avoid degenerate mappings that can tank QWK.'
- What this solution (achieved 0.18981) has done: 'Your current score is far below the target, so the most likely “minimal-change” lever is the regression→class threshold calibration, not the model itself. I keep your exact model/inference path (final=True + flip TTA + same preprocessing) but make threshold optimization more representative by using out-of-fold predictions across all 5 folds (instead of calibrating on only fold 0), then optimize a single global threshold set on those OOF predictions. This preserves evaluation semantics while reducing overfitting/instability in the threshold fit and should move QWK upward toward your target. I also reuse the existing prediction helper for test inference to avoid any subtle mismatch and still write a correctly aligned `submission.csv`.'
- What this solution (achieved 0.0254) has done: 'Your current score is far below the target, so the smallest likely win is fixing a train/test preprocessing mismatch that can severely break calibration: the checkpoint name suggests CLAHE was used, but inference currently never applies CLAHE. I add an optional CLAHE step (OpenCV) inside the existing transform pipeline (keeping resize/normalize and the same model/inference logic), controlled by a flag defaulting to ON to better match the weights. I also make the dataloader deterministic and slightly more stable by setting worker seeds, which can reduce threshold-fitting noise without changing your core approach. Everything else (model, `final=True` regression, flip TTA, OOF threshold optimization, CSV schema/paths) stays the same.'
- What this solution (achieved 0.01578) has done: 'Your current score (0.0254) is extremely far below the target (0.912), so the most likely issue is that the model outputs are not being interpreted on the correct scale (e.g., checkpoint expects logits but inference applies a sigmoid*4.5 again, or vice versa), which can destroy thresholding and QWK. I keep your exact model/weights/TTAs and OOF threshold optimization, but add an automatic, fold-0-only “output calibration mode” selector that tries a couple of *equivalent* post-processing variants (identity vs sigmoid*4.5) on the already-produced raw `final_regressor` outputs and picks the one that yields the best QWK on that fold. Then we run the same 5-fold OOF prediction + threshold optimization using the chosen mode, and apply it consistently at test time, which should move the score sharply upward toward your target without changing architecture or training.'
- What this solution (achieved 0.0) has done: 'I make the checkpoint handling non-fatal so the notebook always runs end-to-end and writes a valid `submission.csv`, even when the external weights aren’t attached. I also fix the CUDA/CPU dtype mismatch by ensuring the loaded `state_dict` is moved onto the model’s device (or moving the model after loading), which is what caused the `FloatTensor` vs `cuda.FloatTensor` crash. Finally, I ensure `OUTPUT_MODE` is always defined (with a safe default) so inference and CSV writing can’t fail due to earlier calibration steps being skipped. These changes are execution/stability fixes; without the intended checkpoint, the score remain far from the target but the pipeline be valid.'
- What this solution (achieved -0.00694) has done: 'I make checkpoint loading non-fatal so the notebook always runs end-to-end and writes `submission.csv`, while still using your intended weights when they are available. I fix the device/dtype mismatch that caused CUDA inputs to hit CPU weights by ensuring the model is moved to `device` after loading and not trying to move individual state_dict tensors. Finally, if the checkpoint is missing, I fall back to the same timm backbone but with `pretrained=True` (score-improving vs random) while keeping your exact inference/calibration logic and submission schema unchanged.'
- What this solution (achieved -0.00694) has done: 'Your score is extremely far below the target, and given your current fallback message it’s very likely the intended checkpoint is not being loaded at all (so the model is essentially untrained), which tank QWK regardless of threshold calibration. I make the smallest change that materially improves this: expand checkpoint discovery to include the common Kaggle dataset location (`/kaggle/input/<dataset_name>/...`) and also accept common filename variations, then load with a slightly more robust key-stripping routine. If we still can’t find weights, we keep your existing pretrained-backbone fallback (so it still runs end-to-end and writes `submission.csv`). Everything else (model, transforms including CLAHE, flip TTA, OOF threshold optimization, and submission formatting) stays the same.'
- What this solution (achieved -0.00694) has done: 'Your current score is far below the target, and the warnings in your history strongly suggest the intended checkpoint still isn’t being found/loaded, leaving you with essentially untrained heads (pretrained backbone alone won’t help enough), which tanks QWK. I make the smallest change that improves checkpoint discovery in a Kaggle-safe way: search also under `/kaggle/data/**` (your environment shows weights may live there) and accept more filename variants, and I add a strict check that prevents silently using random heads when a checkpoint exists but is incompatible. If a checkpoint is found, we load it more robustly (including handling `ema_state_dict` and stripping common prefixes) while keeping your model/inference/threshold-calibration logic identical. The rest of the pipeline (CLAHE, flip TTA, OOF threshold optimization, and submission formatting) remains unchanged.'

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
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT

from PIL import Image, ImageChops
import cv2

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)
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


class apply_CLAHE(object):
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clipLimit = float(clipLimit)
        self.tileGridSize = tuple(tileGridSize)

    def __call__(self, image):
        img = np.array(image)
        if img.ndim != 3 or img.shape[2] != 3:
            return image
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clipLimit, tileGridSize=self.tileGridSize
        )
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        bgr2 = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        rgb2 = cv2.cvtColor(bgr2, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb2)




## === cell 4
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


DATA_DIR = _first_existing(
    [
        "/kaggle/input/aptos2019-blindness-detection",
        "../input/aptos2019-blindness-detection",
    ]
)
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find aptos2019-blindness-detection dataset folder under /kaggle/input or ../input."
    )

TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].values

input_size = 380

USE_CLAHE = True

transform_list = [
    trim(),
    cropTo4_3(),
]
if USE_CLAHE:
    transform_list.append(apply_CLAHE(clipLimit=2.0, tileGridSize=(8, 8)))

transform_list += [
    transforms.Resize((input_size * 3 // 4, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
]
transform = transforms.Compose(transform_list)

net = ThreeStage_Model()

candidate_paths = [
    "../input/weights/B4_3stage_38epoch_CLAHE.pkl",
    "../input/weights/B4_3stage_38epoch_CLAHE.pth",
    "/kaggle/input/weights/B4_3stage_38epoch_CLAHE.pkl",
    "/kaggle/input/weights/B4_3stage_38epoch_CLAHE.pth",
    os.path.join(DATA_DIR, "B4_3stage_38epoch_CLAHE.pkl"),
    os.path.join(DATA_DIR, "B4_3stage_38epoch_CLAHE.pth"),
]

search_globs = [
    "../input/**/B4_3stage_38epoch_CLAHE.pkl",
    "../input/**/B4_3stage_38epoch_CLAHE.pth",
    "/kaggle/input/**/B4_3stage_38epoch_CLAHE.pkl",
    "/kaggle/input/**/B4_3stage_38epoch_CLAHE.pth",
    "../input/**/*B4*3stage*CLAHE*.pth",
    "../input/**/*B4*3stage*CLAHE*.pkl",
    "/kaggle/input/**/*B4*3stage*CLAHE*.pth",
    "/kaggle/input/**/*B4*3stage*CLAHE*.pkl",
    "/kaggle/data/**/B4_3stage_38epoch_CLAHE.pkl",
    "/kaggle/data/**/B4_3stage_38epoch_CLAHE.pth",
    "/kaggle/data/**/*B4*3stage*CLAHE*.pth",
    "/kaggle/data/**/*B4*3stage*CLAHE*.pkl",
    "../input/**/*B4*3stage*38*CLAHE*.pt",
    "/kaggle/input/**/*B4*3stage*38*CLAHE*.pt",
    "/kaggle/data/**/*B4*3stage*38*CLAHE*.pt",
]
for g_glob in search_globs:
    candidate_paths.extend(glob.glob(g_glob, recursive=True))

seen = set()
candidate_paths = [p for p in candidate_paths if not (p in seen or seen.add(p))]

ckpt_path = next((p for p in candidate_paths if os.path.exists(p)), None)

ckpt_loaded = False
if ckpt_path is None:
    print(
        "WARNING: Could not find checkpoint matching 'B4_3stage_38epoch_CLAHE'. "
        "Falling back to timm pretrained backbone weights (architecture unchanged)."
    )
    net.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
    net.backbone.global_pool = GeM(flatten=True)
else:
    state = torch.load(ckpt_path, map_location="cpu")

    if (
        isinstance(state, dict)
        and "ema_state_dict" in state
        and isinstance(state["ema_state_dict"], dict)
    ):
        state_dict = state["ema_state_dict"]
    elif (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state_dict = state["state_dict"]
    elif (
        isinstance(state, dict)
        and "model" in state
        and isinstance(state["model"], dict)
    ):
        state_dict = state["model"]
    elif isinstance(state, dict) and "net" in state and isinstance(state["net"], dict):
        state_dict = state["net"]
    else:
        state_dict = state

    if isinstance(state_dict, dict):

        def _strip_prefix(k):
            for pref in ("module.", "model.", "net."):
                if k.startswith(pref):
                    return k[len(pref) :]
            return k

        state_dict = {_strip_prefix(k): v for k, v in state_dict.items()}

    missing, unexpected = net.load_state_dict(state_dict, strict=False)
    ckpt_loaded = True

    if missing:
        print(
            "WARNING: Missing keys when loading checkpoint (showing up to 20):",
            missing[:20],
        )
    if unexpected:
        print(
            "WARNING: Unexpected keys when loading checkpoint (showing up to 20):",
            unexpected[:20],
        )

    head_keys = [
        "classifier.1.weight",
        "regressor.1.weight",
        "ordinal.1.weight",
        "final_regressor.1.weight",
    ]
    head_missing = [k for k in head_keys if k in missing]
    if len(head_missing) >= 3:
        print(
            "WARNING: Checkpoint found but appears incompatible (most head weights missing). "
            "This will likely score very poorly. ckpt_path=",
            ckpt_path,
        )

net = net.to(device)
net.eval()

print("USE_CLAHE:", USE_CLAHE)
print("Checkpoint loaded:", ckpt_loaded)
print("Checkpoint path:", ckpt_path)




## === cell 5
class ImageIdDataset(Dataset):
    def __init__(self, df, img_dir, transform, has_label=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        idx = str(row["id_code"])
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        img = self.transform(img)
        if self.has_label:
            y = int(row["diagnosis"])
            return idx, img, y
        return idx, img


def _apply_thresholds_np(preds_float, thr):
    preds_float = np.asarray(preds_float, dtype=np.float32)
    thr = np.asarray(thr, dtype=np.float32)
    return (preds_float[:, None] >= thr[None, :]).sum(axis=1).astype(np.int64)


def _sanitize_thresholds(thr, lo=0.0, hi=4.5):
    thr = np.asarray(thr, dtype=np.float32).copy()
    thr = np.clip(thr, lo, hi)
    thr.sort()
    for k in range(1, 4):
        if thr[k] <= thr[k - 1] + 1e-3:
            thr[k] = thr[k - 1] + 1e-3
    thr = np.clip(thr, lo, hi)
    return thr


def _optimize_thresholds_for_qwk(y_true, y_pred_float, init_thr):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred_float = np.asarray(y_pred_float, dtype=np.float32)

    thr = _sanitize_thresholds(init_thr)

    def score(thr_):
        y_hat = _apply_thresholds_np(y_pred_float, thr_)
        return cohen_kappa_score(y_true, y_hat, weights="quadratic")

    best = score(thr)

    step = 0.05
    span = 0.75
    grid = np.arange(-span, span + 1e-9, step, dtype=np.float32)

    for _ in range(3):
        for i in range(4):
            base = float(thr[i])
            best_i = base
            best_s = best
            for d in grid:
                cand = thr.copy()
                cand[i] = base + float(d)
                cand = _sanitize_thresholds(cand)
                s = score(cand)
                if s > best_s:
                    best_s = s
                    best_i = base + float(d)
            thr[i] = best_i
            thr = _sanitize_thresholds(thr)
            best = score(thr)

    return thr.tolist(), float(best)


def _postprocess_final_output(y, mode: str, already_scaled_0_4p5: bool):
    if already_scaled_0_4p5:
        return y
    if mode == "as_is":
        return y
    if mode == "sigmoid4p5":
        return torch.sigmoid(y) * 4.5
    raise ValueError(f"Unknown mode: {mode}")


def _predict_regression_with_flip_tta(model, loader, output_mode: str = "as_is"):
    ids_all = []
    outs_all = []
    for batch in loader:
        if len(batch) == 3:
            batch_ids, batch_imgs, _ = batch
        else:
            batch_ids, batch_imgs = batch
        batch_imgs = batch_imgs.to(device, non_blocking=True)

        with torch.no_grad():
            try:
                out1 = model(batch_imgs, final=True).squeeze(1)
                out2 = model(torch.flip(batch_imgs, dims=[3]), final=True).squeeze(1)
                out1 = _postprocess_final_output(
                    out1, output_mode, already_scaled_0_4p5=True
                )
                out2 = _postprocess_final_output(
                    out2, output_mode, already_scaled_0_4p5=True
                )
                final_out = 0.5 * (out1 + out2)
            except TypeError:
                _, r1, _ = model(batch_imgs)
                _, r2, _ = model(torch.flip(batch_imgs, dims=[3]))
                final_out = 0.5 * (r1.squeeze(1) + r2.squeeze(1))

        ids_all.extend([str(x) for x in batch_ids])
        outs_all.append(final_out.detach().float().cpu())
    outs_all = torch.cat(outs_all, dim=0)
    return ids_all, outs_all


OUTPUT_MODE = "as_is"

train_df = pd.read_csv(TRAIN_CSV)
oof_true = train_df["diagnosis"].astype(int).values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

try:
    fold0_tr_idx, fold0_va_idx = next(skf.split(train_df["id_code"].values, oof_true))
    fold0_val_df = train_df.iloc[fold0_va_idx].reset_index(drop=True)
    fold0_val_ds = ImageIdDataset(
        fold0_val_df, TRAIN_IMG_DIR, transform, has_label=True
    )
    fold0_val_loader = DataLoader(
        fold0_val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=g,
    )

    modes_to_try = ["as_is"]
    mode_scores = {}
    for m in modes_to_try:
        _, vpred = _predict_regression_with_flip_tta(
            net, fold0_val_loader, output_mode=m
        )
        qwk_m = cohen_kappa_score(
            fold0_val_df["diagnosis"].astype(int).values,
            _apply_thresholds_np(vpred.numpy(), threshold),
            weights="quadratic",
        )
        mode_scores[m] = float(qwk_m)
        print(f"[Mode selection] fold0 QWK@initial_thr mode={m}: {qwk_m:.5f}")

    OUTPUT_MODE = max(mode_scores, key=mode_scores.get)
    print("Selected OUTPUT_MODE:", OUTPUT_MODE, "| scores:", mode_scores)

    oof_pred = np.zeros(len(train_df), dtype=np.float32)

    for fold, (tr_idx, va_idx) in enumerate(
        skf.split(train_df["id_code"].values, oof_true)
    ):
        val_df = train_df.iloc[va_idx].reset_index(drop=True)

        val_ds = ImageIdDataset(val_df, TRAIN_IMG_DIR, transform, has_label=True)
        val_loader = DataLoader(
            val_ds,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=seed_worker,
            generator=g,
        )

        _, val_pred_float = _predict_regression_with_flip_tta(
            net, val_loader, output_mode=OUTPUT_MODE
        )
        oof_pred[va_idx] = val_pred_float.numpy()
        fold_qwk = cohen_kappa_score(
            val_df["diagnosis"].astype(int).values,
            _apply_thresholds_np(val_pred_float.numpy(), threshold),
            weights="quadratic",
        )
        print(f"Fold {fold} done. QWK@initial_thr={fold_qwk:.5f}")

    new_thr, oof_qwk = _optimize_thresholds_for_qwk(oof_true, oof_pred, threshold)
    print("Calibrated thresholds (OOF):", new_thr, " | OOF QWK:", oof_qwk)
    threshold = new_thr
except Exception as e:
    print("WARNING: Calibration step failed; proceeding with default thresholds/mode.")
    print("Calibration error:", repr(e))
    OUTPUT_MODE = "as_is"



## === cell 6
test_ds = ImageIdDataset(test_df, TEST_IMG_DIR, transform, has_label=False)
test_loader = DataLoader(
    test_ds,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    worker_init_fn=seed_worker,
    generator=g,
)

test_ids_pred, test_pred_float = _predict_regression_with_flip_tta(
    net, test_loader, output_mode=OUTPUT_MODE
)
test_pred_cls = _apply_thresholds_np(test_pred_float.numpy(), threshold)

pred_df = pd.DataFrame(
    {
        "id_code": np.array(test_ids_pred, dtype=str),
        "diagnosis": test_pred_cls.astype(int),
    }
)
pred_df["diagnosis"] = pred_df["diagnosis"].clip(0, 4)

df = test_df.merge(pred_df, on="id_code", how="left")
df["diagnosis"] = df["diagnosis"].fillna(0).astype(int).clip(0, 4)
df = df[["id_code", "diagnosis"]]

assert (
    len(df) == len(test_df) and df["diagnosis"].notna().all()
), "Submission is empty or misaligned."

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("Final thresholds used:", threshold)
print("OUTPUT_MODE used:", OUTPUT_MODE)
print("Checkpoint loaded:", ckpt_loaded)
print("Checkpoint path:", ckpt_path)
print("Using device:", device)
