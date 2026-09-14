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

0.9121147017905544

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the run blockers by (1) making device selection robust (CPU fallback when no GPU driver is available), and (2) handling the missing external weights file by automatically falling back to a lightweight, deterministic baseline prediction so a non-empty `submission.csv` is always produced. I also correct a few small transform/utility bugs that can silently break inference (e.g., `is` vs `==` for strings and `trim()` sometimes returning `None`). These changes keep your model code intact and only add safe guards to ensure end-to-end execution and a valid submission file is written. Because your current score was “Not yielded”, the priority is producing a valid submission; any score improvement beyond that isn’t possible without the missing weights.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with the model running without the intended trained weights (it falls back to predicting all zeros), so the smallest change that can move you toward the target is to correctly locate and load weights if they exist in the Kaggle dataset folders you already have mounted. I keep the model and transforms identical, but expand the weight search to common Kaggle input locations (including `/kaggle/input/**`) and make the state-dict loading a bit more robust (handling `module.` prefixes and checkpoint wrappers) so it actually uses the trained parameters when available. If no weights are found, it still produce a valid `submission.csv` exactly as before. This should increase QWK substantially when the weights are present, without changing training/inference semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the “no weights loaded → predict all zeros” fallback, so the smallest score-moving change is to make weight loading actually succeed when the checkpoint exists and is compatible. I keep your model and inference identical, but (1) improve `_extract_state_dict` to correctly unwrap common checkpoint formats and (2) load with `strict=False` only as a last resort (and report missing/unexpected keys) so partially-compatible checkpoints can still be used rather than falling back to all-zeros. If weights still can’t be loaded, the script behave exactly as before and still write a valid `submission.csv`. This should move QWK upward toward your target when the intended weights are present in `/kaggle/input/**`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is coming from the explicit “no weights loaded → predict all zeros” fallback, so the smallest score-improving change is to ensure your intended EfficientNet-B4 checkpoint is actually found and loaded in this environment. I keep your model, transforms, and inference logic identical, but (1) fix `DATA_DIR` to the real mounted path (`/kaggle/input/...`) so file discovery is consistent, and (2) expand checkpoint discovery to include common filename patterns and unwrap/load more checkpoint formats robustly (including `.pth`/`.pt`/`.pkl`/`.bin`, nested dicts, and `ema` keys). If weights still aren’t present, the script still produce a valid `submission.csv` exactly as before, but if they are present anywhere under `/kaggle/input`, this should move QWK upward toward your target.'
- What this solution (achieved 0.48823) has done: 'Your 0.0 score is consistent with the current “no weights loaded → predict all zeros” fallback, which cannot reach your 0.912 target; the smallest score-improving change is to (1) reliably find and load a compatible checkpoint if one exists anywhere under the mounted Kaggle input/data trees, and (2) if no weights exist, train the exact same model (same architecture, same regression head and thresholding) for a very small number of epochs on the provided `train.csv` so predictions become non-trivial instead of all-zero. I keep your model, transforms, and inference semantics intact; training uses the same forward branch you already use for inference (`r_out`), with a simple regression loss, and then uses your unchanged `regress2class()` thresholds. This should move QWK upward from 0.0 toward your target while staying within runtime by using a lightweight dataloader, small epoch count, and CPU/GPU depending on availability. The script still always writes a valid `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.47905) has done: 'I keep your model and thresholds exactly the same, but make the training fallback (used when no external weights are found) a bit more effective so the QWK moves upward toward your 0.912 target. The smallest high-impact change is to stop training on an arbitrary 800-row slice and instead use a stratified subset of the full train set (same size), so all grades 0–4 are represented; this typically improves ordinal calibration without changing architecture or loss. I also make inference substantially faster (and therefore allow training to finish reliably) by batching test prediction in a DataLoader (same per-image transform and same `regress2class()`), which doesn’t change semantics but reduces overhead. Finally, I add a quick post-training threshold calibration on a held-out validation split of that subset (only if we trained in-notebook), adjusting only the four scalar thresholds used by your existing mapping—this usually boosts QWK notably while preserving the core regression approach.'
- What this solution (achieved 0.53628) has done: 'Your current score (0.47905) is far below the target (0.91211), so we should cautiously improve performance without changing your model/transform/loss semantics. The biggest issue in your fallback training is that your validation split is created via a separate stratified sampling and then dropped by index, which is incorrect because the sampled indices don’t correspond to `sub_df`’s indices; this silently breaks train/val separation and harms threshold calibration. I fix the split by doing a proper stratified split within `sub_df` (same data budget), then keep your same 3-epoch SmoothL1 regression training and the same threshold-search calibration, just on the correctly held-out set. I also make the inference dataframe alignment follow `test.csv` order explicitly (already mostly true) and keep the submission format identical.'
- What this solution (achieved 0.0) has done: 'The timeout is dominated by the fallback path that does 3-fold OOF training with image caching plus a full training run, which is far beyond 600 seconds on this dataset. To preserve core logic and accuracy while making runtime predictable, the refactor forces the inference-only path by requiring pretrained weights and skipping the expensive training fallback (instead of attempting to train from scratch). Inference is sped up by enabling `torch.inference_mode()`, using channels-last + `torch.compile` (when available) for the model, and reducing per-sample overhead by vectorizing `regress2class_prob` and keeping threshold tensors cached. Data loading is tuned with persistent workers and prefetching while keeping transforms/semantics identical.'
- What this solution (achieved 0.0) has done: 'I remove the hard failure when pretrained weights aren’t found so the notebook always runs end-to-end and writes a valid `submission.csv`. I also broaden checkpoint discovery to search under the mounted Kaggle dataset trees you actually have (`/kaggle/input` and `/kaggle/data`) to increase the chance the intended weights are found/loaded without changing your model. If no checkpoint is available, the script fall back to a deterministic baseline that produces a non-empty submission (score may remain low, but it won’t crash). These changes are focused on unblocking execution and enabling score improvement when weights exist.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score strongly suggests the submission is still effectively predicting a constant class (or weights aren’t being used as intended), so the smallest score-moving change is to stop converting the regression output to hard classes with fixed thresholds and instead optimize those same four thresholds on a small held-out validation split when (and only when) we are using loaded weights. This preserves your exact model, forward path (`r_out`), transforms, and evaluation semantics, but calibrates the only free decision step that directly targets QWK. I also make the weight-file search slightly more reliable by preferring the most likely competition dataset folder first and by allowing `.ckpt` (common in PyTorch Lightning), without changing any model logic. The script still always writes a valid `submission.csv` in the required order.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with the “no weights loaded → predict all zeros” path still being triggered, so the smallest score-moving change is to make weight discovery/load succeed more often without changing your model or inference semantics. I broaden the checkpoint search to include the competition’s `/kaggle/input/**` trees more reliably (including `.ckpt`) and make checkpoint unwrapping handle common PyTorch Lightning formats (where weights live under `state_dict` with prefixes). I also tighten the fallback condition so we only drop to all-zeros when we truly cannot load compatible weights, and keep the same threshold calibration step when weights are loaded. No architecture, transforms, or prediction mapping logic are changed—only the robustness of loading the intended trained parameters so the submission becomes non-trivial and the score moves upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score indicates the “no weights → predict all zeros” fallback is still being used, so the smallest score-improving change is to (1) make checkpoint discovery much more likely to find a real model file by scanning only plausible checkpoint names under `/kaggle/input` and `/kaggle/data` without an enormous directory walk, and (2) make loading robust to common wrappers/prefixes while refusing “mostly-missing” loads so we don’t silently use random weights. If we successfully load weights, we keep your exact inference path (`r_out` + `regress2class`) but also calibrate the four thresholds on a validation split using the same `r_out` outputs, which directly targets QWK and typically lifts score substantially. If no usable weights exist, behavior stays identical (valid submission with zeros), ensuring end-to-end execution and valid CSV. These changes preserve your model architecture, transforms, and prediction semantics—only improving the odds that the intended trained parameters are actually used and properly calibrated.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with still falling back to “all zeros” predictions because no usable checkpoint is being loaded, so the smallest score-moving change is to make checkpoint discovery/load succeed more often without changing the model or inference semantics. I (1) fix the dataset path fallback to also cover the common mounted location `/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection`, (2) improve weight search to look inside that folder and prefer higher-scoring hits, and (3) make checkpoint unwrapping handle common PyTorch Lightning keys (`state_dict` + `model.` prefix) and return a clear load/no-load decision. If weights still aren’t found, the behavior remains identical (valid `submission.csv` with zeros), but when weights are present this should lift QWK substantially toward your target.'

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
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]

_thr_tensor_cpu = None
_thr_tensor_dtype = None


def _get_thr_tensor(dtype):
    global _thr_tensor_cpu, _thr_tensor_dtype
    if _thr_tensor_cpu is None or _thr_tensor_dtype != dtype:
        _thr_tensor_cpu = torch.tensor(threshold, dtype=dtype, device="cpu")
        _thr_tensor_dtype = dtype
    return _thr_tensor_cpu


def regress2class(out):
    if out.ndim > 1:
        out = out.squeeze(1)
    out_cpu = out.detach().to("cpu")
    thr = _get_thr_tensor(out_cpu.dtype)  # (4,)
    pred = (out_cpu[:, None] >= thr[None, :]).sum(dim=1).to(torch.float32)
    return pred


def ordinal2class_prob(out):
    dev = out.device
    pred_prob = torch.zeros(out.size(0), 5, device=dev)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    if out.ndim > 1:
        out = out.squeeze(1)
    out_det = out.detach().to(dtype=torch.float32)
    dev = out_det.device

    out_c = torch.clamp(out_det, 0.0, 4.0)
    l1 = torch.floor(out_c).to(torch.long)  # in [0,4]
    l2 = torch.ceil(out_c).to(torch.long)  # in [0,4]

    pred_prob = torch.zeros((out_det.size(0), 5), device=dev, dtype=torch.float32)

    frac = out_c - l1.to(out_c.dtype)  # in [0,1]
    w1 = 1.0 - frac
    w2 = 1.0 - (l2.to(out_c.dtype) - out_c)

    idx = torch.arange(out_det.size(0), device=dev)
    pred_prob[idx, l1] += w1
    pred_prob[idx, l2] += w2

    ge4 = out_det >= 4.0
    if ge4.any():
        pred_prob[ge4] = 0.0
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
DATA_DIR = "/kaggle/input/aptos2019-blindness-detection"
ALT_DATA_DIR = (
    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection"
)
ALT_DATA_DIR2 = "/kaggle/data/aptos2019-blindness-detection"


def _resolve_data_dir():
    for d in [DATA_DIR, ALT_DATA_DIR, ALT_DATA_DIR2]:
        if os.path.exists(os.path.join(d, "test.csv")) and os.path.exists(
            os.path.join(d, "train.csv")
        ):
            return d
    return DATA_DIR


DATA_DIR = _resolve_data_dir()
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

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

net = ThreeStage_Model().to(device)

WEIGHTS_CANDIDATES = [
    os.path.join(DATA_DIR, "B4_3stage_5epoch_finetune.pkl"),
    os.path.join(DATA_DIR, "B4_3stage_5epoch_finetune.pth"),
    os.path.join(DATA_DIR, "B4_3stage_5epoch_finetune.pt"),
    os.path.join(DATA_DIR, "B4_3stage_5epoch_finetune.ckpt"),
    "/kaggle/input/aptos2019-blindness-detection/B4_3stage_5epoch_finetune.pkl",
    "/kaggle/input/aptos2019-blindness-detection/B4_3stage_5epoch_finetune.pth",
    "/kaggle/input/aptos2019-blindness-detection/B4_3stage_5epoch_finetune.pt",
    "/kaggle/input/aptos2019-blindness-detection/B4_3stage_5epoch_finetune.ckpt",
    "../input/weights/B4_3stage_5epoch_finetune.pkl",
    "../input/weights/B4_3stage_5epoch_finetune.pth",
    "../input/weights/B4_3stage_5epoch_finetune.pt",
    "../input/weights/B4_3stage_5epoch_finetune.ckpt",
    "/kaggle/input/weights/B4_3stage_5epoch_finetune.pkl",
    "/kaggle/input/weights/B4_3stage_5epoch_finetune.pth",
    "/kaggle/input/weights/B4_3stage_5epoch_finetune.pt",
    "/kaggle/input/weights/B4_3stage_5epoch_finetune.ckpt",
]


def _find_weight_file():
    for p in WEIGHTS_CANDIDATES:
        if os.path.exists(p):
            return p

    roots = [
        DATA_DIR,
        ALT_DATA_DIR,
        "/kaggle/input",
        "/kaggle/data",
    ]
    exts = (".pth", ".pt", ".pkl", ".bin", ".ckpt")

    strong_terms = [
        "b4_3stage_5epoch_finetune",
        "3stage",
        "three",
        "tf_efficientnet_b4",
        "efficientnet_b4",
        "efficientnet",
        "aptos",
        "blindness",
        "retina",
        "dr",
    ]
    weak_terms = ["best", "fold", "epoch", "checkpoint", "model", "weights"]

    best_hit = None
    best_score = -1

    for root in roots:
        if not os.path.exists(root):
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                lfn = fn.lower()
                if not lfn.endswith(exts):
                    continue
                score = 0
                for t in strong_terms:
                    if t in lfn:
                        score += 10
                for t in weak_terms:
                    if t in lfn:
                        score += 1
                score2 = score * 1000 - len(os.path.join(dirpath, fn))
                if score2 > best_score:
                    best_score = score2
                    best_hit = os.path.join(dirpath, fn)

    if best_hit is not None and best_score >= 10 * 1000:
        return best_hit
    return None


weights_path = _find_weight_file()


def _extract_state_dict(state_obj):
    if isinstance(state_obj, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
        ]:
            if k in state_obj and isinstance(state_obj[k], dict):
                state_obj = state_obj[k]
                break

    if isinstance(state_obj, dict):
        for k in ["ema_state_dict", "model_ema", "ema", "state_dict_ema"]:
            if k in state_obj and isinstance(state_obj[k], dict):
                state_obj = state_obj[k]
                break

    if not isinstance(state_obj, dict) or len(state_obj) == 0:
        return None

    def _looks_like_state_dict(d):
        if not isinstance(d, dict) or len(d) == 0:
            return False
        for v in d.values():
            if isinstance(v, (torch.Tensor, torch.nn.Parameter)):
                return True
        return False

    if not _looks_like_state_dict(state_obj):
        for v in state_obj.values():
            if isinstance(v, dict) and _looks_like_state_dict(v):
                state_obj = v
                break

    if not _looks_like_state_dict(state_obj):
        return None

    def _strip_prefix(d, prefix):
        if any(isinstance(k, str) and k.startswith(prefix) for k in d.keys()):
            return {k.replace(prefix, "", 1): v for k, v in d.items()}
        return d

    for pref in ["module.", "model.", "net.", "backbone."]:
        state_obj = _strip_prefix(state_obj, pref)

    if not all(isinstance(k, str) for k in state_obj.keys()):
        return None
    return state_obj


has_weights = False
if weights_path is not None:
    try:
        state = torch.load(weights_path, map_location="cpu")
        state = _extract_state_dict(state)
        if state is None:
            raise ValueError(
                "Unsupported checkpoint format (no usable state_dict found)."
            )

        try:
            net.load_state_dict(state, strict=True)
            has_weights = True
        except RuntimeError:
            missing, unexpected = net.load_state_dict(state, strict=False)
            missing = list(missing) if missing is not None else []
            unexpected = list(unexpected) if unexpected is not None else []
            print("Warning: strict=True load failed, tried strict=False.")
            print("Missing keys (first 30):", missing[:30])
            print("Unexpected keys (first 30):", unexpected[:30])

            total = len(net.state_dict())
            ok = (total > 0) and (len(missing) <= 0.25 * total)
            has_weights = bool(ok)
            if not has_weights:
                print(
                    "Warning: too many missing keys; treating as NO WEIGHTS to avoid junk predictions."
                )
    except Exception as e:
        print(f"Warning: failed to load weights from {weights_path}: {e}")
        has_weights = False

print(
    f"Device: {device} | weights_loaded: {has_weights} | weights_path: {weights_path} | DATA_DIR: {DATA_DIR}"
)

if has_weights and device.type == "cuda":
    net = net.to(memory_format=torch.channels_last)
    try:
        net = torch.compile(net, mode="max-autotune", fullgraph=False)
        print("torch.compile enabled.")
    except Exception as e:
        print("torch.compile not available/failed; continuing without it:", str(e))

net.eval()



## === cell 5
from torch.utils.data import Dataset, DataLoader


class APTOSDataset(Dataset):
    def __init__(self, df, img_dir, transform, with_labels=True, cache_images=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.with_labels = with_labels
        self.cache_images = cache_images
        self._cache = {} if cache_images else None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        if self.cache_images and i in self._cache:
            img = self._cache[i]
            row = self.df.iloc[i]
        else:
            row = self.df.iloc[i]
            idx = str(row["id_code"])
            image_name = os.path.join(self.img_dir, f"{idx}.png")
            with Image.open(image_name) as im:
                img = im.convert("RGB")
            img = self.transform(img)
            if self.cache_images:
                self._cache[i] = img

        if self.with_labels:
            y = int(row["diagnosis"])
            return img, torch.tensor([y], dtype=torch.float32)
        return img, str(row["id_code"])


def _make_loader(ds, batch_size, shuffle, num_workers):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(num_workers > 0),
    )
    if num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(ds, **kwargs)




## === cell 6
def _qwk_for_thresholds(y_true_int, y_pred_reg, thr_list):
    thr_t = torch.tensor(thr_list, dtype=torch.float32)
    y_pred_reg_t = torch.tensor(y_pred_reg, dtype=torch.float32).view(-1)
    pred = (
        (y_pred_reg_t[:, None] >= thr_t[None, :]).sum(dim=1).cpu().numpy().astype(int)
    )
    return cohen_kappa_score(y_true_int, pred, weights="quadratic")


def _calibrate_thresholds_on_val(net, train_df, img_dir, transform, max_val_images=800):
    df = train_df.copy()
    df["diagnosis"] = df["diagnosis"].astype(int)
    tr_df, va_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["diagnosis"]
    )

    if len(va_df) > max_val_images:
        va_df_small, _ = train_test_split(
            va_df,
            train_size=max_val_images,
            random_state=SEED,
            stratify=va_df["diagnosis"],
        )
        va_df = va_df_small

    va_ds = APTOSDataset(
        va_df, img_dir, transform, with_labels=True, cache_images=False
    )
    num_workers = min(8, os.cpu_count() or 4) if (os.name != "nt") else 0
    bs = 32 if device.type == "cuda" else 8
    va_dl = _make_loader(va_ds, batch_size=bs, shuffle=False, num_workers=num_workers)

    y_true = []
    y_reg = []
    net.eval()
    with torch.inference_mode():
        for xb, yb in va_dl:
            if device.type == "cuda":
                xb = xb.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
            else:
                xb = xb.to(device)
            _, r_out, _ = net(xb)
            y_reg.append(r_out.detach().float().cpu().view(-1))
            y_true.append(yb.detach().cpu().view(-1))
    y_reg = torch.cat(y_reg).numpy()
    y_true = torch.cat(y_true).numpy().astype(int)

    base_thr = np.array(threshold, dtype=np.float32)
    best_thr = base_thr.copy()
    best = _qwk_for_thresholds(y_true, y_reg, best_thr.tolist())

    for step in [0.30, 0.15, 0.07]:
        improved = True
        while improved:
            improved = False
            for j in range(4):
                for delta in [-step, step]:
                    cand = best_thr.copy()
                    cand[j] = float(cand[j] + delta)
                    cand = np.clip(cand, 0.0, 4.0)
                    cand = np.sort(cand)
                    score = _qwk_for_thresholds(y_true, y_reg, cand.tolist())
                    if score > best + 1e-6:
                        best = score
                        best_thr = cand
                        improved = True

    return best_thr.tolist(), float(best), len(va_df)


if has_weights:
    try:
        new_thr, val_qwk, nval = _calibrate_thresholds_on_val(
            net, train_df, TRAIN_IMG_DIR, transform
        )
        threshold = new_thr
        _thr_tensor_cpu = None
        _thr_tensor_dtype = None
        print(
            f"Calibrated thresholds on {nval} val images; val QWK={val_qwk:.5f}; thresholds={threshold}"
        )
    except Exception as e:
        print(
            "Warning: threshold calibration failed; using default thresholds. Error:",
            str(e),
        )

test_ds = APTOSDataset(
    test_df.assign(diagnosis=0),
    TEST_IMG_DIR,
    transform,
    with_labels=False,
    cache_images=False,
)

num_workers = min(8, os.cpu_count() or 4) if (os.name != "nt") else 0
bs_inf = 48 if (device.type == "cuda" and has_weights) else 8
test_dl = _make_loader(
    test_ds, batch_size=bs_inf, shuffle=False, num_workers=num_workers
)

submission_rows = []
if has_weights:
    with torch.inference_mode():
        for xb, idxs in test_dl:
            if device.type == "cuda":
                xb = xb.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
            else:
                xb = xb.to(device)
            _, r_out, _ = net(xb)
            preds = regress2class(r_out).numpy().astype(int).tolist()
            for idx, p in zip(idxs, preds):
                submission_rows.append([str(idx), int(p)])
else:
    submission_rows = [[str(rid), 0] for rid in test_ids]

pred_map = {rid: diag for rid, diag in submission_rows}
submission_rows = [[rid, int(pred_map.get(rid, 0))] for rid in test_ids]
submission = np.array(submission_rows, dtype=object)



## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

assert (
    len(df) == len(test_ids) and len(df) > 0
), "Submission DataFrame should not be empty and must match test size."
df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("Final thresholds used:", threshold)
print("Used pretrained weights:", has_weights)
