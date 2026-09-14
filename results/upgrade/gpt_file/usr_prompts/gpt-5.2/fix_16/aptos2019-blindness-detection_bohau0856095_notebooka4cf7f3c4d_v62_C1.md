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

0.9144714453611804

# 6. Current score

-0.03395

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the notebook robust to Kaggle’s environment by (1) removing the failing custom `pip install` and instead using the already-installed `timm`, (2) automatically selecting `cuda` only if available (otherwise CPU) to fix the “no NVIDIA driver” crash, and (3) handling the missing `../input/weights/...pkl` by falling back to generating a valid (non-empty) submission using the competition’s required format. I also fix small but real logic issues in transforms (`is` vs `==`, and `trim()` returning `None`) that can break preprocessing. These changes keep the core model/inference logic intact when weights exist, while guaranteeing an end-to-end run that always writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is caused by never loading real model weights (the code falls back to predicting all zeros), so the smallest meaningful improvement is to correctly locate and load weights if they exist inside the provided competition dataset tree. I keep the exact same model and inference logic, but add a robust search for `B4_3stage_1epoch_finetune.pkl` under `../input/` and (if present) load it; otherwise we still emit a valid submission. I also ensure the submission row order exactly matches `test.csv` by building predictions keyed by `id_code` and then reassembling in that order, preventing rare misalignment issues that can tank kappa. These changes are minimal and directly aimed at moving the score up toward your 0.914 target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from the “all zeros” fallback because no weights are being found/loaded, so the smallest real improvement is to reliably load the provided pretrained weights (if present) and make `load_state_dict` robust to common checkpoint formats (`state_dict`, `model`, `net`). To move kappa upward without changing model logic, I also make inference faster/cleaner (batching + `torch.inference_mode`) while keeping identical transforms and prediction mapping. Finally, I keep the submission strictly aligned to `test.csv` order and ensure every `id_code` gets a prediction, avoiding accidental missing keys that can invalidate or damage the score. These changes preserve the same architecture, thresholds, and evaluation semantics—just ensure you’re actually using the trained model when weights exist.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly from the model not loading real weights and therefore outputting the all-zeros fallback (which yields near-random/very poor kappa). To move the score up toward your 0.914 target with minimal changes, I (1) make weight discovery robust to common Kaggle layouts (including the current working directory), (2) make checkpoint loading handle common key-prefix issues (e.g., `module.` and `backbone.`) without changing the model, and (3) switch inference to the model’s `final=True` head (the one trained to fuse the three heads) while keeping your exact discretization thresholds and submission alignment. If weights are still not found, it still produce a valid `submission.csv` (all zeros) exactly as before.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score indicates the model is still effectively predicting a constant class (either because weights aren’t being found/loaded, or because the loaded checkpoint keys don’t match the model). I make the smallest change that increases the chance of successfully loading real weights: broaden the checkpoint discovery to include any `.pkl/.pth/.pt` in common Kaggle locations and robustly extract/normalize state_dict keys (including `backbone.`-prefixed keys) without changing the model. If weights still can’t be loaded, the script behave exactly as before (valid all-zeros submission), but if they can, predictions should move substantially upward toward your 0.914 target. I also keep submission order strictly aligned to `test.csv` as you already do.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the model never actually loading usable weights, so the smallest improvement toward your 0.914 target is to (1) reliably locate a real checkpoint under the Kaggle filesystem, and (2) load it in a way that matches your model keys (handling common wrappers/prefixes and PyTorch serialization formats) without changing the model or thresholds. I also make inference deterministic/consistent by avoiding any randomness sources in the test pipeline and ensuring the model is on the correct device before loading. Finally, I keep your submission alignment checks, but I build the submission directly from `test.csv` to guarantee perfect ordering and row count.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the model still not loading the intended pretrained weights (so it falls back to predicting all zeros). To move the score upward toward the 0.914 target with minimal disruption, I (1) broaden and prioritize checkpoint discovery to look specifically for your expected EfficientNet-B4 three-stage checkpoint (and close name variants) inside the competition dataset tree, and (2) make state_dict loading more robust to common key-prefix mismatches (`backbone.`, `model.`, `net.`, `module.`) while still preferring a strict load when possible. I not change the model, transforms, thresholds, or inference semantics; only the weight-finding/loading reliability is improved. The submission creation/order checks remain intact to prevent silent misalignment that can destroy kappa.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is because the notebook still can’t reliably load a real checkpoint, so it falls back to predicting all zeros. I keep your exact model, transforms, and discretization thresholds, but make the checkpoint discovery/loading more robust in a minimal way: search specifically for likely EfficientNet-B4 three-stage checkpoints under `../input/aptos2019-blindness-detection/` (including nested folders) and then load the *best-overlap* state_dict variant, retrying with common key-prefix fixes. This should move the score upward toward your 0.914 target when weights are present, while preserving identical inference semantics once weights are loaded. The submission is still built strictly in `test.csv` order and always writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score comes from the fallback path where no usable checkpoint is actually being loaded, so predictions collapse to all zeros. I keep the exact same model, transforms, thresholds, and inference semantics, but make the checkpoint discovery stricter-and-broader: search all of `/kaggle/input` (and the provided competition folder) for any likely EfficientNet-B4 three-stage checkpoint, and prefer the best match by filename plus file size. Then I make state_dict loading more robust to common wrapper/prefix patterns and (crucially) handle checkpoints where weights are stored under an EMA key (common in timm training) without changing any model logic. If no checkpoint is found, it still produce the same valid all-zeros `submission.csv`, but if one exists in the environment, this should move the score up toward your 0.914 target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the model producing an almost-constant prediction, which in your code happens whenever weights fail to load (or load mismatched keys silently). I make the smallest changes that increase the probability of actually loading a compatible checkpoint (including common formats like `state_dict` nested under EMA keys and `module.` prefixes) while keeping the same model, transforms, and thresholds. I also ensure the checkpoint is loaded onto the already-moved-to-device model in a stable way and keep the submission strictly aligned to `test.csv` as you already do. If no usable checkpoint exists, the behavior remains identical (valid all-zeros submission), but if one does exist, the score should move upward toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score indicates the model is still not using meaningful weights (so predictions collapse to a constant), so the smallest score-improving change is to reliably load a compatible checkpoint if one exists in the dataset tree. I keep your exact model, transforms, thresholds, and inference semantics, but (1) expand checkpoint discovery to also scan `/kaggle/input/**` for likely EfficientNet-B4 three-stage weights and (2) improve state_dict normalization by auto-detecting the best-matching key transform based on overlap with the current model’s parameters (so we don’t accidentally “load” an incompatible checkpoint). Finally, I add a hard safety check to refuse using a checkpoint unless the overlap ratio is high enough, preventing silent bad loads that can keep kappa near zero while still setting `weights_loaded=True`. These changes are directly aimed at turning the current all-zeros behavior into real model predictions, moving the score upward toward your 0.914 target.'
- What this solution (achieved 0.02381) has done: 'Your score is 0.0 because the script still usually can’t find any usable pretrained checkpoint, so it falls back to predicting all zeros. To move the score up toward the 0.914 target with minimal disruption, I only change the weight discovery/loading to (a) also look for Kaggle “Dataset” attached files by scanning `/kaggle/input/**` for likely EfficientNet-B4 weights and (b) if nothing is found, automatically use timm’s ImageNet pretrained backbone weights (same architecture) rather than outputting constant zeros. I keep your exact model, transforms, thresholds, and submission ordering; the only modeling change is enabling pretrained backbone initialization as a last-resort fallback to produce non-trivial predictions. This should substantially increase kappa from 0.0 while remaining within Kaggle constraints and still writing a valid `submission.csv`.'
- What this solution (achieved -0.0774) has done: 'Your current 0.02381 is far below the 0.914 target, so we should improve the score by making sure the predictions are not dominated by a mis-calibrated final regressor head when we only have an ImageNet-pretrained backbone fallback. I keep your exact model and discretization thresholds, but add a minimal, metric-aligned inference fallback: if a real DR checkpoint is not loaded, use the classifier head’s argmax (already part of your model) instead of `final=True` regression, because the final regressor was never trained for ImageNet features and can collapse to near-constant outputs (hurting kappa). If a compatible checkpoint is successfully loaded, inference remains exactly as you currently do (`final=True` + `regress2class`). This change is directly aimed at moving the score upward toward your target without altering architecture/training/loss, and still writes a valid `submission.csv`.'
- What this solution (achieved -0.03395) has done: 'Your current score is far below the 0.914 target, and the biggest likely cause (given your code) is that you’re still not loading any real DR-trained checkpoint, so you’re effectively using an ImageNet backbone with randomly initialized heads—then taking classifier argmax—which can easily produce very poor kappa. I keep your exact model and transforms, but (1) make checkpoint discovery/loading more reliable by preferring any checkpoint that actually matches your model’s parameter keys (not just filename heuristics), and (2) if no DR checkpoint is found, keep your current fallback but calibrate it minimally by using the regression head (already in your model) with your existing `regress2class` thresholds, which tends to be less arbitrary than an untrained 5-way classifier head. These changes don’t alter architecture/loss/training; they only improve the chance you’re using real weights and make the fallback predictions less degenerate, which should move the score upward toward the target. Submission ordering/format safeguards remain unchanged and it still always write `submission.csv`.'
- What this solution (achieved -0.03395) has done: 'Your current score (-0.03395) is far below the target (0.914), so we should only make small, high-impact fixes that increase the chance you’re actually using a compatible DR-trained checkpoint (the main determinant of kappa here). I keep the exact same model, transforms, and discretization thresholds, but improve checkpoint selection by (1) also scoring by “state_dict overlap ratio” (not just filename/size) across a slightly larger candidate set, and (2) lowering the overlap acceptance threshold a bit while adding a safety rule that the backbone must largely match (so we don’t accept a wrong model). If no DR checkpoint is usable, I keep your ImageNet-backbone fallback but force it to use the regression head + your existing thresholds (avoiding the untrained classifier argmax path entirely), which should reduce degenerate predictions and move kappa upward. All changes are targeted to loading/using the right weights and producing a correctly-aligned `submission.csv`.'

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
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



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
TEST_CSV = "../input/aptos2019-blindness-detection/test.csv"
TEST_IMG_DIR = "../input/aptos2019-blindness-detection/test_images"
SAMPLE_SUB = "../input/aptos2019-blindness-detection/sample_submission.csv"

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
net.eval()

WEIGHTS_BASENAME = "B4_3stage_1epoch_finetune.pkl"

CANDIDATE_PATHS = [
    "./" + WEIGHTS_BASENAME,
    "../input/weights/" + WEIGHTS_BASENAME,
    "../input/aptos2019-blindness-detection/weights/" + WEIGHTS_BASENAME,
    "../input/aptos2019-blindness-detection/" + WEIGHTS_BASENAME,
    "/kaggle/input/weights/" + WEIGHTS_BASENAME,
    "/kaggle/input/aptos2019-blindness-detection/weights/" + WEIGHTS_BASENAME,
    "/kaggle/input/aptos2019-blindness-detection/" + WEIGHTS_BASENAME,
    "/kaggle/working/" + WEIGHTS_BASENAME,
]

found_weights_path = None
for p in CANDIDATE_PATHS:
    if os.path.exists(p):
        found_weights_path = p
        break


def _score_ckpt_path(path: str) -> int:
    """Higher is better: prioritize files that look like the intended B4 three-stage finetune checkpoint."""
    name = os.path.basename(path).lower()
    s = 0
    if "b4" in name:
        s += 30
    if "3stage" in name or "three" in name:
        s += 30
    if "finetune" in name or "fine_tune" in name:
        s += 20
    if "epoch" in name:
        s += 5
    if any(x in name for x in ("efficientnet", "tf_efficientnet", "effnet")):
        s += 10
    if any(x in name for x in ("ema", "model_ema")):
        s += 2
    if name.endswith((".pkl", ".pth", ".pt", ".bin")):
        s += 5
    try:
        s += min(60, int(os.path.getsize(path) / (1024 * 1024)))  # +1 per MB, capped
    except Exception:
        pass
    return s


if found_weights_path is None:
    plausible = []
    search_roots = [
        "../input/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection",
        "../input",
        "/kaggle/input",
        ".",
        "/kaggle/working",
    ]
    exts = (".pkl", ".pth", ".pt", ".bin")

    for sr in search_roots:
        if not os.path.exists(sr):
            continue
        for root, _, files in os.walk(sr):
            for fn in files:
                lfn = fn.lower()
                full = os.path.join(root, fn)
                if lfn == WEIGHTS_BASENAME.lower():
                    plausible.append(full)
                elif lfn.endswith(exts):
                    if (
                        ("b4" in lfn)
                        or ("3stage" in lfn)
                        or ("three" in lfn)
                        or ("finetune" in lfn)
                        or ("fine_tune" in lfn)
                        or ("tf_efficientnet" in lfn)
                        or ("efficientnet" in lfn)
                        or ("aptos" in lfn)
                    ):
                        plausible.append(full)
            if len(plausible) >= 4000:
                break
        if len(plausible) >= 4000:
            break

    if plausible:
        plausible = sorted(plausible, key=_score_ckpt_path, reverse=True)
        found_weights_path = plausible[0]


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
            "parameters",
            "ema",
            "model_ema",
            "state_dict_ema",
            "ema_state_dict",
        ):
            if k in ckpt and isinstance(ckpt[k], dict):
                inner = ckpt[k]
                if "state_dict" in inner and isinstance(inner["state_dict"], dict):
                    return inner["state_dict"]
                return inner
    return ckpt


def _fix_double_backbone_prefix(state: dict) -> dict:
    if not isinstance(state, dict):
        return state
    if any(k.startswith("backbone.backbone.") for k in state.keys()):
        return {
            k.replace("backbone.backbone.", "backbone.", 1): v for k, v in state.items()
        }
    return state


def _best_normalized_state_dict(state: dict, model: nn.Module):
    if not isinstance(state, dict):
        return state, -1.0, {}

    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    model_n = len(model_keys)

    def strip_prefix(d, pref):
        return {k[len(pref) :] if k.startswith(pref) else k: v for k, v in d.items()}

    def add_backbone_prefix(d):
        out = {}
        for k, v in d.items():
            if k.startswith(
                (
                    "classifier.",
                    "regressor.",
                    "ordinal.",
                    "final_regressor.",
                    "backbone.",
                )
            ):
                out[k] = v
            else:
                out["backbone." + k] = v
        return out

    candidates = []
    candidates.append(("raw", state))
    if any(k.startswith("module.") for k in state.keys()):
        candidates.append(("strip_module", strip_prefix(state, "module.")))
    if any(k.startswith("model.") for k in state.keys()):
        candidates.append(("strip_model", strip_prefix(state, "model.")))
    if any(k.startswith("net.") for k in state.keys()):
        candidates.append(("strip_net", strip_prefix(state, "net.")))
    if any(k.startswith("backbone.") for k in state.keys()):
        candidates.append(("strip_backbone", strip_prefix(state, "backbone.")))

    candidates.append(("add_backbone", add_backbone_prefix(state)))
    for name, d in list(candidates):
        if isinstance(d, dict):
            candidates.append((name + "_fixdbl", _fix_double_backbone_prefix(d)))

    if any(k.startswith("module.model.") for k in state.keys()):
        candidates.append(("strip_module_model", strip_prefix(state, "module.model.")))

    best = None
    best_ratio = -1.0
    best_stats = None

    for name, d in candidates:
        if not isinstance(d, dict):
            continue
        overlap = sum(1 for k in d.keys() if k in model_keys)
        ratio = overlap / max(1, model_n)
        stats = {"name": name, "overlap": overlap, "ratio": ratio, "keys": len(d)}
        if ratio > best_ratio:
            best_ratio = ratio
            best = d
            best_stats = stats

    return best, best_ratio, best_stats if best_stats is not None else {}


def _backbone_overlap_ratio(state: dict, model: nn.Module) -> float:
    """Change: ensure we only accept checkpoints whose EfficientNet backbone matches well.
    This avoids 'loading' a wrong checkpoint that still overlaps on head layers and can tank kappa.
    """
    if not isinstance(state, dict):
        return -1.0
    model_keys = set(model.state_dict().keys())
    bb_model = [k for k in model_keys if k.startswith("backbone.")]
    if not bb_model:
        return -1.0
    bb_overlap = sum(
        1 for k in state.keys() if k in model_keys and k.startswith("backbone.")
    )
    return bb_overlap / max(1, len(bb_model))


def _find_best_ckpt_by_overlap(
    paths, model, topk=30, min_ratio=0.55, min_backbone_ratio=0.60
):
    """Change: consider more candidates and accept slightly lower overall overlap, but require strong backbone overlap.
    This increases odds of finding a usable DR-trained checkpoint in varied Kaggle layouts, improving score toward target.
    """
    scored = []
    paths = sorted(paths, key=_score_ckpt_path, reverse=True)[: max(1, topk)]
    for p in paths:
        try:
            ckpt = torch.load(p, map_location="cpu")
            state = _extract_state_dict(ckpt)

            if isinstance(ckpt, dict):
                for ema_key in ("model_ema", "ema", "state_dict_ema", "ema_state_dict"):
                    if ema_key in ckpt and isinstance(ckpt[ema_key], dict):
                        ema_state = _extract_state_dict({ema_key: ckpt[ema_key]})
                        if isinstance(ema_state, dict) and len(ema_state) >= 50:
                            state = ema_state
                            break

            state, ratio, stats = _best_normalized_state_dict(state, model)
            bb_ratio = _backbone_overlap_ratio(state, model)
            stats = dict(stats) if isinstance(stats, dict) else {}
            stats["backbone_ratio"] = bb_ratio
            scored.append((ratio, bb_ratio, stats, p, state))
        except Exception:
            continue

    if not scored:
        return None, None, -1.0, {}

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best_ratio, best_bb_ratio, best_stats, best_path, best_state = scored[0]

    if best_ratio < min_ratio or best_bb_ratio < min_backbone_ratio:
        return None, None, best_ratio, best_stats
    return best_path, best_state, best_ratio, best_stats


weights_loaded = False

fallback_mode = "regressor_threshold"

plausible_all = []
for p in CANDIDATE_PATHS:
    if os.path.exists(p):
        plausible_all.append(p)

if not plausible_all:
    search_roots = [
        "../input/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection",
        "../input",
        "/kaggle/input",
        ".",
        "/kaggle/working",
    ]
    exts = (".pkl", ".pth", ".pt", ".bin")
    for sr in search_roots:
        if not os.path.exists(sr):
            continue
        for root, _, files in os.walk(sr):
            for fn in files:
                lfn = fn.lower()
                if lfn.endswith(exts) and (
                    ("b4" in lfn)
                    or ("3stage" in lfn)
                    or ("three" in lfn)
                    or ("finetune" in lfn)
                    or ("fine_tune" in lfn)
                    or ("tf_efficientnet" in lfn)
                    or ("efficientnet" in lfn)
                    or ("aptos" in lfn)
                ):
                    plausible_all.append(os.path.join(root, fn))
            if len(plausible_all) >= 1200:
                break
        if len(plausible_all) >= 1200:
            break

best_path, best_state, best_ratio, best_stats = _find_best_ckpt_by_overlap(
    list(dict.fromkeys(plausible_all)),
    net,
    topk=30,
    min_ratio=0.55,
    min_backbone_ratio=0.60,
)

if best_path is not None:
    found_weights_path = best_path
    try:
        try:
            net.load_state_dict(best_state, strict=True)
            weights_loaded = True
            print(
                "Loaded best-overlap checkpoint with strict=True:",
                best_path,
                best_stats,
            )
        except RuntimeError:
            incompatible = net.load_state_dict(best_state, strict=False)
            missing = len(incompatible.missing_keys)
            unexpected = len(incompatible.unexpected_keys)
            model_n = len(net.state_dict().keys())
            if missing > int(0.25 * model_n):
                weights_loaded = False
                print(
                    "Refusing strict=False load; too many missing keys.",
                    best_path,
                    best_stats,
                )
            else:
                weights_loaded = True
                print(
                    "Loaded best-overlap checkpoint with strict=False:",
                    best_path,
                    best_stats,
                    "missing_keys=",
                    missing,
                    "unexpected_keys=",
                    unexpected,
                )
    except Exception as e:
        print(
            "Failed to load best-overlap checkpoint:", best_path, "Exception:", repr(e)
        )
        weights_loaded = False

if not weights_loaded:
    try:
        net.backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
        net.backbone.global_pool = GeM(flatten=True)
        net = net.to(device)
        net.eval()
        fallback_mode = "regressor_threshold"
        print(
            "No usable DR checkpoint loaded; using ImageNet-pretrained EfficientNet-B4 backbone fallback "
            "with regressor+thresholds inference."
        )
    except Exception as e:
        print("Failed to enable pretrained-backbone fallback:", repr(e))

print("device =", device)
print(
    "weights_loaded =",
    weights_loaded,
    "found_weights_path =",
    found_weights_path,
    "best_overlap_ratio =",
    best_ratio,
    "fallback_mode =",
    fallback_mode,
)




## === cell 5
class TestDS(Dataset):
    def __init__(self, ids, img_dir, tfm):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.tfm = tfm

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = f"{self.img_dir}/{idx}.png"
        img = Image.open(image_name).convert("RGB")
        img = self.tfm(img)
        return idx, img


pred_by_id = {}

ds = TestDS(test_ids, TEST_IMG_DIR, transform)
batch_size = 8 if device.startswith("cuda") else 2
dl = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=device.startswith("cuda"),
)

with torch.inference_mode():
    for batch_ids, batch_x in dl:
        batch_x = batch_x.to(device, non_blocking=True)

        if weights_loaded:
            out = net(batch_x, final=True)  # (B, 1) in [0, 4.5]
            out = out.data.squeeze(1)
            pred = regress2class(out)  # preserves original discretization thresholds
            pred = pred.to(torch.int64).cpu().numpy().tolist()
        else:
            c_out, r_out, o_out = net(batch_x, final=False)
            out = r_out.data.squeeze(1)  # already sigmoid*4.5 in forward(final=False)
            pred = regress2class(out).to(torch.int64).cpu().numpy().tolist()

        for _id, _p in zip(batch_ids, pred):
            pred_by_id[str(_id)] = int(_p)

missing_ids = [idx for idx in test_ids if str(idx) not in pred_by_id]
if missing_ids:
    for idx in missing_ids:
        pred_by_id[str(idx)] = 0
    print("Warning: filled missing predictions for", len(missing_ids), "ids")

submission_df = pd.DataFrame(
    {
        "id_code": test_df["id_code"].astype(str).values,
        "diagnosis": [
            int(pred_by_id[str(idx)]) for idx in test_df["id_code"].astype(str).values
        ],
    }
)



## === cell 6
df = submission_df.copy()
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

if df.empty:
    raise RuntimeError("Submission DataFrame is empty; check test.csv reading.")

if len(df) != len(test_df):
    raise RuntimeError(f"Row count mismatch: submission={len(df)} test={len(test_df)}")
if not (df["id_code"].values == test_df["id_code"].astype(str).values).all():
    raise RuntimeError(
        "id_code order mismatch vs test.csv; refusing to write invalid submission."
    )

df.to_csv("submission.csv", index=False)

print(df.head())
print(
    "Wrote submission.csv with",
    len(df),
    "rows. weights_loaded =",
    weights_loaded,
    "fallback_mode =",
    fallback_mode,
)
