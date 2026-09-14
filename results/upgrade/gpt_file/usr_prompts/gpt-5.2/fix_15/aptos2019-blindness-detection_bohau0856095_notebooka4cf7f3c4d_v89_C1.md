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

0.9235870138161588

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the pipeline so it runs end-to-end on Kaggle CPU-only environments by (1) selecting `cuda` only if available and otherwise using `cpu`, (2) removing the failing external `pip install` and instead using the already-installed `timm`, and (3) robustly locating the model weights file (or cleanly falling back to an untrained model so a non-empty valid `submission.csv` is always produced). I also fix a couple of small but real transform bugs (`is` vs `==` and `trim()` returning `None`) that can crash inference. Finally, I ensure the submission is written with the exact required columns and that predictions are aligned to `test.csv` order.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with “random/untrained model” predictions because the script often can’t find the pretrained weights (so it falls back to random init). The smallest change that should move the score sharply upward toward your 0.9236 target is to robustly locate the weights inside the competition dataset (including `/kaggle/input/aptos2019-blindness-detection/` and any `.pth/.pt/.pkl` matches) and load them reliably. I also switch inference to a DataLoader (same transforms/model/thresholding) so it finishes within the 600s limit without changing evaluation semantics. Finally, I force submission ordering to exactly match `test.csv` to avoid any accidental misalignment.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission dominated by a single class (often caused by using the regression head + fixed thresholds, or by silently failing to load the intended weights). To move the score upward toward your 0.9236 target with minimal semantic change, I keep your exact model and transforms but (1) make weight discovery more robust by also scanning for any `.pkl/.pth/.pt` in `/kaggle/input/**` and handling common checkpoint key prefixes, and (2) switch prediction to use the classifier logits (`argmax`) when weights are present (while keeping your original regression-threshold path as a fallback if weights are missing). This preserves the core architecture/training (none is added) and only changes the final decoding to something more stable for QWK when using a trained 3-head model. The script still runs end-to-end and writes a valid `submission.csv` aligned to `test.csv` order.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is overwhelmingly consistent with either (a) the pretrained weights not actually being loaded (so predictions are effectively random/degenerate), or (b) weights loading partially/wrongly (e.g., checkpoint contains a nested key like `model_state_dict`) while the script still sets `weights_loaded=True`. I make the smallest targeted change: improve checkpoint parsing and only mark `weights_loaded=True` if we successfully load a plausible state dict into the model (with basic sanity checks on key shapes/prefixes). This keeps your exact model/transform/inference logic, but makes it far more likely you’re using the intended trained weights—moving score up toward 0.9236 without changing evaluation semantics. The rest of the pipeline and submission formatting/order remain unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the model is still running effectively untrained/degenerate (most often because the checkpoint that gets found is not the intended one, or it loads the wrong object so the backbone/classifier stay random). I make the smallest targeted change to weight discovery and loading: restrict the recursive scan to checkpoints whose filenames indicate they belong to this model/competition (avoid accidentally grabbing unrelated `.pth/.pt` from other datasets), and strengthen the “did we really load?” sanity check by verifying that a key tensor (classifier final layer) matches in shape and actually changed from its random initialization. This preserves your exact model, transforms, and prediction decoding, but makes it far more likely you’re actually using the trained weights—moving QWK sharply upward toward your 0.9236 target. The pipeline still runs end-to-end and writes a valid `submission.csv` aligned to `test.csv` order.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model still running effectively untrained because the intended checkpoint isn’t being found/loaded, so the smallest change is to (1) robustly point to the real Kaggle dataset location (`/kaggle/input/...`) and (2) tighten weight discovery to only scan plausible checkpoint filenames instead of grabbing unrelated `.pth/.pt/.pkl` files. I also make the checkpoint loader handle common serialization formats (`state_dict`, `model_state_dict`, and Lightning-style `state_dict`) and only mark `weights_loaded=True` after a stronger compatibility check, which should move QWK upward toward your 0.9236 target without changing the model, transforms, or inference decoding. Finally, I keep the submission ordering aligned to `test.csv` and ensure the CSV is always written.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the checkpoint not actually being loaded (or being loaded in a way that leaves the model effectively random), so the smallest score-moving change is to (1) broaden weight discovery to include `/kaggle/input/**` but still prioritize the exact expected filename, and (2) make the checkpoint parser handle more common nested formats (including Lightning `state_dict` with extra prefixes) while only setting `weights_loaded=True` when a backbone tensor clearly changes. I keep your exact model, transforms, and inference decoding; the only behavioral change is reliably using the trained weights when they exist. I also keep submission alignment to `test.csv` order and still always write a valid `submission.csv`. These changes should move QWK sharply upward toward your 0.9236 target without altering core logic.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with `weights_loaded` effectively being false (or loading a wrong/incompatible checkpoint), which makes predictions degenerate/random; the smallest score-improving step is to make checkpoint discovery/selection and state-dict extraction more robust without changing the model or inference logic. I (1) broaden the search to also include `/kaggle/working/**` (some notebooks copy weights there) and (2) improve checkpoint parsing by handling common nested formats and key prefixes (including `backbone.model.*` / `encoder.*`) while keeping strict=False. I also fix a subtle bug in `regress2class` (it was constructing CPU tensors without an explicit dtype/shape and can behave oddly) but keep the same thresholds/semantics. Everything else (model architecture, transforms, argmax-vs-regression fallback, submission alignment) stays the same and it still write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the run producing effectively untrained/degenerate predictions because the intended checkpoint is not being found/loaded (or is loaded but doesn’t actually match the model). I make the smallest score-moving change: broaden checkpoint discovery within the competition dataset directory and strengthen state-dict extraction to handle more real-world checkpoint wrappers (including nested `model` objects and `OrderedDict`-like formats), while only setting `weights_loaded=True` when we can verify a meaningful backbone weight actually changed. I also keep your exact model and transforms, but switch the “weights_loaded” prediction decode to use the model’s `final=True` regressor (your intended fused head) with the same `regress2class` thresholds, so the output stays consistent with your regression-to-class semantics and typically improves QWK over raw classifier argmax when the checkpoint was trained for the fused regressor. The pipeline still run end-to-end and write a correctly ordered `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with “weights not actually being used” (or being loaded but effectively not applied), which makes predictions degenerate/random. I keep your exact model and decoding (final regressor → `regress2class`), but make weight discovery more likely to find the checkpoint by also scanning common weight filetypes (including `.ckpt`) and Kaggle working/output locations without changing any modeling semantics. I also strengthen checkpoint extraction to handle more wrappers (e.g., `{"model": {"state_dict": ...}}`, PyTorch Lightning `.ckpt`, etc.) and only set `weights_loaded=True` if we can verify a key backbone tensor both matches shape and changes after loading. These are minimal, score-relevant changes intended to move QWK up toward your target by ensuring the intended pretrained weights are actually applied at inference.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the checkpoint either not being found or being found but not actually loading into `ThreeStage_Model` (so predictions collapse to near-random/degenerate). I keep your exact model, transforms, and regression→threshold decoding, but make checkpoint selection/load stricter and more compatible by (1) prioritizing checkpoints inside this competition folder and (2) improving key-prefix handling (including Lightning-style `backbone.*` vs full-model keys) while verifying that multiple tensors (not just one) actually changed after loading. If a checkpoint is present but only partially matches, we still load what matches (as you already do with `strict=False`) but we won’t mistakenly set `weights_loaded=True` unless the load is genuinely effective—this should move QWK sharply upward toward your 0.9236 target without changing evaluation semantics. Everything else remains the same and it still writes a valid `submission.csv` aligned to `test.csv` order.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with “weights not actually loaded (or loaded but ineffective)”, which collapses predictions and tanks QWK. I make the smallest score-moving change by (1) expanding checkpoint discovery to also include the common Kaggle “dataset root” `/kaggle/input/**` plus the competition subfolder that appears duplicated in your file tree, and (2) improving checkpoint key normalization to handle Lightning-style `model.*`/`backbone.*` nesting without changing your model or decoding. I also force deterministic eval-time behavior (no semantic change) and slightly increase CPU DataLoader workers within safe bounds to ensure the full test set finishes reliably within the timeout. Everything else (model architecture, transforms, `final=True` regressor + `regress2class` thresholds, submission schema/order) stays the same.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the run *not actually using a compatible pretrained checkpoint*, so predictions collapse and QWK tanks. I keep your exact model + transforms + regression→threshold decoding, but make the checkpoint discovery/selection more likely to pick the correct file by (1) scanning the competition input tree for any `.ckpt/.pth/.pt/.pkl` and (2) ranking candidates by filename relevance and key-overlap to your model (so we don’t accidentally “load” an unrelated checkpoint). I also strengthen state-dict extraction to handle more real-world wrappers and only set `weights_loaded=True` when overlap and tensor-shape compatibility are genuinely high. This should move the score sharply upward toward your 0.9236 target while preserving your core inference semantics and still writing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the checkpoint selection logic choosing an incompatible file (or none), so the model runs effectively untrained and predictions collapse. I make the smallest score-moving changes by (1) tightening checkpoint discovery to only search within this competition dataset tree (to avoid accidentally grabbing unrelated weights from other inputs), and (2) fixing the checkpoint ranking bug where the tuple sort currently prioritizes *low* key-overlap due to using `-overlap` as the first key. I also make `torch.load(..., weights_only=True)` the first attempt (when supported) to robustly load state dicts from newer PyTorch checkpoints without executing arbitrary pickled objects, falling back to the old behavior if needed. Everything else (model, transforms, final-regressor + `regress2class` decoding, submission format/order) stays the same.'

# 9. Code solution

## === cell 0
import os
import random
import time
import math
import glob
import re
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

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    out = out.view(-1).detach().cpu().to(torch.float32)
    prediction = torch.zeros(out.size(0), dtype=torch.int64)
    for i in range(4):
        prediction += (out >= float(threshold[i])).to(torch.int64)
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
TEST_CSV_PATH = "/kaggle/input/aptos2019-blindness-detection/test.csv"
TEST_IMG_DIR = "/kaggle/input/aptos2019-blindness-detection/test_images"

test_df = pd.read_csv(TEST_CSV_PATH)
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

net = ThreeStage_Model()

allowed_ext = (".pkl", ".pth", ".pt", ".ckpt")

candidate_weight_paths = [
    "/kaggle/input/aptos2019-blindness-detection/B4_3stage_2epoch_finetune3.pkl",
    "/kaggle/input/aptos2019-blindness-detection/B4_3stage_2epoch_finetune3.pth",
    "/kaggle/input/aptos2019-blindness-detection/B4_3stage_2epoch_finetune3.pt",
    "/kaggle/input/aptos2019-blindness-detection/B4_3stage_2epoch_finetune3.ckpt",
    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/B4_3stage_2epoch_finetune3.pkl",
    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/B4_3stage_2epoch_finetune3.pth",
    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/B4_3stage_2epoch_finetune3.pt",
    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/B4_3stage_2epoch_finetune3.ckpt",
    "/kaggle/working/B4_3stage_2epoch_finetune3.pkl",
    "/kaggle/working/B4_3stage_2epoch_finetune3.pth",
    "/kaggle/working/B4_3stage_2epoch_finetune3.pt",
    "/kaggle/working/B4_3stage_2epoch_finetune3.ckpt",
]

patterns = [
    "/kaggle/input/aptos2019-blindness-detection/**/B4_3stage_2epoch_finetune3.*",
    "/kaggle/input/aptos2019-blindness-detection/**/b4_3stage_2epoch_finetune3.*",
    "/kaggle/input/aptos2019-blindness-detection/**/*3stage*finetune*.*",
    "/kaggle/input/aptos2019-blindness-detection/**/*3stage*.*",
    "/kaggle/working/**/*3stage*.*",
    "/kaggle/output/**/*3stage*.*",
]

for pat in patterns:
    for p in glob.glob(pat, recursive=True):
        if isinstance(p, str) and p.lower().endswith(allowed_ext):
            candidate_weight_paths.append(p)

existing = [
    p for p in candidate_weight_paths if isinstance(p, str) and os.path.exists(p)
]
seen = set()
existing = [p for p in existing if not (p in seen or seen.add(p))]


def _is_plausible_ckpt(path: str) -> bool:
    bn = os.path.basename(path).lower()
    if not bn.endswith(allowed_ext):
        return False
    keywords = [
        "3stage",
        "finetune",
        "aptos",
        "blindness",
        "dr",
        "retina",
        "b4",
        "efficientnet",
    ]
    return any(k in bn for k in keywords)


existing = [p for p in existing if _is_plausible_ckpt(p)]


def _path_priority(p: str) -> int:
    pl = p.lower()
    if (
        "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/"
        in pl
    ):
        return 0
    if "/kaggle/input/aptos2019-blindness-detection/" in pl:
        return 1
    if "/kaggle/input/" in pl:
        return 2
    if "/kaggle/working/" in pl:
        return 3
    if "/kaggle/output/" in pl:
        return 4
    return 5


def _torch_load_any(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        priority_keys = [
            "state_dict",  # Lightning
            "model_state_dict",  # common
            "ema_state_dict",
            "weights",
            "params",
            "net",
            "model",
            "module",
            "student",
            "teacher",
            "model_ema",
            "model_state",
        ]
        for key in priority_keys:
            if key in ckpt_obj:
                nested = _extract_state_dict(ckpt_obj[key])
                if nested is not None:
                    return nested

        if len(ckpt_obj) > 0 and any(torch.is_tensor(v) for v in ckpt_obj.values()):
            return ckpt_obj

    if isinstance(ckpt_obj, nn.Module):
        return ckpt_obj.state_dict()

    return None


def _normalize_state_dict_keys(sd, target_keys_set):
    def strip_prefixes(inp_sd, prefixes):
        out = {}
        for k, v in inp_sd.items():
            nk = k
            for pref in prefixes:
                if nk.startswith(pref):
                    nk = nk[len(pref) :]
            out[nk] = v
        return out

    candidate_prefix_sets = [
        ("module.", "model.", "net."),
        ("state_dict.", "module.", "model.", "net."),
        ("module.model.", "model.model.", "net.model."),
        ("module.backbone.", "model.backbone.", "net.backbone.", "backbone."),
        (
            "module.backbone.model.",
            "model.backbone.model.",
            "net.backbone.model.",
            "backbone.model.",
        ),
        ("encoder.", "module.encoder.", "model.encoder.", "net.encoder."),
        (
            "backbone.model.",
            "module.backbone.model.",
            "model.backbone.model.",
            "net.backbone.model.",
        ),
    ]

    cands = [strip_prefixes(sd, prefixes=pfx) for pfx in candidate_prefix_sets]

    best_sd = None
    best_overlap = -1
    for cand in cands:
        overlap = sum((k in target_keys_set) for k in cand.keys())
        if overlap > best_overlap:
            best_overlap = overlap
            best_sd = cand
    return best_sd, best_overlap


def _score_candidate_ckpt(path: str, net_sd_keys_set: set) -> tuple:
    """
    Lower is better (we sort ascending).
    Primary goal: pick a checkpoint that actually matches the model to avoid ineffective loads => 0.0 QWK.
    """
    bn = os.path.basename(path).lower()
    name_bonus = 0
    if bn.startswith("b4_3stage_2epoch_finetune3"):
        name_bonus -= 50
    for kw in ["b4", "3stage", "finetune", "aptos", "blindness", "efficientnet"]:
        if kw in bn:
            name_bonus -= 2

    try:
        ckpt = _torch_load_any(path)
        state = _extract_state_dict(ckpt)
        if state is None or not isinstance(state, dict):
            return (
                10_000,
                _path_priority(path),
                -os.path.getmtime(path),
                name_bonus,
            )
        _, overlap = _normalize_state_dict_keys(state, net_sd_keys_set)

        return (
            -overlap * 0
            + (10_000 - overlap),  # monotone: higher overlap => smaller score
            _path_priority(path),
            -os.path.getmtime(path),
            name_bonus,
        )
    except Exception:
        try:
            mt = -os.path.getmtime(path)
        except Exception:
            mt = 0
        return (20_000, _path_priority(path), mt, name_bonus)


net_sd = net.state_dict()
net_keys_set = set(net_sd.keys())

weight_path = None
if len(existing) > 0:
    ranked = sorted(existing, key=lambda p: _score_candidate_ckpt(p, net_keys_set))
    weight_path = ranked[0]

weights_loaded = False
if weight_path is None:
    print(
        "WARNING: pretrained weights not found; running with randomly initialized model (likely low score)."
    )
else:
    print("Selected weight_path:", weight_path)
    ckpt = _torch_load_any(weight_path)
    state = _extract_state_dict(ckpt)
    if state is None:
        print(
            "WARNING: Could not find a valid state_dict inside checkpoint; running untrained (likely low score)."
        )
    else:
        target_keys_set = set(net_sd.keys())
        state_norm, overlap = _normalize_state_dict_keys(state, target_keys_set)
        state = state_norm

        overlap_ratio = float(overlap) / float(max(1, len(target_keys_set)))

        ref_keys = [
            "backbone.conv_stem.weight",
            "backbone.bn1.weight",
            "classifier.1.weight",
            "regressor.1.weight",
            "ordinal.1.weight",
            "final_regressor.1.weight",
        ]
        ref_keys = [
            k
            for k in ref_keys
            if k in net_sd
            and k in state
            and torch.is_tensor(state[k])
            and tuple(net_sd[k].shape) == tuple(state[k].shape)
        ]
        ref_before = {k: net_sd[k].detach().cpu().clone() for k in ref_keys}

        missing, unexpected = net.load_state_dict(state, strict=False)
        total_params = len(net_sd)

        changed = 0
        for k in ref_keys:
            after = net.state_dict()[k].detach().cpu()
            if not torch.allclose(ref_before[k], after):
                changed += 1

        effective_change = (changed >= 2) or (len(ref_keys) == 1 and changed == 1)
        too_many_missing = len(missing) >= int(0.98 * total_params)

        if (
            (overlap <= 0)
            or (overlap_ratio < 0.10)
            or too_many_missing
            or (not effective_change)
        ):
            print(
                f"WARNING: Checkpoint appears incompatible/ineffective "
                f"(key_overlap={overlap}, overlap_ratio={overlap_ratio:.3f}, missing {len(missing)}/{total_params}, "
                f"ref_keys={len(ref_keys)}, changed={changed}); treating as not loaded."
            )
            weights_loaded = False
        else:
            weights_loaded = True
            if missing or unexpected:
                print(
                    f"NOTE: load_state_dict strict=False; missing={len(missing)} unexpected={len(unexpected)} key_overlap={overlap} overlap_ratio={overlap_ratio:.3f}"
                )

net = net.to(device)
net.eval()




## === cell 5
class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        img = self.transform(img)
        return idx, img


batch_size = 8 if device == "cuda" else 4
num_workers = 2 if device == "cuda" else min(4, max(1, (os.cpu_count() or 2) // 2))

ds = TestDataset(test_ids, TEST_IMG_DIR, transform)
dl = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
)

submission_rows = []
with torch.no_grad():
    for batch_ids, batch_imgs in dl:
        batch_imgs = batch_imgs.to(device, non_blocking=True)

        if weights_loaded:
            out_final = net(batch_imgs, final=True)
            pred = regress2class(out_final.data.squeeze(1))
        else:
            c_out, r_out, _ = net(batch_imgs)
            pred = regress2class(r_out.data.squeeze(1))

        for idx, p in zip(batch_ids, pred.tolist()):
            submission_rows.append([str(idx), int(p)])

submission = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])
submission["id_code"] = submission["id_code"].astype(str)
submission["diagnosis"] = submission["diagnosis"].astype(int)

submission = submission.set_index("id_code").loc[test_ids].reset_index()

assert len(submission) == len(test_ids), "Submission row count mismatch."
assert list(submission.columns) == ["id_code", "diagnosis"]



## === cell 6
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
print("diagnosis value counts:\n", submission["diagnosis"].value_counts().sort_index())
print("weights_loaded:", weights_loaded)
if weight_path is not None:
    print("weight_path:", weight_path)
