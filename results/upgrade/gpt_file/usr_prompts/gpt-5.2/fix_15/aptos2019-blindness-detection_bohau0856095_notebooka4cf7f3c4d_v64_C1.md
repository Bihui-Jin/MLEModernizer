# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.9118531059271592

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) remove the failing `pip install` of an unavailable wheel and use the already-installed `timm`, (2) make device selection robust by falling back to CPU when no GPU/driver is available, and (3) fix the weight-loading failure by searching common input locations and safely running even if weights are missing. I also fix a couple of small logic bugs that can break inference (string comparison with `is`, and `trim()` returning `None`), and make the test-image path resolution robust to the dataset’s duplicate folder layouts. Finally, I ensure a non-empty `submission.csv` is always written with the exact required columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with the model running with randomly initialized weights (your log even warns the weight file is missing), which yields essentially random predictions. The smallest change that should move you strongly toward the target is to correctly locate and load the pretrained checkpoint from the competition dataset folder (it’s typically inside the input dataset tree, not at the input root). I expand the weight search to include the competition directory and common nested locations, and also robustly handle checkpoints saved as `{"state_dict": ...}` to avoid silent partial loads. This preserves your exact model/transform/inference logic and only fixes the missing-weights issue so performance improves toward the target.'
- What this solution (achieved 0.01221) has done: 'I fix the immediate failure by removing the hard stop when the checkpoint file isn’t found, so inference can run end-to-end and always write a valid `submission.csv`. To keep the core model/inference logic intact, I only add a robust fallback to EfficientNet’s built-in ImageNet pretrained weights when your competition checkpoint is missing; this is score-improving versus random weights while staying within the same architecture and forward path. I also make test-id extraction and submission building robust (ensure we iterate the `id_code` column, preserve order, and avoid shape issues that caused the row-mismatch assertion). Finally, I keep paths unchanged but make the directory discovery a bit more tolerant to the duplicated Kaggle folder layout.'
- What this solution (achieved 0.01221) has done: 'Your current score (0.01221) is far below the target (0.91185), and the biggest likely cause is that inference is effectively running without the intended competition-trained head/weights, so predictions are close to random even with an ImageNet backbone. I (1) fix weight discovery to actually find the checkpoint in common nested Kaggle input paths (including any `.pkl`/`.pth` variants) and (2) make weight loading robust to different checkpoint formats and key prefixes so the full model loads when available. If the competition checkpoint still truly isn’t present, I keep your current fallback (ImageNet backbone) to ensure a valid submission is produced. These changes preserve your model, transforms, and inference logic; they only ensure the correct weights are used so the score moves toward the target.'
- What this solution (achieved 0.07297) has done: 'Your current score is far below the target, and the code is still effectively not using the intended competition-trained final regressor because inference calls `net(img)` with `final=False` (default), while the checkpoint name suggests the trained output is the `final_regressor` path. I make the smallest possible change to run the same model but switch inference to `final=True`, then convert that single regression output to classes using your existing `regress2class` thresholds. To avoid the score collapsing due to mismatched thresholds, I also calibrate the 4 thresholds once on the training set using the loaded model (no training, just forward passes) by optimizing QWK over a small grid around your current thresholds; this keeps semantics the same (thresholding regression output) but aligns post-processing with the metric. These changes keep your architecture and weights intact and should move the score substantially toward the target while still writing a valid `submission.csv`.'
- What this solution (achieved 0.08566) has done: 'Your current gap to the target is large, so the most likely “minimal change with big impact” is fixing a subtle alignment bug during threshold calibration: when any train images are skipped/missing, your `y_true_aligned = y_true[:len(train_reg)]` no longer matches the predicted rows, which can produce badly-calibrated thresholds and tank QWK. I change calibration to keep (pred, label) aligned by collecting labels only for images that were actually predicted, without changing the model, transforms, or inference semantics. I also make threshold search slightly more robust (still the same coordinate-descent grid idea) by using a small second refinement pass around the best thresholds; this stays within the same thresholding core logic but should move score upward toward your target. Submission writing stays identical and still produces `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.11536) has done: 'Your score is still far below target, so we should make the smallest change that improves QWK without altering the model/training logic: calibrate the 4 thresholds in a way that directly optimizes QWK on the training set more reliably. I keep your exact model and regression→class conversion, but replace the coarse coordinate-descent grid with a monotonic threshold optimization using `scipy.optimize.minimize`-style logic is unavailable here, so instead we do a slightly stronger (yet still fast) iterative coordinate descent with more candidate points and a final “pairwise swap” check to escape shallow local optima. I also ensure calibration uses a fixed, stratified subset of the training set for speed/stability (still no training and no approximations in inference; just fewer calibration images), because full 3,295-image calibration can be noisy if some images fail to load. These changes only affect post-processing thresholds (evaluation-aligned), preserve your architecture and inference semantics, and still write a valid `submission.csv`.'
- What this solution (achieved 0.11438) has done: 'Your current score (0.115) is far below the target (0.912), so we need a real lift without changing the model/training core logic. The biggest safe lever left is threshold calibration quality: right now it uses a small subset and a local coordinate search that can get stuck, and it doesn’t enforce class-coverage constraints that matter for QWK. I (1) calibrate thresholds on the full training set predictions (still inference-only; no training), (2) keep the same regression→thresholding semantics but upgrade the search to a deterministic, monotonic coordinate descent with a short global “jitter” restart that remains fast, and (3) add a guard to avoid degenerate thresholds that collapse to too-few classes on the calibration predictions. This keeps your architecture/forward path identical and only improves metric-aligned post-processing so the score should move substantially toward the target band.'
- What this solution (achieved 0.11438) has done: 'Your current score is far below the target, and the most likely cause (given your pipeline already does “final=True” and threshold calibration) is that the correct competition-trained checkpoint still isn’t being loaded, so you’re effectively running an ImageNet backbone with random heads. I make the smallest change that strongly increases score: (1) broaden checkpoint discovery to include common nested filenames and search for any “B4_3stage” checkpoint, and (2) make loading robust to PyTorch Lightning-style keys and to shape-mismatched heads, so the backbone at least loads even if the head doesn’t. This preserves your exact model, transforms, and inference semantics; it only fixes weight-loading so predictions become meaningful. I also print a clear summary of how many tensors were actually loaded so we can confirm we’re not silently falling back again.'
- What this solution (achieved 0.11438) has done: 'Your current score is far below the target, so the smallest change likely to move QWK meaningfully upward is to ensure inference is actually using the intended competition-trained weights (right now it’s very likely falling back to ImageNet backbone + random heads, which performs near-random on this task). I tighten checkpoint discovery (search nested paths and any filename containing `B4_3stage`) and make loading robust to common checkpoint wrappers/prefixes while requiring that the *final_regressor* weights load (otherwise we keep searching), because your inference uses `final=True`. I also ensure the calibrated thresholds used by `regress2class()` are the same ones found during calibration (your current calibration updates `threshold`, but `regress2class` is driven by a global list—so we keep that but make it explicitly consistent). These are minimal changes that preserve your model architecture, forward path, transforms, and regression→threshold semantics, while fixing the main cause of low performance (wrong/missing checkpoint).'
- What this solution (achieved 0.11438) has done: 'Your current score is far below the target, so the most likely remaining issue is still that the intended competition checkpoint isn’t being loaded (or is being rejected by the “must include final_regressor weights” constraint), leaving you with random heads and near-random predictions. I make the smallest change that increases the chance of correctly loading *some* useful weights by (1) expanding checkpoint discovery to include common nested “weights/checkpoints/models” folders and more flexible filename matching, and (2) relaxing the “strict final_regressor required” rule to prefer the checkpoint that loads the most tensors (especially backbone), instead of rejecting partially-compatible checkpoints. This preserves your model/forward path and your regression→threshold semantics, and keeps your threshold calibration intact; it only changes how we find/select/load an existing checkpoint. If no checkpoint exists in the environment, the behavior remains the same as your current fallback and still writes a valid `submission.csv`.'
- What this solution (achieved 0.11438) has done: 'Your score is far below the target, so the most likely “minimal but high-impact” fix is that the model is still not actually loading the intended competition checkpoint (or not loading it correctly), making predictions effectively random. I keep your model/transform/inference logic unchanged, but (1) expand checkpoint discovery to also consider files inside the dataset’s zip/structure-friendly locations and *any* `.pth/.pt/.ckpt` whose state_dict contains `final_regressor` keys, and (2) make loading robust to Lightning-style `state_dict` keys and keep the checkpoint that loads the most tensors (with a strong preference for loading `final_regressor`). This should move the score substantially upward without changing the architecture or evaluation semantics. The rest of the pipeline (threshold calibration + submission writing) stays the same, aside from making sure the chosen checkpoint is actually applied to the same `net` instance used for calibration/inference.'

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
BASE_INPUT_CANDIDATES = [
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "../input",
    "/kaggle/input",
    "/kaggle/data/input/aptos2019-blindness-detection",
    "/kaggle/data/input",
]


def first_existing(path_list):
    for p in path_list:
        if p is not None and os.path.exists(p):
            return p
    return None


base_comp = first_existing(BASE_INPUT_CANDIDATES)
if base_comp is None:
    raise FileNotFoundError(
        "Could not locate Kaggle input directory for competition data."
    )

test_csv_candidates = [
    os.path.join(base_comp, "test.csv"),
    os.path.join(base_comp, "aptos2019-blindness-detection", "test.csv"),
]
test_csv_path = first_existing(test_csv_candidates)
if test_csv_path is None:
    raise FileNotFoundError("test.csv not found in expected locations.")

train_csv_candidates = [
    os.path.join(base_comp, "train.csv"),
    os.path.join(base_comp, "aptos2019-blindness-detection", "train.csv"),
]
train_csv_path = first_existing(train_csv_candidates)
if train_csv_path is None:
    raise FileNotFoundError("train.csv not found in expected locations.")

test_ids_df = pd.read_csv(test_csv_path)
if "id_code" not in test_ids_df.columns:
    raise KeyError(
        f"Expected column 'id_code' in test.csv, found: {list(test_ids_df.columns)}"
    )
test_ids = test_ids_df["id_code"].astype(str).tolist()

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


def _rebuild_heads_for_backbone(model: ThreeStage_Model):
    feat_dim = int(getattr(model.backbone, "num_features", 1000))
    model.classifier = nn.Sequential(
        nn.SiLU(),
        nn.Linear(feat_dim, 500),
        nn.SiLU(),
        nn.Linear(500, 5),
    )
    model.regressor = nn.Sequential(
        nn.SiLU(),
        nn.Linear(feat_dim, 500),
        nn.SiLU(),
        nn.Linear(500, 1),
    )
    model.ordinal = nn.Sequential(
        nn.SiLU(),
        nn.Linear(feat_dim, 500),
        nn.SiLU(),
        nn.Linear(500, 4),
    )
    model.final_regressor = nn.Sequential(
        nn.SiLU(),
        nn.Linear(10, 1),
    )
    return feat_dim


_ = _rebuild_heads_for_backbone(net)

weight_base_name = "B4_3stage_3epoch_finetune"

weight_search_roots = [
    base_comp,
    os.path.join(base_comp, "aptos2019-blindness-detection"),
    "../input",
    "/kaggle/input",
    "/kaggle/data/input",
]
extra_subdirs = [
    "",
    "weights",
    "weight",
    "checkpoints",
    "checkpoint",
    "models",
    "model",
    "ckpt",
    "output",
    "outputs",
    "train",
    "trained",
]
expanded_roots = []
for r in weight_search_roots:
    for sd in extra_subdirs:
        expanded_roots.append(os.path.join(r, sd) if sd else r)
weight_search_roots = expanded_roots

exact_name_candidates = [
    weight_base_name + ext for ext in [".pkl", ".pth", ".pt", ".bin", ".ckpt"]
]
fuzzy_tokens = [
    "b4_3stage",
    "3stage",
    "b4",
    "3epoch",
    "3_epoch",
    "finetune",
    "fine_tune",
    "3stage_model",
]


def _extract_state_dict(obj):
    if not isinstance(obj, dict):
        return obj
    for k in [
        "state_dict",
        "model_state_dict",
        "model",
        "net",
        "weights",
        "ema_state_dict",
    ]:
        if k in obj and isinstance(obj[k], dict):
            return obj[k]
    for _, v in obj.items():
        if isinstance(v, dict):
            for kk in ["state_dict", "model_state_dict", "model", "net"]:
                if kk in v and isinstance(v[kk], dict):
                    return v[kk]
    return obj


def _strip_prefixes(state):
    if not isinstance(state, dict):
        return state
    prefixes = ["module.", "model.", "net."]
    out = state
    changed = True
    while changed:
        changed = False
        for pfx in prefixes:
            if any(k.startswith(pfx) for k in out.keys()):
                out = {k[len(pfx) :]: v for k, v in out.items() if k.startswith(pfx)}
                changed = True
    return out


def _load_state_dict_forgiving(model, state):
    model_sd = model.state_dict()
    filtered = {}
    matched, shape_mismatch, missing_in_model = 0, 0, 0
    for k, v in state.items():
        if k not in model_sd:
            missing_in_model += 1
            continue
        if tuple(model_sd[k].shape) != tuple(v.shape):
            shape_mismatch += 1
            continue
        filtered[k] = v
        matched += 1
    missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)
    return {
        "matched": matched,
        "shape_mismatch": shape_mismatch,
        "missing_in_model": missing_in_model,
        "missing_keys_after": len(missing_keys),
        "unexpected_keys_after": len(unexpected_keys),
        "filtered_keys": set(filtered.keys()),
    }


def _checkpoint_candidates():
    cands = []

    for root in weight_search_roots:
        for wn in exact_name_candidates:
            p = os.path.join(root, wn)
            if os.path.exists(p):
                cands.append(p)

    for root in weight_search_roots:
        if not os.path.exists(root):
            continue
        for ext in ["*.pkl", "*.pth", "*.pt", "*.bin", "*.ckpt"]:
            for p in glob.glob(os.path.join(root, "**", ext), recursive=True):
                bn = os.path.basename(p).lower()
                if (
                    ("b4" in bn)
                    and ("stage" in bn)
                    and any(tok in bn for tok in fuzzy_tokens)
                ):
                    cands.append(p)

    for root in weight_search_roots:
        if not os.path.exists(root):
            continue
        for ext in ["*.pth", "*.pt", "*.ckpt"]:
            for p in glob.glob(os.path.join(root, "**", ext), recursive=True):
                cands.append(p)

    seen = set()
    out = []
    for p in cands:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


def _has_final_regressor_keys(state_dict):
    if not isinstance(state_dict, dict):
        return False
    keys = set(state_dict.keys())
    return ("final_regressor.1.weight" in keys) or any(
        k.startswith("final_regressor.") for k in keys
    )


def _try_load_checkpoint_ranked(model, ckpt_path):
    before = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    try:
        raw = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        return False, {
            "reason": f"torch_load_failed: {type(e).__name__}",
            "score": -1,
            "matched": 0,
            "loaded_final": False,
        }

    state = _strip_prefixes(_extract_state_dict(raw))
    if not isinstance(state, dict):
        return False, {
            "reason": "state_not_dict",
            "score": -1,
            "matched": 0,
            "loaded_final": False,
        }

    loaded_final = _has_final_regressor_keys(state)
    stats = _load_state_dict_forgiving(model, state)

    after = model.state_dict()
    changed_any = any(
        (k in after) and (not torch.equal(before[k], after[k].detach().cpu()))
        for k in before.keys()
    )

    score = int(stats["matched"]) + (5000 if loaded_final else 0)
    ok = bool(changed_any and stats["matched"] > 0)

    stats_out = {
        "ok": ok,
        "score": score,
        "loaded_final": bool(
            loaded_final
            and (
                {"final_regressor.1.weight", "final_regressor.1.bias"}.issubset(
                    stats["filtered_keys"]
                )
                or any(k.startswith("final_regressor.") for k in stats["filtered_keys"])
            )
        ),
        "changed_any": changed_any,
        "matched": stats["matched"],
        "shape_mismatch": stats["shape_mismatch"],
        "missing_in_model": stats["missing_in_model"],
        "missing_keys_after": stats["missing_keys_after"],
        "unexpected_keys_after": stats["unexpected_keys_after"],
    }
    return ok, stats_out


best = None  # (score, path, stats)
for cand in _checkpoint_candidates():
    ok, st = _try_load_checkpoint_ranked(net, cand)
    if st.get("score", -1) > 0:
        print(f"Checkpoint try: {cand} -> {st}")
    if not ok:
        continue
    cur = (st["score"], cand, st)
    if best is None or cur[0] > best[0]:
        best = cur

if best is not None:
    chosen_weight_path = best[1]
    chosen_stats = best[2]
    print(f"Using checkpoint: {chosen_weight_path}")
    print("Checkpoint stats:", chosen_stats)
else:
    print(
        "WARNING: Could not find/load any compatible checkpoint; using ImageNet-pretrained EfficientNet-B4 backbone + rebuilt heads."
    )
    net.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
    net.backbone.global_pool = GeM(flatten=True)
    feat_dim = _rebuild_heads_for_backbone(net)
    print(f"Backbone num_features={feat_dim}. Heads rebuilt to match feature dim.")

net = net.to(device)
net.eval()

test_img_dir_candidates = [
    os.path.join(os.path.dirname(test_csv_path), "test_images"),
    os.path.join(base_comp, "test_images"),
    os.path.join(base_comp, "aptos2019-blindness-detection", "test_images"),
]
test_img_dir = first_existing(test_img_dir_candidates)
if test_img_dir is None:
    raise FileNotFoundError("test_images directory not found in expected locations.")

train_img_dir_candidates = [
    os.path.join(os.path.dirname(train_csv_path), "train_images"),
    os.path.join(base_comp, "train_images"),
    os.path.join(base_comp, "aptos2019-blindness-detection", "train_images"),
]
train_img_dir = first_existing(train_img_dir_candidates)
if train_img_dir is None:
    raise FileNotFoundError("train_images directory not found in expected locations.")



## === cell 5
train_df = pd.read_csv(train_csv_path)
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["diagnosis"] = train_df["diagnosis"].astype(int)


def _predict_train_regression_and_labels(df, batch_size=10, max_items=None):
    preds = []
    labels = []
    ids = df["id_code"].tolist()
    ys = df["diagnosis"].to_numpy()
    use_n = len(ids) if max_items is None else min(len(ids), int(max_items))

    with torch.inference_mode():
        for start in range(0, use_n, batch_size):
            batch_ids = ids[start : start + batch_size]
            batch_y = ys[start : start + batch_size]
            imgs = []
            ok_y = []
            for idx, y in zip(batch_ids, batch_y):
                image_name = os.path.join(train_img_dir, f"{idx}.png")
                if not os.path.exists(image_name):
                    alt = os.path.join(train_img_dir, "train_images", f"{idx}.png")
                    if os.path.exists(alt):
                        image_name = alt
                    else:
                        continue
                try:
                    img = Image.open(image_name).convert("RGB")
                except Exception:
                    continue
                img = transform(img)
                imgs.append(img)
                ok_y.append(int(y))

            if len(imgs) == 0:
                continue

            xb = torch.stack(imgs, dim=0).to(device)
            out = net(xb, final=True).view(-1).detach().cpu().numpy()
            preds.extend(out.tolist())
            labels.extend(ok_y)

    return np.asarray(preds, dtype=np.float32), np.asarray(labels, dtype=np.int64)


t0 = time.time()
train_reg, y_true_aligned = _predict_train_regression_and_labels(
    train_df, batch_size=10, max_items=None
)
if len(train_reg) == 0:
    raise RuntimeError(
        "No training predictions were produced; cannot calibrate thresholds."
    )
print(
    f"Calibration pairs (full train): {len(train_reg)} / {len(train_df)}  (took {time.time()-t0:.1f}s)"
)


def _apply_thresholds(pred_cont, thr):
    thr = list(thr)
    pred_cls = np.zeros_like(pred_cont, dtype=np.int64)
    for t in thr:
        pred_cls += (pred_cont >= t).astype(np.int64)
    pred_cls = np.clip(pred_cls, 0, 4)
    return pred_cls


def _qwk(pred_cont, y_true, thr):
    return cohen_kappa_score(
        y_true, _apply_thresholds(pred_cont, thr), weights="quadratic"
    )


def _class_coverage_ok(pred_cont, thr, min_classes=4):
    pred_cls = _apply_thresholds(pred_cont, thr)
    return len(np.unique(pred_cls)) >= int(min_classes)


def _calibrate_thresholds_monotonic_cd(
    pred_cont, y_true, base_thr, deltas, n_passes=3, min_classes=4
):
    cur = np.array(base_thr, dtype=np.float32).copy()
    best = _qwk(pred_cont, y_true, cur)

    for _ in range(int(n_passes)):
        for j in range(4):
            local_best_thr = cur.copy()
            local_best = best

            lo = -1e9 if j == 0 else (cur[j - 1] + 1e-4 - cur[j])
            hi = 1e9 if j == 3 else (cur[j + 1] - 1e-4 - cur[j])

            for d in deltas:
                if d < lo or d > hi:
                    continue
                trial = cur.copy()
                trial[j] = float(trial[j] + d)
                if not (trial[0] < trial[1] < trial[2] < trial[3]):
                    continue
                if not _class_coverage_ok(pred_cont, trial, min_classes=min_classes):
                    continue
                k = _qwk(pred_cont, y_true, trial)
                if k > local_best + 1e-12:
                    local_best = k
                    local_best_thr = trial

            cur = local_best_thr
            best = local_best

    return cur, float(best)


base_thr = np.array(threshold, dtype=np.float32)

deltas1 = np.array(
    [-0.60, -0.45, -0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30, 0.45, 0.60],
    dtype=np.float32,
)
thr1, k1 = _calibrate_thresholds_monotonic_cd(
    train_reg, y_true_aligned, base_thr, deltas1, n_passes=2
)

deltas2 = np.array(
    [
        -0.18,
        -0.12,
        -0.08,
        -0.05,
        -0.03,
        -0.02,
        -0.01,
        0.0,
        0.01,
        0.02,
        0.03,
        0.05,
        0.08,
        0.12,
        0.18,
    ],
    dtype=np.float32,
)
thr2, k2 = _calibrate_thresholds_monotonic_cd(
    train_reg, y_true_aligned, thr1, deltas2, n_passes=3
)

deltas3 = np.array(
    [-0.06, -0.04, -0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03, 0.04, 0.06],
    dtype=np.float32,
)
thr3, k3 = _calibrate_thresholds_monotonic_cd(
    train_reg, y_true_aligned, thr2, deltas3, n_passes=3
)

best_thr = thr3
best_k = k3

rng = np.random.RandomState(42)
for jitter_scale in [0.06, 0.03]:
    for _ in range(3):
        jit = rng.uniform(-jitter_scale, jitter_scale, size=4).astype(np.float32)
        trial0 = best_thr + jit
        trial0 = np.sort(trial0)
        trial0[1] = max(trial0[1], trial0[0] + 1e-3)
        trial0[2] = max(trial0[2], trial0[1] + 1e-3)
        trial0[3] = max(trial0[3], trial0[2] + 1e-3)

        thrj, kj = _calibrate_thresholds_monotonic_cd(
            train_reg, y_true_aligned, trial0, deltas3, n_passes=2
        )
        if kj > best_k + 1e-12:
            best_k = kj
            best_thr = thrj

threshold = [float(x) for x in best_thr.tolist()]
print("Calibrated thresholds:", threshold)
print("Train QWK on full-train calibration:", float(best_k))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2967660122.py in <cell line: 0>()
     45 
     46 t0 = time.time()
---> 47 train_reg, y_true_aligned = _predict_train_regression_and_labels(
     48     train_df, batch_size=10, max_items=None
     49 )

/tmp/ipykernel_55/2967660122.py in _predict_train_regression_and_labels(df, batch_size, max_items)
     37 
     38             xb = torch.stack(imgs, dim=0).to(device)
---> 39             out = net(xb, final=True).view(-1).detach().cpu().numpy()
     40             preds.extend(out.tolist())
     41             labels.extend(ok_y)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/3027555046.py in forward(self, x, final)
     78         x = self.backbone(x)
     79 
---> 80         c_out = self.classifier(x)
     81         r_out = self.regressor(x)
     82         o_out = self.ordinal(x)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: mat1 and mat2 shapes cannot be multiplied (10x1000 and 1792x500)

## === cell 6
submission_rows = []
with torch.inference_mode():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print(i)

        image_name = os.path.join(test_img_dir, f"{idx}.png")
        if not os.path.exists(image_name):
            alt = os.path.join(test_img_dir, "test_images", f"{idx}.png")
            if os.path.exists(alt):
                image_name = alt
            else:
                raise FileNotFoundError(f"Missing test image: {image_name}")

        img = Image.open(image_name).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        out = net(img, final=True)  # (1,1) regression in [0, 4.5]
        out = out.view(-1)  # (1,)
        pred = regress2class(out)
        submission_rows.append([idx, int(pred.item())])

df = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

assert len(df) == len(
    test_ids
), f"Submission rows mismatch with test set: {len(df)} vs {len(test_ids)}"

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with", len(df), "rows")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/590030984.py in <cell line: 0>()
     16         img = transform(img).unsqueeze(0).to(device)
     17 
---> 18         out = net(img, final=True)  # (1,1) regression in [0, 4.5]
     19         out = out.view(-1)  # (1,)
     20         pred = regress2class(out)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/3027555046.py in forward(self, x, final)
     78         x = self.backbone(x)
     79 
---> 80         c_out = self.classifier(x)
     81         r_out = self.regressor(x)
     82         o_out = self.ordinal(x)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py in forward(self, input)
    123 
    124     def forward(self, input: Tensor) -> Tensor:
--> 125         return F.linear(input, self.weight, self.bias)
    126 
    127     def extra_repr(self) -> str:

RuntimeError: mat1 and mat2 shapes cannot be multiplied (1x1000 and 1792x500)
