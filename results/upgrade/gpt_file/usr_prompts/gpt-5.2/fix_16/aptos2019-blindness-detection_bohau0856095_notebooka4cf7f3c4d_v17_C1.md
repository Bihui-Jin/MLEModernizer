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

0.8636653723796762

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved -0.11541) has done: 'I fix the missing weight file crash by making the code automatically pick an available weight file if the hardcoded path doesn’t exist, and otherwise fall back to running the backbone with `pretrained=True` so the pipeline always completes. I also make the device selection robust (CPU fallback) and load checkpoints with `map_location` so it works regardless of GPU availability. Finally, I add a small safety clamp/round so predictions are guaranteed to be valid integer classes 0–4 and always write a correctly formatted `submission.csv`.'
- What this solution (achieved 0.00807) has done: 'Your negative kappa strongly suggests the current predictions are miscalibrated for the competition’s ordinal metric (and likely using the wrong head: rounding the regression output directly is usually suboptimal for QWK). I keep your exact model and weights-loading behavior, but change only the prediction post-processing to use the model’s ordinal outputs (4 probabilities) and convert them into classes via your existing `regress2class` thresholds, which is the intended mapping for ordinal/QWK. I also fix a small scaling inconsistency in `final=True` (kept unused here) and ensure the transform variable name is consistent to avoid silent mistakes. These minimal changes should move the score upward toward the target without changing architecture or training.'
- What this solution (achieved 0.00807) has done: 'Your current score (0.00807) is far below the target (0.8637), so we should improve, but with minimal changes that don’t alter your model/training logic. The biggest likely issue is that `regress2class` expects a *regression-like scalar in the same scale as the thresholds* (0–4-ish), but you currently feed it `o_out.sum()` which is in 0–4 but not calibrated to your hardcoded thresholds (0.7,1.5,2.5,3.5) and can be quite noisy; a more stable ordinal-to-class mapping is to convert the 4 ordinal probabilities into an expected class `E[y]=sum_k P(y>k)` and then round to nearest integer (this preserves ordinal semantics and typically boosts QWK without changing the model). I also make checkpoint loading slightly more robust (handle `state_dict` prefixes) without changing what weights are used, and keep the submission formatting identical. These are small post-processing/load fixes aimed specifically at moving QWK upward toward your target.'
- What this solution (achieved 0.0204) has done: 'Your current QWK (0.00807) is far below target, and the biggest low-risk issue is that you’re using an EfficientNet-B7 backbone but applying B7’s ImageNet normalization rather than the correct EfficientNet normalization used by timm for that model, which can severely degrade predictions. I keep your exact model/weights and inference loop, but change the input transform to use `timm.data.create_transform` with the model’s own `default_cfg` mean/std (and keep the same resize size) so the network sees correctly normalized inputs. I also switch inference to a DataLoader-style batch loop (same logic) to reduce per-image overhead and keep runtime stable under the 600s limit, without changing semantics. These minimal changes should move the score upward toward the target without altering architecture or training.'
- What this solution (achieved 0.00395) has done: 'Your score is far below the target, and the current post-processing likely discards useful information by rounding the ordinal expected value; for QWK on this competition, a very common minimal fix is to tune the 4 ordinal decision thresholds on a small validation split and then apply those thresholds to test predictions. This keeps your exact model, weights-loading, transforms, and inference loop intact, but changes only the class-mapping step to better match the ordinal/QWK metric. I add a lightweight (single-epoch-free) calibration step: run inference on a stratified validation fold, search thresholds that maximize QWK, and then use those thresholds for the test set. This is deterministic (seeded), doesn’t change training, and should move performance substantially toward the target while staying within the 600s budget.'
- What this solution (achieved 0.00395) has done: 'Your current score is extremely far below the target, so we should improve it with the smallest changes that don’t alter your model or training/inference semantics. The biggest likely issue is that the checkpoint you’re loading may not actually match the `ThreeStage_Model` parameter names, so `strict=False` silently leaves most weights at random initialization, producing near-random predictions and a QWK near 0. I keep the exact same model and inference pipeline, but make checkpoint loading “name-aware” by remapping common key prefixes and loading only the matching tensors, and I print how many parameters were actually loaded to avoid silent failure. This should move predictions from random toward meaningful without changing architecture, transforms, loss, or training.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so the most likely issue is that you are not actually loading the intended trained weights (the glob search can pick unrelated `.pkl/.pth` files and the current “first sorted match” can leave most of the model randomly initialized). I make checkpoint selection stricter and deterministic by preferring files that look like the known competition checkpoint (B7 + epoch/70epoch keywords) and, crucially, choosing the candidate that yields the highest number of matching tensors when compared to the model’s `state_dict` (still no training changes). I also enforce a minimum “matched tensor ratio” sanity check: if a candidate matches too few tensors, we skip it and fall back to ImageNet pretrained, which is usually better than near-random weights. These changes keep your model/inference/post-processing logic intact and are directly aimed at moving QWK upward toward the target.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score strongly indicates the submission is effectively random; the most likely cause (given your history) is that the checkpoint loading is still failing silently and leaving most weights uninitialized, and/or the threshold tuning is overfitting a weak holdout because it uses only 20% of already-small training data. I make two minimal, score-relevant changes: (1) make checkpoint selection/load stricter by requiring that the chosen checkpoint matches a large fraction of tensors (otherwise fall back to ImageNet pretrained), and (2) replace the single 80/20 holdout threshold fit with a small deterministic stratified CV threshold fit (same thresholding logic, just more stable), then average thresholds across folds. These keep your model, transforms, inference, and metric semantics intact, but should move QWK upward toward your target by ensuring meaningful weights are used and thresholds generalize better. The script still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 QWK strongly suggests inference isn’t using the intended trained weights (or is selecting an incompatible checkpoint), so predictions are effectively random. I make checkpoint discovery deterministic and relevant by searching only inside this competition’s input directory and preferring filenames that look like known APTOS B7 checkpoints, then still selecting by best tensor-match to your `ThreeStage_Model`. I also make checkpoint loading more robust by handling cases where the checkpoint is a whole `nn.Module` or nested dict, while keeping your model, transforms, ordinal expected-value inference, and CV threshold tuning exactly the same. These changes are minimal, score-relevant (ensure meaningful weights are actually loaded), and still produce a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.0) has done: 'Your 0.0 QWK suggests predictions are essentially uninformative; the most likely culprit (given this code) is that the “best-match checkpoint” being loaded is still not the intended trained weights, leaving most layers randomly initialized or mismatched. I make a minimal, score-relevant change to checkpoint selection: require both a high tensor-match ratio and a “backbone coverage” check (many backbone tensors matched), otherwise skip that checkpoint and fall back to ImageNet pretrained (which is usually better than random). I also make the tensor-match scoring prefer checkpoints that match the backbone specifically (since that dominates performance), without changing your model, transforms, inference, or threshold-tuning logic. This keeps core logic intact but should move QWK upward toward your target by preventing silent bad weight loads.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 QWK is consistent with “garbage-inference” caused by weak/incorrect checkpoint loading plus thresholds being tuned on predictions that don’t reflect a trained model. I keep your exact model, inference, ordinal expected-value mapping, and CV threshold tuning, but make checkpoint discovery/load both stricter (so we don’t accidentally load random/unrelated weights) and more compatible with common checkpoint formats (including full `state_dict` nesting and non-`backbone.` key naming). The key minimal fix is to explicitly try mapping checkpoint keys into your `ThreeStage_Model` namespace (including mapping `model.*`/`net.*` weights into `backbone.*` when appropriate) and only accept a checkpoint if it loads a high fraction of backbone tensors; otherwise we fall back to ImageNet pretrained (which should beat random and move QWK upward). These changes are directly score-relevant and should keep runtime within the 600s limit while still producing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with generating essentially constant or misaligned predictions; the most likely minimal fix is to ensure inference uses a consistent, correct image normalization/resize for EfficientNet-B7 and that the ordinal head is mapped to classes in a way that matches QWK. I keep your model and inference logic intact, but (1) enforce a deterministic test-time preprocessing pipeline from `timm` (with explicit resize and center-crop to your `input_size`) and (2) tune thresholds using out-of-fold predictions (train-fold predicts val-fold) instead of predicting each fold with the exact same model state without any fold separation (which can make the tuned thresholds unstable/unhelpful). Finally, I add a small sanity check that aborts threshold tuning and falls back to default thresholds if the validation predictions are near-constant (which often produces QWK≈0). These are minimal, score-relevant changes that should move QWK upward toward your target while still writing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 QWK is so far below the target that the most likely issue is still “bad weights loaded” (effectively random inference), not the thresholding itself. I keep your model/inference/threshold-tuning core logic unchanged, but make checkpoint selection/load *less brittle* by (1) restricting the search to this competition’s dataset folder first, (2) also accepting checkpoints that store weights under common nested keys like `model_state_dict`, and (3) improving the key-remapping so checkpoints that omit the `backbone.` prefix can still populate `net.backbone.*` reliably. This should increase the chance that a real trained APTOS checkpoint is actually loaded (instead of mismatched tensors), moving QWK upward toward the target while still producing a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.0) has done: 'Your 0.0 QWK indicates the model’s test predictions are essentially uninformative, and the most likely cause in this code is that the selected checkpoint is still not truly compatible (so the network is effectively ImageNet-only or partially random), while thresholds are being tuned on those weak outputs. I keep your exact model/inference/threshold-tuning logic, but make one score-critical change: (1) prefer and correctly load the strongest available weights by explicitly checking for the well-known APTOS B7 checkpoint name first and, if found, load it with a broader key-unwrapping/remapping strategy; otherwise fall back to your existing “best tensor-match” selection. I also add a small sanity check to print the distribution of expected-values on train/test so you can confirm the pipeline isn’t collapsing to near-constant outputs (a common reason for QWK≈0), without changing predictions themselves. These changes are minimal, keep runtime within limits, and directly aim to move your score upward toward the target by ensuring you’re actually using meaningful trained weights.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 QWK is most consistent with a label/prediction misalignment or a collapsing prediction distribution rather than a subtle modeling issue, so I make minimal, score-relevant fixes that preserve your model and inference core logic. I (1) enforce deterministic, *index-aligned* OOF collection (so `oof_exp[val_idx]` always matches the correct sample even if DataLoader ordering changes), (2) make threshold tuning strictly monotonic and robust by optimizing thresholds on the aligned OOF arrays only, and (3) add a small safety fix to ensure the transform matches EfficientNet-B7’s expected input (resize + center crop) without changing the model. These changes don’t alter your architecture, losses, or training (none is performed), but they remove common causes of QWK≈0 and should move the score upward toward your target while still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import timm
from timm.data import resolve_data_config, create_transform

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score

device = "cuda:0" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.7, 1.5, 2.5, 3.5]


def regress2class(out, thr=None):
    thr = threshold if thr is None else thr
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= thr[i]).squeeze().cpu()
    return prediction


def ordinal_probs_to_expected(o_out: torch.Tensor) -> torch.Tensor:
    """
    o_out: [B,4] sigmoid outputs interpreted as P(y > k) for k=0..3
    returns: [B] expected value in [0,4] on CPU float32
    """
    return o_out.sum(dim=1).detach().float().cpu()


def apply_thresholds_to_expected(exp_y: np.ndarray, thr) -> np.ndarray:
    """exp_y: shape [N] float, thr: list/array length 4 increasing -> class 0..4"""
    thr = np.asarray(thr, dtype=np.float32)
    return (exp_y[:, None] >= thr[None, :]).sum(axis=1).astype(np.int64)


def _qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))




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


class Pretrain_Model(nn.Module):
    def __init__(self, backbone=None, pretrain=False):
        super(Pretrain_Model, self).__init__()

        if backbone is None:
            self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=pretrain)
            self.backbone.global_pool = GeM(flatten=True)
        else:
            self.backbone = backbone

        self.classifier1 = nn.Linear(1000, 500)
        self.classifier2 = nn.Linear(500, 5)

        self.regressor1 = nn.Linear(1000, 500)
        self.regressor2 = nn.Linear(500, 1)

        self.ordinal1 = nn.Linear(1000, 500)
        self.ordinal2 = nn.Linear(500, 4)

    def forward(self, x):
        x = self.backbone(x)

        c_out = self.classifier1(x)
        c_out = self.classifier2(c_out)

        r_out = self.regressor1(x)
        r_out = self.regressor2(r_out)
        r_out = torch.sigmoid(r_out) * 5 - 0.5

        o_out = self.ordinal1(x)
        o_out = self.ordinal2(o_out)
        o_out = torch.sigmoid(o_out)

        return c_out, r_out, o_out


class Maintrain_Model(Pretrain_Model):
    def __init__(self, weight_path):
        model = Pretrain_Model()
        model.load_state_dict(torch.load(weight_path))
        super(Maintrain_Model, self).__init__(model.backbone)


class Posttrain_Model(nn.Module):
    def __init__(self, weight_path=None):
        super(Posttrain_Model, self).__init__()

        self.model = Pretrain_Model()
        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path))

        self.regressor = nn.Linear(10, 1)

    def forward(self, x):
        c_out, r_out, o_out = self.model(x)

        out = torch.cat((c_out, r_out, o_out), 1)
        out = self.regressor(out)
        out = torch.sigmoid(out) * 5 - 0.5

        return out




## === cell 3
DATA_DIR = "../input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_ids = train_df["id_code"].astype(str).values
train_y = train_df["diagnosis"].astype(int).values
test_ids = test_df["id_code"].astype(str).values

input_size = 256


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None, pretrained=False):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.models.tf_efficientnet_b7_ns(pretrained=pretrained)
        self.backbone.global_pool = GeM(flatten=True)

        self.classifier1 = nn.Linear(1000, 500)
        self.classifier2 = nn.Linear(500, 5)

        self.regressor1 = nn.Linear(1000, 500)
        self.regressor2 = nn.Linear(500, 1)

        self.ordinal1 = nn.Linear(1000, 500)
        self.ordinal2 = nn.Linear(500, 4)

        self.final_regressor = nn.Linear(10, 1)

    def forward(self, x, final=False):
        x = self.backbone(x)

        c_out = self.classifier1(x)
        c_out = self.classifier2(c_out)

        r_out = self.regressor1(x)
        r_out = self.regressor2(r_out)
        r_out = torch.sigmoid(r_out) * 5 - 0.5

        o_out = self.ordinal1(x)
        o_out = self.ordinal2(o_out)
        o_out = torch.sigmoid(o_out)

        if final:
            out = torch.cat((c_out, r_out, o_out), 1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 5 - 0.5
            return out
        else:
            return c_out, r_out, o_out


def _clean_state_dict(state):
    """
    Change (score-relevant): broaden checkpoint unwrapping to catch common Kaggle formats.
    This increases the chance we load the actual trained weights (vs near-random), improving QWK.
    """
    if isinstance(state, nn.Module):
        state = state.state_dict()

    if not isinstance(state, dict):
        return state

    unwrap_keys = (
        "state_dict",
        "model_state_dict",
        "net",
        "model",
        "weights",
        "params",
        "ema_state_dict",
        "student",
        "teacher",
    )
    for key in unwrap_keys:
        if key in state and isinstance(state[key], dict) and len(state[key]) > 0:
            state = state[key]
            break

    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        if nk.startswith("net."):
            nk = nk[len("net.") :]
        new_state[nk] = v
    return new_state


def _remap_and_filter_state_dict_for_model(model: nn.Module, state: dict) -> dict:
    """
    Change (score-relevant): keep your core matching, but accept more backbone naming variants.
    """
    if not isinstance(state, dict):
        return {}

    model_sd = model.state_dict()
    out = {}

    def try_add(dst_k, v):
        if (
            dst_k in model_sd
            and torch.is_tensor(v)
            and model_sd[dst_k].shape == v.shape
        ):
            out[dst_k] = v
            return True
        return False

    backbone_like_prefixes = (
        "conv_stem",
        "bn1",
        "blocks",
        "conv_head",
        "bn2",
        "global_pool",
        "classifier",
    )

    for k, v in state.items():
        if not torch.is_tensor(v):
            continue

        if try_add(k, v):
            continue

        candidates = []

        for pref in ("module.", "model.", "net."):
            if k.startswith(pref):
                candidates.append(k[len(pref) :])

        if k.startswith(backbone_like_prefixes) or any(
            k.startswith(p + ".") for p in backbone_like_prefixes
        ):
            candidates.append("backbone." + k)

        for pref in ("module.", "model.", "net."):
            if k.startswith(pref):
                kk = k[len(pref) :]
                if kk.startswith(backbone_like_prefixes) or any(
                    kk.startswith(p + ".") for p in backbone_like_prefixes
                ):
                    candidates.append("backbone." + kk)

        candidates.append("backbone." + k)
        for pref in ("module.", "model.", "net."):
            if k.startswith(pref):
                candidates.append("backbone." + k[len(pref) :])

        for pref in (
            "model.backbone.",
            "module.backbone.",
            "net.backbone.",
            "model.model.backbone.",
            "backbone.backbone.",
        ):
            if k.startswith(pref):
                candidates.append("backbone." + k[len(pref) :])

        if "." in k:
            candidates.append(k.split(".", 1)[1])
            candidates.append("backbone." + k.split(".", 1)[1])

        for kc in candidates:
            if try_add(kc, v):
                break

    return out


def _score_filtered_sd(model: nn.Module, filtered_sd: dict):
    if not isinstance(filtered_sd, dict) or len(filtered_sd) == 0:
        return 0, 0, 0

    total = len(model.state_dict())
    matched = len(filtered_sd)

    backbone_keys = [k for k in model.state_dict().keys() if k.startswith("backbone.")]
    matched_backbone = sum(1 for k in filtered_sd.keys() if k.startswith("backbone."))
    total_backbone = max(1, len(backbone_keys))

    score = int(1000 * (matched_backbone / total_backbone) + 100 * (matched / total))
    return score, matched, matched_backbone


def _find_known_aptos_b7_checkpoint():
    """
    Change (score-relevant): explicitly prefer the known/standard APTOS checkpoint name if present.
    """
    roots = [
        "../input/aptos2019-blindness-detection",
        "../input",
    ]
    known_names = [
        "B7_ns_70epoch.pkl",
        "b7_ns_70epoch.pkl",
    ]
    for r in roots:
        for nm in known_names:
            p = os.path.join(r, nm)
            if os.path.isfile(p):
                return p
        for nm in known_names:
            hits = glob.glob(os.path.join(r, "**", nm), recursive=True)
            hits = [h for h in hits if os.path.isfile(h)]
            if hits:
                return sorted(hits)[0]
    return None


def _find_best_checkpoint_for_model(model: nn.Module):
    """
    Keep your existing selection, but after checking for a known APTOS B7 checkpoint first.
    """
    known = _find_known_aptos_b7_checkpoint()
    if known is not None:
        try:
            raw = torch.load(known, map_location="cpu")
            state = _clean_state_dict(raw)
            filtered = _remap_and_filter_state_dict_for_model(model, state)
            wscore, matched, _ = _score_filtered_sd(model, filtered)
            msg = (
                f"Found known APTOS checkpoint: {known} "
                f"(weighted_score={wscore}, matched {matched}/{len(model.state_dict())})."
            )
            return known, filtered, msg
        except Exception as e:
            print(f"Known checkpoint found but failed to load ({known}): {repr(e)}")

    search_roots = [
        "../input/aptos2019-blindness-detection",
        "../input",
    ]

    patterns = [
        "**/B7_ns_70epoch.pkl",
        "**/*aptos*B7*epoch*.pkl",
        "**/*aptos*b7*epoch*.pth",
        "**/*b7*70*epoch*.pkl",
        "**/*b7*epoch*.pkl",
        "**/*efficientnet*b7*.pkl",
        "**/*.pth",
        "**/*.pt",
        "**/*.pkl",
    ]

    candidates = []
    for root in search_roots:
        for pat in patterns:
            candidates.extend(glob.glob(os.path.join(root, pat), recursive=True))
    candidates = [p for p in candidates if os.path.isfile(p)]

    if len(candidates) == 0:
        return None, None, "No checkpoint candidates found."

    def priority(p):
        b = os.path.basename(p).lower()
        pl = p.lower()
        s = 0
        if "aptos2019-blindness-detection" in pl:
            s += 50
        if "aptos" in pl:
            s += 10
        if "blindness" in pl:
            s += 5
        if "b7" in b:
            s += 10
        if "70epoch" in b or ("70" in b and "epoch" in b):
            s += 8
        if "fold" in b:
            s += 1
        if b.endswith(".pkl"):
            s += 1
        return s

    candidates = sorted(candidates, key=lambda p: (-priority(p), p))

    best = (None, -1, -1, None)  # path, weighted_score, matched, filtered_sd
    tried = 0
    for p in candidates[:120]:
        try:
            raw = torch.load(p, map_location="cpu")
        except Exception:
            continue
        state = _clean_state_dict(raw)
        filtered = _remap_and_filter_state_dict_for_model(model, state)
        wscore, matched, matched_backbone = _score_filtered_sd(model, filtered)
        tried += 1
        if wscore > best[1]:
            best = (p, wscore, matched, filtered)

        backbone_total = max(
            1, sum(1 for k in model.state_dict().keys() if k.startswith("backbone."))
        )
        if matched_backbone >= int(0.98 * backbone_total):
            break

    if best[0] is None:
        return None, None, f"Tried {tried} checkpoints but none could be read."

    msg = (
        f"Selected checkpoint by backbone-weighted match: {best[0]} "
        f"(weighted_score={best[1]}, matched {best[2]}/{len(model.state_dict())})."
    )
    return best[0], best[3], msg


net = ThreeStage_Model(pretrained=False)

ckpt_path, filtered_sd, select_msg = _find_best_checkpoint_for_model(net)
print(select_msg)

loaded_msg = "No usable checkpoint; using ImageNet pretrained backbone only."
if ckpt_path is not None and isinstance(filtered_sd, dict):
    total_sd = len(net.state_dict())
    match_ratio = len(filtered_sd) / max(1, total_sd)

    backbone_total = max(
        1, sum(1 for k in net.state_dict().keys() if k.startswith("backbone."))
    )
    backbone_matched = sum(1 for k in filtered_sd.keys() if k.startswith("backbone."))
    backbone_ratio = backbone_matched / backbone_total

    if (match_ratio >= 0.60) and (backbone_ratio >= 0.85):
        missing, unexpected = net.load_state_dict(filtered_sd, strict=False)
        loaded_msg = (
            f"Loaded checkpoint: {ckpt_path}\n"
            f"Matched tensors: {len(filtered_sd)}/{total_sd} (match_ratio={match_ratio:.3f}).\n"
            f"Backbone matched: {backbone_matched}/{backbone_total} (backbone_ratio={backbone_ratio:.3f}).\n"
            f"load_state_dict: missing={len(missing)} unexpected={len(unexpected)}"
        )
    else:
        net = ThreeStage_Model(pretrained=True)
        loaded_msg = (
            f"Checkpoint rejected (match_ratio={match_ratio:.3f}, backbone_ratio={backbone_ratio:.3f}); "
            f"falling back to ImageNet pretrained backbone."
        )
else:
    net = ThreeStage_Model(pretrained=True)
    loaded_msg = "No checkpoint selected; falling back to ImageNet pretrained backbone."

print(loaded_msg)

net = net.to(device)
net.eval()

data_cfg = resolve_data_config({}, model=net.backbone)
data_cfg["input_size"] = (3, input_size, input_size)
if "crop_pct" not in data_cfg or data_cfg["crop_pct"] is None:
    data_cfg["crop_pct"] = 0.875
transform = create_transform(**data_cfg, is_training=False)


class ImgDataset(Dataset):
    def __init__(self, ids, img_dir, transform, labels=None):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = None if labels is None else np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        img = self.transform(img)
        if self.labels is None:
            return idx, img
        return idx, img, int(self.labels[i])


def infer_expected_values(ids, img_dir, labels=None, batch_size=16):
    ds = ImgDataset(ids, img_dir, transform, labels=labels)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.startswith("cuda"),
    )

    all_ids = []
    all_exp = []
    all_y = [] if labels is not None else None

    with torch.no_grad():
        for batch in loader:
            if labels is None:
                ids_b, imgs = batch
            else:
                ids_b, imgs, y_b = batch
                all_y.extend(list(y_b))

            imgs = imgs.to(device, non_blocking=True)
            _, _, o_out = net(imgs)
            exp_y = ordinal_probs_to_expected(o_out).numpy()
            all_exp.append(exp_y)
            all_ids.extend([str(x) for x in list(ids_b)])

    all_exp = np.concatenate(all_exp, axis=0)
    if labels is None:
        return np.asarray(all_ids), all_exp
    return np.asarray(all_ids), all_exp, np.asarray(all_y, dtype=np.int64)




## === cell 4
def tune_thresholds(exp_y: np.ndarray, y_true: np.ndarray, base_thr=None):
    thr = np.array(threshold if base_thr is None else base_thr, dtype=np.float32)

    eps = 1e-4
    thr = np.sort(thr)
    for i in range(1, 4):
        if thr[i] <= thr[i - 1] + eps:
            thr[i] = thr[i - 1] + eps

    best_pred = apply_thresholds_to_expected(exp_y, thr)
    best_score = _qwk(y_true, best_pred)

    grids = [
        np.linspace(0.2, 1.6, 29),
        np.linspace(1.0, 2.6, 33),
        np.linspace(2.0, 3.4, 29),
        np.linspace(2.8, 4.0, 25),
    ]

    for _ in range(3):
        for i in range(4):
            best_local = (best_score, float(thr[i]))
            for t in grids[i]:
                cand = thr.copy()
                cand[i] = float(t)
                cand = np.sort(cand)

                for j in range(1, 4):
                    if cand[j] <= cand[j - 1] + eps:
                        cand[j] = cand[j - 1] + eps

                if cand[0] < 0.0 or cand[3] > 4.0:
                    continue
                pred = apply_thresholds_to_expected(exp_y, cand)
                sc = _qwk(y_true, pred)
                if sc > best_local[0]:
                    best_local = (sc, float(t))
                    best_pred = pred
                    best_score = sc
                    thr = cand
            thr = np.sort(thr)
            for j in range(1, 4):
                if thr[j] <= thr[j - 1] + eps:
                    thr[j] = thr[j - 1] + eps

    return thr.tolist(), float(best_score)


bs = 16 if device.startswith("cuda") else 4

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

id_to_index = {str(i): idx for idx, i in enumerate(train_ids.tolist())}
oof_exp = np.full(len(train_ids), np.nan, dtype=np.float32)
oof_y = train_y.copy()

for fold, (tr_idx, val_idx) in enumerate(skf.split(train_ids, train_y), start=1):
    val_ids = train_ids[val_idx]
    val_y = train_y[val_idx]

    val_ids_out, val_exp, val_y_out = infer_expected_values(
        val_ids, TRAIN_IMG_DIR, labels=val_y, batch_size=bs
    )

    for idc, ev in zip(val_ids_out.tolist(), val_exp.tolist()):
        oof_exp[id_to_index[str(idc)]] = float(ev)

    print(
        f"Fold {fold}: collected val expected-values: mean={val_exp.mean():.4f} std={val_exp.std():.4f} "
        f"min={val_exp.min():.4f} max={val_exp.max():.4f}"
    )

if np.isnan(oof_exp).any():
    missing_n = int(np.isnan(oof_exp).sum())
    raise RuntimeError(
        f"OOF collection has {missing_n} missing entries (nan). Check data loading/alignment."
    )

print(
    "OOF expected-values summary:",
    f"mean={oof_exp.mean():.4f} std={oof_exp.std():.4f} min={oof_exp.min():.4f} max={oof_exp.max():.4f}",
)

if float(np.std(oof_exp)) < 1e-3:
    best_thr = threshold
    print("OOF expected-values are near-constant; using base thresholds:", best_thr)
else:
    best_thr, oof_qwk = tune_thresholds(oof_exp, oof_y, base_thr=threshold)
    print("Base thresholds:", threshold)
    print("OOF tuned thresholds:", [round(float(x), 4) for x in best_thr])
    print("OOF QWK:", round(float(oof_qwk), 6))



## === cell 5
test_ids_out, test_exp = infer_expected_values(
    test_ids, TEST_IMG_DIR, labels=None, batch_size=bs
)

print(
    "TEST expected-values summary:",
    f"mean={test_exp.mean():.4f} std={test_exp.std():.4f} min={test_exp.min():.4f} max={test_exp.max():.4f}",
)

test_pred = apply_thresholds_to_expected(test_exp, best_thr)

pred_map = {idc: int(p) for idc, p in zip(test_ids_out.tolist(), test_pred.tolist())}
ordered_pred = [pred_map[str(i)] for i in test_ids.tolist()]

submission_df = pd.DataFrame(
    {
        "id_code": test_ids.astype(str),
        "diagnosis": np.asarray(ordered_pred, dtype=np.int64),
    }
)
submission_df["diagnosis"] = submission_df["diagnosis"].clip(0, 4).astype(int)
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
print("diagnosis value counts:")
print(submission_df["diagnosis"].value_counts().sort_index())
