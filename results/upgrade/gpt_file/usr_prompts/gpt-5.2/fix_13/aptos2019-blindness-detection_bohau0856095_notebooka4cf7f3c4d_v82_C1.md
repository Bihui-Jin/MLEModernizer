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

0.9228298069522016

# 6. Current score

0.05174

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing weights dependency by loading whichever pretrained weight file is actually present (and otherwise fall back to an untrained model so the notebook still completes and writes a CSV). I also make the script run on CPU when no CUDA driver is available, and fix a couple of tensor device bugs in the probability helper functions that could crash on CPU. Finally, I make the image loop robust and fast enough (DataLoader batching instead of per-image prints) so it reliably produces a non-empty `submission.csv` with the required columns.'
- What this solution (achieved -0.03395) has done: 'I make the script robust to missing external weight files by allowing a safe fallback that uses timm’s built-in ImageNet pretrained backbone weights (so it still produces a non-empty, valid `submission.csv` instead of crashing). I also fix the device/type mismatch causing CUDA inputs to hit CPU weights by ensuring the model is moved to `device` before any forward pass and by forcing consistent dtype/device in inference. Finally, I make the submission generation always define `submission` and write a correctly formatted CSV even if an image fails to load, without changing the model architecture or inference semantics beyond these stability fixes.'
- What this solution (achieved 0.02584) has done: 'Your current negative kappa strongly suggests the predictions are systematically mis-calibrated for this metric, not just noisy. I keep your model and inference the same, but change only the final discretization step: instead of fixed hand thresholds, we compute optimal regression→class thresholds on the training set using the same regressor outputs to directly maximize quadratic weighted kappa. This aligns post-processing with the competition metric and typically yields a large improvement without changing architecture or training. The rest of the pipeline (weights loading, transforms, DataLoader, submission format) stays the same, and it still write a valid `submission.csv`.'
- What this solution (achieved 0.01886) has done: 'Your current score is far below the target, so we should improve QWK with the smallest changes that don’t alter your model/training logic. The biggest remaining issue is that you fit thresholds on the same data used to generate the regression outputs, which overfits and can yield poor generalization; we instead fit thresholds on out-of-fold predictions using a simple StratifiedKFold loop (same model, same forward pass, just a safer way to tune thresholds). We also make the threshold search a bit more fine-grained (still a tiny post-processing change) and add a robust fallback to default thresholds if anything goes wrong, keeping submission generation intact. All paths, model architecture, and inference semantics remain the same; only the threshold calibration procedure is corrected to better match the QWK metric.'
- What this solution (achieved 0.12777) has done: 'Your very low QWK suggests the model outputs are not aligned with the metric and/or the checkpoint isn’t being used as intended. To move toward the target with minimal change and without altering model/training logic, I (1) ensure we actually use the model’s intended `final=True` scalar output when available (still the same architecture, just the designed inference path), and (2) fit QWK thresholds on out-of-fold predictions from that same final output for better generalization. As a safety net, if the final head behaves badly (e.g., constant outputs), we fall back to the regressor head exactly as before so the notebook remains stable and always writes a valid `submission.csv`. Everything else (paths, transforms, no training loop, submission formatting) remains unchanged.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so we should improve QWK with the smallest changes that keep your model and inference intact. The biggest likely issue is preprocessing mismatch: these EfficientNet backbones expect `timm`’s own normalization/config, but your code uses custom mean/std, which can severely degrade predictions even with good weights. I switch only the normalization part of the transform to `timm.data.resolve_model_data_config(...)` + `create_transform(...)` (keeping your trim/crop and the same final image size), and keep the same OOF threshold fitting/inference logic. This typically yields a large, legitimate uplift without changing architecture, training, or loss.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with a silent submission misalignment: you build predictions in a `DataFrame`, but then convert them to a NumPy `object` array and back, which can introduce subtle dtype/order issues; we keep your model/inference unchanged but keep predictions as a DataFrame keyed by `id_code` throughout and assert perfect row-count/alignment before writing. Next, to legitimately improve QWK without changing architecture/training, we make the OOF threshold fitting more robust by (a) ensuring OOF predictions are filled in the exact original row order (no reliance on subset indices) and (b) adding a monotonicity-safe fallback if any NaNs appear. Finally, we also force deterministic evaluation transforms and disable any accidental randomness (your current `random.shuffle` augmentation class exists but isn’t used; we keep it unused and just ensure the eval pipeline is deterministic).'
- What this solution (achieved 0.00024) has done: 'Your 0.0 score is most consistent with a semantic mismatch between how your regressor outputs are used and how thresholds are calibrated (and then applied) for QWK. I keep your model, weights logic, and inference identical, but make threshold fitting respect the natural bounds/shape of your regressor by (1) clipping OOF regression outputs to the same [0, 4.5] range your model produces and (2) initializing thresholds from the OOF prediction quantiles per class (then doing the same local QWK search as you already do). This is a minimal post-processing change that usually prevents pathological threshold sets (e.g., collapsing to extreme classes), which can yield near-zero kappa. The submission writing and alignment remain unchanged and still produce a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current QWK is extremely low, so we need a legitimate improvement with minimal disruption to your existing model and no training changes. The most likely culprit is a preprocessing mismatch: you’re creating the timm transform with a config that doesn’t match the actual backbone you instantiate (your model is B4, but the config is resolved from a separately created B4 model object and then wrapped inside an extra `Resize` + custom crop/trim), which can produce badly scaled inputs. I switch to timm’s recommended evaluation preprocessing for the exact instantiated backbone (same input size 380, still deterministic), and I keep your trim/crop logic but move resizing/normalization into a single consistent timm pipeline. This is a minimal change that often yields a large QWK lift when weights are correct, while keeping architecture/inference/threshold fitting identical.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 QWK is most consistent with the checkpoint not actually being applied to the intended model parameters, leaving you with essentially random/constant predictions even though the notebook completes. I keep your model, transforms, and OOF threshold-fitting logic intact, but make weight loading robust to common checkpoint formats (including `torch.nn.Module` objects and PyTorch Lightning keys) and explicitly verify that a meaningful fraction of parameters changed after loading. If loading still fails/mismatches, we still fall back to ImageNet-pretrained backbone as before (so a valid CSV is always produced), but this patch should much more often activate the real competition-trained weights and move QWK toward your target. No training is introduced and inference/post-processing semantics remain the same.'
- What this solution (achieved 0.05174) has done: 'Your current 0.0 score strongly suggests the model is effectively producing near-random/constant predictions (or the checkpoint still isn’t being correctly applied), so the smallest meaningful improvement is to make checkpoint loading actually succeed across common formats and to ensure the model’s backbone/head dimensions match the saved weights. I keep your architecture and inference logic intact, but (1) instantiate the timm backbones with `num_classes=0` so the feature dimension is correct for `GeM + Linear` heads (avoiding silent mismatches), and (2) broaden the checkpoint key-cleaning/extraction to handle `backbone.*` / Lightning-style keys while loading with a more reliable “changed tensor fraction” check. These changes should move your predictions from essentially broken to at least reasonable without changing training, loss, threshold fitting, or submission semantics. The rest of your pipeline (timm eval preprocessing, OOF threshold fitting, final CSV writing) remains the same.'

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
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor, thr=None):
    out = out.detach()
    thr = threshold if thr is None else thr
    prediction = torch.zeros(out.size(0), device=out.device)
    for i in range(4):
        prediction += (out >= thr[i]).to(prediction.dtype)
    return prediction


def ordinal2class_prob(out: torch.Tensor):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out: torch.Tensor):
    out = out.detach()
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
        self.backbone = timm.create_model(
            "tf_efficientnet_b5_ns", pretrained=False, num_classes=0
        )
        self.backbone.global_pool = GeM(flatten=True)
        feat_dim = getattr(self.backbone, "num_features", 1000)
        self.regressor = nn.Linear(feat_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=False, num_classes=0
        )
        self.backbone.global_pool = GeM(flatten=True)
        feat_dim = getattr(self.backbone, "num_features", 1000)

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_dim, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )

        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_dim, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )

        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_dim, 500),
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
from timm.data import resolve_model_data_config, create_transform

DATA_DIR_CANDIDATES = [
    "/kaggle/input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "../kaggle/data/aptos2019-blindness-detection",
]
DATA_DIR = next((p for p in DATA_DIR_CANDIDATES if os.path.exists(p)), None)
if DATA_DIR is None:
    raise FileNotFoundError(
        f"Could not find dataset directory. Tried: {DATA_DIR_CANDIDATES}"
    )

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

input_size = 380

_backbone_for_cfg = timm.create_model(
    "tf_efficientnet_b4_ns", pretrained=False, num_classes=0
)
_data_cfg = resolve_model_data_config(_backbone_for_cfg)
_data_cfg["input_size"] = (3, input_size, input_size)
_timm_eval_tf = create_transform(**_data_cfg, is_training=False)

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        _timm_eval_tf,  # includes Resize, ToTensor, Normalize as expected by this backbone family
    ]
)

net = ThreeStage_Model()


def _find_weight_file():
    candidates = [
        "../input/weights/B4_3stage_17epoch_finetune2.pkl",
        "../input/weights/B4_3stage_17epoch_finetune2.pth",
        "../input/weights/B4_3stage_17epoch_finetune2.pt",
        "../input/weights/0.912_tf_efficientnet_b4_ns_regress.pth",
        "/kaggle/input/weights/B4_3stage_17epoch_finetune2.pkl",
        "/kaggle/input/weights/B4_3stage_17epoch_finetune2.pth",
        "/kaggle/input/weights/B4_3stage_17epoch_finetune2.pt",
        "/kaggle/input/weights/0.912_tf_efficientnet_b4_ns_regress.pth",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    scan_roots = ["../input", "/kaggle/input", "/kaggle/data", "../kaggle/data"]
    exts = (".pth", ".pt", ".pkl")
    for scan_root in scan_roots:
        if not os.path.exists(scan_root):
            continue
        for root, _, files in os.walk(scan_root):
            for fn in files:
                if fn.endswith(exts) and (
                    "B4" in fn or "b4" in fn or "3stage" in fn or "efficientnet" in fn
                ):
                    return os.path.join(root, fn)
    return None


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, nn.Module):
        return ckpt_obj.state_dict()

    if isinstance(ckpt_obj, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
            "ema_state_dict",
        ]:
            if k in ckpt_obj:
                v = ckpt_obj[k]
                if isinstance(v, nn.Module):
                    return v.state_dict()
                if isinstance(v, dict):
                    return v

        if len(ckpt_obj) > 0 and all(isinstance(kk, str) for kk in ckpt_obj.keys()):
            if any(
                kk.startswith(("model.", "module.", "net.", "backbone."))
                for kk in ckpt_obj.keys()
            ):
                return ckpt_obj
            if all(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
                return ckpt_obj

    return None


def _clean_state_dict_keys(state_dict):
    new_state = {}
    for k, v in state_dict.items():
        nk = k
        for pref in ("module.", "model.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        new_state[nk] = v
    return new_state


def _load_with_change_check(net, state_dict, min_changed_frac=0.02):
    before = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    missing, unexpected = net.load_state_dict(state_dict, strict=False)

    after = net.state_dict()
    changed = 0
    total = 0
    for k in before.keys():
        if k in after:
            total += 1
            try:
                if not torch.equal(before[k], after[k].detach().cpu()):
                    changed += 1
            except Exception:
                pass

    changed_frac = changed / max(1, total)
    ok = changed_frac >= min_changed_frac
    return ok, changed_frac, missing, unexpected


weight_path = _find_weight_file()
missing, unexpected = [], []
if weight_path is None:
    print(
        "WARNING: No external pretrained weights found. Falling back to ImageNet-pretrained EfficientNet-B4 backbone."
    )
    net.backbone = timm.create_model(
        "tf_efficientnet_b4_ns", pretrained=True, num_classes=0
    )
    net.backbone.global_pool = GeM(flatten=True)
else:
    ckpt = torch.load(weight_path, map_location="cpu")
    state_dict = _extract_state_dict(ckpt)

    if state_dict is None or not isinstance(state_dict, dict) or len(state_dict) == 0:
        print(
            "WARNING: Checkpoint found but could not extract a valid state_dict; "
            "falling back to ImageNet-pretrained EfficientNet-B4 backbone.\n"
            f"weight_path={weight_path}"
        )
        net = ThreeStage_Model()
        net.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=True, num_classes=0
        )
        net.backbone.global_pool = GeM(flatten=True)
        weight_path = None
        missing, unexpected = [], []
    else:
        state_dict = _clean_state_dict_keys(state_dict)

        ok, changed_frac, missing, unexpected = _load_with_change_check(
            net, state_dict, min_changed_frac=0.02
        )

        if (not ok) or (
            len(missing) > 0 and len(missing) >= 0.9 * len(net.state_dict())
        ):
            print(
                "WARNING: Checkpoint found but did not change model parameters sufficiently; "
                "falling back to ImageNet-pretrained EfficientNet-B4 backbone.\n"
                f"weight_path={weight_path}\nchanged_frac={changed_frac:.4f} "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
            net = ThreeStage_Model()
            net.backbone = timm.create_model(
                "tf_efficientnet_b4_ns", pretrained=True, num_classes=0
            )
            net.backbone.global_pool = GeM(flatten=True)
            weight_path = None
            missing, unexpected = [], []
        else:
            print(
                f"Loaded checkpoint OK: weight_path={weight_path} changed_frac={changed_frac:.4f} "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

net = net.to(device)
net.eval()

print("Using DATA_DIR:", DATA_DIR)
print(
    "External weights:",
    weight_path if weight_path is not None else "(none; backbone pretrained=True)",
)
print("Device:", device)
print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))



## === cell 5
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold


class TrainDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True).copy()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        idx = str(self.df.loc[i, "id_code"])
        y = int(self.df.loc[i, "diagnosis"])
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        try:
            img = Image.open(image_name).convert("RGB")
        except Exception:
            img = Image.new("RGB", (input_size, input_size), (0, 0, 0))
        img = self.transform(img)
        return i, idx, img, y


def _preds_from_thr(y_reg_np, thr):
    thr = list(thr)
    pred = np.zeros_like(y_reg_np, dtype=np.int64)
    for t in thr:
        pred += (y_reg_np >= t).astype(np.int64)
    return np.clip(pred, 0, 4)


def _init_thresholds_from_oof_quantiles(y_reg_np, y_true_np):
    thr = []
    for c in range(4):
        left = y_reg_np[y_true_np <= c]
        right = y_reg_np[y_true_np >= (c + 1)]
        if left.size == 0 or right.size == 0:
            thr.append([0.75, 1.5, 2.5, 3.5][c])
            continue
        ql = float(np.quantile(left, 0.90))
        qr = float(np.quantile(right, 0.10))
        t = 0.5 * (ql + qr)
        thr.append(t)
    thr = np.clip(np.array(thr, dtype=np.float64), 0.0, 4.5)
    for i in range(1, 4):
        if thr[i] <= thr[i - 1] + 1e-6:
            thr[i] = min(4.5, thr[i - 1] + 0.05)
    if not (thr[0] < thr[1] < thr[2] < thr[3]) or thr[3] >= 4.5:
        return [0.75, 1.5, 2.5, 3.5]
    return thr.tolist()


def _fit_thresholds_qwk(y_reg_np, y_true_np, init_thr=(0.75, 1.5, 2.5, 3.5)):
    thr = np.array(init_thr, dtype=np.float64)

    def score(thr_vec):
        pred = _preds_from_thr(y_reg_np, thr_vec)
        return cohen_kappa_score(y_true_np, pred, weights="quadratic")

    if not np.isfinite(y_reg_np).all():
        y_reg_np = np.nan_to_num(
            y_reg_np,
            nan=float(np.nanmedian(y_reg_np[np.isfinite(y_reg_np)])),
            posinf=4.5,
            neginf=0.0,
        )

    best = score(thr)

    steps = [0.25, 0.10, 0.05, 0.02]
    for step in steps:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for delta in (-step, step):
                    cand = thr.copy()
                    cand[i] = cand[i] + delta
                    cand = np.clip(cand, 0.0, 4.5)
                    if not (cand[0] < cand[1] < cand[2] < cand[3]):
                        continue
                    s = score(cand)
                    if s > best:
                        best = s
                        thr = cand
                        improved = True
    return thr.tolist(), float(best)


def _infer_reg_values(net, imgs: torch.Tensor):
    out_final = net(imgs, final=True).squeeze(1)
    if torch.isfinite(out_final).all():
        if float(out_final.std().detach().cpu().item()) > 1e-6:
            return out_final
    _, r_out, _ = net(imgs)
    return r_out.squeeze(1)


train_ds = TrainDataset(train_df, TRAIN_IMG_DIR, transform)

y_all = train_df["diagnosis"].astype(int).values
n = len(train_df)

oof_reg = np.full(n, np.nan, dtype=np.float64)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

net.eval()
with torch.no_grad():
    for fold, (_, val_idx) in enumerate(skf.split(np.zeros(n), y_all), 1):
        val_subset = torch.utils.data.Subset(train_ds, val_idx.tolist())
        val_dl = DataLoader(
            val_subset,
            batch_size=8,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

        for i_batch, ids, imgs, y in val_dl:
            imgs = imgs.to(
                device=device,
                dtype=torch.float32,
                non_blocking=torch.cuda.is_available(),
            )
            r = _infer_reg_values(net, imgs)
            r = r.detach().to("cpu").numpy().astype(np.float64)
            i_batch_np = np.asarray(i_batch, dtype=np.int64)
            oof_reg[i_batch_np] = r

oof_reg = np.clip(oof_reg, 0.0, 4.5)

if not np.isfinite(oof_reg).all():
    finite = oof_reg[np.isfinite(oof_reg)]
    fill_value = float(np.median(finite)) if finite.size else 2.0
    oof_reg = np.nan_to_num(oof_reg, nan=fill_value, posinf=4.5, neginf=0.0)

init_thr = _init_thresholds_from_oof_quantiles(oof_reg, y_all)

try:
    fitted_thr, oof_qwk = _fit_thresholds_qwk(oof_reg, y_all, init_thr=init_thr)
except Exception as e:
    print(
        "WARNING: threshold fitting failed; falling back to default thresholds.",
        repr(e),
    )
    fitted_thr = list(threshold)
    oof_qwk = float("nan")

print("Init thresholds (OOF quantiles):", init_thr)
print("Fitted thresholds (OOF):", fitted_thr)
print("OOF QWK (sanity):", oof_qwk)




## === cell 6
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
        try:
            img = Image.open(image_name).convert("RGB")
        except Exception:
            img = Image.new("RGB", (input_size, input_size), (0, 0, 0))
        img = self.transform(img)
        return idx, img


ds = TestDataset(test_ids, TEST_IMG_DIR, transform)
dl = DataLoader(
    ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available()
)

pred_records = []
net.eval()
with torch.no_grad():
    for ids, imgs in dl:
        imgs = imgs.to(
            device=device, dtype=torch.float32, non_blocking=torch.cuda.is_available()
        )
        r_out = _infer_reg_values(net, imgs)
        preds = regress2class(r_out, thr=fitted_thr).to("cpu").numpy().astype(int)
        for idx, p in zip(ids, preds):
            pred_records.append((str(idx), int(p)))

pred_df = pd.DataFrame(pred_records, columns=["id_code", "diagnosis"])
pred_df["id_code"] = pred_df["id_code"].astype(str)
pred_df["diagnosis"] = pred_df["diagnosis"].astype(int)

pred_df = pred_df.groupby("id_code", as_index=False)["diagnosis"].mean()
pred_df["diagnosis"] = pred_df["diagnosis"].round().astype(int).clip(0, 4)

sub_df = test_df.copy()
sub_df["id_code"] = sub_df["id_code"].astype(str)
sub_df = sub_df.merge(pred_df, on="id_code", how="left")
sub_df["diagnosis"] = sub_df["diagnosis"].fillna(0).astype(int).clip(0, 4)

assert sub_df.shape[0] == test_df.shape[0]
assert sub_df["id_code"].isna().sum() == 0

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("Unique predictions:", sorted(sub_df["diagnosis"].unique().tolist()))
