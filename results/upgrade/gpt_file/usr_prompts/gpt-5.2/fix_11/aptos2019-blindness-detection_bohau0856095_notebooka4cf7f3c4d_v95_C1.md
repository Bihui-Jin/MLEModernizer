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

0.9246456187118622

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) make the notebook run on CPU-only Kaggle sessions by selecting `cuda` only when available, which fixes the “no NVIDIA driver” crash and allows inference to complete. I (2) fix the missing weights error by loading from a weights path that actually exists if available, otherwise falling back to running with random weights (still producing a valid submission CSV). I (3) correct two small transform/augmentation bugs that can cause undefined behavior (`is` vs `==` for strings, and `trim()` returning `None`), ensuring images always flow through the pipeline. Finally, I ensure the submission file is always written with the required columns and `.csv` suffix.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with running inference using randomly initialized weights (your code warns when weights are missing), so the smallest legitimate improvement is to reliably load a real pretrained checkpoint if it exists anywhere under the provided dataset tree. I keep your model and transforms unchanged, but add an automatic filesystem search for `B4_3stage_20epoch_finetune3.pkl` under `/kaggle/input` and `/kaggle/data`, then load it with safe key-handling (full state_dict vs nested `state_dict`). If no checkpoint is found, the code still produce a valid `submission.csv` (as before), but when the checkpoint is present it should move the score strongly upward toward your target. I also add a deterministic, no-accuracy-change improvement by batching test inference (same predictions, faster and less error-prone under time limits) without changing the model outputs.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the model is still running with random weights (or loading an incompatible checkpoint), so the smallest improvement toward the 0.9246 target is to (1) make the data root auto-detect the real Kaggle input path you actually have, and (2) harden checkpoint loading so it correctly finds/loads weights even when key prefixes differ (e.g., `module.`, `net.`, `model.`). These changes preserve your exact model, transforms, and inference logic; they only increase the chance the intended trained weights are applied. I also force CPU/GPU inference determinism-related settings off and keep the same thresholds to preserve evaluation semantics, while ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model still running with random (or mismatched) weights, so the smallest change that should move you sharply upward toward the 0.9246 target is to (1) reliably locate the intended checkpoint anywhere under the available Kaggle filesystem and (2) load it in a way that correctly matches keys (including EfficientNet naming differences) while failing loudly if nothing meaningful was loaded. I keep your model, transforms, thresholds, and inference exactly the same; the only behavioral change is making checkpoint loading robust and verifiable. If the checkpoint cannot be found/loaded, the script still produce a valid `submission.csv`, but it clearly warn that the expected score be low.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the checkpoint is still not being found/loaded correctly (so you’re predicting with random weights). I make the smallest change that increases the chance of loading the real trained weights: search specifically for the known weight filename (and common variants) under the *actual* competition dataset folder(s) you have (`/kaggle/input/aptos2019-blindness-detection` and `/kaggle/data/aptos2019-blindness-detection`), and then verify that critical layer tensors match in shape before proceeding. If the checkpoint exists but is incompatible (e.g., different architecture), the code clearly warn and still write a valid submission (but we avoid silently “loading” a useless partial match that yields near-random predictions). This preserves your model, transforms, thresholds, and inference semantics; it only hardens checkpoint discovery/loading to move the score upward toward your 0.9246 target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the model is still running with random (or mostly-unloaded) weights, so the smallest score-improving change is to make checkpoint discovery/loading actually succeed for common Kaggle situations. I (1) broaden the weight-file search to include all `.pth/.pt/.bin/.pkl` files under the dataset root and automatically pick the best-matching checkpoint by state_dict key overlap, and (2) tighten loading so we only proceed with a checkpoint that loads a high fraction of parameters (otherwise we keep the current safe fallback). This preserves your exact model, transforms, thresholds, and inference semantics; it only increases the chance that real trained weights are used. The submission writing and column schema remain unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is overwhelmingly consistent with the model still running with random weights (or silently “loading” an incompatible checkpoint), so the minimal change that should move you sharply upward toward the 0.9246 target is to reliably find and load a truly compatible checkpoint if one exists in the dataset tree. I keep your exact model, transforms, thresholds, and inference flow, but (1) expand checkpoint search to include common Kaggle locations (including `/kaggle/working`), (2) add a fast “compatibility scoring” step that selects the best-matching checkpoint by tensor-shape matches (not just key overlap), and (3) only accept a checkpoint if it loads a high fraction of parameters, otherwise clearly fall back (still producing a valid submission). This should convert “random predictions” into “trained predictions” without altering evaluation semantics.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with still running on random weights, so the minimal improvement is to make checkpoint selection/loading actually succeed and to refuse “mostly-unloaded” checkpoints that silently degrade predictions. I keep your exact model, transforms, thresholds, and inference (still using the regressor head + `regress2class`), but tighten compatibility scoring to use tensor-shape matches (not just key overlap) and try a small set of common key-mappings (including `backbone.` and `model.` nesting) so real checkpoints load reliably when present. I also add an explicit sanity print (loaded parameter ratio) and ensure submission ordering matches `test.csv` exactly (no change if already correct), which avoids accidental misalignment penalties. These changes should move the score upward toward your 0.9246 target without altering evaluation semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with effectively random predictions because the checkpoint is not being found or not being loaded (the code can silently fall back). I make the smallest change that increases the probability of successfully using a real trained checkpoint: broaden checkpoint discovery to include *all* weight files under the known dataset roots (not just filenames containing “b4/3stage”), then select the best one by tensor-shape match as you already do. I also tighten the “loaded ok” decision to use the already-computed shape-match ratio as the primary gate and print a clear warning when we fall back, without changing your model, transforms, thresholds, or inference semantics. This should move the score upward toward your 0.9246 target if any compatible weights exist in the filesystem; otherwise it still produce a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model still running on random/mostly-unloaded weights, so the smallest change that should move you sharply upward toward the 0.9246 target is to (1) prioritize finding a real compatible checkpoint (including inside `train_images/`, where Kaggle datasets sometimes stash extra files) and (2) make checkpoint loading succeed when the state_dict is nested under common wrapper keys (e.g., `ema_state_dict`, `student`, `teacher`) while still refusing low shape-match loads. These changes preserve your exact model architecture, transforms, thresholds, and inference semantics; they only increase the probability that the intended trained weights are actually applied. I also add a lightweight verification print of the prediction class distribution to quickly catch the “all one class” failure mode that commonly yields near-zero kappa, without altering outputs. The script still always writes a valid `submission.csv` with correct ordering.'

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
import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



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
            l1 = int(math.floor(out[i]))
            l2 = int(math.ceil(out[i]))
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
def _pick_data_root():
    candidates = [
        "../input/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/input",
        "/kaggle/data",
        "../input",
        "../data",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "test.csv")
        ):
            return c
        nested = os.path.join(c, "aptos2019-blindness-detection")
        if os.path.exists(os.path.join(nested, "train.csv")) and os.path.exists(
            os.path.join(nested, "test.csv")
        ):
            return nested
    return "../input/aptos2019-blindness-detection"


DATA_ROOT = _pick_data_root()
print("Using DATA_ROOT:", DATA_ROOT)

test_csv_path = os.path.join(DATA_ROOT, "test.csv")
test_img_dir = os.path.join(DATA_ROOT, "test_images")

test_ids = pd.read_csv(test_csv_path)["id_code"].values

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


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in [
            "state_dict",
            "model",
            "net",
            "model_state_dict",
            "ema_state_dict",
            "ema",
            "teacher",
            "student",
            "module",
        ]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key]
    return ckpt


def _strip_prefixes(k: str):
    for prefix in ["module.", "model.", "net."]:
        if k.startswith(prefix):
            return k[len(prefix) :]
    return k


def _clean_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[_strip_prefixes(k)] = v
    return cleaned


def _make_state_dict_variants(sd0):
    variants = []

    variants.append(sd0)

    sd_backbone = {}
    for k, v in sd0.items():
        sd_backbone[k] = v
        sd_backbone["backbone." + k] = v
    variants.append(sd_backbone)

    sd_rm_backbone = {}
    for k, v in sd0.items():
        if k.startswith("backbone."):
            sd_rm_backbone[k[len("backbone.") :]] = v
        sd_rm_backbone[k] = v
    variants.append(sd_rm_backbone)

    sd_wrap_model = {}
    for k, v in sd0.items():
        sd_wrap_model[k] = v
        sd_wrap_model["model." + k] = v
        sd_wrap_model["net." + k] = v
    variants.append(sd_wrap_model)

    return variants


def _best_variant_by_shape_match(model, raw_state_dict):
    sd0 = _clean_state_dict_keys(raw_state_dict)
    if not isinstance(sd0, dict):
        return None, 0.0, 0, 0

    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    best_sd = None
    best_shape_ratio = -1.0
    best_n_match_shape = 0
    best_inter = 0

    for sd in _make_state_dict_variants(sd0):
        inter = len(model_keys.intersection(sd.keys()))
        n_match_shape = 0
        for k in model_keys.intersection(sd.keys()):
            try:
                if tuple(sd[k].shape) == tuple(model_sd[k].shape):
                    n_match_shape += 1
            except Exception:
                pass
        shape_ratio = n_match_shape / max(1, len(model_sd))
        if (shape_ratio > best_shape_ratio) or (
            shape_ratio == best_shape_ratio and inter > best_inter
        ):
            best_shape_ratio = shape_ratio
            best_sd = sd
            best_n_match_shape = n_match_shape
            best_inter = inter

    return best_sd, best_shape_ratio, best_n_match_shape, best_inter


def _load_checkpoint_into_model(model, weight_path: str):
    ckpt = torch.load(weight_path, map_location="cpu")
    raw_state_dict = _extract_state_dict(ckpt)
    if not isinstance(raw_state_dict, dict):
        print("WARNING: checkpoint is not a state_dict-like object; skipping load.")
        return False, 0.0, 0.0, 0, 0

    best_sd, shape_ratio, n_match_shape, inter = _best_variant_by_shape_match(
        model, raw_state_dict
    )
    if best_sd is None:
        print("WARNING: could not derive a usable state_dict; skipping load.")
        return False, 0.0, 0.0, 0, 0

    if shape_ratio < 0.60:
        print(
            f"WARNING: very low shape-match ratio ({shape_ratio:.3f}) for {weight_path}; skipping to avoid random-like initialization."
        )
        return False, 0.0, shape_ratio, n_match_shape, inter

    missing, unexpected = model.load_state_dict(best_sd, strict=False)

    n_total = len(model.state_dict())
    n_missing = len(missing)
    n_loaded_est = n_total - n_missing
    loaded_ratio = n_loaded_est / max(1, n_total)

    print("Loaded weights from:", weight_path)
    print(
        f"State dict keys: total_model={n_total}, missing={n_missing}, unexpected={len(unexpected)}, loaded_ratio~{loaded_ratio:.3f}, "
        f"shape_match_ratio={shape_ratio:.3f}, match_shape_keys={n_match_shape}, key_overlap={inter}"
    )

    if not (loaded_ratio >= 0.70 or shape_ratio >= 0.80):
        print(
            "WARNING: checkpoint appears to load too little of the model; treating as incompatible to avoid low-score submissions."
        )
        return False, loaded_ratio, shape_ratio, n_match_shape, inter

    return True, loaded_ratio, shape_ratio, n_match_shape, inter


def _candidate_weight_files():
    roots = [
        os.path.join(DATA_ROOT, "weights"),
        DATA_ROOT,
        os.path.join(DATA_ROOT, "train_images"),
        os.path.join(DATA_ROOT, "test_images"),
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection/weights",
        "/kaggle/input/aptos2019-blindness-detection/train_images",
        "/kaggle/input/aptos2019-blindness-detection/test_images",
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection/weights",
        "/kaggle/data/aptos2019-blindness-detection/train_images",
        "/kaggle/data/aptos2019-blindness-detection/test_images",
        "/kaggle/working",
        "/kaggle/working/weights",
        "/kaggle/input",
        "/kaggle/data",
    ]
    exts = (".pth", ".pt", ".bin", ".pkl")
    seen = set()
    out = []

    preferred = [
        "B4_3stage_20epoch_finetune3.pkl",
        "B4_3stage_20epoch_finetune3.pth",
        "B4_3stage_20epoch_finetune3.pt",
    ]
    for r in roots:
        if not r or not os.path.exists(r):
            continue
        for fn in preferred:
            p = os.path.join(r, fn)
            if os.path.exists(p) and p not in seen:
                out.append(p)
                seen.add(p)

    for r in roots:
        if not r or not os.path.exists(r):
            continue
        for dirpath, _, filenames in os.walk(r):
            for f in filenames:
                fl = f.lower()
                if fl.endswith(exts):
                    p = os.path.join(dirpath, f)
                    if p not in seen:
                        out.append(p)
                        seen.add(p)
            if len(out) >= 120:
                return out
    return out


def _score_checkpoint_compat(model, weight_path: str):
    try:
        ckpt = torch.load(weight_path, map_location="cpu")
        raw_state_dict = _extract_state_dict(ckpt)
        if not isinstance(raw_state_dict, dict):
            return -1.0, 0.0, 0, 0

        _, shape_ratio, n_match_shape, inter = _best_variant_by_shape_match(
            model, raw_state_dict
        )
        score = shape_ratio + 1e-6 * inter
        return score, shape_ratio, n_match_shape, inter
    except Exception:
        return -1.0, 0.0, 0, 0


def _pick_best_checkpoint(model):
    candidates = _candidate_weight_files()
    if not candidates:
        return None

    best_path = None
    best_score = -1.0

    for p in candidates:
        s, shape_ratio, n_match_shape, inter = _score_checkpoint_compat(model, p)
        if s > best_score:
            best_score = s
            best_path = p

    return best_path


best_ckpt = _pick_best_checkpoint(net)
if best_ckpt is not None:
    print("Best checkpoint candidate:", best_ckpt)
    loaded_ok, _, _, _, _ = _load_checkpoint_into_model(net, best_ckpt)
    if not loaded_ok:
        print(
            "WARNING: checkpoint search found candidates but none were sufficiently compatible; running with randomly initialized weights (expected low score)."
        )
else:
    print(
        "WARNING: no checkpoint files found under dataset roots; running with randomly initialized weights (expected low score)."
    )

net = net.to(device)
net.eval()



## === cell 5
submission_rows = []

batch_size = 8 if device.startswith("cuda") else 4
batch_imgs = []
batch_ids = []

with torch.no_grad():
    for idx in test_ids:
        image_name = os.path.join(test_img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        img = transform(img)
        batch_imgs.append(img)
        batch_ids.append(idx)

        if len(batch_imgs) == batch_size:
            x = torch.stack(batch_imgs, dim=0).to(device)
            _, r_out, _ = net(x)
            preds = regress2class(r_out.data.squeeze(1))
            for i in range(len(batch_ids)):
                submission_rows.append([batch_ids[i], int(preds[i].item())])
            batch_imgs, batch_ids = [], []

    if len(batch_imgs) > 0:
        x = torch.stack(batch_imgs, dim=0).to(device)
        _, r_out, _ = net(x)
        preds = regress2class(r_out.data.squeeze(1))
        for i in range(len(batch_ids)):
            submission_rows.append([batch_ids[i], int(preds[i].item())])

submission = np.array(submission_rows, dtype=object)

try:
    preds_arr = submission[:, 1].astype(int)
    uniq, cnt = np.unique(preds_arr, return_counts=True)
    print("Predicted class distribution:", dict(zip(uniq.tolist(), cnt.tolist())))
except Exception:
    pass



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

assert len(df) == len(test_ids), "Submission row count does not match test set."

df = df.set_index("id_code").loc[list(map(str, test_ids))].reset_index()

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
