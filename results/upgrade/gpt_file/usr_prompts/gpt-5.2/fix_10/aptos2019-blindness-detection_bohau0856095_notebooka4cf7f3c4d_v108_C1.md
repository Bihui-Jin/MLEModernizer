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

0.9257328956321108

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) remove the failing custom `pip install` and instead use the already-installed `timm` to avoid missing wheel/runtime issues, (2) make device selection robust so the notebook runs on CPU-only environments (fixing the CUDA driver error), and (3) fix the missing weights path by loading from a set of likely Kaggle input locations when available, otherwise falling back to a randomly initialized model so a valid (non-empty) `submission.csv` is always produced. I also fix a small logic bug in `trim()` that can return `None`, which breaks the transform pipeline and can silently empty the submission loop. These changes preserve the model/feature logic and only add defensive I/O/device handling so the pipeline completes end-to-end.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with the model running without pretrained weights (random predictions collapse QWK), so the smallest change that should move you toward the 0.925 target is to reliably locate and load the correct checkpoint when it exists in common Kaggle input locations. I keep the exact same model and inference logic, but expand the weight search to include the dataset directory and recursively search under `/kaggle/input` for the expected filename, then load either `state_dict` or full checkpoint formats safely. I also ensure `regress2class()` always returns integer class indices on CPU deterministically (no `.data`), preserving semantics but avoiding edge cases. If weights still aren’t found, the script still produce a valid `submission.csv` as before.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the model running without the intended pretrained checkpoint (random predictions yield near-zero QWK). The smallest change likely to move you toward the 0.9257 target is to (1) broaden and harden checkpoint discovery to also accept common alternate filenames and locations, and (2) make weight loading robust to common training wrappers (e.g., `module.` prefixes) while still using the exact same model and inference logic. I also ensure we never skip test rows due to missing-image handling (instead, we fill a default prediction) so the submission always aligns perfectly with `test.csv`. These are defensive I/O/loading changes only; the model architecture, transforms, and prediction rule remain the same.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the intended checkpoint still isn’t being found/loaded, so the smallest change to move toward the 0.9257 target is to (a) broaden checkpoint discovery to include any `.pth/.pt/.pkl` under common Kaggle data roots, and (b) make state_dict loading robust to common nesting keys and prefixes (including `model.`) while still requiring shape-matching strict loads. I also add a safety print of missing/unexpected keys when a near-match load is attempted, so you can confirm whether the correct weights are being applied. Core model, transforms, and prediction rule remain unchanged; this only hardens weight loading so inference uses the trained model rather than random initialization. The script still always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the intended checkpoint still not being loaded, so the minimal change to move toward the 0.9257 target is to make checkpoint discovery/loading actually succeed. I (1) fix the most likely data_root issue (your code can pick `/kaggle/data` or `/kaggle/input` where `test_images/` may not exist) by validating both `test.csv` and `test_images/`, and (2) harden `torch.load` for modern PyTorch by explicitly handling `weights_only=True/False` and common checkpoint key nesting so the same model can load trained weights instead of staying random. I also add a deterministic fallback to use the model’s class head if regression output is degenerate (NaNs/unloaded weights), which preserves semantics (still outputs 0-4) but prevents pathological all-zero submissions. Core model/transform/prediction logic remains the same; this is defensive I/O and loading so inference uses the trained network.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the intended trained checkpoint is still not being loaded (so predictions are effectively random / collapsed), so the smallest change to move toward the 0.9257 target is to make weight discovery and loading actually succeed. I (1) broaden checkpoint discovery to include common Kaggle dataset layouts and accept any plausible `.pth/.pt/.pkl` in the competition folder (not just names containing “b4/3stage”), and (2) make state-dict extraction handle more common nesting keys while still requiring a strict shape match (so we don’t silently load the wrong thing). I also validate that `data_root` points to a folder that contains both `test.csv` and `test_images/`, because a wrong root can cause missing-image fallbacks that destroy score. Core model, transforms, and prediction logic remain unchanged; these are defensive I/O/loading fixes aimed specifically at moving the score up from 0.0.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the model still running effectively untrained or with a mismatched checkpoint load, so the smallest meaningful step toward the 0.9257 target is to make checkpoint discovery/loading actually succeed (and verify it did). I (1) constrain checkpoint auto-discovery to only files that plausibly match your `ThreeStage_Model` keys (instead of “any .pth”), (2) add robust state-dict extraction/cleanup to handle common wrappers while keeping strict shape matching, and (3) add a hard guard that reports `loaded=False` unless we can confirm a real strict load occurred. This keeps the same model, transforms, and regression-to-class rule; it only fixes weight loading so predictions are not random/collapsed. The script still always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the model not actually loading the intended trained checkpoint, so the smallest change likely to move you toward the 0.9257 target is to make checkpoint discovery succeed in your current environment and to ensure we only accept a strict, shape-matching load. I broaden discovery to include any `.pth/.pt/.pkl` under the APTOS dataset folder and Kaggle roots, then select the first checkpoint whose extracted state_dict is “plausible” for `ThreeStage_Model` and can be loaded strictly. This preserves the exact model, transforms, and regression-to-class prediction semantics; it only fixes weight I/O so inference uses trained weights instead of random initialization. The script still always produce a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is still most consistent with “running untrained” because no matching checkpoint is actually being loaded, so the minimal score-improving change is to make checkpoint discovery/load succeed (or loudly prove it cannot). I keep your exact model, transforms, and regression→class thresholds, but (1) fix the most likely root cause: the notebook never searches the real dataset roots you actually have (`/kaggle/data/...` and `/kaggle/input/...` without the extra nested folder), and (2) add a strict-but-robust loader that can handle common checkpoint formats (plain state_dict, wrapped under keys, `module.` prefixes) and confirms a real strict load happened. If no valid checkpoint exists in the environment, the code still produce a valid `submission.csv`, but now you reliably get a trained-weight submission whenever the weights are present, moving the score toward your 0.9257 target.'

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
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2

from sklearn.metrics import cohen_kappa_score
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    out = out.detach()
    prediction = torch.zeros(out.size(0), device=out.device, dtype=torch.long)
    for i in range(4):
        prediction += (out >= threshold[i]).squeeze().to(torch.long)
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5).to(out.device)
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
DATA_ROOT_CANDIDATES = [
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection/aptos2019-blindness-detection",
    "../kaggle/data/aptos2019-blindness-detection",
    "../kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection",
]


def _looks_like_aptos_root(p):
    return os.path.exists(os.path.join(p, "test.csv")) and os.path.isdir(
        os.path.join(p, "test_images")
    )


data_root = None
for p in DATA_ROOT_CANDIDATES:
    if _looks_like_aptos_root(p):
        data_root = p
        break

if data_root is None:
    for umbrella in ["/kaggle/input", "/kaggle/data", "../input", "../kaggle/data"]:
        cand = os.path.join(umbrella, "aptos2019-blindness-detection")
        if _looks_like_aptos_root(cand):
            data_root = cand
            break
        cand2 = os.path.join(
            umbrella, "aptos2019-blindness-detection", "aptos2019-blindness-detection"
        )
        if _looks_like_aptos_root(cand2):
            data_root = cand2
            break

if data_root is None:
    data_root = "../input/aptos2019-blindness-detection"

test_csv_path = os.path.join(data_root, "test.csv")
test_img_dir = os.path.join(data_root, "test_images")

print("Using data_root:", data_root)
print("test_csv_path exists:", os.path.exists(test_csv_path))
print("test_img_dir exists:", os.path.isdir(test_img_dir))

test_ids = pd.read_csv(test_csv_path)["id_code"].values

input_size = 384
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

EXPECTED_WEIGHT_NAMES = [
    "B4_3stage_15epoch_320finetune.pkl",
    "B4_3stage_15epoch_320finetune.pth",
    "B4_3stage_15epoch_320finetune.pt",
    "b4_3stage_15epoch_320finetune.pkl",
    "b4_3stage_15epoch_320finetune.pth",
    "b4_3stage_15epoch_320finetune.pt",
]

WEIGHT_CANDIDATES = []
for name in EXPECTED_WEIGHT_NAMES:
    WEIGHT_CANDIDATES.extend(
        [
            "../input/weights/" + name,
            "/kaggle/input/weights/" + name,
            "/kaggle/data/weights/" + name,
            "../kaggle/data/weights/" + name,
            os.path.join(data_root, name),
            os.path.join(data_root, "weights", name),
        ]
    )


def _recursive_find(root, filenames, max_hits=3):
    hits = []
    if not os.path.exists(root):
        return hits
    fn_set = set(filenames)
    for dirpath, dirnames, files in os.walk(root):
        inter = fn_set.intersection(files)
        if inter:
            for fn in sorted(inter):
                hits.append(os.path.join(dirpath, fn))
                if len(hits) >= max_hits:
                    return hits
    return hits


def _recursive_find_any_ckpt(root, exts=(".pth", ".pt", ".pkl"), max_hits=200):
    hits = []
    if not os.path.exists(root):
        return hits
    exts = tuple(e.lower() for e in exts)
    for dirpath, dirnames, files in os.walk(root):
        for fn in files:
            lfn = fn.lower()
            if lfn.endswith(exts):
                hits.append(os.path.join(dirpath, fn))
                if len(hits) >= max_hits:
                    return hits
    return hits


def _strip_prefix_from_state_dict(sd, prefix):
    if not isinstance(sd, dict):
        return sd
    if not any(k.startswith(prefix) for k in sd.keys()):
        return sd
    return {k[len(prefix) :]: v for k, v in sd.items()}


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "ema_state_dict",
            "model_ema",
            "params",
        ]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict) and len(ckpt_obj[k]) > 0:
                return ckpt_obj[k]
        tensor_like = (
            all(hasattr(v, "shape") for v in ckpt_obj.values()) if ckpt_obj else False
        )
        if tensor_like and len(ckpt_obj) > 0:
            return ckpt_obj
    return None


def _state_dict_plausible_for_model(sd, model, min_key_hits=10):
    if not isinstance(sd, dict) or len(sd) == 0:
        return False
    model_keys = set(model.state_dict().keys())
    sd_keys = set(sd.keys())
    hit = len(model_keys.intersection(sd_keys))
    if hit >= min_key_hits:
        return True
    for prefix in ["module.", "model.", "net."]:
        stripped = _strip_prefix_from_state_dict(sd, prefix)
        if stripped is not sd:
            hit2 = len(model_keys.intersection(set(stripped.keys())))
            if hit2 >= min_key_hits:
                return True
    return False


def _load_state_dict_safely_strict(model, sd):
    if not isinstance(sd, dict) or len(sd) == 0:
        return False
    try:
        model.load_state_dict(sd, strict=True)
        return True
    except Exception:
        pass
    for prefix in ["module.", "model.", "net."]:
        try:
            model.load_state_dict(
                _strip_prefix_from_state_dict(sd, prefix), strict=True
            )
            return True
        except Exception:
            pass
    return False


def _torch_load_robust(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu", weights_only=False)


weight_path = None
for wp in WEIGHT_CANDIDATES:
    if os.path.exists(wp):
        weight_path = wp
        break

if weight_path is None:
    for root in [
        data_root,
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/input",
        "../input",
        "/kaggle/data",
        "../kaggle/data",
    ]:
        hits = _recursive_find(root, EXPECTED_WEIGHT_NAMES, max_hits=1)
        if hits:
            weight_path = hits[0]
            break

loaded = False
load_diag = ""
best_ckpt_scan = []

if weight_path is None:
    scan_roots = [
        os.path.join(data_root, "weights"),
        data_root,
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/input",
        "../input",
        "/kaggle/data",
        "../kaggle/data",
    ]
    seen = set()
    for sr in scan_roots:
        for p in _recursive_find_any_ckpt(
            sr, exts=(".pth", ".pt", ".pkl"), max_hits=200
        ):
            if p not in seen:
                seen.add(p)
                best_ckpt_scan.append(p)

    def _ckpt_rank(p):
        lp = os.path.basename(p).lower()
        score = 0
        for tok, w in [
            ("3stage", 5),
            ("b4", 4),
            ("efficientnet", 2),
            ("finetune", 2),
            ("epoch", 1),
        ]:
            if tok in lp:
                score += w
        return (-score, len(lp))

    best_ckpt_scan = sorted(best_ckpt_scan, key=_ckpt_rank)

    for cand_path in best_ckpt_scan:
        try:
            ckpt_obj = _torch_load_robust(cand_path)
            sd = _extract_state_dict(ckpt_obj)
            if sd is None:
                continue
            if not _state_dict_plausible_for_model(sd, net, min_key_hits=10):
                continue
            if _load_state_dict_safely_strict(net, sd):
                weight_path = cand_path
                loaded = True
                load_diag = "Strict load succeeded (autodiscovered ckpt)"
                break
        except Exception:
            continue

if not loaded and weight_path is not None:
    try:
        ckpt_obj = _torch_load_robust(weight_path)
        sd = _extract_state_dict(ckpt_obj)
        if sd is None:
            loaded = False
            load_diag = "Checkpoint did not contain a recognizable state_dict"
        else:
            if not _state_dict_plausible_for_model(sd, net, min_key_hits=10):
                loaded = False
                load_diag = (
                    "Found checkpoint but keys do not match ThreeStage_Model (skipped)"
                )
            else:
                loaded = _load_state_dict_safely_strict(net, sd)
                load_diag = (
                    "Strict load succeeded"
                    if loaded
                    else "Strict load failed (shape/key mismatch)"
                )
    except Exception as e:
        loaded = False
        load_diag = f"Exception during load: {repr(e)}"

if loaded:
    print("Loaded pretrained weights from:", weight_path)
else:
    if weight_path is not None:
        print("WARNING:", load_diag)
        print("WARNING: checkpoint path was:", weight_path)
    else:
        print(
            "WARNING: pretrained weights not found in candidates or autodiscovery scan."
        )
        print("Scanned ckpt candidates (first 10):", best_ckpt_scan[:10])
    print("WARNING: running with randomly initialized model (score may be ~0.0).")

net = net.to(device)
net.eval()



## === cell 5
submission = []
missing_images = 0
used_classifier_fallback = 0

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        image_name = os.path.join(test_img_dir, f"{idx}.png")
        pred_int = 0

        if not os.path.exists(image_name):
            missing_images += 1
        else:
            try:
                img = Image.open(image_name).convert("RGB")
                img = transform(img).unsqueeze(0).to(device)
                c_out, r_out, _ = net(img)

                r_val = r_out.squeeze(1)
                if torch.isnan(r_val).any() or torch.isinf(r_val).any():
                    pred_int = int(torch.argmax(c_out, dim=1).item())
                    used_classifier_fallback += 1
                else:
                    pred = regress2class(r_val)
                    pred_int = int(pred.item())
            except Exception:
                missing_images += 1
                pred_int = 0

        submission.append([idx, pred_int])

if missing_images > 0:
    print(
        f"WARNING: {missing_images} test images were missing/unreadable; predicted 0 for those rows."
    )
if used_classifier_fallback > 0:
    print(
        f"WARNING: classifier fallback used for {used_classifier_fallback} rows due to degenerate regressor output."
    )

submission = np.array(submission, dtype=object)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df = df.set_index("id_code").reindex(test_ids).reset_index()
df["diagnosis"] = df["diagnosis"].astype(int)
df.to_csv("submission.csv", index=False)

out_df = pd.read_csv("submission.csv")
assert list(out_df.columns) == ["id_code", "diagnosis"]
assert len(out_df) == len(test_ids)
print(out_df.head())
print("Wrote submission.csv with", len(out_df), "rows")
print(
    "Weights loaded:", loaded, "| weight_path:", weight_path, "| load_diag:", load_diag
)
