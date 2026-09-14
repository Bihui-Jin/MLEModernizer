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

0.9121102035104214

# 6. Current score

0.02917

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) remove the failing external `pip install` and add a robust weight-file discovery/fallback so the notebook no longer crashes when `../input/weights/...` is missing. I (2) make device selection automatic (`cuda` if available else `cpu`) to fix the “no NVIDIA driver” runtime error and ensure inference runs everywhere. I (3) fix dataset path handling by using the provided `../input/aptos2019-blindness-detection/...` structure with a safe fallback to `../input/...` if needed. Finally, I (4) ensure the transforms never return `None` (bug in `trim`) and write a non-empty `submission.csv` with the correct columns and row count.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the model is running with random weights (the expected weight file isn’t present), so the smallest meaningful improvement is to load a valid pretrained checkpoint if available and otherwise fall back to a “reasonable” prediction prior instead of random guesses. I keep your model and transforms unchanged, but I (1) expand weight-file discovery to pick up common EfficientNet-B4/B5 DR checkpoints that might already exist in your input dataset and (2) make weight loading tolerant (`strict=False`) so partially matching checkpoints can still be used rather than failing silently into random weights. If no weights are found, I replace random-weight inference with a deterministic label prior computed from `train.csv` (majority-class fallback), which typically scores above 0.0 and moves you toward the target while remaining legitimate (no test leakage). The submission writing and schema stay identical.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from producing almost-constant predictions (either because weights aren’t found/loaded or because the model output isn’t converted to the correct ordinal classes). I keep your model and transforms intact, but make two minimal changes that directly affect kappa: (1) always produce predictions via the model’s `final=True` regression head when weights are available (it’s explicitly designed for the final prediction), and (2) replace the fixed thresholds with thresholds fitted on the training set using out-of-fold style inference on a small held-out split to maximize quadratic weighted kappa (calibration only; architecture/training untouched). If no weights are found, we still fall back to the train-set majority class as before to ensure a valid non-empty submission.'
- What this solution (achieved 0.02917) has done: 'Your 0.0 score is most consistent with the model running without a real trained checkpoint (random/near-constant predictions), so the smallest meaningful way to move toward 0.912 is to (1) reliably load the correct weights if they exist and (2) avoid collapsing predictions when weights are missing by using a calibrated, deterministic prior distribution from `train.csv` instead of a single majority class. I keep your model, transforms, and inference path the same, but add a safer weight discovery that prefers checkpoints whose keys match your `ThreeStage_Model` backbone/heads. If no compatible weights are found, I generate predictions by sampling from the train label distribution with a fixed seed (legitimate, no test leakage) to improve QWK above 0.0 while staying minimal. I also ensure the submission ordering matches `test.csv` without creating duplicate rows during merge.'
- What this solution (achieved 0.02917) has done: 'Your current score (0.02917) is far below the target (0.9121), so we should improve performance with minimal, evaluation-aligned changes while preserving your model and inference flow. The biggest likely issue is that `regress2class()` thresholds are being applied to a regressor output scaled to `0..4.5`, but your defaults (and the calibration search start) are tuned closer to a `0..4` scale, which can collapse predictions and hurt QWK. I make the thresholds scale-consistent (default and calibration initialization), clamp regressor outputs to `[0, 4]` before thresholding, and slightly strengthen threshold calibration by using a stratified split (so the validation set reflects label distribution) without changing training/architecture. All I/O paths and submission formatting remain unchanged, and the script still produces `submission.csv`.'
- What this solution (achieved 0.02917) has done: 'Your score is far below the target, so the most likely bottleneck is prediction calibration rather than architecture. I keep your model and inference exactly the same, but replace the small greedy threshold search with a more robust, still-lightweight coordinate-descent calibration on a stratified validation split (same split idea, just better optimization), which directly targets QWK. I also ensure the regressor outputs are clamped to `[0, 4]` consistently during calibration and inference so thresholds operate on the correct scale. Everything else (paths, transforms, model, submission format) stays unchanged and it still write `submission.csv`.'
- What this solution (achieved 0.02917) has done: 'Your current score is far below the target, so the most likely issue is that inference is still effectively “near-random” because the checkpoint being loaded is not actually compatible (partial overlap can still yield unusable predictions). I make weight loading stricter in a minimal way: require that key tensor shapes match (not just key names), and only accept checkpoints that cover the backbone and final head with sufficient correctly-shaped parameters; otherwise we fall back to the (already legitimate) label-prior sampler. If weights are accepted, I also ensure we use the same image preprocessing for calibration/inference and compute thresholds using a slightly larger batch size to stabilize predictions (no change in model or training). These changes directly target improving QWK by ensuring we only run “real” model inference when the weights are truly compatible, rather than producing garbage outputs that hurt kappa.'
- What this solution (achieved 0.02917) has done: 'Your score is far below the target, so the most plausible issue is still “bad/near-constant predictions” from using an incompatible checkpoint or the wrong head, plus per-image inference overhead that can introduce subtle inconsistencies. I keep your exact model and transforms, but (1) tighten checkpoint acceptance so we only use weights that truly match the EfficientNet-B4 backbone + all heads (otherwise we deterministically fall back), (2) batch test inference with a DataLoader (same predictions, more stable + faster within limits), and (3) calibrate thresholds a bit more robustly by using more validation images (still stratified, still only calibration) and by ensuring thresholding is done on the exact `[0,4]` clamped regressor output used at test time. These are minimal, evaluation-aligned changes that should move QWK upward toward your target without changing architecture/training semantics. The script still writes a valid `submission.csv` with correct ordering and row count.'

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

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
threshold = [0.9, 1.8, 2.8, 3.7]


def regress2class(out: torch.Tensor) -> torch.Tensor:
    out = out.detach()
    out = torch.clamp(out, 0.0, 4.0)
    prediction = torch.zeros(out.size(0), dtype=torch.long)
    for i in range(4):
        prediction += (out >= threshold[i]).to(torch.long).cpu()
    return prediction


def ordinal2class_prob(out: torch.Tensor) -> torch.Tensor:
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out: torch.Tensor) -> torch.Tensor:
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)
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




## === cell 4
BASE = "../input/aptos2019-blindness-detection"
if not os.path.exists(BASE):
    BASE = "../input"
print("BASE:", BASE)

test_csv = os.path.join(BASE, "test.csv")
test_img_dir = os.path.join(BASE, "test_images")

if not os.path.exists(test_csv) and os.path.exists(
    os.path.join(BASE, "aptos2019-blindness-detection", "test.csv")
):
    BASE = os.path.join(BASE, "aptos2019-blindness-detection")
    test_csv = os.path.join(BASE, "test.csv")
    test_img_dir = os.path.join(BASE, "test_images")
print("Resolved BASE:", BASE)
print("test_csv:", test_csv)
print("test_img_dir exists:", os.path.isdir(test_img_dir))

test_df = pd.read_csv(test_csv)
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


def _extract_state_dict(obj):
    if (
        isinstance(obj, dict)
        and "state_dict" in obj
        and isinstance(obj["state_dict"], dict)
    ):
        return obj["state_dict"]
    if isinstance(obj, dict):
        return obj
    return None


def _normalize_state_dict_keys(sd):
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_sd[nk] = v
    return new_sd


def _shape_compatible_overlap(sd, model):
    msd = model.state_dict()
    ok = 0
    total = 0
    for k, v in sd.items():
        if k in msd:
            total += 1
            try:
                if tuple(v.shape) == tuple(msd[k].shape):
                    ok += 1
            except Exception:
                pass
    return ok, total


def _compat_score_for_checkpoint(path, model):
    try:
        ckpt = torch.load(path, map_location="cpu")
        sd = _extract_state_dict(ckpt)
        if sd is None:
            return -1
        sd = _normalize_state_dict_keys(sd)

        ok, total = _shape_compatible_overlap(sd, model)
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(sd.keys())
        overlap = len(model_keys & ckpt_keys)

        need = ["backbone", "classifier", "regressor", "ordinal", "final_regressor"]
        bonus = 0
        for n in need:
            if any(k.startswith(n + ".") for k in ckpt_keys):
                bonus += 200

        return bonus + overlap + 10 * ok
    except Exception:
        return -1


def find_weight_file(model):
    candidates = [
        "../input/weights/B4_3stage_54epoch_CLAHE.pkl",
        "../input/weights/B4_3stage_54epoch_CLAHE.pth",
        "../input/weights/B4_3stage_54epoch_CLAHE.pt",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    name_patterns = [
        "*B4*3stage*CLAHE*.pth",
        "*B4*3stage*CLAHE*.pkl",
        "*B4*3stage*CLAHE*.pt",
        "*3stage*.pth",
        "*three*stage*.pth",
        "*efficientnet*b4*.pth",
        "*effnet*b4*.pth",
        "*.pth",
        "*.pt",
        "*.pkl",
    ]
    search_roots = ["../input"]
    hits = []
    for root in search_roots:
        for pat in name_patterns:
            hits.extend(glob.glob(os.path.join(root, "**", pat), recursive=True))
    hits = sorted(list(set(hits)))

    if not hits:
        return None

    scored = []
    for p in hits:
        sc = _compat_score_for_checkpoint(p, model)
        if sc >= 0:
            scored.append((sc, p))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], len(x[1])))
    return scored[0][1]


weight_path = find_weight_file(net)
weights_loaded = False
loaded_state_for_infer = None

if weight_path is None:
    print("WARNING: weight file not found.")
else:
    print("Loading weights:", weight_path)
    state_obj = torch.load(weight_path, map_location="cpu")
    state = _extract_state_dict(state_obj)
    if state is None:
        print("WARNING: checkpoint did not contain a usable state_dict; skipping.")
    else:
        state = _normalize_state_dict_keys(state)

        ok, total = _shape_compatible_overlap(state, net)
        missing, unexpected = net.load_state_dict(state, strict=False)

        model_key_overlap = len(set(net.state_dict().keys()) & set(state.keys()))
        print(
            f"Checkpoint key-overlap={model_key_overlap}, shape-compatible={ok} (out of {total} matching-name tensors)"
        )

        ckpt_keys = set(state.keys())
        need_prefixes = [
            "backbone.",
            "classifier.",
            "regressor.",
            "ordinal.",
            "final_regressor.",
        ]
        has_all_parts = all(
            any(k.startswith(p) for k in ckpt_keys) for p in need_prefixes
        )

        if (not has_all_parts) or (ok < 600):
            print(
                f"WARNING: checkpoint deemed incompatible (has_all_parts={has_all_parts}, shape_ok={ok}); "
                f"treating as not loaded to avoid near-random predictions."
            )
            weights_loaded = False
        else:
            weights_loaded = True
            loaded_state_for_infer = weight_path
            if missing:
                print(
                    f"NOTE: missing keys while loading ({len(missing)}): e.g. {missing[:5]}"
                )
            if unexpected:
                print(
                    f"NOTE: unexpected keys while loading ({len(unexpected)}): e.g. {unexpected[:5]}"
                )

net = net.to(device)
net.eval()

fallback_label = 0
fallback_probs = None

train_csv = os.path.join(BASE, "train.csv")
if not os.path.exists(train_csv) and os.path.exists(
    os.path.join(os.path.dirname(BASE), "train.csv")
):
    train_csv = os.path.join(os.path.dirname(BASE), "train.csv")

if os.path.exists(train_csv):
    try:
        train_df = pd.read_csv(train_csv)
        if "diagnosis" in train_df.columns:
            vc = train_df["diagnosis"].value_counts().sort_index()
            fallback_label = int(vc.idxmax())
            probs = np.zeros(5, dtype=np.float64)
            for k, v in vc.items():
                if 0 <= int(k) <= 4:
                    probs[int(k)] = float(v)
            probs = probs / probs.sum()
            fallback_probs = probs
    except Exception as e:
        print("Could not compute fallback priors from train.csv:", repr(e))

print(
    "weights_loaded:",
    weights_loaded,
    "| weight_path_used:",
    loaded_state_for_infer,
    "| fallback_label:",
    fallback_label,
)
print("fallback_probs:", fallback_probs)




## === cell 5
class ImgDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.ids = df["id_code"].astype(str).values
        self.y = df["diagnosis"].values if "diagnosis" in df.columns else None
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        path = os.path.join(self.img_dir, f"{idx}.png")
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        if self.y is None:
            return x, idx
        return x, int(self.y[i])


def fit_thresholds_from_val(y_true, y_pred_cont):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred_cont = np.asarray(y_pred_cont, dtype=np.float32)

    y_pred_cont = np.clip(y_pred_cont, 0.0, 4.0)

    def kappa_for(t):
        global threshold
        threshold = [float(x) for x in t]
        preds = regress2class(torch.tensor(y_pred_cont, dtype=torch.float32)).numpy()
        return cohen_kappa_score(y_true, preds, weights="quadratic")

    starts = [
        [0.9, 1.8, 2.8, 3.7],
        [0.5, 1.5, 2.5, 3.5],
        [0.7, 1.7, 2.7, 3.7],
        [1.0, 2.0, 3.0, 3.8],
    ]

    best_t = starts[0]
    best = -1.0
    for st in starts:
        sc = kappa_for(st)
        if sc > best:
            best, best_t = sc, st

    steps = [0.2, 0.1, 0.05, 0.02]
    for step in steps:
        improved = True
        it = 0
        while improved and it < 80:
            improved = False
            it += 1
            for k in range(4):
                cur = best_t[k]
                for cand in (cur - step, cur + step, cur - 2 * step, cur + 2 * step):
                    t = best_t.copy()
                    t[k] = float(cand)
                    t[0] = max(0.0, min(t[0], 4.0))
                    t[1] = max(t[0] + 1e-3, min(t[1], 4.0))
                    t[2] = max(t[1] + 1e-3, min(t[2], 4.0))
                    t[3] = max(t[2] + 1e-3, min(t[3], 4.0))
                    sc = kappa_for(t)
                    if sc > best + 1e-8:
                        best, best_t = sc, t
                        improved = True
    return [float(x) for x in best_t], float(best)


if weights_loaded and os.path.exists(train_csv):
    full_train = pd.read_csv(train_csv)
    full_train["id_code"] = full_train["id_code"].astype(str)

    train_img_dir = os.path.join(BASE, "train_images")
    if not os.path.isdir(train_img_dir) and os.path.isdir(
        os.path.join(os.path.dirname(BASE), "train_images")
    ):
        train_img_dir = os.path.join(os.path.dirname(BASE), "train_images")

    y_all = full_train["diagnosis"].astype(int).values

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
    tr_idx, va_idx = next(sss.split(np.zeros(len(y_all)), y_all))
    val_df = full_train.iloc[va_idx].reset_index(drop=True)

    val_ds = ImgDataset(val_df, train_img_dir, transform)
    val_loader = DataLoader(
        val_ds,
        batch_size=24,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    y_true = []
    y_pred_cont = []

    net.eval()
    with torch.inference_mode():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = net(xb, final=True).squeeze(1).detach().float().cpu().numpy()
            out = np.clip(out, 0.0, 4.0)
            y_pred_cont.extend(out.tolist())
            y_true.extend([int(v) for v in yb])

    new_t, best_kappa = fit_thresholds_from_val(np.array(y_true), np.array(y_pred_cont))
    threshold = new_t
    print("Calibrated thresholds:", threshold, "| val QWK:", best_kappa)
else:
    print(
        "Skipping threshold calibration (no usable weights or no train.csv). Using default thresholds:",
        threshold,
    )




## === cell 6
class TestImgDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = np.asarray(ids, dtype=str)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        path = os.path.join(self.img_dir, f"{idx}.png")
        if not os.path.exists(path):
            return None, idx
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, idx


def _collate_test(batch):
    xs = []
    ids = []
    for x, idx in batch:
        ids.append(idx)
        xs.append(x)
    return xs, ids


submission_rows = []
missing_images = 0

prior_rng = np.random.RandomState(42)
if fallback_probs is None:
    fallback_probs = np.array([1, 0, 0, 0, 0], dtype=np.float64)

test_ds = TestImgDataset(test_ids, test_img_dir, transform)
test_loader = DataLoader(
    test_ds,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=(device == "cuda"),
    collate_fn=_collate_test,
)

net.eval()
with torch.inference_mode():
    for xs, ids in test_loader:
        if not weights_loaded:
            for idx in ids:
                image_name = os.path.join(test_img_dir, f"{idx}.png")
                if not os.path.exists(image_name):
                    missing_images += 1
                pred_f = int(prior_rng.choice(np.arange(5), p=fallback_probs))
                submission_rows.append([idx, pred_f])
            continue

        present_mask = [x is not None for x in xs]
        if any(not m for m in present_mask):
            for m, idx in zip(present_mask, ids):
                if not m:
                    missing_images += 1

        if any(present_mask):
            xb = torch.stack([x for x in xs if x is not None], dim=0).to(device)
            out = net(xb, final=True).squeeze(1)
            preds = regress2class(out).numpy().astype(int).tolist()
        else:
            preds = []

        pi = 0
        for m, idx in zip(present_mask, ids):
            if not m:
                pred_f = int(prior_rng.choice(np.arange(5), p=fallback_probs))
                submission_rows.append([idx, pred_f])
            else:
                submission_rows.append([idx, int(preds[pi])])
                pi += 1

print("Missing images:", missing_images)
submission = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])



## === cell 7
submission = submission.drop_duplicates(subset=["id_code"], keep="first")
submission = pd.DataFrame({"id_code": test_ids}).merge(
    submission, on="id_code", how="left"
)
submission["diagnosis"] = submission["diagnosis"].fillna(fallback_label).astype(int)
submission = submission[["id_code", "diagnosis"]]

assert len(submission) == len(test_ids), "Submission row count mismatch with test set."
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
