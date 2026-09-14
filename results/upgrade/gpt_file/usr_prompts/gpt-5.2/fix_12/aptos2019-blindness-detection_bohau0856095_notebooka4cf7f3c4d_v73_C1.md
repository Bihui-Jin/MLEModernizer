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

0.922894019000856

# 6. Current score

0.13625

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing weights crash by making the script robust to the absence of `../input/weights/*` (so it still runs end-to-end), and I also fix the CUDA-only device setting so it works on Kaggle CPU-only sessions. To avoid creating an empty submission, I ensure the test IDs are loaded correctly as strings and wrap image loading/inference with a safe fallback prediction if any single file fails. Finally, I keep your model/transform logic intact, but when weights are unavailable I fall back to a deterministic, label-distribution prior computed from `train.csv` so the submission is valid and non-empty (score won’t be good without weights, but it run and submit).'
- What this solution (achieved -0.14491) has done: 'Your current 0.0 score is driven by always falling back to a single “prior class” because the intended pretrained weights aren’t being found/loaded, so predictions become nearly constant and kappa collapses. The smallest score-improving change is to (1) search for the weights file in the actual dataset tree you have mounted (including `/kaggle/data/...`), and (2) if weights still aren’t present, use a deterministic, per-image heuristic based on simple intensity/contrast features to produce non-constant ordinal predictions (still legitimate, fast, and preserves your model logic when weights exist). I keep your model, transforms, and inference path unchanged when weights load; the heuristic only activates when `loaded_weights=False`. This should move the score upward toward your target without altering the core deep model behavior.'
- What this solution (achieved 0.05009) has done: 'Your score is extremely low because the model is effectively untrained (weights aren’t available/loaded), so even with the heuristic fallback the predictions won’t correlate well with labels and QWK collapses. The minimal change that legitimately boosts score toward your target without changing your model/training logic is to load an ImageNet-pretrained EfficientNet backbone when your finetuned weights are missing, so the regressor/heads operate on meaningful features (still the same architecture and inference path). I keep your exact transforms, thresholds, and prediction decoding; only the `pretrained` flag is conditionally enabled as a fallback when `loaded_weights=False`. This should move QWK sharply upward compared to the current heuristic-only path, while remaining deterministic and within the runtime limit.'
- What this solution (achieved -0.01033) has done: 'Your current score is far below target because the model usually runs with missing finetuned weights, and the current fallback only enables ImageNet pretrained weights on one backbone (B4) while inference actually uses only the regressor head output, which is poorly calibrated without the finetuned weights. I keep your exact model/transform/inference logic, but make the fallback stronger and still architecture-identical by (1) also enabling ImageNet pretrained weights for the EfficientNet-B4 backbone used in inference (already) and (2) switching inference to the model’s `final=True` path (same network, already implemented) only when finetuned weights are missing, so the prediction uses all three heads + final regressor instead of the weak regressor head alone. This is a minimal, metric-aligned change that should legitimately increase QWK (toward your target) without altering the training approach or adding new techniques. Submission writing stays the same and remains deterministic.'
- What this solution (achieved -0.05739) has done: 'Your score is far below the target, so we should improve it while keeping your architecture and inference semantics intact. The biggest likely issue is that the finetuned weights still aren’t being found/loaded in this environment, causing the weak fallback path to dominate; I add an on-disk search for the expected weights filename under the dataset root (and common Kaggle roots) without changing how weights are applied. Next, because QWK is highly sensitive to class distribution, I keep your regression output but calibrate the fixed `threshold` values using the training set label distribution (quantile-based thresholds) when finetuned weights are missing, which typically improves ordinal agreement without changing the model or loss. Finally, I make submission order exactly match `test.csv` (already mostly true) and keep the existing heuristic/prior fallbacks only for genuine read/inference failures.'
- What this solution (achieved 0.03332) has done: 'Your current score is far below the target, so we should improve it with the smallest changes that keep your model/inference logic intact. The biggest issue is that when finetuned weights are missing, your continuous regressor outputs are being discretized with thresholds derived only from the training label histogram, which is usually poorly calibrated for ImageNet-pretrained (non-finetuned) outputs and can collapse QWK. I keep your model and `final=True` fallback exactly as-is, but fit the 4 thresholds on a small deterministic validation split by directly maximizing QWK (a lightweight grid search around the existing thresholds). This only activates when weights are not loaded, stays within runtime, and should move the score upward toward your target while preserving evaluation semantics.'
- What this solution (achieved 0.07147) has done: 'Your score is far below the target, so the goal is to legitimately increase QWK with the smallest change that doesn’t alter your model/training semantics. The biggest current issue is that the “threshold fit” uses a random sample and evaluates on the same images it fits on, which can yield unstable/overfit thresholds that generalize poorly to test; we instead do a deterministic train/val split and fit thresholds on train-split while selecting by QWK on val-split (only when finetuned weights are missing). This keeps the exact model, transforms, and regress-to-class logic unchanged, but makes the fallback discretization calibration more reliable and usually improves QWK. Everything else (weight search/loading, inference path, submission format) remains the same and still writes a valid `submission.csv`.'
- What this solution (achieved 0.10448) has done: 'Your score is still far below the target, so we should cautiously increase QWK with the smallest change that improves ordinal calibration without changing your model or training/inference structure. Right now, when finetuned weights are missing, you fit thresholds on a small subset using a coarse grid; a very small, metric-aligned improvement is to fit thresholds using a deterministic 1D coordinate search that directly maximizes QWK on a held-out validation split, which is typically more effective than a coarse 4D grid at similar runtime. This keeps the exact “regression output + fixed thresholds => class” evaluation semantics intact, and only changes how thresholds are calibrated in the no-weights fallback. I also ensure threshold search remains bounded and deterministic to stay within the 600s limit and avoid instability.'
- What this solution (achieved 0.06133) has done: 'Your current score is far below target, so we should improve QWK in the no-finetuned-weights path with the smallest metric-aligned change. Right now you discretize the regression output with thresholds fit on a small split, but the regressor’s scale can drift; we can stabilize this by learning a simple monotonic calibration (affine scale + shift) on the fit split before threshold search, then fitting thresholds on the calibrated outputs using the same coordinate-ascent routine. This keeps your model, transforms, inference flow, and “regression → thresholds → class” semantics intact, only improving calibration when `loaded_weights=False`. The weights-loaded path and submission formatting remain unchanged, and the script still runs end-to-end within time.'
- What this solution (achieved 0.09162) has done: 'Your current gap to the target is very large (0.06133 vs 0.92289), and the main limiter is that the finetuned weights still aren’t being found/loaded, so you’re effectively relying on an ImageNet-pretrained fallback that won’t reach the target. I make the smallest change that increases the chance of actually loading the intended finetuned weights by broadening and prioritizing the on-disk weight search to include common Kaggle dataset structures and any `.pkl` matching the expected stem, while keeping the same model/forward/inference logic. Additionally, when weights are missing, I fit the affine calibration + thresholds using a slightly larger (still deterministic) subset to reduce variance and improve QWK without changing the architecture or inference semantics. The weights-loaded path (which is what can realistically approach your target) remains identical except for more robust weight discovery.'
- What this solution (achieved 0.13625) has done: 'Your score is far below the target, and the biggest likely blocker is still that the finetuned weights are not being loaded, so you’re effectively submitting predictions from an ImageNet-pretrained fallback that cannot approach ~0.92 QWK. I make the smallest change that increases the chance of actually loading the correct checkpoint by expanding the weight-file discovery to include any `.pth/.pt/.bin` and any file containing key substrings (b4/3stage/finetune/512) under the dataset roots, while keeping the same model and `load_state_dict` logic. If weights remain missing, I keep your exact inference semantics but make the no-weights calibration use a slightly larger deterministic fit/val subset to reduce variance (still fast enough) and I also ensure inference uses `model.eval()` + `torch.inference_mode()` for deterministic, stable outputs (no semantic change). These changes should move QWK upward toward the target by prioritizing the “weights-loaded” path and stabilizing the fallback path without altering architecture/training/loss.'

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
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2

from sklearn.metrics import cohen_kappa_score
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
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
    pred_prob = torch.zeros((out.size(0), 5)).to(out.device)
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
    def __init__(self, pretrained_backbone: bool = False):
        super(Regressor, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b5_ns(
            pretrained=pretrained_backbone
        )
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None, pretrained_backbone: bool = False):
        super(ThreeStage_Model, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b4_ns(
            pretrained=pretrained_backbone
        )
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
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data/input/aptos2019-blindness-detection",
]

data_root = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        data_root = p
        break

if data_root is None:
    raise FileNotFoundError(
        "Could not find aptos2019-blindness-detection dataset under expected locations"
    )

test_csv_path = os.path.join(data_root, "test.csv")
train_csv_path = os.path.join(data_root, "train.csv")
test_img_dir = os.path.join(data_root, "test_images")
train_img_dir = os.path.join(data_root, "train_images")

test_df = pd.read_csv(test_csv_path)
test_ids = test_df["id_code"].astype(str).tolist()

input_size = 512
transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

WEIGHT_FILENAME = "B4_3stage_8epoch_finetune512.pkl"

WEIGHT_CANDIDATES = [
    os.path.join(data_root, WEIGHT_FILENAME),
    os.path.join(data_root, "weights", WEIGHT_FILENAME),
    os.path.join("/kaggle/input", WEIGHT_FILENAME),
    os.path.join("/kaggle/input", "weights", WEIGHT_FILENAME),
    os.path.join("/kaggle/data", WEIGHT_FILENAME),
    os.path.join("/kaggle/data", "weights", WEIGHT_FILENAME),
    os.path.join("/kaggle/data/input", WEIGHT_FILENAME),
    os.path.join("/kaggle/data/input", "weights", WEIGHT_FILENAME),
    "../input/weights/" + WEIGHT_FILENAME,
    "/kaggle/input/weights/" + WEIGHT_FILENAME,
    "/kaggle/data/weights/" + WEIGHT_FILENAME,
    "/kaggle/data/input/weights/" + WEIGHT_FILENAME,
]


def find_weight_file():
    for wp in WEIGHT_CANDIDATES:
        if wp and os.path.exists(wp):
            return wp

    expected_stem = os.path.splitext(WEIGHT_FILENAME)[0]
    expected_tokens = ["b4", "3stage", "finetune", "512"]
    exts = {".pkl", ".pth", ".pt", ".bin"}

    search_roots = [
        data_root,
        "/kaggle/input",
        "/kaggle/data",
        "/kaggle/working",
    ]
    seen = set()
    best_token_match = None
    best_token_score = -1

    for root in search_roots:
        if not root or root in seen or not os.path.exists(root):
            continue
        seen.add(root)
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                if WEIGHT_FILENAME in filenames:
                    return os.path.join(dirpath, WEIGHT_FILENAME)

                for fn in filenames:
                    lfn = fn.lower()
                    ext = os.path.splitext(lfn)[1]
                    if ext not in exts:
                        continue

                    full = os.path.join(dirpath, fn)

                    if expected_stem.lower() in lfn:
                        return full

                    score = sum(tok in lfn for tok in expected_tokens)
                    if score > best_token_score:
                        best_token_score = score
                        best_token_match = full
        except Exception:
            pass

    if best_token_score >= 2 and best_token_match is not None:
        return best_token_match

    return None


weights_path = find_weight_file()

loaded_weights = False
use_pretrained_backbone_fallback = weights_path is None

net = ThreeStage_Model(pretrained_backbone=use_pretrained_backbone_fallback)

if weights_path is not None:
    state = torch.load(weights_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    try:
        net.load_state_dict(state, strict=True)
    except RuntimeError:
        cleaned = {}
        for k, v in state.items():
            ck = k
            if ck.startswith("module."):
                ck = ck[len("module.") :]
            cleaned[ck] = v
        net.load_state_dict(cleaned, strict=False)
    loaded_weights = True

net = net.to(device)
net.eval()

train_df = pd.read_csv(train_csv_path)
label_prior = (
    train_df["diagnosis"]
    .value_counts(normalize=True)
    .reindex([0, 1, 2, 3, 4], fill_value=0.0)
    .values
)
prior_class = int(np.argmax(label_prior))


def _predict_outputs_for_df(
    model: torch.nn.Module,
    df_in: pd.DataFrame,
    img_dir: str,
):
    outs = []
    ys = []
    with torch.inference_mode():
        for _, row in df_in.iterrows():
            img_path = os.path.join(img_dir, f"{row['id_code']}.png")
            try:
                img_pil = Image.open(img_path).convert("RGB")
                img = transform(img_pil).unsqueeze(0).to(device)
                out = (
                    model(img, final=True)
                    .data.squeeze(1)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)[0]
                )
                outs.append(float(out))
                ys.append(int(row["diagnosis"]))
            except Exception:
                continue
    return np.array(outs, dtype=np.float32), np.array(ys, dtype=np.int64)


def _apply_thresholds_np(out_cont: np.ndarray, thr: list):
    out_cont = out_cont.reshape(-1)
    pred = np.zeros_like(out_cont, dtype=np.int64)
    for t in thr:
        pred += (out_cont >= t).astype(np.int64)
    return pred


def _fit_affine_calibration_by_qwk(
    out_fit: np.ndarray,
    y_fit: np.ndarray,
    base_thr: list,
):
    if len(out_fit) == 0:
        return 1.0, 0.0, float("nan")

    best_a, best_b = 1.0, 0.0
    best_k = -1e9

    def _score(a, b):
        o = out_fit * a + b
        pred = _apply_thresholds_np(o, base_thr)
        return float(cohen_kappa_score(y_fit, pred, weights="quadratic"))

    a, b = 1.0, 0.0
    for a_step, b_step, a_span, b_span in [
        (0.10, 0.10, 0.80, 0.80),
        (0.05, 0.05, 0.40, 0.40),
        (0.02, 0.02, 0.20, 0.20),
    ]:
        a_grid = np.arange(
            max(0.25, a - a_span), a + a_span + 1e-9, a_step, dtype=np.float32
        )
        for cand_a in a_grid:
            k = _score(float(cand_a), b)
            if k > best_k + 1e-12:
                best_k = k
                best_a, best_b = float(cand_a), float(b)
        a = best_a

        b_grid = np.arange(b - b_span, b + b_span + 1e-9, b_step, dtype=np.float32)
        for cand_b in b_grid:
            k = _score(a, float(cand_b))
            if k > best_k + 1e-12:
                best_k = k
                best_a, best_b = float(a), float(cand_b)
        b = best_b

    return best_a, best_b, best_k


def _fit_thresholds_qwk_coordinate_ascent(
    out_fit: np.ndarray,
    y_fit: np.ndarray,
    out_val: np.ndarray,
    y_val: np.ndarray,
    base_thr: list,
):
    thr = [float(x) for x in base_thr]

    def _enforce_monotonic(t):
        eps = 1e-3
        t = [float(x) for x in t]
        for i in range(1, 4):
            if t[i] <= t[i - 1] + eps:
                t[i] = t[i - 1] + eps
        return t

    def _score(t):
        if len(out_val) == 0:
            pred = _apply_thresholds_np(out_fit, t)
            return float(cohen_kappa_score(y_fit, pred, weights="quadratic"))
        pred = _apply_thresholds_np(out_val, t)
        return float(cohen_kappa_score(y_val, pred, weights="quadratic"))

    thr = _enforce_monotonic(thr)
    best_k = _score(thr)

    step_schedule = [0.12, 0.06, 0.03]
    span0 = 0.45

    for step in step_schedule:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                cur = thr[i]
                lo = cur - span0
                hi = cur + span0
                grid = np.arange(lo, hi + 1e-9, step, dtype=np.float32)
                for cand in grid:
                    t2 = thr.copy()
                    t2[i] = float(cand)
                    t2 = _enforce_monotonic(t2)
                    k2 = _score(t2)
                    if k2 > best_k + 1e-12:
                        best_k = k2
                        thr = t2
                        improved = True
                        cur = thr[i]
        span0 = max(span0 * 0.7, 0.18)

    return thr, best_k


calib_a, calib_b = 1.0, 0.0

if not loaded_weights:
    df_all = train_df.copy()
    df_all = df_all.sample(frac=1.0, random_state=42).reset_index(drop=True)

    n_all = len(df_all)
    n_fit = min(2200, n_all)
    n_val = min(650, max(300, int(0.30 * n_fit)))
    df_sub = df_all.iloc[:n_fit].reset_index(drop=True)
    df_fit = df_sub.iloc[:-n_val].reset_index(drop=True)
    df_val = df_sub.iloc[-n_val:].reset_index(drop=True)

    counts = (
        train_df["diagnosis"]
        .value_counts()
        .reindex([0, 1, 2, 3, 4], fill_value=0)
        .values.astype(float)
    )
    probs = counts / max(counts.sum(), 1.0)
    cum = np.cumsum(probs)
    q = [float(cum[0]), float(cum[1]), float(cum[2]), float(cum[3])]
    base_threshold = [4.5 * qi for qi in q]
    eps = 1e-3
    for i in range(1, 4):
        if base_threshold[i] <= base_threshold[i - 1] + eps:
            base_threshold[i] = base_threshold[i - 1] + eps

    out_fit, y_fit = _predict_outputs_for_df(net, df_fit, train_img_dir)
    out_val, y_val = _predict_outputs_for_df(net, df_val, train_img_dir)

    calib_a, calib_b, calib_fit_k = _fit_affine_calibration_by_qwk(
        out_fit, y_fit, base_threshold
    )
    out_fit_cal = out_fit * calib_a + calib_b
    out_val_cal = out_val * calib_a + calib_b

    thr_candidate, best_val_k = _fit_thresholds_qwk_coordinate_ascent(
        out_fit_cal, y_fit, out_val_cal, y_val, base_threshold
    )

    threshold = thr_candidate
    print(
        f"[calibration] a={calib_a:.4f} b={calib_b:.4f} | fit_qwk(base_thr)={calib_fit_k:.5f}"
    )
    print(
        f"[threshold-fit] fit_n={len(out_fit_cal)} val_n={len(out_val_cal)} "
        f"| best_val_qwk={best_val_k:.5f}"
    )

print(
    f"Data root: {data_root}\n"
    f"Device: {device} | Weights loaded: {loaded_weights} | weights_path: {weights_path}\n"
    f"Pretrained-backbone fallback active: {use_pretrained_backbone_fallback}\n"
    f"Thresholds: {threshold}\n"
    f"Fallback prior class (if needed): {prior_class}\n"
    f"Calibration (no-weights only): a={calib_a:.4f}, b={calib_b:.4f}"
)




## === cell 5
def heuristic_predict_from_pil(pil_img: Image.Image) -> int:
    img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    p10, p50, p90 = np.percentile(gray, [10, 50, 90])
    spread = float(p90 - p10)  # contrast proxy
    med = float(p50)  # brightness proxy

    score = 0
    if spread > 35:
        score += 1
    if spread > 55:
        score += 1
    if med < 70:
        score += 1
    if med < 55:
        score += 1

    return int(np.clip(score, 0, 4))


submission = []
with torch.inference_mode():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print(i)
        image_name = os.path.join(test_img_dir, f"{idx}.png")
        img_pil = None
        try:
            img_pil = Image.open(image_name).convert("RGB")
            img = transform(img_pil).unsqueeze(0).to(device)

            if loaded_weights:
                _, r_out, _ = net(img)
                pred = regress2class(r_out.data.squeeze(1))
                pred_int = int(pred.item())
            else:
                final_out = net(img, final=True).data.squeeze(1)
                final_out = final_out * float(calib_a) + float(calib_b)
                pred = regress2class(final_out)
                pred_int = int(pred.item())

        except Exception:
            try:
                if img_pil is None:
                    img_pil = Image.open(image_name).convert("RGB")
                pred_int = heuristic_predict_from_pil(img_pil)
            except Exception:
                pred_int = prior_class

        submission.append([idx, pred_int])

submission = np.array(submission, dtype=object)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df = test_df.assign(id_code=test_df["id_code"].astype(str)).merge(
    df, on="id_code", how="left", validate="one_to_one"
)
assert df["diagnosis"].isna().sum() == 0, "Some test ids did not get predictions."
df = df[["id_code", "diagnosis"]]

assert df.shape[0] == len(test_ids)
assert list(df.columns) == ["id_code", "diagnosis"]

df.to_csv("submission.csv", index=False)
print(df.head())
print(df["diagnosis"].value_counts().sort_index())
print("Wrote submission.csv with", len(df), "rows")
