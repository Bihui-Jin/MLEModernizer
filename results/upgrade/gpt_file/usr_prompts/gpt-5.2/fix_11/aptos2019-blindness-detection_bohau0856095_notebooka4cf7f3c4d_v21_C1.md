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

0.912829436921204

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24331) has done: 'I (1) remove the failing `pip install` and make `timm` usage compatible with the already-installed version, (2) make device selection robust by falling back to CPU when CUDA isn’t available (fixing the “no NVIDIA driver” crash), and (3) fix weight-loading so it doesn’t hard-fail when the referenced checkpoint file isn’t present (and instead loads any available `.pth/.pt` in `../input/weights` or runs with the initialized model). Finally, I ensure inference always runs end-to-end and writes a non-empty `submission.csv` with the exact required columns and id alignment. These changes are execution/stability fixes; they don’t alter the core model definition or prediction logic beyond safely handling missing files/hardware.'
- What this solution (achieved 0.0) has done: 'The runtime error comes from a feature-dimension mismatch: with `timm` EfficientNet models, `self.backbone(x)` returns the classifier logits (1000-d) rather than pooled features, so your `Linear(n_features,1)` expects 1792 but receives 1000. The minimal fix is to replace the backbone classifier head with an identity layer so the forward pass returns pooled features, preserving the rest of the architecture and checkpoint-loading logic. I also make `regress2class` device-safe and slightly more robust (no `.data`), and I keep all paths and submission formatting unchanged so a valid `submission.csv` is always written.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with a “submission is valid CSV but predictions are effectively uninformative,” which can happen when the checkpoint isn’t loaded (wrong path) and the randomly-initialized regressor outputs almost-constant values that mostly map to class 2 via your thresholds. To move the score toward the target with minimal, core-logic-preserving changes, I (1) make checkpoint discovery robust by searching common Kaggle input locations (while keeping the original preferred path first), and (2) fix the B4 backbone head removal more completely using `reset_classifier(0)` when available so the feature dimension matches `num_features` deterministically. Everything else (same model, GeM pooling, sigmoid*4.5 regression, fixed thresholds, and per-image inference loop) stays the same, and the script still writes `submission.csv` in the required format.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model running essentially untrained (or loading the wrong weights), producing near-constant predictions that map poorly to the 5 ordinal classes. To move the score upward toward the target with minimal disruption, I (1) make checkpoint discovery prefer the exact expected filename anywhere under `../input/**` (not just `../input/weights`), and (2) align the `Regressor` definition with the actual inference `Model` (B4 + head removed + GeM + correct `num_features`) so that if the checkpoint contains that architecture it loads cleanly instead of partially/incorrectly. Everything else (thresholds, sigmoid*4.5 regression, per-image inference loop, and submission formatting) stays the same to preserve evaluation semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission that is technically valid but badly miscalibrated for quadratic weighted kappa, which commonly happens when fixed regression thresholds don’t match the model’s output scale/distribution on this environment. I keep your exact model/inference logic (EfficientNet-B4 -> GeM -> Linear -> sigmoid*4.5) and only add a minimal, metric-aligned post-processing step: fit 4 optimal thresholds on a small validation split of the provided training set using your current model outputs, then apply those thresholds to test predictions. This preserves the core architecture and prediction semantics (still regression + thresholding), but adapts the thresholds to the actual loaded checkpoint (or untrained model) so the kappa moves upward toward your target. I also make image path resolution robust to both `../input/...` and `/kaggle/input/...` without changing the dataset used, and ensure `submission.csv` is always written.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the “CSV is valid but the model is effectively untrained / mismatched,” so the smallest reliable way to move upward is to ensure the checkpoint actually loads into the exact same module names and to make the threshold-fitting step run (right now `scipy` isn’t installed, so the script would fail before writing a meaningful submission). I (1) remove the SciPy dependency by fitting thresholds via a small deterministic grid/local search using only NumPy (keeping the same “regression + thresholds” semantics), and (2) make checkpoint loading stricter/cleaner by unwrapping common wrappers (e.g., `model`, `state_dict`) and stripping both `module.` and `net.` prefixes so weights don’t silently miss. Everything else—EfficientNet-B4 backbone, GeM pooling, sigmoid*4.5 regression head, and submission formatting—stays the same, but you should now get non-random predictions and a score that increases toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the submission is valid but predictions are effectively uninformative, most commonly because the checkpoint didn’t load (so the model is random) or because keys didn’t match and most weights stayed uninitialized. I keep your exact model/inference semantics (EfficientNet-B4 + GeM + Linear + sigmoid*4.5 + thresholding) but make checkpoint loading robust to common nesting/prefix patterns and enforce that we only accept a checkpoint if it actually loads most backbone weights. I also make the train/val alignment correct in threshold fitting (use val_df ids_used to index y) and ensure deterministic behavior (seeds + cudnn flags) so the fitted thresholds don’t vary run-to-run. These are minimal, score-relevant fixes that should move QWK upward toward your target without changing the core approach.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with either (a) predictions not matching the submission schema/order (misalignment) or (b) the model effectively running untrained due to a checkpoint-loading mismatch, producing near-constant classes. I keep your exact model/inference semantics (EfficientNet-B4 + GeM + Linear + sigmoid*4.5 + thresholding) and make two minimal, score-relevant fixes: (1) correct state-dict key cleaning so we don’t accidentally strip required prefixes like `backbone.` (which can prevent backbone weights from loading), and (2) use the fitted thresholds consistently for test by applying them via the same NumPy digitize logic (removing any subtle CPU/torch threshold handling differences). These changes are intended to make weight loading actually work when the checkpoint exists and ensure identical thresholding behavior between validation and test, which should move QWK upward toward your target while keeping the approach unchanged. The script still runs end-to-end and writes a valid `submission.csv` with correct columns and id alignment.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission that is valid CSV but effectively garbage (e.g., NaNs or misaligned ids), or a model that is running untrained because no checkpoint was actually found/loaded in this environment. To move the score upward with minimal core-logic changes, I (1) make the data-root resolution prefer the known existing `/kaggle/data` and its nested `aptos2019-blindness-detection` directory first (so images/CSVs always resolve correctly), (2) make checkpoint search include the *local working directory* as well (in case the weight file was added alongside the notebook/script), and (3) harden inference against missing/corrupt images by falling back to the untrimmed image (preventing empty outputs/NaNs). These changes keep your exact model (EfficientNet-B4 + GeM + Linear + sigmoid*4.5) and the same “fit thresholds on a validation split then apply to test” semantics, but they greatly reduce the chance of producing an uninformative 0.0 submission.'

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
import torchvision.transforms as transforms
from PIL import Image, ImageChops, UnidentifiedImageError

import timm


def _seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_seed_everything(42)

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -10)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)
    return im


def _resolve_data_root():
    candidates = [
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection",
        "/kaggle/data",
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection",
        "/kaggle/input",
        "../input/aptos2019-blindness-detection",
        "../input",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "test.csv")
        ):
            if os.path.isdir(os.path.join(c, "train_images")) and os.path.isdir(
                os.path.join(c, "test_images")
            ):
                return c
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "test.csv")
        ):
            return c
    return "/kaggle/data/aptos2019-blindness-detection"


DATA_ROOT = _resolve_data_root()
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")


def _resolve_img_dir(root, name):
    cands = [
        os.path.join(root, name),
        os.path.join("/kaggle/data/aptos2019-blindness-detection", name),
        os.path.join(
            "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection",
            name,
        ),
        os.path.join("/kaggle/input/aptos2019-blindness-detection", name),
        os.path.join(
            "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection",
            name,
        ),
        os.path.join("../input/aptos2019-blindness-detection", name),
        os.path.join("../input", name),
    ]
    for p in cands:
        if os.path.isdir(p):
            return p
    return os.path.join(root, name)


TRAIN_IMG_DIR = _resolve_img_dir(DATA_ROOT, "train_images")
TEST_IMG_DIR = _resolve_img_dir(DATA_ROOT, "test_images")

print("DATA_ROOT:", DATA_ROOT)
print("TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
print("TEST_IMG_DIR:", TEST_IMG_DIR)
print("Device:", device)



## === cell 1
threshold = [0.7, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor, thr=None) -> torch.Tensor:
    """
    Convert regression output (float) to ordinal class {0..4} using thresholds.
    Keeps tensor on CPU for safe int conversion later.
    """
    if thr is None:
        thr = threshold
    out_cpu = out.detach().view(-1).cpu()
    prediction = torch.zeros(out_cpu.size(0), dtype=torch.long)
    for i in range(4):
        prediction += (out_cpu >= float(thr[i])).long()
    return prediction




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
        self.backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=False)

        if hasattr(self.backbone, "reset_classifier"):
            self.backbone.reset_classifier(0)
        else:
            if hasattr(self.backbone, "classifier"):
                self.backbone.classifier = nn.Identity()
            if hasattr(self.backbone, "fc"):
                self.backbone.fc = nn.Identity()
            if hasattr(self.backbone, "head"):
                self.backbone.head = nn.Identity()

        self.backbone.global_pool = GeM(flatten=True)

        n_features = getattr(self.backbone, "num_features", 1000)
        self.regressor = nn.Linear(n_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out




## === cell 3
test_df = pd.read_csv(TEST_CSV)
test_ids = np.squeeze(test_df["id_code"].values)

input_size = 384

tranforms = transforms.Compose(
    [
        transforms.Resize(int(input_size * 1.15)),
        transforms.CenterCrop((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=False)

        if hasattr(self.backbone, "reset_classifier"):
            self.backbone.reset_classifier(0)
        else:
            if hasattr(self.backbone, "classifier"):
                self.backbone.classifier = nn.Identity()
            if hasattr(self.backbone, "fc"):
                self.backbone.fc = nn.Identity()
            if hasattr(self.backbone, "head"):
                self.backbone.head = nn.Identity()

        self.backbone.global_pool = GeM(flatten=True)

        n_features = getattr(self.backbone, "num_features", 1000)
        self.regressor = nn.Linear(n_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
        ]:
            if k in obj:
                obj = obj[k]
                break
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        if len(obj) == 1:
            v = next(iter(obj.values()))
            if isinstance(v, dict) and all(isinstance(kk, str) for kk in v.keys()):
                return v
        return obj
    return None


def _clean_state_keys(state):
    new_state = {}
    for k, v in state.items():
        nk = k
        for pref in ["module.", "net.", "model."]:
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        new_state[nk] = v
    return new_state


def _load_checkpoint_into_model(model, ckpt_path, device):
    obj = torch.load(ckpt_path, map_location=device)
    state = _unwrap_state_dict(obj)
    if state is None:
        return model, 0.0

    state = _clean_state_keys(state)

    model_state_keys = set(model.state_dict().keys())
    backbone_state_keys = set(model.backbone.state_dict().keys())

    state_keys = set(state.keys())
    overlap_model = len(model_state_keys.intersection(state_keys))
    overlap_backbone = len(backbone_state_keys.intersection(state_keys))

    if overlap_model >= overlap_backbone:
        missing, unexpected = model.load_state_dict(state, strict=False)
        load_ratio = overlap_model / max(1, len(model_state_keys))
        print("Loaded ckpt into full model:", ckpt_path)
        print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
    else:
        missing_b, unexpected_b = model.backbone.load_state_dict(state, strict=False)
        load_ratio = overlap_backbone / max(1, len(backbone_state_keys))
        print("Loaded ckpt into backbone:", ckpt_path)
        print(
            "Backbone missing keys:",
            len(missing_b),
            "Unexpected keys:",
            len(unexpected_b),
        )

    print("Approx load ratio:", float(load_ratio))
    return model, float(load_ratio)


net = Model().to(device)

preferred_ckpt = "../input/weights/0.912_tf_efficientnet_b4_ns_regress.pth"

search_globs = [
    preferred_ckpt,
    "./0.912_tf_efficientnet_b4_ns_regress.pth",
    "/kaggle/working/0.912_tf_efficientnet_b4_ns_regress.pth",
    "../input/**/0.912_tf_efficientnet_b4_ns_regress.pth",
    "/kaggle/input/**/0.912_tf_efficientnet_b4_ns_regress.pth",
    "/kaggle/data/**/0.912_tf_efficientnet_b4_ns_regress.pth",
    "./*.pth",
    "./*.pt",
    "/kaggle/working/*.pth",
    "/kaggle/working/*.pt",
    "../input/weights/*.pth",
    "../input/weights/*.pt",
    "../input/**/*.pth",
    "../input/**/*.pt",
    "/kaggle/input/**/*.pth",
    "/kaggle/input/**/*.pt",
    "/kaggle/data/**/*.pth",
    "/kaggle/data/**/*.pt",
]
ckpt_candidates = []
for g in search_globs:
    ckpt_candidates.extend(glob.glob(g, recursive=True))


def _ckpt_rank(p: str) -> tuple:
    base = os.path.basename(p).lower()
    exact = 0 if base == "0.912_tf_efficientnet_b4_ns_regress.pth" else 1
    return (
        exact,
        0 if "0.912" in base else 1,
        0 if "efficientnet_b4" in base else 1,
        0 if "regress" in base else 1,
        len(p),
        p,
    )


ckpt_candidates = sorted(list(dict.fromkeys(ckpt_candidates)), key=_ckpt_rank)

chosen_ckpt = None
chosen_ratio = 0.0
for p in ckpt_candidates[:80]:
    if not os.path.exists(p):
        continue
    try:
        tmp = Model().to(device)
        tmp, ratio = _load_checkpoint_into_model(tmp, p, device)
    except Exception as e:
        print("Skipping ckpt (load failed):", p, "error:", repr(e))
        continue
    if ratio > chosen_ratio:
        chosen_ckpt = p
        chosen_ratio = ratio
        net = tmp
    if chosen_ratio >= 0.60:
        break

if chosen_ckpt is None:
    print("No usable checkpoint found (running untrained).")
else:
    print("Chosen checkpoint:", chosen_ckpt)
    print("Chosen load ratio:", chosen_ratio)

net.eval()
print("Backbone num_features:", getattr(net.backbone, "num_features", "NA"))



## === cell 4
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score


def _qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def _apply_thresholds(preds_1d: np.ndarray, thr: np.ndarray) -> np.ndarray:
    thr = np.asarray(thr, dtype=np.float64)
    thr = np.sort(thr)
    return np.digitize(preds_1d, thr, right=False).astype(np.int64)


def _fit_thresholds_numpy(y_true: np.ndarray, preds: np.ndarray, init_thr=None):
    """
    Deterministic NumPy local search for 4 thresholds (regression -> ordinal).
    """
    y_true = y_true.astype(np.int64)
    preds = preds.astype(np.float64)

    if init_thr is None:
        thr = np.array([0.7, 1.5, 2.5, 3.5], dtype=np.float64)
    else:
        thr = np.asarray(init_thr, dtype=np.float64).copy()

    def score(thr_vec):
        y_pred = _apply_thresholds(preds, thr_vec)
        return _qwk(y_true, y_pred)

    best_thr = np.sort(thr)
    best_score = score(best_thr)

    lo, hi = 0.0, 4.5
    steps = [0.25, 0.10, 0.05, 0.02]
    for st in steps:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for delta in (-st, st):
                    cand = best_thr.copy()
                    cand[i] = np.clip(cand[i] + delta, lo, hi)
                    cand = np.sort(cand)
                    if not (cand[0] < cand[1] < cand[2] < cand[3]):
                        continue
                    sc = score(cand)
                    if sc > best_score + 1e-6:
                        best_score = sc
                        best_thr = cand
                        improved = True
    return best_thr, best_score


def _safe_open_image(path: str) -> Image.Image:
    try:
        return Image.open(path).convert("RGB")
    except (FileNotFoundError, UnidentifiedImageError, OSError) as e:
        print("Warning: failed to read image:", path, "error:", repr(e))
        return Image.new("RGB", (input_size, input_size), (0, 0, 0))


def _predict_regression_for_ids(ids, img_dir, batch_size=8, max_n=None):
    outs = []
    ids_used = []
    batch = []
    batch_ids = []
    with torch.no_grad():
        for k, idx in enumerate(ids):
            if max_n is not None and k >= max_n:
                break
            image_name = os.path.join(img_dir, f"{idx}.png")
            img = _safe_open_image(image_name)

            try:
                img2 = trim(img)
                if img2.size[0] < 10 or img2.size[1] < 10:
                    img2 = img
            except Exception:
                img2 = img

            tens = tranforms(img2)
            batch.append(tens)
            batch_ids.append(idx)
            if len(batch) == batch_size:
                x = torch.stack(batch, dim=0).to(device)
                r_out = net(x).view(-1).detach().cpu().numpy()
                outs.append(r_out)
                ids_used.extend(batch_ids)
                batch, batch_ids = [], []
        if len(batch) > 0:
            x = torch.stack(batch, dim=0).to(device)
            r_out = net(x).view(-1).detach().cpu().numpy()
            outs.append(r_out)
            ids_used.extend(batch_ids)

    if len(outs) == 0:
        return np.zeros((0,), dtype=np.float64), np.array([], dtype=object)
    return np.concatenate(outs, axis=0), np.array(ids_used)


train_df = pd.read_csv(TRAIN_CSV)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
tr_idx, va_idx = next(
    sss.split(train_df["id_code"].values, train_df["diagnosis"].values)
)
val_df = train_df.iloc[va_idx].reset_index(drop=True)

val_ids = val_df["id_code"].values
val_preds, val_ids_used = _predict_regression_for_ids(
    val_ids, TRAIN_IMG_DIR, batch_size=8
)

val_map = dict(zip(val_df["id_code"].astype(str).values, val_df["diagnosis"].values))
val_y = np.array([val_map[str(i)] for i in val_ids_used], dtype=np.int64)

thr_fitted, val_best = _fit_thresholds_numpy(
    val_y, val_preds, init_thr=np.array(threshold, dtype=np.float64)
)
val_pred_cls = _apply_thresholds(val_preds, thr_fitted)
val_qwk = _qwk(val_y, val_pred_cls)

print("Default thresholds:", threshold)
print("Fitted thresholds:", thr_fitted.tolist())
print("Validation QWK (fitted thresholds):", float(val_qwk))



## === cell 5
submission = []
test_reg_outs = []
test_ids_used = []

batch = []
batch_ids = []
batch_size = 8

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
        img = _safe_open_image(image_name)

        try:
            img2 = trim(img)
            if img2.size[0] < 10 or img2.size[1] < 10:
                img2 = img
        except Exception:
            img2 = img

        tens = tranforms(img2)
        batch.append(tens)
        batch_ids.append(idx)

        if len(batch) == batch_size:
            x = torch.stack(batch, dim=0).to(device)
            r_out = net(x).view(-1).detach().cpu().numpy()
            test_reg_outs.append(r_out)
            test_ids_used.extend(batch_ids)
            batch, batch_ids = [], []

    if len(batch) > 0:
        x = torch.stack(batch, dim=0).to(device)
        r_out = net(x).view(-1).detach().cpu().numpy()
        test_reg_outs.append(r_out)
        test_ids_used.extend(batch_ids)

if len(test_reg_outs) == 0:
    test_reg_outs = np.zeros((len(test_ids),), dtype=np.float64)
    test_ids_used = list(test_ids)
else:
    test_reg_outs = np.concatenate(test_reg_outs, axis=0)

test_pred_cls = _apply_thresholds(test_reg_outs, thr_fitted)

for idx, pred in zip(test_ids_used, test_pred_cls.tolist()):
    submission.append([idx, int(pred)])

submission = np.array(submission, dtype=object)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df = df.set_index("id_code").reindex(test_df["id_code"].astype(str)).reset_index()

if df["diagnosis"].isna().any():
    df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print(
    "Diagnosis value counts:\n", df["diagnosis"].value_counts(dropna=False).sort_index()
)
