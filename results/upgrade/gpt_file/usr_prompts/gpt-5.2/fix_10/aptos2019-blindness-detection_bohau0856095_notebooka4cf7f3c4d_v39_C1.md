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

0.9099378838043536

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.01221) has done: 'I fix the missing weights issue by adding a safe fallback that loads the provided pretrained EfficientNet backbone when the external `.pkl` file is not available, so inference can still run end-to-end. I also make the device selection robust by automatically using CPU when no CUDA driver/GPU is present, which fixes the runtime CUDA error. To prevent an empty submission, I add basic file existence checks and ensure `trim()` always returns an image (it currently can return `None`). Finally, I keep the model architecture and prediction logic intact and ensure the script always writes `submission.csv` with the required columns.'
- What this solution (achieved 0.01787) has done: 'Your score is extremely low because the current code falls back to a randomly-initialized head when the external `.pkl` weights aren’t present, so predictions are essentially arbitrary even though inference runs. To move the score toward the 0.9099 target with minimal logic change, I keep the exact same model and preprocessing, but fix the fallback to load the full `tf_efficientnet_b4_ns` pretrained backbone **into the existing backbone module** (so GeM pooling and output dims stay consistent). I also ensure inference calls the intended regression head by using `net(img, final=True)` when available (this uses the model’s built-in final regressor instead of ignoring it), while keeping the same thresholding logic for class conversion. Finally, I add a strict test-id order alignment so the submission rows match `test.csv` exactly.'
- What this solution (achieved 0.01787) has done: 'Your current score is far below the target because when the competition-trained `.pkl` weights are missing, the fallback only loads an ImageNet backbone and leaves all heads (classifier/regressor/ordinal/final_regressor) randomly initialized, which makes predictions essentially arbitrary. To move score toward the 0.9099 target with minimal semantic changes, I keep your exact model and preprocessing, but (1) correctly try multiple likely weight locations (both `../input/...` and `/kaggle/input/...`) and (2) load whatever matching keys exist from the checkpoint in a non-strict way (so partial checkpoints still improve predictions). If no checkpoint is found, I keep your current ImageNet-backbone fallback (so the script still runs), but this change should significantly increase score whenever the weights are actually present in the environment. I also keep submission ordering aligned to `test.csv` and still write `submission.csv` unchanged.'
- What this solution (achieved 0.01787) has done: 'Your score is far below the target because the inference is effectively running with mostly random heads when the competition checkpoint isn’t found (and your current `../input/...` paths don’t match the provided dataset location), so predictions are near-random. I make a minimal change to search the *actual* dataset/working directories you listed for the `.pkl` weights and load it robustly (supporting common checkpoint key prefixes like `module.` and `model.`) while keeping your model and transforms unchanged. If no checkpoint exists anywhere, we still fall back to the ImageNet backbone exactly as you already do (so it always produces a valid submission). This should move the score sharply upward toward the target whenever the weights file is present in the environment.'
- What this solution (achieved 0.01787) has done: 'Your current score is far below the target because the code is pointing at a non-existent `../input/...` dataset location in your environment, which makes it silently generate default predictions (mostly zeros) and yields a near-random kappa. I minimally fix data-path resolution by auto-detecting the real dataset root from the provided file tree (preferring `/kaggle/data/input/...`), while keeping your exact model, transforms, and prediction logic. I also broaden the weight-file search to recursively look for `B4_3stage_46epoch_CLAHE.pkl` under the detected dataset root (and common locations), so if it exists anywhere in the environment it gets loaded without changing architecture. These changes should move the score sharply upward toward the target whenever the checkpoint exists; if it truly does not exist, the code still runs end-to-end and writes a valid `submission.csv`.'
- What this solution (achieved 0.01787) has done: 'Your score is far below the target because the script is almost certainly not loading the intended competition-trained checkpoint (so most heads remain random and predictions are near-random). I keep your exact model, transforms, and inference logic, but make the weight search reliably find the checkpoint by (a) also searching the working directory and (b) accepting common filename variants (e.g., `.pt/.pth/.bin`) and picking the largest match (often the real model). I also broaden state-dict normalization to handle nested keys like `ema_state_dict` and strip `backbone.` prefixes, so more weights actually load into your existing modules without changing architecture. These are minimal, execution-safe changes that should move QWK sharply upward toward your target if the checkpoint exists in the environment; otherwise behavior remains the same and still produces a valid `submission.csv`.'
- What this solution (achieved 0.01787) has done: 'Your score is far below the target because the model is effectively doing inference with untrained/random heads when the competition checkpoint isn’t available, so predictions are close to arbitrary. With minimal changes and without altering the model architecture or transforms, I (1) make the checkpoint search also look for any plausible `.pth/.pt/.pkl` under the available dataset roots and pick the best candidate, and (2) improve checkpoint loading by handling common nested/prefixed key formats and safely loading either a full state_dict or a raw checkpoint. If no checkpoint is found, I keep your ImageNet-backbone fallback exactly, but I also ensure inference uses deterministic settings and correct test-order alignment (already mostly correct) so the submission is valid and stable.'
- What this solution (achieved -0.07456) has done: 'Your current score is far below the target because the script is almost certainly not loading the intended competition-trained checkpoint (so the final regressor head remains random), and it’s also not applying the CLAHE preprocessing that your checkpoint filename implies it expects. I make two minimal, score-relevant changes: (1) expand checkpoint discovery to also look under `/kaggle/data/**` and `/kaggle/data/input/**` recursively for the exact `.pkl` (and close variants), and (2) add a CLAHE step in the test transform pipeline while keeping your model, thresholds, and inference exactly the same. These changes preserve core logic/semantics (same architecture and regression->threshold class mapping) but should move QWK substantially upward toward your 0.9099 target if the checkpoint exists and was trained with CLAHE. The script still run end-to-end and always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your score is far below the target because the inference is likely running without the intended competition-trained checkpoint (so the final regressor head is effectively random) and/or using a slightly mismatched EfficientNet-B4 input preprocessing. I keep your exact model/thresholding logic, but (1) expand checkpoint discovery to also look for any plausible “B4 3stage CLAHE” checkpoint names under the provided dataset roots, and (2) load the checkpoint more robustly by handling common key prefixes and nested dict formats so more weights actually land in the right modules. Then I align the image normalization to timm’s default for `tf_efficientnet_b4_ns` (while keeping CLAHE, crop, GeM, and everything else the same) because this often yields a big QWK jump when using ImageNet-pretrained or competition checkpoints trained with timm defaults. These are minimal, score-relevant changes and still produce a valid `submission.csv` end-to-end.'

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

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

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


class CLAHE(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image: Image.Image) -> Image.Image:
        arr = np.array(image)
        if arr.ndim == 2:
            gray = arr
            clahe = cv2.createCLAHE(
                clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
            )
            out = clahe.apply(gray)
            return Image.fromarray(out, mode="L").convert("RGB")
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        rgb = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb)




## === cell 4
def _detect_data_dir():
    candidates = [
        "/kaggle/data/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection",
        "../input/aptos2019-blindness-detection",
        "/kaggle/data/input",
        "/kaggle/data",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "test.csv")) and os.path.exists(
            os.path.join(c, "test_images")
        ):
            return c

    for base in ["/kaggle/data/input", "/kaggle/data", "/kaggle/input", "../input"]:
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            if "test.csv" in files and "test_images" in dirs:
                return root
    return "../input/aptos2019-blindness-detection"


DATA_DIR = _detect_data_dir()
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

if not os.path.exists(TEST_CSV):
    raise FileNotFoundError(f"test.csv not found at {TEST_CSV} (DATA_DIR={DATA_DIR})")
if not os.path.isdir(TEST_IMG_DIR):
    raise FileNotFoundError(
        f"test_images dir not found at {TEST_IMG_DIR} (DATA_DIR={DATA_DIR})"
    )

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

input_size = 380

cfg = timm.data.resolve_model_data_config(
    timm.create_model("tf_efficientnet_b4_ns", pretrained=False)
)
mean = cfg.get("mean", (0.485, 0.456, 0.406))
std = cfg.get("std", (0.229, 0.224, 0.225))

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

net = ThreeStage_Model()


def _find_weight_file(
    filename="B4_3stage_46epoch_CLAHE.pkl",
    roots=None,
    exts=(".pkl", ".pth", ".pt", ".bin"),
):
    if roots is None:
        roots = []
    base = os.path.splitext(os.path.basename(filename))[0]

    patterns = set([os.path.basename(filename)])
    for e in exts:
        patterns.add(base + e)

    patterns.update(
        {
            "B4_3stage_46epoch_CLAHE.pth",
            "B4_3stage_46epoch_CLAHE.pt",
            "B4_3stage_46epoch.pkl",
            "B4_3stage_46epoch.pth",
            "B4_3stage.pkl",
            "3stage_b4.pkl",
            "three_stage_b4.pkl",
            "best.pth",
            "best.pt",
            "checkpoint.pth",
            "checkpoint.pt",
            "model.pth",
            "model.pt",
            "weights.pth",
            "weights.pt",
        }
    )

    found = []
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        for root, dirs, files in os.walk(r):
            for f in files:
                if f in patterns:
                    fp = os.path.join(root, f)
                    try:
                        sz = os.path.getsize(fp)
                    except OSError:
                        sz = -1
                    found.append((sz, fp))
    if not found:
        return None
    found.sort(reverse=True, key=lambda x: x[0])
    return found[0][1]


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if all(hasattr(v, "shape") for v in obj.values() if v is not None):
            return obj
        for key in (
            "ema_state_dict",
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "student",
            "teacher",
        ):
            if key in obj and isinstance(obj[key], dict):
                sd = _extract_state_dict(obj[key])
                if sd is not None and len(sd) > 0:
                    return sd
    return None


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return {}

    prefixes = (
        "module.",
        "model.",
        "net.",
        "backbone.",
        "encoder.",
    )
    out = {}
    for k, v in state_dict.items():
        nk = k
        changed = True
        while changed:
            changed = False
            for pref in prefixes:
                if (
                    nk.startswith(pref) and pref != "backbone."
                ):  # keep backbone. when intended
                    nk = nk[len(pref) :]
                    changed = True

        if nk.startswith("backbone.backbone."):
            nk = "backbone." + nk[len("backbone.backbone.") :]

        out[nk] = v
    return out


candidate_roots = [
    "../input/weights",
    "/kaggle/input/weights",
    DATA_DIR,
    os.path.dirname(DATA_DIR),
    "/kaggle/data/input",
    "/kaggle/data",
    "/kaggle/data/working",
    "/kaggle/working",
]

weights_path = _find_weight_file(roots=candidate_roots)

if weights_path is not None:
    ckpt = torch.load(weights_path, map_location="cpu")
    sd = _extract_state_dict(ckpt)
    if sd is None:
        sd = ckpt if isinstance(ckpt, dict) else {}
    sd = _normalize_state_dict_keys(sd)

    model_state = net.state_dict()
    filtered = {
        k: v
        for k, v in sd.items()
        if (
            k in model_state and hasattr(v, "shape") and v.shape == model_state[k].shape
        )
    }
    missing, unexpected = net.load_state_dict(filtered, strict=False)
    print(f"Loaded checkpoint from: {weights_path}")
    print(
        f"Checkpoint keys loaded: {len(filtered)}/{len(model_state)} (missing={len(missing)}, unexpected={len(unexpected)})"
    )
else:
    pretrained_backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
    missing, unexpected = net.backbone.load_state_dict(
        pretrained_backbone.state_dict(), strict=False
    )
    net.backbone.global_pool = GeM(flatten=True)
    print(
        "WARNING: No competition checkpoint found anywhere; using ImageNet backbone fallback (score will be much lower)."
    )

net = net.to(device)
net.eval()



## === cell 5
submission_dict = {}

for idx in test_ids:
    image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
    if not os.path.exists(image_name):
        submission_dict[idx] = 0
        continue

    img = Image.open(image_name).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    r_out = net(img, final=True)  # shape (1,1), already scaled to [0,4.5]
    pred = regress2class(r_out.data.squeeze(1))
    submission_dict[idx] = int(pred.item())

if len(submission_dict) == 0:
    raise RuntimeError(
        "No test predictions were generated; check test.csv and test_images path."
    )



## === cell 6
df = pd.DataFrame({"id_code": test_ids})
df["diagnosis"] = [submission_dict.get(i, 0) for i in test_ids]
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("DATA_DIR used:", DATA_DIR)
print("Weights used:", weights_path if weights_path is not None else "None")
