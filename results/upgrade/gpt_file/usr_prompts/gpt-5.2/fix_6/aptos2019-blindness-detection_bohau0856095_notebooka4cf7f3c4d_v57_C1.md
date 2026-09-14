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

0.9058477549711728

# 6. Current score

-0.05005

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix three execution blockers while keeping the modeling/prediction logic the same: (1) remove the broken `pip install` of an unavailable timm wheel and rely on the already-installed `timm`, (2) make device selection robust so it runs on CPU when no GPU is available, and (3) load the pretrained weights only if present; otherwise it still run end-to-end with randomly initialized weights (score be poor, but it produce a valid submission). I also fix a couple of transform bugs that can silently break preprocessing (`is` vs `==` and `trim()` returning `None`) and speed/stability issues by using `torch.no_grad()` and avoiding per-image print spam. Finally, I ensure the submission is always non-empty and matches the required column names and `.csv` suffix.'
- What this solution (achieved 0.0) has done: 'I fix the EfficientNet feature-dimension mismatch that causes the linear layer shape error by correctly taking the backbone’s feature dimension (using `classifier.in_features` when `num_features` is not reliable) and resetting the backbone classifier head so `forward()` returns feature vectors compatible with your custom heads. I keep the model architecture and inference logic the same, only adjusting how features are extracted so the existing heads receive the expected tensor shape. I also make the test image directory resolution a bit more robust to avoid path issues, without changing I/O locations. This should make the notebook run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved -0.15779) has done: 'Your score is 0.0 mainly because the code often runs with randomly initialized weights (no pretrained checkpoint found), which makes predictions essentially random for QWK. The smallest, score-relevant change is to load a standard ImageNet-pretrained EfficientNet backbone (same architecture) when the competition-specific weights are missing; this improves feature quality without changing your heads, loss, or inference logic. I also make weight loading tolerant to common checkpoint key prefixes so that if the file is present, it actually loads into the right submodules. Everything else (transforms, thresholds, regress2class mapping, and submission format) stays the same, and it still produce `submission.csv`.'
- What this solution (achieved -0.05005) has done: 'Your current score is far below the target, so we should make the smallest score-relevant change that improves QWK without changing the model/training core (and you currently do no training). The biggest issue is that you’re using only the regressor head (`r_out`) with fixed thresholds, while your model also has a 5-class classifier head (`c_out`) that can provide a more stable ordinal signal when weights are missing/mismatched. I keep the exact same model definition and weight-loading logic, but change inference to fuse `classifier` and `regressor` outputs into a single continuous prediction (expected value from class probs blended with regressor), then apply the same thresholding to get 0–4. I also add a tiny, deterministic calibration step that aligns the regressor scale to the classifier expected value on-the-fly (per-image blend only, no data leakage), which typically reduces extreme randomness and should move QWK upward toward your target.'

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor) -> torch.Tensor:
    prediction = torch.zeros(out.size(0), device="cpu")
    out_cpu = out.detach().view(-1).cpu()
    for i in range(4):
        prediction += (out_cpu >= threshold[i]).to(torch.float32)
    return prediction


def ordinal2class_prob(out: torch.Tensor) -> torch.Tensor:
    dev = out.device
    pred_prob = torch.zeros(out.size(0), 5, device=dev)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out: torch.Tensor) -> torch.Tensor:
    dev = out.device
    pred_prob = torch.zeros((out.size(0), 5), device=dev)
    out_flat = out.view(-1)
    for i in range(out_flat.size(0)):
        oi = float(out_flat[i].detach().cpu())
        if oi < 4.0:
            l1 = int(math.floor(oi))
            l2 = int(math.ceil(oi))
            pred_prob[i, l1] = 1 - (oi - l1)
            pred_prob[i, l2] = 1 - (l2 - oi)
        else:
            pred_prob[i, 4] = 1.0
    return pred_prob




## === cell 2
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.models.tf_efficientnet_b5_ns(pretrained=True)
        self.backbone.reset_classifier(0, global_pool="")
        self.backbone.global_pool = GeM(flatten=True)

        in_features = getattr(self.backbone, "num_features", None)
        if in_features is None or in_features == 1000:
            in_features = getattr(
                getattr(self.backbone, "classifier", None), "in_features", in_features
            )
        if in_features is None:
            raise RuntimeError("Could not infer in_features for Regressor backbone.")

        self.regressor = nn.Linear(in_features, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None, pretrained_backbone: bool = False):
        super().__init__()
        self.backbone = timm.models.tf_efficientnet_b4_ns(
            pretrained=pretrained_backbone
        )
        self.backbone.reset_classifier(0, global_pool="")
        self.backbone.global_pool = GeM(flatten=True)

        in_features = getattr(self.backbone, "num_features", None)
        clf = getattr(self.backbone, "classifier", None)
        if hasattr(clf, "in_features"):
            in_features = clf.in_features
        if in_features is None:
            raise RuntimeError(
                "Could not infer in_features for ThreeStage_Model backbone."
            )

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )

        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )

        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_features, 500),
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
        return image.crop(bbox) if bbox else image




## === cell 4
BASE1 = "../input/aptos2019-blindness-detection"
BASE2 = "/kaggle/data/aptos2019-blindness-detection"
BASE3 = "/kaggle/input/aptos2019-blindness-detection"
base = BASE1 if os.path.exists(BASE1) else (BASE2 if os.path.exists(BASE2) else BASE3)

test_csv_path = os.path.join(base, "test.csv")
test_img_dir = os.path.join(base, "test_images")
if not os.path.isdir(test_img_dir) and os.path.isdir(
    os.path.join(test_img_dir, "test_images")
):
    test_img_dir = os.path.join(test_img_dir, "test_images")

WEIGHTS_FILENAME = "B4_3stage_56epoch_CLAHE.pkl"
CANDIDATE_WEIGHT_PATHS = [
    "../input/weights/B4_3stage_56epoch_CLAHE.pkl",  # original
    f"../input/{WEIGHTS_FILENAME}",
    f"/kaggle/input/{WEIGHTS_FILENAME}",
]
KAGGLE_INPUT_ROOT = "/kaggle/input"
if os.path.isdir(KAGGLE_INPUT_ROOT):
    for d in os.listdir(KAGGLE_INPUT_ROOT):
        CANDIDATE_WEIGHT_PATHS.append(
            os.path.join(KAGGLE_INPUT_ROOT, d, WEIGHTS_FILENAME)
        )

weights_path = None
for p in CANDIDATE_WEIGHT_PATHS:
    if os.path.exists(p):
        weights_path = p
        break

test_df = pd.read_csv(test_csv_path)
test_ids = test_df["id_code"].values

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

use_pretrained_backbone = weights_path is None
net = ThreeStage_Model(pretrained_backbone=use_pretrained_backbone)


def _normalize_state_dict_keys_for_net(state_dict: dict) -> dict:
    keys = list(state_dict.keys())
    if any(k.startswith("module.") for k in keys):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        keys = list(state_dict.keys())

    if any(k.startswith("model.") for k in keys):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
        keys = list(state_dict.keys())

    if any(k.startswith("net.") for k in keys):
        state_dict = {k.replace("net.", "", 1): v for k, v in state_dict.items()}
        keys = list(state_dict.keys())

    return state_dict


if weights_path is not None and os.path.exists(weights_path):
    ckpt = torch.load(weights_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            state = ckpt["state_dict"]
        elif "model_state_dict" in ckpt and isinstance(ckpt["model_state_dict"], dict):
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
    else:
        state = ckpt

    if isinstance(state, dict):
        state = _normalize_state_dict_keys_for_net(state)

    missing, unexpected = net.load_state_dict(state, strict=False)
    print("Loaded weights:", weights_path)
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
else:
    print("WARNING: weights file not found in candidates:")
    for p in CANDIDATE_WEIGHT_PATHS:
        print(" -", p)
    print(
        "Using ImageNet-pretrained EfficientNet backbone (heads randomly initialized). "
        "This should score better than fully-random weights while still producing a valid submission."
    )

net = net.to(device)
net.eval()



## === cell 5
submission = []

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        image_name = os.path.join(test_img_dir, f"{idx}.png")
        img = Image.open(image_name).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        c_out, r_out, _ = net(img)

        c_prob = F.softmax(c_out, dim=1)
        classes = torch.arange(5, device=c_prob.device, dtype=c_prob.dtype).view(1, -1)
        c_ev = (c_prob * classes).sum(dim=1)  # (B,)

        r_val = r_out.view(-1).clamp(0.0, 4.0)

        eps = 1e-12
        entropy = -(c_prob * (c_prob + eps).log()).sum(dim=1)  # (B,)
        entropy_norm = (entropy / math.log(5.0)).clamp(0.0, 1.0)
        alpha = (0.70 - 0.20 * entropy_norm).clamp(
            0.50, 0.70
        )  # weight on classifier EV

        fused = alpha * c_ev + (1.0 - alpha) * r_val  # (B,)

        pred = regress2class(fused)
        submission.append([idx, int(pred.item())])

submission = np.array(submission, dtype=object)
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

if df.empty:
    raise RuntimeError("Submission DataFrame is empty; inference produced no rows.")
df["diagnosis"] = df["diagnosis"].astype(int).clip(0, 4)

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
