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

0.9218033364342396

# 6. Current score

0.24593

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the code robust to the Kaggle CPU-only environment by selecting `cuda` only if available and mapping weights to CPU when loading. I also fix a couple of small but critical logic bugs in transforms (`trim()` returning `None` sometimes and the incorrect `is` string comparison) that can break preprocessing. Since your provided weights path doesn’t exist in this dataset, I add a minimal fallback that runs end-to-end by generating a valid submission via a simple baseline derived from the training label distribution (so the notebook always produces a non-empty `submission.csv`). These changes are focused on unblocking execution and producing a valid CSV; if you later provide the correct weights file, the same script automatically use it.'
- What this solution (achieved 0.02131) has done: 'I fix the immediate crash when loading a timm pretrained backbone by allowing the missing `global_pool.p` key (your model replaces `global_pool` with GeM, so that parameter doesn’t exist in the pretrained state dict). Then I prevent the cascade `NameError`s by making sure `train_df` and `train_label_mode` are always defined (even if weight-loading fails), so threshold tuning and the submission writer can run. Finally, I keep the same inference core logic, but make the test prediction loop robust to missing/corrupt images so the notebook always produces a valid `submission.csv`.'
- What this solution (achieved 0.08813) has done: 'Your current score is far below the target, so the smallest change that plausibly moves it upward is to use the model’s intended “final” regression head at inference time (it was trained to combine classifier+regressor+ordinal outputs, but your code currently ignores it). I keep the same architecture and preprocessing, and only change the forward call used for both threshold tuning and test prediction to `net(img, final=True)`, so predictions come from the combined head. I also make `regress2class` device-safe (avoid `.data`/CPU mixing) to prevent subtle dtype/device issues, while keeping identical thresholding semantics. Everything else (paths, transforms, submission formatting) stays the same.'
- What this solution (achieved -0.00858) has done: 'Your score is far below the target, so we should make small, legitimate changes that improve agreement with the quadratic weighted kappa metric without altering the model/training core. The biggest low-risk gain here is to align inference with how the final regressor was trained: apply the same input normalization to the `final_regressor` inputs at inference by passing class probabilities (softmax) and ordinal probabilities (sigmoid) instead of raw logits, while keeping the architecture and weights untouched. We implement a single helper `predict_continuous()` used consistently for threshold tuning and test inference, and keep the same threshold-search logic and submission formatting. This should move performance upward significantly compared to feeding mismatched logits into the final head.'
- What this solution (achieved -0.00858) has done: 'Your score is far below the target, so the smallest change likely to increase QWK is to fix a mismatch between the `final_regressor`’s expected inputs and what it currently receives at inference. Right now `predict_continuous()` feeds raw ordinal logits (not passed through sigmoid) into the final head, even though the forward path applies sigmoid to ordinal outputs; this can severely break calibration and class boundaries. I change `predict_continuous()` to apply `sigmoid` to `o_out` and (for consistency) ensure `r_out` has the same sigmoid-scaling used elsewhere before concatenation, without changing the model, transforms, thresholding logic, or submission format. Everything else stays the same so runtime and semantics remain stable.'
- What this solution (achieved 0.0) has done: 'Your current score is far below the target, so we should make the smallest legitimate changes that plausibly improve QWK without changing the model/training core. The main issue is that in the “no custom weights” fallback, only the backbone is pretrained while all heads (classifier/regressor/ordinal/final_regressor) are random, so the continuous outputs and threshold tuning become effectively noise, yielding near-random kappa. I keep your exact architecture and inference flow, but (1) disable threshold tuning when `WEIGHTS_PATH` is missing and (2) in that same missing-weights case, output a deterministic, distribution-informed baseline label (mode) to avoid random predictions. When weights are present, behavior remains the same (including threshold tuning and `predict_continuous()`), so this change only improves the broken fallback path and should move the score upward toward the target rather than oscillating around 0.'
- What this solution (achieved 0.02917) has done: 'Your 0.0 score is coming from the “missing weights” fallback path, which currently predicts a constant class (the train mode) and therefore yields very poor QWK. To move toward the target with minimal changes and without altering your model/training logic, I keep your architecture and inference intact but replace the constant fallback with a deterministic, distribution-matching baseline that samples labels according to the training label histogram (seeded for reproducibility). This preserves valid submission formatting while producing a more reasonable label distribution that should lift QWK above 0.0 when no weights are available. When weights are present, behavior remains unchanged (same `predict_continuous()`, same threshold tuning, same transforms).'
- What this solution (achieved -0.15091) has done: 'Your score is far below the target, and the main reason is that the current run is using the “missing weights” fallback path, which produces essentially random/distribution-sampled labels and thus very low QWK. The smallest legitimate improvement is to remove randomness in the fallback and instead predict using a simple, deterministic image-derived signal (mean brightness) mapped to classes via thresholds fit on the training set; this keeps the same I/O and submission semantics while producing predictions correlated with image quality/severity. When the weights file is present, nothing changes: the same model, `predict_continuous()`, threshold tuning, and inference are used. This should move the score upward toward your target without altering your core model logic.'
- What this solution (achieved 0.24593) has done: 'Your current score is far below the target, and it’s coming from the “missing weights” fallback path; the brightness-only heuristic is too weak and can easily be anti-correlated with severity, producing very low QWK. Keeping your core model/inference logic unchanged, I replace only the fallback with a deterministic, image-derived feature baseline that is more directly related to DR severity: vessel/lesion “texture” strength via Laplacian variance + darkness ratio. I fit simple per-class centroids and class-to-score calibration on a subset of training images (same paths), then predict the closest class for test images; this stays lightweight and deterministic under the 600s limit. When `WEIGHTS_PATH` exists, nothing changes (same `predict_continuous()`, threshold tuning, and regression-to-class mapping), so this only improves the broken fallback path toward your target.'
- What this solution (achieved 0.24593) has done: 'Your current score (0.24593) is far below the target (0.9218), and it’s almost certainly because the run is still going through the “missing weights fallback” path where the network heads are untrained. The smallest change that can move you meaningfully toward the target—without changing your architecture, transforms, or loss/training semantics—is to ensure `WEIGHTS_PATH` resolves to an actually-existing file in this Kaggle environment by searching common locations under `/kaggle/input` (and only falling back to the heuristic baseline if no weights exist anywhere). I keep your existing inference (`predict_continuous`, threshold tuning, `regress2class`) identical; the only “score” change is enabling the intended pretrained model weights to load when present. I also make the weight loading tolerant to `global_pool.*`/missing GeM keys (which you already encountered earlier) while keeping strictness for the rest, to avoid silently not loading useful weights.'

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
from sklearn.model_selection import train_test_split
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


def regress2class(out):
    out = out.view(-1)
    prediction = torch.zeros(out.size(0), device=out.device, dtype=torch.long)
    for i in range(4):
        prediction += (out >= threshold[i]).to(torch.long)
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
DATA_ROOT_CANDIDATES = [
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "../input",
    "/kaggle/input",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data",
]
DATA_ROOT = None
for cand in DATA_ROOT_CANDIDATES:
    if os.path.exists(os.path.join(cand, "train.csv")) and os.path.exists(
        os.path.join(cand, "test.csv")
    ):
        DATA_ROOT = cand
        break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate dataset root. Checked: " + str(DATA_ROOT_CANDIDATES)
    )

WEIGHTS_PATH = "../input/weights/B4_3stage_17epoch_finetune.pkl"


def _find_existing_weights_path(preferred_path: str):
    if preferred_path and os.path.exists(preferred_path):
        return preferred_path

    candidates = []
    roots = ["/kaggle/input", "../input", "/kaggle/data", "../data"]
    for r in roots:
        if os.path.isdir(r):
            candidates.append(
                os.path.join(r, "weights", "B4_3stage_17epoch_finetune.pkl")
            )
            candidates.append(os.path.join(r, "B4_3stage_17epoch_finetune.pkl"))

    target_names = {
        "B4_3stage_17epoch_finetune.pkl",
        "b4_3stage_17epoch_finetune.pkl",
    }
    for r in roots:
        if not os.path.isdir(r):
            continue
        try:
            for ds in os.listdir(r):
                p = os.path.join(r, ds)
                if not os.path.isdir(p):
                    continue
                for name in target_names:
                    candidates.append(os.path.join(p, name))
                    candidates.append(os.path.join(p, "weights", name))
        except Exception:
            pass

    for c in candidates:
        if os.path.exists(c):
            return c
    return preferred_path  # keep original for warning message


WEIGHTS_PATH = _find_existing_weights_path(WEIGHTS_PATH)

test_df = pd.read_csv(f"{DATA_ROOT}/test.csv")
test_ids = np.squeeze(test_df["id_code"].values)

train_df = pd.read_csv(f"{DATA_ROOT}/train.csv")
train_label_mode = int(train_df["diagnosis"].mode().iloc[0])

train_counts = train_df["diagnosis"].value_counts().sort_index()
fallback_labels = train_counts.index.values.astype(int)
fallback_probs = (train_counts.values / train_counts.values.sum()).astype(np.float64)

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

net = ThreeStage_Model().to(device)
net.eval()

has_weights = os.path.exists(WEIGHTS_PATH)
if has_weights:
    state = torch.load(WEIGHTS_PATH, map_location=device)
    try:
        net.load_state_dict(state, strict=True)
        print("Loaded weights (strict):", WEIGHTS_PATH)
    except RuntimeError as e:
        print(
            "Strict load failed, retrying strict=False to allow minor key mismatch:",
            str(e).split("\n")[0],
        )
        missing, unexpected = net.load_state_dict(state, strict=False)
        print("Loaded weights (strict=False):", WEIGHTS_PATH)
        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)
else:
    print("WARNING: weights file not found:", WEIGHTS_PATH)
    print("Initializing backbone from timm pretrained weights.")
    pretrained_backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
    missing, unexpected = net.backbone.load_state_dict(
        pretrained_backbone.state_dict(), strict=False
    )
    print(
        "Backbone load_state_dict strict=False; missing:",
        missing,
        "unexpected:",
        unexpected,
    )
    del pretrained_backbone


def predict_continuous(img_tensor_1x3xhxw: torch.Tensor) -> torch.Tensor:
    """
    Keep inference consistent with model components: classifier->softmax probs,
    ordinal->sigmoid probs, regressor already scaled to [0..4.5].
    """
    net.eval()
    c_out, r_out_scaled, o_out_sig = net(img_tensor_1x3xhxw, final=False)
    c_prob = F.softmax(c_out, dim=1)
    x = torch.cat((c_prob, r_out_scaled, o_out_sig), dim=1)  # 5 + 1 + 4 = 10
    out = net.final_regressor(x)
    out = torch.sigmoid(out) * 4.5
    return out


_fallback_feat_resize = 256


def _safe_imread_rgb(path: str):
    try:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception:
        return None


def _retina_features_from_path(path: str):
    img = _safe_imread_rgb(path)
    if img is None:
        return None

    try:
        h, w = img.shape[:2]
        if w / h >= (4 / 3):
            new_h = h
            new_w = int(h * 4 / 3)
        else:
            new_h = int(w * 3 / 4)
            new_w = w
        left = max(0, (w - new_w) // 2)
        top = max(0, (h - new_h) // 2)
        img = img[top : top + new_h, left : left + new_w]

        img_small = cv2.resize(
            img,
            (_fallback_feat_resize, int(_fallback_feat_resize * 3 / 4)),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(img_small, cv2.COLOR_RGB2GRAY)

        lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
        lap_var = float(lap.var())

        dark_ratio = float((gray < 35).mean())

        p10, p90 = np.percentile(gray, [10, 90])
        contrast = float((p90 - p10) / 255.0)

        feat = np.array([math.log1p(lap_var), dark_ratio, contrast], dtype=np.float64)
        if not np.all(np.isfinite(feat)):
            return None
        return feat
    except Exception:
        return None


_fallback_class_centroids = None
_fallback_global_centroid = None
_fallback_class_order = None

if not has_weights:
    tr_ids = train_df["id_code"].values
    y = train_df["diagnosis"].values.astype(int)

    max_fit = 1200
    idx_all = np.arange(len(tr_ids))
    rng = np.random.RandomState(42)
    rng.shuffle(idx_all)
    idx_fit = idx_all[: min(max_fit, len(idx_all))]

    feats = []
    labels = []
    for k in idx_fit:
        img_path = f"{DATA_ROOT}/train_images/{tr_ids[k]}.png"
        if not os.path.exists(img_path):
            alt = f"{DATA_ROOT}/train_images/{tr_ids[k]}.jpg"
            img_path = alt if os.path.exists(alt) else img_path
        f = _retina_features_from_path(img_path)
        if f is None:
            continue
        feats.append(f)
        labels.append(int(y[k]))

    if len(feats) >= 200:
        feats = np.vstack(feats)
        labels = np.asarray(labels, dtype=np.int64)

        centroids = {}
        for c in range(5):
            fc = feats[labels == c]
            if len(fc) == 0:
                continue
            centroids[c] = np.median(fc, axis=0)

        if len(centroids) >= 2:
            _fallback_class_centroids = centroids
            _fallback_class_order = sorted(list(centroids.keys()))
            _fallback_global_centroid = np.median(feats, axis=0)
            print(
                "Fitted fallback centroids for classes:",
                _fallback_class_order,
                "n_fit:",
                len(feats),
            )
        else:
            print("WARNING: could not fit enough class centroids; will use mode.")
    else:
        print(
            "WARNING: insufficient features extracted to fit fallback; will use mode."
        )


def _fallback_predict_label_from_features(path: str) -> int:
    if _fallback_class_centroids is None:
        return int(train_label_mode)
    f = _retina_features_from_path(path)
    if f is None:
        return int(train_label_mode)

    scale = np.array([1.0, 6.0, 2.0], dtype=np.float64)
    f2 = f * scale

    best_c = None
    best_d = None
    for c, cen in _fallback_class_centroids.items():
        d = float(np.sum((f2 - (cen * scale)) ** 2))
        if (best_d is None) or (d < best_d):
            best_d = d
            best_c = c
    return int(np.clip(best_c if best_c is not None else train_label_mode, 0, 4))




## === cell 5
def find_best_thresholds(y_true, y_pred_cont, init_thr=None, n_iters=2):
    thr = np.array(
        init_thr if init_thr is not None else [0.75, 1.5, 2.5, 3.5], dtype=np.float64
    )

    def apply_thr(pred, t):
        pred = np.asarray(pred)
        out = np.zeros_like(pred, dtype=np.int64)
        out += pred >= t[0]
        out += pred >= t[1]
        out += pred >= t[2]
        out += pred >= t[3]
        return out

    best_kappa = cohen_kappa_score(
        y_true, apply_thr(y_pred_cont, thr), weights="quadratic"
    )

    for _ in range(n_iters):
        for i in range(4):
            cur = thr[i]
            grid = np.arange(cur - 0.35, cur + 0.3501, 0.025)
            grid = np.clip(grid, 0.05, 4.25)
            for cand in grid:
                t2 = thr.copy()
                t2[i] = cand
                if not (t2[0] < t2[1] < t2[2] < t2[3]):
                    continue
                k = cohen_kappa_score(
                    y_true, apply_thr(y_pred_cont, t2), weights="quadratic"
                )
                if k > best_kappa:
                    best_kappa = k
                    thr = t2
    return thr.tolist(), float(best_kappa)


USE_THRESHOLD_TUNING = bool(has_weights)

if USE_THRESHOLD_TUNING:
    tr_ids = train_df["id_code"].values
    y = train_df["diagnosis"].values.astype(int)

    tr_idx, va_idx = train_test_split(
        np.arange(len(tr_ids)),
        test_size=0.15,
        random_state=42,
        stratify=y,
    )
    max_val = 256
    va_idx = va_idx[:max_val]

    y_true = y[va_idx]
    y_pred_cont = []

    net.eval()
    with torch.no_grad():
        for k in va_idx:
            image_name = f"{DATA_ROOT}/train_images/{tr_ids[k]}.png"
            if not os.path.exists(image_name):
                alt = f"{DATA_ROOT}/train_images/{tr_ids[k]}.jpg"
                image_name = alt if os.path.exists(alt) else image_name

            try:
                img = Image.open(image_name).convert("RGB")
            except Exception:
                y_pred_cont.append(float(train_label_mode))
                continue

            img = transform(img).unsqueeze(0).to(device)

            r_out = predict_continuous(img)
            y_pred_cont.append(float(r_out.squeeze(1).item()))

    new_thr, val_kappa = find_best_thresholds(
        y_true, y_pred_cont, init_thr=threshold, n_iters=2
    )
    print("Threshold tuning val_kappa:", val_kappa)
    print("Old thresholds:", threshold, "New thresholds:", new_thr)
    threshold = new_thr
else:
    print(
        "Threshold tuning disabled because WEIGHTS_PATH is missing (fallback baseline)."
    )



## === cell 6
submission = []

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print(i, "/", len(test_ids))

        image_name = f"{DATA_ROOT}/test_images/{idx}.png"
        if not os.path.exists(image_name):
            alt = f"{DATA_ROOT}/test_images/{idx}.jpg"
            image_name = alt if os.path.exists(alt) else image_name

        if not has_weights:
            pred_label = _fallback_predict_label_from_features(image_name)
            submission.append([idx, pred_label])
            continue

        try:
            img = Image.open(image_name).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            r_out = predict_continuous(img)
            pred = regress2class(r_out.squeeze(1))
            submission.append([idx, int(pred.item())])
        except Exception:
            submission.append([idx, int(train_label_mode)])

submission = np.array(submission, dtype=object)



## === cell 7
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = (
    pd.to_numeric(df["diagnosis"], errors="coerce").fillna(train_label_mode).astype(int)
)
df["diagnosis"] = df["diagnosis"].clip(0, 4)

df = test_df.merge(df, on="id_code", how="left")
df["diagnosis"] = df["diagnosis"].fillna(train_label_mode).astype(int)
df = df[["id_code", "diagnosis"]]

out_path = "submission.csv"
df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(df.head())
print("Rows:", len(df), "Cols:", df.shape[1])
print("Diagnosis value counts:\n", df["diagnosis"].value_counts().sort_index())
