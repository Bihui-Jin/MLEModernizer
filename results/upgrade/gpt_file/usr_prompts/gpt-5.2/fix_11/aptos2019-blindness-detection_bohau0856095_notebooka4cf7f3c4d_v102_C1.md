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

0.9141751547995498

# 6. Current score

0.41492

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.00014) has done: 'I make the code robust to the missing `../input/weights` directory by falling back to `pretrained=True` EfficientNet weights when no custom checkpoint exists, so the notebook can always run end-to-end and produce a valid CSV. I also fix the CUDA/CPU mismatch by ensuring the loaded state dict is moved onto the same device as the model (and avoiding hard-coded `.cuda()` in helper functions). Finally, I guard submission creation so `submission` is always defined and aligned to `test.csv` order, writing `submission.csv` with the required columns.'
- What this solution (achieved -0.0597) has done: 'Your current score is near-random because, when no custom weights are available, you still run the untrained heads (classifier/regressor/ordinal) and then hard-threshold their outputs, which collapses predictive signal. To move the score toward the 0.914 target with minimal change, I keep your exact model/transform/inference loop, but add a safe fallback: if no competition-specific checkpoint is found, do test-time inference using the backbone’s built-in pretrained classifier logits (ImageNet) mapped into 5 classes via argmax%5, instead of using random heads. This preserves your pipeline and produces a valid submission while making predictions non-degenerate, which should substantially increase QWK versus random. If weights are found, behavior remains identical to your current approach.'
- What this solution (achieved 0.0) has done: 'Your negative QWK is coming from the fallback path: when no DR-trained checkpoint is available you map ImageNet logits to 5 classes with `% 5`, which is essentially random with respect to DR labels. To move the score toward your 0.914 target with minimal change and without altering your model/training logic, I keep your exact model and transforms but replace the fallback with a deterministic “no-weights prior” that uses the competition’s known label distribution from `train.csv` and predicts the modal class (0) for all test images. This typically yields a non-random, stable baseline QWK (often ~0 or slightly above) and should significantly improve from -0.0597 toward the target. If a proper checkpoint exists in `../input/weights`, your original inference path remains unchanged.'
- What this solution (achieved -0.11884) has done: 'Your 0.0 score is coming from the “no checkpoint found” path, which currently predicts the majority class for every test image; that’s a stable baseline but far from your 0.914 target. To move the score upward with minimal changes and without altering your model or training logic, I keep your exact `ThreeStage_Model` inference, but remove the majority-class shortcut and always run the model (using `pretrained=True` backbone when no DR-specific weights exist). To make those pretrained features produce less-collapsed predictions, I calibrate the regression thresholds using the training set distribution (quantiles) instead of hard-coded `[0.75, 1.5, 2.5, 3.5]`, which is a small post-processing change aligned with QWK. The code still writes a valid `submission.csv` in the required format and keeps the checkpoint path behavior unchanged if you later provide weights.'
- What this solution (achieved -0.13158) has done: 'Your current negative QWK is most consistent with producing largely uncorrelated predictions because, in the no-DR-checkpoint case, you’re effectively using an ImageNet-pretrained backbone with randomly initialized heads. To move the score upward toward the 0.914 target with minimal change (and without changing your model/training core), I keep your exact model and inference loop but switch the prediction used at inference to the classifier head (argmax of 5 logits) when no checkpoint is found, instead of thresholding the regressor output. When a real competition checkpoint is present, behavior remains the same as your original intent (still uses the regressor+thresholds path). This avoids the regressor-from-random-head failure mode while preserving the rest of your pipeline and writing a valid `submission.csv`.'
- What this solution (achieved 0.12932) has done: 'Your score is far below the 0.914 target, and the main issue is that when no DR-specific checkpoint is found you’re effectively using randomly initialized heads (classifier/regressor/ordinal) on top of an ImageNet backbone, so predictions are near-random and can go negative QWK. With minimal change and without touching the model/training core, I keep your same architecture and transforms but swap the no-checkpoint inference to use the backbone’s own ImageNet logits as a stable feature signal, then map them to a 0–4 label via a learned linear mapping fit on the training set using frozen backbone logits (fast, no training loop changes). When a real checkpoint exists, your original regressor+threshold path remains unchanged. This should move you substantially upward from -0.13 toward the target while still producing the same valid `submission.csv`.'
- What this solution (achieved 0.25066) has done: 'You’re far below the 0.914 target (gap ≈ -0.785), and the main driver is that in the no-checkpoint case you’re using a DR-untrained head (argmax of `c_out`) or a weak ridge mapping; both are effectively uncorrelated with DR severity. To move the score upward with minimal disruption to your existing architecture/inference semantics, I keep the same model and transforms but improve the no-weights fallback by fitting a tiny **ordinal calibration** on a small training subset: learn 5 class centroids in the backbone feature space, predict by nearest-centroid, then apply the same quantile-based thresholding you already use to better match the label distribution. This keeps runtime bounded (small subset + batched feature extraction) and leaves the “weights exist” path unchanged. The submission writing/alignment remains identical and still produces a valid `submission.csv`.'
- What this solution (achieved 0.44928) has done: 'Your current score (0.25066) is far below the target (0.91418), so we should increase performance with the smallest change that fixes the main failure mode: in the no-checkpoint path you’re using a DR-untrained/random head (`argmax(c_out)`) or weak centroid heuristic, which yields mostly uncorrelated labels. I keep your model, transforms, and inference structure, but replace the no-weights fallback with a minimal, fast supervised calibration on top of the **frozen backbone features**: fit a 5-class multinomial Logistic Regression on a moderate training subset, then apply it to test features and (optionally) blend with your centroid prediction for stability. This preserves your core logic (same backbone forward, same thresholding mechanism kept for the checkpoint path) while making the no-weights path actually learn DR separation from `train.csv`, which should move QWK substantially upward toward the target without changing training loops or architecture. Submission writing and alignment remain unchanged and it still produce `submission.csv` end-to-end.'
- What this solution (achieved 0.41492) has done: 'Your current gap to the target is large (0.449 → 0.914), and the biggest limiter is that the “no checkpoint” path relies on a light LR-on-features calibration that is likely underpowered and mis-calibrated for QWK. I keep your exact backbone/model and inference flow, but make the no-weights calibrator stronger and more stable by (1) fitting LR on a larger, stratified subset of training data, and (2) standardizing backbone features before LR (same features, just better-conditioned optimization). I also make the fallback blend (LR + centroid) enabled at low weight to reduce brittle argmax errors without changing the checkpoint path at all. Submission writing, paths, transforms, and the checkpoint inference branch remain unchanged.'

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
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops
import cv2

from sklearn.metrics import cohen_kappa_score
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
    return prediction


def ordinal2class_prob(out):
    dev = out.device
    pred_prob = torch.zeros(out.size(0), 5, device=dev)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    dev = out.device
    pred_prob = torch.zeros((out.size(0), 5), device=dev)
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
    def __init__(self, pretrained=False):
        super(Regressor, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b5_ns(pretrained=pretrained)
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.5
        return out


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None, pretrained=False):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=pretrained)
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
TEST_CSV = os.path.join(BASE, "test.csv")
TEST_IMG_DIR = os.path.join(BASE, "test_images")
TRAIN_CSV = os.path.join(BASE, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

train_df = pd.read_csv(TRAIN_CSV)
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["diagnosis"] = (
    pd.to_numeric(train_df["diagnosis"], errors="coerce")
    .fillna(0)
    .astype(int)
    .clip(0, 4)
)

label_counts = train_df["diagnosis"].value_counts().sort_index()
label_priors = (
    (label_counts / label_counts.sum()).reindex(range(5), fill_value=0.0).values
)
cum_priors = np.cumsum(label_priors)
cum_priors = np.clip(cum_priors, 1e-6, 1 - 1e-6)
threshold = (4.5 * cum_priors[:4]).tolist()

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

WEIGHTS_DIR = "../input/weights"
preferred = os.path.join(WEIGHTS_DIR, "B4_3stage_20epoch_320.pkl")

weights_path = None
if os.path.exists(preferred):
    weights_path = preferred
elif os.path.isdir(WEIGHTS_DIR):
    candidates = []
    for pat in ["*.pkl", "*.pth", "*.pt", "*.bin"]:
        candidates.extend(sorted(glob.glob(os.path.join(WEIGHTS_DIR, pat))))
    if len(candidates) > 0:
        weights_path = candidates[0]

use_pretrained_backbone = weights_path is None
net = ThreeStage_Model(pretrained=use_pretrained_backbone)

if weights_path is not None:
    state = torch.load(weights_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_state[nk] = v
        state = new_state
    net.load_state_dict(state, strict=True)

net = net.to(device)
net.eval()

ridge_model = None  # kept for compatibility with existing structure
centroids = None
centroid_means = None

lr_model = None  # multinomial logistic regression on backbone features
scaler = None

use_blend = True

if weights_path is None:
    max_fit_total = 2600  # still bounded for runtime; train has 3295
    per_class_cap = max(50, max_fit_total // 5)

    parts = []
    for k in range(5):
        dfk = train_df[train_df["diagnosis"] == k]
        if len(dfk) == 0:
            continue
        parts.append(dfk.sample(n=min(per_class_cap, len(dfk)), random_state=42))
    sample_train = (
        pd.concat(parts, axis=0)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    feats = []
    ys = []
    batch_imgs = []
    batch_y = []

    bs = 24  # slightly larger batch to keep extraction faster; does not change model logic
    with torch.no_grad():
        for _, row in sample_train.iterrows():
            img_path = os.path.join(TRAIN_IMG_DIR, f"{row['id_code']}.png")
            try:
                img = Image.open(img_path).convert("RGB")
                batch_imgs.append(transform(img))
                batch_y.append(int(row["diagnosis"]))
                if len(batch_imgs) >= bs:
                    x = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
                    f = net.backbone(x).detach().float().cpu().numpy()
                    feats.append(f)
                    ys.extend(batch_y)
                    batch_imgs, batch_y = [], []
            except Exception:
                continue

        if len(batch_imgs) > 0:
            x = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
            f = net.backbone(x).detach().float().cpu().numpy()
            feats.append(f)
            ys.extend(batch_y)

    if len(feats) > 0 and len(ys) >= 400:
        X = np.concatenate(feats, axis=0)
        y = np.asarray(ys, dtype=np.int64)

        mu = X.mean(axis=0)
        centroids = np.zeros((5, X.shape[1]), dtype=np.float32)
        counts = np.zeros(5, dtype=np.int64)

        for k in range(5):
            idxs = np.where(y == k)[0]
            counts[k] = len(idxs)
            if counts[k] > 0:
                ck = X[idxs].mean(axis=0)
                alpha = counts[k] / (counts[k] + 50.0)
                centroids[k] = (alpha * ck + (1.0 - alpha) * mu).astype(np.float32)
            else:
                centroids[k] = mu.astype(np.float32)

        centroid_means = np.arange(5, dtype=np.float32)

        scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = scaler.fit_transform(X)

        try:
            lr_model = LogisticRegression(
                multi_class="multinomial",
                solver="lbfgs",
                max_iter=600,
                C=2.0,
                class_weight="balanced",
                n_jobs=1,
            )
            lr_model.fit(Xs, y)
        except Exception:
            lr_model = None
            scaler = None
    else:
        centroids = None
        centroid_means = None
        lr_model = None
        scaler = None

print(f"Using device: {device}")
print(
    f"Loaded weights: {weights_path if weights_path is not None else '[NONE] (using pretrained backbone)'}"
)
print(f"Calibrated thresholds (0..4.5): {threshold}")
print(f"Num test ids: {len(test_ids)}")
print(
    f"No-weights calibrator: {'LR+SCALER' if (lr_model is not None and scaler is not None) else ('CENTROID' if centroids is not None else 'DISABLED')}"
)



## === cell 5
submission_rows = []

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print(f"Infer {i}/{len(test_ids)}")

        image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")

        try:
            img = Image.open(image_name).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)

            if weights_path is None:
                if lr_model is not None and scaler is not None:
                    f = (
                        net.backbone(img_t)
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .reshape(1, -1)
                    )
                    fs = scaler.transform(f)
                    prob = lr_model.predict_proba(fs)[0]  # 5

                    if use_blend and centroids is not None:
                        d = ((centroids[None, :, :] - f[:, None, :]) ** 2).sum(axis=2)[
                            0
                        ]
                        k_cent = int(np.argmin(d))
                        prob2 = np.full(5, 0.0, dtype=np.float32)
                        prob2[k_cent] = 1.0
                        prob = 0.90 * prob + 0.10 * prob2

                    pred = int(np.argmax(prob))
                elif centroids is not None:
                    f = (
                        net.backbone(img_t)
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                        .reshape(1, -1)
                    )
                    d = ((centroids[None, :, :] - f[:, None, :]) ** 2).sum(axis=2)
                    k = int(np.argmin(d, axis=1)[0])
                    y_cont = float(centroid_means[k])  # 0..4
                    y_cont = float(np.clip(y_cont * (4.5 / 4.0), 0.0, 4.5))
                    pred = int(sum([y_cont >= t for t in threshold]))
                else:
                    c_out, _, _ = net(img_t)
                    pred = int(torch.argmax(c_out, dim=1).item())
            else:
                c_out, r_out, _ = net(img_t)
                pred = int(regress2class(r_out.data.squeeze(1)).item())

        except Exception:
            pred = 0

        submission_rows.append([idx, pred])

submission = np.array(submission_rows, dtype=object)



## === cell 6
sub_df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
sub_df["id_code"] = sub_df["id_code"].astype(str)
sub_df["diagnosis"] = (
    pd.to_numeric(sub_df["diagnosis"], errors="coerce").fillna(0).astype(int)
)

sub_df = sub_df.set_index("id_code").reindex(test_ids).reset_index()
sub_df["diagnosis"] = sub_df["diagnosis"].fillna(0).astype(int).clip(0, 4)

out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print(sub_df.head())
print(f"Wrote {out_path} with shape {sub_df.shape}")
print(sub_df["diagnosis"].value_counts(dropna=False).sort_index())
