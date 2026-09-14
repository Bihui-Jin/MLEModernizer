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

0.9146394946393326

# 6. Current score

0.47869

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the two blockers preventing an end-to-end run: (1) the missing `../input/weights` directory by falling back to a deterministic, minimal “no-weights” inference path that still produces a valid submission.csv, and (2) the CUDA/CPU dtype mismatch by ensuring the model parameters and inputs are on the same device before inference. I also make the test image directory resolution robust to the dataset being mounted at either `../input/...` or `/kaggle/input/...`, and I harden image loading so a single bad/missing image won’t zero out the submission. These changes preserve your model code and prediction semantics; they only adjust runtime robustness so a CSV is always produced.'
- What this solution (achieved 0.03993) has done: 'Your current 0.0 score is because the model is effectively untrained at inference time (no weights), so the smallest legitimate improvement is to load a strong pretrained baseline for the same backbone and keep everything else (model, transforms, regression-to-class thresholds, and inference loop) identical. I therefore switch the EfficientNet-B4 backbone inside `ThreeStage_Model` to `pretrained=True` only when no external weights are found, and I keep the exact same forward semantics and post-processing. This is a minimal change that should move QWK up substantially (toward your target) without altering the core architecture or training approach (there is still no training loop). I also make the prediction tensor device/dtype handling slightly safer (no semantic change) to avoid rare edge failures.'
- What this solution (achieved -0.00672) has done: 'Your score is far below the target because the current inference uses the regressor head (`r_out`) whose weights are random when no competition checkpoint is available; only the backbone is pretrained, so predictions are essentially noise. To move toward the target without changing your model or training logic, I keep the exact same architecture and inference loop but (1) use the classifier logits (`c_out`) as the prediction source when running in the no-checkpoint fallback mode, since the classifier head at least provides a stable argmax signal, and (2) set `threshold` adaptively from the training label distribution to avoid systematic under/over-prediction when still using the regressor path. These are minimal, metric-relevant changes (better ordinal class assignment), preserve evaluation semantics (still predicting 0–4), and keep the original regressor-based path unchanged when real weights exist. The script still runs end-to-end and writes `submission.csv` with the required columns.'
- What this solution (achieved -0.07646) has done: 'Your current score is far below the target because, in the no-checkpoint fallback, the model’s heads are randomly initialized, so using `argmax(c_out)` is effectively noise. To move the score upward without changing your model or training loop, I keep the same architecture and inference flow but change only the fallback prediction source to use the pretrained EfficientNet-B4 features directly (via the backbone’s 1000-d logits) and map them deterministically to classes. I also calibrate that mapping to the training label distribution (prior matching) so predicted class frequencies are reasonable for QWK, while leaving the “real weights” path (regressor + thresholds) unchanged. This is a minimal, metric-relevant fix that should substantially increase QWK versus the current random-head fallback and still produces a valid `submission.csv`.'
- What this solution (achieved 0.41942) has done: 'Your score is far below the target because the “no-checkpoint fallback” is currently mapping ImageNet logits to DR classes, which is essentially unrelated to the task and yields near-random ordering for QWK. Keeping your architecture and inference loop intact, I make the fallback use the backbone features but predict with a simple nearest-class-prototype rule computed from the training set using the same backbone (no training, just forward passes). This is a minimal, metric-relevant change that typically boosts ordinal agreement because it anchors predictions to the training distribution in feature space. The normal “real weights found” path (regressor + thresholds) remains unchanged, and the script still writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current score (0.41942) is far below the target (0.91464), so we should improve QWK with minimal, metric-relevant changes while keeping your model/inference flow intact. The biggest safe gain here is to replace the hard argmax-over-prototypes fallback with a distance-to-class-centroid regression (expected value over softmaxed similarities), then discretize using thresholds computed from the training labels—this preserves the same “backbone features only, no training” core approach but better matches the ordinal nature of QWK. I also compute thresholds via cumulative class proportions (more appropriate for 0–4 labels than raw quantiles on integer labels) and keep the “real weights found” path completely unchanged. These changes should increase ordinal agreement without changing architecture, training loops, or loss functions, and the script still write a valid `submission.csv`.'
- What this solution (achieved 0.47869) has done: 'Your current 0.0 score indicates the fallback inference path is still effectively uncorrelated with the DR labels, so the smallest legitimate move toward the 0.9146 target is to keep your backbone+prototype fallback but compute class thresholds from the training feature distribution (instead of fixed midpoints), which directly optimizes the ordinal discretization step used by QWK. I leave the “real weights found” path unchanged, and only adjust the no-checkpoint fallback to (a) build prototypes with a slightly larger, deterministic per-class sample and (b) derive 4 cutpoints by matching the predicted continuous severity distribution on the train subset to the observed train label cumulative proportions. This preserves your model architecture and no-training approach (still just forward passes), but makes the final 0–4 mapping much better calibrated to this dataset, which should increase QWK substantially from 0.0. The script still run end-to-end within the time limit and write a valid `submission.csv`.'

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
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



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
    def __init__(self, backbone=None, backbone_pretrained=False):
        super(ThreeStage_Model, self).__init__()

        self.backbone = timm.models.tf_efficientnet_b4_ns(
            pretrained=backbone_pretrained
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
def resolve_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


DATA_DIR = resolve_existing_path(
    [
        "../input/aptos2019-blindness-detection",
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
    ]
)
if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not locate aptos2019-blindness-detection dataset directory."
    )

TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

train_df = pd.read_csv(TRAIN_CSV)
y = train_df["diagnosis"].astype(int).values

try:
    class_counts = np.bincount(y, minlength=5).astype(float)
    priors = class_counts / max(class_counts.sum(), 1.0)
    print("Train class priors:", priors.round(4).tolist())
except Exception as e:
    priors = np.array([1, 0, 0, 0, 0], dtype=float)
    print("Could not compute priors; using fallback. Reason:", repr(e))

class_counts_int = np.bincount(y, minlength=5).astype(int)
print("Train class counts:", class_counts_int.tolist())

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

weights_dir = resolve_existing_path(
    ["../input/weights", "/kaggle/input/weights", "/kaggle/data/weights"]
)
weights_path = None
if weights_dir is not None:
    preferred = os.path.join(weights_dir, "B4_3stage_49epoch_CLAHE.pkl")
    if os.path.exists(preferred):
        weights_path = preferred
    else:
        candidates = (
            glob.glob(os.path.join(weights_dir, "*.pkl"))
            + glob.glob(os.path.join(weights_dir, "*.pth"))
            + glob.glob(os.path.join(weights_dir, "*.pt"))
        )
        candidates = sorted(candidates)
        if len(candidates) > 0:
            weights_path = candidates[0]

use_pretrained_backbone = weights_path is None
net = ThreeStage_Model(backbone_pretrained=use_pretrained_backbone)

if weights_path is not None:
    print("Loading weights:", weights_path)
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
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_state[nk] = v
        net.load_state_dict(new_state, strict=False)
else:
    print(
        "WARNING: No competition weights found; using ImageNet-pretrained backbone. Using train-set prototypes for fallback."
    )

net = net.to(device)
net.eval()

USE_PROTOTYPE_FALLBACK = weights_path is None


def build_class_prototypes(
    net,
    train_df,
    img_dir,
    transform,
    device,
    max_per_class=200,
    seed=0,
):
    rng = np.random.RandomState(seed)
    df = train_df.copy()
    df["diagnosis"] = df["diagnosis"].astype(int)

    selected_rows = []
    for cls in range(5):
        cls_rows = df[df["diagnosis"] == cls]
        if len(cls_rows) == 0:
            continue
        take = min(max_per_class, len(cls_rows))
        idxs = rng.choice(len(cls_rows), size=take, replace=False)
        selected_rows.append(cls_rows.iloc[idxs])
    sel = pd.concat(selected_rows, axis=0).reset_index(drop=True)

    feats_by_class = {c: [] for c in range(5)}
    with torch.no_grad():
        for i in range(len(sel)):
            img_id = str(sel.loc[i, "id_code"])
            cls = int(sel.loc[i, "diagnosis"])
            path = os.path.join(img_dir, f"{img_id}.png")
            try:
                img = Image.open(path).convert("RGB")
                x = transform(img).unsqueeze(0).to(device)
                f = net.backbone(x)  # [1,1000]
                f = F.normalize(f, dim=1)
                feats_by_class[cls].append(f.squeeze(0).detach().float().cpu())
            except Exception:
                continue

    protos = []
    for cls in range(5):
        if len(feats_by_class[cls]) == 0:
            protos.append(torch.zeros(1000, dtype=torch.float32))
        else:
            m = torch.stack(feats_by_class[cls], dim=0).mean(dim=0)
            m = m / (m.norm(p=2) + 1e-12)
            protos.append(m)
    protos = torch.stack(protos, dim=0)  # [5,1000]
    return protos.to(device)


def calibrate_thresholds_from_train_predictions(
    net,
    prototypes,
    train_df,
    img_dir,
    transform,
    device,
    priors,
    max_samples=1200,
    seed=0,
):
    rng = np.random.RandomState(seed)
    df = train_df.copy()
    if len(df) > max_samples:
        idxs = rng.choice(len(df), size=max_samples, replace=False)
        df = df.iloc[idxs].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    proto_labels = torch.arange(5, device=device, dtype=torch.float32)

    sevs = []
    with torch.no_grad():
        for i in range(len(df)):
            img_id = str(df.loc[i, "id_code"])
            path = os.path.join(img_dir, f"{img_id}.png")
            try:
                img = Image.open(path).convert("RGB")
                x = transform(img).unsqueeze(0).to(device)
                f = net.backbone(x)
                f = F.normalize(f, dim=1)
                sims = torch.matmul(f, prototypes.t()).squeeze(0)  # [5]
                probs = F.softmax(sims, dim=0)
                sev = float((probs * proto_labels).sum().item())
                if math.isfinite(sev):
                    sevs.append(sev)
            except Exception:
                continue

    if len(sevs) < 50:
        print("Threshold calibration skipped: too few train predictions:", len(sevs))
        return None

    sevs = np.asarray(sevs, dtype=np.float64)
    sevs.sort()

    cum_priors = np.cumsum(np.asarray(priors, dtype=np.float64))
    cuts = []
    n = len(sevs)
    for k in range(4):
        q = float(cum_priors[k])  # proportion of <=k
        q = min(max(q, 0.0), 1.0)
        idx = int(round(q * (n - 1)))
        idx = max(0, min(n - 1, idx))
        cuts.append(float(sevs[idx]))
    for i in range(1, 4):
        if cuts[i] <= cuts[i - 1]:
            cuts[i] = cuts[i - 1] + 1e-3
    return cuts


prototypes = None
if USE_PROTOTYPE_FALLBACK:
    t0 = time.time()
    prototypes = build_class_prototypes(
        net=net,
        train_df=train_df,
        img_dir=TRAIN_IMG_DIR,
        transform=transform,
        device=device,
        max_per_class=200,  # CHANGE: more stable class centroids, still fast enough
        seed=0,
    )
    print("Built prototypes in %.1fs" % (time.time() - t0))

    t1 = time.time()
    new_thr = calibrate_thresholds_from_train_predictions(
        net=net,
        prototypes=prototypes,
        train_df=train_df,
        img_dir=TRAIN_IMG_DIR,
        transform=transform,
        device=device,
        priors=priors,
        max_samples=1200,  # CHANGE: modest sample for stable quantiles within time limit
        seed=1,
    )
    if new_thr is not None and len(new_thr) == 4 and np.all(np.isfinite(new_thr)):
        threshold = [float(x) for x in new_thr]
        print(
            "Calibrated thresholds (fallback):",
            threshold,
            "in %.1fs" % (time.time() - t1),
        )
    else:
        print("Using default thresholds (fallback):", threshold)

print("Final thresholds used by regress2class:", threshold)



## === cell 5
submission = []

proto_labels = (
    torch.arange(5, device=device, dtype=torch.float32)
    if USE_PROTOTYPE_FALLBACK
    else None
)

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print("Processed", i, "/", len(test_ids))

        image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
        try:
            img = Image.open(image_name).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)

            if USE_PROTOTYPE_FALLBACK:
                f = net.backbone(img_t)  # [1,1000]
                f = F.normalize(f, dim=1)
                sims = torch.matmul(f, prototypes.t()).squeeze(0)  # [5]
                probs = F.softmax(sims, dim=0)  # [5]
                sev = (probs * proto_labels).sum().view(1)  # [1] continuous in [0,4]
                pred_class = int(regress2class(sev)[0].item())
                submission.append([idx, pred_class])
            else:
                c_out, r_out, _ = net(img_t)
                r_out_1d = r_out.view(-1)
                pred = regress2class(r_out_1d)
                pred_class = int(pred.item())
                submission.append([idx, pred_class])
        except Exception:
            submission.append([idx, 0])

submission = np.array(submission, dtype=object)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

if df.shape[0] != len(test_ids):
    raise RuntimeError(
        f"Submission row count mismatch: got {df.shape[0]}, expected {len(test_ids)}"
    )

df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)
df = test_df.merge(df, on="id_code", how="left", validate="one_to_one")
if df["diagnosis"].isna().any():
    df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
