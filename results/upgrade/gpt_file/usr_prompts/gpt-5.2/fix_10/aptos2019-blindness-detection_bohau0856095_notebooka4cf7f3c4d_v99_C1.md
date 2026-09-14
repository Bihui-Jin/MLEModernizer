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

0.9237673810862916

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02917) has done: 'I fix the pipeline so it runs end-to-end even when the external checkpoint file is missing by (1) loading the model checkpoint only if it exists, otherwise cleanly falling back to a deterministic, label-distribution-based baseline that still produces a valid submission. I also correct the inference call to the model: the current code calls `net(imgs)` expecting three outputs, but the model returns three outputs only when `final=False` (which we keep), and we ensure `net` is always defined. Finally, I make the submission generation robust by always producing exactly the test set rows in the right order and writing `submission.csv` with the required columns.'
- What this solution (achieved -0.02576) has done: 'Your score is far below the target, and the main reason is that you’re not actually using a trained model: `pretrained=False` plus a missing checkpoint forces the deterministic random/majority baseline, which yields near-random kappa. The smallest legitimate change to move toward the target is to (1) switch the backbones to `pretrained=True` so inference has meaningful features even without the external `.pkl`, and (2) add a minimal, deterministic test-time augmentation (horizontal flip) and average predictions, which typically improves kappa without changing the core model/loop. I also keep your existing checkpoint-loading behavior unchanged, so if a checkpoint is present it’s still used exactly as before. The submission writing and ordering logic stays the same.'
- What this solution (achieved -0.03171) has done: 'Your current score is far below the target, so we should improve the actual predictive signal without changing your model design or training approach (you do no training here). The smallest meaningful fix is to make inference consistent regardless of checkpoint availability: always use the same (slightly stronger) test-time augmentation averaging (original + horizontal flip), not only in the no-checkpoint branch. Additionally, we run the model in `final=True` mode and then threshold that single regressed output, which matches how the architecture is intended to be used when combining classifier/regressor/ordinal heads (this is still the same model, just using its proper forward path). These changes should increase kappa substantially toward your target while keeping everything deterministic and preserving the core logic.'
- What this solution (achieved 0.10474) has done: 'Your score is far below the target, so the smallest meaningful improvement is to better align post-processing with the quadratic weighted kappa metric without changing your model or adding training. I keep your exact model/inference flow, but replace the fixed regression thresholds with thresholds optimized on the training label distribution (a common kappa-aligned calibration step) and use them in `regress2class`. This is a minimal, deterministic change that often boosts kappa substantially when the raw regressor is miscalibrated. I also make sure the calibrated thresholds are used regardless of whether a checkpoint is found, and keep the submission writing/order logic identical.'
- What this solution (achieved 0.11312) has done: 'Your current score (0.10474) is far below the target (0.92377), so we should improve true predictive signal without changing the model, training loop (none), or loss. The biggest issue is that you’re “calibrating” thresholds by fitting them on the same training predictions you later rely on, which overfits and can harm generalization/QWK on test; I keep the same threshold-optimization logic but make it out-of-fold by adding a deterministic train/val split and optimizing thresholds on the held-out fold only. I also ensure the calibration pass runs in true inference mode (`net.eval()` and `torch.inference_mode()`) and avoid any accidental gradient/mode differences. Everything else (architecture, transforms, TTA, submission ordering/format, paths) stays the same.'
- What this solution (achieved 0.0) has done: 'Your current score (0.11312) is far below the target (0.92377), so we should improve real predictive signal without changing your model, transforms, or inference semantics. The biggest likely issue is input normalization: EfficientNet backbones in `timm` expect ImageNet normalization, but your custom mean/std can heavily miscalibrate features and collapse predictions; switching to the backbone’s default `timm` data config is a minimal, metric-relevant fix. I keep your exact architecture, forward usage (`final=True`), TTA (hflip average), and OOF threshold calibration logic, only swapping the normalization to the correct pretrained config and keeping the rest identical. This should move QWK substantially upward toward the target while remaining deterministic and within Kaggle constraints.'
- What this solution (achieved -0.0037) has done: 'Your current score (0.0) is far below the target, so the priority is to fix the most likely cause of a “broken” submission: predictions collapsing to a constant class due to a preprocessing mismatch with the pretrained EfficientNet backbones. I keep your model, forward path (`final=True`), and OOF threshold calibration, but replace the custom PIL trim/crop/resize pipeline with the exact `timm` pretrained inference pipeline (resize/crop + normalization) for both train-calibration and test inference. This is a minimal, metric-relevant change that typically restores meaningful logits for pretrained models and should move QWK substantially upward toward your target band. I also add a small safety clamp on the regressed outputs before thresholding to avoid any NaN/Inf edge cases affecting the submission.'
- What this solution (achieved 0.0) has done: 'Your score is extremely far below the target, so the most likely issue is that inference is running on a randomly-initialized “head” (classifier/regressor/ordinal/final_regressor) even though the EfficientNet backbone is pretrained; this typically collapses predictions and destroys QWK. I keep your exact architecture and inference flow, but (1) load the checkpoint from the competition dataset if it exists (it commonly ships with the notebook dataset) by expanding the candidate paths, and (2) if no checkpoint is found, fall back to using only the pretrained backbone features with a deterministic, metric-aligned label-distribution mapping (instead of the untrained heads), so predictions aren’t random. I also make threshold calibration conditional on having meaningful continuous predictions (checkpoint available), because calibrating thresholds on untrained outputs is counterproductive. All changes are minimal and keep your submission format/order unchanged.'

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
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

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
        self.backbone = timm.models.tf_efficientnet_b5_ns(pretrained=True)
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

        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
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
DATA_DIR = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

input_size = 380

_timm_ref_model = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
timm_cfg = timm.data.resolve_model_data_config(_timm_ref_model)
imagenet_mean = timm_cfg["mean"]
imagenet_std = timm_cfg["std"]
transform = timm.data.create_transform(**timm_cfg, is_training=False)


def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


ckpt_candidates = [
    "../input/weights/B4_3stage_18epoch_finetune3.pkl",
    "/kaggle/input/weights/B4_3stage_18epoch_finetune3.pkl",
    "/kaggle/input/aptos2019-blindness-detection/B4_3stage_18epoch_finetune3.pkl",
    "../input/aptos2019-blindness-detection/B4_3stage_18epoch_finetune3.pkl",
    os.path.join(DATA_DIR, "B4_3stage_18epoch_finetune3.pkl"),
    os.path.join(DATA_DIR, "weights", "B4_3stage_18epoch_finetune3.pkl"),
]
ckpt_path = find_first_existing(ckpt_candidates)

net = ThreeStage_Model().to(device)
has_ckpt = False
if ckpt_path is not None:
    try:
        state = torch.load(ckpt_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = net.load_state_dict(state, strict=False)
        has_ckpt = True
        print(f"Loaded checkpoint: {ckpt_path}")
        if len(missing) or len(unexpected):
            print(
                f"Warning: missing keys={len(missing)}, unexpected keys={len(unexpected)} (loaded with strict=False)."
            )
    except Exception as e:
        print(f"Warning: failed to load checkpoint at {ckpt_path}: {e}")
        has_ckpt = False
else:
    print(
        "Warning: checkpoint not found; will avoid using random untrained heads by falling back to a deterministic baseline."
    )

net.eval()

train_label_probs = None
train_majority_class = 0
train_labels = None
train_df = None

if os.path.exists(TRAIN_CSV):
    train_df = pd.read_csv(TRAIN_CSV)
    train_labels = train_df["diagnosis"].astype(int).values
    counts = (
        train_df["diagnosis"]
        .value_counts()
        .reindex([0, 1, 2, 3, 4], fill_value=0)
        .values.astype(np.float64)
    )
    probs = counts / max(counts.sum(), 1.0)
    train_label_probs = probs
    train_majority_class = int(np.argmax(counts))
else:
    train_label_probs = np.array([1, 0, 0, 0, 0], dtype=np.float64)
    train_majority_class = 0
    print("Baseline: train.csv not found; defaulting all predictions to 0")


def _apply_thresholds(y_cont, thr):
    y_cont = np.asarray(y_cont, dtype=np.float64).reshape(-1)
    thr = np.asarray(thr, dtype=np.float64).reshape(4)
    return np.digitize(y_cont, bins=thr, right=False).astype(int)


def _optimize_qwk_thresholds(y_true, y_cont, init_thr=(0.75, 1.5, 2.5, 3.5), n_iters=4):
    y_true = np.asarray(y_true, dtype=int).reshape(-1)
    y_cont = np.asarray(y_cont, dtype=np.float64).reshape(-1)

    thr = np.array(init_thr, dtype=np.float64)

    for _ in range(n_iters):
        for j in range(4):
            lo = 0.0 if j == 0 else thr[j - 1] + 1e-3
            hi = 4.5 if j == 3 else thr[j + 1] - 1e-3
            if hi <= lo:
                continue

            window = 0.6
            a = max(lo, thr[j] - window)
            b = min(hi, thr[j] + window)
            if b <= a:
                continue

            candidates = np.linspace(a, b, 41)
            best_t = thr[j]
            best_k = -1e9

            for t in candidates:
                thr_try = thr.copy()
                thr_try[j] = t
                pred = _apply_thresholds(y_cont, thr_try)
                k = cohen_kappa_score(y_true, pred, weights="quadratic")
                if k > best_k:
                    best_k = k
                    best_t = t

            thr[j] = best_t

    return thr


calibrated_threshold = None
if (
    has_ckpt
    and train_labels is not None
    and train_df is not None
    and os.path.exists(os.path.join(DATA_DIR, "train_images"))
):
    TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")

    class TrainDataset(Dataset):
        def __init__(self, df, img_dir, transform=None):
            self.ids = df["id_code"].astype(str).values
            self.labels = df["diagnosis"].astype(int).values
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            id_code = self.ids[idx]
            y = int(self.labels[idx])
            image_name = os.path.join(self.img_dir, f"{id_code}.png")
            img = Image.open(image_name).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            return y, img

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    idx_all = np.arange(len(train_df))
    tr_idx, va_idx = next(skf.split(idx_all, train_labels))

    calib_df = train_df.iloc[va_idx].reset_index(drop=True)

    calib_ds = TrainDataset(calib_df, TRAIN_IMG_DIR, transform=transform)
    calib_dl = DataLoader(
        calib_ds,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    y_true_all = []
    y_cont_all = []

    t0 = time.time()
    with torch.inference_mode():
        for ys, imgs in calib_dl:
            imgs = imgs.to(device, non_blocking=True)
            out = net(imgs, final=True).squeeze(1)  # (B,)
            out = torch.nan_to_num(out, nan=0.0, posinf=4.5, neginf=0.0).clamp(0.0, 4.5)
            y_true_all.append(ys.numpy())
            y_cont_all.append(out.detach().cpu().numpy())

    y_true_all = np.concatenate(y_true_all, axis=0)
    y_cont_all = np.concatenate(y_cont_all, axis=0)

    try:
        thr_opt = _optimize_qwk_thresholds(
            y_true_all, y_cont_all, init_thr=threshold, n_iters=4
        )
        calibrated_threshold = thr_opt.tolist()

        base_k = cohen_kappa_score(
            y_true_all, _apply_thresholds(y_cont_all, threshold), weights="quadratic"
        )
        opt_k = cohen_kappa_score(
            y_true_all,
            _apply_thresholds(y_cont_all, calibrated_threshold),
            weights="quadratic",
        )
        threshold = calibrated_threshold  # update global used by regress2class
        print(f"Threshold calibration (OOF fold) done in {time.time()-t0:.1f}s")
        print("OOF QWK with default thresholds:", float(base_k))
        print("OOF QWK with calibrated thresholds:", float(opt_k))
        print("Using thresholds:", threshold)
    except Exception as e:
        print("Warning: threshold calibration failed; keeping defaults. Error:", e)
else:
    print(
        "Threshold calibration skipped (no checkpoint / train images missing). Using thresholds:",
        threshold,
    )


class PrototypeBaseline(nn.Module):
    def __init__(
        self, backbone_name="tf_efficientnet_b4_ns", num_classes=5, max_per_class=60
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=True, num_classes=0, global_pool="avg"
        )
        self.num_classes = num_classes
        self.max_per_class = max_per_class
        self.register_buffer(
            "prototypes", torch.zeros(num_classes, self.backbone.num_features)
        )
        self._fitted = False

    @torch.no_grad()
    def fit(self, train_df, img_dir, transform, device):
        df = train_df.copy()
        df["id_code"] = df["id_code"].astype(str)
        df["diagnosis"] = df["diagnosis"].astype(int)

        feats_by_class = [[] for _ in range(self.num_classes)]
        counts = [0] * self.num_classes

        class _DS(Dataset):
            def __init__(self, df, img_dir, transform):
                self.ids = df["id_code"].values
                self.y = df["diagnosis"].values
                self.img_dir = img_dir
                self.transform = transform

            def __len__(self):
                return len(self.ids)

            def __getitem__(self, idx):
                idc = self.ids[idx]
                y = int(self.y[idx])
                p = os.path.join(self.img_dir, f"{idc}.png")
                img = Image.open(p).convert("RGB")
                img = self.transform(img) if self.transform is not None else img
                return y, img

        df = df.sort_values("id_code").reset_index(drop=True)
        parts = []
        for c in range(self.num_classes):
            part = df[df["diagnosis"] == c].head(self.max_per_class)
            parts.append(part)
        sub = pd.concat(parts, axis=0).reset_index(drop=True)

        dl = DataLoader(
            _DS(sub, img_dir, transform),
            batch_size=16,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
        self.backbone.to(device)
        self.backbone.eval()

        sums = [None] * self.num_classes
        ns = [0] * self.num_classes

        for ys, imgs in dl:
            imgs = imgs.to(device, non_blocking=True)
            f = self.backbone(imgs)  # (B, F)
            f = torch.nan_to_num(f)
            for i in range(f.size(0)):
                c = int(ys[i].item())
                if sums[c] is None:
                    sums[c] = f[i].detach().float().clone()
                else:
                    sums[c] += f[i].detach().float()
                ns[c] += 1

        protos = []
        for c in range(self.num_classes):
            if ns[c] == 0:
                protos.append(torch.zeros(self.backbone.num_features, device=device))
            else:
                protos.append(sums[c] / float(ns[c]))
        self.prototypes = torch.stack(protos, dim=0).detach().to(device)
        self._fitted = True

    @torch.no_grad()
    def forward(self, x):
        f = self.backbone(x)
        f = torch.nan_to_num(f)
        f = F.normalize(f, dim=1)
        p = F.normalize(self.prototypes, dim=1)
        sim = f @ p.t()  # (B,5), higher is closer
        prob = F.softmax(sim, dim=1)
        score = (prob * torch.arange(5, device=prob.device, dtype=prob.dtype)).sum(
            dim=1, keepdim=True
        )
        score = (score / 4.0) * 4.5
        return sim, score


baseline_model = None
if (
    not has_ckpt
    and train_df is not None
    and os.path.exists(os.path.join(DATA_DIR, "train_images"))
):
    try:
        baseline_model = PrototypeBaseline(max_per_class=60).to(device)
        baseline_model.fit(
            train_df, os.path.join(DATA_DIR, "train_images"), transform, device=device
        )
        print("Fitted PrototypeBaseline for no-checkpoint fallback.")
    except Exception as e:
        baseline_model = None
        print(
            "Warning: PrototypeBaseline fitting failed; will revert to majority-class baseline. Error:",
            e,
        )




## === cell 5
class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform=None):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_code = self.ids[idx]
        image_name = os.path.join(self.img_dir, f"{id_code}.png")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return id_code, img


ds = TestDataset(test_ids, TEST_IMG_DIR, transform=transform)
dl = DataLoader(
    ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available()
)

submission_rows = []

with torch.inference_mode():
    for id_codes, imgs in dl:
        imgs = imgs.to(device, non_blocking=True)

        if has_ckpt:
            out1 = net(imgs, final=True)  # (B,1)
            imgs_flip = torch.flip(imgs, dims=[3])
            out2 = net(imgs_flip, final=True)

            out = 0.5 * (out1 + out2)  # (B,1)
            out = torch.nan_to_num(out, nan=0.0, posinf=4.5, neginf=0.0).clamp(0.0, 4.5)

            pred = regress2class(out.data.squeeze(1)).numpy().astype(int)
        else:
            if baseline_model is not None and getattr(baseline_model, "_fitted", False):
                _, score1 = baseline_model(imgs)  # (B,1) in [0,4.5]
                imgs_flip = torch.flip(imgs, dims=[3])
                _, score2 = baseline_model(imgs_flip)
                score = 0.5 * (score1 + score2)
                score = torch.nan_to_num(score, nan=0.0, posinf=4.5, neginf=0.0).clamp(
                    0.0, 4.5
                )
                pred = regress2class(score.squeeze(1)).numpy().astype(int)
            else:
                pred = np.full((len(id_codes),), train_majority_class, dtype=int)

        for i, idc in enumerate(id_codes):
            submission_rows.append([str(idc), int(pred[i])])



## === cell 6
df = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])

df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df = df.set_index("id_code").reindex(test_ids).reset_index()
if df["diagnosis"].isna().any():
    raise RuntimeError("Some test ids are missing predictions after reindexing.")

df["diagnosis"] = df["diagnosis"].astype(int)

if df.empty:
    raise RuntimeError(
        "Submission DataFrame is empty; inference did not produce any rows."
    )

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with", len(df), "rows.")
print("Checkpoint loaded:", has_ckpt)
print("Final thresholds used:", threshold)
print("Normalization mean/std used:", imagenet_mean, imagenet_std)
