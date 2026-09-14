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

0.9231115838857415

# 6. Current score

0.74529

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.02381) has done: 'I remove the hard dependency on external weight files (which aren’t present in your environment) by loading ImageNet pretrained weights from `timm` instead, so the notebook can run end-to-end and produce a valid `submission.csv`. I also fix the device/type mismatch by moving the model to the target device *after* loading weights, and ensuring inputs and model weights share the same dtype/device. Finally, I make submission generation robust by reading `test.csv` as a DataFrame (keeping `id_code` strings) and using a DataLoader so inference is stable and non-empty.'
- What this solution (achieved 0.08416) has done: 'Your low score is consistent with using an essentially untrained head (random classifier/regressor/ordinal/final layers) while only loading an ImageNet backbone, so predictions collapse and kappa stays near zero. To move the score toward your target with minimal logic change, I (1) load the official `train.csv` and run a short, deterministic fine-tuning pass on the existing `final_regressor` pathway using the model’s own `final=True` output, and (2) compute thresholds on a held-out validation split to maximize quadratic weighted kappa (same evaluation semantics, just calibrated cutpoints). This keeps your architecture and overall inference approach intact, but replaces fixed thresholds with data-driven ones and ensures the final head is trained instead of random. The script still runs end-to-end within the time limit and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.70702) has done: 'The timeout is dominated by repeatedly loading/transforming images and by re-instantiating/loading EfficientNet weights for each fold. I (1) remove the ineffective per-process Python dict cache and replace it with a deterministic on-disk preprocessed tensor cache (one-time cost, then reused across folds/full-train/test), (2) avoid recreating/loading the backbone per fold by deep-copying a single initialized model state, and (3) speed up dataloading without changing training by using a faster sampler (same shuffle semantics), persistent workers, and precomputed integer indexing to avoid pandas `.loc` overhead. All changes preserve the same transforms, model, losses, epochs, folds, and threshold tuning logic; only redundant work is eliminated.'
- What this solution (achieved 0.73724) has done: 'Your gap to the target is large (0.707 → 0.923), so we need a real but still minimal improvement without changing the model or training setup. The biggest likely issue is that you compute OOF continuous predictions using *fold-specific fine-tuned heads*, but you then train a *single final model* from the original base weights and use the OOF-tuned thresholds on that different model—this mismatch can cost a lot in QWK. I keep the exact same architecture/loss/epochs/folds, but additionally (1) calibrate thresholds on **full-train predictions from the final trained model** (same labels available; no leakage beyond calibration), and (2) make inference use the same batch size variable and ensure persistent_workers doesn’t error when num_workers=0. This should move the public score upward toward your target while keeping changes small and within the same evaluation semantics (continuous-to-ordinal thresholding for QWK).'
- What this solution (achieved 0.74681) has done: 'The timeout is dominated by repeated, CPU-heavy PIL transforms (especially `trim()` + `ImageChops`) executed for every epoch across 3 CV folds + full-data training + calibration, causing the same images to be decoded and preprocessed many times. To preserve identical training/evaluation semantics while cutting redundant work, I add a deterministic on-disk cache of the *post-transform tensors* keyed by image path + transform “mode” (train/eval) so each image is preprocessed once per mode and then reused across all epochs/folds. I also enable safe DataLoader settings that reduce overhead (`drop_last=True` for train loaders, `prefetch_factor`, persistent workers) while keeping shuffling/epochs/loss/model unchanged. These changes keep the core model and training loops identical, but remove the biggest repeated computation and disk decode cost so it can finish within 600 seconds.'
- What this solution (achieved 0.74529) has done: 'Your current pipeline is already end-to-end and the biggest gap to your target likely comes from a metric mismatch: you train with MSE on a bounded continuous target but threshold to discrete classes for QWK, and your threshold search is a coarse local search. I keep the same model, same training loops, same epochs/folds, and same inference semantics, but (1) replace the simple step search with a small coordinate-descent + ternary refinement that more reliably finds better cutpoints for QWK, and (2) ensure threshold ordering/valid range constraints during tuning so the optimizer doesn’t get stuck in poor regions. These are minimal, calibration-only changes that should increase QWK without altering architecture or training. The submission format and paths remain unchanged.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import time
import math
import copy
import hashlib
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold
import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_float32_matmul_precision("high")


def _seed_worker(worker_id: int):
    base_seed = 42
    s = base_seed + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


_threshold_t_cpu = torch.as_tensor(threshold, dtype=torch.float32).view(1, -1)


def regress2class(out):
    out_cpu = out.detach().view(-1).cpu()
    thr = _threshold_t_cpu.to(dtype=out_cpu.dtype)
    return (out_cpu.view(-1, 1) >= thr).sum(dim=1).to(torch.float32)


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
DATA_ROOT = "../input/aptos2019-blindness-detection"

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")

TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.isfile(TEST_CSV), f"Missing test.csv at {TEST_CSV}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test_images directory at {TEST_IMG_DIR}"
assert os.path.isfile(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.isdir(
    TRAIN_IMG_DIR
), f"Missing train_images directory at {TRAIN_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
assert {"id_code", "diagnosis"}.issubset(train_df.columns)
train_df["id_code"] = train_df["id_code"].astype(str)
train_df["diagnosis"] = train_df["diagnosis"].astype(int)

test_df = pd.read_csv(TEST_CSV)
assert "id_code" in test_df.columns
test_df["id_code"] = test_df["id_code"].astype(str)

input_size = 380

transform_train = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        photometric_distort(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)
transform_eval = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model()

preferred_weight_path = "../input/weights/B4_3stage_19epoch_finetune2.pkl"
candidate_paths = [
    preferred_weight_path,
    os.path.join(DATA_ROOT, "B4_3stage_19epoch_finetune2.pkl"),
]
candidate_paths += glob.glob(
    "../input/**/B4_3stage_19epoch_finetune2.pkl", recursive=True
)

weight_path = None
for p in candidate_paths:
    if os.path.isfile(p):
        weight_path = p
        break

preloaded_state = None
if weight_path is not None:
    print("Loading weights from:", weight_path)
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_state[nk] = v
        state = new_state
    preloaded_state = state

if preloaded_state is not None:
    missing, unexpected = net.load_state_dict(preloaded_state, strict=False)
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
else:
    print(
        "No provided weights found; using timm ImageNet pretrained backbone weights for tf_efficientnet_b4_ns."
    )
    pretrained_backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
    backbone_sd = pretrained_backbone.state_dict()
    net.backbone.load_state_dict(backbone_sd, strict=False)

net = net.to(device)

_base_model_state_cpu = {
    k: v.detach().cpu().clone() for k, v in net.state_dict().items()
}



## === cell 5
CACHE_ROOT = os.path.join("../kaggle/working", "aptos_cache_v1")
os.makedirs(CACHE_ROOT, exist_ok=True)


def _cache_key(path: str, mode: str) -> str:
    h = hashlib.sha1((mode + "::" + os.path.abspath(path)).encode("utf-8")).hexdigest()
    return h


def _cache_path(path: str, mode: str) -> str:
    return os.path.join(CACHE_ROOT, f"{_cache_key(path, mode)}.pt")


def _load_or_compute_tensor(image_path: str, mode: str, transform):
    cp = _cache_path(image_path, mode)
    if os.path.isfile(cp):
        return torch.load(cp, map_location="cpu")
    with Image.open(image_path) as im:
        img = im.convert("RGB")
        if transform is not None:
            img = transform(img)
    tmp = cp + f".tmp_{os.getpid()}_{random.randint(0, 1_000_000)}"
    torch.save(img, tmp)
    try:
        os.replace(tmp, cp)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
    return img


class TrainDataset(Dataset):
    def __init__(
        self, df, img_dir, transform=None, cache=None, mode="train", use_cache=True
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.cache = cache  # kept for API compatibility; unused
        self.mode = mode
        self.use_cache = use_cache
        self.ids = df["id_code"].to_numpy(dtype=object)
        self.y = df["diagnosis"].to_numpy(dtype=np.int64)

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, i):
        idx = self.ids[i]
        y = int(self.y[i])
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        if self.use_cache:
            img = _load_or_compute_tensor(image_name, self.mode, self.transform)
        else:
            with Image.open(image_name) as im:
                img = im.convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
        return img, y


@torch.inference_mode()
def predict_continuous(model, loader):
    ys = []
    preds = []
    model.eval()
    for imgs, y in loader:
        imgs = imgs.to(device, non_blocking=True)
        out = model(imgs, final=True).squeeze(1)
        preds.append(out.detach().float().cpu().numpy())
        ys.append(np.asarray(y, dtype=np.int64))
    return np.concatenate(ys, axis=0), np.concatenate(preds, axis=0)


def apply_thresholds(cont, thr):
    thr = np.asarray(thr, dtype=np.float64)
    pred = (cont.reshape(-1, 1) >= thr.reshape(1, -1)).sum(axis=1).astype(np.int64)
    return np.clip(pred, 0, 4)


def _enforce_threshold_constraints(thr, lo=0.0, hi=4.5, min_gap=1e-4):
    t = np.asarray(thr, dtype=np.float64).copy()
    t = np.clip(t, lo, hi)
    for i in range(1, len(t)):
        if t[i] <= t[i - 1] + min_gap:
            t[i] = t[i - 1] + min_gap
    if t[-1] > hi:
        shift = t[-1] - hi
        t = t - shift
        t = np.clip(t, lo, hi)
        for i in range(1, len(t)):
            if t[i] <= t[i - 1] + min_gap:
                t[i] = t[i - 1] + min_gap
        t = np.clip(t, lo, hi)
    return t


def tune_thresholds(y_true, cont_pred, init_thr=(0.75, 1.5, 2.5, 3.5), n_iter=25):
    y_true = np.asarray(y_true, dtype=np.int64)
    cont_pred = np.asarray(cont_pred, dtype=np.float64)

    thr = _enforce_threshold_constraints(np.array(init_thr, dtype=np.float64))
    best_thr = thr.copy()
    best = cohen_kappa_score(
        y_true, apply_thresholds(cont_pred, best_thr), weights="quadratic"
    )

    step = 0.5
    for _ in range(n_iter):
        improved_any = False
        for i in range(4):
            base = best_thr.copy()
            candidates = [base[i] - step, base[i], base[i] + step]
            local_best_thr = best_thr.copy()
            local_best = best
            for v in candidates:
                cand = best_thr.copy()
                cand[i] = v
                cand = _enforce_threshold_constraints(cand)
                score = cohen_kappa_score(
                    y_true, apply_thresholds(cont_pred, cand), weights="quadratic"
                )
                if score > local_best + 1e-12:
                    local_best = score
                    local_best_thr = cand
            if local_best > best + 1e-12:
                best = local_best
                best_thr = local_best_thr
                improved_any = True
        if not improved_any:
            step *= 0.5
            if step < 1e-3:
                break

    for i in range(4):
        span = 0.25
        lo = best_thr[i] - span
        hi = best_thr[i] + span
        for _ in range(18):
            m1 = lo + (hi - lo) / 3.0
            m2 = hi - (hi - lo) / 3.0

            cand1 = best_thr.copy()
            cand1[i] = m1
            cand1 = _enforce_threshold_constraints(cand1)
            s1 = cohen_kappa_score(
                y_true, apply_thresholds(cont_pred, cand1), weights="quadratic"
            )

            cand2 = best_thr.copy()
            cand2[i] = m2
            cand2 = _enforce_threshold_constraints(cand2)
            s2 = cohen_kappa_score(
                y_true, apply_thresholds(cont_pred, cand2), weights="quadratic"
            )

            if s1 < s2:
                lo = m1
            else:
                hi = m2

        cand = best_thr.copy()
        cand[i] = (lo + hi) / 2.0
        cand = _enforce_threshold_constraints(cand)
        score = cohen_kappa_score(
            y_true, apply_thresholds(cont_pred, cand), weights="quadratic"
        )
        if score > best + 1e-12:
            best = score
            best_thr = cand

    return best_thr.tolist(), float(best)




## === cell 6
def _suggest_num_workers():
    cpu = os.cpu_count() or 2
    return max(2, min(8, cpu))


NUM_WORKERS = _suggest_num_workers()
PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = NUM_WORKERS > 0
PREFETCH_FACTOR = 2 if NUM_WORKERS > 0 else None

_dl_gen = torch.Generator()
_dl_gen.manual_seed(42)


def _make_loaders_from_df(df_tr, df_va, batch_size=8):
    tr_ds = TrainDataset(
        df_tr,
        TRAIN_IMG_DIR,
        transform=transform_train,
        cache=None,
        mode="train",
        use_cache=True,
    )
    va_ds = TrainDataset(
        df_va,
        TRAIN_IMG_DIR,
        transform=transform_eval,
        cache=None,
        mode="eval",
        use_cache=True,
    )

    tr_loader = DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=_seed_worker,
        generator=_dl_gen,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR if PREFETCH_FACTOR is not None else None,
        drop_last=True,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=_seed_worker,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR if PREFETCH_FACTOR is not None else None,
    )
    return tr_loader, va_loader


def set_trainable_for_finetune(model):
    for p in model.parameters():
        p.requires_grad = False
    for m in [model.classifier, model.regressor, model.ordinal, model.final_regressor]:
        for p in m.parameters():
            p.requires_grad = True


def finetune_one_fold(model, tr_loader, va_loader, epochs=3, lr=1e-3):
    set_trainable_for_finetune(model)
    model.train()
    params = []
    params += list(model.classifier.parameters())
    params += list(model.regressor.parameters())
    params += list(model.ordinal.parameters())
    params += list(model.final_regressor.parameters())

    optimizer = optim.Adam(params, lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        running = 0.0
        n = 0
        for imgs, y in tr_loader:
            imgs = imgs.to(device, non_blocking=True)
            y = y.to(device, dtype=torch.float32).view(-1, 1)

            optimizer.zero_grad(set_to_none=True)
            out = model(imgs, final=True)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            running += float(loss.item()) * imgs.size(0)
            n += imgs.size(0)
        if epoch == epochs - 1:
            print(f"Fold train last-epoch MSE: {running / max(1, n):.5f}")

    y_va, cont_va = predict_continuous(model, va_loader)
    return y_va, cont_va


N_SPLITS = 3
EPOCHS = 3
BATCH_SIZE = 8

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
y_all = train_df["diagnosis"].values.astype(int)

oof_cont = np.zeros(len(train_df), dtype=np.float32)

start = time.time()
for fold, (tr_idx, va_idx) in enumerate(skf.split(np.zeros(len(train_df)), y_all), 1):
    df_tr = train_df.iloc[tr_idx].reset_index(drop=True)
    df_va = train_df.iloc[va_idx].reset_index(drop=True)

    tr_loader, va_loader = _make_loaders_from_df(df_tr, df_va, batch_size=BATCH_SIZE)

    fold_net = ThreeStage_Model().to(device)
    fold_net.load_state_dict(_base_model_state_cpu, strict=True)

    print(f"Fold {fold}/{N_SPLITS}: train={len(df_tr)} valid={len(df_va)}")
    y_va, cont_va = finetune_one_fold(
        fold_net, tr_loader, va_loader, epochs=EPOCHS, lr=1e-3
    )
    oof_cont[va_idx] = cont_va.astype(np.float32)

print("OOF fine-tuning seconds:", round(time.time() - start, 1))

tuned_thr_oof, oof_kappa = tune_thresholds(
    y_all, oof_cont, init_thr=threshold, n_iter=40
)
print("Old thresholds:", threshold)
print("Tuned thresholds (OOF):", tuned_thr_oof)
print("OOF QWK (tuned):", oof_kappa)

threshold = tuned_thr_oof
_threshold_t_cpu = torch.as_tensor(threshold, dtype=torch.float32).view(1, -1)

full_loader = DataLoader(
    TrainDataset(
        train_df,
        TRAIN_IMG_DIR,
        transform=transform_train,
        cache=None,
        mode="train",
        use_cache=True,
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    worker_init_fn=_seed_worker,
    generator=_dl_gen,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR if PREFETCH_FACTOR is not None else None,
    drop_last=True,
)

for p in net.parameters():
    p.requires_grad = False
for m in [net.classifier, net.regressor, net.ordinal, net.final_regressor]:
    for p in m.parameters():
        p.requires_grad = True

net.train()
optimizer = optim.Adam(
    list(net.classifier.parameters())
    + list(net.regressor.parameters())
    + list(net.ordinal.parameters())
    + list(net.final_regressor.parameters()),
    lr=1e-3,
)
loss_fn = nn.MSELoss()

start = time.time()
for epoch in range(EPOCHS):
    running = 0.0
    n = 0
    for imgs, y in full_loader:
        imgs = imgs.to(device, non_blocking=True)
        y = y.to(device, dtype=torch.float32).view(-1, 1)
        optimizer.zero_grad(set_to_none=True)
        out = net(imgs, final=True)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        running += float(loss.item()) * imgs.size(0)
        n += imgs.size(0)
    print(f"Full-data epoch {epoch+1}/{EPOCHS} train MSE: {running / max(1, n):.5f}")
print("Full-data fine-tuning seconds:", round(time.time() - start, 1))
net.eval()

calib_loader = DataLoader(
    TrainDataset(
        train_df,
        TRAIN_IMG_DIR,
        transform=transform_eval,
        cache=None,
        mode="eval",
        use_cache=True,
    ),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    worker_init_fn=_seed_worker,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR if PREFETCH_FACTOR is not None else None,
)
y_cal, cont_cal = predict_continuous(net, calib_loader)
tuned_thr_final, train_kappa_final = tune_thresholds(
    y_cal, cont_cal, init_thr=tuned_thr_oof, n_iter=40
)
print("Tuned thresholds (FINAL model on full train):", tuned_thr_final)
print("Train QWK (FINAL tuned):", train_kappa_final)
threshold = tuned_thr_final
_threshold_t_cpu = torch.as_tensor(threshold, dtype=torch.float32).view(1, -1)




## === cell 7
class TestDataset(Dataset):
    def __init__(
        self, df, img_dir, transform=None, cache=None, mode="eval", use_cache=True
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.cache = cache  # kept for API compatibility; unused
        self.mode = mode
        self.use_cache = use_cache
        self.ids = df["id_code"].to_numpy(dtype=object)

    def __len__(self):
        return self.ids.shape[0]

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        if self.use_cache:
            img = _load_or_compute_tensor(image_name, self.mode, self.transform)
        else:
            with Image.open(image_name) as im:
                img = im.convert("RGB")
                if self.transform is not None:
                    img = self.transform(img)
        return idx, img


test_ds = TestDataset(
    test_df,
    TEST_IMG_DIR,
    transform=transform_eval,
    cache=None,
    mode="eval",
    use_cache=True,
)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    worker_init_fn=_seed_worker,
    persistent_workers=PERSISTENT_WORKERS,
    prefetch_factor=PREFETCH_FACTOR if PREFETCH_FACTOR is not None else None,
)

submission = []
with torch.inference_mode():
    seen = 0
    for batch_i, (ids, imgs) in enumerate(test_loader):
        if batch_i % 10 == 0:
            print("Predicting batch", batch_i, "/", len(test_loader))
        imgs = imgs.to(device, non_blocking=True)

        r_out = net(imgs, final=True)
        preds = regress2class(r_out.data.squeeze(1))

        preds = preds.to(torch.int64).cpu().numpy().tolist()
        ids = [str(x) for x in ids]
        submission.extend(list(zip(ids, preds)))
        seen += len(ids)

assert seen == len(test_df), f"Predicted {seen} rows, expected {len(test_df)}"



## === cell 8
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])

df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

assert len(df) > 0, "Submission DataFrame is empty"
assert list(df.columns) == ["id_code", "diagnosis"], "Wrong submission columns"
assert len(df) == len(test_df), "Submission row count does not match test.csv"
assert df["diagnosis"].between(0, 4).all(), "Found predictions outside [0,4]"

df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df.shape)
print(df.head())
