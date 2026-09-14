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

0.9211626749752142

# 6. Current score

-0.15854

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing-weights crash by making the script robust to absent pretrained files: it search common `/kaggle/input/**/weights` locations, and if none are found it still run end-to-end with randomly initialized weights (score be poor, but you get a valid CSV). I also fix the CUDA hardcoding by selecting `cuda` only when available, otherwise running on CPU to avoid the “no NVIDIA driver” runtime error. I correct a couple of small logic/runtime issues in the transforms (`is` vs `==`, and `trim` returning `None`), and I make inference batched via a proper `Dataset/DataLoader` so it finishes within the timeout. Finally, I ensure `submission.csv` is always written with the exact required columns and row count matching `test.csv`.'
- What this solution (achieved -0.03395) has done: 'I fix the device mismatch crash by ensuring the model is moved to the selected `device` *after* any backbone replacement (the current code replaces `net.backbone` with a CPU model, leaving weights on CPU while inputs are on CUDA). I also add a safe CPU fallback if a similar mismatch still occurs at runtime so the notebook always completes and writes `submission.csv`. These changes are score-neutral except that they allow inference to run; when the checkpoint is present it use it as intended (which should move the score up toward the target). Finally, I make cell 6 robust by guaranteeing `submission` exists before saving.'
- What this solution (achieved -0.12665) has done: 'Your current score is far below the target, and given your notes it’s likely because the intended checkpoint isn’t being found/loaded (so predictions are effectively random or mismatched). I make the smallest change that increases the chance of loading the correct weights by (1) expanding checkpoint discovery beyond only a `weights/` subfolder and (2) selecting the “best match” checkpoint by filename keyword instead of picking the first arbitrary file. I also make thresholding robust by auto-tuning the 4 cutpoints to match the training label distribution (only when no checkpoint is loaded), which usually boosts QWK versus fixed thresholds while preserving the same regression-to-class core logic. Everything else (model, transforms, inference loop, submission schema) stays the same and it still always write a valid `submission.csv`.'
- What this solution (achieved -0.12665) has done: 'Your score is far below the target, so the most likely issue is that the intended fine-tuned checkpoint still isn’t being loaded correctly (or is being loaded but not applied due to key mismatches). I make minimal, score-relevant changes to (1) improve checkpoint discovery ranking for this specific competition (prefer “aptos”, “blindness”, “b4/b5”, “3stage”, etc.), and (2) enforce a stricter, safer checkpoint load that detects when essentially nothing matched and falls back cleanly (instead of silently running near-random). If we do fall back (no usable checkpoint), I switch the discrete prediction mapping to a distribution-matching quantile binning on train labels (still regression→class, but more robust than fixed cutpoints) to lift QWK from negative toward reasonable values. Everything else (model definitions, transforms, inference loop, submission schema) stays the same and it still always write a valid `submission.csv`.'
- What this solution (achieved -0.15854) has done: 'Your score is far below target and the code already falls back to pretrained ImageNet weights when the fine-tuned checkpoint can’t be loaded, so the main low-risk way to move QWK upward is to (1) make sure we’re actually using the strongest available pretrained backbone variant (B4 *ns*) consistently, and (2) replace the crude “threshold based on label priors × 4.5” fallback with a distribution-matching quantile binning computed from the model’s own test predictions (still regression→class via thresholds, same core logic). I keep the model/forward/inference loop intact and only adjust the fallback threshold calibration path when no checkpoint is loaded. I also make checkpoint matching slightly more permissive (but still safe) so a compatible fine-tuned file is less likely to be rejected, which should substantially raise the score if any real checkpoint exists. The script still always write a valid `submission.csv` with the required columns and row count.'

# 9. Code solution

## === cell 0
import os
import glob
import random
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

import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor) -> torch.Tensor:
    """
    out: shape (B,) or (B,1) on any device.
    returns: shape (B,) on CPU, dtype float
    """
    if out.ndim == 2 and out.size(1) == 1:
        out = out.squeeze(1)
    prediction = torch.zeros(out.size(0), device=out.device, dtype=torch.float32)
    for i in range(4):
        prediction += (out >= threshold[i]).to(torch.float32)
    return prediction.detach().cpu()


def ordinal2class_prob(out: torch.Tensor) -> torch.Tensor:
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out: torch.Tensor) -> torch.Tensor:
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)
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
TEST_CSV = "../input/aptos2019-blindness-detection/test.csv"
TEST_IMG_DIR = "../input/aptos2019-blindness-detection/test_images"
TRAIN_CSV = "../input/aptos2019-blindness-detection/train.csv"

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

train_df = pd.read_csv(TRAIN_CSV)

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


def find_checkpoint(preferred_path: str):
    """
    Change (score-relevant): slightly broaden checkpoint discovery so we more often
    find the intended fine-tuned weights (big score lift vs random/ImageNet-only).
    """
    if os.path.isfile(preferred_path):
        return preferred_path

    preferred_base = os.path.basename(preferred_path).lower()
    preferred_stem = os.path.splitext(preferred_base)[0]

    candidates = []
    for pat in [
        "../input/**/*.pkl",
        "../input/**/*.pth",
        "../input/**/*.pt",
        "../input/**/*.ckpt",
        "/kaggle/input/**/*.pkl",
        "/kaggle/input/**/*.pth",
        "/kaggle/input/**/*.pt",
        "/kaggle/input/**/*.ckpt",
    ]:
        candidates.extend(glob.glob(pat, recursive=True))

    def looks_like_model_file(p: str) -> bool:
        b = os.path.basename(p).lower()
        if any(
            x in b for x in ["submission", "sample_submission", "train.csv", "test.csv"]
        ):
            return False
        if b.endswith((".pt", ".pth", ".pkl", ".ckpt")):
            return True
        return False

    candidates = [c for c in candidates if looks_like_model_file(c)]
    if not candidates:
        return None

    def score_path(p: str) -> tuple:
        b = os.path.basename(p).lower()
        pl = p.lower()
        s = 0

        if b == preferred_base:
            s += 2000
        if preferred_stem in b:
            s += 600

        for kw, w in [
            ("aptos", 120),
            ("blind", 120),
            ("retina", 60),
            ("diabet", 60),
            ("dr", 40),
            ("b4", 80),
            ("b5", 60),
            ("efficientnet", 20),
            ("3stage", 140),
            ("three", 40),
            ("final", 40),
            ("finetune", 60),
            ("best", 50),
            ("epoch", 10),
        ]:
            if kw in b or kw in pl:
                s += w

        if "weights" in pl or "checkpoint" in pl or "ckpt" in pl:
            s += 15

        return (-s, len(p))

    candidates_sorted = sorted(candidates, key=score_path)
    return candidates_sorted[0]


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _clean_state_dict(sd):
    if not isinstance(sd, dict):
        return sd
    cleaned = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        cleaned[nk] = v
    return cleaned


def _load_checkpoint_safely(model: nn.Module, ckpt_path: str) -> tuple[bool, dict]:
    """
    Change (score-relevant): make the compatibility check a bit less strict.
    Some checkpoints may only store part of the model (e.g., backbone+head), and
    rejecting them can force ImageNet-only fallback (very low QWK).
    """
    state = torch.load(ckpt_path, map_location="cpu")
    sd = _clean_state_dict(_extract_state_dict(state))
    if not isinstance(sd, dict) or len(sd) == 0:
        return False, {}

    load_res = model.load_state_dict(sd, strict=False)

    model_keys = set(model.state_dict().keys())
    sd_keys = set(sd.keys())
    matched = len(model_keys & sd_keys)
    match_ratio = matched / max(1, len(model_keys))

    loaded_ok = match_ratio >= 0.10
    info = {
        "matched": matched,
        "total_model_keys": len(model_keys),
        "match_ratio": match_ratio,
        "missing_keys_n": len(load_res.missing_keys),
        "unexpected_keys_n": len(load_res.unexpected_keys),
    }
    return loaded_ok, sd


def compute_quantile_thresholds_from_outputs(
    outputs_1d: np.ndarray, train_labels: np.ndarray
) -> list[float]:
    """
    Change (score-relevant): when we don't have a fine-tuned checkpoint, map the regressor
    outputs to classes by matching the TRAIN label distribution via quantiles of TEST outputs.
    This preserves the same regression->threshold->class core logic but is typically much
    better calibrated than fixed cutpoints for QWK.
    """
    outputs_1d = np.asarray(outputs_1d, dtype=np.float64)
    outputs_1d = np.clip(outputs_1d, 0.0, 4.5)

    counts = pd.Series(train_labels).value_counts().sort_index()
    probs = (counts / counts.sum()).reindex([0, 1, 2, 3, 4]).fillna(0).values
    cum = np.cumsum(probs)
    q = np.clip(cum[:4], 1e-6, 1 - 1e-6)

    thr = np.quantile(outputs_1d, q).astype(np.float64)

    eps = 1e-6
    for i in range(1, 4):
        if thr[i] <= thr[i - 1]:
            thr[i] = thr[i - 1] + eps

    thr = np.clip(thr, 0.0, 4.49)
    return [float(t) for t in thr]


net = ThreeStage_Model()
ckpt_path = find_checkpoint("../input/weights/B4_3stage_15epoch_finetune.pkl")

loaded_checkpoint = False
use_final_head = False

if ckpt_path is not None:
    loaded_checkpoint, sd = _load_checkpoint_safely(net, ckpt_path)
    if loaded_checkpoint:
        use_final_head = any(k.startswith("final_regressor.") for k in sd.keys())
    else:
        ckpt_path = None  # treat as absent if not meaningfully compatible

if ckpt_path is None:
    net.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
    net.backbone.global_pool = GeM(flatten=True)
    loaded_checkpoint = False
    use_final_head = False

net = net.to(device)
net.eval()

print(
    f"device={device} ckpt_found={ckpt_path is not None} ckpt_path={ckpt_path} ckpt_loaded={loaded_checkpoint} use_final_head={use_final_head} threshold={threshold}"
)



## === cell 5
ds = TestDataset(test_ids, TEST_IMG_DIR, transform=transform)
dl = DataLoader(
    ds, batch_size=8, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available()
)

submission_rows = []


def _run_inference(current_device: torch.device, return_raw: bool = False):
    rows = []
    raw = []
    net.to(current_device)
    net.eval()
    with torch.no_grad():
        for id_codes, imgs in dl:
            imgs = imgs.to(current_device, non_blocking=True)
            if use_final_head:
                out = net(imgs, final=True)
                out_1d = out.squeeze(1)
            else:
                _, r_out, _ = net(imgs)
                out_1d = r_out.squeeze(1)

            if return_raw:
                raw.append(out_1d.detach().float().cpu().numpy())

            preds = regress2class(out_1d)  # CPU tensor float
            preds = preds.numpy().astype(np.int64)
            for i, idc in enumerate(id_codes):
                rows.append([str(idc), int(preds[i])])
    if return_raw:
        return rows, np.concatenate(raw, axis=0)
    return rows


try:
    if not loaded_checkpoint:
        _, raw_outputs = _run_inference(device, return_raw=True)
        threshold = compute_quantile_thresholds_from_outputs(
            outputs_1d=raw_outputs,
            train_labels=train_df["diagnosis"].values,
        )
        print(f"Fallback quantile thresholds (no checkpoint): {threshold}")

    submission_rows = _run_inference(device, return_raw=False)

except RuntimeError as e:
    msg = str(e).lower()
    if ("input type" in msg and "weight type" in msg) or (
        "expected all tensors" in msg and "same device" in msg
    ):
        device = torch.device("cpu")
        if not loaded_checkpoint:
            _, raw_outputs = _run_inference(device, return_raw=True)
            threshold = compute_quantile_thresholds_from_outputs(
                outputs_1d=raw_outputs,
                train_labels=train_df["diagnosis"].values,
            )
            print(f"Fallback quantile thresholds (no checkpoint, CPU): {threshold}")

        submission_rows = _run_inference(device, return_raw=False)
    else:
        raise

submission = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])

if len(submission) != len(test_df):
    submission = test_df.merge(submission, on="id_code", how="left")
    submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int)
else:
    submission["diagnosis"] = submission["diagnosis"].astype(int)

submission = submission[["id_code", "diagnosis"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print(f"Saved submission.csv with shape={submission.shape} on device={device}")
print(
    "Pred label distribution:",
    submission["diagnosis"].value_counts(dropna=False).sort_index().to_dict(),
)
