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

0.9144714453611804

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the notebook robust to Kaggle’s environment by (1) removing the failing custom `pip install` and instead using the already-installed `timm`, (2) automatically selecting `cuda` only if available (otherwise CPU) to fix the “no NVIDIA driver” crash, and (3) handling the missing `../input/weights/...pkl` by falling back to generating a valid (non-empty) submission using the competition’s required format. I also fix small but real logic issues in transforms (`is` vs `==`, and `trim()` returning `None`) that can break preprocessing. These changes keep the core model/inference logic intact when weights exist, while guaranteeing an end-to-end run that always writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is caused by never loading real model weights (the code falls back to predicting all zeros), so the smallest meaningful improvement is to correctly locate and load weights if they exist inside the provided competition dataset tree. I keep the exact same model and inference logic, but add a robust search for `B4_3stage_1epoch_finetune.pkl` under `../input/` and (if present) load it; otherwise we still emit a valid submission. I also ensure the submission row order exactly matches `test.csv` by building predictions keyed by `id_code` and then reassembling in that order, preventing rare misalignment issues that can tank kappa. These changes are minimal and directly aimed at moving the score up toward your 0.914 target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from the “all zeros” fallback because no weights are being found/loaded, so the smallest real improvement is to reliably load the provided pretrained weights (if present) and make `load_state_dict` robust to common checkpoint formats (`state_dict`, `model`, `net`). To move kappa upward without changing model logic, I also make inference faster/cleaner (batching + `torch.inference_mode`) while keeping identical transforms and prediction mapping. Finally, I keep the submission strictly aligned to `test.csv` order and ensure every `id_code` gets a prediction, avoiding accidental missing keys that can invalidate or damage the score. These changes preserve the same architecture, thresholds, and evaluation semantics—just ensure you’re actually using the trained model when weights exist.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly from the model not loading real weights and therefore outputting the all-zeros fallback (which yields near-random/very poor kappa). To move the score up toward your 0.914 target with minimal changes, I (1) make weight discovery robust to common Kaggle layouts (including the current working directory), (2) make checkpoint loading handle common key-prefix issues (e.g., `module.` and `backbone.`) without changing the model, and (3) switch inference to the model’s `final=True` head (the one trained to fuse the three heads) while keeping your exact discretization thresholds and submission alignment. If weights are still not found, it still produce a valid `submission.csv` (all zeros) exactly as before.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score indicates the model is still effectively predicting a constant class (either because weights aren’t being found/loaded, or because the loaded checkpoint keys don’t match the model). I make the smallest change that increases the chance of successfully loading real weights: broaden the checkpoint discovery to include any `.pkl/.pth/.pt` in common Kaggle locations and robustly extract/normalize state_dict keys (including `backbone.`-prefixed keys) without changing the model. If weights still can’t be loaded, the script behave exactly as before (valid all-zeros submission), but if they can, predictions should move substantially upward toward your 0.914 target. I also keep submission order strictly aligned to `test.csv` as you already do.'

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
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)



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




## === cell 4
TEST_CSV = "../input/aptos2019-blindness-detection/test.csv"
TEST_IMG_DIR = "../input/aptos2019-blindness-detection/test_images"
SAMPLE_SUB = "../input/aptos2019-blindness-detection/sample_submission.csv"

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

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

WEIGHTS_BASENAME = "B4_3stage_1epoch_finetune.pkl"

CANDIDATE_PATHS = [
    "./" + WEIGHTS_BASENAME,
    "../input/weights/" + WEIGHTS_BASENAME,
    "../input/aptos2019-blindness-detection/weights/" + WEIGHTS_BASENAME,
    "/kaggle/input/weights/" + WEIGHTS_BASENAME,
    "/kaggle/input/aptos2019-blindness-detection/weights/" + WEIGHTS_BASENAME,
    "/kaggle/working/" + WEIGHTS_BASENAME,
]

found_weights_path = None
for p in CANDIDATE_PATHS:
    if os.path.exists(p):
        found_weights_path = p
        break

if found_weights_path is None:
    plausible = []
    search_roots = ["../input", "/kaggle/input", ".", "/kaggle/working"]
    exts = (".pkl", ".pth", ".pt", ".bin")
    keywords = (
        "b4",
        "efficientnet",
        "3stage",
        "three",
        "stage",
        "finetune",
        "blindness",
        "aptos",
    )
    for sr in search_roots:
        if not os.path.exists(sr):
            continue
        for root, _, files in os.walk(sr):
            for fn in files:
                lfn = fn.lower()
                if lfn.endswith(exts) and any(k in lfn for k in keywords):
                    plausible.append(os.path.join(root, fn))
            if len(plausible) >= 30:
                break
        if len(plausible) >= 30:
            break

    for p in plausible:
        if os.path.basename(p) == WEIGHTS_BASENAME:
            found_weights_path = p
            break
    if found_weights_path is None and plausible:
        plausible = sorted(plausible, key=lambda x: os.path.getsize(x), reverse=True)
        found_weights_path = plausible[0]


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("state_dict", "model_state_dict", "model", "net", "weights"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _normalize_state_dict_keys(state):
    if not isinstance(state, dict):
        return state

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    for pref in ("model.", "net."):
        if any(k.startswith(pref) for k in state.keys()):
            state = {
                k.replace(pref, "", 1) if k.startswith(pref) else k: v
                for k, v in state.items()
            }

    model_keys = set(net.state_dict().keys())
    if any(k.startswith("backbone.") for k in state.keys()):
        stripped = {
            k.replace("backbone.", "", 1) if k.startswith("backbone.") else k: v
            for k, v in state.items()
        }
        overlap_old = sum(1 for k in state.keys() if k in model_keys)
        overlap_new = sum(1 for k in stripped.keys() if k in model_keys)
        if overlap_new > overlap_old:
            state = stripped

    return state


weights_loaded = False
if found_weights_path is not None and os.path.exists(found_weights_path):
    try:
        ckpt = torch.load(found_weights_path, map_location=device)
        state = _extract_state_dict(ckpt)
        state = _normalize_state_dict_keys(state)
        try:
            net.load_state_dict(state, strict=True)
            weights_loaded = True
        except RuntimeError:
            missing, unexpected = net.load_state_dict(state, strict=False)
            weights_loaded = True
            print(
                "Loaded with strict=False. missing_keys=",
                len(missing),
                "unexpected_keys=",
                len(unexpected),
            )
    except Exception as e:
        print("Failed to load checkpoint from:", found_weights_path)
        print("Exception:", repr(e))
        weights_loaded = False

print("device =", device)
print("weights_loaded =", weights_loaded, "found_weights_path =", found_weights_path)




## === cell 5
class TestDS(Dataset):
    def __init__(self, ids, img_dir, tfm):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.tfm = tfm

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = f"{self.img_dir}/{idx}.png"
        img = Image.open(image_name).convert("RGB")
        img = self.tfm(img)
        return idx, img


pred_by_id = {}

if weights_loaded:
    ds = TestDS(test_ids, TEST_IMG_DIR, transform)
    batch_size = 8 if device.startswith("cuda") else 2
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=device.startswith("cuda"),
    )

    with torch.inference_mode():
        seen = 0
        for batch_ids, batch_x in dl:
            batch_x = batch_x.to(device, non_blocking=True)

            out = net(batch_x, final=True)  # (B, 1) in [0, 4.5]
            out = out.data.squeeze(1)

            pred = regress2class(out)  # preserves original discretization thresholds
            pred = pred.to(torch.int64).cpu().numpy().tolist()

            for _id, _p in zip(batch_ids, pred):
                pred_by_id[str(_id)] = int(_p)
            seen += len(batch_ids)
            if seen % 50 == 0:
                print(seen)
else:
    sample = pd.read_csv(SAMPLE_SUB)
    for rid in sample["id_code"].astype(str).values:
        pred_by_id[str(rid)] = 0

missing_ids = [idx for idx in test_ids if str(idx) not in pred_by_id]
if missing_ids:
    for idx in missing_ids:
        pred_by_id[str(idx)] = 0
    print("Warning: filled missing predictions for", len(missing_ids), "ids")

submission_rows = [[idx, pred_by_id[str(idx)]] for idx in test_ids]
submission = np.array(submission_rows, dtype=object)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

if df.empty:
    raise RuntimeError(
        "Submission DataFrame is empty; check test.csv/sample_submission.csv reading."
    )

if len(df) != len(test_df):
    raise RuntimeError(f"Row count mismatch: submission={len(df)} test={len(test_df)}")
if not (
    df["id_code"].astype(str).values == test_df["id_code"].astype(str).values
).all():
    raise RuntimeError(
        "id_code order mismatch vs test.csv; refusing to write invalid submission."
    )

df.to_csv("submission.csv", index=False)

print(df.head())
print("Wrote submission.csv with", len(df), "rows. weights_loaded =", weights_loaded)
