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

0.908239205248382

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the notebook robust to the Kaggle CPU-only environment by selecting `cuda` only if it’s available, and by loading model weights only when the file exists (otherwise it still run end-to-end with random weights and create a valid CSV). I also fix a couple of logic/runtime pitfalls: `trim()` sometimes returns `None` (breaking transforms), `ordinal2class_prob()` mixes `.cuda()` and `.cpu()` incorrectly, and `photometric_distort` uses `is` for string comparison. Finally, I ensure the submission is always non-empty and aligned with `test.csv`, and write `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is almost certainly because the code is running with random weights (the weights path points to a non-existent `../input/weights/...`), so the smallest change that legitimately moves the score toward the target is to load the actual competition checkpoint if it exists under the provided dataset directory. I keep the exact model and inference logic, but make the weight-loading robust by (1) searching a few likely locations under `../input/aptos2019-blindness-detection` and `../input` for a `.pkl/.pth` file, and (2) handling common checkpoint formats (`state_dict`, `model`, `net`) and `module.` prefixes. I also switch inference to use `final=True` if (and only if) the checkpoint contains `final_regressor` weights; otherwise it keep the existing `(_, r_out, _)` path—this preserves core semantics while allowing the intended head to be used when available. The rest (transforms, thresholds, submission alignment) stays the same and it still always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the notebook running inference from random weights (no checkpoint found/loaded), so the smallest legitimate change to move toward the target is to make weight discovery/loading actually succeed in this environment. I keep the exact model/inference logic, but (1) broaden the weight-file search to include common Kaggle locations like `/kaggle/input` and the provided `/kaggle/data/...` mirror, and (2) make checkpoint loading robust to common formats (`state_dict` nesting and `module.` prefixes) while still using `strict=False`. If no weights are found, behavior remains identical (valid CSV, low score), but if the intended checkpoint exists anywhere under the available inputs, it be used and the kappa should increase substantially toward your target. Submission writing and row alignment remain unchanged.'

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    """
    out: shape (B,) or (B,1) tensor on any device.
    returns: float tensor (B,) on CPU with values in {0,1,2,3,4}
    """
    if out.dim() == 2 and out.size(1) == 1:
        out = out.squeeze(1)
    prediction = torch.zeros(out.size(0), dtype=torch.float32)
    out_cpu = out.detach().float().cpu()
    for i in range(4):
        prediction += (out_cpu >= threshold[i]).float()
    return prediction


def ordinal2class_prob(out):
    """
    out: shape (B,4) tensor with sigmoid outputs in [0,1], on any device.
    returns: (B,5) probability tensor on same device.
    """
    dev = out.device
    pred_prob = torch.zeros(out.size(0), 5, device=dev, dtype=out.dtype)
    pred_prob[:, 0] = 1 - out[:, 0]
    pred_prob[:, 1] = out[:, 0] * (1 - out[:, 1])
    pred_prob[:, 2] = out[:, 1] * (1 - out[:, 2])
    pred_prob[:, 3] = out[:, 2] * (1 - out[:, 3])
    pred_prob[:, 4] = out[:, 3]
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    """
    out: shape (B,) tensor in [0,4.5] on any device
    returns: (B,5) probability tensor on same device
    """
    if out.dim() == 2 and out.size(1) == 1:
        out = out.squeeze(1)
    dev = out.device
    pred_prob = torch.zeros((out.size(0), 5), device=dev, dtype=out.dtype)

    out_clamped = out.detach()
    for i in range(out.size(0)):
        v = float(out_clamped[i].item())
        if v < 4.0:
            l1 = int(math.floor(v))
            l2 = int(math.ceil(v))
            pred_prob[i, l1] = 1 - (v - l1)
            pred_prob[i, l2] = 1 - (l2 - v)
        else:
            pred_prob[i, 4] = 1.0
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
DATA_DIR = "../input/aptos2019-blindness-detection"
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).tolist()

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

net = ThreeStage_Model()


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if all(isinstance(k, str) and k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model", "net", "model_state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def _find_weight_file():
    candidates = [
        "../input/weights/B4_3stage_32epoch_CLAHE.pkl",
        os.path.join(DATA_DIR, "B4_3stage_32epoch_CLAHE.pkl"),
        os.path.join(DATA_DIR, "weights", "B4_3stage_32epoch_CLAHE.pkl"),
        "/kaggle/input/weights/B4_3stage_32epoch_CLAHE.pkl",
        "/kaggle/input/aptos2019-blindness-detection/B4_3stage_32epoch_CLAHE.pkl",
        "/kaggle/input/aptos2019-blindness-detection/weights/B4_3stage_32epoch_CLAHE.pkl",
        "/kaggle/data/weights/B4_3stage_32epoch_CLAHE.pkl",
        "/kaggle/data/aptos2019-blindness-detection/B4_3stage_32epoch_CLAHE.pkl",
        "/kaggle/data/aptos2019-blindness-detection/weights/B4_3stage_32epoch_CLAHE.pkl",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    search_roots = [
        "../input",
        DATA_DIR,
        "/kaggle/input",
        "/kaggle/data",
    ]
    exts = (".pkl", ".pth", ".pt")
    hits = []
    for root in search_roots:
        if not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            rel_depth = os.path.relpath(dirpath, root).count(os.sep)
            if rel_depth > 5:
                dirnames[:] = []
                continue
            for fn in filenames:
                if fn.endswith(exts) and (
                    ("B4" in fn) or ("3stage" in fn.lower()) or ("clahe" in fn.lower())
                ):
                    hits.append(os.path.join(dirpath, fn))
    if hits:
        hits.sort()
        return hits[0]
    return None


WEIGHTS_PATH = _find_weight_file()
use_final_head = False

if WEIGHTS_PATH is not None:
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
    state = _strip_module_prefix(_extract_state_dict(ckpt))
    missing, unexpected = net.load_state_dict(state, strict=False)

    has_final = (
        any(k.startswith("final_regressor.") for k in state.keys())
        if isinstance(state, dict)
        else False
    )
    use_final_head = bool(has_final)

    print(f"Loaded weights from: {WEIGHTS_PATH}")
    print(
        f"use_final_head={use_final_head} | missing={len(missing)} unexpected={len(unexpected)}"
    )
else:
    print(
        "WARNING: No weights found. Running with random weights (submission will be low score)."
    )

net = net.to(device)
net.eval()



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

            if use_final_head:
                r_out = net(img_t, final=True)
            else:
                _, r_out, _ = net(img_t)

            pred = regress2class(r_out)  # returns CPU tensor
            diagnosis = int(pred.item())
        except Exception as e:
            print(f"WARNING: failed on {idx}: {e}. Using fallback prediction 0.")
            diagnosis = 0
        submission_rows.append([idx, diagnosis])

submission = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])

submission = submission.merge(test_df[["id_code"]], on="id_code", how="right")
submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int)



## === cell 6
assert submission.shape[0] == len(
    test_df
), "Submission row count mismatch with test.csv"
assert list(submission.columns) == [
    "id_code",
    "diagnosis",
], "Submission columns mismatch"
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
