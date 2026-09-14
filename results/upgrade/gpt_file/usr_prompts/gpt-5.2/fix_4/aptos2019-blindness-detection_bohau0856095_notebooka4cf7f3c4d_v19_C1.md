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

0.9053000843210448

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the failing `pip install` of a missing local timm wheel and instead rely on the already-installed `timm` package. Then I fix the missing weights path by automatically searching common Kaggle input locations for `D5_regre_50epoch.pkl` and, if it truly doesn’t exist, fall back to running the same model architecture with default (untrained) weights so the notebook still completes and writes a valid `submission.csv`. I also make the device selection safe (use CPU if CUDA isn’t available) and make inference robust (no-grad, deterministic ordering, proper dtype/shape) without changing the model logic or prediction thresholds. Finally, I ensure a submission CSV with exactly `id_code,diagnosis` is always produced.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the model is running with random weights (because `D5_regre_50epoch.pkl` is not found), so the minimal path toward the target is to (1) reliably locate the provided weight file inside the competition dataset tree and load it, and (2) ensure the test `id_code` order exactly matches `sample_submission.csv` (some Kaggle scorers can be sensitive to ordering/alignment issues). I keep the exact same model, transforms, and thresholds, only improving weight discovery/loading robustness and aligning output order to the official sample submission. This should move your score upward toward the target without changing the core logic. The script still always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with either (a) missing pretrained weights (random model) or (b) weights loading but not actually matching the model, leaving most parameters uninitialized. I make the smallest changes that (1) more reliably find the weight file anywhere under `/kaggle/input` or `/kaggle/data`, (2) load it in a way that handles common checkpoint formats (including nested `model`, `net`, etc.) while preserving the same architecture and thresholds, and (3) add a hard sanity check so we don’t silently submit random-weight predictions if the load mostly failed. This should move the score upward toward your target without changing the model, transforms, or post-processing logic. The script still always produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
from PIL import Image

from sklearn.metrics import (
    cohen_kappa_score,
)  # kept for parity with original environment
import timm

device = "cuda:0" if torch.cuda.is_available() else "cpu"

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.7, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
    return prediction




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




## === cell 3
TEST_CSV_CANDIDATES = [
    "../input/aptos2019-blindness-detection/test.csv",
    "/kaggle/input/aptos2019-blindness-detection/test.csv",
    "/kaggle/data/aptos2019-blindness-detection/test.csv",
    "/kaggle/data/test.csv",
    "/kaggle/input/test.csv",
]

SAMPLE_SUB_CANDIDATES = [
    "../input/aptos2019-blindness-detection/sample_submission.csv",
    "/kaggle/input/aptos2019-blindness-detection/sample_submission.csv",
    "/kaggle/data/aptos2019-blindness-detection/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
]

test_csv_path = None
for p in TEST_CSV_CANDIDATES:
    if os.path.exists(p):
        test_csv_path = p
        break
if test_csv_path is None:
    raise FileNotFoundError(f"Could not find test.csv. Tried: {TEST_CSV_CANDIDATES}")

sample_sub_path = None
for p in SAMPLE_SUB_CANDIDATES:
    if os.path.exists(p):
        sample_sub_path = p
        break
if sample_sub_path is None:
    raise FileNotFoundError(
        f"Could not find sample_submission.csv. Tried: {SAMPLE_SUB_CANDIDATES}"
    )

sample_df = pd.read_csv(sample_sub_path)
test_ids = sample_df["id_code"].astype(str).tolist()

input_size = 384

tranforms = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)


def _find_weight_file(filename="D5_regre_50epoch.pkl"):
    candidates = [
        f"../input/weights/{filename}",
        f"/kaggle/input/weights/{filename}",
        f"../input/aptos2019-blindness-detection/{filename}",
        f"/kaggle/input/aptos2019-blindness-detection/{filename}",
        f"/kaggle/data/aptos2019-blindness-detection/{filename}",
        f"/kaggle/data/{filename}",
        f"/kaggle/input/{filename}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    patterns = [
        f"../input/**/{filename}",
        f"/kaggle/input/**/{filename}",
        f"/kaggle/data/**/{filename}",
    ]
    hits_all = []
    for pat in patterns:
        hits_all.extend(glob.glob(pat, recursive=True))
    hits_all = [h for h in hits_all if os.path.isfile(h)]
    hits_all.sort(key=lambda x: (len(x), x))
    return hits_all[0] if hits_all else None


def _unwrap_state_dict(ckpt_obj):
    state = ckpt_obj
    if isinstance(state, dict):
        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
        ]:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    return state


weight_path = _find_weight_file("D5_regre_50epoch.pkl")

net = Regressor()

loaded_ok = False
if weight_path is not None:
    print("Loading weights from:", weight_path)
    ckpt = torch.load(weight_path, map_location="cpu")
    state = _unwrap_state_dict(ckpt)

    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if isinstance(nk, str) and nk.startswith("module."):
                nk = nk[len("module.") :]
            new_state[nk] = v
        state = new_state

        missing, unexpected = net.load_state_dict(state, strict=False)

        total_params = len(net.state_dict().keys())
        miss_ratio = len(missing) / max(1, total_params)
        print(
            f"State dict load report: total={total_params}, missing={len(missing)}, "
            f"unexpected={len(unexpected)}, missing_ratio={miss_ratio:.3f}"
        )
        loaded_ok = (
            miss_ratio < 0.20
        )  # tolerant of minor mismatches, but blocks near-empty loads
        if not loaded_ok:
            raise RuntimeError(
                "Checkpoint was found but did not match the model sufficiently "
                f"(missing_ratio={miss_ratio:.3f}). This would likely score ~0. "
                "Please ensure the correct D5_regre_50epoch.pkl is available."
            )
    else:
        raise RuntimeError("Loaded checkpoint is not a state_dict-like mapping.")
else:
    print(
        "WARNING: Pretrained weights file D5_regre_50epoch.pkl not found. "
        "Proceeding with randomly initialized weights to produce a valid submission.csv (score will be low)."
    )

net = net.to(device)
net.eval()

TEST_IMG_DIR_CANDIDATES = [
    os.path.join(os.path.dirname(test_csv_path), "test_images"),
    os.path.join(os.path.dirname(sample_sub_path), "test_images"),
    "../input/aptos2019-blindness-detection/test_images",
    "/kaggle/input/aptos2019-blindness-detection/test_images",
    "/kaggle/data/aptos2019-blindness-detection/test_images",
    "/kaggle/data/test_images",
    "/kaggle/input/test_images",
]
test_img_dir = None
for d in TEST_IMG_DIR_CANDIDATES:
    if os.path.isdir(d):
        test_img_dir = d
        break
if test_img_dir is None:
    raise FileNotFoundError(
        f"Could not find test_images directory. Tried: {TEST_IMG_DIR_CANDIDATES}"
    )

submission = []
with torch.no_grad():
    for idx_str in test_ids:
        image_name = os.path.join(test_img_dir, f"{idx_str}.png")
        if not os.path.exists(image_name):
            alt = os.path.join(test_img_dir, f"{idx_str}.jpg")
            if os.path.exists(alt):
                image_name = alt
            else:
                raise FileNotFoundError(
                    f"Missing image for id_code={idx_str}: {image_name}"
                )

        img = Image.open(image_name).convert("RGB")
        img = tranforms(img).unsqueeze(0).to(device)

        r_out = net(img)
        pred = regress2class(r_out.data.squeeze(1)).cpu()
        submission.append([idx_str, int(pred.item())])

submission = np.array(submission, dtype=object)



## === cell 4
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df = sample_df[["id_code"]].merge(df, on="id_code", how="left")
if df["diagnosis"].isna().any():
    df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

df.to_csv("submission.csv", index=False)

print(df.head())
print("Wrote submission.csv with shape:", df.shape)
