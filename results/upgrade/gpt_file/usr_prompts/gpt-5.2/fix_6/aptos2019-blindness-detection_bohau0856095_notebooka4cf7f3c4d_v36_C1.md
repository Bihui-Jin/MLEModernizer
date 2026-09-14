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

0.9027226348454676

# 6. Current score

-0.12665

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix three execution blockers: (1) remove the missing `weights` dependency by selecting a weight file only if it exists (otherwise fall back to an untrained model so a submission is still produced), (2) make the script run on CPU when no GPU driver is available, and (3) fix preprocessing bugs that can yield `None` images (the `trim()` transform) and a Python `is` string-comparison issue. I also make inference use `torch.no_grad()` and batch prediction via a `Dataset/DataLoader` to finish reliably within the time limit while preserving the same model forward/thresholding logic. Finally, I ensure a non-empty `submission.csv` with exactly the required columns and row alignment to `test.csv`.'
- What this solution (achieved -0.03395) has done: 'Your 0.0 score is consistent with using an untrained model (your log already shows weights often don’t load), so the smallest legitimate step toward the 0.90 target is to ensure pretrained ImageNet backbone weights are used when the competition weights aren’t available. I keep your architecture and inference logic identical, but switch `pretrained=False` to `pretrained=True` (timm) as a fallback so the backbone produces meaningful features even without your external weight file. I also make the weight loading `strict=False` to tolerate minor key mismatches (common with timm versions) while still loading whatever matches, which is usually better than silently running random weights. Everything else (thresholds, transforms, regress2class mapping, submission alignment/format) remains unchanged.'
- What this solution (achieved 0.02381) has done: 'Your current score suggests the model is still effectively untrained for this competition; the biggest minimal win is to actually use the better “final” head you already defined, because right now inference ignores it and uses only the regression branch (discarding classifier + ordinal signals). I keep the same architecture, transforms, and thresholding logic, but change inference to call `net(imgs, final=True)` and then apply `regress2class` on that output. I also set deterministic flags to reduce run-to-run noise (helps move score reliably upward toward the target without changing semantics). Everything else (paths, submission alignment, CSV schema) stays the same.'
- What this solution (achieved -0.03395) has done: 'Your score (0.02381) is far below the target (0.9027), so the most likely blocker is that the “final” head is being used without actually having the competition-trained weights for `final_regressor` (your log indicates weights often don’t load), making predictions close to random. To move the score upward with minimal semantic change, I keep the same model and transforms but (1) prefer loading a local `train_images`-trained checkpoint if present under the provided dataset tree, and (2) if no checkpoint is found, fall back to using the non-final regression head (which is at least compatible with ImageNet-pretrained backbone features) instead of the untrained final head. I also make `num_workers` safe on Kaggle CPU-only environments to avoid intermittent dataloader hangs/timeouts (stability helps score reproducibility). The submission format, thresholds, and preprocessing stay unchanged.'
- What this solution (achieved -0.12665) has done: 'Your current score is far below the target, so we should make the smallest change that most plausibly increases QWK without changing your model/training logic: tune the fixed regression-to-class thresholds using the training labels distribution (a common minimal boost for this competition). I keep your exact model forward pass, weights loading behavior, transforms, and `regress2class` mapping—only the threshold values are recalibrated from `train.csv` quantiles to better match class balance. If competition weights are found we still use the final head; otherwise we keep your existing fallback behavior. This should move the score upward toward the target while staying within Kaggle constraints and producing the same submission format.'

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

import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
BASE = "../input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE, "train.csv")


def _compute_thresholds_from_train(train_csv_path: str):
    df = pd.read_csv(train_csv_path)
    y = df["diagnosis"].astype(int).values
    counts = np.bincount(y, minlength=5).astype(float)
    probs = counts / counts.sum()
    cum = np.cumsum(probs)  # probabilities for classes 0..k
    cut_quantiles = np.clip(cum[:4], 1e-3, 1 - 1e-3)
    thr = (cut_quantiles * 4.5).tolist()
    for i in range(1, 4):
        if thr[i] <= thr[i - 1]:
            thr[i] = thr[i - 1] + 1e-3
    return thr


threshold = [0.75, 1.5, 2.5, 3.5]
if os.path.exists(TRAIN_CSV):
    threshold = _compute_thresholds_from_train(TRAIN_CSV)
print(f"Using thresholds: {threshold}")


def regress2class(out: torch.Tensor) -> torch.Tensor:
    """
    out: shape (B,) or (B,1), continuous regression in [0,4.5]
    returns: shape (B,) integer class in {0..4} on CPU
    """
    if out.ndim == 2 and out.size(1) == 1:
        out = out.squeeze(1)
    prediction = torch.zeros(out.size(0), dtype=torch.long, device="cpu")
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().detach().cpu().long()
    return prediction


def ordinal2class_prob(out: torch.Tensor) -> torch.Tensor:
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out: torch.Tensor) -> torch.Tensor:
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
        super().__init__()
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
        super().__init__()

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
TEST_CSV = os.path.join(BASE, "test.csv")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

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

net = ThreeStage_Model()

candidate_weight_paths = [
    "../input/weights/B4_3stage_43epoch_CLAHE.pkl",
    "../input/weights/B4_3stage_43epoch_CLAHE.pth",
    os.path.join(BASE, "B4_3stage_43epoch_CLAHE.pkl"),
    os.path.join(BASE, "B4_3stage_43epoch_CLAHE.pth"),
    os.path.join(
        "../input", "aptos2019-blindness-detection", "B4_3stage_43epoch_CLAHE.pkl"
    ),
    os.path.join(
        "../input", "aptos2019-blindness-detection", "B4_3stage_43epoch_CLAHE.pth"
    ),
]

loaded = False
loaded_path = None
for wp in candidate_weight_paths:
    if os.path.exists(wp):
        state = torch.load(wp, map_location="cpu")
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
        missing, unexpected = net.load_state_dict(state, strict=False)
        loaded = True
        loaded_path = wp
        print(f"Loaded weights from: {wp}")
        print(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        break

net = net.to(device)
net.eval()

print(f"Device: {device}")
print(f"Loaded competition weights: {loaded}")
if loaded_path is not None:
    print(f"Checkpoint: {loaded_path}")




## === cell 5
class AptosTestDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_code = self.ids[idx]
        image_path = os.path.join(self.img_dir, f"{id_code}.png")
        img = Image.open(image_path).convert("RGB")
        img = self.transform(img)
        return id_code, img


batch_size = 8 if device.type == "cuda" else 4
num_workers = 2 if device.type == "cuda" else 0

ds = AptosTestDataset(test_ids, TEST_IMG_DIR, transform)
dl = DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
)

all_ids = []
all_preds = []

use_final_head = bool(loaded)

with torch.no_grad():
    for ids, imgs in dl:
        imgs = imgs.to(device, non_blocking=True)

        if use_final_head:
            out = net(imgs, final=True)  # (B,1) in [0,4.5]
        else:
            _, out, _ = net(imgs, final=False)  # r_out already in [0,4.5]

        pred = regress2class(out.squeeze(1))

        all_ids.extend(list(ids))
        all_preds.extend(pred.numpy().tolist())

submission_df = pd.DataFrame(
    {"id_code": all_ids, "diagnosis": np.array(all_preds, dtype=int)}
)

submission_df = submission_df.set_index("id_code").loc[test_df["id_code"]].reset_index()



## === cell 6
out_path = "submission.csv"
submission_df.to_csv(out_path, index=False)

print(submission_df.head())
print(f"Wrote {out_path} with shape {submission_df.shape}")
print(f"Columns: {list(submission_df.columns)}")
assert (
    submission_df.shape[0] == test_df.shape[0]
), "Submission row count must match test.csv"
assert list(submission_df.columns) == [
    "id_code",
    "diagnosis",
], "Submission columns must be exactly: id_code, diagnosis"
assert out_path.endswith(".csv")
