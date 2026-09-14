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

0.9042101245565856

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing-weights failure by loading the model weights only if they exist; if they don’t, the code fall back to a deterministic lightweight baseline that still produces a valid submission. I also fix the CUDA crash by selecting CPU automatically when no GPU is available, and ensure tensor/device code does not hardcode `.cuda()`. Finally, I fix transform bugs (`is` vs `==`, and `trim()` sometimes returning `None`) so inference runs end-to-end and always writes a non-empty `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is because the model almost certainly isn’t loading weights in this environment, so it falls back to predicting all zeros. To move the score toward your 0.904 target with minimal changes and without changing the model/training logic, I (1) point the input path to the actual available dataset location here, and (2) add a robust weights search that looks both in `../input/weights` and inside the competition dataset folder; if weights still don’t exist, we still generate a valid submission (but you need to add weights as a Kaggle Dataset for real improvement). I also make the `load_state_dict` slightly more tolerant (`strict=False`) to avoid silent failures due to minor key mismatches, which can be the difference between “no model” and “working model” while keeping inference semantics the same. Finally, I keep the exact prediction path the same (use regressor head + `regress2class`) and ensure submission ordering matches `test.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from the “no weights found → predict all zeros” fallback, which is valid-format but performs terribly. To move toward the 0.904 target with minimal changes and without altering your model/inference semantics, I (1) expand the dataset and weights path discovery to match the actual `/kaggle/data/...` layout you have, and (2) add a recursive search for the expected weight filename under common roots so the model actually loads when present. I also make test image path resolution robust (some environments nest `test_images/test_images`) so inference doesn’t silently fail or read the wrong directory. If weights truly aren’t available in the environment, the code still produce a valid `submission.csv`, but your score remain near 0 until weights are provided.'

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
import cv2  # kept to preserve original environment assumptions

from sklearn.metrics import cohen_kappa_score  # kept (not used in inference)
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor):
    prediction = torch.zeros(out.size(0), device=out.device)
    for i in range(4):
        prediction += (out >= threshold[i]).squeeze()
    return prediction.detach().cpu()


def ordinal2class_prob(out: torch.Tensor):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out: torch.Tensor):
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
DATA_DIR_CANDIDATES = [
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "../kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data/input/aptos2019-blindness-detection",
]
DATA_DIR = next((p for p in DATA_DIR_CANDIDATES if os.path.exists(p)), None)
if DATA_DIR is None:
    raise FileNotFoundError(
        f"Could not find aptos dataset in any of: {DATA_DIR_CANDIDATES}"
    )

TEST_CSV = os.path.join(DATA_DIR, "test.csv")

TEST_IMG_DIR_CANDIDATES = [
    os.path.join(DATA_DIR, "test_images"),
    os.path.join(DATA_DIR, "test_images", "test_images"),
]
TEST_IMG_DIR = next((p for p in TEST_IMG_DIR_CANDIDATES if os.path.isdir(p)), None)
if TEST_IMG_DIR is None:
    raise FileNotFoundError(f"Could not find test_images in: {TEST_IMG_DIR_CANDIDATES}")

test_ids_df = pd.read_csv(TEST_CSV)
test_ids = np.squeeze(test_ids_df.values)

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

EXPECTED_WEIGHT_BASENAMES = [
    "B4_3stage_31epoch_CLAHE.pkl",
    "B4_3stage_31epoch_CLAHE.pth",
    "B4_3stage_31epoch_CLAHE.pt",
]

WEIGHTS_CANDIDATES = [
    "../input/weights/B4_3stage_31epoch_CLAHE.pkl",
    "../input/weights/B4_3stage_31epoch_CLAHE.pth",
    "../input/weights/B4_3stage_31epoch_CLAHE.pt",
    os.path.join(DATA_DIR, "B4_3stage_31epoch_CLAHE.pkl"),
    os.path.join(DATA_DIR, "B4_3stage_31epoch_CLAHE.pth"),
    os.path.join(DATA_DIR, "B4_3stage_31epoch_CLAHE.pt"),
    os.path.join(DATA_DIR, "weights", "B4_3stage_31epoch_CLAHE.pkl"),
    os.path.join(DATA_DIR, "weights", "B4_3stage_31epoch_CLAHE.pth"),
    os.path.join(DATA_DIR, "weights", "B4_3stage_31epoch_CLAHE.pt"),
]


def find_weight_recursively(search_roots, basenames):
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            for bn in basenames:
                if bn in filenames:
                    return os.path.join(dirpath, bn)
    return None


weights_path = next((p for p in WEIGHTS_CANDIDATES if os.path.exists(p)), None)
if weights_path is None:
    weights_path = find_weight_recursively(
        search_roots=[
            "../input",
            "/kaggle/input",
            "/kaggle/data",
            DATA_DIR,
        ],
        basenames=EXPECTED_WEIGHT_BASENAMES,
    )

net = ThreeStage_Model().to(device)
net.eval()

model_loaded = False
if weights_path is not None:
    try:
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if isinstance(state, dict) and any(
            k.startswith("module.") for k in state.keys()
        ):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}

        missing, unexpected = net.load_state_dict(state, strict=False)
        net = net.to(device)
        net.eval()
        model_loaded = True
        print(f"Loaded model weights from: {weights_path}")
        if len(missing) > 0 or len(unexpected) > 0:
            print(
                f"State dict loaded with missing={len(missing)} unexpected={len(unexpected)}"
            )
    except Exception as e:
        print(f"Failed to load weights from {weights_path}: {e}")
        model_loaded = False
else:
    print(
        "No pretrained weights found. Will use deterministic baseline predictions (all zeros)."
    )



## === cell 5
submission = []

if not model_loaded:
    for idx in test_ids:
        submission.append([idx, 0])
else:
    with torch.inference_mode():
        for i, idx in enumerate(test_ids):
            if i % 50 == 0:
                print(i)
            image_name = os.path.join(TEST_IMG_DIR, f"{idx}.png")
            img = Image.open(image_name).convert("RGB")
            img = transform(img).unsqueeze(0).to(device)

            _, r_out, _ = net(img)
            pred = regress2class(r_out.squeeze(1))
            submission.append([idx, int(pred.item())])

submission = np.array(submission, dtype=object)



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df = df.set_index("id_code").loc[test_ids_df["id_code"].astype(str)].reset_index()

out_path = "submission.csv"
df.to_csv(out_path, index=False)

print(df.head())
print(f"Wrote {len(df)} rows to {out_path}")
assert out_path.endswith(".csv") and os.path.exists(out_path) and len(df) > 0
assert list(df.columns) == ["id_code", "diagnosis"]
assert len(df) == len(test_ids_df)
