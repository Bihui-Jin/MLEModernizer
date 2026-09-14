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

0.9118531059271592

# 6. Current score

0.01221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) remove the failing `pip install` of an unavailable wheel and use the already-installed `timm`, (2) make device selection robust by falling back to CPU when no GPU/driver is available, and (3) fix the weight-loading failure by searching common input locations and safely running even if weights are missing. I also fix a couple of small logic bugs that can break inference (string comparison with `is`, and `trim()` returning `None`), and make the test-image path resolution robust to the dataset’s duplicate folder layouts. Finally, I ensure a non-empty `submission.csv` is always written with the exact required columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with the model running with randomly initialized weights (your log even warns the weight file is missing), which yields essentially random predictions. The smallest change that should move you strongly toward the target is to correctly locate and load the pretrained checkpoint from the competition dataset folder (it’s typically inside the input dataset tree, not at the input root). I expand the weight search to include the competition directory and common nested locations, and also robustly handle checkpoints saved as `{"state_dict": ...}` to avoid silent partial loads. This preserves your exact model/transform/inference logic and only fixes the missing-weights issue so performance improves toward the target.'
- What this solution (achieved 0.01221) has done: 'I fix the immediate failure by removing the hard stop when the checkpoint file isn’t found, so inference can run end-to-end and always write a valid `submission.csv`. To keep the core model/inference logic intact, I only add a robust fallback to EfficientNet’s built-in ImageNet pretrained weights when your competition checkpoint is missing; this is score-improving versus random weights while staying within the same architecture and forward path. I also make test-id extraction and submission building robust (ensure we iterate the `id_code` column, preserve order, and avoid shape issues that caused the row-mismatch assertion). Finally, I keep paths unchanged but make the directory discovery a bit more tolerant to the duplicated Kaggle folder layout.'

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
BASE_INPUT_CANDIDATES = [
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "../input",
    "/kaggle/input",
    "/kaggle/data/input/aptos2019-blindness-detection",
    "/kaggle/data/input",
]


def first_existing(path_list):
    for p in path_list:
        if p is not None and os.path.exists(p):
            return p
    return None


base_comp = first_existing(BASE_INPUT_CANDIDATES)
if base_comp is None:
    raise FileNotFoundError(
        "Could not locate Kaggle input directory for competition data."
    )

test_csv_candidates = [
    os.path.join(base_comp, "test.csv"),
    os.path.join(base_comp, "aptos2019-blindness-detection", "test.csv"),
]
test_csv_path = first_existing(test_csv_candidates)
if test_csv_path is None:
    raise FileNotFoundError("test.csv not found in expected locations.")

test_ids_df = pd.read_csv(test_csv_path)
if "id_code" not in test_ids_df.columns:
    raise KeyError(
        f"Expected column 'id_code' in test.csv, found: {list(test_ids_df.columns)}"
    )
test_ids = test_ids_df["id_code"].astype(str).tolist()

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

weight_name = "B4_3stage_3epoch_finetune.pkl"

weight_search_roots = [
    base_comp,
    os.path.join(base_comp, "aptos2019-blindness-detection"),
    "../input",
    "/kaggle/input",
    "/kaggle/data/input",
    "../input/weights",
    "/kaggle/input/weights",
    "/kaggle/data/input/weights",
]
weight_path = None
for root in weight_search_roots:
    cand = os.path.join(root, weight_name)
    if os.path.exists(cand):
        weight_path = cand
        break

if weight_path is None:
    hits = []
    for r in ["../input", "/kaggle/input", "/kaggle/data/input", base_comp]:
        if os.path.exists(r):
            hits.extend(glob.glob(os.path.join(r, "**", weight_name), recursive=True))
    hits = sorted(set(hits))
    if len(hits) > 0:
        weight_path = hits[0]


def _extract_state_dict(obj):
    if not isinstance(obj, dict):
        return obj
    for k in ["state_dict", "model_state_dict", "model", "net", "weights"]:
        if k in obj and isinstance(obj[k], dict):
            return obj[k]
    for _, v in obj.items():
        if isinstance(v, dict):
            for kk in ["state_dict", "model_state_dict", "model"]:
                if kk in v and isinstance(v[kk], dict):
                    return v[kk]
    return obj


def _strip_prefix_if_present(state, prefix):
    if not isinstance(state, dict):
        return state
    if any(k.startswith(prefix) for k in state.keys()):
        return {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    return state


loaded_ok = False
if weight_path is not None:
    _before = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}

    state = torch.load(weight_path, map_location="cpu")
    state = _extract_state_dict(state)

    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    state = _strip_prefix_if_present(state, "model.")
    state = _strip_prefix_if_present(state, "net.")

    missing, unexpected = net.load_state_dict(state, strict=False)

    after = net.state_dict()
    changed = 0
    for k in _before.keys():
        if k in after and not torch.equal(_before[k], after[k].detach().cpu()):
            changed += 1
            break
    loaded_ok = changed > 0

    print(f"Loaded weights from: {weight_path}")
    if len(missing) > 0 or len(unexpected) > 0:
        print(
            f"NOTE: load_state_dict strict=False; missing={len(missing)} unexpected={len(unexpected)}"
        )
    if not loaded_ok:
        print(
            "WARNING: Checkpoint found but did not appear to modify weights; continuing with fallback backbone pretrained."
        )
        loaded_ok = False

if not loaded_ok:
    net.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
    net.backbone.global_pool = GeM(flatten=True)
    print(
        "WARNING: Competition checkpoint not found/loaded; using ImageNet-pretrained EfficientNet-B4 backbone as fallback."
    )

net = net.to(device)
net.eval()

test_img_dir_candidates = [
    os.path.join(os.path.dirname(test_csv_path), "test_images"),
    os.path.join(base_comp, "test_images"),
    os.path.join(base_comp, "aptos2019-blindness-detection", "test_images"),
]
test_img_dir = first_existing(test_img_dir_candidates)
if test_img_dir is None:
    raise FileNotFoundError("test_images directory not found in expected locations.")



## === cell 5
submission_rows = []
with torch.inference_mode():
    for i, idx in enumerate(test_ids):
        if i % 50 == 0:
            print(i)

        image_name = os.path.join(test_img_dir, f"{idx}.png")
        if not os.path.exists(image_name):
            alt = os.path.join(test_img_dir, "test_images", f"{idx}.png")
            if os.path.exists(alt):
                image_name = alt
            else:
                raise FileNotFoundError(f"Missing test image: {image_name}")

        img = Image.open(image_name).convert("RGB")
        img = transform(img).unsqueeze(0).to(device)

        _, r_out, _ = net(img)
        pred = regress2class(r_out.data.squeeze(1))
        submission_rows.append([idx, int(pred.item())])

df = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

assert len(df) == len(
    test_ids
), f"Submission rows mismatch with test set: {len(df)} vs {len(test_ids)}"

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with", len(df), "rows")
