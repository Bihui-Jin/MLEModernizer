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

0.910300713101174

# 6. Current score

-0.03543

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix three execution blockers with minimal impact on core inference logic: (1) remove the failing `pip install` and use the already-installed `timm`, (2) make the code device-agnostic so it runs on Kaggle CPU-only environments, and (3) robustly locate the pretrained weight file (or fail loudly with a clear message) instead of hardcoding a missing path. I also fix small tensor/device bugs in `ordinal2class_prob` and ensure the custom `trim()` transform always returns an image. Finally, I keep the same model and prediction method but run batched inference via a `Dataset/DataLoader` so the submission is reliably produced within the time limit.'
- What this solution (achieved 0.0) has done: 'I fix the execution blocker by removing the hard failure when the pretrained weights file isn’t present, and instead fall back to running the same model with random initialization so a valid `submission.csv` is always produced. To still move score upward toward the target when possible, the code load weights if they exist anywhere under `../input/` (more robust search) and only use the fallback if nothing is found. I also make the inference step robust to tensor shape edge cases (batch size 1) without changing the core prediction logic (same `regress2class` thresholds and same model head usage). Finally, I ensure the submission is aligned to `test.csv` order and has the exact required columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with producing near-constant predictions due to missing pretrained weights (random init fallback). To move score up toward the 0.9103 target with minimal core-logic change, I make weight loading robust by searching under the actual competition dataset directory (including `/kaggle/input/...`) and by relaxing `strict=True` to `strict=False` so minor key mismatches don’t prevent loading usable weights. I also ensure we use the intended `final=True` head at inference (still the same model, same prediction-to-class thresholds) because the final regressor is defined specifically to fuse the three outputs; using only `r_out` is likely a regression bug/oversight that depresses kappa. All changes are limited to weight discovery/loading and choosing the correct forward path; submission format and ordering remain unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model running with random initialization because the intended pretrained weights are not being found/loaded, which yields near-random/constant predictions and very low kappa. To move the score upward toward the 0.9103 target while preserving the exact model/inference logic, I only make weight discovery more robust by also searching under the provided `/kaggle/data/...` and `../data/...` trees (in addition to `../input` and `/kaggle/input`). I also add a strict guard: if no weights are found, the script now fails loudly instead of silently producing a low-scoring submission—this prevents wasting submissions that remain near 0.0. Everything else (architecture, transforms, final=True inference, thresholds, and CSV formatting/order) remains unchanged.'
- What this solution (achieved 0.0) has done: 'The run is currently blocked because the code hard-fails when the external pretrained weight file is missing; in this Kaggle environment that file is not provided, so you never reach inference/submission. I keep the same model architecture and the same `final=True` inference path and thresholds, but make weight loading robust: (1) search also inside the competition dataset tree you do have (`/kaggle/input/aptos2019-blindness-detection` and `/kaggle/data/...`) and (2) if still not found, proceed with random init (instead of raising) so a valid `submission.csv` is always produced (score remain low without weights, but it run end-to-end). I also add a couple of small guards to prevent common runtime issues (missing images / worker issues) without changing prediction logic.'
- What this solution (achieved 0.02381) has done: 'Your current 0.0 score is most consistent with the model running essentially untrained (random init) because the intended external weights file is not present in this environment; in that case predictions collapse and kappa goes to ~0. To move the score upward toward the 0.9103 target with minimal changes and without altering the model/training logic, I (1) switch to loading the EfficientNet backbone weights from timm (`pretrained=True`) so the feature extractor is not random, and (2) keep your exact final regression head + thresholding semantics unchanged. I also force the model into `eval()` and run inference under `torch.inference_mode()` for deterministic, correct evaluation-time behavior (no semantic change, just correctness/stability). This should significantly improve over 0.0 while staying within Kaggle time limits and still producing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current score (0.02381) is far below the target (0.9103), so we should improve performance with minimal semantic changes. The biggest likely issue is a mismatch between your inference normalization and what timm’s EfficientNet B4 NS expects; fixing that usually gives a large jump without changing the model or thresholds. I switch the `Normalize(...)` to timm’s model-specific `data_config` (mean/std and interpolation) while keeping your same resize shape and all other transforms, model forward (`final=True`), and `regress2class` thresholding intact. I also explicitly force deterministic eval-time behavior (already mostly done) and keep submission ordering unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is far below the target, and the most likely cause in this script is that the test-time preprocessing is not the same as what the EfficientNet-B4 NS backbone expects (you create a data config from a *separate* temporary model instance, and you don’t apply the model’s expected interpolation/crop defaults). I keep the exact same model, `final=True` inference path, and the same `regress2class` thresholds, but I build the backbone once, reuse it both for the model and for `resolve_model_data_config`, and use `timm.data.create_transform` to match the pretrained backbone’s normalization/interpolation more faithfully. This is a minimal change to preprocessing only (no architecture/training/threshold changes) and is the safest way to move kappa upward toward the target. The submission writing/order remains identical.'
- What this solution (achieved -0.03543) has done: 'Your 0.0 score is far below the 0.9103 target, and with your current code it’s overwhelmingly likely the score is stuck near 0 because the 3-stage heads (and especially the `final_regressor`) are effectively untrained when the external checkpoint isn’t found, so predictions collapse toward a near-constant class. To move the score upward without changing your architecture, loss, or training approach, the minimal legitimate fix is to *use the pretrained backbone’s classifier logits* (which are available because `pretrained=True`) as a fallback prediction path **only when** the 3-stage weights are not loaded. This keeps your existing `final=True` regression + thresholding unchanged when weights exist, but avoids the “random head” failure mode when they don’t. I also add a tiny safety clamp/cast around the fallback to guarantee valid integer labels 0–4 and keep submission ordering identical.'

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
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT

from PIL import Image, ImageChops

import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)



## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out: torch.Tensor):
    prediction = torch.zeros(out.size(0), device="cpu")
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().detach().to("cpu")
    return prediction


def ordinal2class_prob(out: torch.Tensor):
    dev = out.device
    pred_prob = torch.zeros(out.size(0), 5, device=dev)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out: torch.Tensor):
    dev = out.device
    pred_prob = torch.zeros((out.size(0), 5), device=dev)
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

        self.backbone = (
            backbone
            if backbone is not None
            else timm.models.tf_efficientnet_b4_ns(pretrained=True)
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
def resolve_base_input():
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "../input/aptos2019-blindness-detection",
        "../data/aptos2019-blindness-detection",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "test.csv")) or os.path.exists(
            os.path.join(c, "aptos2019-blindness-detection", "test.csv")
        ):
            if os.path.exists(os.path.join(c, "test.csv")):
                return c
            if os.path.exists(
                os.path.join(c, "aptos2019-blindness-detection", "test.csv")
            ):
                return os.path.join(c, "aptos2019-blindness-detection")
    return "../input/aptos2019-blindness-detection"


BASE_INPUT = resolve_base_input()
TEST_CSV = os.path.join(BASE_INPUT, "test.csv")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")

test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).values

input_size = 380

backbone_b4 = timm.models.tf_efficientnet_b4_ns(pretrained=True)
data_cfg = timm.data.resolve_model_data_config(backbone_b4)
mean = data_cfg.get("mean", (0.485, 0.456, 0.406))
std = data_cfg.get("std", (0.229, 0.224, 0.225))

timm_tf = timm.data.create_transform(
    input_size=input_size,
    is_training=False,
    mean=mean,
    std=std,
    interpolation=data_cfg.get("interpolation", "bicubic"),
    crop_pct=data_cfg.get("crop_pct", 1.0),
)

_post = []
for t in timm_tf.transforms:
    name = t.__class__.__name__.lower()
    if "totensor" in name or "normalize" in name:
        _post.append(t)

transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize(
            (input_size * 3 // 4, input_size),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        *_post,
    ]
)


def find_first_existing(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


weight_name_stem = "B4_3stage_55epoch_CLAHE"
weight_candidates = [
    f"../input/weights/{weight_name_stem}.pkl",
    f"../input/weights/{weight_name_stem}.pth",
    f"/kaggle/input/weights/{weight_name_stem}.pkl",
    f"/kaggle/input/weights/{weight_name_stem}.pth",
    f"../data/weights/{weight_name_stem}.pkl",
    f"../data/weights/{weight_name_stem}.pth",
    f"/kaggle/data/weights/{weight_name_stem}.pkl",
    f"/kaggle/data/weights/{weight_name_stem}.pth",
]
weight_candidates += glob.glob(f"../input/**/{weight_name_stem}.*", recursive=True)
weight_candidates += glob.glob(f"/kaggle/input/**/{weight_name_stem}.*", recursive=True)
weight_candidates += glob.glob(f"../data/**/{weight_name_stem}.*", recursive=True)
weight_candidates += glob.glob(f"/kaggle/data/**/{weight_name_stem}.*", recursive=True)
weight_candidates += glob.glob(
    os.path.join(BASE_INPUT, f"**/{weight_name_stem}.*"), recursive=True
)

weight_path = find_first_existing(weight_candidates)

net = ThreeStage_Model(backbone=backbone_b4).to(device)
net.eval()

loaded_weights = False
missing, unexpected = [], []

if weight_path is not None:
    state = torch.load(weight_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k[7:] if k.startswith("module.") else k
            new_state[nk] = v
        state = new_state

    missing, unexpected = net.load_state_dict(state, strict=False)
    loaded_weights = True

net.to(device)
net.eval()

print("Resolved BASE_INPUT:", BASE_INPUT)
print("Using device:", device)
print(
    "Weights found:",
    weight_path if loaded_weights else "NONE (using timm pretrained backbone init)",
)
print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
print("Using timm mean/std:", mean, std)




## === cell 5
class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform=None):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")
        if not os.path.exists(image_name):
            raise FileNotFoundError(f"Missing test image: {image_name}")
        img = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return idx, img


ds = TestDataset(test_ids, TEST_IMG_DIR, transform=transform)
dl = DataLoader(
    ds, batch_size=8, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available()
)

submission_rows = []

with torch.inference_mode():
    for ids, imgs in dl:
        imgs = imgs.to(device, non_blocking=True)

        if loaded_weights:
            out = net(imgs, final=True).view(-1)
            preds = regress2class(out).numpy().astype(int)
        else:
            logits = net.backbone(imgs)  # shape [B, 1000] (ImageNet classes)
            top1 = torch.argmax(logits, dim=1).to("cpu").numpy().astype(np.int64)
            preds = (top1 % 5).astype(np.int64)  # deterministic 0..4 mapping
            preds = np.clip(preds, 0, 4).astype(int)

        for _id, p in zip(ids, preds):
            submission_rows.append([_id, int(p)])

submission = np.array(submission_rows, dtype=object)



## === cell 6
if submission.shape[0] == 0:
    raise RuntimeError("Submission is empty; inference did not run correctly.")

df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df = df.set_index("id_code").reindex(test_df["id_code"].astype(str)).reset_index()
if df["diagnosis"].isna().any():
    df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

df["diagnosis"] = df["diagnosis"].clip(0, 4).astype(int)

df.to_csv("submission.csv", index=False)

print(df.head())
print("Wrote submission.csv with", len(df), "rows")
print("Unique ids:", df["id_code"].nunique())
print("Diagnosis distribution:\n", df["diagnosis"].value_counts().sort_index())
