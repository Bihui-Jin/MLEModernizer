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

0.916054033611703

# 6. Current score

-0.00991

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I (1) remove the failing `pip install` cell and make the script use the already-installed `timm`, (2) make device selection robust so it runs on CPU when no GPU is available, and (3) fix the missing weight-file path by searching common Kaggle input locations and falling back safely if the weights aren’t present. I also fix a couple of runtime issues in transforms/utilities (`is` vs `==`, `trim()` returning `None`) that can silently break inference. Finally, I ensure we always create a non-empty `submission.csv` with the exact required columns and order aligned to `test.csv`.'
- What this solution (achieved 0.00014) has done: 'Your 0.0 score is almost certainly coming from using randomly initialized weights because the expected `.pkl` weight file is not actually present in your Kaggle inputs, so the model outputs near-constant/garbage predictions. I keep your exact model and inference logic, but make the weight loading robust to common checkpoint formats (`state_dict` wrappers, `module.` prefixes) and point the search to the actual competition dataset directory you already have. If the intended weight still isn’t found, I deterministically fall back to using `timm` pretrained backbone weights (without changing architecture) so predictions become meaningful and the kappa should move upward toward your target band. The submission writing/alignment be kept identical, just made slightly safer against missing/corrupt images.'
- What this solution (achieved -0.00991) has done: 'Your current score is extremely low because inference is effectively using untrained/random heads (and possibly a missing checkpoint), plus the prediction post-processing is not well aligned to the kappa metric (fixed thresholds without any calibration). I keep your exact model and inference flow, but (1) make checkpoint discovery also look in `/kaggle/data/...` (where your dataset actually is) and support common key-prefix mismatches, and (2) add a tiny in-notebook threshold calibration step using a train/val split to choose the 4 regression thresholds that maximize quadratic weighted kappa on a held-out fold. This does not change the model architecture, loss, or training loop (there is no training), it only improves the mapping from your regressor output to the 0–4 labels, which is directly what the metric evaluates. The rest of the submission writing/alignment stays the same, ensuring a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import math
import random
import glob
import time
import copy
import json
import pickle
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT

from PIL import Image, ImageChops, ImageEnhance, ImageOps
import cv2

from sklearn.metrics import cohen_kappa_score
import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
from torch import Tensor
from torch.jit.annotations import List, Optional, Tuple


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


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed
    """

    def __init__(self, num_features: int, eps: float = 1e-5, n: Optional[int] = None):
        if n is not None:
            warnings.warn(
                "`n` argument is deprecated and has been renamed `num_features`",
                DeprecationWarning,
            )
            num_features = n
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.weight.shape[0]}, eps={self.eps})"


class Linear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if torch.jit.is_scripting():
            bias = self.bias.to(dtype=input.dtype) if self.bias is not None else None
            return F.linear(input, self.weight.to(dtype=input.dtype), bias=bias)
        return F.linear(input, self.weight, self.bias)


class backboneNet_efficient(nn.Module):
    def __init__(self):
        super().__init__()
        net = timm.create_model("tf_efficientnet_b4_ns", pretrained=False)
        layers_to_train = ["blocks"]
        for name, parameter in net.named_parameters():
            if all([not name.startswith(layer) for layer in layers_to_train]):
                parameter.requires_grad_(False)

        self.num_features = 1792
        self.conv_stem = net.conv_stem
        self.bn1 = net.bn1
        self.act1 = net.act1
        self.block0 = net.blocks[0]
        self.block1 = net.blocks[1]
        self.block2 = net.blocks[2]
        self.block3 = net.blocks[3]
        self.block4 = net.blocks[4]
        self.block5 = net.blocks[5]
        self.block6 = net.blocks[6]
        self.conv_head = net.conv_head
        self.bn2 = net.bn2
        self.act2 = net.act2
        self.global_pool = net.global_pool
        self.drop_rate = 0.4

        self.rg_cls = Linear(self.num_features, 1, bias=True)
        self.cls_cls = Linear(self.num_features, 5, bias=True)
        self.ord_cls = Linear(self.num_features, 4, bias=True)

    def forward(self, x):
        x1 = self.conv_stem(x)
        x2 = self.bn1(x1)
        x3 = self.act1(x2)
        x4 = self.block0(x1)
        x5 = self.block1(x4)
        x6 = self.block2(x5)
        x7 = self.block3(x6)
        x8 = self.block4(x7)
        x9 = self.block5(x8)
        x10 = self.block6(x9)
        x11 = self.conv_head(x10)
        x12 = self.bn2(x11)
        x13 = self.act2(x12)
        x14 = self.global_pool(x13)
        if self.drop_rate > 0.0:
            x14 = F.dropout(x14, p=self.drop_rate, training=self.training)

        x15 = self.rg_cls(x14)
        x16 = self.cls_cls(x14)
        x17 = self.ord_cls(x14)
        return x15, x16, x17




## === cell 2
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = 0
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu().item()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device, dtype=out.dtype)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(out[i].item()))
            l2 = int(math.ceil(out[i].item()))
            pred_prob[i][l1] = 1 - (out[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob


def combine3output(r_out, c_out, o_out):
    R = regress2class(r_out.data)
    _, C = torch.max(c_out.data, 1)
    C = C.squeeze().item()
    _, O = torch.max(o_out.data, 1)
    O = O.squeeze().item()
    P = (R + C + O) / 3.0
    P = int(round(P))
    return P




## === cell 3
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
        super().__init__()

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




## === cell 4
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


def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img




## === cell 5
BASE_INPUT = "/kaggle/input/aptos2019-blindness-detection"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "../input/aptos2019-blindness-detection"
if not os.path.exists(BASE_INPUT):
    BASE_INPUT = "/kaggle/data/aptos2019-blindness-detection"

train_csv_path = os.path.join(BASE_INPUT, "train.csv")
train_img_dir = os.path.join(BASE_INPUT, "train_images")

test_csv_path = os.path.join(BASE_INPUT, "test.csv")
test_img_dir = os.path.join(BASE_INPUT, "test_images")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
test_ids = test_df["id_code"].values

transform1 = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((288, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

transform2 = transforms.Compose(
    [
        transforms.Resize((280, 280)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def find_weight_file(filename: str) -> str:
    candidates = [
        os.path.join("/kaggle/input", "weights", filename),
        os.path.join("../input", "weights", filename),
        os.path.join("/kaggle/input", filename),
        os.path.join("../input", filename),
        os.path.join("/kaggle/input/aptos2019-blindness-detection", filename),
        os.path.join("../input/aptos2019-blindness-detection", filename),
        os.path.join("/kaggle/data/aptos2019-blindness-detection", filename),
        os.path.join("/kaggle/data", filename),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    hits = glob.glob(os.path.join("/kaggle/input", "**", filename), recursive=True)
    if hits:
        return hits[0]
    hits = glob.glob(os.path.join("../input", "**", filename), recursive=True)
    if hits:
        return hits[0]
    hits = glob.glob(os.path.join("/kaggle/data", "**", filename), recursive=True)
    if hits:
        return hits[0]
    return ""


def _unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "network"]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _strip_common_prefixes(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if any(k.startswith("model.") for k in state_dict.keys()):
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}
    return state_dict


net1 = ThreeStage_Model().to(device)
weight_name = "B4_3stage_4epoch_384finetune.pkl"
weight_path = find_weight_file(weight_name)

loaded_ckpt = False
if weight_path:
    ckpt = torch.load(weight_path, map_location=device)
    sd = _strip_common_prefixes(_unwrap_state_dict(ckpt))
    try:
        net1.load_state_dict(sd, strict=True)
        loaded_ckpt = True
        print(f"Loaded checkpoint strictly from: {weight_path}")
    except Exception:
        missing, unexpected = net1.load_state_dict(sd, strict=False)
        loaded_ckpt = True
        print(f"Loaded checkpoint non-strictly from: {weight_path}")
        print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
else:
    warnings.warn(
        f"Weight file '{weight_name}' not found under /kaggle/input, ../input, or /kaggle/data. "
        "Will fall back to ImageNet-pretrained backbone weights to avoid a near-random 0.0 kappa submission."
    )

if not loaded_ckpt:
    try:
        net1.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
        net1.backbone.global_pool = GeM(flatten=True)
        net1 = net1.to(device)
        print("Initialized ThreeStage_Model backbone with pretrained=True as fallback.")
    except Exception as e:
        warnings.warn(
            f"Pretrained backbone fallback failed; proceeding with random weights. Error: {e}"
        )

net1.eval()




## === cell 6
def _predict_regression_value(model, img_path: str) -> float:
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return 0.0
    x = transform1(img).unsqueeze(0).to(device)
    with torch.no_grad():
        _, r_out, _ = model(x)
    return float(r_out.squeeze().detach().cpu().item())


def _apply_thresholds(vals: np.ndarray, thr: np.ndarray) -> np.ndarray:
    thr = np.asarray(thr, dtype=np.float32)
    preds = np.zeros(vals.shape[0], dtype=np.int64)
    for t in thr:
        preds += (vals >= t).astype(np.int64)
    return preds


def _fit_thresholds_for_qwk(y_true: np.ndarray, y_val: np.ndarray) -> np.ndarray:
    thr = np.array([0.75, 1.5, 2.5, 3.5], dtype=np.float32)
    best_thr = thr.copy()
    best_score = cohen_kappa_score(
        y_true, _apply_thresholds(y_val, best_thr), weights="quadratic"
    )

    steps = [0.15, 0.08, 0.04]
    for step in steps:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                lo = 0.0 if i == 0 else best_thr[i - 1] + 1e-3
                hi = 4.5 if i == 3 else best_thr[i + 1] - 1e-3
                candidates = np.clip(
                    np.array(
                        [best_thr[i] - step, best_thr[i], best_thr[i] + step],
                        dtype=np.float32,
                    ),
                    lo,
                    hi,
                )
                for cand in candidates:
                    trial = best_thr.copy()
                    trial[i] = cand
                    score = cohen_kappa_score(
                        y_true, _apply_thresholds(y_val, trial), weights="quadratic"
                    )
                    if score > best_score + 1e-6:
                        best_score = score
                        best_thr = trial
                        improved = True
    return best_thr


rng = np.random.RandomState(42)
idxs = np.arange(len(train_df))
rng.shuffle(idxs)
val_n = min(512, len(train_df))
val_idxs = idxs[:val_n]

y_true = train_df.loc[val_idxs, "diagnosis"].astype(int).values
y_reg = np.zeros(val_n, dtype=np.float32)

t0 = time.time()
for j, ridx in enumerate(val_idxs):
    img_id = train_df.loc[ridx, "id_code"]
    img_path = os.path.join(train_img_dir, f"{img_id}.png")
    y_reg[j] = _predict_regression_value(net1, img_path)
    if (j + 1) % 128 == 0:
        print(f"Calib inference {j+1}/{val_n} done. Elapsed {time.time()-t0:.1f}s")

best_thr = _fit_thresholds_for_qwk(y_true, y_reg)
print("Calibrated thresholds:", best_thr.tolist())
print(
    "Val QWK with calibrated thresholds:",
    cohen_kappa_score(y_true, _apply_thresholds(y_reg, best_thr), weights="quadratic"),
)



## === cell 7
submission_rows = []

with torch.no_grad():
    for i, idx in enumerate(test_ids):
        image_name = os.path.join(test_img_dir, f"{idx}.png")
        try:
            img = Image.open(image_name).convert("RGB")
        except Exception:
            submission_rows.append([idx, 0])
            continue

        img_t = transform1(img).unsqueeze(0).to(device)

        _, r_out, _ = net1(img_t)
        rv = float(r_out.data.squeeze().cpu().item())
        pred = int(np.sum(rv >= best_thr))
        pred = max(0, min(4, pred))
        submission_rows.append([idx, pred])

submission = pd.DataFrame(submission_rows, columns=["id_code", "diagnosis"])



## === cell 8
assert submission.shape[0] == len(test_df), "Submission row count mismatch vs test.csv"
submission = submission.merge(test_df[["id_code"]], on="id_code", how="right")
submission["diagnosis"] = submission["diagnosis"].fillna(0).astype(int)
submission = submission[["id_code", "diagnosis"]]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
