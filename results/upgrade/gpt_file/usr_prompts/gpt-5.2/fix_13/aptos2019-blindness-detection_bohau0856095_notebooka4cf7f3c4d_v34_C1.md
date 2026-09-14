# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.9030625594169336

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I make the pipeline robust to the missing external weights by falling back to a lightweight, deterministic baseline that still produces a valid submission (so you always get a .csv). I also fix the CUDA/CPU dtype mismatch by ensuring the loaded model state is placed onto the same device as the inputs. Finally, I keep your existing model definitions and inference logic intact, only adding a minimal backup path and a safe CPU fallback when weights aren’t available, so the notebook runs end-to-end in the provided environment.'
- What this solution (achieved 0.16543) has done: 'The timeout is dominated by per-image I/O and transforms in inference (opening 367 images one-by-one, doing PIL ops each time) and by inefficient CPU/GPU transfer patterns. I keep the exact model and transforms, but make inference batched with a DataLoader, reuse workers to parallelize image decode/transform, and avoid repeated Python overhead. I also vectorize `regress2class` (same thresholds/semantics) and enable `inference_mode()`/channels-last on CUDA for equivalent but faster execution. No training logic, architecture, loss, or augmentation behavior is changed.'
- What this solution (achieved 0.16543) has done: 'Your current score (0.16543) is far below the target (0.90306), and the biggest reason is that you are almost certainly *not* loading the intended pretrained weights, so you fall back to training only the classifier head for 3 epochs with a frozen backbone (which cannot reach the target). I keep your model, transforms, and inference logic unchanged, but make the weight discovery robust by searching the competition dataset directories for the `.pkl` and also allowing common serialization wrappers (`state_dict`, `model`, `net`) while still loading strictly when possible. Additionally, I ensure we always infer with `final=True` when weights exist only if the checkpoint matches that head; otherwise we keep your exact current weighted-regression path to preserve semantics. These are minimal changes focused on actually using the intended model weights (the only realistic way to move toward 0.90 without changing core training/architecture).'
- What this solution (achieved 0.16543) has done: 'Your score is far below the target, and the dominant issue is that inference is not using the strong “final” head that the provided 3‑stage model defines; instead it only uses the intermediate regressor output, which is typically much weaker for QWK. I keep the exact architecture, transforms, loss, and training loop unchanged, but when weights are available I automatically attempt the `final=True` forward path first and only fall back to the current `(_, r_out, _)` path if the checkpoint clearly doesn’t support it. This is a minimal inference-only change that better matches how a 3‑stage model is usually intended to be used and should move the score substantially toward the 0.903 target without changing training semantics. I also keep the submission formatting/alignment checks intact so you still always get a valid `submission.csv`.'
- What this solution (achieved 0.26833) has done: 'Your score is far below the target, so we should improve (not degrade) QWK with minimal, inference-only changes. The biggest low-risk gain for QWK here is calibrating the regression-to-class thresholds on a held-out validation split when you don’t have external weights, instead of using the fixed `[0.75, 1.5, 2.5, 3.5]` thresholds that are unlikely to be optimal. This preserves your exact model, loss, transforms, and training loop; it only learns 4 scalar thresholds post-training to better match the QWK metric. When pretrained weights are available, nothing changes (we keep your existing `final=True` path and fixed thresholds), preserving semantics and stability.'
- What this solution (achieved 0.04029) has done: 'Your current score (0.26833) is far below the target (0.90306), so we should improve QWK with minimal, metric-aligned changes while keeping your model/training intact. The biggest low-risk issue is that when `have_weights=False` you train the classifier head but then optimize thresholds on a *softmax-expected-class* continuous value, which is poorly matched to QWK; instead, we can calibrate thresholds on the model’s **regressor output** (already in your architecture) without changing the model or loss. To preserve your current training approach, we train the classifier head exactly as before, but at threshold-optimization time and test time we use the frozen regressor’s continuous predictions to apply thresholds (still just a post-processing step). This typically yields a materially higher QWK than thresholding the softmax-expected-class, while staying within your constraints.'
- What this solution (achieved 0.26833) has done: 'Your current score is far below the target, so we should improve QWK with minimal risk while keeping your model, training loop, and inference structure intact. The biggest issue is that when no external weights are found you train only the classifier head, but then you threshold the **untrained regressor** output, which collapses predictions and tanks QWK. I keep the exact same head-only training, but switch the no-weights path to calibrate and apply thresholds on the classifier’s continuous signal (expected class from softmax), which matches what you actually trained. When weights are available, behavior is unchanged.'

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
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.transforms import functional as FT
from PIL import Image, ImageChops

from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedShuffleSplit
import timm

import cv2

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(True)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass


def seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def _thr_tensor(device_, dtype_):
    return torch.as_tensor(threshold, device=device_, dtype=dtype_).view(1, -1)


def regress2class(out: torch.Tensor) -> torch.Tensor:
    thr = _thr_tensor(out.device, out.dtype)
    return (out.view(-1, 1) >= thr).sum(dim=1).to(dtype=torch.float32)


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5).to(out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    out = out.view(-1).to(dtype=torch.float32)
    n = out.numel()
    pred_prob = torch.zeros((n, 5), device=out.device, dtype=out.dtype)
    out_clamped = torch.clamp(out, 0.0, 4.0)
    l1 = torch.floor(out_clamped).to(torch.long)
    l2 = torch.ceil(out_clamped).to(torch.long)
    w2 = out_clamped - l1.to(out_clamped.dtype)  # distance to floor
    w1 = 1.0 - w2
    idx = torch.arange(n, device=out.device)
    pred_prob[idx, l1] += w1
    pred_prob[idx, l2] += 1.0 - (
        l2.to(out_clamped.dtype) - out_clamped
    )  # matches original
    mask4 = out >= 4.0
    if mask4.any():
        pred_prob[mask4] = 0.0
        pred_prob[mask4, 4] = 1.0
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


def _cv_trim_bbox(img_bgr: np.ndarray):
    h, w = img_bgr.shape[:2]
    corner = img_bgr[0, 0, :]  # BGR
    diff = cv2.absdiff(img_bgr, corner.reshape(1, 1, 3))
    diff = diff.astype(np.int16) * 2 - 10
    diff = np.clip(diff, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > 0)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, w, h
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return x0, y0, x1, y1


def _cv_center_crop_to_4_3(img_bgr: np.ndarray):
    h, w = img_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return img_bgr
    if (w / h) >= (4 / 3):
        new_h = h
        new_w = int(h * 4 / 3)
    else:
        new_h = int(w * 3 / 4)
        new_w = w
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return img_bgr[top : top + new_h, left : left + new_w, :]


def _cv_preprocess_to_tensor(path: str, out_w: int, out_h: int, mean, std):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    x0, y0, x1, y1 = _cv_trim_bbox(img)
    img = img[y0:y1, x0:x1, :]
    img = _cv_center_crop_to_4_3(img)
    img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img)  # float32




## === cell 4
BASE1 = "../input/aptos2019-blindness-detection"
BASE2 = "/kaggle/input/aptos2019-blindness-detection"
BASE3 = "/kaggle/data/aptos2019-blindness-detection"
BASE4 = "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection"
base = (
    BASE1
    if os.path.exists(BASE1)
    else (
        BASE2 if os.path.exists(BASE2) else (BASE3 if os.path.exists(BASE3) else BASE4)
    )
)

train_csv_path = os.path.join(base, "train.csv")
test_csv_path = os.path.join(base, "test.csv")
train_img_dir = os.path.join(base, "train_images")
test_img_dir = os.path.join(base, "test_images")

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
test_ids = test_df["id_code"].values

input_size = 256

mean = [0.384, 0.258, 0.174]
std = [0.124, 0.089, 0.094]

train_transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        photometric_distort(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

test_transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

net = ThreeStage_Model()

expected_name = "B4_3stage_41epoch_CLAHE.pkl"

candidate_paths = [
    os.path.join("../input", "weights", expected_name),
    os.path.join("/kaggle/input", "weights", expected_name),
    os.path.join("/kaggle/data", "weights", expected_name),
]
candidate_paths += glob.glob(os.path.join("../input", "*", expected_name))
candidate_paths += glob.glob(os.path.join("/kaggle/input", "*", expected_name))
candidate_paths += glob.glob(os.path.join("/kaggle/data", "*", expected_name))
candidate_paths += glob.glob(os.path.join("../input", "*", "weights", expected_name))
candidate_paths += glob.glob(
    os.path.join("/kaggle/input", "*", "weights", expected_name)
)
candidate_paths += glob.glob(
    os.path.join("/kaggle/data", "*", "weights", expected_name)
)
candidate_paths += glob.glob(os.path.join(base, "**", expected_name), recursive=True)

weight_path = next((p for p in candidate_paths if os.path.exists(p)), None)
have_weights = weight_path is not None


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "net"):
            if k in obj and isinstance(obj[k], (dict,)):
                return obj[k]
    return obj


def _strip_prefix_if_present(sd, prefix):
    if not isinstance(sd, dict):
        return sd
    if all(isinstance(k, str) and k.startswith(prefix) for k in sd.keys()):
        return {k[len(prefix) :]: v for k, v in sd.items()}
    return sd


ckpt_has_final_head = False

if have_weights:
    raw = torch.load(weight_path, map_location="cpu")
    state = _extract_state_dict(raw)

    loaded = False
    errors = []
    for sd in (
        state,
        _strip_prefix_if_present(state, "module."),
        _strip_prefix_if_present(state, "net."),
        _strip_prefix_if_present(state, "model."),
    ):
        try:
            net.load_state_dict(sd, strict=True)
            if isinstance(sd, dict):
                ckpt_has_final_head = any(
                    isinstance(k, str) and k.startswith("final_regressor.")
                    for k in sd.keys()
                )
            loaded = True
            break
        except Exception as e:
            errors.append(str(e))

    if loaded:
        print("Loaded weights:", weight_path)
        print("Checkpoint has final_regressor head:", ckpt_has_final_head)
    else:
        have_weights = False
        ckpt_has_final_head = False
        print(
            "WARNING: Found weights file but could not load strictly; falling back to training head.\n"
            f"Path: {weight_path}\n"
            f"Last error: {errors[-1] if errors else 'unknown'}"
        )
else:
    print(
        f"WARNING: Could not find model weights '{expected_name}'. "
        "Will train a small model on the provided training set to improve score over 0.0 baseline."
    )

net = net.to(device)

if device.startswith("cuda"):
    net = net.to(memory_format=torch.channels_last)

if have_weights and device.startswith("cuda"):
    try:
        net = torch.compile(net, mode="reduce-overhead", fullgraph=False)
    except Exception:
        pass




## === cell 5
class APTOSDataset(Dataset):
    def __init__(
        self,
        df,
        img_dir,
        transform,
        with_label=True,
        use_cv2=False,
        input_size=256,
        mean=None,
        std=None,
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.with_label = with_label

        id_codes = df["id_code"].astype(str).values
        self.ids = id_codes.tolist()
        self.paths = [os.path.join(img_dir, f"{i}.png") for i in self.ids]

        self.labels = None
        if with_label:
            self.labels = df["diagnosis"].astype(np.int64).values

        self.use_cv2 = bool(use_cv2)
        self.out_h = input_size * 3 // 4
        self.out_w = input_size
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img_path = self.paths[i]
        if self.use_cv2:
            img = _cv_preprocess_to_tensor(
                img_path, self.out_w, self.out_h, self.mean, self.std
            )
        else:
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)

        if self.with_label:
            y = int(self.labels[i])
            return img, y
        else:
            return img, self.ids[i]


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def _apply_thresholds(preds_cont: np.ndarray, thr_list):
    thr_arr = np.asarray(thr_list, dtype=np.float32).reshape(1, -1)
    preds = preds_cont.reshape(-1, 1)
    return (preds >= thr_arr).sum(axis=1).astype(np.int64)


def _optimize_thresholds_for_qwk(
    y_true: np.ndarray, y_cont: np.ndarray, init_thr, max_iter=30
):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_cont = np.asarray(y_cont, dtype=np.float32)

    thr = np.sort(np.asarray(init_thr, dtype=np.float32)).tolist()

    best_thr = thr[:]
    best_score = qwk(y_true, _apply_thresholds(y_cont, best_thr))

    for _ in range(max_iter):
        improved = False
        for j in range(4):
            lo = 0.0 if j == 0 else best_thr[j - 1] + 1e-3
            hi = 4.5 if j == 3 else best_thr[j + 1] - 1e-3
            if hi <= lo:
                continue

            grid = np.linspace(lo, hi, 41, dtype=np.float32)
            local_best_t = best_thr[j]
            local_best_s = best_score

            for t in grid:
                cand = best_thr[:]
                cand[j] = float(t)
                s = qwk(y_true, _apply_thresholds(y_cont, cand))
                if s > local_best_s + 1e-12:
                    local_best_s = s
                    local_best_t = float(t)

            if local_best_t != best_thr[j]:
                best_thr[j] = local_best_t
                best_score = local_best_s
                improved = True

        if not improved:
            break

    return best_thr, float(best_score)




## === cell 6
best_threshold = None

if not have_weights:
    y_all = train_df["diagnosis"].astype(int).values
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=0)
    tr_idx, va_idx = next(sss.split(np.zeros(len(y_all)), y_all))
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    train_ds = APTOSDataset(
        tr_df,
        train_img_dir,
        train_transform,
        with_label=True,
        use_cv2=True,
        input_size=input_size,
        mean=mean,
        std=std,
    )
    val_ds = APTOSDataset(
        va_df,
        train_img_dir,
        test_transform,
        with_label=True,
        use_cv2=True,
        input_size=input_size,
        mean=mean,
        std=std,
    )

    batch_size = 8 if device.startswith("cuda") else 4
    num_workers = (
        min(8, os.cpu_count() or 2)
        if device.startswith("cuda")
        else min(4, os.cpu_count() or 2)
    )

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.startswith("cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=seed_worker,
    )

    net.train()
    for p in net.backbone.parameters():
        p.requires_grad = False
    for p in net.classifier.parameters():
        p.requires_grad = True
    for p in net.regressor.parameters():
        p.requires_grad = False
    for p in net.ordinal.parameters():
        p.requires_grad = False
    for p in net.final_regressor.parameters():
        p.requires_grad = False

    optimizer = optim.Adam(net.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 8

    for ep in range(1, epochs + 1):
        t0 = time.time()
        net.train()
        running_loss = 0.0
        n = 0
        for xb, yb in train_loader:
            if device.startswith("cuda"):
                xb = xb.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
                xb = xb.to(device)
            yb = torch.as_tensor(yb, device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            c_out, _, _ = net(xb)
            loss = criterion(c_out, yb)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * xb.size(0)
            n += xb.size(0)

        net.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                if device.startswith("cuda"):
                    xb = xb.to(device, non_blocking=True).contiguous(
                        memory_format=torch.channels_last
                    )
                else:
                    xb = xb.to(device)
                c_out, _, _ = net(xb)
                pred = torch.argmax(c_out, dim=1).detach().cpu().numpy().tolist()
                y_pred.extend(pred)
                y_true.extend(list(yb))
        score = qwk(y_true, y_pred) if len(set(y_true)) > 1 else 0.0
        print(
            f"epoch {ep}/{epochs} | loss {running_loss/max(n,1):.4f} | val_qwk {score:.4f} | time {time.time()-t0:.1f}s"
        )

    net.eval()
    y_true_val = []
    y_cont_val = []
    arange5 = torch.arange(5, device=device, dtype=torch.float32).view(1, 5)
    with torch.no_grad():
        for xb, yb in val_loader:
            if device.startswith("cuda"):
                xb = xb.to(device, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
                xb = xb.to(device)
            c_out, _, _ = net(xb)
            prob = torch.softmax(c_out, dim=1)
            cont = (prob * arange5.to(dtype=prob.dtype)).sum(dim=1)
            y_cont_val.append(cont.detach().cpu().numpy().astype(np.float32))
            y_true_val.append(np.asarray(yb, dtype=np.int64))

    y_true_val = np.concatenate(y_true_val, axis=0)
    y_cont_val = np.concatenate(y_cont_val, axis=0)

    best_threshold, best_thr_score = _optimize_thresholds_for_qwk(
        y_true=y_true_val,
        y_cont=y_cont_val,
        init_thr=threshold,
        max_iter=25,
    )
    print(
        "Optimized thresholds on classifier expected-class (used only when have_weights=False):",
        best_threshold,
        "| val_qwk:",
        best_thr_score,
    )

    net.eval()
else:
    net.eval()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_55/3520882012.py in <cell line: 0>()
     86         running_loss = 0.0
     87         n = 0
---> 88         for xb, yb in train_loader:
     89             if device.startswith("cuda"):
     90                 xb = xb.to(device, non_blocking=True).contiguous(

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

error: Caught error in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/2207100367.py", line 37, in __getitem__
    img = _cv_preprocess_to_tensor(
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/679216782.py", line 83, in _cv_preprocess_to_tensor
    x0, y0, x1, y1 = _cv_trim_bbox(img)
                     ^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/679216782.py", line 52, in _cv_trim_bbox
    diff = cv2.absdiff(img_bgr, corner.reshape(1, 1, 3))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
cv2.error: OpenCV(4.12.0) /io/opencv/modules/core/src/arithm.cpp:667: error: (-215:Assertion failed) type2 == CV_64F && (sz2.height == 1 || sz2.height == 4) in function 'arithm_op'



## === cell 7
torch.set_grad_enabled(False)

test_ds = APTOSDataset(
    test_df,
    test_img_dir,
    test_transform,  # unused when use_cv2=True
    with_label=False,
    use_cv2=True,
    input_size=input_size,
    mean=mean,
    std=std,
)

if device.startswith("cuda"):
    infer_bs = 32
    num_workers = min(8, os.cpu_count() or 2)
else:
    infer_bs = 4
    num_workers = min(2, os.cpu_count() or 2)

test_loader = DataLoader(
    test_ds,
    batch_size=infer_bs,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=device.startswith("cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
)

submission_ids = []
submission_pred = []

use_final_forward = bool(have_weights and ckpt_has_final_head)

arange5_infer = torch.arange(5, device=device, dtype=torch.float32).view(1, 5)

with torch.inference_mode():
    for xb, idb in test_loader:
        if device.startswith("cuda"):
            xb = xb.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
        else:
            xb = xb.to(device)

        if have_weights:
            if use_final_forward:
                try:
                    out = net(xb, final=True)
                    pred = (
                        regress2class(out.squeeze(1))
                        .to(dtype=torch.int64)
                        .cpu()
                        .numpy()
                    )
                except Exception:
                    _, r_out, _ = net(xb)
                    pred = (
                        regress2class(r_out.squeeze(1))
                        .to(dtype=torch.int64)
                        .cpu()
                        .numpy()
                    )
            else:
                _, r_out, _ = net(xb)
                pred = (
                    regress2class(r_out.squeeze(1)).to(dtype=torch.int64).cpu().numpy()
                )
        else:
            c_out, _, _ = net(xb)
            prob = torch.softmax(c_out, dim=1)
            cont = (prob * arange5_infer.to(dtype=prob.dtype)).sum(dim=1)
            cont = cont.detach().cpu().numpy().astype(np.float32)
            thr_use = best_threshold if best_threshold is not None else threshold
            pred = _apply_thresholds(cont, thr_use)

        submission_ids.extend(list(idb))
        submission_pred.extend(pred.tolist())

submission = np.column_stack(
    [np.array(submission_ids, dtype=object), np.array(submission_pred, dtype=np.int64)]
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_55/3650251786.py in <cell line: 0>()
     39 
     40 with torch.inference_mode():
---> 41     for xb, idb in test_loader:
     42         if device.startswith("cuda"):
     43             xb = xb.to(device, non_blocking=True).contiguous(

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

error: Caught error in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/2207100367.py", line 37, in __getitem__
    img = _cv_preprocess_to_tensor(
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/679216782.py", line 83, in _cv_preprocess_to_tensor
    x0, y0, x1, y1 = _cv_trim_bbox(img)
                     ^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/679216782.py", line 52, in _cv_trim_bbox
    diff = cv2.absdiff(img_bgr, corner.reshape(1, 1, 3))
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
cv2.error: OpenCV(4.12.0) /io/opencv/modules/core/src/arithm.cpp:667: error: (-215:Assertion failed) type2 == CV_64F && (sz2.height == 1 || sz2.height == 4) in function 'arithm_op'



## === cell 8
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

assert (
    len(df) == len(test_ids) and len(df) > 0
), "Submission DataFrame is empty or mis-sized."

df = df.set_index("id_code").loc[pd.Index(test_ids, name="id_code")].reset_index()

df.to_csv("submission.csv", index=False)
print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print(
    "Weights used:",
    have_weights,
    "| weight_path:",
    weight_path,
    "| ckpt_has_final_head:",
    ckpt_has_final_head,
    "| use_final_forward:",
    use_final_forward,
    "| best_threshold:",
    best_threshold,
    "| base:",
    base,
    "| device:",
    device,
)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/845897336.py in <cell line: 0>()
----> 1 df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
      2 df["id_code"] = df["id_code"].astype(str)
      3 df["diagnosis"] = df["diagnosis"].astype(int)
      4 
      5 assert (

NameError: name 'submission' is not defined
