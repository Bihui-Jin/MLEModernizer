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

0.926329653636144

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the missing-weights crash by loading from `../input/weights/` if present, otherwise falling back to running the same model with randomly initialized weights so the notebook still completes and writes a valid `submission.csv`. I also fix the CUDA-only device setting so it runs on Kaggle CPU-only environments by using `cuda` only when available, and I remove hard `.cuda()` tensor creation to avoid device mismatch errors. Finally, I make the test id extraction robust (keep `id_code` strings) and ensure the submission is always populated and written with the exact required columns.'
- What this solution (achieved 0.00014) has done: 'Your current 0.0 score is consistent with the model running with randomly initialized weights (because `/kaggle/input/weights` is not part of the provided dataset), so the smallest meaningful improvement is to load ImageNet-pretrained weights for the same EfficientNet backbone you already use. This keeps the exact model architecture and inference logic intact, but makes the features non-random, which should move QWK substantially upward toward your target. I also fix one correctness bug in the backbone forward (it mistakenly feeds `block0(x1)` instead of `block0(x3)`), which is a minimal change that aligns with EfficientNet’s intended forward pass and improves predictions without changing the overall approach. Everything else (transforms, thresholding, submission schema, paths) is preserved, and the script still writes a valid `submission.csv`.'
- What this solution (achieved -0.05081) has done: 'Your current score is near-zero because you’re only using the regressor head output with fixed thresholds, while the provided ThreeStage model is designed to combine the classifier/regressor/ordinal heads into a more stable discrete prediction. To move the score upward toward your target with minimal disruption, I keep the same model and transforms, but change inference to use your existing `combine3output()` (and make it batch-safe) instead of regressor-only thresholding. I also load weights more robustly by stripping common prefixes (`module.`, `model.`) so that if your intended checkpoint exists it actually populates the correct keys; otherwise it still falls back to ImageNet-pretrained backbone as before. These changes preserve architecture/training semantics and only adjust prediction assembly and checkpoint compatibility, which is directly relevant to QWK.'
- What this solution (achieved -0.13126) has done: 'Your score is extremely low for this competition, which strongly suggests an inference-time mismatch rather than a small modeling weakness. I keep your exact model and thresholds, but fix the key bug in `combine3output()` where the ordinal head logits are incorrectly treated as class indices (argmax over 4 thresholds yields 0–3, not a 0–4 class). I also switch the test loop to batched inference (same predictions, much faster) and correctly convert ordinal logits to probabilities before combining, which should move QWK substantially upward toward your target without changing architecture, training, or loss semantics. The submission writing and paths remain unchanged and it still always produce a valid `submission.csv`.'
- What this solution (achieved -0.00284) has done: 'Your current negative QWK strongly suggests the inference-time label assembly is still mismatched to how the (supposed) 3-head model was trained, rather than the backbone capacity. I keep the exact model and transforms, but change `combine3output()` to fuse heads in a safer, more standard way: convert classifier logits to probabilities, convert ordinal logits to class probabilities (without an extra softmax that distorts them), and convert the regressor output into a soft class distribution, then average these distributions and take `argmax`. This is a minimal change (only post-processing) that typically increases QWK a lot when fixed-threshold/rounding is unstable, moving you toward the 0.926 target. I also ensure all outputs are properly shaped (B,1) vs (B,) and keep submission alignment intact.'
- What this solution (achieved -0.00284) has done: 'Your current negative QWK is consistent with an inference-time mismatch: the model returns `(c_out, r_out, o_out)` but `combine3output()` is currently written as if the first argument is the regressor output. I make the smallest direct fix by calling `combine3output(r_out, c_out, o_out)` with the correct tensors (and keep everything else unchanged), which should immediately move the score upward toward your target without changing architecture or training logic. I also add a tiny safety cast to ensure `r_out` is treated as `(B,)` float and `o_out` as `(B,4)` float inside `combine3output()` to avoid shape edge-cases that can silently degrade predictions. Submission formatting, paths, transforms, and model definition remain the same, and the notebook still write a valid `submission.csv`.'
- What this solution (achieved -0.00284) has done: 'Your score is extremely far below the target, and the code is currently doing inference with a randomly initialized 3-head classifier/regressor/ordinal head (because your private checkpoint path likely isn’t available), which yield near-random QWK. To move the score strongly upward while preserving your core model and inference logic, I keep the same `ThreeStage_Model` and `combine3output()`, but (1) deterministically load a matching EfficientNet-B4 NoisyStudent ImageNet checkpoint into the backbone only when the full task checkpoint is missing, and (2) freeze dropout by ensuring `.eval()` and avoiding any accidental training-mode behavior. I also add a strict submission alignment check to guarantee `id_code` order matches `test.csv`, preventing accidental row-order mistakes that can tank QWK. These are minimal, inference-only changes (no training, no architecture change) and should move QWK much closer to your target than random heads.'
- What this solution (achieved -0.00284) has done: 'Your current score is far below the target and the code is already producing a valid submission, so the main issue is prediction quality rather than formatting. The smallest likely improvement (without changing the model/training approach) is to ensure we actually load the intended trained checkpoint from the competition dataset tree instead of only looking in `/kaggle/input/weights` (which typically doesn’t exist). I add a minimal checkpoint search that first tries the exact filename anywhere under `/kaggle/input/aptos2019-blindness-detection/`, then falls back to any `.pkl/.pth/.pt/.bin` found there, and only then falls back to ImageNet-backbone-only. This keeps your architecture, transforms, and `combine3output()` logic intact, but should move QWK substantially upward toward your target if the checkpoint is present in the provided input.'
- What this solution (achieved 0.56783) has done: 'Your current score is extremely far below the target, and the code shows you are doing inference with an ImageNet-pretrained backbone but randomly initialized task heads when the real 3-stage checkpoint isn’t found, which yields near-random QWK. The smallest score-relevant change that preserves your core model/inference logic is to actually train only the existing heads (classifier/regressor/ordinal + final_regressor) for a single short CPU-friendly epoch on the provided `train.csv` images, while keeping the ImageNet backbone frozen. This does not change architecture, loss family, or prediction assembly (`combine3output`), but it converts random heads into weakly-trained heads, which should move QWK sharply upward toward the target. I also reuse your existing test transform for consistency and keep all paths and submission formatting identical.'
- What this solution (achieved 0.6266) has done: 'I remove expensive unused imports and duplicate model helper definitions that add noticeable startup time, and I prevent any accidental fallback training by requiring the checkpoint load path logic to avoid recursive globbing over `/kaggle/input` (which is very slow). I speed up dataloading and GPU transfer without changing the model or transforms by enabling persistent workers, prefetching, and setting `torch.set_num_threads` to reduce CPU oversubscription. I also vectorize a few hot-path postprocessing functions (`regress2class_prob` and ordinal target creation) to eliminate Python loops while keeping identical math. Finally, I enable safe inference optimizations (`inference_mode`, channels_last, and TF32 on Ampere+) that preserve evaluation semantics with only negligible floating-point differences.'

# 9. Code solution

## === cell 0
import os, glob, math, time, random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.transforms import functional as FT

from PIL import Image, ImageChops
import cv2

from sklearn.metrics import cohen_kappa_score
import timm

torch.set_num_threads(min(4, os.cpu_count() or 4))
torch.set_num_interop_threads(1)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def ordinal2class_prob(out):
    out = out.clamp(0.0, 1.0)
    pred_prob = torch.zeros(out.size(0), 5, device=out.device, dtype=out.dtype)
    pred_prob[:, 0] = 1 - out[:, 0]
    pred_prob[:, 1] = out[:, 0] * (1 - out[:, 1])
    pred_prob[:, 2] = out[:, 1] * (1 - out[:, 2])
    pred_prob[:, 3] = out[:, 2] * (1 - out[:, 3])
    pred_prob[:, 4] = out[:, 3]
    pred_prob = pred_prob / (pred_prob.sum(dim=1, keepdim=True) + 1e-12)
    return pred_prob


def regress2class_prob(out):
    out = out.view(-1)
    B = out.size(0)
    pred_prob = torch.zeros((B, 5), device=out.device, dtype=out.dtype)

    out_clamped = torch.clamp(out, 0.0, 4.0)
    low = torch.floor(out_clamped).to(torch.long)
    high = torch.ceil(out_clamped).to(torch.long)

    w_low = 1.0 - (out_clamped - low.to(out.dtype))
    w_high = 1.0 - (high.to(out.dtype) - out_clamped)

    is_four = out >= 4.0
    if is_four.any():
        pred_prob[is_four, 4] = 1.0

    not_four = ~is_four
    if not_four.any():
        idx = torch.arange(B, device=out.device)[not_four]
        l = low[not_four].clamp(0, 4)
        h = high[not_four].clamp(0, 4)
        pred_prob[idx, l] += w_low[not_four]
        pred_prob[idx, h] += w_high[not_four]

    pred_prob = pred_prob / (pred_prob.sum(dim=1, keepdim=True) + 1e-12)
    return pred_prob


def combine3output(r_out, c_out, o_out):
    r_out = r_out.float()
    c_out = c_out.float()
    o_out = o_out.float()

    if r_out.ndim == 2 and r_out.size(1) == 1:
        r_out = r_out.squeeze(1)

    if o_out.ndim == 1:
        o_out = o_out.view(-1, 4)

    if o_out.min().item() < 0.0 or o_out.max().item() > 1.0:
        o_out = torch.sigmoid(o_out)

    c_prob = F.softmax(c_out, dim=1)  # (B,5)
    r_prob = regress2class_prob(r_out)  # (B,5)
    o_prob = ordinal2class_prob(o_out)  # (B,5)

    p = (c_prob + r_prob + o_prob) / 3.0
    preds = torch.argmax(p, dim=1).detach().cpu().numpy().tolist()
    preds = [int(max(0, min(4, x))) for x in preds]
    return preds




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
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img




## === cell 4
DATA_ROOT = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].astype(str).tolist()

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

net1 = ThreeStage_Model()


def _clean_state_dict_keys(state_dict: dict) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("module.", "model."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        cleaned[nk] = v
    return cleaned


def _find_checkpoint():
    preferred_name = "0.926_B4_3stage_5epoch_320finetune.pkl"
    weights_dir = "/kaggle/input/weights"
    preferred_weight = os.path.join(weights_dir, preferred_name)
    if os.path.isfile(preferred_weight):
        return preferred_weight

    cand = os.path.join("/kaggle/input/aptos2019-blindness-detection", preferred_name)
    if os.path.isfile(cand):
        return cand

    hits = glob.glob(
        os.path.join(
            "/kaggle/input/aptos2019-blindness-detection", "**", preferred_name
        ),
        recursive=True,
    )
    hits = [h for h in hits if os.path.isfile(h)]
    if hits:
        return sorted(hits)[0]

    return None


weight_path = _find_checkpoint()

loaded_any = False
if weight_path is not None:
    state = torch.load(weight_path, map_location="cpu")
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]
    if isinstance(state, dict):
        state = _clean_state_dict_keys(state)
        missing, unexpected = net1.load_state_dict(state, strict=False)
        loaded_any = True
        print("Loaded task checkpoint:", weight_path)
        print("Missing keys (first 10):", missing[:10])
        print("Unexpected keys (first 10):", unexpected[:10])

if not loaded_any:
    b4 = timm.create_model("tf_efficientnet_b4_ns", pretrained=True)
    net1.backbone.load_state_dict(b4.state_dict(), strict=False)
    print("No task checkpoint found; loaded ImageNet pretrained backbone weights.")

net1 = net1.to(device)

if torch.cuda.is_available():
    net1 = net1.to(memory_format=torch.channels_last)




## === cell 5
_MEAN1 = np.array([0.384, 0.258, 0.174], dtype=np.float32)
_STD1 = np.array([0.124, 0.089, 0.094], dtype=np.float32)


def _trim_bbox_like_pil(rgb: np.ndarray) -> np.ndarray:
    bg = rgb[0, 0].astype(np.int16)  # (3,)
    diff = np.abs(rgb.astype(np.int16) - bg[None, None, :]).astype(np.int16)
    diff = diff * 2 - 10
    mask = (diff > 0).any(axis=2)
    if not mask.any():
        return rgb
    ys, xs = np.where(mask)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return rgb[y0:y1, x0:x1]


def _center_crop_to_4_3(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    if (w / h) >= (4 / 3):
        new_h = h
        new_w = int(h * 4 / 3)
    else:
        new_h = int(w * 3 / 4)
        new_w = w
    left = int((w - new_w) / 2)
    top = int((h - new_h) / 2)
    return rgb[top : top + new_h, left : left + new_w]


def _transform1_cv2_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    rgb = _trim_bbox_like_pil(rgb)
    rgb = _center_crop_to_4_3(rgb)
    rgb = cv2.resize(rgb, (384, 288), interpolation=cv2.INTER_AREA)  # (W,H)
    x = rgb.astype(np.float32) / 255.0
    x = (x - _MEAN1) / _STD1
    x = np.transpose(x, (2, 0, 1))  # CHW
    return torch.from_numpy(x)


class TrainImageDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self._cache = None  # set by enable_cache()

    def enable_cache(self):
        self._cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        idx = str(row["id_code"])
        y = int(row["diagnosis"])

        if self._cache is not None:
            cached = self._cache.get(idx)
            if cached is not None:
                return cached, y

        image_name = os.path.join(self.img_dir, f"{idx}.png")
        bgr = cv2.imread(image_name, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_name)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.transform is transform1:
            img = _transform1_cv2_to_tensor(rgb)
        else:
            img = Image.fromarray(rgb)
            img = self.transform(img)

        if self._cache is not None:
            self._cache[idx] = img
        return img, y


def _set_trainable_head_only(model: ThreeStage_Model):
    for p in model.parameters():
        p.requires_grad_(False)
    for module in [
        model.classifier,
        model.regressor,
        model.ordinal,
        model.final_regressor,
    ]:
        for p in module.parameters():
            p.requires_grad_(True)


def _qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def _class_weights_from_train_df(
    df: pd.DataFrame, num_classes: int = 5
) -> torch.Tensor:
    counts = (
        df["diagnosis"]
        .value_counts()
        .reindex(range(num_classes), fill_value=0)
        .values.astype(np.float32)
    )
    counts = np.maximum(counts, 1.0)
    inv = 1.0 / counts
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32)


def _train_heads_fixed_epochs(
    model: ThreeStage_Model, train_df: pd.DataFrame, epochs: int = 6
):
    _set_trainable_head_only(model)
    model = model.to(device)

    model.backbone.eval()
    for p in model.backbone.parameters():
        p.requires_grad_(False)

    ce_w = _class_weights_from_train_df(train_df, num_classes=5).to(device)
    ce = nn.CrossEntropyLoss(weight=ce_w)

    mse = nn.MSELoss()
    bce_prob = nn.BCELoss()

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=3e-4)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5
    )

    n = len(train_df)
    idx = np.arange(n)
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    val_n = max(200, int(0.15 * n))
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[val_idx].reset_index(drop=True)

    tr_ds = TrainImageDataset(tr_df, TRAIN_IMG_DIR, transform1)
    va_ds = TrainImageDataset(va_df, TRAIN_IMG_DIR, transform1)

    tr_ds.enable_cache()

    nw = min(4, os.cpu_count() or 2)
    bs = 8 if torch.cuda.is_available() else 4
    loader = DataLoader(
        tr_ds,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )

    ord_arange = torch.arange(4, device=device).view(1, -1)

    for ep in range(epochs):
        model.train()
        model.backbone.eval()
        t0 = time.time()
        running = 0.0

        for batch_imgs, batch_y in loader:
            if torch.cuda.is_available():
                batch_imgs = batch_imgs.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
            else:
                batch_imgs = batch_imgs.to(device)
            y = batch_y.to(device)

            feats = model.backbone(batch_imgs)
            c_out = model.classifier(feats)
            r_out_raw = model.regressor(feats)
            o_out_raw = model.ordinal(feats)

            r_out = torch.sigmoid(r_out_raw) * 4.5
            o_out = torch.sigmoid(o_out_raw)

            loss_c = ce(c_out, y)

            y_float = y.float()
            r_out_1d = (
                r_out.squeeze(1) if (r_out.ndim == 2 and r_out.size(1) == 1) else r_out
            )
            loss_r = mse(r_out_1d, y_float)

            t_ord = (y.view(-1, 1) > ord_arange).float()
            o_prob = o_out.clamp(1e-5, 1 - 1e-5)
            loss_o = bce_prob(o_prob, t_ord)

            out_final_in = torch.cat((c_out, r_out_raw, o_out_raw), 1)
            out_final = model.final_regressor(out_final_in)
            out_final = torch.sigmoid(out_final) * 4.5
            out_final_1d = (
                out_final.squeeze(1)
                if (out_final.ndim == 2 and out_final.size(1) == 1)
                else out_final
            )
            loss_f = mse(out_final_1d, y_float)

            loss = loss_c + 0.5 * loss_r + 0.5 * loss_o + 0.5 * loss_f

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += float(loss.item())

        scheduler.step()

        model.eval()
        y_true_all, y_pred_all = [], []
        with torch.inference_mode():
            for batch_imgs, batch_y in va_loader:
                if torch.cuda.is_available():
                    batch_imgs = batch_imgs.to(device, non_blocking=True).to(
                        memory_format=torch.channels_last
                    )
                else:
                    batch_imgs = batch_imgs.to(device)
                c_out, r_out, o_out = model(batch_imgs)
                preds = combine3output(r_out, c_out, o_out)
                y_true_all.extend(batch_y.cpu().numpy().tolist())
                y_pred_all.extend(preds)

        print(
            f"epoch {ep+1}/{epochs} "
            f"lr={scheduler.get_last_lr()[0]:.2e} "
            f"loss={running/max(1,len(loader)):.4f} "
            f"val_qwk={_qwk(y_true_all, y_pred_all):.4f} "
            f"time={time.time()-t0:.1f}s"
        )


if not loaded_any:
    start_t = time.time()
    _train_heads_fixed_epochs(net1, train_df, epochs=6)
    print(f"Head-only training finished in {time.time()-start_t:.1f}s")

net1.eval()
for m in net1.modules():
    if isinstance(m, (nn.Dropout, nn.BatchNorm2d, nn.BatchNorm1d)):
        m.eval()




## === cell 6
class TestImageDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        image_name = os.path.join(self.img_dir, f"{idx}.png")

        bgr = cv2.imread(image_name, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(image_name)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if self.transform is transform1:
            img = _transform1_cv2_to_tensor(rgb)
        else:
            img = Image.fromarray(rgb)
            img = self.transform(img)
        return idx, img


test_ds = TestImageDataset(test_ids, TEST_IMG_DIR, transform1)

nw = min(4, os.cpu_count() or 2)
bs = 16 if torch.cuda.is_available() else 4
test_loader = DataLoader(
    test_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=nw,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(nw > 0),
    prefetch_factor=2 if nw > 0 else None,
)

submission = []
with torch.inference_mode():
    for batch_ids, batch_imgs in test_loader:
        if torch.cuda.is_available():
            batch_imgs = batch_imgs.to(device, non_blocking=True).to(
                memory_format=torch.channels_last
            )
        else:
            batch_imgs = batch_imgs.to(device)

        c_out, r_out, o_out = net1(batch_imgs)
        preds = combine3output(r_out, c_out, o_out)

        for i, idx in enumerate(batch_ids):
            submission.append([str(idx), int(preds[i])])

df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["id_code"] = df["id_code"].astype(str)
df["diagnosis"] = df["diagnosis"].astype(int)

df = df.set_index("id_code").reindex(test_ids).reset_index()
if df["diagnosis"].isna().any():
    raise RuntimeError("Some test ids are missing predictions after reindexing.")
df["diagnosis"] = df["diagnosis"].astype(int)

df.to_csv("submission.csv", index=False)

print(df.head())
print("Wrote submission.csv with shape:", df.shape)
print("diagnosis value counts:\n", df["diagnosis"].value_counts().sort_index())
