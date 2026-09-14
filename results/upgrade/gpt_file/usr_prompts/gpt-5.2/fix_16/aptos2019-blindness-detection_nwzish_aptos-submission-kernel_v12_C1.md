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

3.7

# 3. Installed packages

No external packages required in the script and installed.

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

0.9080662143998636

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the fastai import breakage by making the notebook compatible with the fastai version available in your environment (your traceback shows fastai v2), while keeping the same overall pipeline: load all images into RAM, build a databunch-like object, load EfficientNet-B5 weights if present, and run TTA on the test set. Specifically, I replace the fastai v1 `ImageList/DataBunch/Learner.TTA` usage with a minimal PyTorch Dataset/DataLoader + deterministic TTA that matches the original intent (argmax over averaged augmented predictions). I also fix missing imports (`tqdm`, fastai symbols), ensure paths resolve, and guarantee a valid `submission.csv` with `id_code,diagnosis` aligned to `test.csv` order. The model architecture and weight-loading behavior remain unchanged; if the external weights dataset isn’t available, the script still run end-to-end and produce a submission (with likely low score).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with a broken inference setup: the model is initialized with `num_classes = train['diagnosis'].nunique()` (5), but the provided checkpoint name strongly suggests it was trained as a *regression/ordinal* head with 1 output, so your current argmax over 5 random logits yields near-constant/wrong classes. To move the score toward the target with minimal core-logic change, I (1) make the head size match the checkpoint when it exists (use 1 output), (2) switch post-processing to the standard “round-and-clip” used for 1D regression outputs (still producing 0–4), and (3) keep the existing TTA/normalization/data loading intact. If the checkpoint is not found, the script fall back to the existing 5-class argmax behavior to still generate a valid submission.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an inference/weight mismatch: the checkpoint may not be found (so you submit random predictions), or it is loaded but the head/output interpretation doesn’t match what the checkpoint was trained for. I make the weight discovery more robust by searching under `/kaggle/input/**/effi_b_30000_1.pth` (and also accept common alternative filenames), so the intended pretrained checkpoint is actually used when present. I also auto-detect whether the loaded checkpoint’s final FC layer is 1-d (regression) or 5-d (classification) and set `use_regression_head/num_classes` accordingly, while keeping your exact model/tta/prediction semantics unchanged. Finally, I keep submission alignment strict to `test.csv` order and ensure `submission.csv` is always produced.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with “random/constant predictions” caused by failing to actually use the intended checkpoint and/or producing outputs that are badly calibrated for QWK (especially if the checkpoint isn’t found). I make the weight-file discovery robust across the provided dataset tree and enforce strict loading when a checkpoint is found so we don’t silently run with random weights. Then, without changing the model or TTA core logic, I add a tiny validation split on the training set to fit 4 optimal regression thresholds (only when using a 1-output head) and apply them to test predictions; this is a standard minimal post-processing step that directly targets QWK. If no checkpoint is found, the script still produce a valid `submission.csv` (but that case likely remain low-scoring).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests you’re still submitting essentially untrained/random predictions, most likely because the intended checkpoint is not being loaded (or is mismatched) and the script silently continues. I make weight loading strict when a checkpoint is found (fail fast instead of silently producing random outputs), and I broaden the weight-file search to include common EfficientNet-B5 DR checkpoint names/locations so the model actually uses learned weights. To move QWK upward with minimal semantic change, I also ensure that when using a 1-d regression head we *always* convert outputs to 0–4 via fitted thresholds (and I keep the same small holdout calibration you already have). These are minimal, directly score-relevant changes and the script still always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a broken/incorrect checkpoint load path (so you’re effectively predicting with random weights) and/or silent mismatch between the checkpoint keys and your model keys. I make weight-file discovery more robust across `/kaggle/input`, and I load non-strictly but with an explicit check that the final FC layer actually loaded (so we don’t accidentally submit random heads). If the checkpoint is a regression head (1 output), I keep your existing threshold-fitting but ensure it only runs when we truly loaded a usable head; otherwise we fall back to default rounding/clip so predictions aren’t degenerate. These are minimal changes that keep your model/TTA pipeline intact while making it far more likely you’re actually using the intended trained weights, moving QWK up toward the target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with an inference mismatch that makes predictions essentially random/constant: the checkpoint head may not be loaded (your `loaded_ok` check is too narrow and can be false even when the head is loaded under a different key), and regression checkpoints commonly store weights as `fc.*` not `_fc.*`. I make weight loading more reliable by (1) detecting the true head keys in the checkpoint, (2) mapping `fc.*` ↔ `_fc.*` when needed, and (3) setting `loaded_ok` based on actual head tensor shape match rather than only missing-keys. This keeps your model, preprocessing, and TTA identical, but makes it far more likely you actually use the intended trained weights and therefore move QWK upward toward the target. The rest of the pipeline (threshold fitting, submission alignment, output format) stays the same and still always writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with still submitting effectively random/degenerate predictions due to a checkpoint/head mismatch that results in a “loaded_ok” false-positive/false-negative and a bad output-to-class conversion. I make weight loading head-safe by (1) detecting whether the checkpoint is 1-output or 5-output and instantiating the model accordingly, (2) remapping `fc.*` and `_fc.*` keys both ways and also handling common prefixes, and (3) verifying that the loaded head tensors actually landed in the model (not just present in the checkpoint). Then, keeping your exact inference/TTA and threshold-fitting approach, I ensure regression outputs are converted to 0–4 via calibrated thresholds (or a safe default) and that classification outputs use softmax-averaged TTA (same semantics, just correct averaging). This is a minimal, directly score-relevant fix aimed at moving QWK up toward your target while still always producing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with “the intended checkpoint was not actually used”, so the model runs with random weights and produces near-random/degenerate predictions. I make checkpoint discovery/loading stricter and more compatible (handle common key prefixes and `_fc` vs `fc` naming, plus `_global_params.` / `global_params.`), and I only trust calibration when we can confirm the head truly loaded. These changes keep your exact EfficientNet/TTA/thresholding core logic intact, but greatly increase the chance you’re doing real inference with the trained weights (which should move QWK up toward the target). The script still always produce a valid `submission.csv` aligned to `test.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with the script never producing a scored submission because it raises an error when the expected external EfficientNet-B5 checkpoint isn’t available (or doesn’t load cleanly). I make weight loading “safe”: if no compatible checkpoint is found, the code continue and still generate a valid `submission.csv` (instead of failing), while keeping the same model/TTA/thresholding logic when the checkpoint does load. To move the score upward when you do have a checkpoint, I also expand the checkpoint key remapping slightly (common `state_dict` nesting + `model.` prefixes are already handled; we add a couple of frequent head-name variants) and keep threshold calibration only when the regression head is confirmed loaded. Finally, I keep the submission strictly aligned to `test.csv` order, but simplify mapping to avoid any accidental misalignment.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with submitting essentially untrained/random predictions because the intended EfficientNet-B5 checkpoint is not actually being used (missing or head not loaded), so the model outputs are degenerate for QWK. To move the score upward with minimal changes, I make checkpoint discovery/loading slightly more robust (handle `ema_state_dict`, `teacher`, `student`, and `model_state` nestings; handle `state_dict` values stored as lists/tuples; and support an additional common head name `head.*`). I also strengthen the “did the head really load?” verification to check both `_fc` and `fc` modules safely, so we only trust regression-threshold calibration when the checkpoint truly loaded. Core model, preprocessing, TTA, and threshold-fitting logic remain the same; the script still always writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from submitting effectively random/degenerate predictions because no compatible checkpoint is actually being used (or the head weights aren’t really landing in the model), so the inference outputs don’t correlate with labels. I make weight loading more reliable but still minimal: (1) broaden checkpoint unwrapping to handle more common nestings, (2) additionally remap `fc.*` ↔ `_fc.*` keys both directions (currently you only remap one way in one place), and (3) strengthen the “head really loaded” check to use shape+value-change on whichever head key exists, not only the remapped key. This preserves your exact EfficientNet-B5 definition, preprocessing, TTA, and threshold-fitting logic, but increases the chance that when a checkpoint is present it is actually used—moving QWK upward toward the target. If no checkpoint exists in the environment, the code still run end-to-end and produce a valid `submission.csv` (score likely still low in that case).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with “weights not actually being used” (random model) and/or a silent partial load where the head stays random, which makes QWK collapse. I make checkpoint loading fail-fast when a checkpoint is found but the head didn’t truly load (so you don’t unknowingly submit random predictions), while still keeping a safe fallback to a constant-class submission only when *no checkpoint exists at all* (to ensure a valid CSV in every environment). To move the score up toward your target without changing the model/TTA/core logic, I also fit thresholds on a deterministic holdout split that is stratified by diagnosis (same size as before), which usually improves QWK stability versus a purely random split. Finally, I enforce strict alignment of `test_preds` length/order to `test.csv` and guard against any missing IDs in RAM to prevent accidental row shifts.'

# 9. Code solution

## === cell 0
import os
import re
import gc
import math
import shutil
import collections
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
from tqdm import tqdm



## === cell 1
bs = 8
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 2
path = Path(".")

INPUT_ROOT = Path("../input/aptos2019-blindness-detection")
if not INPUT_ROOT.exists():
    INPUT_ROOT = Path("/kaggle/input/aptos2019-blindness-detection")
if not INPUT_ROOT.exists():
    alt = Path("/kaggle/data/aptos2019-blindness-detection")
    if alt.exists():
        INPUT_ROOT = alt

path_train = INPUT_ROOT / "train_images"
path_test = INPUT_ROOT / "test_images"

assert (INPUT_ROOT / "train.csv").exists(), f"train.csv not found under {INPUT_ROOT}"
assert (INPUT_ROOT / "test.csv").exists(), f"test.csv not found under {INPUT_ROOT}"
assert (
    INPUT_ROOT / "sample_submission.csv"
).exists(), f"sample_submission.csv not found under {INPUT_ROOT}"
assert path_train.exists(), f"train_images folder not found under {INPUT_ROOT}"
assert path_test.exists(), f"test_images folder not found under {INPUT_ROOT}"

print("Using INPUT_ROOT:", INPUT_ROOT)



## === cell 3
train = pd.read_csv(INPUT_ROOT / "train.csv")
test = pd.read_csv(INPUT_ROOT / "test.csv")  # inference set is test.csv

train["id_code"] = train["id_code"].astype(str)
test["id_code"] = test["id_code"].astype(str)

print(train.head())
print(test.head())
print("Train:", train.shape, "Test:", test.shape)




## === cell 4
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # too dark
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img
    return img




## === cell 5
IMG_SIZE = 512


def load_ben_color(path, sigmaX=0):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    if sigmaX != 0:
        image = cv2.addWeighted(
            image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128
        )
    return image




## === cell 6
IMG_SIZE = 400
Test = True


def read_image(path):
    image = load_ben_color(path, 20)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return image




## === cell 7
image_dic = {}


def load_image_on_ram(folder_path, names):
    folder_path = Path(folder_path)
    for i in tqdm(names, desc=f"Loading images from {folder_path.name}"):
        img_path = folder_path / f"{i}.png"
        image = read_image(img_path)
        image_dic[i] = image


gc.collect()



## === cell 8
load_image_on_ram(path_train, train.id_code.unique())
load_image_on_ram(path_test, test.id_code.unique())
print("Loaded images into RAM:", len(image_dic))
gc.collect()



## === cell 9
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_tensor_normalized(img_rgb_0_255: np.ndarray) -> torch.Tensor:
    x = img_rgb_0_255.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = np.transpose(x, (2, 0, 1))  # CHW
    return torch.from_numpy(x)


def tta_variants(img: np.ndarray):
    yield img
    yield np.ascontiguousarray(img[:, ::-1, :])  # hflip
    yield np.ascontiguousarray(img[::-1, :, :])  # vflip


class RetinaTestDataset(Dataset):
    def __init__(self, df):
        self.ids = df["id_code"].tolist()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        if img_id not in image_dic:
            raise KeyError(f"Image id {img_id} not found in RAM cache.")
        img = image_dic[img_id]  # HWC RGB 0..255
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img_id, img


test_ds = RetinaTestDataset(test)
test_loader = DataLoader(
    test_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

print("Test loader ready:", len(test_ds))



## === cell 10
GlobalParams = collections.namedtuple(
    "GlobalParams",
    [
        "batch_norm_momentum",
        "batch_norm_epsilon",
        "dropout_rate",
        "num_classes",
        "width_coefficient",
        "depth_coefficient",
        "depth_divisor",
        "min_depth",
        "drop_connect_rate",
        "image_size",
    ],
)

BlockArgs = collections.namedtuple(
    "BlockArgs",
    [
        "kernel_size",
        "num_repeat",
        "input_filters",
        "output_filters",
        "expand_ratio",
        "id_skip",
        "stride",
        "se_ratio",
    ],
)

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def relu_fn(x):
    return x * torch.sigmoid(x)


def round_filters(filters, global_params):
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand(
        [batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device
    )
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv2d(image_size=None):
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super().__init__(
            in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias
        )
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input


class Conv2dStaticSamePadding(nn.Conv2d):
    def __init__(
        self, in_channels, out_channels, kernel_size, image_size=None, **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d(
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            )
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return x


def efficientnet_params(model_name):
    params_dict = {
        "efficientnet-b0": (1.0, 1.0, 224, 0.2),
        "efficientnet-b1": (1.0, 1.1, 240, 0.2),
        "efficientnet-b2": (1.1, 1.2, 260, 0.3),
        "efficientnet-b3": (1.2, 1.4, 300, 0.3),
        "efficientnet-b4": (1.4, 1.8, 380, 0.4),
        "efficientnet-b5": (1.6, 2.2, 456, 0.4),
        "efficientnet-b6": (1.8, 2.6, 528, 0.5),
        "efficientnet-b7": (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    @staticmethod
    def _decode_block_string(block_string):
        assert isinstance(block_string, str)

        ops = block_string.split("_")
        options = {}
        for op in ops:
            splits = re.split(r"(\d.*)", op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        assert ("s" in options and len(options["s"]) == 1) or (
            len(options["s"]) == 2 and options["s"][0] == options["s"][1]
        )

        return BlockArgs(
            kernel_size=int(options["k"]),
            num_repeat=int(options["r"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            expand_ratio=int(options["e"]),
            id_skip=("noskip" not in block_string),
            se_ratio=float(options["se"]) if "se" in options else None,
            stride=[int(options["s"][0])],
        )

    @staticmethod
    def decode(string_list):
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args


def efficientnet(
    width_coefficient=None,
    depth_coefficient=None,
    dropout_rate=0.2,
    drop_connect_rate=0.2,
    image_size=None,
    num_classes=1000,
):
    blocks_args = [
        "r1_k3_s11_e1_i32_o16_se0.25",
        "r2_k3_s22_e6_i16_o24_se0.25",
        "r2_k5_s22_e6_i24_o40_se0.25",
        "r3_k3_s22_e6_i40_o80_se0.25",
        "r3_k5_s11_e6_i80_o112_se0.25",
        "r4_k5_s22_e6_i112_o192_se0.25",
        "r1_k3_s11_e6_i192_o320_se0.25",
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    if model_name.startswith("efficientnet"):
        w, d, s, p = efficientnet_params(model_name)
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s
        )
    else:
        raise NotImplementedError("model name is not pre-defined: %s" % model_name)
    if override_params:
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


class MBConvBlock(nn.Module):
    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (
            0 < self._block_args.se_ratio <= 1
        )
        self.id_skip = block_args.id_skip

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(
                in_channels=inp, out_channels=oup, kernel_size=1, bias=False
            )
            self._bn0 = nn.BatchNorm2d(
                num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
            )

        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup,
            out_channels=oup,
            groups=oup,
            kernel_size=k,
            stride=s,
            bias=False,
        )
        self._bn1 = nn.BatchNorm2d(
            num_features=oup, momentum=self._bn_mom, eps=self._bn_eps
        )

        if self.has_se:
            num_squeezed_channels = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio)
            )
            self._se_reduce = Conv2d(
                in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1
            )
            self._se_expand = Conv2d(
                in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1
            )

        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(
            in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False
        )
        self._bn2 = nn.BatchNorm2d(
            num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps
        )

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        input_filters, output_filters = (
            self._block_args.input_filters,
            self._block_args.output_filters,
        )
        if (
            self.id_skip
            and self._block_args.stride == 1
            and input_filters == output_filters
        ):
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x


class EfficientNet(nn.Module):
    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list)
        assert len(blocks_args) > 0
        self._global_params = global_params
        self._blocks_args = blocks_args

        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        in_channels = 3
        out_channels = round_filters(32, self._global_params)
        self._conv_stem = Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, bias=False
        )
        self._bn0 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )

        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:
            block_args = block_args._replace(
                input_filters=round_filters(
                    block_args.input_filters, self._global_params
                ),
                output_filters=round_filters(
                    block_args.output_filters, self._global_params
                ),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params),
            )

            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        in_channels = block_args.output_filters
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(
            num_features=out_channels, momentum=bn_mom, eps=bn_eps
        )

        self._dropout = self._global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

    def extract_features(self, inputs):
        x = relu_fn(self._bn0(self._conv_stem(inputs)))
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        x = relu_fn(self._bn1(self._conv_head(x)))
        return x

    def forward(self, inputs):
        x = self.extract_features(inputs)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self._dropout:
            x = F.dropout(x, p=self._dropout, training=self.training)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = EfficientNet.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        return model

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ["efficientnet_b" + str(i) for i in range(num_models)]
        if model_name.replace("-", "_") not in valid_models:
            raise ValueError("model_name should be one of: " + ", ".join(valid_models))




## === cell 11
def _find_weight_file():
    explicit = [
        Path("../input/b-3000/effi_b_30000_1.pth"),
        Path("/kaggle/input/b-3000/effi_b_30000_1.pth"),
        Path("/kaggle/input/effi-b-3000/effi_b_30000_1.pth"),
        Path("/kaggle/input/effi_b_30000_1.pth"),
        Path("/kaggle/input/effi_b_30000_1.pth.tar"),
        Path("/kaggle/input/effi_b_30000_1.bin"),
    ]
    for p in explicit:
        if p.exists():
            return p

    roots = [Path("/kaggle/input"), Path("../input"), Path("/kaggle/data")]
    patterns = [
        "**/effi_b_30000_1.pth",
        "**/effi_b_30000_1.pth.tar",
        "**/effi_b_30000_1.bin",
        "**/*efficientnet*b5*.pth",
        "**/*efficientnet*b5*.pt",
        "**/*eff*net*b5*.pth",
        "**/*eff*b5*.pth",
        "**/*b5*aptos*.pth",
        "**/*aptos*efficientnet*b5*.pth",
        "**/*blindness*efficientnet*b5*.pth",
        "**/*retina*efficientnet*b5*.pth",
    ]
    for root in roots:
        if root.exists():
            for pat in patterns:
                hits = sorted(root.glob(pat))
                if hits:
                    for h in hits:
                        if h.name == "effi_b_30000_1.pth":
                            return h
                    return hits[0]
    return None


def _unwrap_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in [
            "state_dict",
            "model_state_dict",
            "model",
            "model_state",
            "net",
            "weights",
            "ema_state_dict",
            "teacher",
            "student",
            "ema",
            "params",
            "model_ema",
            "ema_model",
            "module",
            "backbone",
        ]:
            if key in ckpt:
                v = ckpt[key]
                if isinstance(v, dict):
                    ckpt = v
                    break
                if (
                    isinstance(v, (list, tuple))
                    and len(v) > 0
                    and isinstance(v[0], dict)
                ):
                    ckpt = v[0]
                    break
    if isinstance(ckpt, dict):
        ckpt = {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in ckpt.items()
        }
    return ckpt


def _maybe_strip_prefixes(sd):
    if not isinstance(sd, dict):
        return sd
    if any(k.startswith("model.") for k in sd.keys()):
        sd = {
            k[len("model.") :] if k.startswith("model.") else k: v
            for k, v in sd.items()
        }
    if any(k.startswith("net.") for k in sd.keys()):
        sd = {k[len("net.") :] if k.startswith("net.") else k: v for k, v in sd.items()}
    if any(k.startswith("student.") for k in sd.keys()):
        sd = {
            k[len("student.") :] if k.startswith("student.") else k: v
            for k, v in sd.items()
        }
    if any(k.startswith("teacher.") for k in sd.keys()):
        sd = {
            k[len("teacher.") :] if k.startswith("teacher.") else k: v
            for k, v in sd.items()
        }
    return sd


def _maybe_remap_effnet_internal_keys(sd):
    if not isinstance(sd, dict):
        return sd
    remap = {}
    for k in sd.keys():
        nk = k
        if nk.startswith("global_params."):
            nk = "_global_params." + nk[len("global_params.") :]
        if nk.startswith("blocks_args."):
            nk = "_blocks_args." + nk[len("blocks_args.") :]
        if nk.startswith("conv_stem."):
            nk = "_conv_stem." + nk[len("conv_stem.") :]
        if nk.startswith("bn0."):
            nk = "_bn0." + nk[len("bn0.") :]
        if nk.startswith("blocks."):
            nk = "_blocks." + nk[len("blocks.") :]
        if nk.startswith("conv_head."):
            nk = "_conv_head." + nk[len("conv_head.") :]
        if nk.startswith("bn1."):
            nk = "_bn1." + nk[len("bn1.") :]
        if nk.startswith("fc."):
            nk = "_fc." + nk[len("fc.") :]
        if nk.startswith("classifier."):
            nk = "_fc." + nk[len("classifier.") :]
        if nk.startswith("head."):
            nk = "_fc." + nk[len("head.") :]
        remap[k] = nk
    if any(k != v for k, v in remap.items()):
        sd = {remap[k]: v for k, v in sd.items()}
    return sd


def _infer_num_classes_from_state(state_dict):
    if not isinstance(state_dict, dict):
        return None
    for k in list(state_dict.keys()):
        if k.endswith(("_fc.weight", "fc.weight", "classifier.weight", "head.weight")):
            w = state_dict[k]
            if hasattr(w, "shape") and len(w.shape) == 2:
                return int(w.shape[0])
    return None


def _maybe_remap_fc_keys_for_model(state_dict, model_keys):
    if not isinstance(state_dict, dict):
        return state_dict

    sd = dict(state_dict)

    if ("head.weight" in sd) and ("head.bias" in sd):
        if ("_fc.weight" in model_keys) and ("_fc.weight" not in sd):
            sd["_fc.weight"] = sd.pop("head.weight")
            sd["_fc.bias"] = sd.pop("head.bias")
        elif ("fc.weight" in model_keys) and ("fc.weight" not in sd):
            sd["fc.weight"] = sd.pop("head.weight")
            sd["fc.bias"] = sd.pop("head.bias")

    if ("classifier.weight" in sd) and ("classifier.bias" in sd):
        if (
            ("_fc.weight" in model_keys)
            and ("_fc.bias" in model_keys)
            and ("_fc.weight" not in sd)
        ):
            sd["_fc.weight"] = sd.pop("classifier.weight")
            sd["_fc.bias"] = sd.pop("classifier.bias")
        elif (
            ("fc.weight" in model_keys)
            and ("fc.bias" in model_keys)
            and ("fc.weight" not in sd)
        ):
            sd["fc.weight"] = sd.pop("classifier.weight")
            sd["fc.bias"] = sd.pop("classifier.bias")

    has_fc = ("fc.weight" in sd) and ("fc.bias" in sd)
    has__fc = ("_fc.weight" in sd) and ("_fc.bias" in sd)

    model_has__fc = ("_fc.weight" in model_keys) and ("_fc.bias" in model_keys)
    model_has_fc = ("fc.weight" in model_keys) and ("fc.bias" in model_keys)

    if has_fc and model_has__fc and (not has__fc):
        sd["_fc.weight"] = sd.pop("fc.weight")
        sd["_fc.bias"] = sd.pop("fc.bias")
        return sd
    if has__fc and model_has_fc and (not has_fc):
        sd["fc.weight"] = sd.pop("_fc.weight")
        sd["fc.bias"] = sd.pop("_fc.bias")
        return sd
    return sd


def _head_tensor_matches(model, ckpt_sd):
    msd = model.state_dict()
    for head_w_key in ["_fc.weight", "fc.weight"]:
        if head_w_key in msd and head_w_key in ckpt_sd:
            if (
                hasattr(ckpt_sd[head_w_key], "shape")
                and ckpt_sd[head_w_key].shape == msd[head_w_key].shape
            ):
                return True
    return False


def _get_model_fc_module(model):
    if hasattr(model, "_fc"):
        return model._fc
    if hasattr(model, "fc"):
        return model.fc
    return None


def _head_loaded_ok_after_load(model, head_w_before, head_b_before):
    fc = _get_model_fc_module(model)
    if fc is None:
        return False
    with torch.no_grad():
        w_now = fc.weight.detach().cpu()
        b_now = fc.bias.detach().cpu()
        changed = (not torch.allclose(w_now, head_w_before)) or (
            not torch.allclose(b_now, head_b_before)
        )
    return bool(changed)


weight_path = _find_weight_file()
ckpt_state = None
if weight_path is not None:
    raw = torch.load(str(weight_path), map_location="cpu")
    ckpt_state = _unwrap_state_dict(raw)
    ckpt_state = _maybe_strip_prefixes(ckpt_state)
    ckpt_state = _maybe_remap_effnet_internal_keys(ckpt_state)

inferred = _infer_num_classes_from_state(ckpt_state) if ckpt_state is not None else None
if inferred in (1, 5):
    num_classes = inferred
else:
    num_classes = 1 if weight_path is not None else int(train["diagnosis"].nunique())

use_regression_head = num_classes == 1

model = EfficientNet.from_pretrained("efficientnet-b5", num_classes=num_classes)

loaded_ok = False
if weight_path is None:
    print(
        "Warning: pretrained weights not found under Kaggle inputs. Will still generate submission.csv, but score will likely be low."
    )
else:
    print("Loading weights:", weight_path)
    if not isinstance(ckpt_state, dict):
        raise RuntimeError(
            "Checkpoint found but unrecognized format; refusing to run with random weights."
        )
    ckpt_state = _maybe_remap_fc_keys_for_model(
        ckpt_state, set(model.state_dict().keys())
    )

    fc_mod = _get_model_fc_module(model)
    if fc_mod is not None:
        head_w_before = fc_mod.weight.detach().cpu().clone()
        head_b_before = fc_mod.bias.detach().cpu().clone()
    else:
        head_w_before = None
        head_b_before = None

    incompatible = model.load_state_dict(ckpt_state, strict=False)

    head_shape_ok = _head_tensor_matches(model, ckpt_state)
    head_changed_ok = (
        _head_loaded_ok_after_load(model, head_w_before, head_b_before)
        if (head_shape_ok and head_w_before is not None)
        else False
    )
    loaded_ok = bool(head_shape_ok and head_changed_ok)

    print(
        "Non-strict load. Missing keys:",
        len(incompatible.missing_keys),
        "| Unexpected keys:",
        len(incompatible.unexpected_keys),
    )
    print(
        "Head shape matches:",
        head_shape_ok,
        "| Head actually updated:",
        head_changed_ok,
        "| loaded_ok:",
        loaded_ok,
    )

    if not loaded_ok:
        raise RuntimeError(
            "Checkpoint was found but head did not load correctly; stopping to avoid a near-random 0.0-scoring submission."
        )

model = model.to(device)
model.eval()

print(
    "Model output dim:",
    num_classes,
    "| regression head:",
    use_regression_head,
    "| weights_loaded_ok:",
    loaded_ok,
)




## === cell 12
def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    assert y_true.shape == y_pred.shape

    O = np.zeros((n_classes, n_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < n_classes and 0 <= b < n_classes:
            O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=n_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=n_classes).astype(np.float64)

    E = np.outer(act_hist, pred_hist)
    E = E / E.sum() * O.sum() if E.sum() > 0 else E

    W = np.zeros((n_classes, n_classes), dtype=np.float64)
    for i in range(n_classes):
        for j in range(n_classes):
            W[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    return 1.0 - num / den if den > 0 else 0.0


def apply_thresholds(preds_float, thresholds):
    t1, t2, t3, t4 = thresholds
    preds_float = np.asarray(preds_float, dtype=np.float32)
    out = np.zeros_like(preds_float, dtype=int)
    out[preds_float > t1] = 1
    out[preds_float > t2] = 2
    out[preds_float > t3] = 3
    out[preds_float > t4] = 4
    return out


def fit_thresholds_by_coordinate_descent(
    preds_float, y_true, init=(0.5, 1.5, 2.5, 3.5), n_iter=6
):
    preds_float = np.asarray(preds_float, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=int)

    thresholds = np.array(init, dtype=np.float32)

    for _ in range(n_iter):
        for k in range(4):
            base = thresholds.copy()
            lo = max(-1.0, base[k] - 1.0)
            hi = min(5.0, base[k] + 1.0)
            grid = np.linspace(lo, hi, 81, dtype=np.float32)

            best_t = base[k]
            best_score = -1e9
            for cand in grid:
                trial = base.copy()
                trial[k] = cand
                trial.sort()
                pred_cls = apply_thresholds(preds_float, trial)
                score = quadratic_weighted_kappa(y_true, pred_cls, n_classes=5)
                if score > best_score:
                    best_score = score
                    best_t = cand
            thresholds[k] = best_t
            thresholds.sort()
    return tuple(float(x) for x in thresholds)


val_frac = 0.1
rng = np.random.RandomState(42)

val_idx = []
for c in range(5):
    cls_idx = np.where(train["diagnosis"].astype(int).values == c)[0]
    rng.shuffle(cls_idx)
    n_val_c = max(1, int(round(len(cls_idx) * val_frac)))
    val_idx.append(cls_idx[:n_val_c])
val_idx = np.unique(np.concatenate(val_idx))
target_n_val = max(1, int(len(train) * val_frac))
if len(val_idx) > target_n_val:
    rng.shuffle(val_idx)
    val_idx = val_idx[:target_n_val]

val_ids = train.loc[val_idx, "id_code"].tolist()
val_y = train.loc[val_idx, "diagnosis"].astype(int).values


class RetinaValDataset(Dataset):
    def __init__(self, ids, y):
        self.ids = list(ids)
        self.y = np.asarray(y, dtype=int)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        if img_id not in image_dic:
            raise KeyError(f"Val image id {img_id} not found in RAM cache.")
        img = image_dic[img_id]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return img_id, img, int(self.y[idx])


val_ds = RetinaValDataset(val_ids, val_y)
val_loader = DataLoader(
    val_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)



## === cell 13
thresholds = (0.5, 1.5, 2.5, 3.5)

if use_regression_head and (weight_path is not None) and loaded_ok:
    val_preds_float = []
    val_true = []

    with torch.no_grad():
        for _, img, y in tqdm(
            val_loader, total=len(val_loader), desc="Calibrating thresholds (val)"
        ):
            img_np = img[0].numpy()
            logits_sum = None
            n_views = 0
            for v in tta_variants(img_np):
                x = to_tensor_normalized(v).unsqueeze(0).to(device)
                out = model(x)
                logits_sum = out if logits_sum is None else (logits_sum + out)
                n_views += 1
            out_avg = logits_sum / float(n_views)
            val_preds_float.append(float(out_avg.view(-1)[0].item()))
            val_true.append(int(y))

    thresholds = fit_thresholds_by_coordinate_descent(val_preds_float, val_true)
    val_pred_cls = apply_thresholds(val_preds_float, thresholds)
    qwk = quadratic_weighted_kappa(np.array(val_true), val_pred_cls, n_classes=5)
    print("Fitted thresholds:", thresholds, "| holdout QWK:", qwk)
else:
    print("Skipping threshold calibration. Using default thresholds:", thresholds)



## === cell 14
all_ids = []
all_preds = []

with torch.no_grad():
    for img_id, img in tqdm(test_loader, total=len(test_loader), desc="Predicting"):
        img_id = img_id[0]
        img_np = img[0].numpy()  # HWC RGB

        if use_regression_head:
            out_sum = None
            n_views = 0
            for v in tta_variants(img_np):
                x = to_tensor_normalized(v).unsqueeze(0).to(device)
                out = model(x)
                out_sum = out if out_sum is None else (out_sum + out)
                n_views += 1
            out_avg = out_sum / float(n_views)
            pred_float = float(out_avg.view(-1)[0].item())
            pred = int(apply_thresholds([pred_float], thresholds)[0])
        else:
            prob_sum = None
            n_views = 0
            for v in tta_variants(img_np):
                x = to_tensor_normalized(v).unsqueeze(0).to(device)
                out = model(x)
                prob = torch.softmax(out, dim=1)
                prob_sum = prob if prob_sum is None else (prob_sum + prob)
                n_views += 1
            prob_avg = prob_sum / float(n_views)
            pred = int(torch.argmax(prob_avg, dim=1).item())

        all_ids.append(img_id)
        all_preds.append(pred)

pred_map = dict(zip(all_ids, all_preds))
test_preds = np.array([pred_map[i] for i in test["id_code"].tolist()], dtype=int)

print(
    "Preds shape:",
    test_preds.shape,
    "min/max:",
    int(test_preds.min()),
    int(test_preds.max()),
)



## === cell 15
sub = pd.read_csv(INPUT_ROOT / "sample_submission.csv")

sub["id_code"] = test["id_code"].values
sub["diagnosis"] = test_preds.astype(int)

sub["diagnosis"] = sub["diagnosis"].clip(0, 4).astype(int)

if weight_path is None:
    sub["diagnosis"] = 0

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Saved submission.csv with shape:", sub.shape)
print("submission.csv exists:", Path("submission.csv").exists())
print("diagnosis value counts:\n", sub["diagnosis"].value_counts().sort_index())
