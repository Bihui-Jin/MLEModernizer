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

0.911371255572809

# 6. Current score

0.58066

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on `fastai` (it’s not available in your environment) and replace only the data-loading/inference parts with a minimal pure-PyTorch pipeline while preserving the core model (EfficientNet-B5 definition) and the “optimized rounding” post-processing logic. I fix missing imports (`os`, `collections`, etc.), remove notebook magics and shell commands that break a `.py` run, and ensure paths match your Kaggle filesystem (`/kaggle/data/...`). Because the provided weights file path (`../input/kaggle-public/abcdef.pth`) doesn’t exist here, I load weights only if present; otherwise the script still run end-to-end and create a valid `submission.csv`. Finally, I guarantee the submission has exactly the required columns (`id_code`, `diagnosis`) and correct row order matching `test.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with generating essentially random predictions because the script never loads any fine-tuned competition weights; it only instantiates an ImageNet-pretrained EfficientNet and uses arbitrary fixed rounding thresholds. To move the score toward your 0.911 target with minimal changes and without altering the model core, I (1) reliably locate and load a `.pth` from the competition dataset directory (common in Kaggle notebooks) and (2) if weights are present, derive rounding thresholds from the training-label distribution using an out-of-fold–free, deterministic quantile mapping (so the class proportions match train), instead of fixed `[0.5,1.5,2.5,3.5]`. If no weights are found, the script still produces a valid `submission.csv` (score likely remains low, but it run end-to-end). I also ensure the submission `id_code` order matches `test.csv` exactly (not just DataLoader order) to avoid any accidental misalignment.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the submission `id_code` order being mismatched to the predictions: you build `test_df` from `test_ids` coming out of the DataLoader, then reindex again using the official `test.csv`, which can silently mis-map predictions if any ordering/duplication differences occur. I minimally change the inference pipeline to predict in the *exact* `test.csv` order and then write `submission.csv` directly from that same `test_df`, eliminating any possibility of misalignment (this should move your score upward toward the target without changing the model). I also make image size consistent with EfficientNet-B5’s native 456 input (your current 224 harms accuracy materially), which is a small preprocessing fix and not a core-logic rewrite. Finally, I ensure the weight-loader prefers plausible EfficientNet-B5 checkpoints (by filtering out tiny/unrelated `.pth` files) to avoid accidentally loading a wrong file that degrades predictions.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model producing essentially uninformative outputs because it never reliably loads a correct fine-tuned checkpoint and also applies ImageNet dropout at inference (the EfficientNet global dropout is non-zero, and a mismatched head can leave the model near-random). I make two minimal, score-relevant fixes without changing the model definition: (1) make checkpoint loading stricter and only accept checkpoints that actually contain `_fc.weight` with output size 1 (so we don’t silently run with a random head), and (2) disable dropout at inference by setting `_dropout=0.0` after loading (evaluation semantics: deterministic inference). If no suitable checkpoint is found, the script still produce a valid submission, but it clearly warn you that the score likely remain low. Finally, I keep the exact `test.csv` ordering guarantee and still use your quantile-based thresholds.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the script producing invalid/uninformative predictions because it never actually loads a usable fine-tuned checkpoint and also never uses the ImageNet backbone weights as a fallback (so the network is randomly initialized). I make the smallest score-relevant change: load EfficientNet-B5 ImageNet pretrained weights for the backbone when no compatible competition checkpoint is found, keeping your model architecture (EfficientNet-B5 with a 1-unit regression head) and the same quantile-based rounding logic. I also ensure the FC layer remains randomly initialized (since ImageNet FC is 1000-class) and keep inference deterministic. This should move the score upward from 0.0 toward your target while preserving the core semantics and still writing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with predictions being essentially uncorrelated with the true labels (i.e., no usable fine-tuned checkpoint is being loaded, so even with ImageNet backbone the regression head is random). To move the score upward toward 0.911 with minimal core-logic change, I (1) broaden checkpoint compatibility to also accept common EfficientNet checkpoint naming (e.g., `fc.*` / `classifier.*`) and 5-class classification heads, and (2) add the smallest safe adaptation: if a 5-logit head is loaded, convert to a continuous score via expected value (softmax-weighted class index) before applying your existing optimized rounding/quantile thresholds. This preserves your EfficientNet-B5 backbone and the same post-processing approach, but makes it far more likely you actually use a valid competition-trained checkpoint if one exists in `/kaggle/input` or `/kaggle/data`. If no suitable checkpoint is found, it still write a valid `submission.csv` as before.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model producing uncorrelated predictions because no competition-trained checkpoint is actually being loaded, and the ImageNet-weight loading currently depends on downloading weights via `model_zoo.load_url` (internet is typically disabled on Kaggle). I make a minimal, score-relevant change to load EfficientNet-B5 ImageNet backbone weights from `torchvision` (offline, already installed in Kaggle) as a reliable fallback when no fine-tuned `.pth` is found, while keeping your EfficientNet-B5 architecture and the same inference + rounding logic. This should move the score up from 0.0 toward your target without changing training loops (none exist) or the post-processing semantics. I also keep the strict test order check and still write a valid `submission.csv`.'
- What this solution (achieved 0.10623) has done: 'Your 0.0 score is most consistent with the model producing essentially uncorrelated predictions because the “ImageNet fallback” currently loads only a tiny subset (stem/head) of weights, leaving almost all blocks random. To move your score upward toward the 0.911 target without changing the model architecture or inference approach, I load the full EfficientNet-B5 backbone weights offline from `torchvision` by mapping `features.*` weights into your EfficientNet implementation (excluding the classifier head). I keep the exact same test ordering, regression/expected-value prediction logic, and the same quantile-based rounding, but this stronger initialization should materially improve predictions versus near-random. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved -0.27332) has done: 'Your score is far below the target, so the most likely issue is that inference is using an ImageNet backbone with a random head (or a mismatched checkpoint), making predictions nearly uncorrelated with DR severity. With minimal change and without altering the model definition or loss/metric semantics, I (1) always build the model with `image_size=None` so it uses dynamic “same” padding (matching torchvision’s EfficientNet behavior and preventing subtle shape/padding mismatches), (2) improve the torchvision-to-custom EfficientNet weight mapping by copying *all* matching tensors by shape (not just a fragile hand-mapped subset), and (3) make test-time preprocessing match torchvision EfficientNet-B5 (center-crop style resize) rather than naive square-resize, which typically improves signal without changing the core approach. The script still run end-to-end, keep strict `test.csv` ordering, and write a valid `submission.csv`.'
- What this solution (achieved -0.17691) has done: 'Your current score is far below the target (gap ≈ -1.1847), so we should increase performance with the smallest safe changes that keep your model/inference core intact. The biggest likely issue here is domain shift from using plain ImageNet preprocessing on retinal fundus images; with no training loop available, the minimal lever is a standard, deterministic fundus “crop-to-circle” preprocessing that many APTOS solutions rely on and that typically boosts kappa a lot without changing the model. I add this crop/normalization inside `load_image_tensor` (keeping your resize+center-crop and EfficientNet-B5 input size), and I also switch PIL resize to BICUBIC (closer to torchvision) and ensure RGB conversion is robust. The rest (checkpoint loading, test order alignment, quantile rounding, submission writing) stays unchanged.'
- What this solution (achieved 0.39395) has done: 'We need to move your score up toward 0.911, and the biggest current failure mode is that you still aren’t using a competition-trained checkpoint, so predictions are essentially uncorrelated (hence negative kappa). With minimal change and keeping the same EfficientNet-B5 model and inference flow, I add an in-notebook “train the head only” calibration step: freeze the backbone, train the final `_fc` layer on the provided `train.csv` for a single short epoch using MSE to predict the ordinal label as a regression target. This is not a new training approach/architecture; it’s a minimal, deterministic fit of the existing head that should substantially improve correlation and kappa compared to a random head, while staying within the 600s limit by using a small image cache and conservative DataLoader settings. I keep your existing fundus crop + resize pipeline, your expected-value handling for 5-logit checkpoints, and your quantile-based rounding to match label distribution.'
- What this solution (achieved 0.58066) has done: 'I fix the runtime error in the head-only training step by ensuring the training DataFrame includes the required `path` column (and uses the same preprocessing pipeline as the rest of the script). This is a correctness/stability fix that unblocks execution and lets the intended “fit head only when no fine-tuned checkpoint exists” logic actually run, which should improve correlation and move the kappa score upward toward your target. I also make the environment consistent with Kaggle’s stated Python 3.7 by removing the incompatible error trace assumptions and keeping everything pure PyTorch/pandas/PIL. No model architecture, loss, or inference semantics are changed; only the data plumbing for training is corrected so the existing approach can execute end-to-end and write `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import re
import math
import json
import random
import collections
from functools import partial

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageOps

from sklearn import metrics

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT_CANDIDATES = [
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data",
    "/kaggle/input",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "test.csv")
    ):
        DATA_ROOT = p
        break
if DATA_ROOT is None:
    nested = "/kaggle/data/aptos2019-blindness-detection/aptos2019-blindness-detection"
    if os.path.exists(os.path.join(nested, "train.csv")):
        DATA_ROOT = nested
    else:
        raise FileNotFoundError(
            "Could not locate aptos2019-blindness-detection train/test CSVs under /kaggle/data or /kaggle/input."
        )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")



## === cell 1
"""
Fix: fastai is not installed in this runtime (ModuleNotFoundError). Keep core model logic,
but run pure PyTorch inference and write a valid submission.csv.
"""
pass



## === cell 2
"""
This file contains helper functions for building the model and for loading model parameters.
These helper functions are built to mirror those in the official TensorFlow implementation.
"""

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
    """Swish activation function"""
    return x * torch.sigmoid(x)


def round_filters(filters, global_params):
    """Calculate and round number of filters based on depth multiplier."""
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
    """Round number of filters based on depth multiplier."""
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """Drop connect."""
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
    """Chooses static padding if image size specified, else dynamic padding."""
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size"""

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
    """2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(
        self, in_channels, out_channels, kernel_size, image_size=None, **kwargs
    ):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        assert image_size is not None
        ih, iw = (
            image_size if isinstance(image_size, list) else [image_size, image_size]
        )
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
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=None
        )
        global_params = global_params._replace(image_size=None)
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
        stride_val = (
            self._block_args.stride[0]
            if isinstance(self._block_args.stride, (list, tuple))
            else self._block_args.stride
        )
        if self.id_skip and stride_val == 1 and input_filters == output_filters:
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
                    input_filters=block_args.output_filters, stride=[1]
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
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = EfficientNet.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        return model




## === cell 3
def load_imagenet_backbone_from_torchvision(effnet_model):
    """
    Change (score-relevant, minimal): instead of a fragile partial mapping, copy ALL tensors
    from torchvision EfficientNet-B5 into our model when (name, shape) match.
    This usually maps most of the backbone and improves predictions vs near-random.
    """
    try:
        import torchvision
        from torchvision.models import efficientnet_b5

        weights = None
        try:
            weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT
        except Exception:
            try:
                from torchvision.models import EfficientNet_B5_Weights

                weights = EfficientNet_B5_Weights.DEFAULT
            except Exception:
                weights = None

        tv_model = efficientnet_b5(weights=weights)
        tv_sd = tv_model.state_dict()
        eff_sd = effnet_model.state_dict()

        tv_by_shape = collections.defaultdict(list)
        for k, v in tv_sd.items():
            if isinstance(v, torch.Tensor):
                tv_by_shape[tuple(v.shape)].append(k)

        mapped = {}
        used_tv = set()

        for k_eff, v_eff in eff_sd.items():
            if not isinstance(v_eff, torch.Tensor):
                continue
            if (
                k_eff in tv_sd
                and isinstance(tv_sd[k_eff], torch.Tensor)
                and tv_sd[k_eff].shape == v_eff.shape
            ):
                mapped[k_eff] = tv_sd[k_eff]
                used_tv.add(k_eff)

        for k_eff, v_eff in eff_sd.items():
            if k_eff in mapped:
                continue
            if not isinstance(v_eff, torch.Tensor):
                continue
            shp = tuple(v_eff.shape)
            cand = [k for k in tv_by_shape.get(shp, []) if k not in used_tv]
            if len(cand) == 1:
                mapped[k_eff] = tv_sd[cand[0]]
                used_tv.add(cand[0])

        missing, unexpected = effnet_model.load_state_dict(mapped, strict=False)
        print(
            f"Loaded torchvision EfficientNet-B5 weights into custom model: mapped_keys={len(mapped)}"
        )
        if len(mapped) < 200:
            print("Warning: too few weights mapped; fallback may still be weak.")
        if missing:
            print(
                f"Backbone missing keys (first 5): {missing[:5]}{'...' if len(missing)>5 else ''}"
            )
        if unexpected:
            print(
                f"Backbone unexpected keys (first 5): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}"
            )
        return True
    except Exception as e:
        print(
            f"Warning: could not load torchvision EfficientNet-B5 weights due to: {repr(e)}"
        )
        return False




## === cell 4
md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)
md_ef.to(DEVICE)
md_ef.eval()


def _iter_candidate_weight_files():
    roots = [
        "/kaggle/input",
        "/kaggle/data",
        "/kaggle/working",
        DATA_ROOT,
    ]
    exts = (".pth", ".pt", ".bin")
    found = []
    for r in roots:
        if not os.path.exists(r):
            continue
        for dirpath, _, filenames in os.walk(r):
            for fn in filenames:
                if fn.lower().endswith(exts):
                    full = os.path.join(dirpath, fn)
                    try:
                        if os.path.getsize(full) < 5 * 1024 * 1024:
                            continue
                    except Exception:
                        pass
                    found.append(full)

    def _score(p):
        base = os.path.basename(p).lower()
        s = 0
        for kw in [
            "aptos",
            "blind",
            "retina",
            "retinopathy",
            "dr",
            "eff",
            "efficientnet",
            "b5",
            "kappa",
            "fold",
            "best",
            "model",
            "checkpoint",
            "stage",
        ]:
            if kw in base:
                s += 1
        try:
            s += int(min(os.path.getsize(p) / (1024 * 1024), 500) // 25)
        except Exception:
            pass
        return s

    found = sorted(set(found), key=_score, reverse=True)
    for p in found:
        yield p


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return ckpt_obj[k]
        if any(isinstance(v, torch.Tensor) for v in ckpt_obj.values()):
            return ckpt_obj
    return None


def _strip_module_prefix(sd):
    if sd is None:
        return None
    if any(k.startswith("module.") for k in sd.keys()):
        return {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _head_shapes_from_state_dict(sd):
    if sd is None:
        return None
    candidates = [
        ("_fc.weight", "_fc.bias"),
        ("fc.weight", "fc.bias"),
        ("classifier.weight", "classifier.bias"),
    ]
    for w_key, b_key in candidates:
        w = sd.get(w_key, None)
        b = sd.get(b_key, None)
        if isinstance(w, torch.Tensor) and isinstance(b, torch.Tensor) and w.ndim == 2:
            return {
                "w_key": w_key,
                "b_key": b_key,
                "out": int(w.shape[0]),
                "in": int(w.shape[1]),
            }
    return None


def _looks_like_effnet_backbone(sd):
    if sd is None:
        return False
    keys = list(sd.keys())
    must_have_any = [
        "_conv_stem.weight",
        "_bn0.weight",
        "_blocks.0._depthwise_conv.weight",
        "_conv_head.weight",
    ]
    return any(k in sd for k in must_have_any)


def try_load_best_weights(model):
    hardcoded = [
        "/kaggle/input/kaggle-public/abcdef.pth",
        "/kaggle/input/abcdef.pth",
        "/kaggle/data/abcdef.pth",
        os.path.join("/kaggle/working", "models", "abcdef.pth"),
        os.path.join("/kaggle/working", "abcdef.pth"),
    ]
    candidates = [p for p in hardcoded if os.path.exists(p)]
    candidates += [p for p in _iter_candidate_weight_files() if p not in candidates]

    for p in candidates:
        try:
            ckpt = torch.load(p, map_location="cpu")
            sd = _strip_module_prefix(_extract_state_dict(ckpt))
            if sd is None:
                continue
            if not _looks_like_effnet_backbone(sd):
                continue

            head = _head_shapes_from_state_dict(sd)
            if head is None:
                continue

            if head["out"] not in (1, 5):
                continue

            tmp = EfficientNet.from_pretrained(
                "efficientnet-b5", num_classes=head["out"]
            ).to("cpu")
            missing, unexpected = tmp.load_state_dict(sd, strict=False)

            if (
                head["w_key"] != "_fc.weight"
                and (head["w_key"] in sd)
                and ("_fc.weight" not in sd)
            ):
                sd2 = dict(sd)
                sd2["_fc.weight"] = sd2.pop(head["w_key"])
                sd2["_fc.bias"] = sd2.pop(head["b_key"])
                tmp2 = EfficientNet.from_pretrained(
                    "efficientnet-b5", num_classes=head["out"]
                ).to("cpu")
                missing, unexpected = tmp2.load_state_dict(sd2, strict=False)
                tmp = tmp2

            print(f"Loaded fine-tuned weights: {p} (head_out={head['out']})")
            if missing:
                print(
                    f"Missing keys (first 5): {missing[:5]}{'...' if len(missing)>5 else ''}"
                )
            if unexpected:
                print(
                    f"Unexpected keys (first 5): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}"
                )

            return True, p, head["out"], tmp.state_dict()
        except Exception:
            continue

    print(
        "Warning: no compatible fine-tuned EfficientNet checkpoint found (head out=1 or 5)."
    )
    print(
        "         Will fall back to offline torchvision ImageNet EfficientNet-B5 backbone weights."
    )
    return False, None, 1, None


loaded_ok, WEIGHTS_PATH, HEAD_OUT, LOADED_STATE = try_load_best_weights(md_ef)

if loaded_ok:
    if HEAD_OUT == 5:
        md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=5).to(
            DEVICE
        )
    else:
        md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1).to(
            DEVICE
        )
    md_ef.load_state_dict(LOADED_STATE, strict=False)
else:
    ok = load_imagenet_backbone_from_torchvision(md_ef)
    if not ok:
        print(
            "Warning: proceeding with randomly initialized weights (score likely very low)."
        )

md_ef._dropout = 0.0
md_ef.eval()



## === cell 5
os.makedirs("models", exist_ok=True)




## === cell 6
def get_df():
    df = pd.read_csv(TRAIN_CSV)
    df["path"] = df["id_code"].map(lambda x: os.path.join(TRAIN_IMG_DIR, f"{x}.png"))
    df = df.drop(columns=["id_code"])
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    test_df = pd.read_csv(TEST_CSV)
    return df, test_df


df, test_df = get_df()



## === cell 7
SZ = 456
BATCH_SIZE = 8  # keep safe memory with 456px on EfficientNet-B5

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _resize_shorter_side_then_center_crop(img, size):
    """
    Change (score-relevant, minimal): torchvision EfficientNet uses a resize+center-crop
    style preprocessing; this tends to outperform naive square-resize for natural images.
    """
    w, h = img.size
    if min(w, h) == size:
        left = max(0, (w - size) // 2)
        top = max(0, (h - size) // 2)
        return img.crop((left, top, left + size, top + size))

    scale = size / float(min(w, h))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    img = img.resize((new_w, new_h), resample=Image.BICUBIC)
    left = max(0, (new_w - size) // 2)
    top = max(0, (new_h - size) // 2)
    return img.crop((left, top, left + size, top + size))


def _fundus_crop_black(img, tol=7):
    """
    Change (score-relevant, minimal): APTOS images have large black borders; removing them
    (deterministically) reduces domain shift and usually improves kappa without changing the model.
    """
    arr = np.asarray(img)
    if arr.ndim == 3:
        gray = arr[..., 0] * 0.299 + arr[..., 1] * 0.587 + arr[..., 2] * 0.114
    else:
        gray = arr.astype(np.float32)

    mask = gray > tol
    if not mask.any():
        return img

    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    x0 = max(int(x0) - 1, 0)
    y0 = max(int(y0) - 1, 0)
    x1 = min(int(x1) + 1, arr.shape[1])
    y1 = min(int(y1) + 1, arr.shape[0])
    return img.crop((x0, y0, x1, y1))


def load_image_tensor(path, size=224):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    img = _fundus_crop_black(img, tol=7)
    img = _resize_shorter_side_then_center_crop(img, size=size)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.from_numpy(arr)


class TestImageDataset(Dataset):
    def __init__(self, df, img_dir, size=224):
        self.ids = df["id_code"].tolist()
        self.img_dir = img_dir
        self.size = size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_code = self.ids[idx]
        path = os.path.join(self.img_dir, f"{id_code}.png")
        x = load_image_tensor(path, size=self.size)
        return id_code, x


class TrainImageDataset(Dataset):
    def __init__(self, train_df, size=224):
        self.paths = train_df["path"].tolist()
        self.y = train_df["diagnosis"].astype(np.float32).values
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = load_image_tensor(self.paths[idx], size=self.size)
        y = self.y[idx]
        return x, y




## === cell 8
test_ds = TestImageDataset(test_df, TEST_IMG_DIR, size=SZ)
test_dl = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 9
def qk_numpy(y_pred, y_true):
    y_pred = np.clip(np.round(y_pred).astype(int), 0, 4)
    y_true = np.clip(np.round(y_true).astype(int), 0, 4)
    return metrics.cohen_kappa_score(y_true, y_pred, weights="quadratic")




## === cell 10
def _freeze_backbone_train_head_only(model):
    for n, p in model.named_parameters():
        p.requires_grad = False
    for p in model._fc.parameters():
        p.requires_grad = True


def _fit_head_only_regression(model, train_df, size, batch_size, max_images_cache=512):
    model.train()
    _freeze_backbone_train_head_only(model)

    def _set_bn_eval(m):
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

    model.apply(_set_bn_eval)

    ds = TrainImageDataset(train_df, size=size)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    opt = torch.optim.Adam(model._fc.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    n_seen = 0
    running = 0.0
    for xb, yb in dl:
        xb = xb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True).view(-1, 1)

        opt.zero_grad(set_to_none=True)
        out = model(xb).float()
        if out.ndim == 2 and out.shape[1] != 1:
            raise RuntimeError(
                "Head-only training is intended for num_classes=1 models."
            )
        loss = loss_fn(out, yb)
        loss.backward()
        opt.step()

        running += float(loss.detach().cpu())
        n_seen += int(xb.shape[0])

    model.eval()
    print(
        f"Head-only fit done on {n_seen} images; mean_mse={running/max(1.0, len(dl)):.4f}"
    )


if not loaded_ok:
    md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1).to(DEVICE)
    ok = load_imagenet_backbone_from_torchvision(md_ef)
    if not ok:
        print(
            "Warning: could not load torchvision weights for head-fit; proceeding anyway."
        )
    md_ef._dropout = 0.0

    train_df_for_head = pd.read_csv(TRAIN_CSV)
    train_df_for_head["path"] = train_df_for_head["id_code"].map(
        lambda x: os.path.join(TRAIN_IMG_DIR, f"{x}.png")
    )
    train_df_for_head = train_df_for_head[["path", "diagnosis"]].copy()

    _fit_head_only_regression(md_ef, train_df_for_head, size=SZ, batch_size=BATCH_SIZE)




## === cell 11
@torch.no_grad()
def predict_test(model, dl):
    model.eval()
    ids_all = []
    preds_all = []
    for ids, xb in dl:
        xb = xb.to(DEVICE, non_blocking=True)
        out = model(xb).float()
        if out.ndim == 2 and out.shape[1] == 5:
            probs = torch.softmax(out, dim=1)
            cls = torch.arange(5, device=probs.device, dtype=probs.dtype).view(1, -1)
            out = (probs * cls).sum(dim=1)  # (bs,)
        else:
            out = out.squeeze(1)  # (bs,)
        ids_all.extend(list(ids))
        preds_all.append(out.detach().cpu().numpy())
    preds_all = np.concatenate(preds_all, axis=0)
    return ids_all, preds_all


test_ids, test_preds = predict_test(md_ef, test_dl)

if list(test_df["id_code"].values) != list(test_ids):
    raise RuntimeError(
        "Prediction id order does not match test.csv order; refusing to write a misaligned submission."
    )



## === cell 12
pass



## === cell 13
pass




## === cell 14
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        ll = metrics.cohen_kappa_score(y, X_p, weights="quadratic")
        return -ll

    def fit(self, X, y):
        raise RuntimeError(
            "scipy.optimize not available in this environment; provide coefficients explicitly."
        )

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 0
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 1
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 2
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 3
            else:
                X_p[i] = 4
        return X_p

    def coefficients(self):
        return self.coef_["x"]




## === cell 15
def _quantile_thresholds_from_train_labels(train_df, raw_preds, eps=1e-6):
    y = train_df["diagnosis"].values
    counts = np.bincount(y.astype(int), minlength=5).astype(np.float64)
    cum = np.cumsum(counts) / counts.sum()
    qs = [cum[0], cum[1], cum[2], cum[3]]  # quantiles for thresholds
    coef = [float(np.quantile(raw_preds, q)) for q in qs]
    for i in range(1, 4):
        if coef[i] <= coef[i - 1]:
            coef[i] = coef[i - 1] + eps
    return coef


def run_subm(test_df, raw_preds, coefficients=None, out_path="submission.csv"):
    raw_preds = np.asarray(raw_preds).reshape(-1)

    if coefficients is None:
        coefficients = _quantile_thresholds_from_train_labels(
            pd.read_csv(TRAIN_CSV), raw_preds
        )

    opt = OptimizedRounder()
    tst_pred = opt.predict(raw_preds, coefficients).astype(int)
    tst_pred = np.clip(tst_pred, 0, 4)

    subm = test_df.copy()
    subm["diagnosis"] = tst_pred
    subm = subm[["id_code", "diagnosis"]]
    subm.to_csv(out_path, index=False)
    print(f"wrote {out_path} with shape {subm.shape}; coefficients={coefficients}")
    return subm


_ = run_subm(
    test_df=test_df,
    raw_preds=test_preds,
    coefficients=None,
    out_path="submission.csv",
)
