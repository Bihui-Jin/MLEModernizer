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

3.8

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

0.9080228716816332

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the broken `fastai` dependency (it’s not available in your environment) and replace it with a minimal PyTorch-only inference pipeline that preserves the core idea: EfficientNet-B5 model + loading a provided `.pth` checkpoint + predicting on test images and writing `submission.csv`. I also fix missing imports (`os`, `collections`, etc.), remove notebook magics/`!cp` commands that fail in script mode, and ensure paths resolve against the Kaggle dataset directory you listed. Finally, I keep the existing “optimized rounding with fixed coefficients” post-processing (score-neutral vs your intent) but make it work with numpy arrays and guaranteed integer outputs in `[0,4]`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with the model running on random weights because the checkpoint (`abcdef.pth`) is not found; the smallest meaningful improvement is to correctly locate and load the actual pretrained `.pth` from the dataset folders you listed. I add a robust checkpoint search that scans common Kaggle input locations under the APTOS dataset for any `.pth` file and loads it (while keeping the same EfficientNet-B5 architecture and inference logic). I also make sure the test image path resolves correctly even if the folder nesting differs, because missing images or wrong paths can silently break predictions/ordering. These changes should move the score sharply upward toward your target without changing the model core or the metric semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the model effectively predicting nonsense (most likely because no real checkpoint is being loaded, or the loaded checkpoint’s head doesn’t match `num_classes=1`). I make the smallest changes that preserve your core logic (EfficientNet-B5 regression + fixed optimized-rounding) but ensure we actually locate and correctly load an APTOS `.pth/.pt` checkpoint, including common cases where the checkpoint was trained with `num_classes=5` and needs head adaptation. I also make the test image directory resolution more robust and add a hard check that all test image files exist (to avoid silent failures/misalignment). These fixes should move the score sharply upward toward your target without changing the inference approach or post-processing semantics.'
- What this solution (achieved 0.0) has done: 'I remove the hard failure when no `.pth/.pt` checkpoint is found and instead fall back to a deterministic, valid baseline predictor so the notebook always runs end-to-end and writes a proper `submission.csv`. This fixes the immediate runtime errors (`FileNotFoundError` in cell 2 and cascading `NameError`s in later cells) while keeping your core EfficientNet-B5 + inference + optimized-rounding pipeline intact when a checkpoint is available. I also define `USE_EXPECTED_VALUE_FROM_LOGITS` safely in all cases, and add a small safety check around `torch.load` so a bad/unsupported checkpoint doesn’t crash the run. If a usable checkpoint is present, behavior stays the same (model loads and predicts); if not, you still get a valid submission (score be low, but not “Not yielded”).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the submission is valid-format but predictions are essentially meaningless (most likely because no real checkpoint is being loaded, or because the loaded checkpoint’s head type doesn’t match how we interpret outputs). I make minimal changes to (1) more reliably find and load a real APTOS `.pth/.pt` checkpoint, (2) correctly handle common checkpoint output heads (5-class logits vs 1-d regression) without changing the EfficientNet-B5 core, and (3) align the post-processing thresholds to the model’s output type (expected value vs raw regression) so the final integers are sensible. These changes should move the score upward toward your target while keeping the architecture and inference pipeline intact. The script still always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with the pipeline producing a valid CSV but using either no checkpoint (all-zero predictions) or a mismatched checkpoint head (e.g., trained as 5-class but loaded into 1-class), yielding near-random labels after rounding. I make the smallest changes to (1) deterministically locate the most likely APTOS checkpoint, (2) robustly load common checkpoint formats and automatically pick the correct prediction interpretation (regression vs 5-class logits), and (3) replace the hardcoded rounding cutoffs with a minimal, data-driven calibration using the training label distribution (no extra training, just monotonic quantile binning) which usually boosts QWK substantially versus fixed thresholds. These changes preserve your core EfficientNet-B5 + checkpoint inference + rounding-to-0..4 semantics, but should move the score sharply upward toward your target band. The script still always writes a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a “valid CSV but essentially random/constant predictions”, which in this script happens when no real checkpoint is loaded (or the loaded weights don’t match the head and you silently get a mostly-untrained head). I make minimal, score-relevant fixes to (1) more reliably locate a real checkpoint by prioritizing `.pth/.pt` files inside the APTOS dataset folder first, (2) load checkpoints that store a full `nn.Module` (not just a state_dict) without changing the model architecture, and (3) when the head is mismatched/missing, derive sensible class cutoffs from the *training label distribution* (quantiles) so the final discrete labels aren’t collapsed. These changes preserve your EfficientNet-B5 inference + rounding semantics, but should move the score sharply upward toward your target band if any usable checkpoint exists. The script still produces `submission.csv` end-to-end even if no checkpoint is found.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with “predictions are essentially constant/meaningless”, which in this script happens when no real checkpoint is actually loaded (or a mismatched head leaves you with near-random outputs). I make a minimal, score-relevant improvement by (1) fixing a bug in the EfficientNet block stride handling that can silently distort shapes/behavior, and (2) tightening checkpoint discovery to prioritize APTOS-like model files and correctly handle common checkpoint dict formats. These changes keep your core approach (EfficientNet-B5 inference + rounding to 0..4) intact, but should materially increase QWK if any usable `.pth/.pt` exists. The script still always run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because the script is not loading a real trained checkpoint, so predictions collapse (often near-constant) and kappa becomes ~0. I make the smallest score-relevant changes to (1) broaden and prioritize checkpoint discovery (including smaller but valid `.pth/.pt` files), (2) load common checkpoint formats more robustly and deterministically decide whether outputs are 5-class logits or 1D regression, and (3) make the rounding coefficients come from the *training label distribution* in a safer way (compute quantiles on *test predictions*, but using the train class proportions) while keeping the same “continuous -> thresholds -> {0..4}” core logic. These changes should move the score upward toward your target without changing the model architecture or adding any training. The script still always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is most consistent with either (a) no real trained checkpoint being loaded, or (b) a checkpoint being loaded but not actually applied to the model due to key mismatches—leading to near-constant/random predictions after rounding. I make a minimal, score-relevant change to checkpoint discovery: prefer `.pth/.pt` files that look like actual model weights (contain `_fc.weight`/`classifier.weight`, etc.) rather than just “any big file”, and explicitly avoid picking unrelated `.pt` artifacts. I also make loading stricter in the sense of detecting when too few parameters matched (so we can fall back to a more compatible head interpretation: 5-class vs 1-d regression) without changing the core EfficientNet-B5 inference+rounding approach. These changes should materially increase QWK toward your target if a usable checkpoint exists in the dataset folders, while still always writing a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'The current 0.0 score is most consistent with the model never loading real trained weights (or loading an incompatible file), causing near-constant/garbage predictions after rounding. I make the smallest score-relevant change by tightening checkpoint discovery to only accept `.pth/.pt` files that actually *contain EfficientNet-like parameter keys*, then load them with a slightly more robust prefix-stripping routine so the state_dict matches your model. I also add a hard assertion that we successfully loaded a meaningful fraction of weights; if not, we keep the existing safe baseline submission (so the script always finishes) but you immediately see why the score would remain low. These changes preserve your core logic: EfficientNet-B5 inference + continuous prediction + threshold rounding to {0..4} and CSV output.'

# 9. Code solution

## === cell 0
import os
import math
import re
import json
import collections
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from sklearn import metrics

torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR_CANDIDATES = [
    Path("/kaggle/input/aptos2019-blindness-detection"),
    Path("../input/aptos2019-blindness-detection"),
    Path("/kaggle/data/aptos2019-blindness-detection"),
]
BASE_DIR = next((p for p in BASE_DIR_CANDIDATES if p.exists()), None)
if BASE_DIR is None:
    raise FileNotFoundError(
        f"Could not find dataset dir in candidates: {BASE_DIR_CANDIDATES}"
    )

TRAIN_CSV = BASE_DIR / "train.csv"
TEST_CSV = BASE_DIR / "test.csv"
SAMPLE_SUB_CSV = BASE_DIR / "sample_submission.csv"
TRAIN_IMG_DIR = BASE_DIR / "train_images"
TEST_IMG_DIR = BASE_DIR / "test_images"

print("BASE_DIR:", BASE_DIR)
print("CUDA available:", torch.cuda.is_available())



## === cell 1
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
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
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
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise."""
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
        pad_w = max((ow - 1) * self.stride[1] + (kh - 1) * self.dilation[0] + 1 - iw, 0)
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
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kh - 1) * self.dilation[0] + 1 - iw, 0)
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
    """Map EfficientNet model name to parameter coefficients."""
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
    """Block Decoder for readability, straight from the official TensorFlow repository"""

    @staticmethod
    def _decode_block_string(block_string):
        """Gets a block through a string notation of arguments."""
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

        stride = (
            [int(options["s"][0]), int(options["s"][1])]
            if len(options["s"]) == 2
            else [int(options["s"][0]), int(options["s"][0])]
        )

        return BlockArgs(
            kernel_size=int(options["k"]),
            num_repeat=int(options["r"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            expand_ratio=int(options["e"]),
            id_skip=("noskip" not in block_string),
            se_ratio=float(options["se"]) if "se" in options else None,
            stride=stride,
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
    """Creates a efficientnet model."""
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
    """Get the block args and global params for a given model"""
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
            and self._block_args.stride == [1, 1]
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
                    input_filters=block_args.output_filters, stride=[1, 1]
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




## === cell 2
def unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for k in [
            "state_dict",
            "model",
            "model_state_dict",
            "net",
            "network",
            "ema_state_dict",
        ]:
            if k in obj and isinstance(obj[k], dict):
                obj = obj[k]
                break

    if isinstance(obj, dict):
        prefixes = ["module.", "model.", "net.", "encoder.", "backbone."]
        for pref in prefixes:
            if any(k.startswith(pref) for k in obj.keys()):
                obj = {k.replace(pref, "", 1): v for k, v in obj.items()}
    return obj


def infer_head_from_state(state):
    if not isinstance(state, dict):
        return None, None
    for w_key, b_key in [
        ("_fc.weight", "_fc.bias"),
        ("fc.weight", "fc.bias"),
        ("classifier.weight", "classifier.bias"),
    ]:
        if (
            w_key in state
            and hasattr(state[w_key], "shape")
            and len(state[w_key].shape) == 2
        ):
            out_dim = int(state[w_key].shape[0])
            return out_dim, (w_key, b_key if b_key in state else None)
    return None, None


def load_checkpoint_flex(path: Path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, nn.Module):
        return obj, None, None
    state = unwrap_state_dict(obj)
    loaded_num_classes, head_keys = infer_head_from_state(state)
    return state, loaded_num_classes, head_keys


def _state_keys_look_efficientnet(state: dict) -> bool:
    if not isinstance(state, dict) or len(state) == 0:
        return False
    keys = set(state.keys())
    must_have_any = [
        "_conv_stem.weight",
        "_blocks.0._depthwise_conv.weight",
        "_bn0.weight",
        "_conv_head.weight",
        "_fc.weight",
    ]
    return any(k in keys for k in must_have_any)


def _checkpoint_looks_like_weights(p: Path) -> bool:
    try:
        obj = torch.load(p, map_location="cpu")
    except Exception:
        return False
    if isinstance(obj, nn.Module):
        return True
    state = unwrap_state_dict(obj)
    if not isinstance(state, dict):
        return False
    out_dim, _ = infer_head_from_state(state)
    if out_dim in (1, 5) and _state_keys_look_efficientnet(state):
        return True
    if _state_keys_look_efficientnet(state):
        return True
    return False


def find_checkpoint():
    preferred_roots = [
        BASE_DIR,
        BASE_DIR / "aptos2019-blindness-detection",
        Path("/kaggle/input/aptos2019-blindness-detection"),
        Path("/kaggle/input"),
        Path("/kaggle/data"),
        Path("/kaggle/working"),
    ]
    search_roots = [p for p in preferred_roots if p.exists()]

    exts = {".pth", ".pt"}
    found = []
    for root in search_roots:
        try:
            for pat in ["*.pth", "*.pt", "**/*.pth", "**/*.pt"]:
                for p in root.glob(pat):
                    if p.is_file() and p.suffix.lower() in exts:
                        if p.stat().st_size >= 50_000:
                            found.append(p)
        except Exception:
            pass

    found = list(dict.fromkeys(found))
    if not found:
        return None

    key_words = (
        "aptos",
        "blind",
        "retina",
        "efficientnet",
        "b5",
        "kappa",
        "best",
        "fold",
        "checkpoint",
        "model",
        "weights",
    )

    def score_path(p: Path):
        s = p.as_posix().lower()
        kw_hits = sum(1 for k in key_words if k in s)
        in_base = int(str(BASE_DIR.as_posix()).lower() in s)
        looks_like_weights = int(_checkpoint_looks_like_weights(p))
        return (-looks_like_weights, -kw_hits, -in_base, -p.stat().st_size, len(s))

    found_sorted = sorted(found, key=score_path)
    best = found_sorted[0]
    if not _checkpoint_looks_like_weights(best):
        return None
    return best


ckpt_path = find_checkpoint()
print("Checkpoint selected:", ckpt_path)

loaded_num_classes = None
state = None
head_keys = None
loaded_module = None

if ckpt_path is not None:
    try:
        maybe_state, loaded_num_classes, head_keys = load_checkpoint_flex(ckpt_path)
        if isinstance(maybe_state, nn.Module):
            loaded_module = maybe_state
            state = None
            print("Loaded checkpoint as nn.Module:", type(loaded_module))
        else:
            state = maybe_state
            print(
                "Inferred checkpoint head classes:",
                loaded_num_classes,
                "head_keys:",
                head_keys,
            )
    except Exception as e:
        print(f"WARNING: Failed to load checkpoint {ckpt_path}: {repr(e)}")
        state = None
        loaded_num_classes = None
        head_keys = None
        loaded_module = None

if state is not None and loaded_num_classes is None:
    loaded_num_classes = 1
    print("Could not infer head classes; defaulting to num_classes=1 (regression).")

if loaded_module is not None:
    md_ef = loaded_module.to(DEVICE)
    md_ef.eval()
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, 456, 456, device=DEVICE)
        out = md_ef(dummy)
        USE_EXPECTED_VALUE_FROM_LOGITS = bool(out.ndim == 2 and out.shape[1] == 5)
    print("USE_EXPECTED_VALUE_FROM_LOGITS:", USE_EXPECTED_VALUE_FROM_LOGITS)
elif state is not None:
    if loaded_num_classes in (1, 5):
        md_ef = EfficientNet.from_pretrained(
            "efficientnet-b5", num_classes=loaded_num_classes
        ).to(DEVICE)
    else:
        md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1).to(
            DEVICE
        )

    incompatible = md_ef.load_state_dict(state, strict=False)
    missing, unexpected = incompatible.missing_keys, incompatible.unexpected_keys
    print(f"Loaded checkpoint weights from: {ckpt_path}")
    if missing:
        print(f"Missing keys (first 10): {missing[:10]}")
    if unexpected:
        print(f"Unexpected keys (first 10): {unexpected[:10]}")

    total_keys = len(md_ef.state_dict().keys())
    loaded_keys = total_keys - len(missing)
    load_ratio = loaded_keys / max(1, total_keys)
    print(f"Approx load ratio: {load_ratio:.3f} (loaded {loaded_keys}/{total_keys})")

    if load_ratio < 0.50 and loaded_num_classes in (1, 5):
        alt = 5 if loaded_num_classes == 1 else 1
        print(
            f"Low load ratio; retrying model head num_classes={alt} for compatibility."
        )
        md_alt = EfficientNet.from_pretrained("efficientnet-b5", num_classes=alt).to(
            DEVICE
        )
        incompatible2 = md_alt.load_state_dict(state, strict=False)
        missing2 = incompatible2.missing_keys
        total_keys2 = len(md_alt.state_dict().keys())
        loaded_keys2 = total_keys2 - len(missing2)
        load_ratio2 = loaded_keys2 / max(1, total_keys2)
        print(
            f"Retry load ratio: {load_ratio2:.3f} (loaded {loaded_keys2}/{total_keys2})"
        )
        if load_ratio2 > load_ratio:
            md_ef = md_alt
            loaded_num_classes = alt
            load_ratio = load_ratio2

    if load_ratio < 0.50:
        print(
            "WARNING: Load ratio still low (<0.50). Predictions may be near-random; "
            "if you expect a high score, ensure the dataset actually includes the trained checkpoint."
        )

    md_ef.eval()

    with torch.inference_mode():
        dummy = torch.zeros(1, 3, 456, 456, device=DEVICE)
        out = md_ef(dummy)
        USE_EXPECTED_VALUE_FROM_LOGITS = bool(out.ndim == 2 and out.shape[1] == 5)
    print("USE_EXPECTED_VALUE_FROM_LOGITS:", USE_EXPECTED_VALUE_FROM_LOGITS)
else:
    md_ef = None
    loaded_num_classes = None
    USE_EXPECTED_VALUE_FROM_LOGITS = False
    print(
        "WARNING: No suitable checkpoint found. Will generate a valid submission using a baseline predictor."
    )




## === cell 3
def resolve_test_img_dir():
    candidates = [
        TEST_IMG_DIR,
        BASE_DIR / "aptos2019-blindness-detection" / "test_images",
        BASE_DIR / "test_images" / "test_images",
        Path("/kaggle/input/aptos2019-blindness-detection/test_images"),
        Path(
            "/kaggle/input/aptos2019-blindness-detection/aptos2019-blindness-detection/test_images"
        ),
    ]
    for p in candidates:
        if p.exists() and any(p.glob("*.png")):
            return p
    return TEST_IMG_DIR


def get_df():
    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    test_img_dir = resolve_test_img_dir()
    df_test = df_test.copy()
    df_test["path"] = df_test["id_code"].map(lambda x: str(test_img_dir / f"{x}.png"))

    missing = [p for p in df_test["path"].values if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Some test image files were not found. Example missing path: "
            + str(missing[0])
            + "\nResolved test_img_dir="
            + str(test_img_dir)
        )

    return df_train, df_test


train_df, test_df = get_df()
assert "id_code" in test_df.columns and "path" in test_df.columns
print(train_df.shape, test_df.shape)
print("Example test path:", test_df["path"].iloc[0])



## === cell 4
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

IMG_SIZE = 456
BATCH_SIZE = 16  # safe for GPU memory; inference only


def load_image_as_tensor(path, img_size=IMG_SIZE):
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(arr)


class RetinopathyTestDataset(Dataset):
    def __init__(self, df):
        self.ids = df["id_code"].values
        self.paths = df["path"].values

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = load_image_as_tensor(self.paths[idx])
        return self.ids[idx], x


test_ds = RetinopathyTestDataset(test_df)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)




## === cell 5
@torch.inference_mode()
def predict_continuous(model, loader, use_expected_value_from_logits=False):
    model.eval()
    all_ids = []
    all_preds = []
    for ids, x in loader:
        x = x.to(DEVICE, non_blocking=True)
        out = model(x)

        if use_expected_value_from_logits:
            prob = torch.softmax(out, dim=1)  # (bs,5)
            idx = torch.arange(
                prob.shape[1], device=prob.device, dtype=prob.dtype
            ).unsqueeze(0)
            p = (prob * idx).sum(dim=1)  # expected value in [0,4]
        else:
            if out.ndim == 2 and out.shape[1] == 1:
                p = out.squeeze(1)
            elif out.ndim == 2 and out.shape[1] == 5:
                prob = torch.softmax(out, dim=1)
                idx = torch.arange(
                    prob.shape[1], device=prob.device, dtype=prob.dtype
                ).unsqueeze(0)
                p = (prob * idx).sum(dim=1)
            else:
                p = out.reshape(out.shape[0], -1)[:, 0]

        all_ids.extend(list(ids))
        all_preds.append(p.detach().float().cpu().numpy())
    return np.array(all_ids), np.concatenate(all_preds, axis=0)


if md_ef is not None:
    test_ids, test_pred_cont = predict_continuous(
        md_ef,
        test_loader,
        use_expected_value_from_logits=USE_EXPECTED_VALUE_FROM_LOGITS,
    )
else:
    test_ids = test_df["id_code"].values
    test_pred_cont = np.zeros(len(test_ids), dtype=np.float32)

print(
    test_ids.shape,
    test_pred_cont.shape,
    "pred stats:",
    float(np.min(test_pred_cont)),
    float(np.max(test_pred_cont)),
)




## === cell 6
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

    def predict(self, X, coef):
        X = np.asarray(X).reshape(-1)
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




## === cell 7
def _compute_quantile_coefficients(pred_cont, train_labels):
    y = np.asarray(train_labels).astype(int)
    pred = np.asarray(pred_cont).reshape(-1)

    counts = np.bincount(y, minlength=5).astype(np.float64)
    probs = counts / counts.sum()
    cum = np.cumsum(probs)  # ends at 1.0

    qs = cum[:4]
    qs = np.clip(qs, 1e-6, 1 - 1e-6)

    coef = np.quantile(pred, qs).astype(np.float64).tolist()

    for i in range(1, len(coef)):
        if coef[i] <= coef[i - 1]:
            coef[i] = coef[i - 1] + 1e-3

    if (max(coef) - min(coef)) < 1e-6:
        coef = [0.5, 1.5, 2.5, 3.5]

    return tuple(float(c) for c in coef)


def run_subm(
    test_ids,
    test_pred_cont,
    coefficients=None,
    out_path="submission.csv",
):
    if coefficients is None:
        if np.std(test_pred_cont) > 1e-8:
            coefficients = _compute_quantile_coefficients(
                test_pred_cont, train_df["diagnosis"].values
            )
            print("Using quantile-calibrated coefficients:", coefficients)
        else:
            coefficients = (0.5, 1.5, 2.5, 3.5)
            print("Using fallback fixed coefficients:", coefficients)

    opt = OptimizedRounder()
    diag = opt.predict(test_pred_cont, coefficients).astype(np.int64)
    diag = np.clip(diag, 0, 4)

    subm = pd.DataFrame({"id_code": test_ids, "diagnosis": diag})
    order = pd.read_csv(TEST_CSV)["id_code"].values
    subm = subm.set_index("id_code").loc[order].reset_index()

    assert subm.shape[0] == len(order)
    assert list(subm.columns) == ["id_code", "diagnosis"]

    subm.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with shape={subm.shape} and columns={list(subm.columns)}")
    print(
        "Diagnosis value counts:",
        subm["diagnosis"].value_counts().sort_index().to_dict(),
    )
    return subm


_ = run_subm(
    test_ids,
    test_pred_cont,
    coefficients=None,
    out_path="submission.csv",
)
