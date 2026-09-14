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

0.9114099378652444

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The main blocker is that this notebook relies on fastai v1 symbols (`fastai.vision`, `Learner`, `ImageList`, `get_transforms`, etc.) that are not available in your environment, so the pipeline never creates `learn` and never writes `submission.csv`. To keep the core EfficientNet model logic intact while making the script runnable end-to-end, I replace the fastai data/learner parts with a minimal pure-PyTorch inference pipeline that loads images from the provided folders, runs the existing EfficientNet-B5 model, and applies the same fixed rounding thresholds to produce class labels 0–4. I also make pretrained weight loading fully offline-safe (skip download if unavailable) and ensure the submission CSV matches `sample_submission.csv` exactly (`id_code,diagnosis`) with the correct row count. This generate a valid submission file; score likely be low without pretrained/finetuned weights, but it run correctly in the Kaggle environment and can use cached weights if present.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with effectively-untrained predictions (random init or ImageNet-only features with a mismatched regression head), so the smallest legitimate move toward the 0.911 target is to (1) ensure we load competition-specific finetuned weights if they exist anywhere under `/kaggle/input` or `/kaggle/data`, and (2) avoid a hardcoded fake basename (`"abcdef"`) that guarantees no weights are found. I keep your EfficientNet-B5 model and inference logic intact, but expand the weight search to automatically pick up any `.pth/.pt/.bin` file that matches the model’s state dict shape, and I also switch input size to the native EfficientNet-B5 resolution (456) to align preprocessing with the architecture (no change to core modeling). Finally, I keep your fixed rounding thresholds (to preserve semantics) and still write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with “effectively untrained” predictions, and the smallest reliable way to move toward the 0.911 target (without changing the core EfficientNet-B5 logic) is to ensure we actually use meaningful weights. I (1) load ImageNet EfficientNet-B5 weights robustly in an offline Kaggle setting by checking the torch hub cache before attempting any URL download, and (2) add a compatibility-aware preference for checkpoints that match the model including the regression `_fc` head (so competition-finetuned weights are chosen when present). I also make the state-dict loading a bit more tolerant (handling common nesting like `model`, `net`, `ema`, etc.) without changing the model architecture or inference semantics. The submission generation stays identical (same thresholds, same CSV schema), but predictions should be far less random, which should increase QWK from 0.0 toward your target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the current submission is effectively constant/degenerate (most likely because the EfficientNet `_fc` head is random and never trained, while the fixed thresholds assume a trained regression output). To move toward the 0.911 target without changing your core model/inference logic, I keep EfficientNet-B5 + fixed threshold rounding, but I (1) actually use the provided `train.csv` to fit the existing `OptimizedRounder` thresholds on out-of-fold-ish predictions from the same model (no training loop added), and (2) apply those learned thresholds to test predictions. This typically improves QWK a lot versus arbitrary `[0.5,1.5,2.5,3.5]` when outputs are miscalibrated, while preserving architecture and overall semantics. I also make image loading paths robust by using `train_images/` and `test_images/` directly from the same base dir and ensure we always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a degenerate submission (often all-one-class) driven by an effectively untrained/random regression head and/or mismatched preprocessing. I keep your EfficientNet-B5 regression + fixed “OptimizedRounder” discretization, but make two minimal changes that typically move QWK upward: (1) always load ImageNet EfficientNet-B5 backbone weights offline-safely (so the feature extractor isn’t random even when Kaggle has no internet), and (2) add a very small, standard center-crop after resizing to stabilize the image content seen by the model (reduces border/background variance without changing the model). Everything else (model definition, inference loop, rounding approach, submission schema/path) remains the same and it still writes `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from a degenerate model output (random or near-constant regression values) because you never load competition-finetuned weights and you also can’t fit `OptimizedRounder` if `scipy` isn’t available. To move the score upward with minimal changes and without altering the core EfficientNet-B5 + regression + thresholding semantics, I (1) make the training-threshold fitting work without SciPy by adding a tiny pure-NumPy coordinate-descent optimizer inside `OptimizedRounder.fit`, and (2) keep your exact inference/model code but ensure we always derive non-default thresholds from the model’s own train-set predictions. This typically improves QWK substantially versus fixed `[0.5,1.5,2.5,3.5]` when the regression scale is miscalibrated. The script still runs end-to-end and writes a valid `submission.csv` with the correct schema and row order.'
- What this solution (achieved 0.0) has done: 'The current 0.0 score is most consistent with a degenerate submission (often almost-all-one-class) caused by a regression head that’s effectively untrained/miscalibrated; your fitted thresholds help, but they’re currently fit on the same data used to generate them, which can overfit and still produce poor generalization. I keep your EfficientNet-B5 model, pure-PyTorch inference, and the same OptimizedRounder discretization, but I make one minimal evaluation-aligned change: fit the rounding thresholds using out-of-fold (OOF) train predictions via a simple K-fold split (no training added). This typically yields much more stable thresholds and moves QWK upward versus in-sample threshold fitting, while preserving core logic and producing the same submission format. I also ensure we don’t accidentally load ImageNet weights after loading a finetuned checkpoint by keeping the existing load order and only changing threshold fitting.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a degenerate submission (e.g., almost all one class) caused by a regression output scale mismatch: you’re using a single-image regression head but discretizing with thresholds that aren’t fit to the model’s actual prediction distribution. To move the score toward 0.911 with minimal changes, I keep your EfficientNet-B5 model and inference exactly as-is, but I fit the rounding thresholds using true out-of-fold (OOF) predictions by running the same model on each fold’s held-out images (instead of copying in-sample predictions). I also ensure the model uses ImageNet backbone weights only as a fallback (no change in architecture), and keep the submission schema/ordering identical. This should materially improve QWK versus the current pseudo-OOF thresholding while remaining within your constraints.'
- What this solution (achieved 0.0) has done: 'The timeout is dominated by repeated image decoding/preprocessing and repeated inference: you run full train inference once, then you run 5 more full passes over train data for OOF (total ~6x), all with `num_workers=0` and no caching. I keep the exact same model and prediction logic, but remove redundant train inference by reusing the already-computed full-train predictions to populate OOF (equivalent because the model is fixed and inference is deterministic). I also speed up dataloading by using multiple workers, persistent workers, prefetching, and a faster collate path for the test loader, while preserving identical transforms and outputs. Finally, I restrict the expensive checkpoint search to a small number of likely directories and use `torch.load(..., weights_only=True)` when available to reduce load overhead, without changing selection semantics.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely due to a degenerate prediction distribution (often all-one-class) caused by an uncalibrated regression output being discretized with thresholds that are not fit robustly to the model’s output scale. To move the score upward toward your 0.911 target without changing the model or training approach, I keep your EfficientNet-B5 inference exactly as-is but (1) fit the `OptimizedRounder` thresholds on *true* out-of-fold (OOF) predictions (so thresholds generalize better than in-sample fitting), and (2) use a small stratified subsample for OOF threshold fitting to keep runtime under 600s while preserving semantics (threshold fitting only, no training). The submission writing logic and column/order format remain identical, and the script still produces a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from the fact that you never actually load a competition-finetuned checkpoint for the regression head, so outputs are effectively uncalibrated and thresholding collapses into a degenerate class distribution. I keep your EfficientNet-B5 regression + inference + OptimizedRounder logic identical, but make weight loading reliably pick up finetuned checkpoints by (1) searching the full `/kaggle/input` and `/kaggle/data` trees (not just the competition folder), and (2) accepting checkpoints that match most of the backbone even if `_fc` is stored under common alternate names (e.g., `fc`, `classifier`) by mapping them into `_fc` when shapes match. This is a minimal, semantics-preserving change that increases the chance of using real trained weights and should move QWK upward toward your target without changing your model or post-processing. Submission writing and ordering remain unchanged and the script still produces `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import math
import re
import json
import collections
from functools import partial

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import cohen_kappa_score
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

torch.set_grad_enabled(False)

torch.manual_seed(42)
np.random.seed(42)



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
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kh - 1) * self.dilation[1] + 1 - iw, 0)
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
    """Block Decoder"""

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


url_map = {
    "efficientnet-b0": "http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth",
    "efficientnet-b1": "http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth",
    "efficientnet-b2": "http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth",
    "efficientnet-b3": "http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth",
    "efficientnet-b4": "http://storage.googleapis.com/public-models/efficientnet-b4-e116e8b3.pth",
    "efficientnet-b5": "http://storage.googleapis.com/public-models/efficientnet-b5-586e6cc6.pth",
}


def _torch_hub_default_dir():
    try:
        return torch.hub.get_dir()
    except Exception:
        return os.path.expanduser(os.path.join("~", ".cache", "torch", "hub"))


def _find_cached_url_file(url):
    fn = os.path.basename(url)
    return os.path.join(_torch_hub_default_dir(), "checkpoints", fn)


def _torch_load_state(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_pretrained_weights(model, model_name, load_fc=True, weights_path=None):
    """
    Loads pretrained weights.
    Offline-safe: prefer local explicit path; else prefer local torch hub cache; else try URL.
    """
    state_dict = None

    if weights_path is not None and os.path.exists(weights_path):
        state_dict = _torch_load_state(weights_path, map_location="cpu")
        src = f"local file: {weights_path}"
    else:
        cached_path = _find_cached_url_file(url_map[model_name])
        if os.path.exists(cached_path):
            state_dict = _torch_load_state(cached_path, map_location="cpu")
            src = f"torch hub cache: {cached_path}"
        else:
            try:
                state_dict = model_zoo.load_url(
                    url_map[model_name], progress=False, map_location="cpu"
                )
                src = "cache/url"
            except Exception as e:
                print(
                    f"[WARN] Could not load pretrained weights for {model_name} (offline/no cache). Using random init. Reason: {e}"
                )
                return False

    if load_fc:
        model.load_state_dict(state_dict, strict=False)
    else:
        state_dict.pop("_fc.weight", None)
        state_dict.pop("_fc.bias", None)
        model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights for {model_name} from {src}")
    return True


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
        assert isinstance(blocks_args, list), "blocks_args should be a list"
        assert len(blocks_args) > 0, "block args must be greater than 0"
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
    def from_pretrained(cls, model_name, num_classes=1000, weights_path=None):
        model = EfficientNet.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        load_pretrained_weights(
            model, model_name, load_fc=False, weights_path=weights_path
        )
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ["efficientnet_b" + str(i) for i in range(num_models)]
        if model_name.replace("-", "_") not in valid_models:
            raise ValueError("model_name should be one of: " + ", ".join(valid_models))




## === cell 2
md_ef = EfficientNet.from_name("efficientnet-b5", override_params={"num_classes": 1})

imagenet_loaded = load_pretrained_weights(
    md_ef, "efficientnet-b5", load_fc=False, weights_path=None
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
md_ef = md_ef.to(device)
md_ef.eval()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

print("device:", device, "| imagenet_loaded:", imagenet_loaded)




## === cell 3
def get_df():
    base_image_dir = os.path.join(
        "/", "kaggle", "data", "aptos2019-blindness-detection"
    )
    train_dir = os.path.join(base_image_dir, "train_images")
    df = pd.read_csv(os.path.join(base_image_dir, "train.csv"))
    df["path"] = df["id_code"].map(lambda x: os.path.join(train_dir, f"{x}.png"))
    df = df.drop(columns=["id_code"])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.read_csv(os.path.join(base_image_dir, "sample_submission.csv"))
    return df, test_df, base_image_dir


df, test_df, base_image_dir = get_df()
print(df.head())
print(test_df.head())



## === cell 4
bs = 32
sz = EfficientNet.get_image_size("efficientnet-b5")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _load_png_as_rgb(path):
    from PIL import Image

    img = Image.open(path).convert("RGB")
    return img


def _resize_squish(img, size):
    return img.resize((size, size))


def _center_crop(img, crop_size):
    w, h = img.size
    cs = int(crop_size)
    if cs >= w or cs >= h:
        return img
    left = (w - cs) // 2
    top = (h - cs) // 2
    return img.crop((left, top, left + cs, top + cs))


def _to_tensor_normalized(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    return torch.from_numpy(arr)


class TestImagesDataset(Dataset):
    def __init__(self, base_image_dir, test_df, size=224):
        self.base_image_dir = base_image_dir
        self.test_df = test_df.reset_index(drop=True)
        self.size = size

    def __len__(self):
        return len(self.test_df)

    def __getitem__(self, idx):
        id_code = self.test_df.loc[idx, "id_code"]
        path = os.path.join(self.base_image_dir, "test_images", f"{id_code}.png")
        img = _load_png_as_rgb(path)
        img = _resize_squish(img, self.size)
        img = _center_crop(img, int(self.size * 0.94))
        img = _resize_squish(img, self.size)
        x = _to_tensor_normalized(img)
        return x, id_code


class TrainImagesDataset(Dataset):
    def __init__(self, df, size=224):
        self.df = df.reset_index(drop=True)
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "path"]
        y = int(self.df.loc[idx, "diagnosis"])
        img = _load_png_as_rgb(path)
        img = _resize_squish(img, self.size)
        img = _center_crop(img, int(self.size * 0.94))
        img = _resize_squish(img, self.size)
        x = _to_tensor_normalized(img)
        return x, y


def _num_workers():
    try:
        cpu = os.cpu_count() or 2
    except Exception:
        cpu = 2
    return max(2, min(4, cpu))


_NW = _num_workers()
_pin = torch.cuda.is_available()

test_ds = TestImagesDataset(base_image_dir=base_image_dir, test_df=test_df, size=sz)
test_dl = DataLoader(
    test_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=_NW,
    pin_memory=_pin,
    persistent_workers=(_NW > 0),
    prefetch_factor=2 if _NW > 0 else None,
)

train_ds = TrainImagesDataset(df=df, size=sz)
train_dl = DataLoader(
    train_ds,
    batch_size=bs,
    shuffle=False,
    num_workers=_NW,
    pin_memory=_pin,
    persistent_workers=(_NW > 0),
    prefetch_factor=2 if _NW > 0 else None,
)




## === cell 5
def qk(y_pred, y):
    yp = torch.round(y_pred).detach().cpu().numpy().reshape(-1)
    yt = y.detach().cpu().numpy().reshape(-1)
    return torch.tensor(cohen_kappa_score(yp, yt, weights="quadratic"))




## === cell 6
class SimpleLearnerLike:
    def __init__(self, model, test_dl, device):
        self.model = model
        self.test_dl = test_dl
        self.device = device

    def get_test_preds(self):
        preds = []
        ids = []
        self.model.eval()
        with torch.no_grad():
            for xb, idb in self.test_dl:
                xb = xb.to(self.device, non_blocking=True)
                out = self.model(xb).view(-1)
                preds.append(out.detach().cpu().numpy())
                ids.extend(list(idb))
        preds = np.concatenate(preds, axis=0)
        return preds, ids


learn = SimpleLearnerLike(md_ef, test_dl, device)


def get_train_preds(model, train_dl, device):
    model.eval()
    preds = []
    ys = []
    with torch.no_grad():
        for xb, yb in train_dl:
            xb = xb.to(device, non_blocking=True)
            out = model(xb).view(-1)
            preds.append(out.detach().cpu().numpy())
            ys.append(yb.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0), np.concatenate(ys, axis=0)




## === cell 7
def _iter_weight_files(search_roots, exts=(".pth", ".pt", ".bin")):
    for root in search_roots:
        if not os.path.exists(root):
            continue
        if os.path.isfile(root):
            if root.lower().endswith(exts):
                yield root
            continue
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(exts):
                    yield os.path.join(dirpath, fn)


def _maybe_unwrap_state_dict(obj):
    if isinstance(obj, dict):
        for k in [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "ema",
            "student",
            "teacher",
        ]:
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def _clean_state_dict(sd):
    sd = _maybe_unwrap_state_dict(sd)
    if not isinstance(sd, dict):
        return None
    new_sd = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        new_sd[nk] = v

    if "_fc.weight" not in new_sd and "fc.weight" in new_sd:
        new_sd["_fc.weight"] = new_sd["fc.weight"]
    if "_fc.bias" not in new_sd and "fc.bias" in new_sd:
        new_sd["_fc.bias"] = new_sd["fc.bias"]

    if "_fc.weight" not in new_sd and "classifier.weight" in new_sd:
        new_sd["_fc.weight"] = new_sd["classifier.weight"]
    if "_fc.bias" not in new_sd and "classifier.bias" in new_sd:
        new_sd["_fc.bias"] = new_sd["classifier.bias"]

    return new_sd


def _score_state_dict_compat(model, sd):
    """
    Higher is better.
    Prefer checkpoints that match tensor shapes for many keys, and *especially* include _fc for num_classes=1.
    """
    if sd is None:
        return -1
    model_sd = model.state_dict()
    match = 0
    total = 0
    for k, v in sd.items():
        if k in model_sd:
            total += 1
            if hasattr(v, "shape") and v.shape == model_sd[k].shape:
                match += 1

    fc_match = 0
    if (
        "_fc.weight" in sd
        and "_fc.weight" in model_sd
        and hasattr(sd["_fc.weight"], "shape")
    ):
        fc_match += int(sd["_fc.weight"].shape == model_sd["_fc.weight"].shape)
    if "_fc.bias" in sd and "_fc.bias" in model_sd and hasattr(sd["_fc.bias"], "shape"):
        fc_match += int(sd["_fc.bias"].shape == model_sd["_fc.bias"].shape)

    return match + 0.1 * total + 200.0 * fc_match


search_roots = [
    os.path.join("/", "kaggle", "working"),
    os.path.join("/", "kaggle", "input"),
    os.path.join("/", "kaggle", "data"),
]

best_path = None
best_score = -1
candidates = list(_iter_weight_files(search_roots))
print(f"Found {len(candidates)} candidate weight files under Kaggle roots.")

for p in candidates:
    try:
        sd_raw = _torch_load_state(p, map_location="cpu")
        sd = _clean_state_dict(sd_raw)
        sc = _score_state_dict_compat(md_ef, sd)
        if sc > best_score:
            best_score = sc
            best_path = p
    except Exception:
        continue

loaded = False
if best_path is not None:
    try:
        sd_raw = _torch_load_state(best_path, map_location="cpu")
        sd = _clean_state_dict(sd_raw)

        model_sd = md_ef.state_dict()
        shape_matches = 0
        considered = 0
        for k, v in sd.items():
            if k in model_sd and hasattr(v, "shape"):
                considered += 1
                if v.shape == model_sd[k].shape:
                    shape_matches += 1
        match_ratio = (shape_matches / considered) if considered > 0 else 0.0
        fc_ok = (
            "_fc.weight" in sd
            and "_fc.weight" in model_sd
            and hasattr(sd["_fc.weight"], "shape")
            and sd["_fc.weight"].shape == model_sd["_fc.weight"].shape
        ) and (
            "_fc.bias" in sd
            and "_fc.bias" in model_sd
            and hasattr(sd["_fc.bias"], "shape")
            and sd["_fc.bias"].shape == model_sd["_fc.bias"].shape
        )

        if fc_ok and match_ratio >= 0.85:
            missing, unexpected = md_ef.load_state_dict(sd, strict=False)
            md_ef.to(device).eval()
            print(f"Loaded model weights from: {best_path}")
            print(
                f"State dict load summary: missing={len(missing)} unexpected={len(unexpected)} "
                f"compat_score={best_score:.1f} match_ratio={match_ratio:.3f} fc_ok={fc_ok}"
            )
            loaded = True
        else:
            print(
                f"[WARN] Best checkpoint rejected due to low compatibility: path={best_path} "
                f"match_ratio={match_ratio:.3f} fc_ok={fc_ok} compat_score={best_score:.1f}"
            )
    except Exception as e:
        print(f"[WARN] Best candidate weights at {best_path} could not be loaded: {e}")

if not loaded:
    print(
        "[WARN] No finetuned weights loaded. Predictions will be from ImageNet-pretrained backbone (if available), else random init."
    )




## === cell 8
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = None

    def _kappa_loss(self, coef, X, y):
        X = np.asarray(X).reshape(-1)
        y = np.asarray(y).reshape(-1)
        X_p = np.zeros_like(X, dtype=np.int64)
        c0, c1, c2, c3 = coef
        X_p[X < c0] = 0
        X_p[(X >= c0) & (X < c1)] = 1
        X_p[(X >= c1) & (X < c2)] = 2
        X_p[(X >= c2) & (X < c3)] = 3
        X_p[X >= c3] = 4
        ll = metrics.cohen_kappa_score(y, X_p, weights="quadratic")
        return -ll

    def fit(self, X, y):
        X = np.asarray(X).reshape(-1)
        y = np.asarray(y).reshape(-1)

        coef = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)

        def _enforce_monotonic(c):
            c = np.asarray(c, dtype=np.float32)
            eps = 1e-3
            for i in range(1, 4):
                if c[i] <= c[i - 1] + eps:
                    c[i] = c[i - 1] + eps
            return c

        coef = _enforce_monotonic(coef)
        best = -self._kappa_loss(coef, X, y)

        step = float(np.std(X) if np.std(X) > 1e-6 else 1.0)
        step = max(step * 0.25, 0.05)
        for _outer in range(18):
            improved = False
            for j in range(4):
                for direction in (-1.0, 1.0):
                    trial = coef.copy()
                    trial[j] = trial[j] + direction * step
                    trial = _enforce_monotonic(trial)
                    score = -self._kappa_loss(trial, X, y)
                    if score > best + 1e-12:
                        best = score
                        coef = trial
                        improved = True
            step *= 0.7
            if not improved and step < 1e-3:
                break

        self.coef_ = {"x": coef}
        print(best)

    def predict(self, X, coef):
        X = np.asarray(X).reshape(-1)
        X_p = np.zeros_like(X, dtype=np.int64)
        c0, c1, c2, c3 = coef
        X_p[X < c0] = 0
        X_p[(X >= c0) & (X < c1)] = 1
        X_p[(X >= c1) & (X < c2)] = 2
        X_p[(X >= c2) & (X < c3)] = 3
        X_p[X >= c3] = 4
        return X_p

    def coefficients(self):
        return self.coef_["x"]




## === cell 9
def run_subm(
    learn, test_df, coefficients=[0.5, 1.5, 2.5, 3.5], out_path="submission.csv"
):
    opt = OptimizedRounder()

    preds_np, ids = learn.get_test_preds()
    if list(ids) != list(test_df["id_code"].values):
        pred_map = {i: p for i, p in zip(ids, preds_np)}
        preds_np = np.array(
            [pred_map[i] for i in test_df["id_code"].values], dtype=np.float32
        )

    tst_pred = opt.predict(preds_np, coefficients).astype(int)
    tst_pred = np.clip(tst_pred, 0, 4)

    sub = test_df.copy()
    sub["diagnosis"] = tst_pred
    sub = sub[["id_code", "diagnosis"]]
    sub.to_csv(out_path, index=False)
    print(f"done -> wrote {out_path} with shape {sub.shape}")
    return sub


def _build_indexed_train_ds(df, indices, size):
    df_sub = df.iloc[indices].reset_index(drop=True)
    return TrainImagesDataset(df=df_sub, size=size)


def get_train_preds_for_indices(
    model, df, indices, device, size, batch_size, num_workers, pin_memory
):
    ds = _build_indexed_train_ds(df, indices, size=size)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
    )
    model.eval()
    preds = []
    ys = []
    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            out = model(xb).view(-1)
            preds.append(out.detach().cpu().numpy())
            ys.append(yb.detach().cpu().numpy().reshape(-1))
    return np.concatenate(preds, axis=0), np.concatenate(ys, axis=0)


print("Computing full-train prediction stats (single pass)...")
train_preds_full, train_y_full = get_train_preds(md_ef, train_dl, device)
print(
    "Train preds stats:",
    float(np.min(train_preds_full)),
    float(np.max(train_preds_full)),
    float(np.mean(train_preds_full)),
)
print(
    "Train labels stats:",
    int(np.min(train_y_full)),
    int(np.max(train_y_full)),
    float(np.mean(train_y_full)),
)

coefficients_to_use = [0.5, 1.5, 2.5, 3.5]
try:
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    max_fit_n = 1200
    rng = np.random.RandomState(42)
    y_all = df["diagnosis"].values
    if len(df) > max_fit_n:
        sel = []
        for cls in np.unique(y_all):
            idx_cls = np.where(y_all == cls)[0]
            take = max(1, int(round(max_fit_n * (len(idx_cls) / len(df)))))
            take = min(take, len(idx_cls))
            sel.append(rng.choice(idx_cls, size=take, replace=False))
        fit_idx = np.unique(np.concatenate(sel))
        if len(fit_idx) > max_fit_n:
            fit_idx = rng.choice(fit_idx, size=max_fit_n, replace=False)
        fit_idx = np.sort(fit_idx)
    else:
        fit_idx = np.arange(len(df))

    print(f"OOF threshold fit using {len(fit_idx)}/{len(df)} stratified train samples.")

    oof_pred = np.zeros(len(fit_idx), dtype=np.float32)
    oof_y = y_all[fit_idx].astype(int)

    fold = 0
    for tr_local, va_local in skf.split(fit_idx, oof_y):
        fold += 1
        va_idx_global = fit_idx[va_local]
        preds_va, ys_va = get_train_preds_for_indices(
            model=md_ef,
            df=df,
            indices=va_idx_global,
            device=device,
            size=sz,
            batch_size=bs,
            num_workers=_NW,
            pin_memory=_pin,
        )
        oof_pred[va_local] = preds_va.astype(np.float32)
        if fold == 1:
            print(
                "OOF fold 1 preds stats:",
                float(np.min(preds_va)),
                float(np.max(preds_va)),
                float(np.mean(preds_va)),
            )

    opt = OptimizedRounder()
    opt.fit(oof_pred, oof_y)
    coefficients_to_use = list(opt.coefficients())
    print("Using OOF-fitted coefficients:", coefficients_to_use)
except Exception as e:
    print(
        "[WARN] Could not fit OptimizedRounder with true OOF. Falling back to defaults.",
        e,
    )

sub = run_subm(
    learn=learn,
    test_df=test_df,
    coefficients=coefficients_to_use,
    out_path="submission.csv",
)
print(sub.head())



## === cell 10
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["id_code", "diagnosis"]
assert len(chk) == len(test_df)
print(
    "submission checks:",
    chk["diagnosis"].isna().sum(),
    int(chk["diagnosis"].min()),
    int(chk["diagnosis"].max()),
)
