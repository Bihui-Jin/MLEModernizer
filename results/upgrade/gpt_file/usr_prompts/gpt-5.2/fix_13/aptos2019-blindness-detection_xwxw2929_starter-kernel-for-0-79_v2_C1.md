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

0.8321871294659671

# 6. Current score

0.02917

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on fastai (it’s not available in your environment) and replace only the missing pieces with a small PyTorch inference pipeline that keeps the same core idea: EfficientNet-B5 backbone + single-regression output + rounding to 0–4 for submission. I also fix missing imports (`os`, `collections`, etc.), remove notebook magics/`!cp` that break in script mode, and make paths robust for both `/kaggle/input/...` and your listed `/kaggle/data/...` mirrors. Since you provided no working score (no valid submission yet), the priority is to run end-to-end and write a valid `submission.csv` with columns `id_code,diagnosis`. Where pretrained weights cannot be downloaded (no internet), the script still run and produce a valid submission, using random-init weights as a fallback (score be low but valid).'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is consistent with an offline run where EfficientNet-B5 never actually gets pretrained (so predictions are essentially random), plus the fixed thresholds (0.5/1.5/2.5/3.5) are not calibrated to the model’s regression output distribution. To move toward the 0.832 target without changing the core approach (EfficientNet-B5 + single regression output + rounding), I (1) load pretrained weights from common local Kaggle paths if present (no internet), and (2) fit the four rounding thresholds on the training set using out-of-fold style predictions from the same model (no training, just calibration), optimizing QWK on a validation split. This keeps the architecture and inference logic the same, but makes the discretization match the model’s output scale, which typically yields a large jump over fixed generic thresholds. The script still fall back gracefully to fixed thresholds if calibration can’t run, and it always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with predictions being essentially random/constant because the EfficientNet backbone is not actually loading any useful weights (offline URL download fails and no local weights are found). To move the score up toward the 0.832 target while keeping the exact same core approach (EfficientNet-B5 single-output regression + thresholding to 0–4), I (1) correctly implement `from_pretrained()` so it really attempts to load ImageNet weights (local first, then URL if available), and (2) fix a critical bug in the block stride decoding (it currently makes strides lists, breaking the residual/shape logic), which can silently degrade outputs. I also make the validation split deterministic and non-leaky (use a random split rather than “first 20% after shuffle”) to make threshold calibration more stable, without changing the model or adding training. The script still always write a valid `submission.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model running with effectively random weights and/or producing nearly constant predictions, because the script never actually trains a DR head and likely fails to load any meaningful pretrained checkpoint beyond ImageNet (and even ImageNet may fail offline). To move the score upward toward the 0.832 target while keeping the same core logic (EfficientNet-B5 backbone + 1D regression output + thresholding to 0–4), I (1) add a robust local weight search that can load common Kaggle EfficientNet/competition checkpoints if present and correctly handle `module.` prefixes, and (2) make threshold calibration more stable by using a deterministic stratified split so validation contains all classes. These are minimal, execution-safe changes that don’t alter the model architecture or inference semantics, but they reduce the chance of “all one class” outputs and improve QWK via better discretization. The script still always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model producing near-random/near-constant outputs because it never loads a meaningful DR-trained checkpoint (and ImageNet weights may also fail offline), so threshold calibration can’t rescue performance. To move the score upward toward the 0.832 target while preserving the exact core idea (EfficientNet-B5 backbone + single regression output + thresholding to 0–4), I (1) force robust loading of a locally available APTOS/DR checkpoint if present by correctly handling common checkpoint key prefixes (`module.`, `_fc.` vs `fc.`) and by preferring strict-ish compatible loads, and (2) calibrate thresholds on a stratified validation split using the model’s current outputs but with a safer initialization based on prediction quantiles to avoid getting stuck when outputs are poorly scaled. These changes don’t alter the architecture, loss, or training loop (there is none), but they substantially reduce the chance of “all one class” submissions and typically increase QWK when any reasonable weights are available. The script still runs end-to-end offline and always writes a valid `submission.csv` in the required format.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with a submission/file alignment issue or with the model output being essentially random because no meaningful weights are loaded; the smallest safe improvement is to (1) guarantee we’re reading/writing exactly the competition’s `sample_submission.csv` order and row count, and (2) make checkpoint loading actually take effect by preferring DR-trained checkpoints (if present) and correctly handling common key mismatches (`_fc.` vs `fc.`) while keeping the same EfficientNet-B5 + 1D regression + thresholding core. I also add a deterministic sanity fallback so if predictions become NaN/Inf or nearly constant, we fall back to a reasonable label prior (train distribution) rather than producing a pathological all-one-class submission that often scores ~0. These changes don’t alter the model architecture, loss, or training loop (still none), but they directly reduce the chance of a broken/invalid submission and should move the score upward toward your 0.832 target when any usable weights exist locally.'
- What this solution (achieved 0.16725) has done: 'Your 0.0 score is most likely coming from a “no-real-weights” situation (offline download fails and no DR-trained checkpoint is found), which yields near-constant/random predictions that QWK scores extremely poorly. I keep your exact core approach (EfficientNet-B5, 1D regression head, thresholding to 0–4) but add two minimal, score-relevant safeguards: (1) if no good checkpoint is loaded, fit the regression head (and only the head) on the provided train set for a few epochs, and (2) calibrate thresholds on a deterministic stratified validation split using those model outputs. This stays within the same architecture/loss semantics (still regression output + discretization), but ensures predictions are at least correlated with labels so the score moves upward toward your 0.832 target. The script still runs end-to-end offline and always writes a valid `submission.csv` with correct ordering/columns.'
- What this solution (achieved 0.34372) has done: 'Your current gap to the 0.832 target is large (0.167 → needs to increase), and the biggest score limiter is that you only train the final FC head while leaving the backbone frozen (random or ImageNet-only features are not DR-specific enough). I keep the same EfficientNet-B5 + 1D regression output + MSE + thresholding core, but (1) unfreeze the last EfficientNet stage + head for a few epochs to learn DR-specific features, and (2) use a more stable optimizer setup (AdamW + small weight decay) plus a light LR schedule to improve generalization without changing the approach. I also tighten determinism in the data split and ensure calibration always runs on the post-finetune validation predictions so thresholds match the model’s output scale. These are minimal, directly score-relevant changes that should move QWK upward toward your target while still producing a valid `submission.csv` end-to-end within the time limit.'
- What this solution (achieved 0.0) has done: 'The timeout is dominated by two things: repeated expensive PIL decode/resize for train/val across multiple epochs, and the very heavy EfficientNet-B5 forward/backward at 456px. To keep core logic identical while cutting runtime, I (1) add a deterministic on-disk feature cache for resized+normalized tensors so each image is decoded/resized once, (2) enable persistent DataLoader workers + larger prefetch to reduce input stalls, and (3) use `torch.inference_mode()` for prediction paths (exactly equivalent to `no_grad()` for outputs, but faster). These changes preserve the same model, loss, training loop, threshold optimization, and evaluation semantics; they only remove redundant preprocessing work and reduce overhead.'
- What this solution (achieved 0.02917) has done: 'Your current 0.0 score is most consistent with either (a) submission `diagnosis` being misaligned to `id_code` order (mapping + fillna can silently introduce wrong labels) or (b) the model producing nearly-constant outputs that collapse QWK. I make the smallest score-relevant changes to guarantee perfect row alignment by generating predictions in exactly `sample_submission.csv` order (no dict mapping), and I add a sanity fallback that uses the *train label distribution* to create a non-degenerate prediction vector if outputs are constant/NaN (better than constant class). I also make threshold calibration deterministic and constrained (strictly increasing thresholds) to avoid pathological threshold sets that can worsen QWK. These changes keep the same EfficientNet-B5 + 1D regression + thresholding core logic and still write a valid `submission.csv`.'
- What this solution (achieved 0.02917) has done: 'I fix the DataLoader crashes by making the on-disk image cache safe for multi-worker use (atomic writes plus a “don’t cache if it fails” fallback) and by preventing multiple workers from writing the same cache file simultaneously. To keep the core model/training logic unchanged, I only adjust the caching mechanism and DataLoader worker settings so training/inference can complete reliably within the environment’s disk constraints. I also add a small robustness guard so if caching repeatedly fails, it transparently switches to in-memory preprocessing without killing workers. This should restore end-to-end execution and, since the model can now actually finetune + calibrate + infer, it should move the score up from the current ~0.03 toward your target.'

# 9. Code solution

## === cell 0
import os
import re
import math
import json
import warnings
import collections
from functools import partial
from collections import Counter

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils import model_zoo

from PIL import Image

warnings.filterwarnings("ignore")

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
        self.stride = (
            self.stride if len(self.stride) == 2 else (self.stride[0], self.stride[0])
        )

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * sh + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * self.dilation[1] + 1 - iw, 0)
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
        self.stride = (
            self.stride if len(self.stride) == 2 else (self.stride[0], self.stride[0])
        )

        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * sh + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * self.dilation[1] + 1 - iw, 0)
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

        assert ("s" in options and len(options["s"]) == 2) or (
            "s" in options and len(options["s"]) == 1
        )

        s = options["s"]
        if len(s) == 2:
            assert s[0] == s[1]
            stride_val = int(s[0])
        else:
            stride_val = int(s[0])

        return BlockArgs(
            kernel_size=int(options["k"]),
            num_repeat=int(options["r"]),
            input_filters=int(options["i"]),
            output_filters=int(options["o"]),
            expand_ratio=int(options["e"]),
            id_skip=("noskip" not in block_string),
            se_ratio=float(options["se"]) if "se" in options else None,
            stride=stride_val,
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


def load_pretrained_weights(model, model_name, load_fc=True):
    state_dict = model_zoo.load_url(url_map[model_name])
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop("_fc.weight", None)
        state_dict.pop("_fc.bias", None)
        model.load_state_dict(state_dict, strict=False)
    print("Loaded pretrained weights for {}".format(model_name))


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
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000, load_fc=True):
        model = EfficientNet.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        try:
            load_pretrained_weights(model, model_name, load_fc=load_fc)
        except Exception as e:
            print(
                f"Warning: from_pretrained could not download weights (offline likely): {e}"
            )
        return model




## === cell 2
md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1, load_fc=False)


def _clean_state_dict_keys(sd):
    if not isinstance(sd, dict):
        return sd
    if any(k.startswith("module.") for k in sd.keys()):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    return sd


def _remap_head_keys_for_1d_regression(sd):
    """
    Directly relevant to score: many APTOS/DR checkpoints store head keys as 'fc.*' not '_fc.*'.
    Without this remap the DR-trained head won't load, causing near-random/constant predictions.
    """
    if not isinstance(sd, dict):
        return sd
    if "fc.weight" in sd and "_fc.weight" not in sd:
        sd["_fc.weight"] = sd.pop("fc.weight")
    if "fc.bias" in sd and "_fc.bias" not in sd:
        sd["_fc.bias"] = sd.pop("fc.bias")
    return sd


def _try_load_efficientnet_imagenet_weights(model, model_name="efficientnet-b5"):
    candidate_files = [
        "/kaggle/input/efficientnet-pytorch/efficientnet-b5-586e6cc6.pth",
        "/kaggle/input/efficientnet-b5/efficientnet-b5-586e6cc6.pth",
        "/kaggle/input/efficientnet-b5-weights/efficientnet-b5-586e6cc6.pth",
        "/kaggle/input/efficientnet-weights/efficientnet-b5-586e6cc6.pth",
        "/kaggle/input/pretrained-models/efficientnet-b5-586e6cc6.pth",
        "/kaggle/data/efficientnet-pytorch/efficientnet-b5-586e6cc6.pth",
        "/kaggle/data/efficientnet-b5/efficientnet-b5-586e6cc6.pth",
        "/kaggle/data/efficientnet-b5-weights/efficientnet-b5-586e6cc6.pth",
        "/kaggle/data/efficientnet-weights/efficientnet-b5-586e6cc6.pth",
        "/kaggle/data/pretrained-models/efficientnet-b5-586e6cc6.pth",
        "/kaggle/working/efficientnet-b5-586e6cc6.pth",
        "./efficientnet-b5-586e6cc6.pth",
    ]
    for p in candidate_files:
        if os.path.exists(p):
            sd = torch.load(p, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            sd = _clean_state_dict_keys(sd)
            sd.pop("_fc.weight", None)
            sd.pop("_fc.bias", None)
            sd.pop("fc.weight", None)
            sd.pop("fc.bias", None)
            model.load_state_dict(sd, strict=False)
            print("Loaded local ImageNet weights:", p)
            return True
    return False


loaded_any = _try_load_efficientnet_imagenet_weights(md_ef, "efficientnet-b5")
if not loaded_any:
    try:
        load_pretrained_weights(md_ef, "efficientnet-b5", load_fc=False)
        loaded_any = True
    except Exception as e:
        print(
            f"Warning: could not load pretrained weights (offline environment likely). Proceeding with random init. Error: {e}"
        )

md_ef = md_ef.to(device)
md_ef.eval()




## === cell 3
def _find_base_dir():
    candidates = [
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/data/aptos2019-blindness-detection",
        "../input/aptos2019-blindness-detection",
        "../data/aptos2019-blindness-detection",
        "./aptos2019-blindness-detection",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "test.csv")
        ):
            return c
    for root in ["/kaggle/input", "/kaggle/data", ".."]:
        if os.path.exists(root):
            for dirpath, dirnames, filenames in os.walk(root):
                if (
                    "train.csv" in filenames
                    and "test.csv" in filenames
                    and dirpath.endswith("aptos2019-blindness-detection")
                ):
                    return dirpath
    raise FileNotFoundError(
        "Could not find aptos2019-blindness-detection directory with train.csv/test.csv"
    )


BASE_DIR = _find_base_dir()
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")

print("BASE_DIR:", BASE_DIR)
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))




## === cell 4
def _try_load_local_checkpoint(model):
    """
    Directly relevant to score: prefer DR-trained checkpoints if present locally.
    Minimal change: broaden the scan (still shallow) and only load when compatible.
    """
    scan_roots = ["/kaggle/input", "/kaggle/working"]
    name_patterns = ("aptos", "blind", "retina", "efficientnet", "b5", "dr", "kappa")
    candidates = []

    for root in scan_roots:
        if os.path.exists(root):
            for dirpath, dirnames, filenames in os.walk(root):
                for fn in filenames:
                    lfn = fn.lower()
                    if lfn.endswith((".pth", ".pt")) and any(
                        p in lfn for p in name_patterns
                    ):
                        candidates.append(os.path.join(dirpath, fn))
                if dirpath.count(os.sep) - root.count(os.sep) >= 3:
                    dirnames[:] = []

    seen = set()
    candidates = [p for p in candidates if not (p in seen or seen.add(p))]

    best_loaded = False
    for p in candidates:
        try:
            sd = torch.load(p, map_location="cpu")
            if isinstance(sd, dict) and "state_dict" in sd:
                sd = sd["state_dict"]
            sd = _clean_state_dict_keys(sd)
            sd = _remap_head_keys_for_1d_regression(sd)

            missing, unexpected = model.load_state_dict(sd, strict=False)
            head_loaded = ("_fc.weight" not in missing) and ("_fc.bias" not in missing)
            if head_loaded:
                print("Loaded local DR checkpoint (head loaded):", p)
                best_loaded = True
                break
        except Exception:
            continue

    if not best_loaded:
        print(
            "No suitable local DR checkpoint found (with head loaded); using current model weights."
        )
    return best_loaded


os.makedirs("models", exist_ok=True)
loaded_dr_checkpoint = _try_load_local_checkpoint(md_ef)




## === cell 5
def get_df():
    df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    train_dir = os.path.join(BASE_DIR, "train_images")
    df["path"] = df["id_code"].map(lambda x: os.path.join(train_dir, f"{x}.png"))
    df = (
        df[["id_code", "path", "diagnosis"]]
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )

    test_df["path"] = test_df["id_code"].map(
        lambda x: os.path.join(TEST_IMG_DIR, f"{x}.png")
    )
    return df, test_df


df, test_df = get_df()
print(df.head())
print(test_df.head())



## === cell 6
IMG_SIZE = 456  # EfficientNet-B5 default
BATCH_SIZE = 16  # keep runtime under limits on CPU/GPU

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CACHE_DIR = os.path.join("/kaggle/working", f"img_cache_{IMG_SIZE}")
os.makedirs(CACHE_DIR, exist_ok=True)

CACHE_ENABLED = True


def _cache_path_for_image(path, img_size=IMG_SIZE):
    safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", path)
    if len(safe) > 180:
        safe = safe[-180:]
    return os.path.join(CACHE_DIR, f"{safe}__{img_size}.pt")


def _lock_path_for_cache(cp):
    return cp + ".lock"


def _acquire_lock(lock_path, timeout_s=10.0, poll_s=0.05):
    start = None
    try:
        import time

        start = time.time()
        while True:
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                return True
            except FileExistsError:
                if (time.time() - start) > timeout_s:
                    return False
                time.sleep(poll_s)
    except Exception:
        return False


def _release_lock(lock_path):
    try:
        os.remove(lock_path)
    except Exception:
        pass


def _decode_and_preprocess_image(path, img_size=IMG_SIZE):
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # CHW
    t = torch.from_numpy(arr)
    return t


def _load_image(path, img_size=IMG_SIZE):
    global CACHE_ENABLED

    cp = _cache_path_for_image(path, img_size=img_size)
    if CACHE_ENABLED and os.path.exists(cp):
        try:
            return torch.load(cp, map_location="cpu")
        except Exception:
            try:
                os.remove(cp)
            except Exception:
                pass

    t = _decode_and_preprocess_image(path, img_size=img_size)

    if not CACHE_ENABLED:
        return t

    lockp = _lock_path_for_cache(cp)
    got_lock = _acquire_lock(lockp, timeout_s=10.0)
    try:
        if os.path.exists(cp):
            try:
                return torch.load(cp, map_location="cpu")
            except Exception:
                try:
                    os.remove(cp)
                except Exception:
                    pass

        if got_lock:
            tmp = cp + f".tmp_{os.getpid()}_{np.random.randint(0, 1_000_000)}"
            try:
                torch.save(t, tmp)
                os.replace(tmp, cp)  # atomic on same filesystem
            except Exception as e:
                CACHE_ENABLED = False
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except Exception:
                    pass
    finally:
        if got_lock:
            _release_lock(lockp)

    return t


class TestDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = _load_image(row["path"])
        return row["id_code"], x


def _make_loader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )


test_ds = TestDataset(test_df)
test_loader = _make_loader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)




## === cell 7
class TrainDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = _load_image(row["path"])
        y = int(row["diagnosis"])
        return x, y


def predict_loader(model, loader):
    model.eval()
    all_out = []
    all_y = []
    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 2:
                xb, yb = batch
                all_y.append(yb.numpy())
            else:
                raise ValueError("Unexpected batch format.")
            xb = xb.to(device, non_blocking=True)
            out = model(xb).view(-1)
            out = out.detach().float().cpu().numpy()
            all_out.append(out)
    all_out = np.concatenate(all_out, axis=0)
    all_y = np.concatenate(all_y, axis=0)
    return all_out, all_y




## === cell 8
def apply_coefficients(preds_1d, coefficients=(0.5, 1.5, 2.5, 3.5)):
    coef = list(coefficients)
    x = np.array(preds_1d, copy=True)
    y = np.zeros_like(x, dtype=np.int64)
    y[x >= coef[0]] = 1
    y[x >= coef[1]] = 2
    y[x >= coef[2]] = 3
    y[x >= coef[3]] = 4
    y = np.clip(y, 0, 4)
    return y


def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape

    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < num_classes and 0 <= b < num_classes:
            O[a, b] += 1.0

    act_hist = np.bincount(y_true, minlength=num_classes).astype(np.float64)
    pred_hist = np.bincount(y_pred, minlength=num_classes).astype(np.float64)
    E = np.outer(act_hist, pred_hist)
    E = E / E.sum() * O.sum() if E.sum() > 0 else E

    W = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            W[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def _ensure_strictly_increasing(thr, eps=1e-5):
    thr = np.array(thr, dtype=np.float64)
    thr = np.sort(thr)
    for i in range(1, len(thr)):
        if thr[i] <= thr[i - 1]:
            thr[i] = thr[i - 1] + eps
    return thr


def optimize_thresholds(preds, y_true, init=(0.5, 1.5, 2.5, 3.5), n_iter=25):
    best = _ensure_strictly_increasing(init)

    def score_for(thr):
        thr = _ensure_strictly_increasing(thr)
        yp = apply_coefficients(preds, coefficients=tuple(thr))
        return quadratic_weighted_kappa(y_true, yp, num_classes=5)

    best_score = score_for(best)

    pmin, pmax = float(np.min(preds)), float(np.max(preds))
    span = max(1e-6, pmax - pmin)
    step0 = span / 8.0

    for it in range(n_iter):
        improved = False
        step = step0 * (0.7**it)
        for k in range(4):
            candidates = []
            for delta in [-2 * step, -step, 0.0, step, 2 * step]:
                thr = best.copy()
                thr[k] = thr[k] + delta
                thr = _ensure_strictly_increasing(thr)
                thr = np.clip(thr, pmin - 0.1 * span, pmax + 0.1 * span)
                thr = _ensure_strictly_increasing(thr)
                candidates.append(thr)

            for thr in candidates:
                sc = score_for(thr)
                if sc > best_score + 1e-12:
                    best_score = sc
                    best = thr
                    improved = True
        if not improved:
            break
    return tuple(best.tolist()), float(best_score)


def _init_thresholds_from_quantiles(preds):
    """
    Directly relevant to score: outputs are often not on [0,4] scale, so quantile init stabilizes calibration.
    """
    qs = np.quantile(preds, [0.2, 0.4, 0.6, 0.8])
    qs = _ensure_strictly_increasing(qs, eps=1e-4).astype(np.float64)
    return tuple(qs.tolist())




## === cell 9
def _set_batchnorm_eval(m):
    if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.eval()


def _finetune_last_stage_and_head_if_needed(
    model, train_df, val_df, epochs=4, lr_head=3e-4, lr_backbone=3e-5, weight_decay=1e-4
):
    for p in model.parameters():
        p.requires_grad = False

    for p in model._fc.parameters():
        p.requires_grad = True

    unfreeze_n_blocks = min(12, len(model._blocks))
    for blk in model._blocks[-unfreeze_n_blocks:]:
        for p in blk.parameters():
            p.requires_grad = True

    for p in model._conv_head.parameters():
        p.requires_grad = True
    for p in model._bn1.parameters():
        p.requires_grad = True

    model.train()
    model.apply(_set_batchnorm_eval)

    head_params = list(model._fc.parameters())
    backbone_params = []
    for blk in model._blocks[-unfreeze_n_blocks:]:
        backbone_params += list(blk.parameters())
    backbone_params += list(model._conv_head.parameters()) + list(
        model._bn1.parameters()
    )

    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    loss_fn = torch.nn.MSELoss()

    nw = 2 if CACHE_ENABLED else 0

    tr_loader = _make_loader(
        TrainDataset(train_df),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=nw,
    )
    va_loader = _make_loader(
        TrainDataset(val_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=nw,
    )

    for ep in range(epochs):
        running = 0.0
        nobs = 0
        for xb, yb in tr_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).float()

            opt.zero_grad(set_to_none=True)
            out = model(xb).view(-1)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

            running += float(loss.item()) * xb.size(0)
            nobs += xb.size(0)

        scheduler.step()

        model.eval()
        with torch.inference_mode():
            vp, vy = predict_loader(model, va_loader)
            init_thr = _init_thresholds_from_quantiles(vp)
            thr, qwk = optimize_thresholds(vp, vy, init=init_thr, n_iter=15)
        model.train()
        model.apply(_set_batchnorm_eval)

        lrs = [pg["lr"] for pg in opt.param_groups]
        print(
            f"Finetune epoch {ep+1}/{epochs} - train MSE: {running/max(1,nobs):.4f} - val QWK (calibrated): {qwk:.4f} - lrs: {lrs} - cache_enabled: {CACHE_ENABLED}"
        )

    model.eval()
    return model


rng = np.random.RandomState(42)
y_all = df["diagnosis"].astype(int).values
idx_by_class = {c: np.where(y_all == c)[0] for c in range(5)}
val_idx = []
for c, idxs in idx_by_class.items():
    idxs = idxs.copy()
    rng.shuffle(idxs)
    if len(idxs) == 0:
        continue
    take = int(round(0.2 * len(idxs)))
    take = max(1, min(take, 120))
    val_idx.append(idxs[:take])
val_idx = np.concatenate(val_idx) if len(val_idx) else np.array([], dtype=np.int64)
val_idx = np.unique(val_idx)

val_df = df.iloc[val_idx].reset_index(drop=True)
train_df = df.drop(index=val_idx).reset_index(drop=True)

if not loaded_dr_checkpoint:
    md_ef = _finetune_last_stage_and_head_if_needed(
        md_ef,
        train_df,
        val_df,
        epochs=4,
        lr_head=3e-4,
        lr_backbone=3e-5,
        weight_decay=1e-4,
    )



## === cell 10
calibrated_coefficients = (0.5, 1.5, 2.5, 3.5)
do_calibrate = True

try:
    nw = 2 if CACHE_ENABLED else 0
    val_loader = _make_loader(
        TrainDataset(val_df),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=nw,
    )
    val_preds, val_y = predict_loader(md_ef, val_loader)

    if not (np.isfinite(val_preds).all() and len(val_preds) > 10):
        raise ValueError("Invalid validation predictions for calibration.")

    init_thr = calibrated_coefficients
    if (val_preds.max() - val_preds.min()) > 1e-6:
        if (
            (val_preds.mean() < -1.0)
            or (val_preds.mean() > 5.0)
            or (val_preds.std() < 0.05)
        ):
            init_thr = _init_thresholds_from_quantiles(val_preds)

    calibrated_coefficients, val_qwk = optimize_thresholds(
        val_preds, val_y, init=init_thr, n_iter=25
    )
    print("Calibration init:", init_thr)
    print("Calibrated coefficients:", calibrated_coefficients, "val QWK:", val_qwk)
except Exception as e:
    do_calibrate = False
    print("Warning: calibration failed; using default coefficients. Error:", e)



## === cell 11
preds = np.zeros((len(test_df),), dtype=np.float32)

nw = 2 if CACHE_ENABLED else 0
test_loader = _make_loader(
    TestDataset(test_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=nw
)

md_ef.eval()
with torch.inference_mode():
    offset = 0
    for ids, xb in test_loader:
        xb = xb.to(device, non_blocking=True)
        out = md_ef(xb).view(-1)
        out = out.detach().float().cpu().numpy()
        preds[offset : offset + len(out)] = out
        offset += len(out)

print(
    "Preds stats:",
    float(np.nanmin(preds)),
    float(np.nanmean(preds)),
    float(np.nanmax(preds)),
    "cache_enabled:",
    CACHE_ENABLED,
)



## === cell 12
sample_sub = pd.read_csv(SAMPLE_SUB)

sample_ids = sample_sub["id_code"].astype(str).values
test_ids = test_df["id_code"].astype(str).values
id_to_idx = {k: i for i, k in enumerate(test_ids)}
order_idx = np.array([id_to_idx.get(k, -1) for k in sample_ids], dtype=np.int64)

coefficients = calibrated_coefficients
tst_pred = apply_coefficients(preds, coefficients=coefficients)

aligned_pred = np.where(
    order_idx >= 0, tst_pred[np.clip(order_idx, 0, len(tst_pred) - 1)], 0
).astype(np.int64)

bad_preds = (not np.isfinite(preds).all()) or (float(np.std(preds)) < 1e-6)

if bad_preds:
    train_counts = (
        df["diagnosis"]
        .value_counts(normalize=True)
        .reindex(range(5), fill_value=0.0)
        .values.astype(np.float64)
    )
    train_counts = train_counts / max(1e-12, train_counts.sum())
    rs = np.random.RandomState(42)
    aligned_pred = rs.choice(
        np.arange(5, dtype=np.int64), size=len(sample_sub), p=train_counts
    ).astype(np.int64)
    print(
        "Warning: model outputs invalid/nearly-constant; falling back to train-prior sampled labels."
    )

sample_sub["diagnosis"] = aligned_pred.astype(int)
sample_sub.to_csv("submission.csv", index=False)

print("Used coefficients:", coefficients)
print("Wrote submission.csv with shape:", sample_sub.shape)
print(sample_sub.head())
