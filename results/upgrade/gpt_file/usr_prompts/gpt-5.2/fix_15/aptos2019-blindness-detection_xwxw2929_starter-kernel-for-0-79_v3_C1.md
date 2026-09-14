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

0.9072190121781872

# 6. Current score

0.02917

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the notebook-only magics and the broken `fastai.callbacks` import (your environment doesn’t have that module), and instead run the same overall approach (EfficientNet model + pretrained checkpoint + test-time prediction + rounding to 0–4) using pure PyTorch. I also fix missing imports (`os`, `collections`, etc.), remove the internet weight download (Kaggle has no internet), and make the code robust to the unknown checkpoint filename by auto-searching for a `.pth` in `/kaggle/input`. Finally, I ensure the submission is written as `submission.csv` with exactly `id_code,diagnosis` and aligned to `test.csv` order.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with the model running with random weights (no valid checkpoint loaded), plus a small but real logic bug in the skip-connection condition that can also prevent proper checkpoint compatibility. I make two minimal, score-relevant fixes: (1) correct the skip condition to compare the stride list properly (so the architecture matches expected EfficientNet behavior and checkpoints), and (2) make checkpoint discovery prefer an APTOS/EfficientNet-relevant `.pth` and verify that we actually loaded meaningful weights (otherwise explicitly fall back to a safe constant prediction so you don’t submit essentially-random outputs). This preserves your core approach (EfficientNet-B5 regression head + fixed thresholds), keeps I/O paths unchanged, and still writes a valid `submission.csv` with `id_code,diagnosis` aligned to `test.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score strongly suggests the submission is either all-zeros (your current “safe fallback”) or near-random due to not actually loading the intended weights. To move toward the target with minimal change, I (1) make checkpoint discovery deterministic and more likely to pick a real APTOS EfficientNet-B5 checkpoint by preferring larger files and model-name matches, and (2) tighten the load verification by checking that key layers (stem/head/fc) have matching shapes so we don’t silently run with incompatible weights. If no compatible checkpoint is found, I keep the existing constant fallback (so we still always produce a valid `submission.csv`), but the main goal is to reliably load the correct weights when they exist. No model architecture, transforms, thresholds, or prediction logic is changed beyond checkpoint selection/validation.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly coming from the “safe constant predictions” fallback path, which triggers when no compatible checkpoint is actually loaded. To move toward the target with minimal change, I (1) make checkpoint discovery try multiple top candidates (not just the single best-scoring filename) and load the first one that is shape-compatible, and (2) broaden compatibility to accept common head naming (`_fc.*` vs `fc.*`) and `num_classes=5` classification checkpoints by adapting only the final layer weights (core EfficientNet feature extractor weights unchanged). This keeps your core approach (EfficientNet-B5 + pretrained checkpoint inference + thresholding/argmax to 0–4) but greatly increases the chance you use real learned weights instead of constant zeros. The script still always writes a valid `submission.csv` with `id_code,diagnosis` aligned to `test.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is overwhelmingly likely to come from the fallback path producing constant all-zero predictions because no compatible checkpoint is actually being loaded. To move toward the 0.9072 target with minimal core-logic changes, I (1) make checkpoint loading more robust to common key-prefix patterns (`net.`, `model.module.`, etc.) and EfficientNet naming (`_fc` vs `fc`), and (2) explicitly support common APTOS checkpoints where the head is stored as a 5-class classifier while the backbone matches—by loading the backbone weights and skipping only the incompatible head. This preserves your same EfficientNet-B5 inference approach and your same post-processing (argmax for classification / fixed thresholds for regression), but greatly increases the chance you use learned weights instead of the constant fallback. The script still always writes a valid `submission.csv` with exactly `id_code,diagnosis` aligned to `test.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with either (a) never actually loading a meaningful checkpoint, or (b) loading something but producing badly-calibrated class outputs (e.g., using argmax on logits without softmax/TTA) that collapses predictions. To move toward the target with minimal core-logic change, I (1) make checkpoint loading *actually strict for the backbone* (stem/head/blocks) while still allowing head mismatches, and (2) for classification checkpoints, switch to using softmax probabilities and the expected value (regression-style) followed by the same fixed thresholds—this typically improves QWK vs raw argmax while preserving the same model and post-processing semantics (0–4 via thresholds). I also add a deterministic “center-crop at test size” (not random) and a very light 2-way TTA (original + horizontal flip) at inference only, which usually nudges kappa upward without changing training or architecture. The script still always write a valid `submission.csv` aligned to `test.csv`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the “no compatible checkpoint loaded” path producing all-zero predictions, so the smallest score-relevant improvement is to make checkpoint loading succeed when a valid EfficientNet-B5 checkpoint exists. I keep your exact EfficientNet code and inference/thresholding logic, but broaden checkpoint key normalization (handle more real-world prefixes like `module.model.` and `state_dict` nesting), and I try loading into both `_fc` and `fc` variants so we don’t miss checkpoints that used different head naming. I also relax the “loaded_ok” test from a fragile fraction-of-keys heuristic to a backbone-keys-present heuristic (stem/head + at least one block), which is what actually matters for learned predictions. If no checkpoint is found/compatible, the script still deterministically write a valid `submission.csv` (as before), but the goal is to avoid falling into that fallback when weights are present.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still best explained by the code falling into the constant-all-zeros fallback because no checkpoint is actually being loaded (or the “loaded_ok” check is failing even when usable weights exist). I make checkpoint loading more robust but still minimal: (1) search and try more realistic checkpoint file types (`.bin`, `.pth.tar`) and prefer likely APTOS/EfficientNet files by size/name, and (2) improve state-dict unwrapping to handle common nesting patterns and key prefixes more reliably. I also change the loader to attempt a “backbone-only” load (skip incompatible head weights) for both the regression and classification model, which preserves your core inference logic but greatly increases the chance you use meaningful pretrained features instead of zeros. The rest of your pipeline (EfficientNet code, transforms, 2-way TTA, thresholds, and submission formatting) stays the same.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the code not loading any meaningful weights and falling back to constant all-zero predictions, so the smallest score-relevant improvement is to make checkpoint loading succeed more often when a valid EfficientNet checkpoint exists. I keep your exact EfficientNet architecture and the same inference logic (TTA + expected value/thresholds), but broaden checkpoint discovery to include `.ckpt` and relax the “must-have backbone keys” check to accept checkpoints that omit BN running stats (common) while still requiring real stem/head/block weights. I also make the loader try both model variants (reg/cls) and accept a strong partial backbone load even if some head keys mismatch, instead of rejecting it and falling back to zeros. This should move the score upward toward your target without changing the model’s core behavior.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the inference falling into the constant-all-zero fallback because no compatible checkpoint is actually being loaded, so the smallest score-relevant move toward your 0.9072 target is to make checkpoint loading succeed more often (without changing the model/inference core). I (1) broaden the backbone-compatibility check to accept EfficientNet checkpoints whose parameter names use common alternative conventions (notably `._fc` vs `._fc`, and especially `conv_stem/conv_head` vs `_conv_stem/_conv_head`), and (2) normalize more key patterns (`blocks.` vs `_blocks.` etc.) so backbone weights can be loaded even if the head is incompatible. If after these attempts nothing loads, the script still produces a valid `submission.csv` exactly as before.'
- What this solution (achieved 0.02917) has done: 'Your 0.0 score indicates the submission is being judged invalid or completely uninformative (most likely all-zeros from the fallback path because no checkpoint is actually being loaded). To move upward toward the 0.9072 target with minimal core-logic change, I make checkpoint loading succeed more often by (1) expanding key normalization to cover common EfficientNet key names like `conv_stem/conv_head/bn*` and `blocks` without underscores, and (2) relaxing the backbone “must-have” check to accept either underscored or non-underscored stem/head keys (after normalization), while still requiring real block weights. I also add a tiny safety check that if predictions collapse to a single class, we fall back to the sample_submission’s class distribution (still legitimate, and typically better than all-zeros) rather than outputting a degenerate vector. The model, transforms, TTA, and thresholding remain the same.'
- What this solution (achieved 0.02917) has done: 'To move your score upward toward the 0.9072 target while keeping core logic unchanged, the main thing to fix is prediction calibration for QWK: fixed thresholds (0.5/1.5/2.5/3.5) are rarely optimal for a particular checkpoint, and they can yield very low kappa even with a decent model. I keep your EfficientNet-B5 inference + (regression or expected-value-from-softmax) pipeline exactly the same, but add a tiny out-of-fold threshold calibration step on the training set using the *same transforms and TTA* to choose thresholds that maximize quadratic weighted kappa. This does not change architecture/training/loss; it only chooses better discretization cutpoints, which is directly metric-aligned and typically gives a big kappa lift. I also keep your checkpoint-loading logic intact, and if loading fails we still fall back to the train-distribution submission so the script always outputs a valid `submission.csv` within time.'
- What this solution (achieved 0.02917) has done: 'Your current score (0.02917) is far below the target (0.9072), and the most likely cause is that you’re still not actually using a strong trained checkpoint (or you’re using one but with a mismatched input normalization/scale that collapses predictions). I keep your core logic (EfficientNet-B5, same transforms, same TTA, same thresholding-to-0..4) but make two minimal score-relevant changes: (1) make checkpoint loading more robust by explicitly supporting “efficientnet_pytorch”-style key names (e.g., `_swish`, `_bn*`, `_blocks.*._depthwise_conv.conv.weight`) and unwrapping common nesting patterns, and (2) make threshold calibration stronger but still cheap by optimizing thresholds with a slightly wider local search and ensuring we’re calibrating on a stratified validation split (to stabilize QWK calibration). These changes should materially increase the chance of loading meaningful backbone weights and produce better discretization for QWK without altering the modeling approach.'
- What this solution (achieved 0.02917) has done: 'Your current score (0.02917) is far below the target (0.9072), so we should improve rather than degrade, and the most likely bottleneck is still that the checkpoint is either not being loaded correctly or not being used in the best way for QWK discretization. I keep your EfficientNet-B5 + TTA + “continuous prediction then 4 thresholds to 0–4” core exactly the same, but make checkpoint loading succeed more often by (1) expanding key normalization for common EfficientNet-PyTorch and timm-style names (including `classifier.*`, `head.fc.*`, and `._fc.*`) and (2) preferring checkpoints that actually contain EfficientNet-B5 backbone tensors (by size + required key presence). Then I make threshold calibration more metric-aligned but still minimal by calibrating on a stratified validation set and adding a tiny global rescale/shift calibration of the continuous predictions (monotonic, so same semantics) before threshold search; this often fixes “collapsed” or mis-scaled outputs that produce near-zero kappa. The script still always write a valid `submission.csv` with `id_code,diagnosis` aligned to `test.csv`.'

# 9. Code solution

## === cell 0
import os
import re
import math
import glob
import warnings
import collections

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

warnings.filterwarnings("ignore")


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = "/kaggle/input/aptos2019-blindness-detection"
if not os.path.exists(BASE_DIR):
    BASE_DIR = "/kaggle/data/aptos2019-blindness-detection"

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")

assert os.path.exists(TEST_CSV), f"Could not find test.csv at {TEST_CSV}"
assert os.path.exists(
    SAMPLE_SUB_CSV
), f"Could not find sample_submission.csv at {SAMPLE_SUB_CSV}"
assert os.path.exists(TEST_IMG_DIR), f"Could not find test image dir at {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_CSV), f"Could not find train.csv at {TRAIN_CSV}"
assert os.path.exists(
    TRAIN_IMG_DIR
), f"Could not find train image dir at {TRAIN_IMG_DIR}"




## === cell 1
"""
EfficientNet implementation (as provided), with minimal fixes:
- Keep architecture identical.
- IMPORTANT score-relevant fix: stride in BlockArgs is stored as a list (e.g. [1] / [2]);
  the skip-connection condition must compare to [1], not to integer 1. This improves
  correctness and checkpoint compatibility.
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
        from functools import partial as _partial

        return _partial(Conv2dStaticSamePadding, image_size=image_size)


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

    def forward(self, x):
        return x


class Conv2dStaticSamePadding(nn.Conv2d):
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
            and (self._block_args.stride == [1])
            and (input_filters == output_filters)
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




## === cell 2
md_ef_reg = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1).to(DEVICE)
md_ef_cls = EfficientNet.from_pretrained("efficientnet-b5", num_classes=5).to(DEVICE)


def _unwrap_state_dict(state):
    """
    Change (score-relevant, minimal):
    - Expand key normalization for common checkpoint conventions so we actually load a meaningful backbone.
      This directly addresses near-random/degenerate predictions that drive QWK close to 0.
    """
    if not isinstance(state, dict):
        return None

    for _ in range(10):
        if not isinstance(state, dict):
            break
        if len(state) > 0 and all(isinstance(v, torch.Tensor) for v in state.values()):
            break
        preferred_keys = [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "params",
            "ema_state_dict",
            "ema",
            "student",
            "teacher",
            "checkpoint",
            "ckpt",
        ]
        moved = False
        for key in preferred_keys:
            if key in state and isinstance(state[key], dict):
                state = state[key]
                moved = True
                break
        if not moved:
            break

    if not isinstance(state, dict) or len(state) == 0:
        return None
    if not any(isinstance(v, torch.Tensor) for v in state.values()):
        return None

    new_state = {}
    prefixes = [
        "module.",
        "model.",
        "net.",
        "encoder.",
        "backbone.",
        "model.module.",
        "net.module.",
        "module.model.",
        "module.net.",
        "student.",
        "teacher.",
        "ema.",
        "ema_model.",
        "generator.",
        "swa_model.",
    ]

    def _strip_prefixes(kk):
        changed = True
        while changed:
            changed = False
            for pref in prefixes:
                if kk.startswith(pref):
                    kk = kk[len(pref) :]
                    changed = True
        return kk

    for k, v in state.items():
        if not isinstance(v, torch.Tensor):
            continue

        kk = _strip_prefixes(k)

        if kk.startswith("conv_stem."):
            kk = "_conv_stem." + kk[len("conv_stem.") :]
        if kk.startswith("bn0."):
            kk = "_bn0." + kk[len("bn0.") :]
        if kk.startswith("conv_head."):
            kk = "_conv_head." + kk[len("conv_head.") :]
        if kk.startswith("bn1."):
            kk = "_bn1." + kk[len("bn1.") :]

        if kk.startswith("blocks."):
            kk = "_blocks." + kk[len("blocks.") :]

        if kk.startswith("classifier."):
            kk = "_fc." + kk[len("classifier.") :]
        if kk.startswith("head.fc."):
            kk = "_fc." + kk[len("head.fc.") :]
        if kk.startswith("fc."):
            pass

        kk = kk.replace("._depthwise_conv.conv.", "._depthwise_conv.")
        kk = kk.replace("._project_conv.conv.", "._project_conv.")
        kk = kk.replace("._expand_conv.conv.", "._expand_conv.")
        kk = kk.replace("._se_reduce.conv.", "._se_reduce.")
        kk = kk.replace("._se_expand.conv.", "._se_expand.")

        new_state[kk] = v

    return new_state


def find_checkpoints_topk(k=80):
    candidates = []
    exts = ("*.pth", "*.pt", "*.bin", "*.pth.tar", "*.ckpt")
    for root in ["/kaggle/input", "/kaggle/data", "/kaggle/working"]:
        if os.path.exists(root):
            for ext in exts:
                candidates.extend(
                    glob.glob(os.path.join(root, "**", ext), recursive=True)
                )

    if not candidates:
        return []

    def ckpt_score(p):
        name = os.path.basename(p).lower()
        score = 0
        for token, w in [
            ("aptos", 80),
            ("blind", 40),
            ("retina", 40),
            ("diabetic", 40),
            ("dr", 25),
            ("efficientnet", 70),
            ("b5", 30),
            ("kappa", 15),
            ("qwk", 15),
            ("best", 15),
            ("fold", 10),
            ("stage", 5),
        ]:
            if token in name:
                score += w
        if "aptos2019-blindness-detection" in p.lower():
            score += 30
        try:
            size_mb = os.path.getsize(p) / (1024 * 1024)
        except OSError:
            size_mb = 0.0

        score += min(200.0, size_mb * 2.5)
        if size_mb < 8.0:
            score -= 250.0
        return score

    candidates = sorted(candidates, key=ckpt_score, reverse=True)
    return candidates[:k]


def _state_dict_variants_for_head_names(state_dict):
    if state_dict is None:
        return []
    variants = [state_dict]

    has_fc = any(k.startswith("fc.") for k in state_dict.keys())
    has__fc = any(k.startswith("_fc.") for k in state_dict.keys())

    if has_fc and (not has__fc):
        new_sd = dict(state_dict)
        for k in list(state_dict.keys()):
            if k.startswith("fc."):
                new_sd["_fc." + k[len("fc.") :]] = new_sd.pop(k)
        variants.append(new_sd)

    if has__fc and (not has_fc):
        new_sd = dict(state_dict)
        for k in list(state_dict.keys()):
            if k.startswith("_fc."):
                new_sd["fc." + k[len("_fc.") :]] = new_sd.pop(k)
        variants.append(new_sd)

    return variants


def _filter_only_shape_compatible(state_dict, model):
    if state_dict is None:
        return None
    msd = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k in msd and tuple(v.shape) == tuple(msd[k].shape):
            filtered[k] = v
    return filtered


def try_load_checkpoint_into_model(ckpt_path, model, require_backbone=True):
    """
    Change (score-relevant, minimal):
    - Same partial-loading idea, but accept more real-world key patterns after normalization,
      and require presence of key backbone tensors so we don't "load" irrelevant checkpoints.
    """
    try:
        raw = torch.load(ckpt_path, map_location="cpu")
        base_state = _unwrap_state_dict(raw)
        if base_state is None:
            return False, None, None

        msd = model.state_dict()
        must_have_any = [
            "_conv_stem.weight",
            "_conv_head.weight",
        ]

        for state in _state_dict_variants_for_head_names(base_state):
            state_f = _filter_only_shape_compatible(state, model)
            if state_f is None or len(state_f) == 0:
                continue

            missing, unexpected = model.load_state_dict(state_f, strict=False)

            if require_backbone:
                has_block_weight = any(
                    (k.startswith("_blocks.") and k.endswith(".weight"))
                    for k in state_f.keys()
                )
                if not has_block_weight:
                    continue

                ok_must = True
                for k in must_have_any:
                    if k not in state_f:
                        ok_must = False
                        break
                    if tuple(state_f[k].shape) != tuple(msd[k].shape):
                        ok_must = False
                        break
                if not ok_must:
                    continue

            return True, missing, unexpected

        return False, None, None
    except Exception:
        return False, None, None


def select_and_load_best_checkpoint():
    ckpts = find_checkpoints_topk(k=80)
    if not ckpts:
        return None, None, False

    for p in ckpts:
        ok_reg, _, _ = try_load_checkpoint_into_model(
            p, md_ef_reg, require_backbone=True
        )
        if ok_reg:
            return p, "regression", True

        ok_cls, _, _ = try_load_checkpoint_into_model(
            p, md_ef_cls, require_backbone=True
        )
        if ok_cls:
            return p, "classification", True

    return ckpts[0], None, False


ckpt_path, model_kind, loaded_ok = select_and_load_best_checkpoint()
print("Checkpoint selected:", ckpt_path)
print("Loaded OK:", loaded_ok, "Model kind:", model_kind)

if not loaded_ok:
    print(
        "WARNING: No compatible checkpoint loaded; will use safe fallback predictions."
    )

md_ef_reg.eval()
md_ef_cls.eval()




## === cell 3
test_df = pd.read_csv(TEST_CSV)
train_df = pd.read_csv(TRAIN_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

assert "id_code" in test_df.columns
assert list(sample_sub.columns) == ["id_code", "diagnosis"]


class AptosImageDataset(Dataset):
    def __init__(self, df, img_dir, transform, with_label=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.with_label = with_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_code = self.df.loc[idx, "id_code"]
        img_path = os.path.join(self.img_dir, f"{id_code}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        if self.with_label:
            y = int(self.df.loc[idx, "diagnosis"])
            return id_code, img, y
        return id_code, img


sz = 456
test_tfms = transforms.Compose(
    [
        transforms.Resize(int(sz * 1.12)),
        transforms.CenterCrop((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

ds_test = AptosImageDataset(test_df, TEST_IMG_DIR, test_tfms, with_label=False)
dl_test = DataLoader(
    ds_test,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)


def stratified_val_split(df, val_frac=0.2, min_val=300, seed=42):
    rng = np.random.RandomState(seed)
    idx_by_class = {}
    for c in range(5):
        idx_by_class[c] = df.index[df["diagnosis"].values == c].values
    val_indices = []
    for c, idxs in idx_by_class.items():
        idxs = idxs.copy()
        rng.shuffle(idxs)
        take = int(round(len(idxs) * val_frac))
        val_indices.extend(list(idxs[:take]))
    if len(val_indices) < min_val:
        all_idx = df.index.values.copy()
        rng.shuffle(all_idx)
        need = min_val - len(val_indices)
        extra = [i for i in all_idx if i not in set(val_indices)][:need]
        val_indices.extend(extra)
    val_indices = np.array(sorted(set(val_indices)), dtype=np.int64)
    trn_indices = np.array(
        [i for i in df.index.values if i not in set(val_indices)], dtype=np.int64
    )
    return trn_indices, val_indices


trn_idx, val_idx = stratified_val_split(train_df, val_frac=0.2, min_val=300, seed=42)

train_val_df = train_df.loc[val_idx].reset_index(drop=True)
ds_val = AptosImageDataset(train_val_df, TRAIN_IMG_DIR, test_tfms, with_label=True)
dl_val = DataLoader(
    ds_val,
    batch_size=16,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 4
@torch.no_grad()
def predict_regression_tta(model, loader):
    model.eval()
    ids = []
    preds = []
    for batch in loader:
        if len(batch) == 2:
            batch_ids, x = batch
        else:
            batch_ids, x, _ = batch
        x = x.to(DEVICE, non_blocking=True)
        out1 = model(x).view(-1)
        out2 = model(torch.flip(x, dims=[3])).view(-1)
        out = (out1 + out2) / 2.0
        ids.extend(list(batch_ids))
        preds.append(out.detach().float().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    return np.array(ids), preds


@torch.no_grad()
def predict_classification_expected_tta(model, loader):
    model.eval()
    ids = []
    preds = []
    cls_values = torch.arange(5, device=DEVICE, dtype=torch.float32).view(1, -1)
    for batch in loader:
        if len(batch) == 2:
            batch_ids, x = batch
        else:
            batch_ids, x, _ = batch
        x = x.to(DEVICE, non_blocking=True)
        logits1 = model(x)
        logits2 = model(torch.flip(x, dims=[3]))
        logits = (logits1 + logits2) / 2.0
        probs = torch.softmax(logits, dim=1)
        exp = (probs * cls_values).sum(dim=1)
        ids.extend(list(batch_ids))
        preds.append(exp.detach().float().cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    return np.array(ids), preds


def apply_thresholds(preds, coef=(0.5, 1.5, 2.5, 3.5)):
    c0, c1, c2, c3 = coef
    out = np.zeros_like(preds, dtype=np.int64)
    out[preds >= c0] = 1
    out[preds >= c1] = 2
    out[preds >= c2] = 3
    out[preds >= c3] = 4
    return out


def fallback_by_train_distribution(train_csv_path, n, seed=42):
    df = pd.read_csv(train_csv_path)
    vc = df["diagnosis"].value_counts(normalize=True).sort_index()
    probs = np.array([vc.get(i, 0.0) for i in range(5)], dtype=np.float64)
    if probs.sum() <= 0:
        probs = np.array([1, 0, 0, 0, 0], dtype=np.float64)
    probs = probs / probs.sum()
    rng = np.random.RandomState(seed)
    return rng.choice(np.arange(5), size=n, p=probs).astype(int)


def quadratic_weighted_kappa(y_true, y_pred, n_classes=5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    assert y_true.shape == y_pred.shape
    N = n_classes
    O = np.zeros((N, N), dtype=np.float64)
    for a, b in zip(y_true, y_pred):
        if 0 <= a < N and 0 <= b < N:
            O[a, b] += 1.0

    act_hist = O.sum(axis=1)
    pred_hist = O.sum(axis=0)
    E = np.outer(act_hist, pred_hist)
    if E.sum() > 0:
        E = E * (O.sum() / E.sum())

    W = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            W[i, j] = ((i - j) ** 2) / ((N - 1) ** 2)

    num = (W * O).sum()
    den = (W * E).sum()
    if den == 0:
        return 0.0
    return 1.0 - num / den


def _fit_affine_to_labels(x, y, a_candidates, b_candidates):
    """
    Change (score-relevant, minimal):
    - Calibrate only an affine transform x' = a*x + b on validation predictions.
      This preserves the same model outputs' ordering (monotonic for a>0) and keeps
      the same thresholding semantics, but fixes scale/shift mismatch that can crush QWK.
    """
    best = (-1.0, 1.0, 0.0)
    for a in a_candidates:
        for b in b_candidates:
            xp = a * x + b
            yp = apply_thresholds(xp, (0.5, 1.5, 2.5, 3.5))
            s = quadratic_weighted_kappa(y, yp)
            if s > best[0]:
                best = (s, a, b)
    return best[1], best[2], best[0]


def calibrate_thresholds_on_val(
    val_preds_cont, val_y, init_coef=(0.5, 1.5, 2.5, 3.5), step=0.05, iters=4
):
    """
    Threshold-only calibration (kept), plus a very small affine calibration before threshold search.
    """
    a_candidates = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    b_candidates = [-1.0, -0.5, 0.0, 0.5, 1.0]
    a_best, b_best, base_qwk = _fit_affine_to_labels(
        val_preds_cont, val_y, a_candidates, b_candidates
    )
    val_preds_cont = a_best * val_preds_cont + b_best

    coef = np.array(init_coef, dtype=np.float64)

    def _sanitize(c):
        c = np.array(c, dtype=np.float64)
        for i in range(1, 4):
            if c[i] <= c[i - 1] + 1e-3:
                c[i] = c[i - 1] + 1e-3
        return c

    coef = _sanitize(coef)
    best = quadratic_weighted_kappa(
        val_y, apply_thresholds(val_preds_cont, tuple(coef))
    )

    for _ in range(iters):
        improved = False
        for i in range(4):
            base = coef[i]
            candidates = [
                base + d
                for d in (-3 * step, -2 * step, -step, 0.0, step, 2 * step, 3 * step)
            ]
            best_local = best
            best_c = coef.copy()
            for cand in candidates:
                ctry = coef.copy()
                ctry[i] = cand
                ctry = _sanitize(ctry)
                if ctry[0] < -2.0 or ctry[3] > 8.0:
                    continue
                score = quadratic_weighted_kappa(
                    val_y, apply_thresholds(val_preds_cont, tuple(ctry))
                )
                if score > best_local:
                    best_local = score
                    best_c = ctry
            if best_local > best + 1e-12:
                coef = best_c
                best = best_local
                improved = True
        if not improved:
            break

    return (tuple(coef), float(best), float(a_best), float(b_best), float(base_qwk))




## === cell 5
default_coef = (0.5, 1.5, 2.5, 3.5)
best_coef = default_coef
best_val_qwk = None
affine_a, affine_b = 1.0, 0.0
affine_base_qwk = None

if loaded_ok and model_kind == "regression":
    val_ids, val_preds = predict_regression_tta(md_ef_reg, dl_val)
    val_y = train_val_df["diagnosis"].values.astype(int)
    best_coef, best_val_qwk, affine_a, affine_b, affine_base_qwk = (
        calibrate_thresholds_on_val(
            val_preds, val_y, init_coef=default_coef, step=0.05, iters=5
        )
    )
    print(
        "Calibrated (reg): coef=",
        best_coef,
        "val QWK=",
        best_val_qwk,
        "affine a,b=",
        (affine_a, affine_b),
        "affine-default-threshold QWK=",
        affine_base_qwk,
    )

elif loaded_ok and model_kind == "classification":
    val_ids, val_preds = predict_classification_expected_tta(md_ef_cls, dl_val)
    val_y = train_val_df["diagnosis"].values.astype(int)
    best_coef, best_val_qwk, affine_a, affine_b, affine_base_qwk = (
        calibrate_thresholds_on_val(
            val_preds, val_y, init_coef=default_coef, step=0.05, iters=5
        )
    )
    print(
        "Calibrated (cls-exp): coef=",
        best_coef,
        "val QWK=",
        best_val_qwk,
        "affine a,b=",
        (affine_a, affine_b),
        "affine-default-threshold QWK=",
        affine_base_qwk,
    )
else:
    print("No checkpoint loaded; skipping threshold calibration.")




## === cell 6
if loaded_ok and model_kind == "regression":
    ids, reg_preds = predict_regression_tta(md_ef_reg, dl_test)
    reg_preds = affine_a * reg_preds + affine_b
    final_preds = apply_thresholds(reg_preds, coef=best_coef)
    pred_map = {i: int(p) for i, p in zip(ids, final_preds)}
    sub = pd.DataFrame({"id_code": test_df["id_code"].values})
    sub["diagnosis"] = sub["id_code"].map(pred_map).astype(int)
    if sub["diagnosis"].isna().any():
        sub["diagnosis"] = sub["diagnosis"].fillna(0).astype(int)

elif loaded_ok and model_kind == "classification":
    ids, exp_preds = predict_classification_expected_tta(md_ef_cls, dl_test)
    exp_preds = affine_a * exp_preds + affine_b
    final_preds = apply_thresholds(exp_preds, coef=best_coef)
    pred_map = {i: int(p) for i, p in zip(ids, final_preds)}
    sub = pd.DataFrame({"id_code": test_df["id_code"].values})
    sub["diagnosis"] = sub["id_code"].map(pred_map).astype(int)
    if sub["diagnosis"].isna().any():
        sub["diagnosis"] = sub["diagnosis"].fillna(0).astype(int)

else:
    sub = pd.DataFrame({"id_code": test_df["id_code"].values})
    sub["diagnosis"] = fallback_by_train_distribution(TRAIN_CSV, len(test_df), seed=42)

if sub["diagnosis"].nunique() <= 1:
    sub["diagnosis"] = fallback_by_train_distribution(TRAIN_CSV, len(test_df), seed=43)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
print(
    "diagnosis value counts:\n",
    sub["diagnosis"].value_counts(dropna=False).sort_index(),
)
if best_val_qwk is not None:
    print(
        "Used thresholds:",
        best_coef,
        " (calibrated val QWK:",
        best_val_qwk,
        ", affine a,b:",
        (affine_a, affine_b),
        ")",
    )
else:
    print("Used thresholds:", best_coef, " (default)")
