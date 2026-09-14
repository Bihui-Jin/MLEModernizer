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

0.8982007706089853

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the import/runtime errors caused by missing `fastai` in this environment by replacing the fastai `DataBunch/Learner` pipeline with a minimal pure-PyTorch inference wrapper, while keeping your core EfficientNet model definition and the same rounding-to-5-classes submission logic. I also fix the pretrained-weight loading so it doesn’t require internet (it load from a local `.pth` if present, otherwise it still run and produce a valid submission, just with low score). Finally, I correct the dataset paths to match the provided `/kaggle/input/...` structure and ensure `submission.csv` is written with the required `id_code,diagnosis` columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is almost certainly because the checkpoint `models/abcdef.pth` is missing, so you are effectively submitting random/untrained EfficientNet outputs. The smallest change that should move the score toward your target is to load a real local checkpoint if one exists (auto-detect the best `.pth` in `models/` instead of hardcoding a nonexistent filename), while keeping the same model architecture and the same fixed rounding coefficients. To avoid accidentally loading the wrong shape, the patch also prefers checkpoints whose state_dict looks compatible with the current model and reports what was loaded. Everything else (data paths, preprocessing, inference loop, and submission formatting) remains the same.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with submitting essentially untrained/random predictions because no compatible `.pth` weights are being loaded from `models/`. The minimal change to move the score up toward your target is to (1) search common Kaggle locations for a real checkpoint (including `/kaggle/input/**`) and (2) only load it if it matches your model’s key/shape signature, otherwise keep current behavior to avoid silent misloads. I’m also keeping your existing EfficientNet-B5 regression + fixed OptimizedRounder thresholds unchanged, but I make inference deterministic and slightly more robust by using `torch.inference_mode()` and ensuring test order matches `sample_submission.csv` ordering (prevents accidental id misalignment). The script still writes a valid `submission.csv` with `id_code,diagnosis`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with effectively untrained/random predictions because no compatible checkpoint is being loaded (the script falls back to random weights). The smallest change that should move the score up toward your target is to (1) also search common locations for a real EfficientNet-B5 APTOS checkpoint (especially inside the competition folder under `/kaggle/input/aptos2019-blindness-detection/`) and (2) require the checkpoint to match the EfficientNet-B5 head shape (so we don’t silently load the wrong model). I’m keeping your exact model architecture and the same fixed rounding coefficients; the only other change is to run inference at the model’s native image size (456 for EfficientNet-B5) to better match how such checkpoints are typically trained, which should improve kappa without changing the core approach. The pipeline still runs end-to-end and writes `submission.csv` with `id_code,diagnosis`.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 is consistent with submitting essentially random predictions because no trained checkpoint is being loaded (the code frequently won’t find a compatible `.pth` in the listed locations). The smallest change likely to move your score sharply upward toward the 0.898 target is to (1) also look for a checkpoint path explicitly provided via an environment variable, and (2) broaden checkpoint compatibility by supporting common key prefixes (e.g., `model.`, `net.`) and allowing a head-size mismatch by dropping `_fc.*` weights when the backbone matches. This keeps your core EfficientNet-B5 regression + fixed-threshold rounding logic identical, but makes weight loading much more likely to succeed in Kaggle without internet. If no checkpoint exists anywhere, the script still produce a valid `submission.csv` (but score remain low).'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most consistent with either (a) no trained checkpoint actually being loaded (so predictions are essentially random) or (b) producing constant/degenerate predictions because the model output scale doesn’t match the fixed thresholds. I make the smallest score-relevant change by adding a lightweight, training-set-based calibration step that fits the 4 rounding thresholds on out-of-fold predictions from your existing EfficientNet-B5 regression model (same architecture/inference, just better class mapping for QWK). This preserves your core model and prediction loop, but replaces the hardcoded coefficients with coefficients optimized for kappa on a held-out fold, which should move the score up toward your target when a reasonable checkpoint is present. I also keep the existing “auto-find checkpoint” logic and ensure the submission stays aligned to `sample_submission.csv` order.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is overwhelmingly likely to be caused by submitting essentially untrained/random predictions because no compatible checkpoint is actually being loaded; the biggest score gain with the smallest semantic change is to ensure we *reliably* load a real APTOS-trained weight file if one exists. I keep your EfficientNet-B5 regression model and inference identical, but make checkpoint discovery stricter (prefer filenames that look like aptos/efficientnet/b5, and require backbone key shape matches) while also correctly handling common checkpoint formats (`{'model': ..., 'state_dict': ...}` etc.). Then, to improve kappa mapping without changing the model, I fit thresholds using a deterministic split and additionally guard against degenerate predictions (all same bin) by falling back to the initial coefficients. The script still run end-to-end and always write a valid `submission.csv` with `id_code,diagnosis`.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is most likely coming from submitting effectively random/degenerate predictions because no real trained checkpoint is being loaded (the code quietly falls back to random initialization). I make a minimal, score-relevant change to checkpoint discovery so it preferentially loads a real APTOS EfficientNet-B5 regression checkpoint if one exists anywhere under `/kaggle/input`, and I broaden compatibility for common checkpoint formats (including `.pth`/`.pt` with nested keys and `DataParallel` prefixes). I also add a tiny guard to ensure predicted labels are valid ints in `[0,4]` and that the submission order exactly matches `sample_submission.csv` (to avoid accidental id misalignment). Core model architecture, inference loop, and thresholding approach remain the same; this is just to reliably load weights and prevent silent failure modes.'
- What this solution (achieved 0.0) has done: 'Your score is 0.0 because you are almost certainly not loading any trained weights (the code falls back to random initialization when no compatible checkpoint is found), so the predictions are essentially noise. The smallest score-relevant change is to add an on-disk checkpoint fallback by downloading a known APTOS EfficientNet-B5 regression checkpoint from Kaggle *if available in the environment* (i.e., from `/kaggle/input`), and to make our checkpoint loader slightly more permissive by also accepting `OrderedDict`-style checkpoints and `ema_state_dict` keys. I’m keeping your model, inference, and threshold-optimization logic identical; I’m only improving the likelihood that real weights are actually loaded, and I add a strict “loaded weights sanity check” (non-degenerate output variance) to avoid silently submitting junk. The script still runs end-to-end and always writes a valid `submission.csv` with `id_code,diagnosis`.'
- What this solution (achieved 0.0) has done: 'I remove the hard-fail behaviors that currently prevent any submission from being written when a checkpoint isn’t found or when predictions are nearly constant, because your current score is “Not yielded” and the primary bug is that the pipeline stops before producing `submission.csv`. I keep your EfficientNet-B5 regression model, preprocessing, and thresholding logic unchanged, but make checkpoint loading best-effort: it try to load a compatible checkpoint and otherwise continue with random weights while clearly warning (so you still get a valid CSV). I also keep the test ordering aligned to `sample_submission.csv` and ensure the output is valid ints in `[0,4]`. This is the smallest change that makes the notebook run end-to-end and generate a submission; if you provide a real checkpoint via `CKPT_PATH` or by placing it under `./models`, score should move toward the target.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with the model running with random/untrained weights (no compatible checkpoint found), so the most score-relevant minimal change is to reliably load an actual EfficientNet-B5 checkpoint if it exists in the environment. I add a deterministic “search + quick compatibility check” that also accepts common checkpoint layouts and automatically adapts your model head to the checkpoint’s `_fc.*` shape (so we don’t discard the head when it’s present), then keep your exact inference + threshold-fitting logic the same. I also add a small, safe sanity print that confirms whether weights were loaded and what head shape was used (to avoid silent failures that lead to 0.0 submissions). Everything else (architecture blocks, preprocessing, regression output + OptimizedRounder mapping, and submission formatting/order) stays unchanged.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is still most consistent with not actually loading a meaningful trained checkpoint and/or producing predictions on a different preprocessing than the checkpoint expects. To move the score up with minimal semantic change, I (1) make checkpoint discovery/load more reliable by also considering `.pth/.pt` files next to the dataset and preferring ones whose keys/shapes match your EfficientNet-B5 backbone, (2) add the standard APTOS “crop black borders around retina” preprocessing (keeps the same model/inference, but matches common training pipelines and usually improves QWK a lot), and (3) keep your exact regression+thresholding logic but ensure threshold fitting uses a representative stratified validation split (avoids a label-skewed first-chunk split that can hurt kappa mapping). The script still runs end-to-end and always writes `submission.csv` in the required format.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is still most consistent with weights not being loaded (random model) and/or misloading a checkpoint silently, so the smallest score-improving change is to make checkpoint loading both more reliable and verifiable. I add a strict “backbone key shape” compatibility check and only then load, plus handle common checkpoint formats/prefixes and optionally adapt the classification head if the checkpoint’s `_fc.*` shape differs. I also fix the stratified validation sampling (it currently overshoots/duplicates logic) so threshold fitting is based on a proper stratified holdout, which should improve the quadratic kappa mapping without changing model/inference semantics. Everything else (EfficientNet-B5 definition, preprocessing including crop, inference loop, and submission format) remains the same, and the script always write `submission.csv`.'

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

from sklearn.metrics import cohen_kappa_score
from sklearn import metrics


def seed_everything(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)



## === cell 1
"""
EfficientNet helper functions + model definition (minimal fixes for runtime correctness).
Key fix: BlockArgs uses 'stride' but encode referenced 'strides' -> corrected to 'stride'.
We keep the architecture identical to the provided code.
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
        pad_w = max((ow - 1) * self.stride[1] + (kh - 1) * self.dilation[1] + 1 - iw, 0)
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
        ih, iw = (
            image_size if isinstance(image_size, list) else [image_size, image_size]
        )
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
    def _encode_block_string(block):
        args = [
            "r%d" % block.num_repeat,
            "k%d" % block.kernel_size,
            "s%d%d" % (block.stride[0], block.stride[0]),
            "e%s" % block.expand_ratio,
            "i%d" % block.input_filters,
            "o%d" % block.output_filters,
        ]
        if 0 < (block.se_ratio or 0) <= 1:
            args.append("se%s" % block.se_ratio)
        if block.id_skip is False:
            args.append("noskip")
        return "_".join(args)

    @staticmethod
    def decode(string_list):
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


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
            and self._block_args.stride[0] == 1
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
MODEL_NAME = "efficientnet-b5"



## === cell 3
os.makedirs("models", exist_ok=True)
available_pths = sorted(
    [f for f in os.listdir("models") if f.endswith(".pth") or f.endswith(".pt")]
)
print("models/ contains:", available_pths)




## === cell 4
def get_df():
    base_image_dir = os.path.join(
        "/", "kaggle", "input", "aptos2019-blindness-detection"
    )
    train_dir = os.path.join(base_image_dir, "train_images")
    test_dir = os.path.join(base_image_dir, "test_images")

    train_csv = os.path.join(base_image_dir, "train.csv")
    test_csv = os.path.join(base_image_dir, "test.csv")
    sample_csv = os.path.join(base_image_dir, "sample_submission.csv")

    df = pd.read_csv(train_csv)
    df["path"] = df["id_code"].map(lambda x: os.path.join(train_dir, f"{x}.png"))
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    test_df = pd.read_csv(test_csv)
    test_df["path"] = test_df["id_code"].map(
        lambda x: os.path.join(test_dir, f"{x}.png")
    )

    sample_sub = pd.read_csv(sample_csv)

    test_df = sample_sub[["id_code"]].merge(test_df, on="id_code", how="left")

    return df, test_df, sample_sub


df, test_df, sample_sub = get_df()
print(df.shape, test_df.shape, sample_sub.shape)
print(df.head())
print(test_df.head())



## === cell 5
from PIL import Image

bs = 16
sz = int(EfficientNet.get_image_size(MODEL_NAME))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _crop_black_borders_rgb(img, tol=7):
    """
    Score-relevant: APTOS models are commonly trained with a "crop black border" step.
    Keeps the same resize->normalize->forward inference pipeline, but reduces background shift.
    """
    arr = np.asarray(img)
    if arr.ndim != 3:
        return img
    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).astype(
        np.float32
    )
    mask = gray > float(tol)
    if not np.any(mask):
        return img
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    if (y1 - y0) < 10 or (x1 - x0) < 10:
        return img
    return img.crop((int(x0), int(y0), int(x1), int(y1)))


def load_image_tensor(path, size):
    img = Image.open(path).convert("RGB")
    img = _crop_black_borders_rgb(img, tol=7)
    img = img.resize((size, size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(arr)




## === cell 6
def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, collections.OrderedDict):
        return ckpt_obj
    if isinstance(ckpt_obj, dict):
        for k in (
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "weights",
            "ema_state_dict",
        ):
            if k in ckpt_obj and isinstance(
                ckpt_obj[k], (dict, collections.OrderedDict)
            ):
                return ckpt_obj[k]
        return ckpt_obj
    return None


def _clean_state_dict_keys(state):
    new_state = {}
    for k, v in state.items():
        nk = k
        for pref in ("module.", "model.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        new_state[nk] = v
    return new_state


def _is_tensor(v):
    return torch.is_tensor(v) or isinstance(v, np.ndarray)


def _state_dict_compat_score(state, model, path_hint=""):
    if state is None:
        return -(10**9)

    model_sd = model.state_dict()
    score = 0

    strict_keys = [
        ("_conv_stem.weight", 350),
        ("_bn0.weight", 120),
        ("_conv_head.weight", 220),
        ("_bn1.weight", 120),
    ]
    for k, w in strict_keys:
        if k in state and k in model_sd and _is_tensor(state[k]):
            if tuple(state[k].shape) == tuple(model_sd[k].shape):
                score += w
            else:
                score -= w * 2
        else:
            score -= w

    if "_fc.weight" in state and _is_tensor(state["_fc.weight"]):
        score += 40

    overlap = len(set(model_sd.keys()).intersection(set(state.keys())))
    score += min(120, overlap // 15)

    hint = (path_hint or "").lower()
    for token, w in [
        ("aptos", 160),
        ("blindness", 90),
        ("retina", 60),
        ("diabetic", 60),
        ("efficientnet", 70),
        ("b5", 60),
        ("effnet", 30),
        ("reg", 20),
    ]:
        if token in hint:
            score += w
    for token, w in [("best", 30), ("fold", 10), ("kappa", 20), ("qwk", 20)]:
        if token in hint:
            score += w

    return score


def _find_candidate_checkpoints(search_roots):
    exts = (".pth", ".pt", ".bin")
    cands = []
    for root in search_roots:
        if not root or not os.path.exists(root):
            continue
        if os.path.isfile(root) and root.lower().endswith(exts):
            cands.append(root)
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for fn in filenames:
                if fn.lower().endswith(exts):
                    cands.append(os.path.join(dirpath, fn))
    cands = sorted(set(cands), key=lambda p: os.path.getmtime(p), reverse=True)
    return cands


def _infer_num_classes_from_state(state):
    if not isinstance(state, (dict, collections.OrderedDict)):
        return None
    w = state.get("_fc.weight", None)
    b = state.get("_fc.bias", None)
    if _is_tensor(w):
        try:
            out_features = int(w.shape[0])
            if b is None or (hasattr(b, "shape") and int(b.shape[0]) == out_features):
                if 1 <= out_features <= 100:
                    return out_features
        except Exception:
            return None
    return None


def _backbone_shapes_match(state, model):
    """
    Score-relevant minimal change: avoid silently loading wrong checkpoints.
    If backbone shapes don't match, we skip that checkpoint, preventing "random-looking" outputs.
    """
    if state is None:
        return False
    msd = model.state_dict()
    for k in ("_conv_stem.weight", "_conv_head.weight"):
        if k not in state or k not in msd:
            return False
        if not _is_tensor(state[k]):
            return False
        if tuple(state[k].shape) != tuple(msd[k].shape):
            return False
    return True


def _adapt_fc_if_needed(model, state):
    """
    Score-relevant minimal change: if checkpoint contains an _fc head with different out_features,
    reinitialize model._fc to that shape so the head weights can be loaded (common in APTOS ckpts).
    Architecture stays EfficientNet-B5; only output dimension follows the checkpoint.
    """
    if not isinstance(state, (dict, collections.OrderedDict)):
        return model
    w = state.get("_fc.weight", None)
    b = state.get("_fc.bias", None)
    if _is_tensor(w) and hasattr(w, "shape") and len(w.shape) == 2:
        out_f = int(w.shape[0])
        in_f = int(w.shape[1])
        if hasattr(model, "_fc") and isinstance(model._fc, nn.Linear):
            if model._fc.in_features == in_f and model._fc.out_features != out_f:
                model._fc = nn.Linear(in_f, out_f).to(next(model.parameters()).device)
    return model


explicit_ckpt = os.environ.get("CKPT_PATH", "").strip()
if explicit_ckpt:
    print("CKPT_PATH provided:", explicit_ckpt)

base_comp_dir = os.path.join("/", "kaggle", "input", "aptos2019-blindness-detection")

search_roots = [
    explicit_ckpt if explicit_ckpt else None,
    "models",
    "/kaggle/working/models",
    base_comp_dir,
    os.path.join(base_comp_dir, "models"),
    "/kaggle/input",
]

cand_paths = _find_candidate_checkpoints(search_roots)
print("Found candidate checkpoints:", len(cand_paths))
for p in cand_paths[:12]:
    print("  ", p)

_tmp_model = EfficientNet.from_pretrained(MODEL_NAME, num_classes=1).to("cpu")
_tmp_model.eval()

best_path = None
best_score = -(10**18)
best_state = None

for p in cand_paths[:1200]:
    try:
        ckpt_obj = torch.load(p, map_location="cpu")
        state = _extract_state_dict(ckpt_obj)
        state = (
            _clean_state_dict_keys(state)
            if isinstance(state, (dict, collections.OrderedDict))
            else None
        )
        if not _backbone_shapes_match(state, _tmp_model):
            continue
        sc = _state_dict_compat_score(state, _tmp_model, path_hint=p)
        if sc > best_score:
            best_score = sc
            best_path = p
            best_state = state
    except Exception:
        continue

ckpt_num_classes = _infer_num_classes_from_state(best_state)
if ckpt_num_classes is None:
    ckpt_num_classes = 1

md_ef = EfficientNet.from_pretrained(MODEL_NAME, num_classes=int(ckpt_num_classes)).to(
    DEVICE
)
md_ef.eval()

md_ef = _adapt_fc_if_needed(md_ef, best_state)

loaded_any = False
if (
    best_path is not None
    and best_score > 250
    and isinstance(best_state, (dict, collections.OrderedDict))
):
    missing, unexpected = md_ef.load_state_dict(dict(best_state), strict=False)
    loaded_any = True
    print("Loaded checkpoint:", best_path)
    print("Compat score:", int(best_score))
    print("Checkpoint inferred num_classes:", ckpt_num_classes)
    print("Model fc out_features:", int(md_ef._fc.out_features))
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
else:
    print(
        "WARNING: No sufficiently compatible checkpoint found. Continuing with randomly initialized weights "
        "(submission will be produced but score may be poor)."
    )
    print("Best candidate path:", best_path)
    print("Best compat score:", int(best_score))
    print("Checkpoint inferred num_classes:", ckpt_num_classes)

md_ef.eval()
print("Checkpoint loaded:", loaded_any)




## === cell 7
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


def fit_thresholds_bruteforce(preds, y, init=(0.57, 1.57, 2.77, 3.57), max_iter=8):
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.int64).reshape(-1)

    opt = OptimizedRounder()

    def kappa_for(coef):
        p = opt.predict(preds, coef)
        return metrics.cohen_kappa_score(y, p.astype(int), weights="quadratic")

    coef = list(map(float, init))
    best = kappa_for(coef)

    steps = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005][:max_iter]
    for step in steps:
        improved = True
        while improved:
            improved = False
            for i in range(4):
                for direction in (-1.0, 1.0):
                    cand = coef.copy()
                    cand[i] = cand[i] + direction * step
                    cand = sorted(cand)
                    if not (cand[0] < cand[1] < cand[2] < cand[3]):
                        continue
                    sc = kappa_for(cand)
                    if sc > best + 1e-8:
                        best = sc
                        coef = cand
                        improved = True
    return coef, best




## === cell 8
@torch.inference_mode()
def predict_regression(model, paths, size, batch_size=16):
    model.eval()
    preds = np.zeros((len(paths),), dtype=np.float32)
    for start in range(0, len(paths), batch_size):
        end = min(start + batch_size, len(paths))
        batch_paths = paths[start:end]
        batch = torch.stack(
            [load_image_tensor(p, size) for p in batch_paths], dim=0
        ).to(DEVICE)
        out = model(batch).view(-1)
        preds[start:end] = out.detach().float().cpu().numpy()
        if start == 0:
            print(
                "First batch output stats:",
                float(out.min()),
                float(out.max()),
                float(out.mean()),
                "std:",
                float(out.std()),
            )
    return preds




## === cell 9
def run_subm_torch(model, test_df, coefficients=None, size=256, batch_size=16):
    opt = OptimizedRounder()
    preds = predict_regression(
        model, test_df["path"].tolist(), size=size, batch_size=batch_size
    )

    if float(np.std(preds)) < 1e-6:
        print(
            "WARNING: Model outputs are nearly constant; submission will likely score poorly. "
            "If you have a checkpoint, set CKPT_PATH or place it in ./models."
        )

    if coefficients is None:
        coefficients = [0.57, 1.57, 2.77, 3.57]

    tst_pred = opt.predict(preds.reshape(-1), coefficients).astype(np.int64)
    tst_pred = np.clip(tst_pred, 0, 4).astype(int)

    out_df = test_df[["id_code"]].copy()
    out_df["diagnosis"] = tst_pred
    out_df.to_csv("submission.csv", index=False)
    print("done, wrote submission.csv with shape", out_df.shape)
    return out_df




## === cell 10
val_frac = 0.12
n = len(df)
val_n = max(250, int(n * val_frac))

rng = np.random.RandomState(42)
classes = sorted(df["diagnosis"].unique().tolist())
counts = df["diagnosis"].value_counts().to_dict()
alloc = {c: max(1, int(round(val_n * counts.get(c, 0) / float(n)))) for c in classes}
total = int(sum(alloc.values()))
if total != val_n:
    order = sorted(classes, key=lambda c: counts.get(c, 0), reverse=True)
    while total > val_n:
        for c in order:
            if total == val_n:
                break
            if alloc[c] > 1:
                alloc[c] -= 1
                total -= 1
    while total < val_n:
        for c in order:
            if total == val_n:
                break
            alloc[c] += 1
            total += 1

val_idx = []
for c in classes:
    idxs = df.index[df["diagnosis"] == c].values
    k = min(len(idxs), int(alloc[c]))
    if k > 0:
        chosen = rng.choice(idxs, size=k, replace=False)
        val_idx.extend(chosen.tolist())

val_idx = list(dict.fromkeys(val_idx))
if len(val_idx) < val_n:
    remaining = np.setdiff1d(
        df.index.values, np.array(val_idx, dtype=np.int64), assume_unique=False
    )
    extra = rng.choice(remaining, size=(val_n - len(val_idx)), replace=False)
    val_idx.extend(extra.tolist())
elif len(val_idx) > val_n:
    val_idx = val_idx[:val_n]

val_df = df.loc[val_idx].copy().reset_index(drop=True)
y_val = val_df["diagnosis"].values.astype(int)

val_preds = predict_regression(md_ef, val_df["path"].tolist(), size=sz, batch_size=bs)

init_coef = [0.57, 1.57, 2.77, 3.57]
best_coef = init_coef
best_kappa = float("nan")

if float(np.std(val_preds)) < 1e-6:
    print(
        "WARNING: Validation predictions are nearly constant; skipping threshold fitting and using init thresholds."
    )
else:
    best_coef, best_kappa = fit_thresholds_bruteforce(
        val_preds, y_val, init=init_coef, max_iter=8
    )

    opt = OptimizedRounder()
    mapped = opt.predict(val_preds.reshape(-1), best_coef).astype(int)
    unique_bins = int(len(np.unique(mapped)))
    if unique_bins <= 1:
        print(
            "Fitted thresholds produced degenerate mapping (unique bins:",
            unique_bins,
            "). Falling back to init thresholds.",
        )
        best_coef = init_coef
        best_kappa = metrics.cohen_kappa_score(
            y_val,
            opt.predict(val_preds.reshape(-1), init_coef).astype(int),
            weights="quadratic",
        )

print("Init coef:", init_coef)
print("Used coef:", best_coef)
print("Val kappa (if computed):", best_kappa)

out_df = run_subm_torch(md_ef, test_df, coefficients=best_coef, size=sz, batch_size=bs)
print(pd.read_csv("submission.csv").head())
print("Saved:", os.path.abspath("submission.csv"))
print(
    "Unique predicted labels:",
    out_df["diagnosis"].value_counts().sort_index().to_dict(),
)
