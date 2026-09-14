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

0.9080228716816332

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the notebook magics and the `fastai` dependency (it’s not installed in your environment), while keeping the core idea intact: load an EfficientNet-B5 model, load provided weights, run inference on the test images, and write `submission.csv` with `id_code,diagnosis`. I also fix missing imports (`os`, `collections`, etc.), ensure the dataset paths match your provided `/kaggle/data/...` layout, and make sure the pretrained-weight download is not required (offline-safe). Finally, since no valid score was yielded, the main goal is to produce a valid end-to-end run and a correctly formatted submission CSV.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from using random-initialized EfficientNet weights (no `abcdef.pth` exists), so predictions are essentially noise; the smallest legitimate fix is to load a real checkpoint that is already available offline in the competition dataset (typical kernels ship a `weights/` folder). I add an offline-safe search for common APTOS pretrained weight filenames/locations under `/kaggle/input/**` and load the first matching `.pth`/`.pt` found, while keeping the same model and inference pipeline. To avoid accidentally loading an incompatible checkpoint and silently staying near-random, I make the load stricter by requiring the final FC layer weights to match (or clearly fall back), and I also switch inference to `torch.no_grad()` to ensure consistent behavior. This should move the score up substantially toward your target without changing the core model or post-processing logic.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is consistent with effectively random predictions because the script almost certainly fails to load a compatible pretrained checkpoint (it builds `efficientnet-b5` with `num_classes=1`, which reject almost all common APTOS checkpoints trained for 5 classes). To move the score up toward your target with minimal change, I keep the same EfficientNet-B5 + inference pipeline, but switch the head to `num_classes=5` and use `argmax` to output class labels 0–4 directly (no threshold rounder needed for classification). I also tighten checkpoint loading to prefer checkpoints whose final layer matches 5 classes, which makes it much more likely we actually use real weights found under `/kaggle/input` offline. The output CSV format/path stays identical (`submission.csv` with `id_code,diagnosis`), and runtime remains within Kaggle limits.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is almost certainly because no compatible pretrained checkpoint is actually being loaded, so the EfficientNet-B5 runs with random weights and produces near-random labels. I keep the exact same model/inference approach, but tighten and broaden the offline checkpoint discovery to prioritize APTOS/EfficientNet-B5 5-class checkpoints under the provided dataset folder, and then enforce a *hard requirement* that the checkpoint contains a matching `_fc.weight` with 5 outputs (otherwise we stop instead of silently submitting noise). I also ensure the correct image size is taken from the model definition (456 for B5) to avoid accidental mismatch if you later swap variants, without changing preprocessing semantics. These minimal changes should move the score sharply upward toward your target while preserving the core pipeline.'
- What this solution (achieved 0.0) has done: 'I fix the failure in checkpoint discovery/loading so the pipeline can run end-to-end and write a valid `submission.csv`. The root issue is that this environment likely contains no compatible `.pth/.pt` weights (and Kaggle is offline), so hard-failing guarantees a 0.0 score via no submission; I add an offline-safe fallback that still produces a correct-format submission when no checkpoint exists. To move the score up toward your target with minimal semantic change, I also use the provided `train.csv` to compute class priors and, in the no-weights fallback case only, emit the constant majority-class prediction (a common baseline that beats random on this dataset) while keeping the EfficientNet inference path unchanged when weights are found. Finally, I make the data root auto-detect between `/kaggle/input/...` and `/kaggle/data/...` so the script runs reliably in standard Kaggle layouts.'
- What this solution (achieved 0.0) has done: 'Your 0.0 score is coming from near-random predictions because the script almost never finds any compatible offline EfficientNet-B5 5-class checkpoint, so it falls back to a constant majority-class submission. To move the score toward your target with minimal change to the core inference pipeline, I add a strict-but-practical fallback: load ImageNet-pretrained weights from `torchvision` into the same EfficientNet-B5 backbone (excluding the final FC layer) when no competition checkpoint is found, then keep the same `argmax` 5-class prediction logic. This keeps the model architecture and inference semantics the same (EfficientNet-B5 classifier) while ensuring the network is not randomly initialized. The submission writing/alignment remain unchanged and the script still run offline end-to-end within the time limit.'
- What this solution (achieved 0.19211) has done: 'Your 0.0 score is most consistent with a *valid but essentially untrained/random* model: the current “torchvision ImageNet fallback” mapping is not compatible with this custom EfficientNet implementation, so it likely loads almost nothing and behaves close to random. To move the score upward toward your target with minimal change to the core approach (EfficientNet-B5 inference only), I remove the incompatible torchvision-weight mapping and instead use an offline-safe, deterministic fallback that predicts by nearest class mean color statistics computed from `train.csv` + `train_images` (a weak-but-real signal that should beat random and majority-class). The pretrained-checkpoint path stays the primary route and is unchanged; the fallback only activates when no compatible `.pth/.pt` is found. Submission formatting/alignment stays identical and we still always write a valid `submission.csv`.'
- What this solution (achieved 0.19211) has done: 'Your current score (0.19211) is far below the target (0.9080), so we should improve performance; the biggest likely issue is that the “pretrained checkpoint” path is almost never used, and the fallback (RGB mean prototypes) is too weak for this metric. I keep your EfficientNet-B5 inference core intact, but add a minimal, offline-safe way to load ImageNet weights **into this exact EfficientNet implementation** (not torchvision’s incompatible one) by converting the official `efficientnet-b5` weights keys to your model’s key naming and loading them with `strict=False`. This should move predictions from near-random/weak-signal to a much stronger baseline, and still uses the same model architecture and argmax 5-class outputs. The existing submission formatting/alignment stays unchanged, and the RGB-prototype fallback remains only if both APTOS and ImageNet weights can’t be loaded.'
- What this solution (achieved 0.0) has done: 'I fix the runtime error in the ImageNet-weight fallback by preventing `load_state_dict` from crashing on size mismatches: we filter the mapped weights to only those keys whose tensor shapes exactly match the current custom EfficientNet-B5 model. This keeps the same core inference pipeline (EfficientNet-B5, argmax over 5 classes) while making the fallback path actually run end-to-end and produce a valid `submission.csv`. Because your current score (0.19211) is far below the target (0.9080), this change is expected to improve performance versus the current broken fallback (which never successfully loads usable weights). All I/O paths and submission formatting are preserved.'
- What this solution (achieved 0.0) has done: 'Your score is 0.0 because the current pipeline is effectively not using meaningful trained weights (the torchvision-to-custom EfficientNet mapping is very likely loading few/incorrect keys), so predictions are close to noise; to move toward 0.908 we should ensure we load a real offline APTOS checkpoint if it exists, and otherwise fall back to a legitimate model that is actually available offline. I keep your EfficientNet-B5 inference core intact, but (1) broaden and prioritize checkpoint discovery to include all `.pth/.pt/.bin` files (not only name-hinted ones), (2) fix the compatibility scorer to prefer the most complete match to this exact implementation, and (3) if no APTOS checkpoint is found, load ImageNet EfficientNet-B5 weights from `torchvision` by transferring weights layer-by-layer through a small “proxy” module that mirrors your custom EfficientNet blocks (same shapes), avoiding the brittle key-mapping. This keeps the same architecture and still outputs 5-class `argmax` predictions, but should substantially increase performance versus the current near-random fallback while staying offline-safe and within time.'

# 9. Code solution

## === cell 0
import os
import re
import math
import json
import collections
from functools import partial

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image

torch.set_grad_enabled(False)
torch.backends.cudnn.benchmark = True

_CANDIDATE_ROOTS = [
    "/kaggle/input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/input",
    "/kaggle/data",
]
DATA_ROOT = None
for r in _CANDIDATE_ROOTS:
    if os.path.exists(os.path.join(r, "train.csv")) and os.path.exists(
        os.path.join(r, "test.csv")
    ):
        DATA_ROOT = r
        break
if DATA_ROOT is None:
    DATA_ROOT = "/kaggle/data/aptos2019-blindness-detection"

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

print("Using DATA_ROOT:", DATA_ROOT)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Test CSV exists:", os.path.exists(TEST_CSV))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB))
print("Test image dir exists:", os.path.isdir(TEST_IMG_DIR))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



## === cell 1
"""
EfficientNet implementation (as provided), kept intact.
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


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


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
md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=5).to(device)
md_ef.eval()


def _find_weight_files():
    found = []
    roots = [
        DATA_ROOT,
        "/kaggle/input/aptos2019-blindness-detection",
        "/kaggle/input",
        "/kaggle/data",
        "/kaggle/working",
    ]
    exts = (".pth", ".pt", ".bin")
    for r in roots:
        if not os.path.isdir(r):
            continue
        for dirpath, _, filenames in os.walk(r):
            for fn in filenames:
                if fn.lower().endswith(exts):
                    found.append(os.path.join(dirpath, fn))
    seen = set()
    uniq = []
    for p in found:
        if p not in seen:
            uniq.append(p)
            seen.add(p)
    return uniq


def _cleanup_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k.replace("module.", "")
            new_state[nk] = v
        return new_state
    return state


def _checkpoint_fc_out_features(state):
    if not isinstance(state, dict):
        return None
    w = state.get("_fc.weight", None)
    if isinstance(w, torch.Tensor) and w.ndim == 2:
        return int(w.shape[0])
    return None


def _score_checkpoint_compatibility(state, model):
    if not isinstance(state, dict):
        return -10
    model_sd = model.state_dict()
    score = 0.0

    exact = 0
    for k, v in state.items():
        if (
            k in model_sd
            and isinstance(v, torch.Tensor)
            and tuple(v.shape) == tuple(model_sd[k].shape)
        ):
            exact += 1
    score += exact * 2.0  # broad match

    w = state.get("_fc.weight", None)
    b = state.get("_fc.bias", None)
    if isinstance(w, torch.Tensor) and tuple(w.shape) == tuple(model._fc.weight.shape):
        score += 5000.0
    if isinstance(b, torch.Tensor) and tuple(b.shape) == tuple(model._fc.bias.shape):
        score += 1500.0

    score += min(len(state), 5000) * 0.01
    return score


def _filter_state_dict_by_shape(state_dict, model):
    if not isinstance(state_dict, dict):
        return {}
    model_sd = model.state_dict()
    filtered = {}
    for k, v in state_dict.items():
        if k not in model_sd:
            continue
        mv = model_sd[k]
        if (
            isinstance(v, torch.Tensor)
            and isinstance(mv, torch.Tensor)
            and tuple(v.shape) == tuple(mv.shape)
        ):
            filtered[k] = v
    return filtered


def _try_load_imagenet_weights_via_proxy(model):
    try:
        from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
    except Exception as e:
        print("torchvision not available for ImageNet fallback:", repr(e))
        return False

    try:
        tv = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        tv.eval()
    except Exception as e:
        print("Could not construct torchvision efficientnet_b5 weights model:", repr(e))
        return False

    class ProxyBlock(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.params = nn.ParameterDict()
            for k, v in block.state_dict().items():
                if isinstance(v, torch.Tensor):
                    self.params[k.replace(".", "__")] = nn.Parameter(
                        torch.zeros_like(v), requires_grad=False
                    )

        def load_from(self, sd_src):
            for k, v in sd_src.items():
                kk = k.replace(".", "__")
                if kk in self.params and tuple(self.params[kk].shape) == tuple(v.shape):
                    self.params[kk].data.copy_(v)

        def to_state_dict(self):
            out = {}
            for kk, p in self.params.items():
                out[kk.replace("__", ".")] = p.data
            return out

    class ProxyEff(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.stem_conv = ProxyBlock(nn.Module())
            self.head_conv = ProxyBlock(nn.Module())
            self.bn0 = ProxyBlock(nn.Module())
            self.bn1 = ProxyBlock(nn.Module())
            self.blocks = nn.ModuleList([ProxyBlock(b) for b in model._blocks])

        def state_dict_for_target(self):
            sd = {}
            for k, v in self.stem_conv.to_state_dict().items():
                sd["_conv_stem." + k] = v
            for k, v in self.bn0.to_state_dict().items():
                sd["_bn0." + k] = v
            for i, pb in enumerate(self.blocks):
                for k, v in pb.to_state_dict().items():
                    sd[f"_blocks.{i}." + k] = v
            for k, v in self.head_conv.to_state_dict().items():
                sd["_conv_head." + k] = v
            for k, v in self.bn1.to_state_dict().items():
                sd["_bn1." + k] = v
            return sd

    proxy = ProxyEff(model)

    tv_sd = tv.state_dict()

    stem_conv_sd = {}
    bn0_sd = {}
    for k, v in tv_sd.items():
        if k.startswith("features.0.0."):
            stem_conv_sd[k.replace("features.0.0.", "")] = v
        elif k.startswith("features.0.1."):
            bn0_sd[k.replace("features.0.1.", "")] = v
    proxy.stem_conv.load_from(stem_conv_sd)
    proxy.bn0.load_from(bn0_sd)

    head_conv_sd = {}
    bn1_sd = {}
    for k, v in tv_sd.items():
        if k.startswith("features.8.0."):
            head_conv_sd[k.replace("features.8.0.", "")] = v
        elif k.startswith("features.8.1."):
            bn1_sd[k.replace("features.8.1.", "")] = v
    proxy.head_conv.load_from(head_conv_sd)
    proxy.bn1.load_from(bn1_sd)

    block_src_sds = []
    for stage in range(1, 8):
        stage_key = f"features.{stage}."
        idxs = set()
        for k in tv_sd.keys():
            if k.startswith(stage_key):
                m = re.match(rf"features\.{stage}\.(\d+)\.", k)
                if m:
                    idxs.add(int(m.group(1)))
        if not idxs:
            continue
        for bi in range(0, max(idxs) + 1):
            prefix = f"features.{stage}.{bi}."
            sd_b = {}
            for k, v in tv_sd.items():
                if k.startswith(prefix):
                    sd_b[k.replace(prefix, "")] = v
            if sd_b:
                block_src_sds.append(sd_b)

    if len(block_src_sds) == 0:
        print("ImageNet fallback via proxy: could not enumerate torchvision blocks.")
        return False

    if len(block_src_sds) != len(model._blocks):
        print(
            f"ImageNet fallback via proxy: block count mismatch tv={len(block_src_sds)} vs custom={len(model._blocks)}; "
            f"loading first {min(len(block_src_sds), len(model._blocks))} blocks."
        )

    n = min(len(block_src_sds), len(model._blocks))
    for i in range(n):
        proxy.blocks[i].load_from(block_src_sds[i])

    mapped_sd = proxy.state_dict_for_target()
    filtered = _filter_state_dict_by_shape(mapped_sd, model)
    if len(filtered) == 0:
        print(
            "ImageNet fallback via proxy: after shape-filtering, 0 keys match; not loading."
        )
        return False

    try:
        missing, unexpected = model.load_state_dict(filtered, strict=False)
    except RuntimeError as e:
        print("ImageNet fallback via proxy: failed to load weights:", repr(e))
        return False

    print(
        f"ImageNet fallback via proxy: shape-matched loaded keys={len(filtered)}; "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    return True


weight_paths = _find_weight_files()
print("Found candidate weight files:", len(weight_paths))

best_path = None
best_state = None
best_score = -(10**18)

for p in weight_paths:
    try:
        st = torch.load(p, map_location="cpu")
        st = _cleanup_state_dict(st)
        if _checkpoint_fc_out_features(st) != 5:
            continue
        sc = _score_checkpoint_compatibility(st, md_ef)
        base = os.path.basename(p).lower()
        sc2 = (
            sc * 1000
            + (200 if "b5" in base else 0)
            + (50 if "aptos" in base else 0)
            - len(p) * 0.01
        )
        if sc2 > best_score:
            best_score = sc2
            best_path = p
            best_state = st
    except Exception:
        continue

USING_PRETRAINED = False
if best_path is not None:
    print("Loading APTOS weights from:", best_path)
    missing, unexpected = md_ef.load_state_dict(best_state, strict=False)
    print(
        "Loaded APTOS weights. Missing keys:",
        len(missing),
        "Unexpected keys:",
        len(unexpected),
    )
    USING_PRETRAINED = True
else:
    print(
        "WARNING: No compatible APTOS pretrained checkpoint found offline. "
        "Trying ImageNet EfficientNet-B5 weights (proxy transfer into this implementation) before using RGB-statistics fallback."
    )
    USING_PRETRAINED = _try_load_imagenet_weights_via_proxy(md_ef)
    if USING_PRETRAINED:
        print("Using ImageNet-transferred weights for inference.")
    else:
        print(
            "WARNING: ImageNet fallback also unavailable. "
            "Will use an offline-safe image-statistics fallback (not random / not constant) for predictions."
        )



## === cell 3
test_df = pd.read_csv(TEST_CSV)
assert "id_code" in test_df.columns
print("Test rows:", len(test_df))

train_df = None
if os.path.exists(TRAIN_CSV):
    train_df = pd.read_csv(TRAIN_CSV)
    if "diagnosis" in train_df.columns:
        vc = train_df["diagnosis"].value_counts().sort_index()
        majority_class = int(vc.idxmax())
    else:
        majority_class = 0
else:
    majority_class = 0

print("Majority class (from train.csv if available):", majority_class)



## === cell 4
IMG_SIZE = EfficientNet.get_image_size("efficientnet-b5")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image_tensor(path, img_size=IMG_SIZE):
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))  # HWC->CHW
    return torch.from_numpy(arr)


class TestDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_code = self.df.loc[idx, "id_code"]
        path = os.path.join(self.img_dir, f"{id_code}.png")
        x = load_image_tensor(path)
        return id_code, x


test_ds = TestDataset(test_df, TEST_IMG_DIR)
test_loader = DataLoader(
    test_ds,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 5
def _fast_rgb_mean(path, downsample=64):
    img = Image.open(path).convert("RGB")
    img = img.resize((downsample, downsample), resample=Image.BILINEAR)
    arr = np.asarray(img).astype(np.float32) / 255.0  # (H,W,3)
    return arr.reshape(-1, 3).mean(axis=0)  # (3,)


def _compute_class_rgb_means(train_df, train_img_dir, max_per_class=200):
    class_sums = {c: np.zeros(3, dtype=np.float64) for c in range(5)}
    class_counts = {c: 0 for c in range(5)}
    used = {c: 0 for c in range(5)}

    if train_df is None or "diagnosis" not in train_df.columns:
        return None

    df = train_df[["id_code", "diagnosis"]].copy()
    df["diagnosis"] = df["diagnosis"].astype(int)
    df = df.sort_values("id_code").reset_index(drop=True)

    for _, row in df.iterrows():
        c = int(row["diagnosis"])
        if c not in used:
            continue
        if used[c] >= max_per_class:
            continue
        p = os.path.join(train_img_dir, f"{row['id_code']}.png")
        if not os.path.exists(p):
            continue
        try:
            m = _fast_rgb_mean(p, downsample=64)
            class_sums[c] += m
            class_counts[c] += 1
            used[c] += 1
        except Exception:
            continue

    means = {}
    for c in range(5):
        if class_counts[c] > 0:
            means[c] = (class_sums[c] / class_counts[c]).astype(np.float32)
    return means


def _predict_by_nearest_rgb_mean(test_ids, test_img_dir, class_means):
    classes = sorted(class_means.keys())
    proto = np.stack([class_means[c] for c in classes], axis=0)  # (K,3)
    preds = []
    for id_code in test_ids:
        p = os.path.join(test_img_dir, f"{id_code}.png")
        m = _fast_rgb_mean(p, downsample=64)  # (3,)
        d = ((proto - m[None, :]) ** 2).sum(axis=1)  # (K,)
        preds.append(int(classes[int(np.argmin(d))]))
    return np.asarray(preds, dtype=np.int64)




## === cell 6
ids = list(test_df["id_code"].values)

if USING_PRETRAINED:
    preds_cls = []
    out_ids = []
    with torch.no_grad():
        for batch_ids, batch_x in test_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            logits = md_ef(batch_x)  # (bs, 5)
            pred = torch.argmax(logits, dim=1)  # (bs,)
            preds_cls.append(pred.detach().cpu().numpy().astype(np.int64))
            out_ids.extend(list(batch_ids))
    preds_cls = np.concatenate(preds_cls, axis=0)
    ids = out_ids
    print(
        "Preds shape:",
        preds_cls.shape,
        "IDs:",
        len(ids),
        "Unique preds:",
        np.unique(preds_cls),
    )
else:
    class_means = _compute_class_rgb_means(train_df, TRAIN_IMG_DIR, max_per_class=200)
    if class_means is not None and len(class_means) >= 2:
        preds_cls = _predict_by_nearest_rgb_mean(ids, TEST_IMG_DIR, class_means)
        print(
            "Using RGB-mean nearest-prototype fallback predictions. Unique preds:",
            np.unique(preds_cls),
            "Classes with prototypes:",
            sorted(class_means.keys()),
        )
    else:
        preds_cls = np.full(
            shape=(len(ids),), fill_value=majority_class, dtype=np.int64
        )
        print(
            "Using baseline majority-class predictions (no usable class prototypes). Unique preds:",
            np.unique(preds_cls),
        )




## === cell 7
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = None

    def predict(self, X, coef):
        X = np.asarray(X, dtype=np.float32).reshape(-1)
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
        return X_p.astype(np.int64)




## === cell 8
def run_subm(ids, diag, out_path="submission.csv"):
    sub = pd.DataFrame({"id_code": ids, "diagnosis": diag})
    sub = sub.set_index("id_code").loc[test_df["id_code"]].reset_index()
    sub["diagnosis"] = sub["diagnosis"].astype(int)
    sub.to_csv(out_path, index=False)
    print("Wrote:", out_path, "rows:", len(sub), "cols:", list(sub.columns))
    return sub


submission = run_subm(ids, preds_cls, out_path="submission.csv")



## === cell 9
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["id_code", "diagnosis"]
assert len(chk) == len(test_df)
assert chk["diagnosis"].between(0, 4).all()
print(chk.head())
print("Submission is valid.")
