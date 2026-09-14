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

0.9088854482695004

# 6. Current score

0.74516

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the unsupported fastai dependency (it’s not available in your environment) while keeping the core idea intact: load a pretrained EfficientNet-B5, run inference on the test images, then apply the same fixed threshold rounding to produce 0–4 labels. I also fix missing imports (`os`, `collections`) and remove notebook magics/`!cp` cells that break in a .py-style run. Because your current score was “Not yielded”, the priority is to make the pipeline run end-to-end and write a valid `submission.csv` with the required columns; I keep the existing coefficients-based rounding semantics unchanged. Finally, I make pretrained weight loading offline-safe by optionally loading a local `models/abcdef.pth` if present (otherwise it run with random head weights but still produce a valid submission).'
- What this solution (achieved 0.00821) has done: 'Your current 0.0 score is most consistent with a “valid CSV but essentially random predictions” situation, caused here by never loading a trained APTOS head (you only load ImageNet backbone weights and leave the regression head randomly initialized). To move the score upward toward your target while keeping the same EfficientNet-B5 + continuous regression + fixed-threshold rounding core logic, I add an in-notebook training step on `train.csv` images to fit only the final `_fc` layer using MSE on the 0–4 labels. I keep everything else (architecture, loss family, inference path, coefficients-based rounding) unchanged, and ensure the script still finishes quickly by freezing the backbone and training the head for a small fixed number of epochs. Finally, I keep the submission writing exactly in the required format (`id_code,diagnosis`) as `submission.csv`.'
- What this solution (achieved 0.30901) has done: 'Your current gap to the target is large (0.00821 vs 0.9089, higher-is-better), so we need a real (but still minimal) correctness improvement rather than tiny tweaks. The biggest issue is metric mismatch: you train a regression head with MSE but then use fixed thresholds without calibrating them on your trained head outputs, which can collapse kappa. I keep the same EfficientNet-B5 + single-output regression + MSE head-only training core logic, but add a lightweight out-of-fold prediction step on a small validation split and fit the 4 thresholds by simple grid search to directly maximize quadratic weighted kappa. Then we use those calibrated thresholds for test predictions and write a valid `submission.csv` as before.'
- What this solution (achieved 0.29314) has done: 'The timeout is dominated by repeatedly decoding/transforming large PNGs from disk (train twice + full validation inference + test inference) with small batch sizes and low DataLoader parallelism, plus Python-loop thresholding during grid search. I keep the exact model, loss, epochs, and training loops, but speed things up by (1) using faster, deterministic DataLoader settings (more workers, persistent workers, prefetching, pinned memory) and larger inference batches, (2) avoiding repeated pandas `.loc` in `__getitem__`, (3) vectorizing the thresholding step with `np.digitize` (exactly equivalent binning), and (4) using a DataLoader for test inference instead of per-image Python loops. These changes preserve evaluation semantics (same data, same transforms, same training steps) while cutting CPU overhead and improving GPU utilization.'
- What this solution (achieved 0.10696) has done: 'Your current score (0.293) is far below the target (0.909), so we need a small but meaningful improvement without changing the core EfficientNet-B5 regression + MSE + threshold-rounding approach. The biggest quality issue is that you train the head once on all data, then *re-train on a subset* (tr_df) before calibrating thresholds—this “forgets” some learned signal and makes the calibrated thresholds less representative. I remove the redundant first training, train once on the train-split, calibrate thresholds on the val-split, then do a final head-only fit on the full train set starting from those weights (same model, same loss, same head-only training) before running test inference. I also fix a subtle DataLoader bug (`prefetch_factor=None` is invalid when `num_workers>0`) that can silently break or alter performance depending on environment.'
- What this solution (achieved 0.78309) has done: 'Your current score is far below the target (0.10696 vs 0.90889, higher-is-better), and the most likely cause is that the backbone is effectively untrained for this domain because pretrained weights cannot be downloaded in Kaggle’s offline environment, leaving you training only a randomly initialized head on top of random features. To move the score upward with minimal change to your core logic (EfficientNet-B5, single-output regression, MSE, threshold rounding), I keep the exact model and training approach but (1) load EfficientNet-B5 ImageNet weights from `torchvision` (available offline) by mapping matching layers into your EfficientNet implementation (excluding `_fc`), and (2) allow BatchNorm layers to update during head-only training (while still freezing convolution weights) so the feature statistics adapt slightly to APTOS images. Everything else (splitting, threshold calibration via grid search, final full-data head fit, and submission format) stays the same.'
- What this solution (achieved 0.07238) has done: 'Your current score (0.783) is below the target (0.909), so we should improve quality without changing the core EfficientNet-B5 regression + MSE head-only training + threshold rounding pipeline. The largest low-risk gain is to fix the pretrained-weight transfer: the current “match-by-shape in arbitrary order” mapping is effectively random and can corrupt features; instead we should only copy tensors whose parameter names clearly correspond (stem/blocks/head + BN stats) and otherwise leave your existing EfficientNet weights unchanged. This keeps the same architecture/training/metric semantics but makes the backbone initialization meaningfully closer to ImageNet pretrained, which should move QWK upward toward the target. Everything else (data split, training loops, threshold grid search, submission writing) stays the same.'
- What this solution (achieved 0.1237) has done: 'Your score is far below the target, so we need a real quality fix while keeping the same EfficientNet-B5 regression + MSE head-only training + threshold grid-search rounding pipeline. The biggest issue is still the “pretrained weight transfer”: the current block-weight copy is effectively shape-matched in order, which can corrupt features and lead to near-random predictions. I change the torchvision→custom transfer to copy only clearly corresponding tensors by *matching parameter name suffixes* within each block (e.g., `_expand_conv.weight`, `_bn0.*`, `_depthwise_conv.weight`, SE, `_project_conv.weight`, `_bn2.*`) and only within the same block index, leaving unmatched tensors untouched. This preserves architecture/training/metric semantics but should restore meaningful ImageNet initialization and move QWK upward toward your target.'
- What this solution (achieved 0.1237) has done: 'Your current score (0.1237) is far below the target (0.9089), so we should improve real predictive signal with the smallest change that preserves your core pipeline (EfficientNet-B5 regressor + MSE + head-only training + threshold grid-search rounding). The biggest issue is that your torchvision→custom pretrained transfer still aligns blocks by “order” across different stage/repeat layouts, which can silently mis-map many tensors and destroy the ImageNet initialization. I replace that with a deterministic, name-based mapping that targets only tensors that clearly correspond (stem/head + per-stage/per-repeat blocks when present), and I report how many tensors were copied so you can verify it’s no longer near-random. Everything else (data split, training loops, threshold fitting, and submission writing) stays unchanged to keep semantics stable.'
- What this solution (achieved 0.1237) has done: 'Your current score is far below the target, so we need a quality fix (not tiny tweaks) while keeping your core pipeline (EfficientNet-B5 regressor + MSE head-only training + threshold grid-search rounding). The main issue is the pretrained-weight transfer: pairing torchvision blocks to custom blocks “by order” misaligns many tensors because the block indexing/stage layout differs, effectively destroying ImageNet initialization and making the head-only training nearly random. I replace that with a deterministic, structure-aware mapping that aligns blocks by **(input_filters, output_filters, expand_ratio, kernel, stride, se)** signature and then copies only well-defined per-block tensor suffixes, plus stem/head and BN buffers. Everything else (data, transforms, training epochs, loss, threshold fitting, submission writing) is kept the same.'
- What this solution (achieved -0.03769) has done: 'Your score gap to the target is very large (0.1237 vs 0.9089, higher-is-better), so we need a meaningful but still minimal change that preserves your core pipeline (EfficientNet-B5 regressor + MSE + threshold grid-search rounding). The main culprit is still the pretrained initialization: your signature-based block pairing can match the wrong repeated blocks because many blocks share identical signatures, so weights get mis-assigned and the backbone becomes effectively scrambled. I replace the pairing with an index-aligned mapping that mirrors torchvision’s `features.*.*` block order to your custom `_blocks` order, then copy only well-defined tensor suffixes (plus stem/head + BN buffers) with strict shape checks. Everything else (data, transforms, head-only training, threshold fitting, and submission writing) remains unchanged to keep semantics stable while restoring real ImageNet features and moving QWK upward toward the target.'
- What this solution (achieved -0.03769) has done: 'Your current score is far below the target (gap ≈ -0.946), so we need a real quality improvement while keeping your core pipeline (EfficientNet-B5 regressor + MSE head-only training + threshold grid-search rounding) unchanged. The biggest correctness issue is that your torchvision→custom pretrained transfer is still misaligned: you only copy a small, wrongly-indexed subset of blocks (tv has 39 MBConv blocks; you have 27), which scrambles initialization and can yield near-random features. I replace that transfer with a deterministic “flattened MBConv order” mapping (tv’s `features.1..7.*` blocks in execution order) to your `_blocks` order, copying only the same per-block tensor suffixes with strict shape checks (and still excluding `_fc`). Everything else (data split, training epochs, loss, threshold calibration, submission writing) remains the same, but the backbone starts from meaningful ImageNet weights, which should move QWK strongly upward toward the target.'
- What this solution (achieved -0.03769) has done: 'The current negative QWK strongly suggests the backbone initialization is still effectively scrambled, so head-only training and threshold calibration are operating on near-random features. To move the score upward toward your target while keeping the exact same core pipeline (EfficientNet-B5 regressor, head-only MSE training, grid-searched thresholds, and the same transforms), I replace the fragile “stage/rep order” mapping with a strict key-based mapping that copies only tensors whose names correspond between torchvision and your custom EfficientNet (stem, head, and all MBConv blocks) and only when shapes match. This is a minimal, directly relevant change: it only affects pretrained weight transfer, not the model, training loop, loss, or inference semantics. Everything else (data split, epochs, calibration approach, submission format/path) remains unchanged so runtime stays within the same envelope and a valid `submission.csv` is still produced.'
- What this solution (achieved -0.03769) has done: 'Your negative QWK is most consistent with the backbone being effectively uninitialized/scrambled: the current torchvision→custom weight transfer only pairs the first 27 torchvision MBConv blocks to your 39 custom blocks, leaving most of the custom blocks random (and therefore predictions anti-correlated). To move the score upward toward the target while preserving the exact same model, training (head-only MSE), and thresholding logic, I change only the pretrained transfer function to correctly map **all** torchvision MBConv blocks (flattened in execution order) onto **all** custom `_blocks` (also in execution order) with strict shape checks. This keeps semantics identical (same architecture/loss/loops/transforms) but restores a meaningful ImageNet initialization, which should move QWK strongly upward. Everything else (split, epochs, grid-search calibration, submission writing) stays unchanged.'
- What this solution (achieved 0.74516) has done: 'Your negative QWK strongly suggests the model is effectively learning/inferencing on the wrong data distribution because the ImageNet backbone weights are not being transferred correctly (the current block-by-block “flattened order” copy doesn’t match the true torchvision module layout and leaves most of your custom blocks random). To move the score upward toward your target while keeping the exact same core pipeline (EfficientNet-B5 regressor, head-only MSE training, threshold grid-search rounding, same transforms), I change only the pretrained transfer function to do a strict, shape-checked, key-based mapping using torchvision’s own EfficientNet-B5 state_dict keys (stem/head + all MBConv block conv/BN/SE weights) and your custom module keys. This keeps training/inference semantics unchanged but restores a meaningful initialization so head-only training can actually learn signal. Everything else remains the same, and the script still writes a valid `submission.csv`.'

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
from torchvision import transforms, models
from PIL import Image

from sklearn import metrics

torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CPU_COUNT = os.cpu_count() or 2
DEFAULT_WORKERS = min(8, max(2, CPU_COUNT // 2))


def seed_worker(worker_id: int):
    worker_seed = 42 + worker_id
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


DL_GEN = torch.Generator()
DL_GEN.manual_seed(42)


def _dl_kwargs(num_workers: int):
    kw = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        worker_init_fn=seed_worker if num_workers > 0 else None,
    )
    if num_workers > 0:
        kw["prefetch_factor"] = 2
    return kw




## === cell 1
"""
EfficientNet implementation (as provided), kept intact except for offline-safe weight loading.
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
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[0] + 1 - iw, 0)
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
    def __init__(
        self,
    ):
        super(Identity, self).__init__()

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


url_map = {
    "efficientnet-b0": "http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth",
    "efficientnet-b1": "http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth",
    "efficientnet-b2": "http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth",
    "efficientnet-b3": "http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth",
    "efficientnet-b4": "http://storage.googleapis.com/public-models/efficientnet-b4-e116e8b3.pth",
    "efficientnet-b5": "http://storage.googleapis.com/public-models/efficientnet-b5-586e6cc6.pth",
}


def load_pretrained_weights(model, model_name, load_fc=True, local_path=None):
    state_dict = None
    if local_path is not None and os.path.exists(local_path):
        state_dict = torch.load(local_path, map_location="cpu")
    else:
        try:
            state_dict = model_zoo.load_url(url_map[model_name], progress=False)
        except Exception as e:
            print(
                f"WARNING: Could not download pretrained weights for {model_name}: {e}"
            )
            print(
                "WARNING: Proceeding without pretrained weights (submission will be valid but likely low score)."
            )
            return

    if load_fc:
        model.load_state_dict(state_dict, strict=False)
    else:
        if "_fc.weight" in state_dict:
            state_dict.pop("_fc.weight")
        if "_fc.bias" in state_dict:
            state_dict.pop("_fc.bias")
        model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained weights for {model_name}")


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
    def from_pretrained(cls, model_name, num_classes=1000, weights_path=None):
        model = EfficientNet.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        load_pretrained_weights(
            model, model_name, load_fc=False, local_path=weights_path
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
DATA_ROOT = "/kaggle/data/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)



## === cell 3
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

sz = EfficientNet.get_image_size("efficientnet-b5")  # 456
infer_tfms = transforms.Compose(
    [
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def load_image_tensor(path):
    with Image.open(path) as img:
        img = img.convert("RGB")
        return infer_tfms(img)




## === cell 4
class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, tfms, with_labels=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.tfms = tfms
        self.with_labels = with_labels

        self._ids = self.df["id_code"].values
        if self.with_labels and "diagnosis" in self.df.columns:
            self._y = self.df["diagnosis"].values.astype(np.float32)
        else:
            self._y = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        _id = self._ids[idx]
        img_path = os.path.join(self.img_dir, f"{_id}.png")
        x = load_image_tensor(img_path)
        if self.with_labels:
            y = float(self._y[idx])
            return x, torch.tensor([y], dtype=torch.float32)
        return x




## === cell 5
def load_torchvision_efficientnet_b5_backbone_into_custom(md_custom: EfficientNet):
    """
    Change (score-improving, minimal): replace the fragile block-order copying with a strict,
    shape-checked mapping from torchvision EfficientNet-B5 state_dict keys to this custom
    EfficientNet keys. This preserves the same model/training/inference semantics but restores
    a meaningful ImageNet initialization, which is necessary for head-only training to work.
    """
    try:
        tv = models.efficientnet_b5(
            weights=models.EfficientNet_B5_Weights.IMAGENET1K_V1
        )
        tv_sd = tv.state_dict()
    except Exception as e:
        print("WARNING: torchvision EfficientNet-B5 weights not available:", e)
        print("Proceeding with current weights.")
        return

    custom_sd = md_custom.state_dict()
    new_sd = {}
    copied = 0

    def _copy_if_match(custom_key, tv_key):
        nonlocal copied
        if (custom_key in custom_sd) and (tv_key in tv_sd):
            if custom_sd[custom_key].shape == tv_sd[tv_key].shape:
                new_sd[custom_key] = tv_sd[tv_key].detach().clone()
                copied += 1
                return True
        return False

    _copy_if_match("_conv_stem.weight", "features.0.0.weight")
    for suf in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
        _copy_if_match(f"_bn0.{suf}", f"features.0.1.{suf}")

    tv_blocks = []
    for stage in range(1, 8):  # features.1..features.7 are stages of MBConv blocks
        reps = set()
        pat = re.compile(rf"^features\.{stage}\.(\d+)\.block\.")
        for k in tv_sd.keys():
            m = pat.match(k)
            if m:
                reps.add(int(m.group(1)))
        for rep in sorted(reps):
            tv_blocks.append(f"features.{stage}.{rep}.block")

    custom_blocks = list(range(len(md_custom._blocks)))

    if len(tv_blocks) == 0 or len(custom_blocks) == 0:
        print(
            "WARNING: could not find matching MBConv block lists for weight transfer."
        )
        return

    if len(tv_blocks) != len(custom_blocks):
        print(
            f"WARNING: tv MBConv blocks={len(tv_blocks)} != custom blocks={len(custom_blocks)}. "
            f"Will copy the first min(...) blocks in order (unmatched tail remains as-is)."
        )
    pair_count = min(len(tv_blocks), len(custom_blocks))

    for bidx in range(pair_count):
        tv_base = tv_blocks[bidx]
        cbase = f"_blocks.{bidx}"

        _copy_if_match(f"{cbase}._expand_conv.weight", f"{tv_base}.0.0.weight")
        for suf in [
            "weight",
            "bias",
            "running_mean",
            "running_var",
            "num_batches_tracked",
        ]:
            _copy_if_match(f"{cbase}._bn0.{suf}", f"{tv_base}.0.1.{suf}")

        _copy_if_match(f"{cbase}._depthwise_conv.weight", f"{tv_base}.1.0.weight")
        for suf in [
            "weight",
            "bias",
            "running_mean",
            "running_var",
            "num_batches_tracked",
        ]:
            _copy_if_match(f"{cbase}._bn1.{suf}", f"{tv_base}.1.1.{suf}")

        _copy_if_match(f"{cbase}._se_reduce.weight", f"{tv_base}.2.fc1.weight")
        _copy_if_match(f"{cbase}._se_reduce.bias", f"{tv_base}.2.fc1.bias")
        _copy_if_match(f"{cbase}._se_expand.weight", f"{tv_base}.2.fc2.weight")
        _copy_if_match(f"{cbase}._se_expand.bias", f"{tv_base}.2.fc2.bias")

        _copy_if_match(f"{cbase}._project_conv.weight", f"{tv_base}.3.0.weight")
        for suf in [
            "weight",
            "bias",
            "running_mean",
            "running_var",
            "num_batches_tracked",
        ]:
            _copy_if_match(f"{cbase}._bn2.{suf}", f"{tv_base}.3.1.{suf}")

    _copy_if_match("_conv_head.weight", "features.8.0.weight")
    for suf in ["weight", "bias", "running_mean", "running_var", "num_batches_tracked"]:
        _copy_if_match(f"_bn1.{suf}", f"features.8.1.{suf}")

    for k in list(new_sd.keys()):
        if k.startswith("_fc."):
            new_sd.pop(k, None)

    md_custom.load_state_dict(new_sd, strict=False)

    total_backbone = len([k for k in custom_sd.keys() if not k.startswith("_fc.")])
    print(
        f"Key-based load from torchvision: copied {copied} / {total_backbone} backbone tensors "
        f"(paired blocks={pair_count}, tv blocks={len(tv_blocks)}, custom blocks={len(custom_blocks)})."
    )


os.makedirs("models", exist_ok=True)
local_w = os.path.join("models", "abcdef.pth")
md_ef = EfficientNet.from_pretrained(
    "efficientnet-b5", num_classes=1, weights_path=local_w
)

load_torchvision_efficientnet_b5_backbone_into_custom(md_ef)
md_ef = md_ef.to(DEVICE)




## === cell 6
def train_head_only(
    model, df, epochs=2, batch_size=4, lr=1e-3, num_workers=DEFAULT_WORKERS
):
    model.train()

    for p in model.parameters():
        p.requires_grad = False
    for p in model._fc.parameters():
        p.requires_grad = True

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()

    ds = RetinopathyDataset(df, TRAIN_IMG_DIR, infer_tfms, with_labels=True)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        **_dl_kwargs(num_workers),
        generator=DL_GEN,
    )

    opt = torch.optim.Adam(model._fc.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for ep in range(epochs):
        running = 0.0
        n = 0
        for xb, yb in dl:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            opt.zero_grad()
            out = model(xb)  # [bs,1]
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()

            running += float(loss.detach().cpu().item()) * xb.size(0)
            n += xb.size(0)

        print(f"epoch {ep+1}/{epochs} - mse: {running/max(n,1):.5f}")

    model.eval()
    return model




## === cell 7
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = self.predict(X, coef)
        ll = metrics.cohen_kappa_score(y, X_p, weights="quadratic")
        return -ll

    def predict(self, X, coef):
        coef = np.asarray(coef, dtype=np.float32)
        return np.digitize(X, coef, right=False).astype(np.float32)




## === cell 8
def get_oof_predictions(model, df, batch_size=16, num_workers=DEFAULT_WORKERS):
    ds = RetinopathyDataset(df, TRAIN_IMG_DIR, infer_tfms, with_labels=True)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        **_dl_kwargs(num_workers),
    )

    model.eval()
    preds = np.zeros(len(df), dtype=np.float32)
    ys = np.zeros(len(df), dtype=np.int64)

    k = 0
    with torch.no_grad():
        for xb, yb in dl:
            bsz = xb.size(0)
            xb = xb.to(DEVICE, non_blocking=True)
            out = model(xb).view(-1).detach().float().cpu().numpy()
            preds[k : k + bsz] = out
            ys[k : k + bsz] = yb.view(-1).cpu().numpy().astype(np.int64)
            k += bsz
    return preds, ys


def fit_thresholds_grid(
    preds_cont, y_true, base=(0.57, 1.57, 2.57, 3.57), span=0.8, step=0.1
):
    opt = OptimizedRounder()
    base = np.array(base, dtype=np.float32)
    grid = np.arange(-span, span + 1e-9, step, dtype=np.float32)

    coef = base.copy()
    for _ in range(2):  # small fixed passes; deterministic and fast
        for j in range(4):
            local_best = coef[j]
            local_best_k = -1e9
            for delta in grid:
                trial = coef.copy()
                trial[j] = base[j] + float(delta)
                if not (trial[0] < trial[1] < trial[2] < trial[3]):
                    continue
                pred_cls = opt.predict(preds_cont, trial).astype(int)
                kappa = metrics.cohen_kappa_score(y_true, pred_cls, weights="quadratic")
                if kappa > local_best_k:
                    local_best_k = kappa
                    local_best = trial[j]
            coef[j] = local_best

    pred_cls = opt.predict(preds_cont, coef).astype(int)
    best_kappa = metrics.cohen_kappa_score(y_true, pred_cls, weights="quadratic")
    best_coef = coef.copy()
    return best_coef.tolist(), float(best_kappa)


idx = np.arange(len(train_df))
rng = np.random.RandomState(42)
rng.shuffle(idx)
val_size = int(0.15 * len(train_df))
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

md_ef = train_head_only(
    md_ef, tr_df, epochs=2, batch_size=4, lr=1e-3, num_workers=DEFAULT_WORKERS
)

val_preds_cont, val_y = get_oof_predictions(
    md_ef, val_df, batch_size=16, num_workers=DEFAULT_WORKERS
)
best_coef, best_kappa = fit_thresholds_grid(val_preds_cont, val_y)
print("Calibrated thresholds:", best_coef)
print("Validation QWK (with calibrated thresholds):", best_kappa)

md_ef = train_head_only(
    md_ef, train_df, epochs=2, batch_size=4, lr=1e-3, num_workers=DEFAULT_WORKERS
)




## === cell 9
class TestDataset(Dataset):
    def __init__(self, ids, img_dir, tfms):
        self.ids = np.asarray(ids)
        self.img_dir = img_dir
        self.tfms = tfms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        _id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{_id}.png")
        return load_image_tensor(img_path)


def run_subm(coefficients, batch_size=16, num_workers=DEFAULT_WORKERS):
    opt = OptimizedRounder()

    ids = test_df["id_code"].tolist()
    preds_cont = np.zeros(len(ids), dtype=np.float32)

    ds = TestDataset(ids, TEST_IMG_DIR, infer_tfms)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        **_dl_kwargs(num_workers),
    )

    md_ef.eval()
    k = 0
    with torch.no_grad():
        for xb in dl:
            bsz = xb.size(0)
            xb = xb.to(DEVICE, non_blocking=True)
            out = md_ef(xb).view(-1)
            preds_cont[k : k + bsz] = out.detach().float().cpu().numpy()
            k += bsz

    tst_pred = opt.predict(preds_cont, coefficients).astype(int)

    sub = pd.DataFrame({"id_code": ids, "diagnosis": tst_pred})
    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", sub.shape)
    print(sub.head())


run_subm(coefficients=best_coef, batch_size=16, num_workers=DEFAULT_WORKERS)
