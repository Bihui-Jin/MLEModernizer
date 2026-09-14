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

0.901645703725582

# 6. Current score

0.79447

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The script is restructured to remove the unavailable fastai imports, add the missing collections and os imports, and replace the fastai data pipeline with a standard PyTorch Dataset/DataLoader setup. A pretrained EfficientNet‑b0 model (downloaded via torch.utils.model_zoo) is loaded, fine‑tuned on the training images for a few epochs, and the optimal rounding thresholds are learned on the validation set using the provided OptimizedRounder. Finally, test predictions are converted to class labels and saved as submission.csv in the required format.'
- What this solution (achieved 0.0) has done: 'To speed up the run we keep the model and training logic unchanged but eliminate unnecessary overhead:
- Enable cuDNN benchmarking for faster convolution kernels.
- Use the maximum safe number of workers and keep them alive across epochs (`persistent_workers=True`) so workers aren’t recreated each epoch.
- Move the constant class‑index tensor out of the validation/test loops to avoid rebuilding it each batch.'
- What this solution (achieved 0.70894) has done: 'The changes cache each image after resizing so the DataLoader no longer reads and processes files on every batch, drastically cutting I/O and preprocessing time. The loaders now use a single worker because the data is already in memory, preserving the same augmentations and model logic while keeping results identical.'
- What this solution (achieved 0.45832) has done: 'The changes vectorize the threshold‑rounding used in `OptimizedRounder` (replacing slow Python loops with a single NumPy `digitize` call) and increase the DataLoader workers to parallelize the per‑epoch augmentations, which cuts the heavy per‑epoch optimization time while keeping identical behaviour and model‑training logic.'
- What this solution (achieved 0.77297) has done: 'The fix ensures the threshold optimizer never receives non‑monotonic bin values by sorting the coefficients inside `_apply_coef` and after optimization. A fallback is added after training so that `best_coef` is always defined even if no improvement occurs, preventing later reference errors. These small, targeted changes unblock the training loop and allow the model to produce a valid `submission.csv`, moving the score toward the target.'
- What this solution (achieved 0.79447) has done: 'The updates increase data‑loader parallelism, use mixed‑precision (AMP) training and inference, and add non‑blocking tensor transfers. These changes keep the exact model, loss, optimizer, augmentation, and kappa‑optimization logic while reducing GPU‑compute time enough to stay within the 600 s limit.'

# 9. Code solution

## === cell 0
import os
import warnings

warnings.filterwarnings("ignore")
import collections
import math
import re
import numpy as np
import pandas as pd
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import functional as TF
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import scipy as sp

from torch.cuda import amp



## === cell 1
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


def round_filters(filters, gp):
    mult = gp.width_coefficient
    if not mult:
        return filters
    divisor = gp.depth_divisor
    min_depth = gp.min_depth
    filters *= mult
    min_depth = min_depth or divisor
    new_f = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_f < 0.9 * filters:
        new_f += divisor
    return int(new_f)


def round_repeats(repeats, gp):
    mult = gp.depth_coefficient
    if not mult:
        return repeats
    return int(math.ceil(mult * repeats))


def drop_connect(inputs, p, training):
    if not training:
        return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob + torch.rand(
        [batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device
    )
    binary_tensor = torch.floor(random_tensor)
    return inputs / keep_prob * binary_tensor


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
        pad_h = max((oh - 1) * sh + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * sw + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d(
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)
            )
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
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
    def forward(self, x):
        return x


def efficientnet_params(name):
    mapping = {
        "efficientnet-b0": (1.0, 1.0, 224, 0.2),
        "efficientnet-b1": (1.0, 1.1, 240, 0.2),
        "efficientnet-b2": (1.1, 1.2, 260, 0.3),
        "efficientnet-b3": (1.2, 1.4, 300, 0.3),
        "efficientnet-b4": (1.4, 1.8, 380, 0.4),
        "efficientnet-b5": (1.6, 2.2, 456, 0.4),
        "efficientnet-b6": (1.8, 2.6, 528, 0.5),
        "efficientnet-b7": (2.0, 3.1, 600, 0.5),
    }
    return mapping[name]


class MBConvBlock(nn.Module):
    def __init__(self, block_args, gp):
        super().__init__()
        self.block_args = block_args
        self.id_skip = block_args.id_skip
        self.has_se = (block_args.se_ratio is not None) and (
            0 < block_args.se_ratio <= 1
        )
        Conv2d = get_same_padding_conv2d(image_size=gp.image_size)
        inp = block_args.input_filters
        oup = inp * block_args.expand_ratio
        if block_args.expand_ratio != 1:
            self.expand_conv = Conv2d(inp, oup, 1, bias=False)
            self.bn0 = nn.BatchNorm2d(
                oup, momentum=1 - gp.batch_norm_momentum, eps=gp.batch_norm_epsilon
            )
        k = block_args.kernel_size
        s = block_args.stride
        self.depthwise_conv = Conv2d(oup, oup, k, stride=s, groups=oup, bias=False)
        self.bn1 = nn.BatchNorm2d(
            oup, momentum=1 - gp.batch_norm_momentum, eps=gp.batch_norm_epsilon
        )
        if self.has_se:
            se_ch = max(1, int(block_args.input_filters * block_args.se_ratio))
            self.se_reduce = Conv2d(oup, se_ch, 1)
            self.se_expand = Conv2d(se_ch, oup, 1)
        self.project_conv = Conv2d(oup, block_args.output_filters, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(
            block_args.output_filters,
            momentum=1 - gp.batch_norm_momentum,
            eps=gp.batch_norm_epsilon,
        )

    def forward(self, x, drop_connect_rate=None):
        inp = x
        if self.block_args.expand_ratio != 1:
            x = relu_fn(self.bn0(self.expand_conv(x)))
        x = relu_fn(self.bn1(self.depthwise_conv(x)))
        if self.has_se:
            se = F.adaptive_avg_pool2d(x, 1)
            se = self.se_expand(relu_fn(self.se_reduce(se)))
            x = torch.sigmoid(se) * x
        x = self.bn2(self.project_conv(x))
        if (
            self.id_skip
            and self.block_args.stride == 1
            and self.block_args.input_filters == self.block_args.output_filters
        ):
            if drop_connect_rate:
                x = drop_connect(x, drop_connect_rate, self.training)
            x = x + inp
        return x


class EfficientNet(nn.Module):
    def __init__(self, blocks_args, gp):
        super().__init__()
        Conv2d = get_same_padding_conv2d(image_size=gp.image_size)
        self.gp = gp
        in_ch = 3
        out_ch = round_filters(32, gp)
        self.conv_stem = Conv2d(in_ch, out_ch, 3, stride=2, bias=False)
        self.bn0 = nn.BatchNorm2d(
            out_ch, momentum=1 - gp.batch_norm_momentum, eps=gp.batch_norm_epsilon
        )
        self.blocks = nn.ModuleList()
        for ba in blocks_args:
            ba = ba._replace(
                input_filters=round_filters(ba.input_filters, gp),
                output_filters=round_filters(ba.output_filters, gp),
                num_repeat=round_repeats(ba.num_repeat, gp),
            )
            self.blocks.append(MBConvBlock(ba, gp))
            if ba.num_repeat > 1:
                ba = ba._replace(input_filters=ba.output_filters, stride=1)
            for _ in range(ba.num_repeat - 1):
                self.blocks.append(MBConvBlock(ba, gp))
        in_ch = ba.output_filters
        out_ch = round_filters(1280, gp)
        self.conv_head = Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(
            out_ch, momentum=1 - gp.batch_norm_momentum, eps=gp.batch_norm_epsilon
        )
        self.dropout_rate = gp.dropout_rate
        self.fc = nn.Linear(out_ch, gp.num_classes)

    def extract_features(self, x):
        x = relu_fn(self.bn0(self.conv_stem(x)))
        for i, b in enumerate(self.blocks):
            drop_rate = self.gp.drop_connect_rate
            if drop_rate:
                drop_rate *= float(i) / len(self.blocks)
            x = b(x, drop_rate)
        x = relu_fn(self.bn1(self.conv_head(x)))
        return x

    def forward(self, x):
        x = self.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self.dropout_rate:
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc(x)
        return x

    @classmethod
    def from_name(cls, name, override_params=None):
        w, d, sz, p = efficientnet_params(name)
        gp = GlobalParams(
            batch_norm_momentum=0.99,
            batch_norm_epsilon=1e-3,
            dropout_rate=p,
            drop_connect_rate=0.2,
            num_classes=5,
            width_coefficient=w,
            depth_coefficient=d,
            depth_divisor=8,
            min_depth=None,
            image_size=sz,
        )
        if override_params:
            gp = gp._replace(**override_params)
        blocks = [
            "r1_k3_s11_e1_i32_o16_se0.25",
            "r2_k3_s22_e6_i16_o24_se0.25",
            "r2_k5_s22_e6_i24_o40_se0.25",
            "r3_k3_s22_e6_i40_o80_se0.25",
            "r3_k5_s11_e6_i80_o112_se0.25",
            "r4_k5_s22_e6_i112_o192_se0.25",
            "r1_k3_s11_e6_i192_o320_se0.25",
        ]
        blocks_args = BlockDecoder.decode(blocks)
        return cls(blocks_args, gp)

    @classmethod
    def from_pretrained(cls, name, num_classes=5):
        model = cls.from_name(name, override_params={"num_classes": num_classes})
        return model

    @classmethod
    def _check_model_name_is_valid(cls, model_name):
        valid = [f"efficientnet-b{i}" for i in range(8)]
        if model_name not in valid:
            raise ValueError(f"Invalid model name {model_name}")


class BlockDecoder:
    @staticmethod
    def _decode_block_string(s):
        ops = s.split("_")
        opts = {}
        for op in ops:
            k, v = re.split(r"(\d.*)", op)[:2]
            opts[k] = v
        stride = [int(opts["s"][0])]
        return BlockArgs(
            kernel_size=int(opts["k"]),
            num_repeat=int(opts["r"]),
            input_filters=int(opts["i"]),
            output_filters=int(opts["o"]),
            expand_ratio=int(opts["e"]),
            id_skip=("noskip" not in s),
            stride=stride,
            se_ratio=float(opts["se"]) if "se" in opts else None,
        )

    @staticmethod
    def decode(strings):
        return [BlockDecoder._decode_block_string(s) for s in strings]




## === cell 2
url_map = {
    "efficientnet-b0": "http://storage.googleapis.com/public-models/efficientnet-b0-08094119.pth",
    "efficientnet-b1": "http://storage.googleapis.com/public-models/efficientnet-b1-dbc7070a.pth",
    "efficientnet-b2": "http://storage.googleapis.com/public-models/efficientnet-b2-27687264.pth",
    "efficientnet-b3": "http://storage.googleapis.com/public-models/efficientnet-b3-c8376fa2.pth",
    "efficientnet-b4": "http://storage.googleapis.com/public-models/efficientnet-b4-e116e8b3.pth",
    "efficientnet-b5": "http://storage.googleapis.com/public-models/efficientnet-b5-586e6cc6.pth",
}


def load_pretrained_weights(model, name):
    state = model_zoo.load_url(url_map[name])
    model.load_state_dict(state, strict=False)  # ignore mismatched classifier
    print(f"Pretrained weights loaded for {name}")




## === cell 3
def get_model():
    model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=5)
    load_pretrained_weights(model, "efficientnet-b0")
    return model




## === cell 4
class RetinaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, target_size=(250, 250)):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.images = []
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            img_path = os.path.join(img_dir, f"{row['id_code']}.png")
            img = read_image(img_path).float() / 255.0  # C,H,W float tensor
            img = TF.resize(img, target_size)
            self.images.append(img)
        self.labels = torch.tensor(self.df["diagnosis"].values, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


def load_data():
    base = os.path.join("..", "input", "aptos2019-blindness-detection")
    train_csv = os.path.join(base, "train.csv")
    test_csv = os.path.join(base, "test.csv")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    img_dir = os.path.join(base, "train_images")
    return train_df, test_df, img_dir


train_df, test_df, img_dir = load_data()

train_tfms = transforms.Compose(
    [
        transforms.RandomResizedCrop(size=250, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
val_tfms = transforms.Compose(
    [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = None

    @staticmethod
    def _apply_coef(X, coef):
        bins = np.sort(coef)
        return np.digitize(X, bins, right=False)

    def _kappa_loss(self, coef, X, y):
        Xp = self._apply_coef(X, coef)
        return -cohen_kappa_score(y, Xp, weights="quadratic")

    def fit(self, X, y):
        loss = partial(self._kappa_loss, X=X, y=y)
        init_coef = [0.5, 1.5, 2.5, 3.5]
        self.coef_ = sp.optimize.minimize(loss, init_coef, method="nelder-mead")
        self.coef_["x"] = np.sort(self.coef_["x"])
        print("Optimized kappa:", -self._kappa_loss(self.coef_["x"], X, y))

    def predict(self, X, coef):
        return self._apply_coef(X, coef).astype(int)

    def coefficients(self):
        return np.sort(self.coef_["x"])


skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(train_df, train_df["diagnosis"]))
train_subset = train_df.iloc[train_idx].reset_index(drop=True)
val_subset = train_df.iloc[val_idx].reset_index(drop=True)

train_dataset = RetinaDataset(train_subset, img_dir, transform=train_tfms)
val_dataset = RetinaDataset(val_subset, img_dir, transform=val_tfms)

num_workers = min(8, os.cpu_count() or 2)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

model = get_model().to(device)

criterion = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  # slightly lower LR
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2, verbose=True
)

class_range = torch.arange(5, device=device).float()

scaler = amp.GradScaler() if device.type == "cuda" else None


def train_one_epoch():
    model.train()
    total_loss = 0.0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        with amp.autocast(enabled=scaler is not None):
            out = model(xb)
            loss = criterion(out, yb)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
    return total_loss / len(train_loader.dataset)


def evaluate():
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)
            with amp.autocast(enabled=scaler is not None):
                out = model(xb)
                probs = torch.softmax(out, dim=1)
                exp_val = (probs * class_range).sum(dim=1)
            preds.extend(exp_val.cpu().numpy())
            trues.extend(yb.numpy())
    return np.array(preds), np.array(trues)


best_kappa = -1
best_coef = None
best_val_pred = None
best_val_true = None

for epoch in range(30):
    loss = train_one_epoch()
    val_pred, val_true = evaluate()
    opt_tmp = OptimizedRounder()
    opt_tmp.fit(val_pred, val_true)
    val_pred_rounded = opt_tmp.predict(val_pred, opt_tmp.coefficients())
    kappa = cohen_kappa_score(val_pred_rounded, val_true, weights="quadratic")
    print(f"Epoch {epoch+1}: loss={loss:.4f}, val_kappa={kappa:.4f}")
    scheduler.step(kappa)

    if kappa > best_kappa:
        best_kappa = kappa
        best_val_pred = val_pred.copy()
        best_val_true = val_true.copy()
        best_coef = opt_tmp.coefficients()

if best_coef is None:
    best_coef = opt_tmp.coefficients()
    best_val_pred = val_pred
    best_val_true = val_true



## === cell 6
if best_coef is None:
    opt = OptimizedRounder()
    opt.fit(best_val_pred, best_val_true)
    best_coef = opt.coefficients()
else:
    opt = OptimizedRounder()
    opt.coef_ = {"x": best_coef}  # store coefficients for consistency

test_img_dir = os.path.join(
    "..", "input", "aptos2019-blindness-detection", "test_images"
)


class TestDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, target_size=(250, 250)):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.images = []
        for idx in range(len(self.df)):
            id_code = self.df.iloc[idx]["id_code"]
            path = os.path.join(img_dir, f"{id_code}.png")
            img = read_image(path).float() / 255.0
            img = TF.resize(img, target_size)
            self.images.append(img)
        self.ids = self.df["id_code"].tolist()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        return img, self.ids[idx]


test_dataset = TestDataset(test_df, test_img_dir, transform=val_tfms)
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    persistent_workers=True,
)

model.eval()
all_preds = []
all_ids = []
with torch.no_grad():
    for imgs, ids in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        with amp.autocast(enabled=scaler is not None):
            out = model(imgs)
            probs = torch.softmax(out, dim=1)
            exp_val = (probs * class_range).sum(dim=1)
        all_preds.extend(exp_val.cpu().numpy())
        all_ids.extend(ids)

final_preds = opt.predict(np.array(all_preds), best_coef)
submission = pd.DataFrame({"id_code": all_ids, "diagnosis": final_preds})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("submission.csv written, rows:", len(submission))
