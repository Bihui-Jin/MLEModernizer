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

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The fix adds the missing `partial` import, switches the test ID source to the proper `test.csv`, and makes model loading robust by handling a missing weights file (the model be used with random weights if the file isn’t found). These changes resolve the NameError, ensure `test_loader` is defined, and guarantee a valid `submission.csv` is written.'
- What this solution (achieved 0.0) has done: 'I make the data‑loading path robust so the test CSV and images are correctly read. By trying the typical Kaggle absolute path (`/kaggle/input/...`) and falling back to a relative path, the script actually load the test set instead of producing an empty submission, which should move the score away from 0.0 toward the target.'
- What this solution (achieved 0.0) has done: 'I ensure a valid submission is always created and improve the baseline when the model weights are missing. By loading the training labels and using the most frequent class as a fallback prediction, the script produce sensible predictions instead of random ones, moving the score toward the target. The core EfficientNet model logic remains unchanged; only the inference step adds a simple, safe fallback.'
- What this solution (achieved 0.0) has done: 'I keep the overall EfficientNet‑based pipeline but simplify the inference logic: regardless of whether a pretrained weights file is found, the script still run the model (using whatever weights are loaded – random if none are provided). This removes the constant “most common label” fallback, giving the model a chance to generate varied predictions and therefore move the quadratic weighted kappa away from 0 toward the target. A warning is still printed when the weight file is missing.'
- What this solution (achieved 0.0) has done: 'I add a simple fallback that uses the most common class from the training set whenever the pretrained weights are missing. This keeps the original EfficientNet inference unchanged when weights are available, but avoids the random‑weight predictions that give a score of 0.0. Using the mode label as a constant prediction typically yields a much higher quadratic weighted kappa, moving the current score toward the target.'

# 9. Code solution

## === cell 0
import os
import re
import math
import collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import cohen_kappa_score
from functools import partial  # added for Conv2d padding helper



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
    random_tensor = keep_prob + torch.rand(
        [batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device
    )
    binary_tensor = torch.floor(random_tensor)
    return inputs / keep_prob * binary_tensor


def get_same_padding_conv2d(image_size=None):
    return (
        Conv2dDynamicSamePadding
        if image_size is None
        else partial(Conv2dStaticSamePadding, image_size=image_size)
    )


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
    def forward(self, input):
        return input


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
        ops = block_string.split("_")
        options = {}
        for op in ops:
            splits = re.split(r"(\d.*)", op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value
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
        blocks = []
        for s in string_list:
            blocks.append(BlockDecoder._decode_block_string(s))
        return blocks


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
    w, d, s, p = efficientnet_params("efficientnet-b5")
    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        num_classes=num_classes,
        width_coefficient=width_coefficient if width_coefficient else w,
        depth_coefficient=depth_coefficient if depth_coefficient else d,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size if image_size else s,
    )
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
        oup = inp * self._block_args.expand_ratio
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(inp, oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(oup, momentum=self._bn_mom, eps=self._bn_eps)

        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            oup, oup, groups=oup, kernel_size=k, stride=s, bias=False
        )
        self._bn1 = nn.BatchNorm2d(oup, momentum=self._bn_mom, eps=self._bn_eps)

        if self.has_se:
            num_squeezed = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio)
            )
            self._se_reduce = Conv2d(oup, num_squeezed, kernel_size=1)
            self._se_expand = Conv2d(num_squeezed, oup, kernel_size=1)

        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(oup, final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))
        if self.has_se:
            x_s = F.adaptive_avg_pool2d(x, 1)
            x_s = self._se_expand(relu_fn(self._se_reduce(x_s)))
            x = torch.sigmoid(x_s) * x
        x = self._bn2(self._project_conv(x))
        if (
            self.id_skip
            and self._block_args.stride == 1
            and self._block_args.input_filters == self._block_args.output_filters
        ):
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs
        return x


class EfficientNet(nn.Module):
    def __init__(self, blocks_args, global_params):
        super().__init__()
        self._global_params = global_params
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        bn_mom = 1 - global_params.batch_norm_momentum
        bn_eps = global_params.batch_norm_epsilon

        in_ch = 3
        out_ch = round_filters(32, global_params)
        self._conv_stem = Conv2d(in_ch, out_ch, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(out_ch, momentum=bn_mom, eps=bn_eps)

        self._blocks = nn.ModuleList()
        for block_args in blocks_args:
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, global_params),
                output_filters=round_filters(block_args.output_filters, global_params),
                num_repeat=round_repeats(block_args.num_repeat, global_params),
            )
            self._blocks.append(MBConvBlock(block_args, global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, stride=1
                )
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, global_params))

        in_ch = block_args.output_filters
        out_ch = round_filters(1280, global_params)
        self._conv_head = Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(out_ch, momentum=bn_mom, eps=bn_eps)

        self._dropout = global_params.dropout_rate
        self._fc = nn.Linear(out_ch, global_params.num_classes)

    def extract_features(self, inputs):
        x = relu_fn(self._bn0(self._conv_stem(inputs)))
        for idx, block in enumerate(self._blocks):
            drop_rate = self._global_params.drop_connect_rate
            if drop_rate:
                drop_rate = drop_rate * float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_rate)
        x = relu_fn(self._bn1(self._conv_head(x)))
        return x

    def forward(self, inputs):
        x = self.extract_features(inputs)
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        if self._dropout:
            x = F.dropout(x, p=self._dropout, training=self.training)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        blocks_args, global_params = efficientnet(
            width_coefficient=None,
            depth_coefficient=None,
            dropout_rate=0.2,
            drop_connect_rate=0.2,
            image_size=None,
            num_classes=1000,
        )
        if override_params:
            global_params = global_params._replace(**override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1):
        model = EfficientNet.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        return model




## === cell 2
def load_model(weights_path, device):
    model = EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device)
        if "model" in state:
            model.load_state_dict(state["model"])
        else:
            model.load_state_dict(state)
        print(f"Loaded pretrained weights from {weights_path}")
        weights_available = True
    else:
        print(
            f"Warning: Weights file {weights_path} not found. Proceeding with random-initialized weights."
        )
        weights_available = False
    model.to(device)
    model.eval()
    return model, weights_available




## === cell 3
possible_base_dirs = [
    "/kaggle/input/aptos2019-blindness-detection",
    os.path.abspath("../input/aptos2019-blindness-detection"),
    os.path.abspath("./input/aptos2019-blindness-detection"),
]
base_dir = next((d for d in possible_base_dirs if os.path.isdir(d)), None)
if base_dir is None:
    raise FileNotFoundError(
        "Dataset directory not found. Checked paths: " + ", ".join(possible_base_dirs)
    )

test_df = pd.read_csv(os.path.join(base_dir, "test.csv"))
test_img_dir = os.path.join(base_dir, "test_images")
if not os.path.isdir(test_img_dir):
    raise FileNotFoundError(f"Test images directory not found at {test_img_dir}")

train_df = pd.read_csv(os.path.join(base_dir, "train.csv"))
most_common_label = int(train_df["diagnosis"].mode()[0])

imu_mean = [0.485, 0.456, 0.406]
imu_std = [0.229, 0.224, 0.225]
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=imu_mean, std=imu_std),
    ]
)


class TestDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.ids = df["id_code"].values
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), img_id


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_weights_path = os.path.join("models", "abcdef.pth")
model, weights_available = load_model(model_weights_path, device)

test_dataset = TestDataset(test_df, test_img_dir, transform)
test_loader = DataLoader(
    test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
)

predictions = []
ids_order = []

with torch.no_grad():
    for imgs, ids in test_loader:
        if not weights_available:
            batch_preds = np.full(len(ids), most_common_label, dtype=int).tolist()
        else:
            imgs = imgs.to(device)
            outs = model(imgs)  # shape (batch, 1)
            probs = torch.sigmoid(outs).squeeze(1)  # map to [0,1]
            batch_preds = (
                torch.clamp(torch.round(probs * 4), 0, 4)
                .cpu()
                .numpy()
                .astype(int)
                .tolist()
            )
        predictions.extend(batch_preds)
        ids_order.extend(ids)

submission = pd.DataFrame({"id_code": ids_order, "diagnosis": predictions})
submission = submission.set_index("id_code").reindex(test_df["id_code"]).reset_index()

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
