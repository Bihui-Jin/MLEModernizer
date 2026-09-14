# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8932746560488021

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I fix the fastai import/runtime issues by removing the deprecated `DatasetType` import and by avoiding `cnn_learner`/`vision_learner`, since your `EfficientNet` model is already instantiated and shouldn’t be passed as an `arch` callable. Then I create a `Learner` directly with the same model and metric so training/inference semantics stay the same, and I keep your external weight-loading logic intact. Finally, I make `tta` robust by falling back to a plain `get_preds` if TTA is unavailable for this dataloader, ensuring a `submission.csv` is always written with the correct columns.'
- What this solution (achieved 0.0) has done: 'Your current 0.0 score is most consistent with a “model never trained / wrong inference mode / wrong pipeline” issue rather than a small modeling gap, so the smallest improvement is to ensure the loaded checkpoint is actually used for inference and that predictions are made in evaluation mode with the same preprocessing the checkpoint expects. I keep your EfficientNet-B5, RAM caching, and fastai DataBlock logic intact, but I (1) attach the metric you intended (quadratic kappa) for sanity-checking on the validation split, (2) explicitly put the model in `eval()` for inference and disable grads, and (3) make the checkpoint loading stricter by handling common key-prefix patterns so the right weights get applied. This should move the score upward substantially toward the target without changing architecture, loss, or training approach (there is still no training here—only weight-loading + inference).'
- What this solution (achieved 0.0) has done: 'I fix the `DataBlock.new(get_x=...)` call that fails under your fastai version by creating the test dataloader via `dls.test_dl(...)` using the same DataLoaders/transforms, which preserves your preprocessing and avoids changing model logic. This also define `test_dl`, unblocking inference so `learn.get_preds` can run. I keep your RAM image caching/transform, EfficientNet definition, and checkpoint-loading logic intact, only adjusting the test pipeline wiring. Finally, I ensure the script always writes a valid `submission.csv` with the required columns and aligned row order.'
- What this solution (achieved 0.0) has done: 'Your 0.0 QWK is most consistent with a label/target mismatch during training of the checkpoint you’re loading: your `CategoryBlock(vocab=[0..4])` produces fastai-encoded class indices, but if the mapping differs from the checkpoint’s training setup, `argmax` output indices that don’t correspond to the intended diagnosis labels, collapsing QWK. The smallest safe fix (no architecture/loss/training changes) is to force the `CategoryBlock` vocab to be inferred from the actual `diagnosis` values in `train.csv` so the integer↔class mapping matches the dataset labels (0–4) deterministically and consistently. I also add a lightweight validation QWK sanity check right after loading weights to catch issues before submission, without affecting inference semantics. Everything else (EfficientNet-B5, RAM cache, transforms, weight loading, inference, submission writing) is kept intact.'

# 9. Code solution

## === cell 0
import os, gc, math, re, collections
from functools import partial

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import cv2
from tqdm import tqdm

from pathlib import Path

from fastai.vision.all import (
    DataBlock,
    ImageBlock,
    CategoryBlock,
    Transform,
    Resize,
    ResizeMethod,
    aug_transforms,
    imagenet_stats,
    Normalize,
    IndexSplitter,
    accuracy,
    Learner,
)



## === cell 1
bs = 8



## === cell 2
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 3
CANDIDATES = [
    Path("../input/aptos2019-blindness-detection"),
    Path("/kaggle/input/aptos2019-blindness-detection"),
    Path("/kaggle/data/aptos2019-blindness-detection"),
    Path("/kaggle/data"),
    Path("../input"),
]
path = None
for c in CANDIDATES:
    if (c / "train.csv").exists() and (c / "test.csv").exists():
        path = c
        break
if path is None:
    c = Path("/kaggle/data/aptos2019-blindness-detection")
    path = (
        c
        if (c / "train.csv").exists()
        else Path("../input/aptos2019-blindness-detection")
    )


def _resolve_image_dir(base: Path, dirname: str) -> Path:
    base = Path(base)
    direct = base / dirname
    if direct.exists() and direct.is_dir():
        if any(direct.glob("*.png")):
            return direct
    candidates = []
    for p in base.rglob(dirname):
        if p.is_dir() and any(p.glob("*.png")):
            candidates.append(p)
    if candidates:
        candidates.sort(key=lambda x: len(list(x.glob("*.png"))), reverse=True)
        return candidates[0]
    return direct


path_train = _resolve_image_dir(path, "train_images")
path_test = _resolve_image_dir(path, "test_images")

print("Using base path:", path)
print(
    "Resolved train images path:",
    path_train,
    "| exists:",
    path_train.exists(),
    "| pngs:",
    len(list(path_train.glob("*.png"))) if path_train.exists() else 0,
)
print(
    "Resolved test images path :",
    path_test,
    "| exists:",
    path_test.exists(),
    "| pngs:",
    len(list(path_test.glob("*.png"))) if path_test.exists() else 0,
)



## === cell 4
train = pd.read_csv(path / "train.csv")
test = pd.read_csv(path / "test.csv")
sample_sub = pd.read_csv(path / "sample_submission.csv")

assert "id_code" in train.columns and "diagnosis" in train.columns
assert "id_code" in test.columns
assert list(sample_sub.columns) == ["id_code", "diagnosis"]

print("train:", train.shape, "test:", test.shape, "sample:", sample_sub.shape)



## === cell 5
tfms = aug_transforms(
    do_flip=True,
    flip_vert=True,
    max_rotate=10.0,
    max_zoom=1.05,
    max_lighting=0.0,
    max_warp=0.0,
)



## === cell 6
_ = tfms



## === cell 7
tfms = tfms



## === cell 8
_ = tfms




## === cell 9
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # image is too dark so that we crop out everything
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img




## === cell 10
IMG_SIZE = 512


def load_ben_color(path, sigmaX=0):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    if sigmaX != 0:
        image = cv2.addWeighted(
            image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128
        )
    return image




## === cell 11
NUM_SAMP = 7



## === cell 12
gc.collect()



## === cell 13
IMG_SIZE = 400
Test = True


def read_image(path):
    image = load_ben_color(path, 20)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return image




## === cell 14
image_dic = {}




## === cell 15
def load_image_on_ram(p, names):
    p = Path(p)
    for i in tqdm(names):
        img_path = p / f"{i}.png"
        image = read_image(img_path)
        image_dic[f"{i}.png"] = image  # store with filename key as used in open()




## === cell 16
path1 = path_train
path2 = path_test



## === cell 17
if not path1.exists():
    raise FileNotFoundError(f"Resolved train_images directory does not exist: {path1}")
if not path2.exists():
    raise FileNotFoundError(f"Resolved test_images directory does not exist: {path2}")

load_image_on_ram(path1, train.id_code.unique())
load_image_on_ram(path2, test.id_code.unique())

print("Cached images:", len(image_dic))




## === cell 18
class RAMImageTransform(Transform):
    def __init__(self, cache_dict):
        self.cache = cache_dict

    def encodes(self, fn: Path):
        key = fn.name
        arr = self.cache.get(key, None)
        if arr is None:
            arr = read_image(fn)
        arr = arr.astype(np.float32) / 255.0  # HWC RGB in [0,1]
        t = torch.from_numpy(arr).permute(2, 0, 1)
        return t


ram_reader = RAMImageTransform(image_dic)




## === cell 19
def make_train_path(row):
    return path_train / f"{row['id_code']}.png"


def make_test_path(row):
    return path_test / f"{row['id_code']}.png"


rng = np.random.RandomState(42)
idxs = np.arange(len(train))
rng.shuffle(idxs)
n_valid = int(round(0.15 * len(train)))
valid_idx = idxs[:n_valid].tolist()

vocab = sorted(train["diagnosis"].astype(int).unique().tolist())
assert vocab == [0, 1, 2, 3, 4], f"Unexpected label set in train.csv: {vocab}"

dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock(vocab=vocab)),
    get_x=make_train_path,
    get_y=lambda r: int(r["diagnosis"]),
    splitter=IndexSplitter(valid_idx),
    item_tfms=[ram_reader, Resize(256, method=ResizeMethod.Squish, pad_mode="zeros")],
    batch_tfms=[*tfms, Normalize.from_stats(*imagenet_stats)],
)

dls = dblock.dataloaders(train.to_dict("records"), bs=bs, num_workers=2)

test_records = test.to_dict("records")
test_dblock = DataBlock(
    blocks=(ImageBlock,),
    get_x=make_test_path,
    item_tfms=[ram_reader, Resize(256, method=ResizeMethod.Squish, pad_mode="zeros")],
    batch_tfms=[*tfms, Normalize.from_stats(*imagenet_stats)],
)
test_dl = test_dblock.dataloaders(test_records, bs=bs, num_workers=0).train

print(dls)
print("Test rows:", len(test_records), "Test dl len (batches):", len(test_dl))



## === cell 20
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


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

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


def load_pretrained_weights(model, model_name, load_fc=True):
    state_dict = model_zoo.load_url(url_map[model_name])
    if load_fc:
        model.load_state_dict(state_dict)
    else:
        state_dict.pop("_fc.weight")
        state_dict.pop("_fc.bias")
        res = model.load_state_dict(state_dict, strict=False)
        assert str(res.missing_keys) == str(
            ["_fc.weight", "_fc.bias"]
        ), "issue loading pretrained weights"
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

        stride_is_one = (self._block_args.stride == 1) or (
            self._block_args.stride == [1]
        )

        if self.id_skip and stride_is_one and input_filters == output_filters:
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
    def from_pretrained(cls, model_name, num_classes=1000):
        model = EfficientNet.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        return model

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ["efficientnet_b" + str(i) for i in range(num_models)]
        if model_name.replace("-", "_") not in valid_models:
            raise ValueError("model_name should be one of: " + ", ".join(valid_models))




## === cell 21
n_classes = 5
md_ef = EfficientNet.from_pretrained("efficientnet-b5", num_classes=n_classes)



## === cell 22
os.makedirs("models", exist_ok=True)

candidate_weights = [
    "../input/ef5-b-reg-256/ef5_b_256_1.pth",
    "/kaggle/input/ef5-b-reg-256/ef5_b_256_1.pth",
]
src_weight = None
for cw in candidate_weights:
    if os.path.exists(cw):
        src_weight = cw
        break

dst_weight = "models/ef5_b_256_1.pth"
if src_weight is not None and (not os.path.exists(dst_weight)):
    import shutil

    shutil.copy(src_weight, dst_weight)

print("Found src_weight:", src_weight)
print("Weights present:", os.path.exists(dst_weight), "at", dst_weight)



## === cell 23
from sklearn.metrics import cohen_kappa_score


def qwk_metric(preds, targs):
    preds_cls = preds.argmax(dim=1).detach().cpu().numpy()
    targs_np = targs.detach().cpu().numpy()
    return cohen_kappa_score(targs_np, preds_cls, weights="quadratic")




## === cell 24
learn = Learner(
    dls, md_ef, loss_func=nn.CrossEntropyLoss(), metrics=[accuracy, qwk_metric]
)




## === cell 25
def _clean_state_dict_keys(sd):
    if not isinstance(sd, dict):
        return sd
    if "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]

    cleaned = {}
    for k, v in sd.items():
        nk = k
        for pref in ("module.", "model.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        cleaned[nk] = v
    return cleaned


if os.path.exists(dst_weight):
    try:
        state = torch.load(dst_weight, map_location="cpu")
        state = _clean_state_dict_keys(state)
        res = md_ef.load_state_dict(state, strict=False)
        print("Loaded weights from", dst_weight)
        print(
            "Missing keys:",
            len(res.missing_keys),
            "Unexpected keys:",
            len(res.unexpected_keys),
        )
    except Exception as e:
        print("Warning: failed to load weights:", e)
else:
    print(
        "Warning: pretrained weights file not found at",
        src_weight,
        "; proceeding without weight load.",
    )

try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learn.model.to(device)
    learn.model.eval()
    with torch.no_grad():
        vpreds, vtargs = learn.get_preds(dl=dls.valid)
    v_qwk = qwk_metric(vpreds, vtargs)
    v_acc = (vpreds.argmax(1).cpu() == vtargs.cpu()).float().mean().item()
    print(f"Sanity check on valid split -> acc: {v_acc:.4f} | qwk: {v_qwk:.4f}")
except Exception as e:
    print("Warning: sanity-check evaluation skipped due to:", repr(e))



## === cell 26
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learn.model.to(device)
learn.model.eval()

with torch.no_grad():
    preds, _ = learn.get_preds(dl=test_dl)

test_preds = torch.argmax(preds, dim=1).detach().cpu().numpy().astype(int)

print("Preds shape:", preds.shape)
print("Expected test rows:", len(test))
print("Unique predicted labels:", np.unique(test_preds, return_counts=True))
assert len(test_preds) == len(
    test
), f"Pred length {len(test_preds)} != test length {len(test)}"



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/4032754051.py in <cell line: 0>()
     11 print("Expected test rows:", len(test))
     12 print("Unique predicted labels:", np.unique(test_preds, return_counts=True))
---> 13 assert len(test_preds) == len(
     14     test
     15 ), f"Pred length {len(test_preds)} != test length {len(test)}"

AssertionError: Pred length 294 != test length 367

## === cell 27
submission = sample_sub.copy()
submission["id_code"] = test["id_code"].values
submission["diagnosis"] = test_preds

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
assert submission.shape[0] == test.shape[0]
assert list(submission.columns) == ["id_code", "diagnosis"]
assert submission["diagnosis"].between(0, 4).all()

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1506375592.py in <cell line: 0>()
      1 submission = sample_sub.copy()
      2 submission["id_code"] = test["id_code"].values
----> 3 submission["diagnosis"] = test_preds
      4 
      5 submission.to_csv("submission.csv", index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (294) does not match length of index (367)
