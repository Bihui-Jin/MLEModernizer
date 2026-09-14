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

0.882876705421334

# 6. Current score

0.7414

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The failures come from a fastai v1 → v2 API mismatch (e.g., `ImageList`, `DatasetType`, `get_transforms` no longer exist in the installed fastai), which prevents data creation, model loading, and inference from running at all. To keep core logic intact (EfficientNet + fastai-style TTA ensemble and argmax), I switch the pipeline to pure PyTorch for dataset/dataloader + deterministic TTA (4 flips) while preserving the same image RAM-loading and preprocessing functions. I also fix a small bug where `load_ben_color` expected a path but was given a string, and make sure the submission is written with the required columns and `.csv` suffix. The script now run end-to-end and create `submission.csv` in the working directory.'
- What this solution (achieved -0.01156) has done: 'I fix the pipeline so it always produces a valid `submission.csv` even when the expected `.pth` weight files are not present in `/kaggle/input` (your current hard-fail causes the run to stop early and no submission is written). To keep core logic intact when weights exist, I preserve the exact EfficientNet + 2-scale + TTA + argmax inference path; when weights are missing, I fall back to a deterministic, simple image-statistics-based classifier so the notebook runs end-to-end and yields a valid file (score be low but non-crashing). I also fix the downstream `NameError` by making submission creation depend on the actually-produced `ids/preds` variables. Finally, I make the data directory resolution robust for both `/kaggle/input/...` and `/kaggle/data/...` layouts without changing your feature extraction/preprocessing logic.'
- What this solution (achieved -0.02369) has done: 'Your current score is extremely low because the run is using the heuristic fallback (the required `.pth` weights are not present), so the smallest change that can realistically move QWK toward your target is to remove the dependency on external weight files and still keep the same EfficientNet inference core logic. I keep your EfficientNet-B5 + 2-scale (400/320) + 4-flip TTA + logits-sum + argmax prediction semantics, but switch to using torchvision’s built-in pretrained EfficientNet-B5 weights as the backbone so inference is meaningful without Kaggle input `.pth` files. To preserve 5-class output while staying minimal, I keep the EfficientNet trunk pretrained and add a randomly-initialized 5-class head, then use a simple, fast calibration on the training set’s label distribution to map the 1000-way ImageNet predictions into 5 ordinal classes (this keeps “argmax of logits” semantics on a mapped 5-class score). This should materially increase your score from negative toward the target band while still finishing within the time limit and always writing a valid `submission.csv`.'
- What this solution (achieved 0.7829) has done: 'Your score is far below the target, and the main reason is that you’re still not using DR-trained weights; the ImageNet→5-class mapping is too weak for QWK. The smallest change that preserves your core EfficientNet + 2-scale + 4-flip TTA + argmax semantics is to replace the “CDF calibration mapping” with a proper 5-class head and do a short, deterministic fine-tune on the provided training set (no new architecture/loops beyond adding a standard train pass). To keep runtime under 600s and changes minimal, I freeze the backbone and train only the final classifier layer for a couple of epochs using the same preprocessing you already apply, then run your existing TTA inference path and write `submission.csv` unchanged. This should move QWK substantially upward toward your target band compared to the current heuristic mapping.'
- What this solution (achieved 0.7414) has done: 'Your current gap to the target (0.7829 → 0.8829) is large enough that we should improve generalization, but without changing the core “EfficientNet-B5 + 2-scale (400/320) + 4-flip TTA + logits-sum + argmax” inference semantics. The minimal, high-impact issue is that the fallback path fine-tunes the head only at 400px while you also infer at 320px; this mismatch hurts the 320-scale logits you add at test time. I (1) fine-tune the head at both 400 and 320 (same loop, same loss, no new tricks), (2) ensure the model is in `eval()` during prediction (BatchNorm stability), and (3) keep submission formatting identical.'

# 9. Code solution

## === cell 0
import os
import gc
import re
import math
import collections
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
from tqdm import tqdm

from sklearn.metrics import cohen_kappa_score

torch.backends.cudnn.benchmark = True


def seed_everything(seed=42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



## === cell 1
bs = 8




## === cell 2
def _resolve_input_path(preferred: str) -> Path:
    """
    Bugfix: original resolution missed the actual provided layout (/kaggle/data/...).
    Keep behavior minimal but robust across common Kaggle layouts.
    """
    candidates = []
    p = Path(preferred)
    candidates.append(p)

    candidates.append(Path("/kaggle/input") / "aptos2019-blindness-detection")
    candidates.append(Path("../input") / "aptos2019-blindness-detection")

    candidates.append(Path("/kaggle/data") / "aptos2019-blindness-detection")
    candidates.append(Path("/kaggle/data") / "input" / "aptos2019-blindness-detection")

    candidates.append(Path("/kaggle/data"))

    for c in candidates:
        if (c / "train.csv").exists() and (c / "test.csv").exists():
            return c
        nested = c / "aptos2019-blindness-detection"
        if (nested / "train.csv").exists() and (nested / "test.csv").exists():
            return nested

    raise FileNotFoundError(
        "Could not find competition data directory. Tried:\n"
        + "\n".join(map(str, candidates))
    )


COMP_DIR = _resolve_input_path("../input/aptos2019-blindness-detection")
path_train = COMP_DIR / "train_images"
path_test = COMP_DIR / "test_images"

print("COMP_DIR:", COMP_DIR)
print("Train images exists:", path_train.exists())
print("Test images exists:", path_test.exists())



## === cell 3
train = pd.read_csv(COMP_DIR / "train.csv")
test_df = pd.read_csv(COMP_DIR / "test.csv")

assert "id_code" in train.columns and "diagnosis" in train.columns
assert "id_code" in test_df.columns

print("train shape:", train.shape, "test shape:", test_df.shape)
print(train.head())




## === cell 4
def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # too dark, crop removes everything
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            img = np.stack([img1, img2, img3], axis=-1)
        return img




## === cell 5
IMG_SIZE = 400


def load_ben_color(path, sigmaX=0):
    path = str(path)
    image = cv2.imread(path)
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




## === cell 6
def histogram_normalization(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype("uint8")
    img2 = cdf[image]
    return img2


def adap_hist_eql(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final


def test_img(image):
    im_sz = 1024
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (im_sz, im_sz))
    return image




## === cell 7
def read_image2(path, k):
    path = str(path)
    image = None
    if k == 1:
        image = load_ben_color(path, 20)
    if k == 2:
        image = load_ben_color(path, 15)
    if k == 3:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at: {path}")
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        image = adap_hist_eql(image)
    if k == 4:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    return image




## === cell 8
image_dic = {}


def load_image_on_ram(folder_path, names, k):
    folder_path = Path(folder_path)
    for i in tqdm(names, desc=f"Loading k={k}"):
        img_path = folder_path / f"{i}.png"
        image = read_image2(img_path, k)
        image_dic[str(i)] = image


gc.collect()
load_image_on_ram(path_train, train.id_code.unique(), 1)
gc.collect()
load_image_on_ram(path_test, test_df.id_code.unique(), 2)
gc.collect()

print("Images loaded into RAM:", len(image_dic))




## === cell 9
def im_b(path=""):
    img = str(path).split("/")[-1]  # e.g. "xxxx.png"
    img = img.replace(".png", "")
    image = image_dic[img] / 255.0
    return image


def im_n(path=""):
    image = cv2.imread(str(path) + ".png")
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    return image


def im_n2(path=""):
    image = load_ben_color(str(path) + ".png", 15) / 255.0
    return image


im_load = im_b



## === cell 10
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def normalize_imagenet(x_chw: np.ndarray) -> np.ndarray:
    x = x_chw.copy()
    x[0] = (x[0] - IMAGENET_MEAN[0]) / IMAGENET_STD[0]
    x[1] = (x[1] - IMAGENET_MEAN[1]) / IMAGENET_STD[1]
    x[2] = (x[2] - IMAGENET_MEAN[2]) / IMAGENET_STD[2]
    return x


class RetinopathyTestDataset(torch.utils.data.Dataset):
    def __init__(self, df, folder: Path, img_size: int):
        self.df = df.reset_index(drop=True)
        self.folder = Path(folder)
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_code = self.df.loc[idx, "id_code"]
        img = im_load(str(self.folder / f"{id_code}.png"))  # HWC float [0,1]
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = (
                cv2.resize(
                    (img * 255).astype(np.uint8), (self.img_size, self.img_size)
                ).astype(np.float32)
                / 255.0
            )
        x = np.transpose(img.astype(np.float32), (2, 0, 1))  # CHW
        x = normalize_imagenet(x)
        return torch.from_numpy(x), id_code


class RetinopathyTrainDataset(torch.utils.data.Dataset):
    """
    Minimal addition: uses the same in-RAM image loading + same ImageNet normalization,
    but returns labels for a short head fine-tune.
    """

    def __init__(self, df, folder: Path, img_size: int):
        self.df = df.reset_index(drop=True)
        self.folder = Path(folder)
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id_code = self.df.loc[idx, "id_code"]
        y = int(self.df.loc[idx, "diagnosis"])
        img = im_load(str(self.folder / f"{id_code}.png"))  # HWC float [0,1]
        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = (
                cv2.resize(
                    (img * 255).astype(np.uint8), (self.img_size, self.img_size)
                ).astype(np.float32)
                / 255.0
            )
        x = np.transpose(img.astype(np.float32), (2, 0, 1))  # CHW
        x = normalize_imagenet(x)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


def get_test_loader(sz):
    ds = RetinopathyTestDataset(test_df, path_test, sz)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=bs,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return dl


def get_train_loader(sz):
    ds = RetinopathyTrainDataset(train, path_train, sz)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=bs,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return dl




## === cell 11
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

        stride_is_1 = (self._block_args.stride == 1) or (self._block_args.stride == [1])

        if self.id_skip and stride_is_1 and input_filters == output_filters:
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
    def from_pretrained(cls, model_name, num_classes=1000):
        model = EfficientNet.from_name(
            model_name, override_params={"num_classes": num_classes}
        )
        return model




## === cell 12
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

possible_weight_files = [
    "ef5_b15_400_1.pth",
    "ef5_b15_320_1.pth",
    "ef5_400_1.pth",
    "ef5_ah_400_1.pth",
    "ef5_b_400_1.pth",
    "ef5_b_400_2.pth",
]

found = {}
kaggle_input_root = Path("/kaggle/input")
if kaggle_input_root.exists():
    for pth in kaggle_input_root.rglob("*.pth"):
        if pth.name in possible_weight_files and pth.name not in found:
            found[pth.name] = pth

for name, src in found.items():
    dst = models_dir / name
    if not dst.exists():
        dst.write_bytes(src.read_bytes())

weight_400 = models_dir / "ef5_b15_400_1.pth"
weight_320 = models_dir / "ef5_b15_320_1.pth"

print("Found weights:", list(found.keys()))
print(
    "weight_400 exists:", weight_400.exists(), "weight_320 exists:", weight_320.exists()
)




## === cell 13
@torch.no_grad()
def predict_with_tta(model, dl, tta=4):
    model.eval()
    all_logits = []
    all_ids = []
    for xb, ids in tqdm(dl, desc="Predict"):
        xb = xb.to(device, non_blocking=True)

        logits_sum = 0.0
        views = [
            xb,
            torch.flip(xb, dims=[3]),  # hflip
            torch.flip(xb, dims=[2]),  # vflip
            torch.flip(xb, dims=[2, 3]),  # hvflip
        ]
        for v in views[:tta]:
            logits_sum = logits_sum + model(v)

        logits = logits_sum / float(min(tta, len(views)))
        all_logits.append(logits.detach().cpu())
        all_ids.extend(list(ids))

    all_logits = torch.cat(all_logits, dim=0)
    return all_logits, all_ids


def load_weights_into_model(model, weight_path: Path):
    if not weight_path.exists():
        return False
    sd = torch.load(str(weight_path), map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    if isinstance(sd, dict):
        new_sd = {}
        for k, v in sd.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_sd[nk] = v
        sd = new_sd
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(
        f"Loaded {weight_path.name}. Missing keys: {len(missing)} Unexpected keys: {len(unexpected)}"
    )
    return True




## === cell 14
NUM_CLASSES = int(train["diagnosis"].nunique())
assert NUM_CLASSES == 5, f"Expected 5 classes, got {NUM_CLASSES}"


def build_model(num_out: int):
    m = EfficientNet.from_pretrained("efficientnet-b5", num_classes=num_out)
    return m.to(device)


def build_torchvision_efficientnet_b5_5class():
    """
    Change directly aimed at score improvement:
    Instead of mapping ImageNet logits to 5 classes (very low QWK),
    we use a proper 5-class head and fine-tune it briefly on the provided train.csv.
    Core inference semantics remain: 2-scale + 4-flip TTA + logits-sum + argmax.
    """
    import torchvision
    from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights

    weights = EfficientNet_B5_Weights.DEFAULT
    m = efficientnet_b5(weights=weights)
    in_features = m.classifier[1].in_features
    m.classifier[1] = nn.Linear(in_features, 5)
    return m.to(device)


def freeze_backbone_train_head_only(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier[1].parameters():
        p.requires_grad = True


def train_head_quick(
    model: nn.Module, img_size: int, epochs: int = 2, lr: float = 3e-3
):
    """
    Minimal deterministic training loop (no new tricks): CrossEntropy on train set.
    """
    model.train()
    freeze_backbone_train_head_only(model)

    dl = get_train_loader(img_size)
    opt = torch.optim.Adam(model.classifier[1].parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for ep in range(epochs):
        total_loss = 0.0
        n = 0
        for xb, yb in tqdm(dl, desc=f"Train head sz={img_size} ep={ep+1}/{epochs}"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu().item()) * xb.size(0)
            n += xb.size(0)

        print(f"Epoch {ep+1}: loss={total_loss/max(1,n):.5f}")

    model.eval()
    return model


have_weights = weight_400.exists() and weight_320.exists()
ids_out = None
test_preds = None

if have_weights:
    dl400 = get_test_loader(400)
    model400 = build_model(NUM_CLASSES)
    has_400 = load_weights_into_model(model400, weight_400)
    if not has_400:
        have_weights = False
    else:
        logits400, ids400 = predict_with_tta(model400, dl400, tta=4)
    del model400
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if have_weights:
    dl320 = get_test_loader(320)
    model320 = build_model(NUM_CLASSES)
    has_320 = load_weights_into_model(model320, weight_320)
    if not has_320:
        have_weights = False
    else:
        logits320, ids320 = predict_with_tta(model320, dl320, tta=4)
    del model320
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

if have_weights:
    assert ids400 == ids320, "ID order mismatch between loaders"
    logits = logits400 + logits320
    ids_out = ids400
    test_preds = torch.argmax(logits, dim=1).numpy().astype(int)
    print("Used model weights inference. Preds shape:", test_preds.shape)
else:
    print(
        "WARNING: Required DR-trained .pth weight files not found.\n"
        "To improve QWK toward target while preserving inference semantics, "
        "we fine-tune only the torchvision EfficientNet-B5 classifier head on the provided training set "
        "(same preprocessing, same 2-scale + TTA inference, argmax)."
    )

    try:
        tv_model = build_torchvision_efficientnet_b5_5class()
    except Exception as e:
        print("torchvision not available, falling back to heuristic:", repr(e))
        tv_model = None

    if tv_model is None:

        def fallback_predict_simple(test_ids):
            preds = []
            for id_code in tqdm(test_ids, desc="Fallback predict"):
                img = image_dic[str(id_code)].astype(np.float32) / 255.0  # RGB [0,1]
                gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
                mean = float(gray.mean())
                std = float(gray.std())
                score = (1.0 - mean) * 3.0 + std * 2.0
                if score < 0.7:
                    p = 0
                elif score < 1.1:
                    p = 1
                elif score < 1.5:
                    p = 2
                elif score < 1.9:
                    p = 3
                else:
                    p = 4
                preds.append(int(p))
            return np.array(preds, dtype=int)

        ids_out = test_df["id_code"].tolist()
        test_preds = fallback_predict_simple(ids_out)
        print("Used heuristic fallback inference. Preds shape:", test_preds.shape)
    else:
        tv_model = train_head_quick(tv_model, img_size=400, epochs=2, lr=3e-3)
        tv_model = train_head_quick(tv_model, img_size=320, epochs=1, lr=2e-3)

        dl400 = get_test_loader(400)
        dl320 = get_test_loader(320)

        logits400, ids400 = predict_with_tta(tv_model, dl400, tta=4)
        logits320, ids320 = predict_with_tta(tv_model, dl320, tta=4)
        assert ids400 == ids320, "ID order mismatch between loaders"

        logits = logits400 + logits320  # (N,5)
        ids_out = ids400
        test_preds = torch.argmax(logits, dim=1).numpy().astype(int)
        print(
            "Used head-finetuned torchvision inference. Preds shape:", test_preds.shape
        )



## === cell 15
sub = pd.DataFrame({"id_code": ids_out, "diagnosis": test_preds})
sub = sub.set_index("id_code").loc[test_df["id_code"].values].reset_index()

sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Saved submission.csv with shape:", sub.shape)
print("diagnosis value counts:\n", sub["diagnosis"].value_counts().sort_index())
assert sub.shape[0] == test_df.shape[0]
assert list(sub.columns) == ["id_code", "diagnosis"]
assert Path("submission.csv").exists()
