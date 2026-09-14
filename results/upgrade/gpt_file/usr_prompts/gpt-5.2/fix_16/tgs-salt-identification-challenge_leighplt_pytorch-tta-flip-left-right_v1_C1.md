# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Segment regions of salt in seismic images.

## Metric
Mean average precision at different intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated as:

$$\text{IoU}(A, B)=\frac{A \cap B}{A \cup B}$$

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: `(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)`. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value 𝑡t, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:

$$\frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object. The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold:

$$\frac{1}{\mid \text { thresholds } \mid} \sum_t \frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

## Submission Format
Use run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed\
and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.

The file should contain a header and have the following format. Each row in your submission represents a single predicted salt segmentation for the given image.

```
id,rle_mask
3e06571ef3,1 1
a51b08d882,1 1
c32590b06f,1 1
etc.
```

## Dataset
The data is a set of images chosen at various locations chosen at random in the subsurface. The images are 101 x 101 pixels and each pixel is classified as either salt or sediment. In addition to the seismic images, the depth of the imaged location is provided for each image.

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124

# 4. Data file paths

```
/
    kaggle/
        data/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
        input/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
        working/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
```

-> data/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/tgs-salt-identification-challenge/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> data/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> input/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import data
import torchvision
from torchvision import models

import cv2
from pathlib import Path
import glob
import warnings
import random
import time



## === cell 1
torch.__version__




## === cell 2
class TTAFunction(nn.Module):
    """
    Simple TTA function
    """

    def tta_flip(self, x):
        self.eval()
        with torch.no_grad():
            x_flip = x.flip(3)
            x_cat = torch.cat([x, x_flip], dim=0)
            y_cat = self.forward(x_cat)
            y = y_cat[: x.shape[0]]
            y_flip = y_cat[x.shape[0] :].flip(3)
            return 0.5 * (y + y_flip)




## === cell 3
def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(
                middle_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet11(TTAFunction):  # use our class with TTA function
    def __init__(self, num_filters=32):
        """
        :param num_classes:
        :param num_filters:
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        try:
            self.encoder = models.vgg11(weights=None).features
        except TypeError:
            self.encoder = models.vgg11(pretrained=False).features

        self.relu = self.encoder[1]

        self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

        self.center = DecoderBlock(
            num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8
        )
        self.dec5 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8
        )
        self.dec4 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4
        )
        self.dec3 = DecoderBlock(
            num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2
        )
        self.dec2 = DecoderBlock(
            num_filters * (4 + 2), num_filters * 2 * 2, num_filters
        )
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)

        self.final = nn.Conv2d(
            num_filters,
            1,
            kernel_size=1,
        )

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return torch.sigmoid(self.final(dec1))


def unet11(**kwargs):
    model = UNet11(**kwargs)
    return model


def get_model():
    np.random.seed(717)
    random.seed(717)
    torch.manual_seed(717)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(717)
    model = unet11()
    model.train()
    return model.to(device)




## === cell 4
model_pth = "../input/goto-pytorch-fix-for-v0-3/tgs-13.pth"



## === cell 5
directory = "../input/tgs-salt-identification-challenge"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass

try:
    cv2.setNumThreads(0)
except Exception:
    pass



## === cell 6
_HEIGHT, _WIDTH = 101, 101
if _HEIGHT % 32 == 0:
    _Y_MIN_PAD = 0
    _Y_MAX_PAD = 0
else:
    _Y_PAD = 32 - _HEIGHT % 32
    _Y_MIN_PAD = int(_Y_PAD / 2)
    _Y_MAX_PAD = _Y_PAD - _Y_MIN_PAD

if _WIDTH % 32 == 0:
    _X_MIN_PAD = 0
    _X_MAX_PAD = 0
else:
    _X_PAD = 32 - _WIDTH % 32
    _X_MIN_PAD = int(_X_PAD / 2)
    _X_MAX_PAD = _X_PAD - _X_MIN_PAD


def load_image(path, mask=False):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image at path: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.copyMakeBorder(
        img, _Y_MIN_PAD, _Y_MAX_PAD, _X_MIN_PAD, _X_MAX_PAD, cv2.BORDER_REFLECT_101
    )
    if mask:
        img = img[:, :, 0:1] // 255
        return torch.from_numpy(np.transpose(img, (2, 0, 1)).astype("float32"))
    else:
        img = img.astype(np.float32) * (1.0 / 255.0)
        return torch.from_numpy(np.transpose(img, (2, 0, 1)))




## === cell 7
class TGSSaltDataset(data.Dataset):
    def __init__(self, root_path, file_list, is_test=False):
        self.is_test = is_test
        self.root_path = root_path
        self.file_list = file_list
        self.image_folder = os.path.join(self.root_path, "images")
        self.mask_folder = os.path.join(self.root_path, "masks")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id = self.file_list[index]

        image_path = os.path.join(self.image_folder, file_id + ".png")
        image = load_image(image_path)

        if self.is_test:
            return (image,)
        else:
            mask_path = os.path.join(self.mask_folder, file_id + ".png")
            mask = load_image(mask_path, mask=True)
            return image, mask


train_csv_path = os.path.join(directory, "train.csv")
train_df = pd.read_csv(train_csv_path)

train_path = os.path.join(directory, "train")
file_list = list(train_df["id"].values)



## === cell 8
rng = np.random.RandomState(717)
perm = rng.permutation(len(file_list))
val_size = max(1, int(round(0.1 * len(file_list))))
val_idx = set(perm[:val_size].tolist())

file_list_val = [file_list[i] for i in range(len(file_list)) if i in val_idx]
file_list_train = [file_list[i] for i in range(len(file_list)) if i not in val_idx]

dataset = TGSSaltDataset(train_path, file_list_train)
dataset_val = TGSSaltDataset(train_path, file_list_val)

model = get_model()




## === cell 9
def _load_checkpoint(model, ckpt_path, device):
    if not os.path.exists(ckpt_path):
        warnings.warn(
            f"Checkpoint not found at {ckpt_path}. Proceeding with randomly initialized weights."
        )
        return model

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("state_dict", ckpt)

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        warnings.warn(f"Missing keys when loading checkpoint: {len(missing)}")
    if unexpected:
        warnings.warn(f"Unexpected keys when loading checkpoint: {len(unexpected)}")
    return model


model = _load_checkpoint(model, model_pth, device)
model.eval()



## === cell 10
test_path = os.path.join(directory, "test")
test_file_list = glob.glob(os.path.join(test_path, "images", "*.png"))
test_file_list = [os.path.basename(f).split(".")[0] for f in test_file_list]
test_file_list = sorted(test_file_list)
print("First 3 names of test files:", test_file_list[:3])



## === cell 11
pass



## === cell 12
_NUM_WORKERS = min(4, (os.cpu_count() or 2))
_PIN_MEMORY = device == "cuda"

print(f"Test size: {len(test_file_list)}")
test_dataset = TGSSaltDataset(test_path, test_file_list, is_test=True)

bs = 30
all_predictions_stacked = np.empty((len(test_dataset), 1, 128, 128), dtype=np.float32)
offset = 0

with torch.no_grad():
    for batch in data.DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=_PIN_MEMORY,
        persistent_workers=(_NUM_WORKERS > 0),
    ):
        image = batch[0].to(
            device=device, dtype=torch.float32, non_blocking=_PIN_MEMORY
        )
        y_pred = model.tta_flip(image).detach().cpu().numpy()
        b = y_pred.shape[0]
        all_predictions_stacked[offset : offset + b] = y_pred
        offset += b

all_predictions_stacked = all_predictions_stacked[:, 0, :, :]



## === cell 13
height, width = 101, 101

if height % 32 == 0:
    y_min_pad = 0
    y_max_pad = 0
else:
    y_pad = 32 - height % 32
    y_min_pad = int(y_pad / 2)
    y_max_pad = y_pad - y_min_pad

if width % 32 == 0:
    x_min_pad = 0
    x_max_pad = 0
else:
    x_pad = 32 - width % 32
    x_min_pad = int(x_pad / 2)
    x_max_pad = x_pad - x_min_pad



## === cell 14
all_predictions_stacked = all_predictions_stacked[
    :, y_min_pad : 128 - y_max_pad, x_min_pad : 128 - x_max_pad
]
print("Test preds shape:", all_predictions_stacked.shape)



## === cell 15
val_bs = 30
val_predictions_stacked = np.empty((len(dataset_val), 1, 128, 128), dtype=np.float32)
val_masks_stacked = np.empty((len(dataset_val), 1, 128, 128), dtype=np.float32)

offset = 0
with torch.no_grad():
    for image, mask in data.DataLoader(
        dataset_val,
        batch_size=val_bs,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        pin_memory=_PIN_MEMORY,
        persistent_workers=(_NUM_WORKERS > 0),
    ):
        image = image.to(device=device, dtype=torch.float32, non_blocking=_PIN_MEMORY)
        y_pred = model.tta_flip(image).detach().cpu().numpy()
        b = y_pred.shape[0]
        val_predictions_stacked[offset : offset + b] = y_pred
        val_masks_stacked[offset : offset + b] = mask.numpy()
        offset += b

val_predictions_stacked = val_predictions_stacked[:, 0, :, :]
val_masks_stacked = val_masks_stacked[:, 0, :, :]

val_predictions_stacked = val_predictions_stacked[
    :, y_min_pad : 128 - y_max_pad, x_min_pad : 128 - x_max_pad
]
val_masks_stacked = val_masks_stacked[
    :, y_min_pad : 128 - y_max_pad, x_min_pad : 128 - x_max_pad
]

val_masks_stacked = (val_masks_stacked > 0.5).astype(np.uint8)

print("Val masks/preds shape:", val_masks_stacked.shape, val_predictions_stacked.shape)




## === cell 16
def _label_connected_components(mask01: np.ndarray) -> np.ndarray:
    num_labels, labels = cv2.connectedComponents(
        mask01.astype(np.uint8), connectivity=8
    )
    return labels  # 0 is background, 1..num_labels-1 are objects


def _iou_matrix(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    true_labels = true_labels.astype(np.int32, copy=False)
    pred_labels = pred_labels.astype(np.int32, copy=False)

    t_max = int(true_labels.max())
    p_max = int(pred_labels.max())

    if t_max == 0 or p_max == 0:
        return np.zeros((t_max, p_max), dtype=np.float32)

    idx = true_labels.ravel() * (p_max + 1) + pred_labels.ravel()
    inter = (
        np.bincount(idx, minlength=(t_max + 1) * (p_max + 1))
        .reshape((t_max + 1, p_max + 1))
        .astype(np.float32)
    )

    t_area = inter.sum(axis=1)  # (t_max+1,)
    p_area = inter.sum(axis=0)  # (p_max+1,)

    inter_fg = inter[1:, 1:]  # drop background
    union_fg = t_area[1:, None] + p_area[None, 1:] - inter_fg
    iou = np.divide(
        inter_fg,
        union_fg,
        out=np.zeros_like(inter_fg, dtype=np.float32),
        where=(union_fg > 0),
    )
    return iou


def _ap_at_iou_thresholds_from_iou(iou: np.ndarray, n_true: int, n_pred: int) -> float:
    if n_true == 0 and n_pred == 0:
        return 1.0
    if n_true == 0 and n_pred > 0:
        return 0.0
    if n_true > 0 and n_pred == 0:
        return 0.0

    thresholds = np.arange(0.5, 1.0, 0.05, dtype=np.float32)
    precisions = []

    pairs = np.argwhere(iou > 0)
    if pairs.size > 0:
        pair_ious = iou[pairs[:, 0], pairs[:, 1]]
        order = np.argsort(-pair_ious)
        pairs = pairs[order]
        pair_ious = pair_ious[order]
    else:
        pair_ious = np.empty((0,), dtype=np.float32)

    for t in thresholds:
        if pair_ious.size == 0:
            tp = 0
        else:
            k = int(np.searchsorted(-pair_ious, -t, side="left"))
            if k <= 0:
                tp = 0
            else:
                matched_true = np.zeros((n_true,), dtype=np.uint8)
                matched_pred = np.zeros((n_pred,), dtype=np.uint8)
                tp = 0
                for i, j in pairs[:k]:
                    if matched_true[i] or matched_pred[j]:
                        continue
                    matched_true[i] = 1
                    matched_pred[j] = 1
                    tp += 1

        fp = n_pred - tp
        fn = n_true - tp
        denom = tp + fp + fn
        precisions.append(tp / denom if denom > 0 else 0.0)

    return float(np.mean(precisions))


def _postprocess_mask(
    mask01: np.ndarray,
    min_size: int = 10,
    hole_size: int = 10,
    do_open: int = 0,
    do_close: int = 1,
) -> np.ndarray:
    m = (mask01 > 0).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    if int(do_close) == 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=1)
    if int(do_open) == 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels > 1 and int(min_size) > 0:
        for lab in range(1, num_labels):
            if stats[lab, cv2.CC_STAT_AREA] < min_size:
                m[labels == lab] = 0

    inv = (1 - m).astype(np.uint8)
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(
        inv, connectivity=8
    )
    if num_labels2 > 1 and int(hole_size) > 0:
        h, w = m.shape
        for lab in range(1, num_labels2):
            area = stats2[lab, cv2.CC_STAT_AREA]
            x, y, ww, hh = (
                stats2[lab, cv2.CC_STAT_LEFT],
                stats2[lab, cv2.CC_STAT_TOP],
                stats2[lab, cv2.CC_STAT_WIDTH],
                stats2[lab, cv2.CC_STAT_HEIGHT],
            )
            touches_border = (x == 0) or (y == 0) or (x + ww == w) or (y + hh == h)
            if (not touches_border) and area < hole_size:
                m[labels2 == lab] = 1

    return m.astype(np.uint8)


val_true_labels_list = []
val_n_true = np.empty((len(val_masks_stacked),), dtype=np.int32)
for i, m in enumerate(val_masks_stacked):
    tl = _label_connected_components(m.astype(np.uint8))
    val_true_labels_list.append(tl)
    val_n_true[i] = int(tl.max())

metric_by_params = []

threshold_grid = np.linspace(0.30, 0.70, 81)  # step 0.005

min_size_grid = [0, 2, 3, 5, 8, 10, 15]
hole_size_grid = [0, 2, 3, 5, 8, 10, 15]
do_open_grid = [0, 1]
do_close_grid = [0, 1]

best_metric = -1.0
best_threshold = 0.5
best_min_size = 10
best_hole_size = 10
best_do_open = 0
best_do_close = 1

EPS_TIE = 1e-6

binary_cache = {}  # thr -> (N,H,W) uint8
pp_labels_cache = (
    {}
)  # (thr, min_size, hole_size, do_open, do_close, idx) -> (labels, n_pred)

t0 = time.time()
total_configs = (
    len(do_close_grid)
    * len(do_open_grid)
    * len(min_size_grid)
    * len(hole_size_grid)
    * len(threshold_grid)
)
cfg_i = 0

for do_close in do_close_grid:
    for do_open in do_open_grid:
        for min_size in min_size_grid:
            for hole_size in hole_size_grid:
                for threshold in threshold_grid:
                    cfg_i += 1
                    thr_key = float(threshold)

                    if thr_key not in binary_cache:
                        binary_cache[thr_key] = (
                            val_predictions_stacked > thr_key
                        ).astype(np.uint8)
                    val_binary_prediction = binary_cache[thr_key]

                    ap_scores = np.empty((len(val_true_labels_list),), dtype=np.float32)

                    for idx, true_labels in enumerate(val_true_labels_list):
                        cache_key = (
                            thr_key,
                            int(min_size),
                            int(hole_size),
                            int(do_open),
                            int(do_close),
                            int(idx),
                        )
                        cached = pp_labels_cache.get(cache_key, None)
                        if cached is None:
                            p_mask_pp = _postprocess_mask(
                                val_binary_prediction[idx],
                                min_size=min_size,
                                hole_size=hole_size,
                                do_open=do_open,
                                do_close=do_close,
                            )
                            pred_labels = _label_connected_components(
                                p_mask_pp.astype(np.uint8)
                            )
                            n_pred = int(pred_labels.max())
                            pp_labels_cache[cache_key] = (pred_labels, n_pred)
                        else:
                            pred_labels, n_pred = cached

                        iou = _iou_matrix(true_labels, pred_labels)
                        ap_scores[idx] = _ap_at_iou_thresholds_from_iou(
                            iou, int(val_n_true[idx]), int(n_pred)
                        )

                    score = float(np.mean(ap_scores))
                    metric_by_params.append(
                        (
                            score,
                            float(threshold),
                            int(min_size),
                            int(hole_size),
                            int(do_open),
                            int(do_close),
                        )
                    )

                    if (cfg_i % 80) == 0 or cfg_i == 1 or cfg_i == total_configs:
                        elapsed = time.time() - t0
                        print(
                            f"[{cfg_i}/{total_configs}] close={do_close} open={do_open} min_size={min_size:>2d} hole_size={hole_size:>2d} thr={threshold:.4f} -> Metric: {score:.4f} (elapsed {elapsed:.1f}s)"
                        )

                    better = score > best_metric + EPS_TIE
                    tied = abs(score - best_metric) <= EPS_TIE
                    if better or (tied and float(threshold) > best_threshold):
                        best_metric = score
                        best_threshold = float(threshold)
                        best_min_size = int(min_size)
                        best_hole_size = int(hole_size)
                        best_do_open = int(do_open)
                        best_do_close = int(do_close)

print("Best threshold:", best_threshold, "Best metric:", best_metric)
print(
    "Best postprocess: close =",
    best_do_close,
    "open =",
    best_do_open,
    "min_size =",
    best_min_size,
    "hole_size =",
    best_hole_size,
)



## === cell 17
threshold = best_threshold

binary_prediction = (all_predictions_stacked > threshold).astype(np.uint8)
binary_prediction = np.stack(
    [
        _postprocess_mask(
            m,
            min_size=best_min_size,
            hole_size=best_hole_size,
            do_open=best_do_open,
            do_close=best_do_close,
        )
        for m in binary_prediction
    ],
    axis=0,
).astype(np.uint8)


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


all_masks = []
for p_mask in list(binary_prediction):
    rle = rle_encoding(p_mask)
    all_masks.append(" ".join(map(str, rle)))



## === cell 18
pass



## === cell 19
submit = pd.DataFrame({"id": test_file_list, "rle_mask": all_masks})

assert list(submit.columns) == ["id", "rle_mask"]
assert len(submit) == len(test_file_list)

submit_path = "submit_baseline_torch_with_tta.csv"
submit.to_csv(submit_path, index=False)
print("Wrote submission to:", submit_path)
print(submit.head())
