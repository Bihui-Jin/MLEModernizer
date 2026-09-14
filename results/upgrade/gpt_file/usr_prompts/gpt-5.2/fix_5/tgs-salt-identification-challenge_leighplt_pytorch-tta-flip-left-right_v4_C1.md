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

# 5. Target score

0.6982666666666653

# 6. Current score

0.3866

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.3866) has done: 'I fix the runtime errors by making device selection robust (CPU fallback when no GPU is available), ensuring the pretrained checkpoint path is optional so the script still runs end-to-end even if the file is missing, and updating the deprecated `jaccard_similarity_score` to the modern `jaccard_score` without changing evaluation semantics. I also correct the mistaken CSV read (it should use `depths.csv` for IDs, not `train.csv`) and make file path handling OS-independent. Finally, I guarantee a valid submission CSV is always produced by defining a safe default threshold when validation threshold search can’t run (e.g., no weights loaded), while keeping the same UNet+TTA inference logic.'
- What this solution (achieved 0.3866) has done: 'Main runtime bottlenecks here are (1) heavy per-sample PNG decoding + padding in `__getitem__` (called 1000+ times for test and again for val), (2) the recursive glob-based checkpoint search walking huge directory trees, and (3) CPU↔GPU transfer overhead that’s not fully overlapped. The changes below keep the exact same model, weights usage, TTA logic, metric computation, and RLE formatting, but reduce wall time by (a) eliminating expensive recursive checkpoint search in common cases, (b) caching decoded+preprocessed images/masks in RAM for val/test to avoid repeated disk I/O, and (c) using faster DataLoader settings (prefetch + pinned memory usage without redundant `.pin_memory()` calls). These are provably equivalent because they only memoize pure preprocessing outputs and avoid repeated filesystem work, without changing any numerical operations on tensors.'

# 9. Code solution

## === cell 0
import os
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils import data
from torchvision import models

import cv2

SEED = 717
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = (
    False  # preserves evaluation semantics; allows faster kernels
)
torch.backends.cudnn.benchmark = (
    True  # fixed input sizes (128x128) => faster conv selection
)

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())




## === cell 1
class TTAFunction:
    """
    Simple TTA function
    """

    @staticmethod
    def hflip(x):
        return x.flip(3)

    @staticmethod
    def vflip(x):
        return x.flip(2)

    def tta(self, x):
        self.eval()
        with torch.inference_mode():
            result = self.forward(x)
            x = self.hflip(x)
            result += self.hflip(self.forward(x))
        return 0.5 * result




## === cell 2
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


class UNet11(TTAFunction, nn.Module):  # use our class with TTA function
    def __init__(self, num_filters=32):
        """
        :param num_classes:
        :param num_filters:
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11().features

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
    torch.manual_seed(717)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(717)
    model = unet11()
    model.train()
    return model.to(device)




## === cell 3
model_pth = "../input/goto-pytorch-fix-for-v0-3/tgs-13.pth"


def _find_checkpoint_fallback(preferred_path, filename="tgs-13.pth"):
    if preferred_path is not None and os.path.exists(preferred_path):
        return preferred_path

    common_candidates = [
        os.path.join("/kaggle/input", "goto-pytorch-fix-for-v0-3", filename),
        os.path.join("../input", "goto-pytorch-fix-for-v0-3", filename),
        os.path.join("/kaggle/data/input", "goto-pytorch-fix-for-v0-3", filename),
        os.path.join("../kaggle/input", "goto-pytorch-fix-for-v0-3", filename),
    ]
    for p in common_candidates:
        if os.path.exists(p):
            return p

    search_roots = [
        "/kaggle/input",
        "../input",
        "/kaggle/data/input",
        "../kaggle/input",
    ]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        try:
            candidates = glob.glob(os.path.join(root, "*", filename))
        except Exception:
            candidates = []
        if candidates:
            candidates = sorted(candidates, key=lambda p: (len(p), p))
            return candidates[0]
    return None


model_pth = _find_checkpoint_fallback(model_pth, "tgs-13.pth")
print("Resolved checkpoint path:", model_pth)




## === cell 4
directory = "../input/tgs-salt-identification-challenge"
if not os.path.exists(directory):
    directory = "/kaggle/input/tgs-salt-identification-challenge"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using directory:", directory)
print("Using device:", device)

CPU_COUNT = os.cpu_count() or 2
NUM_WORKERS = min(4, CPU_COUNT)  # keep conservative to avoid oversubscription
PIN_MEMORY = torch.cuda.is_available()

PREFETCH_FACTOR = 4 if NUM_WORKERS > 0 else None




## === cell 5
HEIGHT, WIDTH = 101, 101
if HEIGHT % 32 == 0:
    Y_MIN_PAD = 0
    Y_MAX_PAD = 0
else:
    y_pad = 32 - HEIGHT % 32
    Y_MIN_PAD = int(y_pad / 2)
    Y_MAX_PAD = y_pad - Y_MIN_PAD

if WIDTH % 32 == 0:
    X_MIN_PAD = 0
    X_MAX_PAD = 0
else:
    x_pad = 32 - WIDTH % 32
    X_MIN_PAD = int(x_pad / 2)
    X_MAX_PAD = x_pad - X_MIN_PAD

CROP_Y0, CROP_Y1 = Y_MIN_PAD, 128 - Y_MAX_PAD
CROP_X0, CROP_X1 = X_MIN_PAD, 128 - X_MAX_PAD


def load_image(path, mask=False):
    """
    Load image from a given path and pad it on the sides, so that each side is divisible by 32 (network requirement)
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.copyMakeBorder(
        img, Y_MIN_PAD, Y_MAX_PAD, X_MIN_PAD, X_MAX_PAD, cv2.BORDER_REFLECT_101
    )
    if mask:
        img = img[:, :, 0:1] // 255
        return torch.from_numpy(np.transpose(img, (2, 0, 1)).astype("float32"))
    else:
        img = img / 255.0
        return torch.from_numpy(np.transpose(img, (2, 0, 1)).astype("float32"))




## === cell 6
class TGSSaltDataset(data.Dataset):
    def __init__(self, root_path, file_list, is_test=False, cache_images=False):
        self.is_test = is_test
        self.root_path = root_path
        self.file_list = file_list

        self.image_folder = os.path.join(self.root_path, "images")
        self.mask_folder = os.path.join(self.root_path, "masks")

        self.cache_images = cache_images
        self._img_cache = {}  # id -> torch.Tensor(C,128,128) on CPU
        self._mask_cache = {}  # id -> torch.Tensor(1,128,128) on CPU

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id = self.file_list[index]

        if self.cache_images and file_id in self._img_cache:
            image = self._img_cache[file_id]
        else:
            image_path = os.path.join(self.image_folder, file_id + ".png")
            image = load_image(image_path)
            if self.cache_images:
                self._img_cache[file_id] = image

        if self.is_test:
            return (image,)
        else:
            if self.cache_images and file_id in self._mask_cache:
                mask = self._mask_cache[file_id]
            else:
                mask_path = os.path.join(self.mask_folder, file_id + ".png")
                mask = load_image(mask_path, mask=True)
                if self.cache_images:
                    self._mask_cache[file_id] = mask
            return image, mask


depths_df = pd.read_csv(os.path.join(directory, "depths.csv"))

train_path = os.path.join(directory, "train")
file_list = list(depths_df["id"].values)




## === cell 7
file_list_val = file_list[::10]
val_set = set(file_list_val)
file_list_train = [f for f in file_list if f not in val_set]

dataset = TGSSaltDataset(train_path, file_list_train, cache_images=False)
dataset_val = TGSSaltDataset(train_path, file_list_val, cache_images=True)
print("Train/val sizes:", len(dataset), len(dataset_val))




## === cell 8
model = get_model()
print("Model created.")




## === cell 9
loaded_checkpoint = False
if model_pth is not None and os.path.exists(model_pth):
    ckpt = torch.load(model_pth, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    loaded_checkpoint = True
    print("Loaded checkpoint:", model_pth)
else:
    warnings.warn(
        f"Checkpoint not found. Running with random weights will produce a valid submission but likely low score."
    )




## === cell 10
test_path = os.path.join(directory, "test")

sample_sub_path = os.path.join(directory, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
test_file_list = sample_sub["id"].astype(str).tolist()

missing = []
for tid in test_file_list[:10]:
    if not os.path.exists(os.path.join(test_path, "images", tid + ".png")):
        missing.append(tid)
if len(missing) > 0:
    warnings.warn(
        f"Some test image files appear missing (showing up to 10 checked): {missing}"
    )

print("First 3 names of test files:", test_file_list[:3])




## === cell 11
pass




## === cell 12
print(f"Test size: {len(test_file_list)}")

test_dataset = TGSSaltDataset(
    test_path, test_file_list, is_test=True, cache_images=True
)

test_loader = data.DataLoader(
    test_dataset,
    batch_size=30,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=PREFETCH_FACTOR,
)

model.eval()
all_predictions = []
with torch.inference_mode():
    for batch in test_loader:
        image = batch[0].to(device, non_blocking=PIN_MEMORY).float()
        y_pred = model.tta(image).cpu().numpy()  # use tta_flip
        all_predictions.append(y_pred)

all_predictions_stacked = np.vstack(all_predictions)[:, 0, :, :]




## === cell 13
height, width = 101, 101
y_min_pad, y_max_pad = Y_MIN_PAD, Y_MAX_PAD
x_min_pad, x_max_pad = X_MIN_PAD, X_MAX_PAD




## === cell 14
all_predictions_stacked = all_predictions_stacked[:, CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
print("Test preds shape:", all_predictions_stacked.shape)




## === cell 15
val_loader = data.DataLoader(
    dataset_val,
    batch_size=30,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(NUM_WORKERS > 0),
    prefetch_factor=PREFETCH_FACTOR,
)

val_predictions = []
val_masks = []

model.eval()
with torch.inference_mode():
    for image, mask in val_loader:
        image = image.to(device, non_blocking=PIN_MEMORY).float()
        y_pred = model.tta(image).cpu().numpy()
        val_predictions.append(y_pred)
        val_masks.append(mask.numpy())

val_predictions_stacked = np.vstack(val_predictions)[:, 0, :, :]
val_masks_stacked = np.vstack(val_masks)[:, 0, :, :]

val_predictions_stacked = val_predictions_stacked[:, CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]
val_masks_stacked = val_masks_stacked[:, CROP_Y0:CROP_Y1, CROP_X0:CROP_X1]

print("Val masks/preds shapes:", val_masks_stacked.shape, val_predictions_stacked.shape)




## === cell 16
metric_by_threshold = []

y_true_flat = val_masks_stacked.reshape(val_masks_stacked.shape[0], -1).astype(np.uint8)

for threshold in np.linspace(0, 1, 11):
    y_pred_flat = (
        (val_predictions_stacked > threshold)
        .reshape(y_true_flat.shape[0], -1)
        .astype(np.uint8)
    )

    inter = (y_true_flat & y_pred_flat).sum(axis=1).astype(np.float32)
    union = (y_true_flat | y_pred_flat).sum(axis=1).astype(np.float32)

    iou_values = np.divide(
        inter, union, out=np.zeros_like(inter, dtype=np.float32), where=(union != 0)
    )

    accuracies = [
        np.mean(iou_values > iou_threshold)
        for iou_threshold in np.linspace(0.5, 0.95, 10)
    ]
    print("Threshold: %.1f, Metric: %.3f" % (threshold, np.mean(accuracies)))
    metric_by_threshold.append((np.mean(accuracies), threshold))

best_metric, best_threshold = max(metric_by_threshold, key=lambda x: x[0])

if not loaded_checkpoint:
    best_threshold = 0.5
print("Best threshold:", best_threshold, "Best val metric:", best_metric)




## === cell 17
threshold = best_threshold
binary_prediction = (all_predictions_stacked > threshold).astype(np.uint8)


def rle_encoding(x):
    dots = np.flatnonzero(x.T.reshape(-1) == 1)
    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


all_masks = []
for p_mask in binary_prediction:
    rle = rle_encoding(p_mask)
    all_masks.append(" ".join(map(str, rle)))




## === cell 18
pass




## === cell 19
submit = pd.DataFrame({"id": test_file_list, "rle_mask": all_masks})

submit.to_csv("submit_baseline_torch_with_tta.csv", index=False)
print("Wrote submission:", "submit_baseline_torch_with_tta.csv")
print(submit.head())
