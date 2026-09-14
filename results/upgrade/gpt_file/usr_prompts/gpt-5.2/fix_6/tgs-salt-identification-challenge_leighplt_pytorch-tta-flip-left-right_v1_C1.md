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

0.5866083333333325

# 6. Current score

0.5221

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5221) has done: 'I fix the runtime failures by making device selection robust (CPU fallback when no GPU is available) so the model and tensors can be moved safely. I also correct the incorrect CSV being read for IDs (it should use `train.csv` for IDs and `depths.csv` is not needed here) and make the model checkpoint loading resilient to common key mismatches. Finally, I replace the removed `jaccard_similarity_score` import with `jaccard_score` so the threshold search runs, then ensure predictions are cropped back to 101×101 and a valid `submit_*.csv` file is written with the required `id,rle_mask` columns.'
- What this solution (achieved 0.5221) has done: 'Your current score (0.5221) is below the target (0.5866), so we should make small, metric-aligned changes that typically improve IoU/AP without changing the model or training loop. The biggest low-risk win here is to apply the same TTA used on test (`tta_flip`) to validation when selecting the best threshold; otherwise you pick a suboptimal threshold and hurt the final submission. I also ensure `test_file_list` is deterministically ordered (sorted) so IDs and predictions always align reliably, preventing silent submission degradation. Finally, I keep everything else (model, weights, padding/cropping, RLE) identical to preserve core logic and semantics.'
- What this solution (achieved 0.5221) has done: 'Your current score (0.5221) is below the target (0.5866), so we should make a small, metric-aligned improvement without changing the model or training procedure. The lowest-risk gain here is to select the binarization threshold using the *same per-image metric definition as Kaggle* (mAP over IoU thresholds based on connected components), because the current threshold search uses a pixelwise IoU surrogate that can pick a suboptimal threshold for the leaderboard metric. We keep the exact same model, weights, TTA, padding/cropping, and RLE writing, and only replace the validation threshold-selection computation with a faithful implementation of the competition metric. This typically moves the score upward toward your target while preserving your core logic and evaluation semantics.'
- What this solution (achieved 0.5221) has done: 'Your current score (0.5221) is below the target (0.5866), so we should make a small, metric-aligned change that improves mask quality without touching the model/training logic. The lowest-risk gain here is to apply a lightweight morphological post-processing (remove tiny components + fill small holes) during threshold selection and final test binarization, because the Kaggle metric is object-based and is sensitive to small speckles/holes. We keep the exact same UNet, checkpoint loading, TTA, padding/cropping, and RLE format; only the conversion from probabilities to final binary masks changes. The threshold search choose the best threshold under the same post-processing, making it consistent and typically improving leaderboard mAP.'
- What this solution (achieved 0.5221) has done: 'We make the threshold search finer (still deterministic and using the same competition-metric implementation you already added) because the current coarse 0.1 step can easily pick a suboptimal cutoff and leave mAP on the table. We also ensure the validation masks are binarized strictly to {0,1} before connected-components and post-processing, avoiding any edge-case where non-binary values could affect component labeling. Finally, we keep the exact same model, checkpoint loading, TTA, padding/cropping, RLE encoding, and submission writing, so the core logic stays unchanged while improving calibration toward your target score.'

# 9. Code solution

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
            result = self.forward(x)
            result += self.forward(x.flip(2)).flip(2)  # apply flip and back
        return 0.5 * result




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




## === cell 6
def load_image(path, mask=False):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (newtwork requirement)
    """
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image at path: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, _ = img.shape

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

    img = cv2.copyMakeBorder(
        img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101
    )
    if mask:
        img = img[:, :, 0:1] // 255
        return torch.from_numpy(np.transpose(img, (2, 0, 1)).astype("float32"))
    else:
        img = img / 255.0
        return torch.from_numpy(np.transpose(img, (2, 0, 1)).astype("float32"))




## === cell 7
class TGSSaltDataset(data.Dataset):
    def __init__(self, root_path, file_list, is_test=False):
        self.is_test = is_test
        self.root_path = root_path
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if index not in range(0, len(self.file_list)):
            return self.__getitem__(np.random.randint(0, self.__len__()))

        file_id = self.file_list[index]

        image_folder = os.path.join(self.root_path, "images")
        image_path = os.path.join(image_folder, file_id + ".png")

        mask_folder = os.path.join(self.root_path, "masks")
        mask_path = os.path.join(mask_folder, file_id + ".png")

        image = load_image(image_path)

        if self.is_test:
            return (image,)
        else:
            mask = load_image(mask_path, mask=True)
            return image, mask


train_csv_path = os.path.join(directory, "train.csv")
train_df = pd.read_csv(train_csv_path)

train_path = os.path.join(directory, "train")
file_list = list(train_df["id"].values)



## === cell 8
file_list_val = file_list[::10]
file_list_train = [f for f in file_list if f not in file_list_val]
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
print(f"Test size: {len(test_file_list)}")
test_dataset = TGSSaltDataset(test_path, test_file_list, is_test=True)

all_predictions = []
with torch.no_grad():
    for batch in data.DataLoader(
        test_dataset, batch_size=30, shuffle=False, num_workers=0
    ):
        image = batch[0].float().to(device)
        y_pred = model.tta_flip(image).cpu().numpy()  # use tta_flip
        all_predictions.append(y_pred)

all_predictions_stacked = np.vstack(all_predictions)[:, 0, :, :]



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
val_predictions = []
val_masks = []
with torch.no_grad():
    for image, mask in data.DataLoader(
        dataset_val, batch_size=30, shuffle=False, num_workers=0
    ):
        image = image.float().to(device)
        y_pred = model.tta_flip(image).cpu().numpy()
        val_predictions.append(y_pred)
        val_masks.append(mask.numpy())

val_predictions_stacked = np.vstack(val_predictions)[:, 0, :, :]
val_masks_stacked = np.vstack(val_masks)[:, 0, :, :]

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
        mask01.astype(np.uint8), connectivity=4
    )
    return labels  # 0 is background, 1..num_labels-1 are objects


def _iou_matrix(true_labels: np.ndarray, pred_labels: np.ndarray) -> np.ndarray:
    true_ids = np.unique(true_labels)
    pred_ids = np.unique(pred_labels)

    true_ids = true_ids[true_ids != 0]
    pred_ids = pred_ids[pred_ids != 0]

    if true_ids.size == 0 or pred_ids.size == 0:
        return np.zeros((true_ids.size, pred_ids.size), dtype=np.float32)

    ious = np.zeros((true_ids.size, pred_ids.size), dtype=np.float32)
    true_masks = [(true_labels == tid) for tid in true_ids]
    pred_masks = [(pred_labels == pid) for pid in pred_ids]

    for i, tm in enumerate(true_masks):
        tm_sum = tm.sum()
        for j, pm in enumerate(pred_masks):
            inter = np.logical_and(tm, pm).sum()
            if inter == 0:
                continue
            union = tm_sum + pm.sum() - inter
            ious[i, j] = inter / union if union > 0 else 0.0
    return ious


def _ap_at_iou_thresholds(y_true01: np.ndarray, y_pred01: np.ndarray) -> float:
    true_labels = _label_connected_components(y_true01)
    pred_labels = _label_connected_components(y_pred01)

    true_ids = np.unique(true_labels)
    pred_ids = np.unique(pred_labels)
    n_true = int((true_ids != 0).sum())
    n_pred = int((pred_ids != 0).sum())

    if n_true == 0 and n_pred == 0:
        return 1.0
    if n_true == 0 and n_pred > 0:
        return 0.0
    if n_true > 0 and n_pred == 0:
        return 0.0

    ious = _iou_matrix(true_labels, pred_labels)
    thresholds = np.arange(0.5, 1.0, 0.05, dtype=np.float32)

    precisions = []
    for t in thresholds:
        pairs = np.argwhere(ious > t)
        if pairs.size == 0:
            tp = 0
        else:
            pair_ious = ious[pairs[:, 0], pairs[:, 1]]
            order = np.argsort(-pair_ious)
            pairs = pairs[order]

            matched_true = set()
            matched_pred = set()
            tp = 0
            for i, j in pairs:
                if i in matched_true or j in matched_pred:
                    continue
                matched_true.add(int(i))
                matched_pred.add(int(j))
                tp += 1

        fp = n_pred - tp
        fn = n_true - tp
        denom = tp + fp + fn
        precisions.append(tp / denom if denom > 0 else 0.0)

    return float(np.mean(precisions))


def _postprocess_mask(
    mask01: np.ndarray, min_size: int = 10, hole_size: int = 10
) -> np.ndarray:
    m = (mask01 > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
    if num_labels > 1:
        for lab in range(1, num_labels):
            if stats[lab, cv2.CC_STAT_AREA] < min_size:
                m[labels == lab] = 0

    inv = (1 - m).astype(np.uint8)
    num_labels2, labels2, stats2, _ = cv2.connectedComponentsWithStats(
        inv, connectivity=4
    )
    if num_labels2 > 1:
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


metric_by_threshold = []

threshold_grid = np.linspace(0.05, 0.95, 19)  # step 0.05, avoid degenerate 0/1

for threshold in threshold_grid:
    val_binary_prediction = (val_predictions_stacked > threshold).astype(np.uint8)

    ap_scores = []
    for y_mask, p_mask in zip(val_masks_stacked, val_binary_prediction):
        p_mask_pp = _postprocess_mask(p_mask, min_size=10, hole_size=10)
        ap_scores.append(
            _ap_at_iou_thresholds(y_mask.astype(np.uint8), p_mask_pp.astype(np.uint8))
        )

    score = float(np.mean(ap_scores))
    print("Threshold: %.2f, Metric: %.4f" % (threshold, score))
    metric_by_threshold.append((score, float(threshold)))

best_metric, best_threshold = max(metric_by_threshold, key=lambda x: x[0])
print("Best threshold:", best_threshold, "Best metric:", best_metric)



## === cell 17
threshold = best_threshold

binary_prediction = (all_predictions_stacked > threshold).astype(np.uint8)
binary_prediction = np.stack(
    [_postprocess_mask(m, min_size=10, hole_size=10) for m in binary_prediction], axis=0
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
