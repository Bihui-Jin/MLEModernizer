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

0.1293

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.1293) has done: 'I fixed the GPU‑related crashes by forcing the code to run on CPU, added safe handling for missing pretrained weights, replaced the removed `jaccard_similarity_score` with the current `jaccard_score`, and ensured all variables needed for the later steps are defined (fallback threshold = 0.5). These changes let the notebook run end‑to‑end and produce a valid `submit_baseline_torch_with_tta.csv` file while keeping the original model architecture unchanged.'

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



## === cell 1
torch.__version__




## === cell 2
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
        with torch.no_grad():
            result = self.forward(x)
            x_h = self.hflip(x)
            result += self.hflip(self.forward(x_h))
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


class UNet11(TTAFunction, nn.Module):
    def __init__(self, num_filters=32):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11(pretrained=True).features

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

        self.final = nn.Conv2d(num_filters, 1, kernel_size=1)

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
    return UNet11(**kwargs)


def get_model(device):
    np.random.seed(717)
    torch.manual_seed(717)
    model = unet11()
    model.train()
    return model.to(device)




## === cell 4
model_pth = "../input/goto-pytorch-fix-for-v0-3/tgs-13.pth"



## === cell 5
directory = "../input/tgs-salt-identification-challenge"
device = torch.device("cpu")




## === cell 6
def load_image(path, mask=False):
    """
    Load an image, pad it so each side is divisible by 32 (network requirement)
    Returns a torch FloatTensor (C, H, W). If mask=True returns a single‑channel mask tensor.
    """
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, _ = img.shape

    y_pad = (32 - height % 32) if height % 32 != 0 else 0
    x_pad = (32 - width % 32) if width % 32 != 0 else 0
    y_min_pad = y_pad // 2
    y_max_pad = y_pad - y_min_pad
    x_min_pad = x_pad // 2
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
        if index < 0 or index >= len(self.file_list):
            index = np.random.randint(0, len(self.file_list))
        file_id = self.file_list[index]
        image_path = os.path.join(self.root_path, "images", f"{file_id}.png")
        image = load_image(image_path)
        if self.is_test:
            return (image,)
        mask_path = os.path.join(self.root_path, "masks", f"{file_id}.png")
        mask = load_image(mask_path, mask=True)
        return image, mask


depths_df = pd.read_csv(os.path.join(directory, "train.csv"))
train_path = os.path.join(directory, "train")
file_list = list(depths_df["id"].values)



## === cell 8
file_list_val = file_list[::10]
file_list_train = [f for f in file_list if f not in file_list_val]

dataset = TGSSaltDataset(train_path, file_list_train, is_test=False)
dataset_val = TGSSaltDataset(train_path, file_list_val, is_test=False)



## === cell 9
model = get_model(device)



## === cell 10
if os.path.exists(model_pth):
    try:
        checkpoint = torch.load(model_pth, map_location=device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("Loaded pretrained weights.")
    except Exception as e:
        print(f"Warning: could not load weights ({e}). Using random initialization.")
else:
    print("Checkpoint not found – proceeding with random weights.")



## === cell 11
test_path = os.path.join(directory, "test")
test_files_glob = glob.glob(os.path.join(test_path, "images", "*.png"))
test_file_list = [os.path.splitext(os.path.basename(p))[0] for p in test_files_glob]
print("First 3 test ids:", test_file_list[:3])



## === cell 12
model.eval()
all_predictions = []
test_dataset = TGSSaltDataset(test_path, test_file_list, is_test=True)

for batch in data.DataLoader(test_dataset, batch_size=30, shuffle=False):
    imgs = batch[0].to(device).float()
    preds = model.tta(imgs).cpu().numpy()  # shape (B,1,H,W)
    all_predictions.append(preds)

all_predictions_stacked = np.vstack(all_predictions)[:, 0, :, :]



## === cell 13
height, width = 101, 101
y_pad = (32 - height % 32) if height % 32 != 0 else 0
x_pad = (32 - width % 32) if width % 32 != 0 else 0
y_min_pad = y_pad // 2
y_max_pad = y_pad - y_min_pad
x_min_pad = x_pad // 2
x_max_pad = x_pad - x_min_pad

all_predictions_stacked = all_predictions_stacked[
    :, y_min_pad : 101 + y_min_pad, x_min_pad : 101 + x_min_pad
]

print("Test predictions shape:", all_predictions_stacked.shape)



## === cell 14
from sklearn.metrics import jaccard_score

metric_by_threshold = []
for thresh in np.linspace(0, 1, 11):
    bin_pred = (
        np.stack(
            [p for p in data.DataLoader(dataset_val, batch_size=30, shuffle=False)][
                :, 0
            ]
        )
        > thresh
    ).astype(int)
    val_preds = []
    val_masks = []
    for imgs, masks in data.DataLoader(dataset_val, batch_size=30, shuffle=False):
        imgs = imgs.to(device).float()
        preds = model.tta(imgs).cpu().numpy()
        val_preds.append(preds)
        val_masks.append(masks.numpy())
    val_pred_arr = np.vstack(val_preds)[:, 0, :, :]
    val_mask_arr = np.vstack(val_masks)[:, 0, :, :]

    val_pred_arr = val_pred_arr[
        :, y_min_pad : 101 + y_min_pad, x_min_pad : 101 + x_min_pad
    ]
    val_mask_arr = val_mask_arr[
        :, y_min_pad : 101 + y_min_pad, x_min_pad : 101 + x_min_pad
    ]

    bin_pred = (val_pred_arr > thresh).astype(int)
    ious = []
    for y_true, y_pred in zip(val_mask_arr, bin_pred):
        iou = jaccard_score(y_true.flatten(), y_pred.flatten())
        ious.append(iou)
    ious = np.array(ious)
    accuracies = [np.mean(ious > t) for t in np.linspace(0.5, 0.95, 10)]
    metric = np.mean(accuracies)
    print(f"Threshold {thresh:.2f} – Metric {metric:.4f}")
    metric_by_threshold.append((metric, thresh))

best_metric, best_threshold = max(metric_by_threshold, key=lambda x: x[0])
print("Best validation metric:", best_metric, "at threshold", best_threshold)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3143155370.py in <cell line: 0>()
      6     bin_pred = (
      7         np.stack(
----> 8             [p for p in data.DataLoader(dataset_val, batch_size=30, shuffle=False)][
      9                 :, 0
     10             ]

TypeError: list indices must be integers or slices, not tuple

## === cell 15
if "best_threshold" not in globals():
    best_threshold = 0.5
    print("Using default threshold 0.5")



## === cell 16
binary_prediction = (all_predictions_stacked > best_threshold).astype(int)


def rle_encoding(mask):
    """Run‑length encoding as used in the competition."""
    dots = np.where(mask.T.flatten() == 1)[0]
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



## === cell 17
submit = pd.DataFrame({"id": test_file_list, "rle_mask": all_masks})
submit.to_csv("submit_baseline_torch_with_tta.csv", index=False)
print("Submission saved to submit_baseline_torch_with_tta.csv")
