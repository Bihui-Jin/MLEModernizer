# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Overview
Detect the presence of ink from 3d x-ray scans of detached fragments of ancient papyrus scrolls.

## Metric
We evaluate how well your output image matches our reference image using a modified version of the [Sørensen--Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient), where instead of using the F1 score, we are using the F0.5 score. The F0.5 score is given by:

$$
\frac{\left(1+\beta^2\right) p r}{\beta^2 p+r} \text { where } p=\frac{t p}{t p+f p}, r=\frac{t p}{t p+f n}, \beta=0.5
$$

The F0.5 score weights precision higher than recall, which improves the ability to form coherent characters out of detected ink areas.

In order to reduce the submission file size, our metric uses run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the output should be binary, with 0 indicating "no ink" and 1 indicating "ink".

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from left to right, then top to bottom: 1 is pixel (1,1), 2 is pixel (1,2), etc.

Your output should be a single file, **submission.csv**, with this run-length encoded information. This should have a header with two columns, `Id` and `Predicted`, and with one row for every directory under **test/**. For example:

```
Id,Predicted
a,1 1 5 1 etc.
b,10 20 etc.
```

For a real-world example of what these files look like, see `inklabels_rce.csv` in the data directories, which have been generated with [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0).


## Data
- **[train/test]/[fragment_id]/surface_volume/[image_id].tif** slices from the 3d x-ray [surface volume](https://scrollprize.org/tutorial1#3-surface-volumes). Each file contains a greyscale slice in the z-direction. Each fragment contains 65 slices. Combined this image stack gives us `width * height * 65` number of voxels per fragment. You can expect two fragments in the hidden test set, which together are roughly the same size as a single training fragment. The sample slices available to download in the test folders are simply copied from training fragment one, but when you submit your notebook they will be substituted with the real test data.
- **[train/test]/[fragment_id]/mask.png** --- a binary mask of which pixels contain data.
- **train/[fragment_id]/inklabels.png** --- a binary mask of the ink vs no-ink labels.
- **train/[fragment_id]/inklabels_rle.csv** --- a run-length-encoded version of the labels, generated using [this script](https://gist.github.com/janpaul123/ca3477c1db6de4346affca37e0e3d5b0). This is the same format as you should make your submission in.
- **train/[fragment_id]/ir.png** --- the infrared photo on which the binary mask is based.
- **sample_submission.csv**, an example of a submission file in the correct format. You need to output the following file in the home directory: **submission.csv**.

# 2. Python version

3.11

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
pillow==11.3.0
protobuf==6.33.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (135 lines)
            sample_submission.csv (2 lines)
            sample_submission.csv.zip (215 Bytes)
            test.zip (3.0 GB)
            train.zip (15.5 GB)
            test/
                a/
                    mask.png (40.7 kB)
                    surface_volume/
                        06.tif (79.8 MB)
                        01.tif (79.8 MB)
                        ... and 63 other files
                test/
            train/
                1/
                    inklabels.png (92.6 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        42.tif (103.6 MB)
                        45.tif (103.6 MB)
                        ... and 63 other files
                2/
                    inklabels.png (294.3 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        10.tif (281.9 MB)
                        17.tif (281.9 MB)
                        ... and 63 other files
                train/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
        input/
            description.md (135 lines)
            sample_submission.csv (2 lines)
            sample_submission.csv.zip (215 Bytes)
            test.zip (3.0 GB)
            train.zip (15.5 GB)
            test/
                a/
                    mask.png (40.7 kB)
                    surface_volume/
                        06.tif (79.8 MB)
                        01.tif (79.8 MB)
                        ... and 63 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
            train/
                1/
                    inklabels.png (92.6 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        42.tif (103.6 MB)
                        45.tif (103.6 MB)
                        ... and 63 other files
                2/
                    inklabels.png (294.3 kB)
                    inklabels_rle.csv (2 lines)
                    ... and 2 other files
                    surface_volume/
                        10.tif (281.9 MB)
                        17.tif (281.9 MB)
                        ... and 63 other files
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
        working/
            vesuvius-challenge-ink-detection/
                description.md (135 lines)
                sample_submission.csv (2 lines)
                ... and 3 other files
                test/
                    a/
                        mask.png (40.7 kB)
                        surface_volume/
                            ... (max depth reached)
                    test/
                train/
                    1/
                        inklabels.png (92.6 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    2/
                        inklabels.png (294.3 kB)
                        inklabels_rle.csv (2 lines)
                        ... and 2 other files
                        surface_volume/
                            ... (max depth reached)
                    train/
                vesuvius-challenge-ink-detection/
```

-> data/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/train/1/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/train/2/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/train/1/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> data/vesuvius-challenge-ink-detection/train/2/inklabels_rle.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> input/sample_submission.csv has 1 rows and 2 columns.
The columns are: Id, Predicted

-> (stopped after 10 files for performance)

# 5. Target score

0.0395623179948312

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import random
import math
import cv2
from torchvision import transforms
from tqdm import tqdm
import warnings
from collections import defaultdict




## === cell 1
DEVICE = torch.device("cpu")




## === cell 2
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        channels = int(out_channels / 2)
        if in_channels > out_channels:
            channels = int(in_channels / 2)

        layers = [
            nn.Conv2d(in_channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        ]
        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class DownSampling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampling, self).__init__()
        self.maxpool_to_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_to_conv(x)


class UpSampling(nn.Module):
    """
    Upsample the feature map and concatenate with the corresponding skip connection.
    If spatial sizes differ by one pixel (common with odd dimensions), the skip
    connection is cropped to match the upsampled tensor.
    """

    def __init__(self, in_channels, out_channels):
        super(UpSampling, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = DoubleConv(in_channels + int(in_channels / 2), out_channels)

    def forward(self, inputs1, inputs2):
        inputs1 = self.up(inputs1)

        if inputs2.shape[2] != inputs1.shape[2] or inputs2.shape[3] != inputs1.shape[3]:
            diff_y = inputs2.shape[2] - inputs1.shape[2]
            diff_x = inputs2.shape[3] - inputs1.shape[3]
            start_y = diff_y // 2
            start_x = diff_x // 2
            end_y = start_y + inputs1.shape[2]
            end_x = start_x + inputs1.shape[3]
            inputs2 = inputs2[:, :, start_y:end_y, start_x:end_x]

        outputs = torch.cat([inputs1, inputs2], dim=1)
        outputs = self.conv(outputs)
        return outputs


class LastConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LastConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class InkDetector(nn.Module):
    def __init__(self, in_channels=66):
        super(InkDetector, self).__init__()
        self.inputs = DoubleConv(in_channels, 64)
        self.down_1 = DownSampling(64, 128)
        self.down_2 = DownSampling(128, 256)
        self.up_2 = UpSampling(256, 128)
        self.up_3 = UpSampling(128, 64)
        self.outputs = LastConv(64, 1)

    def forward(self, x):
        x1 = self.inputs(x)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x6 = self.up_2(x3, x2)
        x7 = self.up_3(x6, x1)
        return self.outputs(x7)




## === cell 3
model = InkDetector().to(DEVICE)




## === cell 4
WEIGHT_PATH = "/kaggle/input/kalen9099/32.pt"
weights_loaded = False
if os.path.exists(WEIGHT_PATH):
    try:
        model_weights = torch.load(WEIGHT_PATH, map_location=DEVICE)
        model.load_state_dict(model_weights)
        weights_loaded = True
    except Exception as e:
        print(
            f"Warning: failed to load weights – proceeding with random initialization. ({e})"
        )
else:
    print("Weight file not found – proceeding with random initialization.")




## === cell 5
TRAIN_RUN = False




## === cell 6
warnings.simplefilter("ignore")




## === cell 7
test_root = "/kaggle/input/vesuvius-challenge-ink-detection/test"
if not os.path.isdir(test_root):
    test_root = "/kaggle/input/vesuvius-challenge/test"

test_ids = [
    name
    for name in os.listdir(test_root)
    if os.path.isdir(os.path.join(test_root, name))
]

submission_df = pd.DataFrame({"Id": test_ids, "Predicted": [""] * len(test_ids)})




## === cell 8
import torch.nn.functional as F


def rle_encode(mask):
    """
    Convert binary mask (2D numpy array) to run‑length encoding string.
    """
    pixels = mask.flatten(order="C")
    pixels = np.where(pixels > 0, 1, 0).astype(np.int8)

    padding = np.array([0])
    padded = np.concatenate([padding, pixels, padding])
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0] + 1  # 1‑based indexing
    ends = np.where(diffs == -1)[0] + 1
    lengths = ends - starts
    rle = " ".join(str(s) + " " + str(l) for s, l in zip(starts, lengths))
    return rle


THRESHOLD = 0.6

model.eval()
with torch.no_grad():
    for idx, fragment_id in enumerate(tqdm(test_ids, desc="Predicting")):
        fragment_path = os.path.join(test_root, fragment_id)
        vol_path = os.path.join(fragment_path, "surface_volume")
        if not os.path.isdir(vol_path):
            continue

        slice_files = sorted(
            [f for f in os.listdir(vol_path) if f.lower().endswith(".tif")]
        )
        slices = []
        for sf in slice_files:
            img = cv2.imread(os.path.join(vol_path, sf), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            img = img.astype(np.float32) / 255.0
            slices.append(img)
        if len(slices) == 0:
            continue
        volume = np.stack(slices, axis=0)  # shape: (C, H, W)

        mask_path = os.path.join(fragment_path, "mask.png")
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            mask_norm = np.ones((volume.shape[1], volume.shape[2]), dtype=np.float32)
        else:
            mask_resized = cv2.resize(
                mask_img,
                (volume.shape[2], volume.shape[1]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_norm = (mask_resized > 0).astype(np.float32)  # binary mask

        volume = np.concatenate(
            [volume, mask_norm[np.newaxis, ...]], axis=0
        )  # (66, H, W)

        if weights_loaded:
            inp = torch.from_numpy(volume).unsqueeze(0).to(DEVICE)  # (1, 66, H, W)
            logits = model(inp)
            prob = torch.sigmoid(logits)
            pred_binary = (prob > THRESHOLD).float().cpu().numpy()[0, 0]  # (H, W)
        else:
            avg_intensity = volume.mean(axis=0).astype(np.float32)  # (H, W)
            scaled = (avg_intensity * 255).astype(np.uint8)
            _, otsu_thr = cv2.threshold(
                scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            otsu_thr = otsu_thr / 255.0
            pred_binary = (avg_intensity > otsu_thr).astype(np.float32)

        if pred_binary.shape != mask_norm.shape:
            pred_binary = cv2.resize(
                pred_binary.astype(np.float32),
                (mask_norm.shape[1], mask_norm.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        pred_binary = pred_binary * mask_norm

        pred_binary = (pred_binary > 0).astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        pred_binary = cv2.morphologyEx(pred_binary, cv2.MORPH_OPEN, kernel)

        rle = rle_encode(pred_binary)
        submission_df.at[idx, "Predicted"] = rle

output_path = os.path.join(os.getcwd(), "submission.csv")
submission_df.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")




## === cell 9
print(submission_df.head())
