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

No external packages required in the script and installed.

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

0.1964498661486244

# 6. Current score

0.13173

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05259) has done: 'Implemented a robust handling of training masks by resizing all masks to a common shape before averaging, eliminating the heterogeneous‑shape error that prevented submission generation. The rest of the pipeline (RLE encoding, test mask processing, and CSV output) remains unchanged, ensuring a valid `submission.csv` is written and the model’s baseline score moves toward the target.'
- What this solution (achieved 0.15907) has done: 'I lower the probability threshold used to turn the averaged prior ink mask into a binary prediction, which should increase recall while keeping reasonable precision and move the F0.5 score closer to the target. The change is confined to the mask‑handling logic and does not alter the overall pipeline or model architecture.'
- What this solution (achieved 0.15907) has done: 'I slightly lower the probability threshold used to binarize the averaged prior mask (from 0.20 to 0.15). This modest change should increase recall without overly harming precision, moving the F0.5 score upward toward the target while keeping the core pipeline unchanged.'
- What this solution (achieved 0.15907) has done: 'I lower the probability threshold from 0.15 to 0.10, which should increase recall while keeping precision acceptable, moving the F0.5 score closer to the target. This is the only change needed and the rest of the pipeline remains untouched.'
- What this solution (achieved 0.13173) has done: 'Implemented a modest threshold increase (0.20) to favor precision, which aligns better with the F0.5 metric, and replaced the naive `np.resize` with proper bilinear resizing via Pillow. This yields higher‑quality prior mask alignment for both training aggregation and test inference while keeping the core modeling pipeline unchanged.'

# 9. Code solution

## === cell 0
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T2

try:
    import segmentation_models_pytorch as smp
except ModuleNotFoundError:

    class SimpleUnet(nn.Module):
        def __init__(self, in_channels=5, out_channels=1):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
                nn.Conv3d(16, 32, kernel_size=3, padding=1),
                nn.BatchNorm3d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool3d(2),
            )
            self.decoder = nn.Sequential(
                nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
                nn.BatchNorm3d(16),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(16, out_channels, kernel_size=2, stride=2),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    class DummySMP:
        @staticmethod
        def Unet(encoder_name, encoder_weights, in_channels, classes):
            return SimpleUnet(in_channels=in_channels, out_channels=classes)

    smp = DummySMP()

import glob
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import pandas as pd
import os


class Config:
    BASE_PATH = "/kaggle/input"
    TRAIN = False
    TILE_DEPTH = 5


config = Config()




## === cell 1
class CustomModel1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = smp.Unet(
            encoder_name="resnext50_32x4d",
            encoder_weights="imagenet",
            in_channels=config.TILE_DEPTH,
            classes=1,
        )

    def forward(self, X):
        return self.encoder(X)




## === cell 2
def rle(img):
    """Run‑length encoding for a binary mask."""
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def resize_mask(mask, shape):
    """
    Resize a 2‑D binary mask to ``shape`` (height, width) using bilinear interpolation.
    Returns a float32 array with values in [0, 1].
    """
    img = Image.fromarray((mask * 255).astype(np.uint8))
    resized = img.resize((shape[1], shape[0]), resample=Image.BILINEAR)
    return np.array(resized).astype(np.float32) / 255.0


train_root_candidates = [
    os.path.join(config.BASE_PATH, "vesuvius-challenge", "train"),
    os.path.join(config.BASE_PATH, "vesuvius-challenge-ink-detection", "train"),
]
train_root = None
for cand in train_root_candidates:
    if os.path.isdir(cand):
        train_root = cand
        break
if train_root is None:
    possible = glob.glob(
        os.path.join(config.BASE_PATH, "**", "train", "*", "inklabels.png"),
        recursive=True,
    )
    train_root = os.path.dirname(os.path.dirname(possible[0])) if possible else None

prior_masks = []
if train_root:
    fragment_paths = sorted(glob.glob(os.path.join(train_root, "*")))
    for fp in fragment_paths:
        ink_path = os.path.join(fp, "inklabels.png")
        if os.path.isfile(ink_path):
            ink = np.array(Image.open(ink_path).convert("1"))
            ink = (ink > 0).astype(np.float32)
            prior_masks.append(ink)

if not prior_masks:
    raise RuntimeError("No training ink masks found.")

target_shape = prior_masks[0].shape
resized_masks = [resize_mask(m, target_shape) for m in prior_masks]
prior_mean = np.mean(resized_masks, axis=0)  # float probabilities in [0,1]

PROB_THRESHOLD = 0.20

test_root_candidates = [
    os.path.join(config.BASE_PATH, "vesuvius-challenge", "test"),
    os.path.join(config.BASE_PATH, "vesuvius-challenge-ink-detection", "test"),
]
test_root = None
for cand in test_root_candidates:
    if os.path.isdir(cand):
        test_root = cand
        break
if test_root is None:
    raise RuntimeError("Test directory not found.")

fragment_ids = ["a", "b"]  # known test fragment identifiers
results = []
for fragment_id in fragment_ids:
    test_folder = os.path.join(test_root, fragment_id)
    mask_file = os.path.join(test_folder, "mask.png")
    if not os.path.isfile(mask_file):
        continue
    mask = np.array(Image.open(mask_file).convert("1"))
    mask = (mask > 0).astype(np.int32)

    h, w = mask.shape
    prior_resized = resize_mask(prior_mean, (h, w))

    test_result = ((mask * prior_resized) > PROB_THRESHOLD).astype(np.int32)

    plt.figure()
    plt.imshow(test_result, cmap="gray")
    plt.title(f"Predicted mask for fragment {fragment_id}")

    inklabels_rle = rle(test_result)
    results.append((fragment_id, inklabels_rle))

sub = pd.DataFrame(results, columns=["Id", "Predicted"])
sub.to_csv("/kaggle/working/submission.csv", index=False)
