# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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
ipywidgets==8.1.5
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

# 5. Code solution

## === cell 0
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

BASE_INPUT = "/kaggle/input/vesuvius-challenge-ink-detection"
TRAIN_ROOT = os.path.join(BASE_INPUT, "train")
TEST_ROOT = os.path.join(BASE_INPUT, "test")
TRAIN_FRAG = "1"  # use fragment 1 for training
PREFIX = os.path.join(TRAIN_ROOT, TRAIN_FRAG) + "/"

BUFFER = 30  # spatial buffer around each pixel
Z_START = 27  # first slice in Z to use
Z_DIM = 10  # number of slices
TRAINING_STEPS = 20000
LEARNING_RATE = 0.05
BATCH_SIZE = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
mask = np.array(Image.open(os.path.join(PREFIX, "mask.png")).convert("1"))
label_np = np.array(Image.open(os.path.join(PREFIX, "inklabels.png")))
label = torch.from_numpy(label_np).gt(0).float().to(DEVICE)




## === cell 2
tif_paths = sorted(glob.glob(os.path.join(PREFIX, "surface_volume", "*.tif")))
selected = tif_paths[Z_START : Z_START + Z_DIM]

images = [
    np.array(Image.open(p), dtype=np.float32) / 65535.0
    for p in tqdm(selected, desc="Loading slices")
]

image_stack = torch.stack([torch.from_numpy(img) for img in images], dim=0).to(DEVICE)




## === cell 3
class SubvolumeDataset(data.Dataset):
    """Dataset returning a (1, Z, H, W) sub‑volume and the binary label at its centre."""

    def __init__(self, image_stack, label, pixels):
        self.image_stack = image_stack  # (Z, H, W)
        self.label = label  # (H, W) torch tensor
        self.pixels = pixels  # Nx2 array of (y, x)

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        y, x = self.pixels[idx]
        subvol = self.image_stack[
            :, y - BUFFER : y + BUFFER + 1, x - BUFFER : x + BUFFER + 1
        ].unsqueeze(
            0
        )  # shape (1, Z, 2*BUFFER+1, 2*BUFFER+1)
        ink = self.label[y, x].unsqueeze(0)  # shape (1,)
        return subvol, ink


model = nn.Sequential(
    nn.Conv3d(1, 16, 3, 1, 1),
    nn.MaxPool3d(2, 2),
    nn.Conv3d(16, 32, 3, 1, 1),
    nn.MaxPool3d(2, 2),
    nn.Conv3d(32, 64, 3, 1, 1),
    nn.MaxPool3d(2, 2),
    nn.Flatten(start_dim=1),
    nn.LazyLinear(128),
    nn.ReLU(),
    nn.LazyLinear(1),
    nn.Sigmoid(),
).to(DEVICE)



## === cell 4
rect = (1100, 3500, 700, 950)  # (x, y, w, h) as in the original notebook
h, w = mask.shape
not_border = np.zeros_like(mask, dtype=bool)
not_border[BUFFER : h - BUFFER, BUFFER : w - BUFFER] = True
valid_area = mask.astype(bool) & not_border

inside_rect = np.zeros_like(mask, dtype=bool)
inside_rect[rect[1] : rect[1] + rect[3] + 1, rect[0] : rect[0] + rect[2] + 1] = True
inside_rect &= valid_area

outside_rect = valid_area & (~inside_rect)

pixels_inside = np.argwhere(inside_rect)
pixels_outside = np.argwhere(outside_rect)

train_dataset = SubvolumeDataset(image_stack, label, pixels_outside)
train_loader = data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True
)

criterion = nn.BCELoss()
optimizer = optim.ASGD(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5, total_iters=4)

model.train()
for step, (subvolumes, inklabels) in enumerate(
    tqdm(train_loader, total=TRAINING_STEPS, desc="Training")
):
    if step >= TRAINING_STEPS:
        break
    optimizer.zero_grad()
    outputs = model(subvolumes.to(DEVICE))
    loss = criterion(outputs, inklabels.to(DEVICE))
    loss.backward()
    optimizer.step()
    scheduler.step()




## === cell 5
def predict_fragment(fragment_id):
    """Return a binary mask (torch tensor) for the given test fragment."""
    test_prefix = os.path.join(TEST_ROOT, fragment_id) + "/"
    test_mask = np.array(Image.open(os.path.join(test_prefix, "mask.png")).convert("1"))
    tif_paths = sorted(glob.glob(os.path.join(test_prefix, "surface_volume", "*.tif")))
    sel = tif_paths[Z_START : Z_START + Z_DIM]
    imgs = [np.array(Image.open(p), dtype=np.float32) / 65535.0 for p in sel]
    vol = torch.stack([torch.from_numpy(im) for im in imgs], dim=0).to(DEVICE)

    h_t, w_t = test_mask.shape
    nb = np.zeros_like(test_mask, dtype=bool)
    nb[BUFFER : h_t - BUFFER, BUFFER : w_t - BUFFER] = True
    pixels = np.argwhere(test_mask.astype(bool) & nb)

    dataset = SubvolumeDataset(vol, torch.zeros((h_t, w_t), device=DEVICE), pixels)
    loader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    pred = torch.zeros((h_t, w_t), device=DEVICE)
    model.eval()
    with torch.no_grad():
        for i, (subvols, _) in enumerate(
            tqdm(loader, desc=f"Inferencing {fragment_id}")
        ):
            outs = model(subvols.to(DEVICE)).squeeze(1)  # (B,)
            start = i * BATCH_SIZE
            end = start + outs.shape[0]
            coords = pixels[start:end]
            pred[coords[:, 0], coords[:, 1]] = outs
    return pred.cpu()


test_ids = [
    name
    for name in os.listdir(TEST_ROOT)
    if os.path.isdir(os.path.join(TEST_ROOT, name))
]

submission_rows = []
THRESHOLD = 0.4


def rle_encode(mask_tensor):
    """Encode a binary mask (torch tensor) to run‑length string."""
    arr = (mask_tensor.numpy() > THRESHOLD).astype(np.uint8).flatten()
    arr[0] = 0
    arr[-1] = 0
    runs = np.where(arr[1:] != arr[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return " ".join(str(x) for x in runs)


for frag_id in test_ids:
    pred_mask = predict_fragment(frag_id)
    rle_str = rle_encode(pred_mask)
    submission_rows.append({"Id": frag_id, "Predicted": rle_str})

submission_path = "submission.csv"
pd.DataFrame(submission_rows).to_csv(submission_path, index=False)
print(f"Submission written to {submission_path} ({len(submission_rows)} rows)")
