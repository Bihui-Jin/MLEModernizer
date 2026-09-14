# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.038431

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, glob, torch, torch.nn as nn, torch.optim as optim, numpy as np
import PIL.Image as Image
import torch.utils.data as data
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm

possible_prefixes = [
    "/kaggle/input/vesuvius-challenge-ink-detection/train/1/",
    "/kaggle/input/vesuvius-challenge/train/1/",
    "/kaggle/input/vesuvius-challenge-ink-detection/train/1/",
    "/kaggle/input/vesuvius-challenge/train/1/",
]
PREFIX = None
for p in possible_prefixes:
    if os.path.isdir(p):
        PREFIX = p
        break
if PREFIX is None:
    raise FileNotFoundError("Could not locate training data prefix.")
BUFFER = 30  # buffer size in x and y direction
Z_START = 27  # first slice to use
Z_DIM = 10  # number of slices
TRAINING_STEPS = 20000
LEARNING_RATE = 0.05
BATCH_SIZE = 24
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

plt.imshow(Image.open(os.path.join(PREFIX, "ir.png")), cmap="gray")
plt.title("IR image")
plt.axis("off")
plt.show()




## === cell 1
mask = np.array(Image.open(os.path.join(PREFIX, "mask.png")).convert("1"))
label = (
    torch.from_numpy(np.array(Image.open(os.path.join(PREFIX, "inklabels.png"))))
    .gt(0)
    .float()
    .to(DEVICE)
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.set_title("mask.png")
ax1.imshow(mask, cmap="gray")
ax2.set_title("inklabels.png")
ax2.imshow(label.cpu(), cmap="gray")
plt.show()




## === cell 2
tif_paths = sorted(glob.glob(os.path.join(PREFIX, "surface_volume", "*.tif")))[
    Z_START : Z_START + Z_DIM
]
images = [
    np.array(Image.open(fp), dtype=np.float32) / 65535.0
    for fp in tqdm(tif_paths, desc="Loading slices")
]
image_stack = torch.stack([torch.from_numpy(img) for img in images], dim=0).to(
    DEVICE
)  # shape: (Z_DIM, H, W)

fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
for img, ax in zip(images, axes):
    small = Image.fromarray((img * 255).astype(np.uint8)).resize(
        (img.shape[1] // 20, img.shape[0] // 20)
    )
    ax.imshow(small, cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.show()




## === cell 3
rect = (1100, 3500, 700, 950)  # (x, y, width, height)

fig, ax = plt.subplots()
ax.imshow(label.cpu())
patch = patches.Rectangle(
    (rect[0], rect[1]), rect[2], rect[3], linewidth=2, edgecolor="r", facecolor="none"
)
ax.add_patch(patch)
plt.title("Label with ROI")
plt.show()




## === cell 4
class SubvolumeDataset(data.Dataset):
    """Returns a sub‑volume centred on (y, x) and the corresponding ink label."""

    def __init__(self, image_stack, label, pixels):
        self.image_stack = image_stack  # (Z, H, W)
        self.label = label  # (H, W)
        self.pixels = pixels.astype(np.int64)  # N x 2 array of (y, x)

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, idx):
        y, x = self.pixels[idx]
        subvol = self.image_stack[
            :, y - BUFFER : y + BUFFER + 1, x - BUFFER : x + BUFFER + 1
        ]  # (Z, H', W')
        subvol = subvol.unsqueeze(0)  # (1, Z, H', W')
        ink = self.label[y, x].unsqueeze(0)  # (1,)
        return subvol, ink


model = nn.Sequential(
    nn.Conv3d(1, 16, kernel_size=3, padding=1),
    nn.MaxPool3d(2),
    nn.Conv3d(16, 32, kernel_size=3, padding=1),
    nn.MaxPool3d(2),
    nn.Conv3d(32, 64, kernel_size=3, padding=1),
    nn.MaxPool3d(2),
    nn.Flatten(start_dim=1),
    nn.LazyLinear(128),
    nn.ReLU(),
    nn.LazyLinear(1),
    nn.Sigmoid(),
).to(DEVICE)

print("Generating pixel lists...")
not_border = np.zeros(mask.shape, dtype=bool)
not_border[BUFFER : mask.shape[0] - BUFFER, BUFFER : mask.shape[1] - BUFFER] = True
arr_mask = (mask > 0) & not_border

inside_rect = np.zeros(mask.shape, dtype=bool)
inside_rect[rect[1] : rect[1] + rect[3] + 1, rect[0] : rect[0] + rect[2] + 1] = True
inside_rect &= arr_mask

outside_rect = arr_mask & (~inside_rect)

pixels_inside = np.argwhere(inside_rect)
pixels_outside = np.argwhere(outside_rect)

train_dataset = SubvolumeDataset(image_stack, label, pixels_outside)
train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

criterion = nn.BCELoss()
optimizer = optim.ASGD(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.5, total_iters=4)

model.train()
for step, (subvolumes, inklabels) in enumerate(
    tqdm(train_loader, total=TRAINING_STEPS)
):
    if step >= TRAINING_STEPS:
        break
    optimizer.zero_grad()
    outputs = model(subvolumes.to(DEVICE)).squeeze(1)  # (B,)
    loss = criterion(outputs.unsqueeze(1), inklabels.to(DEVICE))
    loss.backward()
    optimizer.step()
    scheduler.step()

model.eval()
all_preds = []
all_true = []
with torch.no_grad():
    for subvolumes, inklabels in tqdm(train_loader, desc="Calibrating on train"):
        preds = model(subvolumes.to(DEVICE)).squeeze(1).cpu().numpy()
        all_preds.append(preds)
        all_true.append(inklabels.numpy())
all_preds = np.concatenate(all_preds)
all_true = np.concatenate(all_true)


def f05_score(y_pred, y_true, beta=0.5):
    y_pred = y_pred.astype(bool)
    y_true = y_true.astype(bool)
    tp = np.logical_and(y_pred, y_true).sum()
    fp = np.logical_and(y_pred, np.logical_not(y_true)).sum()
    fn = np.logical_and(np.logical_not(y_pred), y_true).sum()
    if tp + fp == 0 or tp + fn == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta2 = beta**2
    return (1 + beta2) * precision * recall / (beta2 * precision + recall)


best_thr = 0.5
best_f05 = -1.0
for thr in np.arange(0.1, 0.9, 0.05):
    bin_pred = (all_preds > thr).astype(np.uint8)
    score = f05_score(bin_pred, all_true)
    if score > best_f05:
        best_f05 = score
        best_thr = thr
THRESHOLD = best_thr
print(f"Calibrated THRESHOLD = {THRESHOLD:.3f} with F0.5 ≈ {best_f05:.4f}")


def rle_from_numpy(arr):
    """Convert a binary (uint8) numpy array to RLE string."""
    flat = arr.flatten()
    flat[0] = 0
    flat[-1] = 0
    runs = np.where(flat[1:] != flat[:-1])[0] + 2  # 1‑based indexing & shift
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return " ".join(str(x) for x in runs)


test_root_candidates = [
    os.path.abspath(os.path.join(PREFIX, "..", "..", "test")),
    os.path.abspath(os.path.join(PREFIX, "..", "..", "..", "test")),
]
TEST_ROOT = None
for cand in test_root_candidates:
    if os.path.isdir(cand):
        TEST_ROOT = cand
        break
if TEST_ROOT is None:
    raise FileNotFoundError("Test directory not found.")

test_ids = sorted(
    [
        os.path.basename(d)
        for d in glob.glob(os.path.join(TEST_ROOT, "*"))
        if os.path.isdir(d)
    ]
)

submission_path = "submission.csv"
with open(submission_path, "w") as f:
    f.write("Id,Predicted\n")
    for tid in test_ids:
        mask_path = os.path.join(TEST_ROOT, tid, "mask.png")
        test_mask = np.array(Image.open(mask_path).convert("1"))
        not_border_test = np.zeros(test_mask.shape, dtype=bool)
        not_border_test[
            BUFFER : test_mask.shape[0] - BUFFER, BUFFER : test_mask.shape[1] - BUFFER
        ] = True
        pixels_test = np.argwhere((test_mask > 0) & not_border_test)

        tif_paths = sorted(
            glob.glob(os.path.join(TEST_ROOT, tid, "surface_volume", "*.tif"))
        )[Z_START : Z_START + Z_DIM]
        if not tif_paths:
            f.write(f"{tid},\n")
            continue
        imgs = [
            np.array(Image.open(fp), dtype=np.float32) / 65535.0 for fp in tif_paths
        ]
        test_stack = torch.stack([torch.from_numpy(img) for img in imgs], dim=0).to(
            DEVICE
        )

        class TestDataset(data.Dataset):
            def __init__(self, stack, pixels):
                self.stack = stack
                self.pixels = pixels.astype(np.int64)

            def __len__(self):
                return len(self.pixels)

            def __getitem__(self, idx):
                y, x = self.pixels[idx]
                subvol = self.stack[
                    :, y - BUFFER : y + BUFFER + 1, x - BUFFER : x + BUFFER + 1
                ]
                return subvol.unsqueeze(0)  # (1, Z, H', W')

        test_dataset = TestDataset(test_stack, pixels_test)
        test_loader = data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False
        )

        preds = np.zeros(test_mask.shape, dtype=np.float32)
        model.eval()
        with torch.no_grad():
            for i, subvol_batch in enumerate(
                tqdm(test_loader, desc=f"Infer {tid}", leave=False)
            ):
                batch_preds = model(subvol_batch.to(DEVICE)).squeeze(1).cpu().numpy()
                batch_indices = pixels_test[
                    i * BATCH_SIZE : i * BATCH_SIZE + len(batch_preds)
                ]
                preds[batch_indices[:, 0], batch_indices[:, 1]] = batch_preds

        bin_mask = (preds > THRESHOLD).astype(np.uint8)
        rle = rle_from_numpy(bin_mask)
        f.write(f"{tid},{rle}\n")

print(f"Submission written to {submission_path}")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2747412307.py in <cell line: 0>()
     78         preds = model(subvolumes.to(DEVICE)).squeeze(1).cpu().numpy()
     79         all_preds.append(preds)
---> 80         all_true.append(inklabels.numpy())
     81 all_preds = np.concatenate(all_preds)
     82 all_true = np.concatenate(all_true)

TypeError: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
