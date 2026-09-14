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

0.198602

# 6. Current score

0.10113

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.26337) has done: 'I fix the crashes by making the code discover available train/test fragment IDs from disk instead of hard-coding fragments 1..3 and test fragments a/b (your dataset only has train 1..2 and test a). I also fix a real shape bug in the dataset/model interface: the subvolume slicing was producing 63×63 patches for img_size=64, and the model was instantiated with the wrong subvolume dimension order; both can silently hurt training and predictions. To keep your core approach intact, I won’t change the architecture or training loop semantics, but I make prediction generation robust (no out-of-bounds writes, correct patch size) and ensure a valid `submission.csv` is always produced with the required `Id,Predicted` columns.'
- What this solution (achieved 0.25367) has done: 'I fix the fragment-ID discovery bug that incorrectly includes a nested `test/` directory named `"test"`, which causes attempts to open `/test/test/mask.png` and crashes prediction/submission generation. I make `list_fragment_ids()` ignore known non-fragment folders (like `train`/`test`) and require that fragment folders actually contain `mask.png`, so the pipeline is robust to the dataset’s nested structure. These changes are score-neutral (they only prevent invalid IDs and file-not-found errors) and preserve your training/inference logic. Finally, I ensure `submission.csv` is always written with exactly the `Id,Predicted` columns for the discovered valid test fragments.'
- What this solution (achieved 0.00643) has done: 'Your current issue is “Not yielded”, so the smallest score-improving change is to make submission generation reliably finish within the time limit: the current inference loops write one pixel at a time in Python, which can stall/crash before writing `submission.csv`. I keep your model/training identical, but vectorize the inference write-back (no per-pixel Python loop), which preserves the exact sampled-pixel predictions while making the notebook complete end-to-end. I also ensure `test_pixels` is a NumPy array (stable indexing) and add a safety guard so we never index past the last pixel batch. These changes don’t alter evaluation semantics (same stride, same thresholding, same RLE), but they make producing a valid submission much more likely and typically improve score vs “no submission”.'
- What this solution (achieved 0.01352) has done: 'Your score is far below the target, so we should nudge performance upward with minimal, metric-aligned changes while keeping your model/training/inference core intact. The biggest low-risk win here is to make inference denser (smaller stride) and to tune the binarization threshold on *both* available training fragments (instead of just the first), because the competition metric is computed on the final binary mask. I keep the same prediction semantics (same model, same patch-wise pixel sampling, same RLE), but compute the best threshold by aggregating F0.5 over two small tuning patches and use a slightly denser test stride to improve mask quality. These are small changes that should move your score closer to 0.1986 without altering architecture or training loop behavior.'
- What this solution (achieved 0.01339) has done: 'I make two minimal, score-relevant adjustments that preserve your model/training loop and patch-wise inference semantics, but improve the final binary mask for the F0.5 metric. First, I tune the binarization threshold directly for F0.5 (precision-weighted) by computing it from tp/fp/fn across the tuning patches, instead of using the current approximate dice routine. Second, I apply the provided `mask.png` as a hard constraint during threshold tuning as well (not just at test-time), so the threshold is calibrated on the same valid-pixel region the metric effectively cares about. These changes should move your score upward toward the 0.1986 target without changing the core architecture or training approach.'
- What this solution (achieved 0.13395) has done: 'Your current score is far below the target, so we should carefully increase performance with the smallest metric-aligned tweaks that don’t change your model or training loop. The main low-risk issue is that with sparse strided inference, most pixels default to 0 which hurts recall/structure; we can fill unsampled pixels by nearest-neighbor propagation on the sampled grid (this preserves your patch-wise semantics while producing a dense binary mask for RLE). Second, we tune the threshold on a slightly denser stride to better match the new inference density, keeping the same F0.5-based calibration. Finally, we keep runtime safe by doing propagation with vectorized NumPy repeats/cropping (no slow per-pixel loops) and still writing a valid `submission.csv`.'
- What this solution (achieved 0.07949) has done: 'We’re currently below the target (0.13395 vs 0.198602), so the smallest safe way to move upward is to better match the F0.5 metric by reducing false positives without changing the model/training loop. I keep your architecture and training exactly the same, but tune the binarization threshold more appropriately by (1) optimizing F0.5 only on valid pixels and (2) constraining predictions to be within the mask during scoring, and (3) adding one tiny post-processing step that is metric-aligned and lightweight: remove isolated single-pixel positives (a simple denoise) to boost precision. This changes only final binarization/post-processing semantics (not model outputs), is fast, and should increase score toward the target band. Submission writing and Id alignment remain unchanged.'
- What this solution (achieved 0.10113) has done: 'I make the smallest changes that unblock “Not yielded” by cutting the biggest risk factors for timeout/OOM during training/inference while keeping your model, loss, and patch-wise prediction semantics identical. Concretely, I (1) avoid repeatedly re-loading the same training patch stack every loop by caching it once, (2) speed up DataLoader transfers with `non_blocking=True` and a couple of safe PyTorch backend flags, and (3) ensure the submission RLE is always valid by sorting fragments and producing an empty string when no positive pixels exist. These changes don’t alter architecture, training objective, or thresholding logic; they only make the pipeline reliably finish and thus yield a score (which should move upward from “no submission” toward your target). All paths and output format remain unchanged and it still write `submission.csv` in the working directory.'

# 9. Code solution

## === cell 0
import os
import time
import random
import numpy as np
import pandas as pd
import PIL
from PIL import Image

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib

device = "cuda:0" if torch.cuda.is_available() else "cpu"
vesuvius_data_path = "/kaggle/input/vesuvius-challenge-ink-detection/"

Image.MAX_IMAGE_PIXELS = None

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def list_fragment_ids(root_dir, must_contain_mask=True):
    """Return subdirectory names under root_dir that look like fragment IDs and contain mask.png."""
    if not os.path.isdir(root_dir):
        return []
    out = []
    for d in sorted(os.listdir(root_dir)):
        if d.startswith("."):
            continue
        if d in {"train", "test"}:
            continue
        p = os.path.join(root_dir, d)
        if not os.path.isdir(p):
            continue
        if must_contain_mask and not os.path.exists(os.path.join(p, "mask.png")):
            continue
        out.append(d)
    return out


train_dir = os.path.join(vesuvius_data_path, "train")
test_dir = os.path.join(vesuvius_data_path, "test")

train_ids = [
    d for d in list_fragment_ids(train_dir, must_contain_mask=True) if d.isdigit()
]
test_ids = list_fragment_ids(test_dir, must_contain_mask=True)

print("Discovered train fragments:", train_ids)
print("Discovered test fragments:", test_ids)

test_patches = {
    1: [2000, 400, 2500, 1000],
    2: [1500, 1200, 2200, 1000],
    3: [1800, 500, 2300, 1200],
}

train_patches = [
    [1, [200, 1500, 4500, 6500]],
]


def load_mask_label(fragment, rect=None, display=False):
    """Loads and returns mask and label for a given training fragment (numeric id)."""
    fragment = str(fragment)
    mask_filepath = os.path.join(vesuvius_data_path, f"train/{fragment}/mask.png")
    label_filepath = os.path.join(vesuvius_data_path, f"train/{fragment}/inklabels.png")
    mask = torch.from_numpy(np.array(PIL.Image.open(mask_filepath).convert("1")))
    label = torch.from_numpy(np.array(PIL.Image.open(label_filepath))).float()
    if rect is not None:
        x, y, w, h = rect
        mask = mask[y : y + h, x : x + w]
        label = label[y : y + h, x : x + w]
    if display:
        plt.figure(figsize=(5, 5))
        plt.imshow(label, cmap="gray")
        plt.imshow(mask, cmap="gray", alpha=0.5)
        plt.axis("off")
        plt.show()
    return mask, label


def load_image_stack(
    fragment,
    Z_START,
    Z_DIM,
    rect=None,
    folder="train",
    display_all=False,
    display_one=False,
):
    root_filepath = os.path.join(
        vesuvius_data_path, f"{folder}/{fragment}/surface_volume/"
    )
    tif_files = sorted(
        [x for x in os.listdir(root_filepath) if x.lower().endswith(".tif")]
    )
    layers_to_use = [
        os.path.join(root_filepath, x) for x in tif_files[Z_START : Z_START + Z_DIM]
    ]

    image_stack = []
    for filepath in layers_to_use:
        loaded_img = np.array(PIL.Image.open(filepath), dtype=np.float32) / 65535.0
        if rect is None:
            image_stack.append(loaded_img)
        else:
            x, y, w, h = rect
            image_stack.append(loaded_img[y : y + h, x : x + w])

    if display_one and len(image_stack) > 0:
        plt.imshow(image_stack[0], cmap="gray")
        plt.axis("off")
        plt.show()

    if display_all:
        plt.figure(figsize=(20, 10))
        for j in range(min(len(image_stack), 10)):
            plt.subplot(2, 5, j + 1)
            plt.imshow(image_stack[j], cmap="gray")
            plt.axis("off")
        plt.show()

    image_stack = torch.stack(
        [torch.from_numpy(image) for image in image_stack], dim=0
    )  # (Z, H, W)
    return image_stack


def get_pixels(mask, img_size, stride=0):
    """Return pixel coords (y,x) inside mask and away from borders so a img_size patch fits."""
    radius = int(img_size // 2)
    not_border = np.zeros(mask.shape, dtype=bool)
    not_border[radius : mask.shape[0] - radius, radius : mask.shape[1] - radius] = True
    arr_mask = np.array(mask).astype(bool) & not_border

    if stride != 0:
        assert stride % 2 == 1, "stride has to be an odd number!"
        sparse_mask = np.zeros(mask.shape, dtype=bool)
        sparse_mask[::stride, ::stride] = True
        return np.argwhere(sparse_mask & arr_mask)

    return np.argwhere(arr_mask)


class SubvolumeDataset(torch.utils.data.Dataset):
    def __init__(self, image_stack, label, pixels, img_size):
        self.image_stack = image_stack  # (Z, H, W)
        self.label = label  # (H, W) for labels or mask (unused in test)
        self.pixels = pixels
        self.radius = int(img_size // 2)
        self.img_size = img_size

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        y, x = self.pixels[index]
        subvolume = self.image_stack[
            :, y - self.radius : y + self.radius, x - self.radius : x + self.radius
        ]
        if subvolume.shape[1] != self.img_size or subvolume.shape[2] != self.img_size:
            subvolume = self.image_stack[
                :,
                y - self.radius : y + self.radius + 1,
                x - self.radius : x + self.radius + 1,
            ]
        ink_label = (
            self.label[y, x].float()
            if torch.is_tensor(self.label)
            else torch.tensor(0.0)
        )
        return subvolume, ink_label


def create_dataloader(fragment, patch, Z_START, Z_DIM, img_size, batch_size):
    mask, label = load_mask_label(fragment, rect=patch)
    image_stack = load_image_stack(fragment, Z_START, Z_DIM, rect=patch, folder="train")
    pixels = get_pixels(mask, img_size)
    dataset = SubvolumeDataset(image_stack, label, pixels, img_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.startswith("cuda")),
    )
    return dataloader


frag_nums = [int(x) for x in train_ids]
if len(frag_nums) > 0:
    fig, ax = plt.subplots(1, len(frag_nums), figsize=(5 * len(frag_nums), 5))
    if len(frag_nums) == 1:
        ax = [ax]
    for idx, fragment in enumerate(frag_nums):
        mask, label = load_mask_label(fragment)
        ax[idx].imshow(label, cmap="gray")
        ax[idx].imshow(mask, cmap="gray", alpha=0.5)
        if fragment in test_patches:
            x, y, w, h = test_patches[fragment]
            test_patch_rect = matplotlib.patches.Rectangle(
                (x, y), w, h, linewidth=2, edgecolor="r", facecolor="none"
            )
            ax[idx].add_patch(test_patch_rect)
        for tp in train_patches:
            if tp[0] == fragment:
                x, y, w, h = tp[1]
                train_patch_rect = matplotlib.patches.Rectangle(
                    (x, y), w, h, linewidth=2, edgecolor="b", facecolor="none"
                )
                ax[idx].add_patch(train_patch_rect)
        ax[idx].set_title(f"Train fragment {fragment}")
        ax[idx].axis("off")
    plt.show()
else:
    print("No train fragments found to visualize.")




## === cell 1
class Subvolume3DcnnEncoder(nn.Module):
    def __init__(self, batch_norm_momentum, filters):
        super().__init__()
        strides = [1, 2, 2, 2]
        filter_sizes = [1] + filters
        filter_list_pairs = list(zip(filter_sizes[:-1], filter_sizes[1:]))
        self.conv_layers = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv3d(
                        chan_in, chan_out, kernel_size=3, stride=stride, padding=1
                    ),
                    nn.ReLU(),
                )
                for (chan_in, chan_out), stride, _filter in zip(
                    filter_list_pairs, strides, filters
                )
            ]
        )
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv3d):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.conv_layers(x)


class LinearInkDecoder(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.fc = nn.Linear(int(np.prod(input_shape)), 1)
        self.flatten = nn.Flatten()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc(self.flatten(x)))


class InkClassifier3DCNN(nn.Module):
    def __init__(
        self, subvolume_shape, batch_norm_momentum=0.001, filters=[16, 32, 64, 128]
    ):
        super().__init__()
        self.encoder = Subvolume3DcnnEncoder(batch_norm_momentum, filters)
        with torch.no_grad():
            dummy = torch.zeros((1, 1, *subvolume_shape))
            enc_shape = self.encoder(dummy).shape[1:]
        self.decoder = LinearInkDecoder(enc_shape)

    def forward(self, x):
        return self.decoder(self.encoder(x))




## === cell 2
img_size = 64
Z_START = 27
Z_DIM = 10

batch_size = 32
lr = 3e-3

model = InkClassifier3DCNN(subvolume_shape=[Z_DIM, img_size, img_size]).to(device)

xb = torch.rand((5, 1, Z_DIM, img_size, img_size)).to(device)
print("forward pass:", model(xb).shape)
print(
    f"Number of params: {(sum(p.numel() for p in model.parameters() if p.requires_grad)):,}"
)

eval_print_interval = 1000
total_training_steps = 15000
subvolume_training_steps = 100000

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr, total_steps=total_training_steps
)

training_step = 0
running_loss = []
model.train()

train_patches_existing = []
for frag, patch in train_patches:
    if str(frag) in train_ids:
        train_patches_existing.append([frag, patch])
if len(train_patches_existing) == 0:
    raise FileNotFoundError(
        "No valid train patches found for discovered train fragments."
    )

_cached_train_loaders = {}
for fragment, patch in train_patches_existing:
    key = (int(fragment), tuple(patch))
    if key not in _cached_train_loaders:
        t_load = time.time()
        dl = create_dataloader(fragment, patch, Z_START, Z_DIM, img_size, batch_size)
        _cached_train_loaders[key] = dl
        print(
            f"Cached dataloader for fragment {fragment}, patch {patch} in {time.time()-t_load:.2f}s"
        )

while training_step < total_training_steps:
    print("-" * 20, "New dataloader section", "-" * 20)
    random.shuffle(train_patches_existing)
    for fragment, patch in train_patches_existing:
        if training_step >= total_training_steps:
            break

        dataloader = _cached_train_loaders[(int(fragment), tuple(patch))]
        t_dataloader = time.time()
        t_interval = time.time()
        dataloader_steps = 0

        for i, (subvolumes, ink_labels) in enumerate(dataloader):
            if (
                dataloader_steps >= subvolume_training_steps
                or training_step >= total_training_steps
            ):
                break

            subvolumes = subvolumes.to(device, non_blocking=True)
            ink_labels = ink_labels.to(device, non_blocking=True)

            logits = model(subvolumes.unsqueeze(1))
            loss = loss_fn(logits, ink_labels.unsqueeze(dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss.append(loss.item())
            training_step += 1
            dataloader_steps += 1

            if i % eval_print_interval == eval_print_interval - 1:
                print(
                    f"i: {i} | Running loss: {np.array(running_loss).mean():.4f} | "
                    f"lr: {scheduler.get_last_lr()[0]:.5f} | "
                    f"t_interval {time.time()-t_interval:.2f} t_dataloader: {time.time() - t_dataloader:.2f}s"
                )
                t_interval = time.time()
                running_loss = []




## === cell 3
def generate_predictions(fragment="a", stride=99, patch=None):
    """Return sparse prediction map for a fragment (test: string id, train: int id with patch).

    Note: This preserves your original inference semantics (predict only on a strided grid and
    write those positions into the output map).
    """
    t_load = time.time()

    if isinstance(fragment, str) and fragment in test_ids:
        mask_filepath = os.path.join(vesuvius_data_path, f"test/{fragment}/mask.png")
        mask = torch.from_numpy(np.array(PIL.Image.open(mask_filepath).convert("1")))
        image_stack = load_image_stack(fragment, Z_START, Z_DIM, folder="test")
        out_shape = mask.shape
    else:
        mask, label = load_mask_label(fragment, rect=patch)
        image_stack = load_image_stack(
            fragment, Z_START, Z_DIM, rect=patch, folder="train"
        )
        out_shape = mask.shape

    test_pixels = get_pixels(mask, img_size, stride=stride)
    test_pixels = np.asarray(test_pixels, dtype=np.int64)

    print(
        f"Mask size {mask.shape}, Striding by {stride}, we have {len(test_pixels)} pixels to test | Time to load: {time.time() - t_load:.2f}s"
    )

    test_dataset = SubvolumeDataset(
        image_stack,
        mask.float() if torch.is_tensor(mask) else torch.from_numpy(mask).float(),
        test_pixels,
        img_size,
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.startswith("cuda")),
    )
    print(
        f"Length of test dataloader: {len(test_dataloader)} batches of size {batch_size}"
    )

    t_generate = time.time()
    output = torch.zeros(out_shape, dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        n = len(test_pixels)
        for i, (subvolumes, _) in enumerate(test_dataloader):
            preds = (
                model(subvolumes.to(device, non_blocking=True).unsqueeze(1))
                .detach()
                .cpu()
                .view(-1)
            )
            base = i * batch_size
            end = min(base + preds.numel(), n)
            if end <= base:
                continue

            coords = test_pixels[base:end]  # (m,2) -> (y,x)
            ys = coords[:, 0]
            xs = coords[:, 1]
            output[ys, xs] = preds[: (end - base)]

    print(f"Generated pixels!! Time taken: {time.time() - t_generate:.2f}s")
    return output


def densify_sparse_grid(pred_map, stride):
    """Convert sparse grid predictions to a dense map by nearest-neighbor propagation."""
    if stride is None or stride <= 1:
        return np.asarray(pred_map, dtype=np.float32)

    pm = np.asarray(pred_map, dtype=np.float32)
    H, W = pm.shape

    grid = pm[::stride, ::stride]
    dense = np.repeat(np.repeat(grid, stride, axis=0), stride, axis=1)
    dense = dense[:H, :W]
    dense[::stride, ::stride] = grid
    return dense


def fbeta_from_counts(tp, fp, fn, beta=0.5, smooth=1e-5):
    beta2 = beta * beta
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    return (1 + beta2) * (precision * recall) / (beta2 * precision + recall + smooth)


def f05_score_binary(
    pred_binary, target_binary, valid_mask=None, beta=0.5, smooth=1e-5
):
    """Compute F0.5 only on valid pixels (mask==True)."""
    p = np.asarray(pred_binary, dtype=np.uint8)
    t = np.asarray(target_binary, dtype=np.uint8)
    if valid_mask is not None:
        vm = np.asarray(valid_mask, dtype=bool)
        p = p[vm]
        t = t[vm]
    p = p.reshape(-1)
    t = t.reshape(-1)
    tp = int(((p == 1) & (t == 1)).sum())
    fp = int(((p == 1) & (t == 0)).sum())
    fn = int(((p == 0) & (t == 1)).sum())
    return float(fbeta_from_counts(tp, fp, fn, beta=beta, smooth=smooth))


def remove_isolated_positives(binary_mask):
    """Score-relevant change: slightly stronger tiny denoise to improve precision for F0.5.

    Minimal semantics change vs your existing step:
    - Previously removed only 1-pixel islands (required >=1 positive 8-neighbor).
    - Now also removes 2-pixel specks by requiring a pixel to have at least TWO positive 8-neighbors.
      This tends to cut false positives (precision) with limited recall loss, helping F0.5.
    """
    b = (np.asarray(binary_mask, dtype=np.uint8) > 0).astype(np.uint8)
    if b.ndim != 2:
        raise ValueError("binary_mask must be 2D")
    H, W = b.shape
    bp = np.pad(b, 1, mode="constant", constant_values=0)
    neigh = (
        bp[0:H, 0:W]
        + bp[0:H, 1 : W + 1]
        + bp[0:H, 2 : W + 2]
        + bp[1 : H + 1, 0:W]
        + bp[1 : H + 1, 2 : W + 2]
        + bp[2 : H + 2, 0:W]
        + bp[2 : H + 2, 1 : W + 1]
        + bp[2 : H + 2, 2 : W + 2]
    )
    out = (b & (neigh >= 2)).astype(np.uint8)
    return out




## === cell 4
best_threshold = torch.tensor(0.4)  # safe default if tuning can't run

tune_items = []
for tid in train_ids:
    fid = int(tid)
    if fid in test_patches:
        tune_items.append((fid, test_patches[fid]))

if len(tune_items) == 0:
    print("No tuning patches available; using default threshold=0.4")
else:
    tune_stride = 11

    tuned_preds = []
    tuned_labels = []
    tuned_masks = []

    for fragment, patch in tune_items:
        print(
            f"Tuning: generating predictions for train fragment {fragment} patch {patch}"
        )
        sparse_pred = generate_predictions(
            fragment=fragment, stride=tune_stride, patch=patch
        )
        pred = densify_sparse_grid(sparse_pred, stride=tune_stride)

        mask, label = load_mask_label(fragment, rect=patch)

        mask_np = np.asarray(mask, dtype=bool)
        pred_np = np.asarray(pred, dtype=np.float32)
        pred_np[~mask_np] = 0.0

        tuned_preds.append(torch.from_numpy(pred_np))
        tuned_labels.append(label)
        tuned_masks.append(mask_np)

    plt.figure(figsize=(6, 6))
    plt.imshow(np.asarray(tuned_preds[0]), cmap="gray")
    plt.axis("off")
    plt.show()



## === cell 5
if "tuned_preds" in globals() and len(tuned_preds) > 0:
    thresholds = torch.linspace(0.20, 0.92, steps=145)

    best_f05 = -1.0
    best_threshold = torch.tensor(0.4)

    max_pos_frac = 0.035
    min_pos_frac = 0.0002

    tuned_labels_np = [np.asarray(lab, dtype=np.uint8) for lab in tuned_labels]
    tuned_masks_np = [np.asarray(msk, dtype=bool) for msk in tuned_masks]
    tuned_preds_np = [np.asarray(pred, dtype=np.float32) for pred in tuned_preds]

    for threshold in thresholds:
        f05s = []
        ok = True
        thr = float(threshold.item())
        for pred_np, lab_np, msk_np in zip(
            tuned_preds_np, tuned_labels_np, tuned_masks_np
        ):
            binary_pred = (pred_np > thr).astype(np.uint8)
            binary_pred[~msk_np] = 0

            binary_pred = remove_isolated_positives(binary_pred)
            binary_pred[~msk_np] = 0

            pos_frac = float(binary_pred.mean())
            if pos_frac > max_pos_frac or pos_frac < min_pos_frac:
                ok = False
                break

            f05s.append(
                f05_score_binary(binary_pred, lab_np, valid_mask=msk_np, beta=0.5)
            )

        if not ok:
            continue

        mean_f05 = float(np.mean(f05s)) if len(f05s) else -1.0
        if mean_f05 > best_f05:
            best_f05 = mean_f05
            best_threshold = threshold

    if best_f05 < 0:
        best_f05 = 0.0
        best_threshold = torch.tensor(0.4)
        for threshold in thresholds:
            thr = float(threshold.item())
            f05s = []
            for pred_np, lab_np, msk_np in zip(
                tuned_preds_np, tuned_labels_np, tuned_masks_np
            ):
                binary_pred = (pred_np > thr).astype(np.uint8)
                binary_pred[~msk_np] = 0
                binary_pred = remove_isolated_positives(binary_pred)
                binary_pred[~msk_np] = 0
                f05s.append(
                    f05_score_binary(binary_pred, lab_np, valid_mask=msk_np, beta=0.5)
                )
            mean_f05 = float(np.mean(f05s)) if len(f05s) else 0.0
            if mean_f05 > best_f05:
                best_f05 = mean_f05
                best_threshold = threshold

    print(
        f"Selected threshold: {best_threshold.item():.5f} | Mean tuning-patch F0.5: {best_f05:.6f} | tuned on {len(tuned_preds)} patch(es)"
    )

    binary_pred = (tuned_preds_np[0] > float(best_threshold.item())).astype(np.uint8)
    binary_pred[~tuned_masks_np[0]] = 0
    binary_pred = remove_isolated_positives(binary_pred)
    binary_pred[~tuned_masks_np[0]] = 0

    plt.figure(figsize=(6, 6))
    plt.imshow(binary_pred, cmap="gray")
    plt.imshow(tuned_labels_np[0], cmap="gray", alpha=0.5)
    plt.axis("off")
    plt.show()
else:
    print(
        "Skipping threshold tuning because tuned_preds/tuned_labels were not generated."
    )



## === cell 6
if "binary_pred" in globals():
    plt.figure(figsize=(6, 6))
    plt.imshow(binary_pred, cmap="gray")
    plt.axis("off")
    plt.show()



## === cell 7
for tid in test_ids:
    mask_filepath = os.path.join(vesuvius_data_path, f"test/{tid}/mask.png")
    mask = torch.from_numpy(np.array(PIL.Image.open(mask_filepath).convert("1")))
    plt.figure(figsize=(5, 5))
    plt.imshow(mask, cmap="gray")
    plt.title(f"Test mask: {tid}")
    plt.axis("off")
    plt.show()




## === cell 8
def rle_from_binary_mask(binary_mask):
    """Run-length encode a 2D binary mask using 1-indexed pixel positions."""
    pixels = np.asarray(binary_mask, dtype=np.uint8).reshape(-1)
    pixels = np.concatenate([[0], pixels, [0]])
    changes = np.where(pixels[1:] != pixels[:-1])[0] + 1  # 1..N+1 positions
    starts = changes[0::2]
    ends = changes[1::2]
    lengths = ends - starts
    return (starts).astype(np.int64), lengths.astype(np.int64)


pred_list = []
if len(test_ids) == 0:
    raise FileNotFoundError("No test fragments found; cannot create submission.")

TEST_STRIDE = 11  # keep as before (odd); densification makes this effectively dense

for fragment in sorted(test_ids, key=lambda x: str(x)):
    sparse_pred_map = generate_predictions(fragment=fragment, stride=TEST_STRIDE)
    pred_map_np = densify_sparse_grid(sparse_pred_map, stride=TEST_STRIDE)

    mask_filepath = os.path.join(vesuvius_data_path, f"test/{fragment}/mask.png")
    mask = np.array(PIL.Image.open(mask_filepath).convert("1"), dtype=bool)
    pred_map_np[~mask] = 0.0

    binary = (pred_map_np > float(best_threshold)).astype(np.uint8)
    binary[~mask] = 0
    binary = remove_isolated_positives(binary)
    binary[~mask] = 0

    starts_ix, lengths = rle_from_binary_mask(binary)

    if starts_ix.size == 0:
        inklabels_rle = ""
    else:
        inklabels_rle = " ".join(
            map(str, np.column_stack([starts_ix, lengths]).ravel().tolist())
        )

    pred_list.append({"Id": str(fragment), "Predicted": inklabels_rle})

print("Prepared predictions for test Ids:", [x["Id"] for x in pred_list])



## === cell 9
pd.DataFrame(pred_list, columns=["Id", "Predicted"]).to_csv(
    "submission.csv", index=False
)
print("Wrote submission.csv with shape:", pd.read_csv("submission.csv").shape)
print(pd.read_csv("submission.csv").head())



## === cell 10
print("Final cell: submission.csv is ready.")
