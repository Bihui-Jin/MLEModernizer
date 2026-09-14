# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.11

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import torch
import torch.nn as nn
import numpy as np

vesuvius_data_path = "/kaggle/input/vesuvius-challenge-ink-detection/"

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import os
import time
import PIL
import random
import tqdm

device = "cuda:0" if torch.cuda.is_available() else "cpu"


def load_mask_label(fragment, rect=None, display=False):
    """Loads and returns mask and label for a given fragment
    Parameters:
      fragment (int in [1, 2, 3]): id of the fragment
      rect (tuple): (x, y, w, h) of the subsection of the image to load
    """
    mask_filepath = vesuvius_data_path + f"train/{fragment}/mask.png"
    label_filepath = vesuvius_data_path + f"train/{fragment}/inklabels.png"
    mask = torch.from_numpy(np.array(PIL.Image.open(mask_filepath).convert("1")))
    label = torch.from_numpy(np.array(PIL.Image.open(label_filepath))).float()
    if rect is not None:
        mask = mask[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]
        label = label[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]
    if display:
        plt.figure(figsize=(5, 5))
        plt.imshow(label, cmap="gray")
        plt.imshow(mask, cmap="gray", alpha=0.5)
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
    root_filepath = vesuvius_data_path + f"{folder}/{fragment}/surface_volume/"
    tif_filepaths = [root_filepath + x for x in sorted(os.listdir(root_filepath))[:-4]]
    layers_to_use = tif_filepaths[Z_START : Z_START + Z_DIM]

    image_stack = []
    for filepath in layers_to_use:
        loaded_img = np.array(PIL.Image.open(filepath), dtype=np.float32) / 65535.0
        if rect is None:
            image_stack.append(loaded_img)
        else:
            image_stack.append(
                loaded_img[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]
            )

    if display_one:
        plt.imshow(
            np.array(PIL.Image.fromarray(image_stack[0]), dtype=np.float32), cmap="gray"
        )
        plt.axis("off")
        plt.show()

    if display_all:
        plt.figure(figsize=(20, 10))
        for j in range(len(image_stack)):
            plt.subplot(2, 5, j + 1)
            plt.imshow(
                np.array(PIL.Image.fromarray(image_stack[j]), dtype=np.float32),
                cmap="gray",
            )
            plt.axis("off")
        plt.show()

    image_stack = torch.stack([torch.from_numpy(image) for image in image_stack], dim=0)
    return image_stack


def get_pixels(mask, img_size, stride=0):
    """Returns pixels inside rectangle and mask
    Parameters:
        mask (np.array): mask of the image
        img_size (int): size of the image
        stride (int): how many pixels to skip (e.g. 0 means all pixels, 3 means a gap of 3 horiontally and vertically between pixels)
    """
    radius = int(img_size // 2)
    not_border = np.zeros(mask.shape, dtype=bool)
    not_border[radius : mask.shape[0] - radius, radius : mask.shape[1] - radius] = True
    arr_mask = np.array(mask) * not_border

    if stride != 0:
        assert stride % 2 == 1, "stride has to be an odd number!"
        sparse_mask = np.zeros(mask.shape, dtype=bool)
        sparse_mask[::stride, ::stride] = True
        return np.argwhere(sparse_mask * arr_mask)

    return np.argwhere(arr_mask)


class SubvolumeDataset(torch.utils.data.Dataset):
    def __init__(self, image_stack, label, pixels, img_size):
        self.image_stack = image_stack
        self.label = label
        self.pixels = pixels
        self.radius = int(img_size // 2)

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        y, x = self.pixels[index]
        subvolume = self.image_stack[
            :,
            y - self.radius : y + self.radius + 1,
            x - self.radius : x + self.radius + 1,
        ]
        ink_label = self.label[y, x]
        return subvolume, ink_label


def create_dataloader(fragment, patch, Z_START, Z_DIM, img_size, batch_size):
    mask, label = load_mask_label(fragment, rect=patch)
    image_stack = load_image_stack(fragment, Z_START, Z_DIM, rect=patch)
    pixels = get_pixels(mask, img_size)
    dataset = SubvolumeDataset(image_stack, label, pixels, img_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return dataloader




## === cell 1

test_patches = {
    1: [2000, 400, 2500, 1000],
    2: [1500, 1200, 2200, 1000],
    3: [1800, 500, 2300, 1200],
}

train_patches = [[1, [200, 1500, 4500, 6500]]]


if not os.path.exists(vesuvius_data_path):
    candidate = "/kaggle/data/vesuvius-challenge-ink-detection/"
    if os.path.exists(candidate):
        vesuvius_data_path = candidate

train_root = os.path.join(vesuvius_data_path, "train")
available_fragments = []
if os.path.isdir(train_root):
    for d in sorted(os.listdir(train_root)):
        if d.isdigit() and os.path.exists(os.path.join(train_root, d, "mask.png")):
            available_fragments.append(int(d))

n = len(available_fragments)
fig, ax = plt.subplots(1, n, figsize=(10, 10))
if n == 1:
    ax = [ax]

for i, fragment in enumerate(available_fragments):
    mask, label = load_mask_label(fragment)
    ax[i].imshow(label, cmap="gray")
    ax[i].imshow(mask, cmap="gray", alpha=0.5)

    if fragment in test_patches:
        test_rect = test_patches[fragment]
        test_patch = matplotlib.patches.Rectangle(
            (test_rect[0], test_rect[1]),
            test_rect[2],
            test_rect[3],
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax[i].add_patch(test_patch)

    for train_patch in train_patches:
        if train_patch[0] == fragment:
            train_rect = train_patch[1]
            train_patch_rect = matplotlib.patches.Rectangle(
                (train_rect[0], train_rect[1]),
                train_rect[2],
                train_rect[3],
                linewidth=2,
                edgecolor="b",
                facecolor="none",
            )
            ax[i].add_patch(train_patch_rect)

plt.figure(figsize=(10, 10))
plt.show()




## === cell 2
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
                for (chan_in, chan_out), stride, filter in zip(
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
        return self.sigmoid(self.fc(self.flatten(x)))  # (B, 1)


class InkClassifier3DCNN(nn.Module):

    def __init__(
        self,
        subvolume_shape=[256, 256, 10],
        batch_norm_momentum=0.001,
        filters=[16, 32, 64, 128],
    ):
        super().__init__()
        self.encoder = Subvolume3DcnnEncoder(batch_norm_momentum, filters)
        self.decoder = LinearInkDecoder(
            self.encoder(torch.zeros((1, 1, *subvolume_shape))).shape[1:]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))




## === cell 3
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

img_size = 64  # 256
Z_START = 27
Z_DIM = 10

batch_size = 32
lr = 3e-3

model = InkClassifier3DCNN(subvolume_shape=[Z_DIM, img_size, img_size]).to(device)

with torch.no_grad():
    enc_out_shape = model.encoder(
        torch.zeros((1, 1, Z_DIM, img_size, img_size), device=device)
    ).shape[1:]
model.decoder = LinearInkDecoder(enc_out_shape).to(device)

xb = torch.rand((5, 1, Z_DIM, img_size, img_size)).to(device)
print("forward pass:", model(xb).shape)
print(
    f"Number of params: {(sum(p.numel() for p in model.parameters() if p.requires_grad)):,}"
)

eval_print_interval = 1000
total_training_steps = 15000  # (really means 5000*batch_size steps)
subvolume_training_steps = 100000

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr, total_steps=total_training_steps
)

training_step = 0
running_loss = []
model.train()

while training_step < total_training_steps:
    print("-" * 20, "New dataloader section", "-" * 20)
    random.shuffle(train_patches)
    for fragment, patch in train_patches:
        if training_step >= total_training_steps:
            print("beaking 1")
            break
        t_load = time.time()
        dataloader = create_dataloader(
            fragment, patch, Z_START, Z_DIM, img_size, batch_size
        )
        print(
            f"## Fragment {fragment}, patch {patch}, Taken {time.time() - t_load:.2f}s to load ##"
        )
        t_dataloader = time.time()
        t_interval = time.time()
        dataloader_steps = 0
        for i, (subvolumes, ink_labels) in enumerate(dataloader):
            if (
                dataloader_steps >= subvolume_training_steps
                or training_step >= total_training_steps
            ):
                print("Breaking 2")
                break

            subvolumes, ink_labels = subvolumes.to(device), ink_labels.to(device)

            logits = model(
                subvolumes.permute(0, 2, 3, 1).permute(0, 3, 1, 2).unsqueeze(dim=1)
            )

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
                    f"i: {i} | Running loss: {np.array(running_loss).mean():.4f} | lr: {scheduler.get_last_lr()[0]:.5f} | t_interval {time.time()-t_interval:.2f} t_dataloader: {time.time() - t_dataloader:.2f}s"
                )
                t_interval = time.time()
                running_loss = []


## --- ERROR in cell 3, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mRuntimeError[0m                              Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1871505545.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     67[0m             [0;31m# Fix: ensure Conv3D input layout is (B, 1, Z, H, W) deterministically.[0m[0;34m[0m[0;34m[0m[0m
[1;32m     68[0m             [0;31m# Dataset returns (B, Z, H, W); permute to (B, 1, Z, H, W) to match model init sizing.[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 69[0;31m             logits = model(
[0m[1;32m     70[0m                 [0msubvolumes[0m[0;34m.[0m[0mpermute[0m[0;34m([0m[0;36m0[0m[0;34m,[0m [0;36m2[0m[0;34m,[0m [0;36m3[0m[0;34m,[0m [0;36m1[0m[0;34m)[0m[0;34m.[0m[0mpermute[0m[0;34m([0m[0;36m0[0m[0;34m,[0m [0;36m3[0m[0;34m,[0m [0;36m1[0m[0;34m,[0m [0;36m2[0m[0;34m)[0m[0;34m.[0m[0munsqueeze[0m[0;34m([0m[0mdim[0m[0;34m=[0m[0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     71[0m             )

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/4095435709.py[0m in [0;36mforward[0;34m(self, x)[0m
[1;32m     58[0m [0;34m[0m[0m
[1;32m     59[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mx[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 60[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0mdecoder[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mencoder[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     61[0m [0;34m[0m[0m
[1;32m     62[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/4095435709.py[0m in [0;36mforward[0;34m(self, x)[0m
[1;32m     40[0m [0;34m[0m[0m
[1;32m     41[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mx[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 42[0;31m         [0;32mreturn[0m [0mself[0m[0;34m.[0m[0msigmoid[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mfc[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mflatten[0m[0;34m([0m[0mx[0m[0;34m)[0m[0;34m)[0m[0;34m)[0m  [0;31m# (B, 1)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     43[0m [0;34m[0m[0m
[1;32m     44[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_wrapped_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1737[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_compiled_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m  [0;31m# type: ignore[misc][0m[0;34m[0m[0;34m[0m[0m
[1;32m   1738[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1739[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_impl[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1740[0m [0;34m[0m[0m
[1;32m   1741[0m     [0;31m# torchrec tests the code consistency with the following code[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py[0m in [0;36m_call_impl[0;34m(self, *args, **kwargs)[0m
[1;32m   1748[0m                 [0;32mor[0m [0m_global_backward_pre_hooks[0m [0;32mor[0m [0m_global_backward_hooks[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1749[0m                 or _global_forward_hooks or _global_forward_pre_hooks):
[0;32m-> 1750[0;31m             [0;32mreturn[0m [0mforward_call[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1751[0m [0;34m[0m[0m
[1;32m   1752[0m         [0mresult[0m [0;34m=[0m [0;32mNone[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/nn/modules/linear.py[0m in [0;36mforward[0;34m(self, input)[0m
[1;32m    123[0m [0;34m[0m[0m
[1;32m    124[0m     [0;32mdef[0m [0mforward[0m[0;34m([0m[0mself[0m[0;34m,[0m [0minput[0m[0;34m:[0m [0mTensor[0m[0;34m)[0m [0;34m->[0m [0mTensor[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 125[0;31m         [0;32mreturn[0m [0mF[0m[0;34m.[0m[0mlinear[0m[0;34m([0m[0minput[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mweight[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mbias[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    126[0m [0;34m[0m[0m
[1;32m    127[0m     [0;32mdef[0m [0mextra_repr[0m[0;34m([0m[0mself[0m[0;34m)[0m [0;34m->[0m [0mstr[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mRuntimeError[0m: mat1 and mat2 shapes cannot be multiplied (32x20736 and 16384x1)

## === cell 4
def generate_predictions(fragment="a", stride=99, patch=None):
    """Return predictions for the test fragments
    Parameters:
        fragment: str, 'a' or 'b' OR int fragment id for train
        stride: int, stride for the sliding window
    """
    t_load = time.time()
    if fragment in ["a", "b"]:  # test set
        mask_filepath = vesuvius_data_path + f"test/{fragment}/mask.png"
        mask = torch.from_numpy(np.array(PIL.Image.open(mask_filepath).convert("1")))
        image_stack = load_image_stack(fragment, Z_START, Z_DIM, folder="test")
    elif fragment in [1, 2, 3]:
        mask, label = load_mask_label(fragment, rect=patch)
        image_stack = load_image_stack(fragment, Z_START, Z_DIM, rect=patch)
    else:
        raise ValueError(f"Unknown fragment: {fragment}")

    test_pixels = get_pixels(mask, img_size, stride=stride)
    print(
        f"Mask size {mask.shape}, Striding by {stride}, we have {len(test_pixels)} pixels to test | Time to load: {time.time() - t_load:.2f}s"
    )
    test_dataset = SubvolumeDataset(image_stack, mask, test_pixels, img_size)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    print(
        f"Length of test dataloader: {len(test_dataloader)} batches of size {batch_size}"
    )

    t_generate = time.time()
    output = torch.zeros_like(mask).float()
    model.eval()

    radius = img_size // 2

    H, W = output.shape
    with torch.no_grad():
        for i, (subvolumes, _) in enumerate(test_dataloader):
            preds = model(subvolumes.to(device).permute(0, 2, 3, 1).unsqueeze(dim=1))
            for j, value in enumerate(preds):
                y, x = test_pixels[i * batch_size + j]
                y0 = max(int(y - radius), 0)
                y1 = min(int(y + radius + 1), H)
                x0 = max(int(x - radius), 0)
                x1 = min(int(x + radius + 1), W)
                output[y0:y1, x0:x1] = value
    print(f"Generated pixels!! Time taken: {time.time() - t_generate:.2f}s")
    return output.cpu()


def dice_coef_torch(preds, targets, beta=0.5, smooth=1e-5):
    preds = np.array(preds)
    targets = np.array(targets)
    y_true_count = targets.sum()
    ctp = preds[targets == 1].sum()
    cfp = preds[targets == 0].sum()
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (
        (1 + beta_squared)
        * (c_precision * c_recall)
        / (beta_squared * c_precision + c_recall + smooth)
    )
    return round(dice, 6)
