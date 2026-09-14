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

0.2743

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import torch
import torch.nn as nn
import numpy as np

vesuvius_data_path = '/kaggle/input/vesuvius-challenge-ink-detection/'

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


def load_mask_label(fragment, rect = None, display = False):
    """Loads and returns mask and label for a given fragment
    Parameters:
      fragment (int in [1, 2, 3]): id of the fragment
      rect (tuple): (x, y, w, h) of the subsection of the image to load
    """
    mask_filepath = vesuvius_data_path+f"train/{fragment}/mask.png"
    label_filepath = vesuvius_data_path+f"train/{fragment}/inklabels.png"
    mask = torch.from_numpy(np.array(PIL.Image.open(mask_filepath).convert('1')))
    label = torch.from_numpy(np.array(PIL.Image.open(label_filepath))).float()
    if rect is not None:
        mask = mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        label = label[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
    if display:
        plt.figure(figsize=(5, 5))
        plt.imshow(label, cmap = 'gray')
        plt.imshow(mask, cmap = 'gray', alpha = 0.5)
        plt.show()
    return mask, label

def load_image_stack(fragment, Z_START, Z_DIM, rect = None, folder='train', display_all = False, display_one = False):
    root_filepath = vesuvius_data_path+f"{folder}/{fragment}/surface_volume/"
    tif_filepaths = [root_filepath+ x for x in sorted(os.listdir(root_filepath))[:-4]]
    layers_to_use = tif_filepaths[Z_START:Z_START+Z_DIM]

    image_stack = []
    for filepath in layers_to_use:
        loaded_img = np.array(PIL.Image.open(filepath), dtype=np.float32)/65535.0
        if rect is None:
            image_stack.append(loaded_img)
        else:
            image_stack.append(loaded_img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]])

    if display_one:
        plt.imshow(np.array(PIL.Image.fromarray(image_stack[0]), dtype=np.float32), cmap='gray')
        plt.axis('off')
        plt.show()

    if display_all:
        plt.figure(figsize=(20,10))
        for j in range(len(image_stack)):
            plt.subplot(2, 5,j+1)
            plt.imshow(np.array(PIL.Image.fromarray(image_stack[j]), dtype=np.float32), cmap='gray')
            plt.axis('off')
        plt.show()

    image_stack = torch.stack([torch.from_numpy(image) for image in image_stack], dim=0)
    return image_stack

def get_pixels(mask, img_size, stride = 0):
    """Returns pixels inside rectangle and mask
    Parameters:
        mask (np.array): mask of the image
        img_size (int): size of the image
        stride (int): how many pixels to skip (e.g. 0 means all pixels, 3 means a gap of 3 horiontally and vertically between pixels)
    """
    radius = int(img_size//2)
    not_border = np.zeros(mask.shape, dtype=bool)
    not_border[radius:mask.shape[0]-radius, radius:mask.shape[1]-radius] = True
    arr_mask = np.array(mask) * not_border

    if stride !=0:
        assert stride % 2 == 1, "stride has to be an odd number!"
        sparse_mask = np.zeros(mask.shape, dtype=bool)
        sparse_mask[::stride, ::stride] = True
        return np.argwhere(sparse_mask*arr_mask)
    
    return np.argwhere(arr_mask)

class SubvolumeDataset(torch.utils.data.Dataset):
    def __init__(self, image_stack, label, pixels, img_size):
        self.image_stack = image_stack
        self.label = label
        self.pixels = pixels
        self.radius = int(img_size//2)
    def __len__(self):
        return len(self.pixels)
    def __getitem__(self, index):
        y, x = self.pixels[index] 
        subvolume = self.image_stack[:, y-self.radius:y+self.radius, x-self.radius:x+self.radius]
        ink_label = self.label[y, x]
        return subvolume, ink_label
    

def create_dataloader(fragment, patch, Z_START, Z_DIM, img_size, batch_size):
    mask, label = load_mask_label(fragment, rect = patch) # display = True. Note: #If the mask covers the entire image, matplotlib will not display it
    image_stack = load_image_stack(fragment, Z_START, Z_DIM, rect = patch)
    pixels = get_pixels(mask, img_size)
    dataset = SubvolumeDataset(image_stack, label, pixels, img_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


## === cell 1

test_patches = {1: [2000, 400, 2500, 1000],
                2: [1500, 1200, 2200, 1000],
                3: [1800, 500, 2300, 1200]}

train_patches = [
    [1, [200, 1500, 4500, 6500]]]


fig, ax = plt.subplots(1, 3, figsize=(10, 10))
for fragment in range(1, 4):
    mask, label = load_mask_label(fragment)
    ax[fragment-1].imshow(label, cmap = 'gray') #show mask label overlaid on image
    ax[fragment-1].imshow(mask, cmap = 'gray', alpha=0.5)

    test_rect = test_patches[fragment]
    test_patch = matplotlib.patches.Rectangle((test_rect[0], test_rect[1]), test_rect[2], test_rect[3], linewidth=2, edgecolor='r', facecolor='none')
    ax[fragment-1].add_patch(test_patch)

    for train_patch in train_patches:
        if train_patch[0] == fragment:
            train_rect = train_patch[1]
            train_patch = matplotlib.patches.Rectangle((train_rect[0], train_rect[1]), train_rect[2], train_rect[3], linewidth=2, edgecolor='b', facecolor='none')
            ax[fragment-1].add_patch(train_patch)

plt.figure(figsize=(10, 10))
plt.show()


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3540795029.py in <cell line: 0>()
     35 fig, ax = plt.subplots(1, 3, figsize=(10, 10))
     36 for fragment in range(1, 4):
---> 37     mask, label = load_mask_label(fragment)
     38     ax[fragment-1].imshow(label, cmap = 'gray') #show mask label overlaid on image
     39     ax[fragment-1].imshow(mask, cmap = 'gray', alpha=0.5)

/tmp/ipykernel_11/2731055931.py in load_mask_label(fragment, rect, display)
     33     mask_filepath = vesuvius_data_path+f"train/{fragment}/mask.png"
     34     label_filepath = vesuvius_data_path+f"train/{fragment}/inklabels.png"
---> 35     mask = torch.from_numpy(np.array(PIL.Image.open(mask_filepath).convert('1')))
     36     label = torch.from_numpy(np.array(PIL.Image.open(label_filepath))).float()
     37     if rect is not None:

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/vesuvius-challenge-ink-detection/train/3/mask.png'

## === cell 2
class Subvolume3DcnnEncoder(nn.Module):

    def __init__(self, batch_norm_momentum, filters):
        super().__init__()
        strides = [1, 2, 2, 2]
        filter_sizes = [1] + filters
        filter_list_pairs = list(zip(filter_sizes[:-1], filter_sizes[1:])) # [(1, 16), (16, 32), (32, 64), (64, 128)]
        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv3d(chan_in, chan_out, kernel_size=3 ,stride=stride, padding=1),
                nn.ReLU(),
            )
                for (chan_in, chan_out), stride, filter in zip(filter_list_pairs, strides, filters)])
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
        return self.sigmoid(self.fc(self.flatten(x)))# (B, 1) 
    
class InkClassifier3DCNN(nn.Module):

    def __init__(self, subvolume_shape=[256,256,10], batch_norm_momentum=0.001, filters=[16, 32, 64, 128]):
        super().__init__()
        self.encoder = Subvolume3DcnnEncoder(batch_norm_momentum, filters)
        self.decoder = LinearInkDecoder(self.encoder(torch.zeros((1, 1, *subvolume_shape))).shape[1:])

    def forward(self, x):
        return self.decoder(self.encoder(x))


## === cell 3
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

img_size = 64 # 256
Z_START = 27
Z_DIM = 10

batch_size = 32
lr = 3e-3

model = InkClassifier3DCNN(subvolume_shape=[img_size, img_size, 10]).to(device)
xb = torch.rand((5,1,img_size,img_size,10)).to(device)
print("forward pass:", model(xb).shape)
print(f"Number of params: {(sum(p.numel() for p in model.parameters() if p.requires_grad)):,}")

eval_print_interval = 1000
total_training_steps = 15000 # (really means 5000*batch_size steps)
subvolume_training_steps = 100000

loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=total_training_steps)

training_step = 0
running_loss = []
model.train()

while training_step < total_training_steps:
    print("-"*20, "New dataloader section", "-"*20)
    random.shuffle(train_patches)
    for fragment, patch in train_patches:
        if training_step >= total_training_steps:
            print("beaking 1")
            break
        t_load = time.time()
        dataloader = create_dataloader(fragment, patch, Z_START, Z_DIM, img_size, batch_size)
        print(f"## Fragment {fragment}, patch {patch}, Taken {time.time() - t_load:.2f}s to load ##") #, Length of dataloader: {len(dataloader)} batches of size {batch_size} | ")
        dataloader_loss = []
        t_dataloader = time.time()
        t_interval = time.time()
        dataloader_steps = 0
        for i, (subvolumes, ink_labels) in enumerate(dataloader):
            if dataloader_steps >= subvolume_training_steps or training_step >= total_training_steps:
                print("Breaking 2")
                break
            subvolumes, ink_labels = subvolumes.to(device), ink_labels.to(device)
            logits = model(subvolumes.permute(0, 2, 3, 1).unsqueeze(dim=1))
            loss = loss_fn(logits, ink_labels.unsqueeze(dim=1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            dataloader_loss.append(loss.item())
            running_loss.append(loss.item())
            training_step += 1
            dataloader_steps += 1
            if i % eval_print_interval == eval_print_interval-1:
                print(f"i: {i} | Running loss: {np.array(running_loss).mean():.4f} | lr: {scheduler.get_last_lr()[0]:.5f} | t_interval {time.time()-t_interval:.2f} t_dataloader: {time.time() - t_dataloader:.2f}s")
                t_interval = time.time()
                running_loss = []


## === cell 4
def generate_predictions(fragment = 'a', stride = 99, patch = None):
    """Return predictions for the test fragments
    Parameters:
        fragment: str, 'a' or 'b'
        stride: int, stride for the sliding window
    """
    t_load = time.time()
    if fragment in ['a', 'b']: # test set
        mask_filepath = vesuvius_data_path+f"test/{fragment}/mask.png"
        mask = torch.from_numpy(np.array(PIL.Image.open(mask_filepath).convert('1')))
        image_stack = load_image_stack(fragment, Z_START, Z_DIM, folder = 'test')
    elif fragment in [1, 2, 3]:
        mask, label = load_mask_label(fragment, rect = patch)
        image_stack = load_image_stack(fragment, Z_START, Z_DIM, rect = patch)
        
    test_pixels = get_pixels(mask, img_size, stride=stride)
    print(f"Mask size {mask.shape}, Striding by {stride}, we have {len(test_pixels)} pixels to test | Time to load: {time.time() - t_load:.2f}s")
    test_dataset =  SubvolumeDataset(image_stack, mask, test_pixels, img_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print(f"Length of test dataloader: {len(test_dataloader)} batches of size {batch_size}")

    t_generate = time.time()
    output = torch.zeros_like(mask).float()
    model.eval()
    radius = stride//2
    with torch.no_grad():
        for i, (subvolumes, _) in enumerate(test_dataloader):   
            for j, value in enumerate(model(subvolumes.to(device).permute(0, 2, 3, 1).unsqueeze(dim=1))):
                y, x = test_pixels[i*batch_size+j]
                output[y-radius:y+radius+1, x-radius: x+radius+1] = value
    print(f"Generated pixels!! Time taken: {time.time() - t_generate:.2f}s")
    return output.cpu()

def dice_coef_torch(preds, targets, beta=0.5, smooth=1e-5):
    preds = np.array(preds)
    targets = np.array(targets)
    y_true_count = targets.sum()
    ctp = preds[targets==1].sum() # targets==1, preds==1
    cfp = preds[targets==0].sum() # targets==0, preds==1
    beta_squared = beta * beta
    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)
    return round(dice, 6)


## === cell 5
train_pred = generate_predictions(fragment = fragment, stride = 19, patch = test_patches[fragment])
mask, label = load_mask_label(fragment, rect = test_patches[fragment])
plt.imshow(train_pred, cmap="gray")


## === cell 6
samples_to_test = 200
best_dice = 0
best_threshold = 0
for threshold in torch.rand(samples_to_test):
    binary_pred = train_pred.clone().gt(threshold)
    dice_coef = dice_coef_torch(binary_pred, label, beta = 0.5)
    if dice_coef > best_dice: 
        best_dice = dice_coef
        best_threshold = threshold
        print(f"Threshold: {threshold.item():.5f} | Dice: {dice_coef:.5f}")

binary_pred = train_pred.clone().gt(best_threshold)
plt.imshow(binary_pred, cmap = "gray")
plt.imshow(label, cmap = 'gray', alpha=0.5)
plt.show()


## === cell 7
plt.imshow(binary_pred, cmap = "gray")


## === cell 8
mask_filepath = vesuvius_data_path+f"test/a/mask.png"
mask = torch.from_numpy(np.array(PIL.Image.open(mask_filepath).convert('1')))
plt.imshow(mask, cmap = "gray")


## === cell 9
mask_filepath = vesuvius_data_path+f"test/b/mask.png"
mask = torch.from_numpy(np.array(PIL.Image.open(mask_filepath).convert('1')))
plt.imshow(mask, cmap = "gray")


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3265420793.py in <cell line: 0>()
      1 mask_filepath = vesuvius_data_path+f"test/b/mask.png"
----> 2 mask = torch.from_numpy(np.array(PIL.Image.open(mask_filepath).convert('1')))
      3 plt.imshow(mask, cmap = "gray")

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/vesuvius-challenge-ink-detection/test/b/mask.png'

## === cell 10
def rle(img, thr=best_threshold):
    img = np.array(img)
    flat_img = img.flatten()
    flat_img = np.where(flat_img > thr, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ix = np.where(starts)[0] + 2
    ends_ix = np.where(ends)[0] + 2
    lengths = ends_ix - starts_ix

    return starts_ix, lengths

pred_list = []
for fragment in ['a', 'b']:
    train_pred = generate_predictions(fragment = fragment, stride = 19)
    plt.imshow(train_pred.gt(best_threshold), cmap="gray")
    plt.show()
    starts_ix, lengths = rle(train_pred, thr=0.4)
    inklabels_rle = " ".join(map(str, sum(zip(starts_ix, lengths), ())))
    pred_list.append({"Id": str(fragment).split("/")[-1], "Predicted": inklabels_rle})


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/838594096.py in <cell line: 0>()
     14 pred_list = []
     15 for fragment in ['a', 'b']:
---> 16     train_pred = generate_predictions(fragment = fragment, stride = 19)
     17     plt.imshow(train_pred.gt(best_threshold), cmap="gray")
     18     plt.show()

/tmp/ipykernel_11/274778276.py in generate_predictions(fragment, stride, patch)
      8     if fragment in ['a', 'b']: # test set
      9         mask_filepath = vesuvius_data_path+f"test/{fragment}/mask.png"
---> 10         mask = torch.from_numpy(np.array(PIL.Image.open(mask_filepath).convert('1')))
     11         image_stack = load_image_stack(fragment, Z_START, Z_DIM, folder = 'test')
     12     elif fragment in [1, 2, 3]:

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/vesuvius-challenge-ink-detection/test/b/mask.png'

## === cell 12
print("Final cell:")


## === cell 13
import pandas as pd
pd.DataFrame(pred_list).to_csv("submission.csv", index=False)
