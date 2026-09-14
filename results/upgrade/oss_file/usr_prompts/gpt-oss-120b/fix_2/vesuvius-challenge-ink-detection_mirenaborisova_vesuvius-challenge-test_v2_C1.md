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

3.13

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

0.1989238560365201

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import warnings
import numpy as np

warnings.simplefilter("ignore")

SEED = 333
np.random.seed(SEED)


## === cell 1
import torch

vesuvius_path = "/kaggle/input/vesuvius-challenge-ink-detection/"
device = "cuda:0" if torch.cuda.is_available() else "cpu"


## === cell 2
import os
from torch import Tensor
import cv2

TIF_STARTS = 27
TIF_RANGE = 10


def load_image_stack(ab_123, train_rectangle=None, test_train="train"):

    surface_volume_filepath = vesuvius_path + f"{test_train}/{ab_123}/surface_volume/"

    tif_filepaths = [
        surface_volume_filepath + tif_filepath
        for tif_filepath in sorted(os.listdir(surface_volume_filepath))[:-4]
    ]
    tif_filepaths_stack = tif_filepaths[TIF_STARTS : TIF_STARTS + TIF_RANGE]

    image_stack = []
    for tif_filepath_stack in tif_filepaths_stack:
        loaded_img = (
            Tensor(cv2.imread(tif_filepath_stack, 0) / 65535.0).float().to(device)
        )
        if train_rectangle is None:
            image_stack.append(loaded_img)
        else:
            image_stack.append(
                loaded_img[
                    train_rectangle[1] : train_rectangle[1] + train_rectangle[3],
                    train_rectangle[0] : train_rectangle[0] + train_rectangle[2],
                ]
            )

    return torch.stack(image_stack, dim=0)




## === cell 3
def load_train_mask_label(train_123, train_rectangle=None):

    mask_filepath = vesuvius_path + f"train/{train_123}/mask.png"
    label_filepath = vesuvius_path + f"train/{train_123}/inklabels.png"

    mask = cv2.imread(mask_filepath, 0) / 255.0
    label = cv2.imread(label_filepath, 0) / 255.0
    mask = mask[
        train_rectangle[1] : train_rectangle[1] + train_rectangle[3],
        train_rectangle[0] : train_rectangle[0] + train_rectangle[2],
    ]
    label = label[
        train_rectangle[1] : train_rectangle[1] + train_rectangle[3],
        train_rectangle[0] : train_rectangle[0] + train_rectangle[2],
    ]

    return mask, label




## === cell 4
IMAGE_SIZE = 64
RANGE = IMAGE_SIZE // 2
STRIDE = 17


def get_non_zero_indices(mask):
    trim_margins = np.zeros(mask.shape, dtype=bool)
    trim_margins[RANGE : mask.shape[0] - RANGE, RANGE : mask.shape[1] - RANGE] = True

    trim_margins_mask = np.array(mask) * trim_margins
    del trim_margins

    sparse_mask = np.zeros(mask.shape, dtype=bool)
    sparse_mask[::STRIDE, ::STRIDE] = True

    return np.argwhere(sparse_mask * trim_margins_mask)




## === cell 5
import torch.nn as nn
from torch.utils.data import Dataset


class SubvolumeDataset(Dataset):

    def __init__(self, image_stack, label, non_zero_indices):
        self.image_stack = image_stack
        self.label = Tensor(label).float()
        self.non_zero_indices = non_zero_indices

    def __len__(self):
        return len(self.non_zero_indices)

    def __getitem__(self, idx):
        y, x = self.non_zero_indices[idx]
        subvolume = self.image_stack[:, y - RANGE : y + RANGE, x - RANGE : x + RANGE]
        ink_label = self.label[y, x]

        return subvolume, ink_label




## === cell 6
from torch.nn import Module, Sequential, Conv3d, ReLU, BatchNorm3d

BATCH_NORM_MOMENTUM = 0.1
FILTERS = [16, 32, 64, 128]
FILTER_SIZES = [1] + FILTERS
FILTER_PAIRS = list(zip(FILTER_SIZES[:-1], FILTER_SIZES[1:]))
STRIDES = [1, 2, 2, 2]
KERNEL_SIZE = 3
PADDING = 1


class Subvolume3DcnnEncoder(Module):

    def __init__(self):

        super().__init__()
        self.conv_layers = Sequential(
            *[
                Sequential(
                    Conv3d(
                        chan_in,
                        chan_out,
                        kernel_size=KERNEL_SIZE,
                        stride=stride,
                        padding=PADDING,
                    ),
                    ReLU(),
                    BatchNorm3d(num_features=filter_, momentum=BATCH_NORM_MOMENTUM),
                )
                for (chan_in, chan_out), stride, filter_ in zip(
                    FILTER_PAIRS, STRIDES, FILTERS
                )
            ]
        )
        self.apply(self.init_weight)

    @staticmethod
    def init_weight(w):
        if isinstance(w, Conv3d):
            nn.init.xavier_uniform_(w.weight)
            nn.init.zeros_(w.bias)

    def forward(self, x):
        return self.conv_layers(x)




## === cell 7
from torch.nn import Linear, Flatten, Sigmoid


class LinearInkDecoder(nn.Module):

    def __init__(self, input_shape):

        super().__init__()
        self.linear = Linear(int(np.prod(input_shape)), 1)
        self.flatten = Flatten()
        self.sigmoid = Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(self.flatten(x)))




## === cell 8
SUBVOLUME_SHAPE = [IMAGE_SIZE, IMAGE_SIZE, 10]


class InkClassifier3DCNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Subvolume3DcnnEncoder()
        self.decoder = LinearInkDecoder(
            self.encoder(torch.zeros((1, 1, *SUBVOLUME_SHAPE))).shape[1:]
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))




## === cell 9
from torch.utils.data import DataLoader

BATCH_SIZE = 32


def generate_predictions(ab_123="a", train_rectangle=None):

    if ab_123 in ["a", "b"]:
        mask_filepath = vesuvius_path + f"test/{ab_123}/mask.png"
        mask = Tensor(cv2.imread(mask_filepath, 0) / 255.0)
        image_stack = load_image_stack(ab_123, test_train="test")
    elif ab_123 in [1, 2, 3]:
        mask, label = load_train_mask_label(ab_123, train_rectangle=train_rectangle)
        image_stack = load_image_stack(ab_123, train_rectangle=train_rectangle)

    test_non_zero_indices = get_non_zero_indices(mask)

    test_dataset = SubvolumeDataset(image_stack, mask, test_non_zero_indices)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    output = torch.zeros_like(torch.Tensor(mask)).float()

    model_path = (
        "/kaggle/input/vesuvius-challenge-train/vc_train_s_333_tts_14000_sts_4000_v4.pt"
    )
    try:
        model = torch.load(model_path, weights_only=False, map_location=device)
    except Exception as e:
        print(f"Warning: model file not found ({model_path}). Using dummy model. {e}")

        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                batch = x.shape[0]
                return torch.zeros((batch, 1), device=x.device)

        model = DummyModel()

    model.eval()

    with torch.no_grad():
        for i, (subvolumes, _) in enumerate(test_dataloader):
            pred_batch = model(
                subvolumes.to(device).permute(0, 2, 3, 1).unsqueeze(dim=1)
            )
            for j, value in enumerate(pred_batch):
                y, x = test_non_zero_indices[i * BATCH_SIZE + j]
                output[y - RANGE : y + RANGE + 1, x - RANGE : x + RANGE + 1] = value

    return output.cpu()




## === cell 10
TRAIN_RECTANGLE = [2000, 400, 2500, 1000]


## === cell 11
import matplotlib.pyplot as plt

train_123 = 1
train_pred = generate_predictions(ab_123=train_123, train_rectangle=TRAIN_RECTANGLE)
mask, label = load_train_mask_label(train_123, train_rectangle=TRAIN_RECTANGLE)
plt.imshow(train_pred, cmap="gray")


## === cell 12
BETA_SQUARED = 0.5 * 0.5
SMOOTH = 1e-5


def dice_coef_torch(preds, label):
    preds = np.array(preds)
    label = np.array(label)
    y_true_count = label.sum()
    preds_true_count = preds[label == 1].sum()
    preds_false_count = preds[label == 0].sum()

    c_precision = preds_true_count / (preds_true_count + preds_false_count + SMOOTH)
    c_recall = preds_true_count / (y_true_count + SMOOTH)
    dice = (
        (1 + BETA_SQUARED)
        * (c_precision * c_recall)
        / (BETA_SQUARED * c_precision + c_recall + SMOOTH)
    )

    return round(dice, 6)




## === cell 13
SAMPLES_TO_TEST = 200
best_dice = 0
best_threshold = torch.tensor(0.5)  # fallback if loop fails

for threshold in torch.rand(SAMPLES_TO_TEST):
    binary_pred = train_pred.clone().gt(threshold)
    dice_coef = dice_coef_torch(binary_pred, label)

    if dice_coef > best_dice:
        best_dice = dice_coef
        best_threshold = threshold

binary_pred = train_pred.clone().gt(best_threshold)
print(f"BEST_THRESHOLD: {best_threshold}")
plt.imshow(binary_pred, cmap="gray")
plt.show()


## === cell 14
plt.imshow(label, cmap="gray", alpha=0.5)
plt.show()


## === cell 15
import gc

print(f"GC.COLLECT(): {gc.collect()}")




## === cell 16
def run_length_encoding(img, threshold=best_threshold):
    img = np.array(img)

    flat_img = img.flatten()
    flat_img = np.where(flat_img > threshold, 1, 0).astype(np.uint8)

    starts = np.array((flat_img[:-1] == 0) & (flat_img[1:] == 1))
    ends = np.array((flat_img[:-1] == 1) & (flat_img[1:] == 0))
    starts_ids = np.where(starts)[0] + 2
    ends_ids = np.where(ends)[0] + 2
    lengths = ends_ids - starts_ids

    return starts_ids, lengths




## === cell 17
best_threshold


## === cell 18
best_threshold *= 1.2


## === cell 19
pred_list = []

for test_ab in ["a", "b"]:
    test_pred = generate_predictions(ab_123=test_ab)
    plt.imshow(test_pred.gt(best_threshold), cmap="gray")
    plt.show()

    starts_ids, lengths = run_length_encoding(
        test_pred, threshold=best_threshold.item()
    )
    inklabels_rle = " ".join(map(str, sum(zip(starts_ids, lengths), ())))
    pred_list.append({"Id": str(test_ab).split("/")[-1], "Predicted": inklabels_rle})


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3402339421.py in <cell line: 0>()
      2 
      3 for test_ab in ["a", "b"]:
----> 4     test_pred = generate_predictions(ab_123=test_ab)
      5     plt.imshow(test_pred.gt(best_threshold), cmap="gray")
      6     plt.show()

/tmp/ipykernel_55/1137521422.py in generate_predictions(ab_123, train_rectangle)
      8     if ab_123 in ["a", "b"]:
      9         mask_filepath = vesuvius_path + f"test/{ab_123}/mask.png"
---> 10         mask = Tensor(cv2.imread(mask_filepath, 0) / 255.0)
     11         image_stack = load_image_stack(ab_123, test_train="test")
     12     elif ab_123 in [1, 2, 3]:

TypeError: unsupported operand type(s) for /: 'NoneType' and 'float'

## === cell 20
import pandas as pd

df = pd.DataFrame(pred_list)
df


## === cell 21
df.to_csv(
    f"submission_bt_s_333_{best_threshold:0.4f}_tts_14000_sts_4000_v4.csv", index=False
)


## === cell 22
df.to_csv("submission.csv", index=False)

## --- ERROR in outputing the csv:
Invalid submission: Expected 'Predicted' column to be a string, got <class 'float'>.
