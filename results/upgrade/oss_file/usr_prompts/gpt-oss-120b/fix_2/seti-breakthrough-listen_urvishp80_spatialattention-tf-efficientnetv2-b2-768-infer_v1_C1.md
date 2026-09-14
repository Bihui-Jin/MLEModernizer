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
Identify technosignature signals in cadence snippets taken from a digital spectrometer.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
00034abb3629,0.5
0004be0baf70,0.5
0005be4d0752,0.5
etc.

```

## Dataset
The data is from a digital spectrometer, which takes incoming raw data from the telescope (amounting to hundreds of TB per day) and performs a Fourier Transform to generate a spectrogram. These spectrograms, also referred to as filterbank files, or dynamic spectra, consist of measurements of signal intensity as a function of frequency and time.

Below is an example of an FM radio signal. This is not from the GBT, but from a small antenna attached to a software defined radio dongle (a $20 piece of kit that you can plug into your laptop to pick up signals). The data we get from the GBT are very similar, but split into larger numbers of frequency channels, covering a much broader instantaneous frequency range, and with much better sensitivity.

![frequency-time-plot](https://prod-files-secure.s3.us-west-2.amazonaws.com/667f1cbf-826f-4641-a321-96054292638d/b59a57f3-11a7-4493-8268-55c3fa632f7e/Untitled.png)

The screenshot above shows frequency on the horizontal axis (running from around 88.2 to 89.8 MHz) and time on the vertical axis. The bright orange feature at 88.5 MHz is the FM signal from KQED, a radio station in the San Francisco Bay Area. The solid yellow blocks on either side (one highlighted by the pointer in the screenshot) are the KQED “HD radio” signal (the same data as the FM signal, but encoded digitally). Additional FM stations are visible at different frequencies, including another obvious FM signal (without the corresponding digital sidebands) at 89.5 MHz.

The spectrometer generates similar spectrograms to the one shown above, but typically spanning several GHz of the radio spectrum (rather than the approx. 2 MHz shown above). The data are stored either as filterbank format or HDF5 format files, but essentially are arrays of intensity as a function of frequency and time, accompanied by headers containing metadata such as the direction the telescope was pointed in, the frequency scale, and so on. We generate over 1 PB of spectrograms per year; individual filterbank files can be tens of GB in size. We have discarded the majority of the metadata and are simply presenting numpy arrays consisting of small regions of the spectrograms that we refer to as “snippets”.

The spectrometer is searching for candidate signatures of extraterrestrial technology - so-called technosignatures. The main obstacle to doing so is that our own human technology (not just radio stations, but wifi routers, cellphones, and even electronics that are not deliberately designed to transmit radio signals) also gives off radio signals. We refer to these human-generated signals as “radio frequency interference”, or RFI.

One method we use to isolate candidate technosignatures from RFI is to look for signals that appear to be coming from particular positions on the sky. Typically we do this by alternating observations of our primary target star with observations of three nearby stars: 5 minutes on star “A”, then 5 minutes on star “B”, then back to star “A” for 5 minutes, then “C”, then back to “A”, then finishing with 5 minutes on star “D”. One set of six observations (ABACAD) is referred to as a “cadence”. Since we're just giving you a small range of frequencies for each cadence, we refer to the datasets you'll be analyzing as “cadence snippets”.

An example of an extraterrestrial signal:

![voyager-signal](https://storage.googleapis.com/kaggle-media/competitions/SETI-Berkeley/Screen%20Shot%202021-05-03%20at%2011.39.42.png)

As the plot title suggests, this is the Voyager 1 spacecraft. Even though it's 20 billion kilometers from Earth, it's picked up clearly by the GBT. The first, third, and fifth panels are the “A” target (the spacecraft, in this case). The yellow diagonal line is the radio signal coming from Voyager. It's detected when we point at the spacecraft, and it disappears when we point away. It's a diagonal line in this plot because the relative motion of the Earth and the spacecraft imparts a Doppler drift, causing the frequency to change over time. As it happens, that's another possible way to reject RFI, which has a higher tendency to remain at a fixed frequency over time.

While it would be nice to train our algorithms entirely on observations of interplanetary spacecraft, there are not many examples of them, and we also want to be able to find a wider range of signal types. So we've turned to simulating technosignature candidates.

We've taken tens of thousands of cadence snippets, which we're calling the haystack, and we've hidden needles among them. Some of these needles look similar to the Voyager 1 signal above and should be easy to detect, even with classical detection algorithms. Others are hidden in noisy regions of the spectrum and will be harder, even though they might be relatively obvious on visual inspection:

![needle-signal](https://storage.googleapis.com/kaggle-media/competitions/SETI-Berkeley/Screen%20Shot%202021-05-03%20at%2011.34.06.png)

After we perform the signal injections, we normalize each snippet, so you probably can't identify most of the needles just by looking for excess energy in the corresponding array. You'll likely need a more subtle algorithm that looks for patterns that appear only in the on-target observations.

Not all of the “needle” signals look like diagonal lines, and they may not be present for the entirety of all three “A” observations, but what they do have in common is that they are only present in some or all of the “A” observations (panels 1, 3, and 5 in the cadence snippets). Your challenge is to train an algorithm to find as many needles as you can, while minimizing the number of false positives from the haystack.

- **train/** - a training set of cadence snippet files stored in `numpy` `float16` format (v1.20.1), one file per cadence snippet `id`, with corresponding labels found in the `train_labels.csv` file. Each file has dimension `(6, 273, 256)`, with the 1st dimension representing the 6 positions of the cadence, and the 2nd and 3rd dimensions representing the 2D spectrogram.
- **test/** - the test set cadence snippet files; you must predict whether or not the cadence contains a "needle", which is the `target` for this competition
- **sample_submission.csv** - a sample submission file in the correct format
- **train_labels** - targets corresponding (by `id`) to the cadence snippet files found in the `train/` folder
- **old_leaky_data** - full pre-relaunch data, including test labels; you should not assume this data is helpful (it may or may not be).

# 2. Python version

3.9

# 3. Installed packages

albumentations==2.0.8
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
plotly==5.24.1
plotly-express==0.4.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
timm==1.0.19
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
            description.md (112 lines)
            old_leaky_data.zip (23.6 GB)
            sample_submission.csv (6001 lines)
            sample_submission.csv.zip (60.0 kB)
            test.zip (4.5 GB)
            train.zip (4.7 GB)
            train_labels.csv (54001 lines)
            train_labels.csv.zip (529.5 kB)
            old_leaky_data/
                test_labels_old.csv (35848 lines)
                train_labels_old.csv (50166 lines)
                test_old/
                    0/
                        00034db451c4.npy (838.8 kB)
                        0006316b5ca0.npy (838.8 kB)
                        ... and 2197 other files
                    1/
                        10038983cab1.npy (838.8 kB)
                        100865aff453.npy (838.8 kB)
                        ... and 2278 other files
                    ... and 14 other folders
                train_old/
                    0/
                        00034abb3629.npy (838.8 kB)
                        0004300a0b9b.npy (838.8 kB)
                        ... and 3143 other files
                    1/
                        1000e00b26db.npy (838.8 kB)
                        100148224705.npy (838.8 kB)
                        ... and 3142 other files
                    ... and 14 other folders
            seti-breakthrough-listen/
                description.md (112 lines)
                old_leaky_data.zip (23.6 GB)
                ... and 6 other files
                old_leaky_data/
                    test_labels_old.csv (35848 lines)
                    train_labels_old.csv (50166 lines)
                    test_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    train_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                seti-breakthrough-listen/
                test/
                    0/
                        0016fd6c09d476d.npy (838.8 kB)
                        0017643c1c5c254.npy (838.8 kB)
                        ... and 374 other files
                    1/
                        1001ca1d08f9235.npy (838.8 kB)
                        1016de9cec2dc8a.npy (838.8 kB)
                        ... and 353 other files
                    ... and 15 other folders
                train/
                    0/
                        0000799a2b2c42d.npy (838.8 kB)
                        00042890562ff68.npy (838.8 kB)
                        ... and 3335 other files
                    1/
                        100105755d4c5b1.npy (838.8 kB)
                        1001a55ebce86f2.npy (838.8 kB)
                        ... and 3392 other files
                    ... and 15 other folders
            test/
                0/
                    0016fd6c09d476d.npy (838.8 kB)
                    0017643c1c5c254.npy (838.8 kB)
                    ... and 374 other files
                1/
                    1001ca1d08f9235.npy (838.8 kB)
                    1016de9cec2dc8a.npy (838.8 kB)
                    ... and 353 other files
                ... and 15 other folders
            train/
                0/
                    0000799a2b2c42d.npy (838.8 kB)
                    00042890562ff68.npy (838.8 kB)
                    ... and 3335 other files
                1/
                    100105755d4c5b1.npy (838.8 kB)
                    1001a55ebce86f2.npy (838.8 kB)
                    ... and 3392 other files
                ... and 15 other folders
        input/
            description.md (112 lines)
            old_leaky_data.zip (23.6 GB)
            sample_submission.csv (6001 lines)
            sample_submission.csv.zip (60.0 kB)
            test.zip (4.5 GB)
            train.zip (4.7 GB)
            train_labels.csv (54001 lines)
            train_labels.csv.zip (529.5 kB)
            old_leaky_data/
                test_labels_old.csv (35848 lines)
                train_labels_old.csv (50166 lines)
                test_old/
                    0/
                        00034db451c4.npy (838.8 kB)
                        0006316b5ca0.npy (838.8 kB)
                        ... and 2197 other files
                    1/
                        10038983cab1.npy (838.8 kB)
                        100865aff453.npy (838.8 kB)
                        ... and 2278 other files
                    ... and 14 other folders
                train_old/
                    0/
                        00034abb3629.npy (838.8 kB)
                        0004300a0b9b.npy (838.8 kB)
                        ... and 3143 other files
                    1/
                        1000e00b26db.npy (838.8 kB)
                        100148224705.npy (838.8 kB)
                        ... and 3142 other files
                    ... and 14 other folders
            seti-breakthrough-listen/
                description.md (112 lines)
                old_leaky_data.zip (23.6 GB)
                ... and 6 other files
                old_leaky_data/
                    test_labels_old.csv (35848 lines)
                    train_labels_old.csv (50166 lines)
                    test_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    train_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                seti-breakthrough-listen/
                test/
                    0/
                        0016fd6c09d476d.npy (838.8 kB)
                        0017643c1c5c254.npy (838.8 kB)
                        ... and 374 other files
                    1/
                        1001ca1d08f9235.npy (838.8 kB)
                        1016de9cec2dc8a.npy (838.8 kB)
                        ... and 353 other files
                    ... and 15 other folders
                train/
                    0/
                        0000799a2b2c42d.npy (838.8 kB)
                        00042890562ff68.npy (838.8 kB)
                        ... and 3335 other files
                    1/
                        100105755d4c5b1.npy (838.8 kB)
                        1001a55ebce86f2.npy (838.8 kB)
                        ... and 3392 other files
                    ... and 15 other folders
            test/
                0/
                    0016fd6c09d476d.npy (838.8 kB)
                    0017643c1c5c254.npy (838.8 kB)
                    ... and 374 other files
                1/
                    1001ca1d08f9235.npy (838.8 kB)
                    1016de9cec2dc8a.npy (838.8 kB)
                    ... and 353 other files
                ... and 15 other folders
            train/
                0/
                    0000799a2b2c42d.npy (838.8 kB)
                    00042890562ff68.npy (838.8 kB)
                    ... and 3335 other files
                1/
                    100105755d4c5b1.npy (838.8 kB)
                    1001a55ebce86f2.npy (838.8 kB)
                    ... and 3392 other files
                ... and 15 other folders
        working/
            seti-breakthrough-listen/
                description.md (112 lines)
                old_leaky_data.zip (23.6 GB)
                ... and 6 other files
                old_leaky_data/
                    test_labels_old.csv (35848 lines)
                    train_labels_old.csv (50166 lines)
                    test_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    train_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                seti-breakthrough-listen/
                test/
                    0/
                        0016fd6c09d476d.npy (838.8 kB)
                        0017643c1c5c254.npy (838.8 kB)
                        ... and 374 other files
                    1/
                        1001ca1d08f9235.npy (838.8 kB)
                        1016de9cec2dc8a.npy (838.8 kB)
                        ... and 353 other files
                    ... and 15 other folders
                train/
                    0/
                        0000799a2b2c42d.npy (838.8 kB)
                        00042890562ff68.npy (838.8 kB)
                        ... and 3335 other files
                    1/
                        100105755d4c5b1.npy (838.8 kB)
                        1001a55ebce86f2.npy (838.8 kB)
                        ... and 3392 other files
                    ... and 15 other folders
```

-> data/old_leaky_data/test_labels_old.csv has 35847 rows and 2 columns.
The columns are: id, target

-> data/old_leaky_data/train_labels_old.csv has 50165 rows and 2 columns.
The columns are: id, target

-> data/sample_submission.csv has 6000 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/old_leaky_data/test_labels_old.csv has 35847 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/old_leaky_data/train_labels_old.csv has 50165 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/sample_submission.csv has 6000 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/train_labels.csv has 54000 rows and 2 columns.
The columns are: id, target

-> data/train_labels.csv has 54000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Target score

0.7568656637101057

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'The fix adds a robust checkpoint loader that builds the correct absolute path and falls back to a simple constant‑output model when the file is missing, preventing the FileNotFoundError. It also computes the global mean label to use for the dummy predictions, ensuring the pipeline runs end‑to‑end and creates a valid `sub.csv` submission.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from tqdm import tqdm
import time

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam, SGD
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
)
from torch.optim.optimizer import Optimizer
import torchvision.utils as vutils


import timm
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2



## === cell 1
SEED = 42
N_FOLDS = 5
TRAIN_FOLD = 4
TARGET_COL = "target"
N_EPOCHS = 1
BATCH_SIZE = 32
DIM1 = 768
DIM2 = 768
LR = 1e-4
MAX_LR = 5e-4
PRECISION = 16
GRADIENT_ACCUMULATION = 1
EARLY_STOP = 3
MODEL = "tf_efficientnetv2_b2"
NEW_HEAD = False
CHANNELS = [0, 2, 4]


LR *= BATCH_SIZE / 32
MAX_LR *= BATCH_SIZE / 32




## === cell 2
def set_seed(seed=int):
    """Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    np.random.seed(seed)
    random_state = np.random.RandomState(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    return random_state


random_state = set_seed(SEED)



## === cell 3
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available.")
else:
    device = torch.device("cpu")
    print("GPU not available, going to use CPU instead.")



## === cell 4
train = pd.read_csv("../input/seti-breakthrough-listen/train_labels.csv")
test = pd.read_csv("../input/seti-breakthrough-listen/sample_submission.csv")

GLOBAL_MEAN_TARGET = train[TARGET_COL].mean()


def get_train_file_path(image_id):
    return f"../input/seti-breakthrough-listen/train/{image_id[0]}/{image_id}.npy"


def get_test_file_path(image_id):
    return f"../input/seti-breakthrough-listen/test/{image_id[0]}/{image_id}.npy"


train["file_path"] = train["id"].apply(get_train_file_path)
test["file_path"] = test["id"].apply(get_test_file_path)

display(test.sample(5))




## === cell 5
class TrainDataset(Dataset):
    def __init__(self, df, test=False, transform=None, use_vit=False):
        self.df = df
        self.test = test
        self.file_names = df["file_path"].values
        if not self.test:
            self.labels = df[TARGET_COL].values
        self.transform = transform
        self.use_vit = use_vit

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]

        image = np.load(file_path)[CHANNELS]
        image = image.astype(np.float32)
        image = np.vstack(image).T
        if self.transform:
            image = self.transform(image=image)["image"]
            image = self.inv_stem(image)
        else:
            image = image[np.newaxis, :, :]
            image = torch.from_numpy(image).float()
            image = self.inv_stem(image)
        if not self.test:
            label = torch.unsqueeze(torch.tensor(self.labels[idx]).float(), -1)
            return {"spect": image, "target": label}
        else:
            return {"spect": image}

    def inv_stem(self, x):
        if self.use_vit:
            x1 = x.transpose(0, 1).view(24, 24, 16, 16)
            y = torch.zeros(384, 384, dtype=x.dtype)
            for i in range(24):
                for j in range(24):
                    y[i * 16 : (i + 1) * 16, j * 16 : (j + 1) * 16] = x1[i, j]
            return y
        else:
            return x




## === cell 6
def spec_augment(x, alpha=0.1):
    t0 = np.random.randint(0, x.shape[0])
    delta = np.random.randint(0, int(x.shape[0] * alpha))
    x[t0 : min(t0 + delta, x.shape[0])] = 0
    t0 = np.random.randint(0, x.shape[1])
    delta = np.random.randint(0, int(x.shape[1] * alpha))
    x[:, t0 : min(t0 + delta, x.shape[1])] = 0
    return x


class SpecAugment(ImageOnlyTransform):
    def apply(self, img, **params):
        return spec_augment(img)


def get_transforms(*, data):
    if data == "train":
        return A.Compose(
            [
                A.Resize(DIM1, DIM2),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(rotate_limit=0, p=0.3),
                A.MotionBlur(p=0.3),
                SpecAugment(p=0.3),
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        return A.Compose(
            [
                A.Resize(DIM1, DIM2),
                ToTensorV2(),
            ]
        )




## === cell 7
def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = torch.distributions.Beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]

    index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)




## === cell 8
from torch.distributions import Beta


class Mixup(nn.Module):
    def __init__(self, mix_beta=1.0):

        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, x, y):
        lam = self.beta_distribution.sample().to(device)
        batch_size = x.shape[0]
        index = torch.randperm(batch_size)
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam




## === cell 9
import math
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any


class MADGRAD(Optimizer):

    def __init__(
        self,
        params: _params_t,
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0,
        eps: float = 1e-6,
    ):
        if momentum < 0 or momentum >= 1:
            raise ValueError(f"Momentum {momentum} must be in the range [0,1]")
        if lr <= 0:
            raise ValueError(f"Learning rate {lr} must be positive")
        if weight_decay < 0:
            raise ValueError(f"Weight decay {weight_decay} must be non-negative")
        if eps < 0:
            raise ValueError(f"Eps must be non-negative")

        defaults = dict(lr=lr, eps=eps, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self) -> bool:
        return False

    @property
    def supports_flat_params(self) -> bool:
        return True

    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:

        loss = None
        if closure is not None:
            loss = closure()

        if "k" not in self.state:
            self.state["k"] = torch.tensor([0], dtype=torch.long)
        k = self.state["k"].item()

        for group in self.param_groups:
            eps = group["eps"]
            lr = group["lr"] + eps
            decay = group["weight_decay"]
            momentum = group["momentum"]

            ck = 1 - momentum
            lamb = lr * math.pow(k + 1, 0.5)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if "grad_sum_sq" not in state:
                    state["grad_sum_sq"] = torch.zeros_like(p.data).detach()
                    state["s"] = torch.zeros_like(p.data).detach()
                    if momentum != 0:
                        state["x0"] = torch.clone(p.data).detach()

                if momentum != 0.0 and grad.is_sparse:
                    raise RuntimeError(
                        "momentum != 0 is not compatible with sparse gradients"
                    )

                grad_sum_sq = state["grad_sum_sq"]
                s = state["s"]

                if decay != 0:
                    if grad.is_sparse:
                        raise RuntimeError(
                            "weight_decay option is not compatible with sparse gradients"
                        )

                    grad.add_(p.data, alpha=decay)

                if grad.is_sparse:
                    grad = grad.coalesce()
                    grad_val = grad._values()

                    p_masked = p.sparse_mask(grad)
                    grad_sum_sq_masked = grad_sum_sq.sparse_mask(grad)
                    s_masked = s.sparse_mask(grad)

                    rms_masked_vals = grad_sum_sq_masked._values().pow(1 / 3).add_(eps)
                    x0_masked_vals = p_masked._values().addcdiv(
                        s_masked._values(), rms_masked_vals, value=1
                    )

                    grad_sq = grad * grad
                    grad_sum_sq.add_(grad_sq, alpha=lamb)
                    grad_sum_sq_masked.add_(grad_sq, alpha=lamb)

                    rms_masked_vals = grad_sum_sq_masked._values().pow_(1 / 3).add_(eps)

                    s.add_(grad, alpha=lamb)
                    s_masked._values().add_(grad_val, alpha=lamb)

                    p_kp1_masked_vals = x0_masked_vals.addcdiv(
                        s_masked._values(), rms_masked_vals, value=-1
                    )
                    p_masked._values().add_(p_kp1_masked_vals, alpha=-1)
                    p.data.add_(p_masked, alpha=-1)
                else:
                    if momentum == 0:
                        rms = grad_sum_sq.pow(1 / 3).add_(eps)
                        x0 = p.data.addcdiv(s, rms, value=1)
                    else:
                        x0 = state["x0"]

                    grad_sum_sq.addcmul_(grad, grad, value=lamb)
                    rms = grad_sum_sq.pow(1 / 3).add_(eps)

                    s.data.add_(grad, alpha=lamb)

                    if momentum == 0:
                        p.data.copy_(x0.addcdiv(s, rms, value=-1))
                    else:
                        z = x0.addcdiv(s, rms, value=-1)

                        p.data.mul_(1 - ck).add_(z, alpha=ck)

        self.state["k"] += 1
        return loss




## === cell 10
def get_activation(activ_name: str = "relu"):
    """"""
    act_dict = {
        "relu": nn.ReLU(inplace=True),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "identity": nn.Identity(),
    }
    if activ_name in act_dict:
        return act_dict[activ_name]
    else:
        raise NotImplementedError


class Conv2dBNActiv(nn.Module):
    """Conv2d -> (BN ->) -> Activation"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
        use_bn: bool = True,
        activ: str = "relu",
    ):
        """"""
        super(Conv2dBNActiv, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=bias
            )
        )
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(get_activation(activ))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        """Forward"""
        return self.layers(x)


class SSEBlock(nn.Module):
    """channel `S`queeze and `s`patial `E`xcitation Block."""

    def __init__(self, in_channels: int):
        """Initialize."""
        super(SSEBlock, self).__init__()
        self.channel_squeeze = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward."""
        h = self.sigmoid(self.channel_squeeze(x))
        return x * h


class SpatialAttentionBlock(nn.Module):
    """Spatial Attention for (C, H, W) feature maps"""

    def __init__(
        self,
        in_channels,
        out_channels_list,
    ):
        """Initialize"""
        super(SpatialAttentionBlock, self).__init__()
        self.n_layers = len(out_channels_list)
        channels_list = [in_channels] + out_channels_list
        assert self.n_layers > 0
        assert channels_list[-1] == 1

        for i in range(self.n_layers - 1):
            in_chs, out_chs = channels_list[i : i + 2]
            layer = Conv2dBNActiv(in_chs, out_chs, 3, 1, 1, activ="relu")
            setattr(self, f"conv{i + 1}", layer)

        in_chs, out_chs = channels_list[-2:]
        layer = Conv2dBNActiv(in_chs, out_chs, 3, 1, 1, activ="sigmoid")
        setattr(self, f"conv{self.n_layers}", layer)

    def forward(self, x):
        """Forward"""
        h = x
        for i in range(self.n_layers):
            h = getattr(self, f"conv{i + 1}")(h)

        h = h * x
        return h




## === cell 11
class Backbone(nn.Module):

    def __init__(self, name="resnet18", pretrained=True):
        super(Backbone, self).__init__()
        self.net = timm.create_model(name, pretrained=pretrained)

        if "regnet" in name:
            self.out_features = self.net.head.fc.in_features
        elif "vit" in name:
            self.out_features = self.net.head.in_features
        elif name == "deit_base_distilled_patch16_384":
            self.out_features = 768
        elif "csp" in name:
            self.out_features = self.net.head.fc.in_features
        elif "res" in name:  # works also for resnest
            self.out_features = self.net.fc.in_features
        elif "efficientnet" in name:
            self.out_features = self.net.classifier.in_features
        elif "densenet" in name:
            self.out_features = self.net.classifier.in_features
        elif "senet" in name:
            self.out_features = self.net.fc.in_features
        elif "inception" in name:
            self.out_features = self.net.last_linear.in_features

        else:
            self.out_features = self.net.classifier.in_features

    def forward(self, x):
        x = self.net.forward_features(x)

        return x




## === cell 12
class SETINet(nn.Module):
    def __init__(
        self,
        backbone,
        out_dim,
        embedding_size=512,
        loss=False,
        pretrained=True,
        use_mixup=True,
    ):
        super(SETINet, self).__init__()
        self.backbone_name = backbone
        self.loss = loss
        self.out_dim = out_dim
        self.use_mixup = use_mixup

        self.mixup = Mixup()
        self.backbone = Backbone(backbone, pretrained=pretrained)
        if int(embedding_size) != int(self.backbone.out_features):
            self.embedding_size = self.backbone.out_features // 2
        else:
            self.embedding_size = embedding_size

        self.neck = nn.Sequential(
            SpatialAttentionBlock(self.backbone.out_features, [64, 32, 16, 1]),
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(start_dim=1),
            nn.Linear(self.backbone.out_features, self.embedding_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        self.head = nn.Linear(self.embedding_size, out_dim)

    def forward(
        self, input_dict, training=True, get_embeddings=False, get_attentions=False
    ):

        x = input_dict["spect"]
        if self.use_mixup and training == True:
            x, y_a, y_b, lam = self.mixup(x, input_dict["target"])

        if "deit_base_distilled_patch16_384" == self.backbone_name:
            x = x.unsqueeze(1)
        x = x.expand(-1, 3, -1, -1)

        x = self.backbone(x)
        x = self.neck(x)

        logits = self.head(x)

        output_dict = {"logits": logits}

        if self.loss and self.use_mixup and training == True:
            target = input_dict["target"]
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            output_dict["loss"] = loss
        elif self.loss:
            target = input_dict["target"]
            loss = criterion(logits, target)
            output_dict["loss"] = loss

        return output_dict




## === cell 13
def load_checkpoint(backbone, path):
    """
    Load a pre‑trained checkpoint if it exists.
    If the file is missing, fall back to a dummy model that predicts the global mean target.
    """
    abs_path = os.path.join(
        "/kaggle/input", os.path.basename(os.path.dirname(path)), os.path.basename(path)
    )

    try:
        model = SETINet(
            backbone=backbone,
            out_dim=1,
            loss=False,
            pretrained=False,
        ).to(device)
        checkpoint = torch.load(abs_path, map_location=device)
        model = nn.DataParallel(model)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        return model
    except FileNotFoundError:
        print(f"Checkpoint not found at {abs_path}. Using dummy constant‑output model.")

        class DummyModel(nn.Module):
            def __init__(self, constant):
                super().__init__()
                self.constant = torch.tensor(
                    constant, dtype=torch.float32, device=device
                )

            def forward(self, input_dict, training=False, **kwargs):
                batch_size = input_dict["spect"].size(0)
                logits = self.constant.expand(batch_size, 1)
                return {"logits": logits}

        dummy = DummyModel(GLOBAL_MEAN_TARGET)
        dummy.eval()
        return dummy




## === cell 14
predictions = []

model = load_checkpoint(
    MODEL, "../input/tf-efficientnetv2-b2-cv-8786/tf_efficientnetv2_b2_4_0_9.pt"
)

val_dataset = TrainDataset(
    test, transform=get_transforms(data="valid"), test=True, use_vit=False
)
dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
with torch.no_grad():
    for item in tqdm(dataloader, position=0, leave=True):
        batch = {k: item[k].to(device, non_blocking=True) for k in item.keys()}
        prediction = model(batch, training=False)
        predictions.append(prediction["logits"].flatten().sigmoid())
predictions = torch.cat(predictions).detach().cpu()
final_preds = predictions.squeeze(-1).numpy()



## === cell 15
submission = pd.read_csv("../input/seti-breakthrough-listen/sample_submission.csv")
submission["target"] = final_preds
submission.to_csv("sub.csv", index=False)
submission.head()
