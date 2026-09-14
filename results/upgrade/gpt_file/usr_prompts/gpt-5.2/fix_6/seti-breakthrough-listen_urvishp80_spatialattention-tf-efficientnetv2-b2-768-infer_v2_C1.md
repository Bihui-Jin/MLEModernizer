# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

import timm
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2



## === cell 1
try:
    from IPython.display import display  # type: ignore
except Exception:

    def display(x):
        return x




## === cell 2
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




## === cell 3
def set_seed(seed: int):
    """Sets the seed for reproducibility."""
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

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



## === cell 4
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU is available.")
else:
    device = torch.device("cpu")
    print("GPU not available, going to use CPU instead.")



## === cell 5
DATA_ROOT = "../input/seti-breakthrough-listen"
TRAIN_LABELS_CSV = f"{DATA_ROOT}/train_labels.csv"
SAMPLE_SUB_CSV = f"{DATA_ROOT}/sample_submission.csv"
TRAIN_DIR = f"{DATA_ROOT}/train"
TEST_DIR = f"{DATA_ROOT}/test"

assert os.path.exists(TRAIN_LABELS_CSV), f"Missing {TRAIN_LABELS_CSV}"
assert os.path.exists(SAMPLE_SUB_CSV), f"Missing {SAMPLE_SUB_CSV}"



## === cell 6
train = pd.read_csv(TRAIN_LABELS_CSV)
test = pd.read_csv(SAMPLE_SUB_CSV)


def get_train_file_path(image_id):
    return f"{TRAIN_DIR}/{image_id[0]}/{image_id}.npy"


def get_test_file_path(image_id):
    return f"{TEST_DIR}/{image_id[0]}/{image_id}.npy"


train["file_path"] = train["id"].apply(get_train_file_path)
test["file_path"] = test["id"].apply(get_test_file_path)


def _fast_path_sanity_check(paths: pd.Series, name: str, sample_n: int = 512):
    sample_n = min(sample_n, len(paths))
    idx = np.random.RandomState(SEED).choice(len(paths), size=sample_n, replace=False)
    ok = paths.iloc[idx].map(os.path.exists).mean()
    assert (
        ok > 0.99
    ), f"Some {name} files not found in sanity sample (exist rate={ok:.3f})."


_fast_path_sanity_check(train["file_path"], "train")
_fast_path_sanity_check(test["file_path"], "test")

display(test.sample(5, random_state=SEED))



## === cell 7
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
train["fold"] = -1
for fold, (_, val_idx) in enumerate(
    skf.split(train["id"].values, train[TARGET_COL].values)
):
    train.loc[val_idx, "fold"] = fold

trn_df = train[train["fold"] != TRAIN_FOLD].reset_index(drop=True)
val_df = train[train["fold"] == TRAIN_FOLD].reset_index(drop=True)

print(
    "Train size:",
    len(trn_df),
    "Valid size:",
    len(val_df),
    "Pos rate train:",
    trn_df[TARGET_COL].mean(),
    "Pos rate val:",
    val_df[TARGET_COL].mean(),
)




## === cell 8
def spec_augment(x, alpha=0.1):
    t0 = np.random.randint(0, x.shape[0])
    delta = np.random.randint(0, max(1, int(x.shape[0] * alpha)))
    x[t0 : min(t0 + delta, x.shape[0])] = 0
    t0 = np.random.randint(0, x.shape[1])
    delta = np.random.randint(0, max(1, int(x.shape[1] * alpha)))
    x[:, t0 : min(t0 + delta, x.shape[1])] = 0
    return x


class SpecAugment(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super().__init__(always_apply=always_apply, p=p)

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
    else:
        raise ValueError("data must be 'train' or 'valid'")




## === cell 9
criterion = nn.BCEWithLogitsLoss()




## === cell 10
def mixup_criterion(criterion_fn, pred, y_a, y_b, lam):
    return lam * criterion_fn(pred, y_a) + (1 - lam) * criterion_fn(pred, y_b)




## === cell 11
from torch.distributions import Beta


class Mixup(nn.Module):
    def __init__(self, mix_beta=1.0):
        super(Mixup, self).__init__()
        self.beta_distribution = Beta(mix_beta, mix_beta)

    def forward(self, x, y):
        lam = self.beta_distribution.sample().to(x.device)
        batch_size = x.shape[0]
        index = torch.randperm(batch_size, device=x.device)
        mixed_x = lam * x + (1 - lam) * x[index, ...]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam




## === cell 12
def get_activation(activ_name: str = "relu"):
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
        return self.layers(x)


class SSEBlock(nn.Module):
    def __init__(self, in_channels: int):
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
        h = self.sigmoid(self.channel_squeeze(x))
        return x * h


class SpatialAttentionBlock(nn.Module):
    """Spatial Attention for (C, H, W) feature maps"""

    def __init__(self, in_channels, out_channels_list):
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
        h = x
        for i in range(self.n_layers):
            h = getattr(self, f"conv{i + 1}")(h)
        h = h * x
        return h




## === cell 13
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
        elif "res" in name:
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




## === cell 14
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
        if self.use_mixup and training is True:
            x, y_a, y_b, lam = self.mixup(x, input_dict["target"])

        if "deit_base_distilled_patch16_384" == self.backbone_name:
            x = x.unsqueeze(1)
        x = x.expand(-1, 3, -1, -1)

        x = self.backbone(x)
        x = self.neck(x)
        logits = self.head(x)

        output_dict = {"logits": logits}

        if self.loss and self.use_mixup and training is True:
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            output_dict["loss"] = loss
        elif self.loss:
            target = input_dict["target"]
            loss = criterion(logits, target)
            output_dict["loss"] = loss

        return output_dict




## === cell 15
def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    running_loss = 0.0
    n = 0
    optimizer.zero_grad(set_to_none=True)

    for step, item in enumerate(tqdm(loader, desc="train", leave=False)):
        batch = {k: item[k].to(device, non_blocking=True) for k in item.keys()}
        with autocast(enabled=(device.type == "cuda")):
            out = model(batch, training=True)
            loss = out["loss"] / GRADIENT_ACCUMULATION

        scaler.scale(loss).backward()

        if (step + 1) % GRADIENT_ACCUMULATION == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        bs = batch["spect"].size(0)
        running_loss += loss.item() * bs * GRADIENT_ACCUMULATION
        n += bs

    return running_loss / max(1, n)


@torch.no_grad()
def predict_proba(model, loader):
    model.eval()
    preds = []
    with torch.inference_mode():
        for item in tqdm(loader, desc="infer", leave=False):
            batch = {k: item[k].to(device, non_blocking=True) for k in item.keys()}
            out = model(batch, training=False)
            preds.append(out["logits"].float().sigmoid().flatten())
    return torch.cat(preds).detach().cpu().numpy()




## === cell 16
def load_checkpoint(backbone, path):
    """
    Fix: previously hard-failed due to missing external .pt file.
    Now: return (model, loaded_bool). If path missing, caller can train from scratch.
    """
    model = SETINet(
        backbone=MODEL,
        out_dim=1,
        loss=False,
        pretrained=False,
    ).to(device)

    if path is None or (not os.path.exists(path)):
        return model, False

    checkpoint = torch.load(path, map_location=torch.device(device))
    model = nn.DataParallel(model)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model, True




## === cell 17

import torch.nn.functional as F


class TrainDataset(Dataset):
    def __init__(self, df, test=False, transform=None, use_vit=False, is_train=False):
        self.df = df
        self.test = test
        self.file_names = df["file_path"].values
        if not self.test:
            self.labels = df[TARGET_COL].values
        self.transform = transform
        self.use_vit = use_vit
        self.is_train = is_train

        self._fast_resize_only = False
        if isinstance(transform, A.core.composition.Compose):
            tlist = transform.transforms
            if (
                len(tlist) == 2
                and isinstance(tlist[0], A.Resize)
                and isinstance(tlist[1], ToTensorV2)
                and tlist[0].height == DIM1
                and tlist[0].width == DIM2
            ):
                self._fast_resize_only = True

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]

        arr = np.load(file_path, mmap_mode="r", allow_pickle=False)
        x = np.asarray(arr[CHANNELS], dtype=np.float32, order="C")
        x = x.reshape(-1, x.shape[-1]).T

        if self.transform:
            if self._fast_resize_only:
                t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # (1,1,256,819)
                t = F.interpolate(
                    t, size=(DIM1, DIM2), mode="bilinear", align_corners=False
                )
                image_t = t.squeeze(0)  # (1,DIM1,DIM2)
            else:
                image_t = self.transform(image=x)["image"]
            image_t = self.inv_stem(image_t)
        else:
            image_t = torch.from_numpy(x).unsqueeze(0).float()
            image_t = self.inv_stem(image_t)

        if not self.test:
            label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(-1)
            return {"spect": image_t, "target": label}
        else:
            return {"spect": image_t}

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


def fast_collate(batch):
    if "target" in batch[0]:
        spect = torch.stack([b["spect"] for b in batch], dim=0)
        target = torch.stack([b["target"] for b in batch], dim=0)
        return {"spect": spect, "target": target}
    else:
        spect = torch.stack([b["spect"] for b in batch], dim=0)
        return {"spect": spect}


train_dataset = TrainDataset(
    trn_df,
    transform=get_transforms(data="train"),
    test=False,
    use_vit=False,
    is_train=True,
)
valid_dataset = TrainDataset(
    val_df,
    transform=get_transforms(data="valid"),
    test=False,
    use_vit=False,
    is_train=False,
)
test_dataset = TrainDataset(
    test,
    transform=get_transforms(data="valid"),
    test=True,
    use_vit=False,
    is_train=False,
)

_cpu = os.cpu_count() or 4
_num_workers = min(4, max(2, _cpu // 2))

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=_num_workers,
    pin_memory=True,
    drop_last=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
    collate_fn=fast_collate,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
    collate_fn=fast_collate,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=2 if _num_workers > 0 else None,
    collate_fn=fast_collate,
)



## === cell 18
checkpoint_path = (
    "../input/tf-efficientnetv2-b2-cv-8677/tf_efficientnetv2_b2_4_0_9 (1).pt"
)
model, loaded = load_checkpoint(MODEL, checkpoint_path)

if device.type == "cuda":
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass

if not loaded:
    model = SETINet(
        backbone=MODEL, out_dim=1, loss=True, pretrained=True, use_mixup=True
    ).to(device)
    if device.type == "cuda":
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except Exception:
            pass

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(N_EPOCHS):
        tr_loss = train_one_epoch(model, train_loader, optimizer, scaler)

        val_probs = predict_proba(model, valid_loader)
        val_targets = val_df[TARGET_COL].values.astype(np.float32)
        try:
            from sklearn.metrics import roc_auc_score

            val_auc = roc_auc_score(val_targets, val_probs)
        except Exception:
            val_auc = float("nan")
        print(
            f"Epoch {epoch+1}/{N_EPOCHS} - train_loss: {tr_loss:.5f} - val_auc: {val_auc:.5f}"
        )

final_preds = predict_proba(model, test_loader)



## === cell 19
final_preds = np.nan_to_num(final_preds, nan=0.5, posinf=1.0, neginf=0.0)
final_preds = np.clip(final_preds, 0.0, 1.0)
print("Preds:", final_preds.shape, final_preds.min(), final_preds.max())



## === cell 20
submission = pd.read_csv(SAMPLE_SUB_CSV)
submission["target"] = final_preds
submission.to_csv("sub.csv", index=False)
print(submission.head())
print("Wrote sub.csv with shape:", submission.shape)
