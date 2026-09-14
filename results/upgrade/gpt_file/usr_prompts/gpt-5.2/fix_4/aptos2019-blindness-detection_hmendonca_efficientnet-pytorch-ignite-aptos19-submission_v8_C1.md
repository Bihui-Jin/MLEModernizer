# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.7583415713181103

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on an external `.pth` checkpoint (which is missing in your `/kaggle/input`), and instead train the exact same EfficientNet defined in your code on the provided `train.csv/train_images` so the notebook runs end-to-end and produces `submission.csv`. I also fix the GPU/CPU dtype mismatch by ensuring the model is moved to the same device as the input and by correctly applying drop-connect (your current `_drop_connect` output is computed but never used). Finally, because the competition metric is quadratic weighted kappa on ordinal labels, I keep the classifier setup but add a small validation split and select the best epoch by validation kappa to push score upward in a stable, metric-aligned way without changing the model architecture or loss type.'

# 9. Code solution

## === cell 0
import os, glob, sys, warnings, random, time

warnings.filterwarnings("ignore")

print("Listing ../input top-level directories/files:")
print("\n".join(sorted(os.listdir("../input"))[:50]))

DATA_ROOT = "../input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

for p in [TRAIN_CSV, TEST_CSV, TRAIN_IMG_DIR, TEST_IMG_DIR, SAMPLE_SUB]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing required path: {p}")

SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)



## === cell 1
import torch
import torch.nn as nn

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class SqueezeExcitation(nn.Module):

    def __init__(self, inplanes, se_planes):
        super(SqueezeExcitation, self).__init__()
        self.reduce_expand = nn.Sequential(
            nn.Conv2d(
                inplanes, se_planes, kernel_size=1, stride=1, padding=0, bias=True
            ),
            Swish(),
            nn.Conv2d(
                se_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_se = torch.mean(x, dim=(-2, -1), keepdim=True)
        x_se = self.reduce_expand(x_se)
        return x_se * x


from torch.nn import functional as F


class MBConv(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size,
        stride,
        expand_rate=1.0,
        se_rate=0.25,
        drop_connect_rate=0.2,
    ):
        super(MBConv, self).__init__()

        expand_planes = int(inplanes * expand_rate)
        se_planes = max(1, int(inplanes * se_rate))

        self.expansion_conv = None
        if expand_rate > 1.0:
            self.expansion_conv = nn.Sequential(
                nn.Conv2d(
                    inplanes,
                    expand_planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm2d(expand_planes, momentum=0.01, eps=1e-3),
                Swish(),
            )
            inplanes = expand_planes

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                inplanes,
                expand_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=expand_planes,
                bias=False,
            ),
            nn.BatchNorm2d(expand_planes, momentum=0.01, eps=1e-3),
            Swish(),
        )

        self.squeeze_excitation = SqueezeExcitation(expand_planes, se_planes)

        self.project_conv = nn.Sequential(
            nn.Conv2d(
                expand_planes, planes, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(planes, momentum=0.01, eps=1e-3),
        )

        self.with_skip = stride == 1
        self.drop_connect_rate = float(drop_connect_rate)

    def _drop_connect(self, x):
        if self.drop_connect_rate <= 0.0:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        drop_mask = (
            torch.rand(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype) + keep_prob
        )
        drop_mask = drop_mask.floor()
        return drop_mask * x / keep_prob

    def forward(self, x):
        z = x
        if self.expansion_conv is not None:
            x = self.expansion_conv(x)

        x = self.depthwise_conv(x)
        x = self.squeeze_excitation(x)
        x = self.project_conv(x)

        if x.shape == z.shape and self.with_skip:
            if self.training and self.drop_connect_rate is not None:
                x = self._drop_connect(x)
            x += z
        return x


from collections import OrderedDict
import math


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_out")
    elif isinstance(module, nn.Linear):
        init_range = 1.0 / math.sqrt(module.weight.shape[1])
        nn.init.uniform_(module.weight, a=-init_range, b=init_range)


class EfficientNet(nn.Module):

    def _setup_repeats(self, num_repeats):
        return int(math.ceil(self.depth_coefficient * num_repeats))

    def _setup_channels(self, num_channels):
        num_channels *= self.width_coefficient
        new_num_channels = math.floor(num_channels / self.divisor + 0.5) * self.divisor
        new_num_channels = max(self.divisor, new_num_channels)
        if new_num_channels < 0.9 * num_channels:
            new_num_channels += self.divisor
        return new_num_channels

    def __init__(
        self,
        num_classes,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        se_rate=0.25,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
    ):
        super(EfficientNet, self).__init__()

        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.divisor = 8

        list_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        list_channels = [self._setup_channels(c) for c in list_channels]

        list_num_repeats = [1, 2, 2, 3, 3, 4, 1]
        list_num_repeats = [self._setup_repeats(r) for r in list_num_repeats]

        expand_rates = [1, 6, 6, 6, 6, 6, 6]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]

        self.stem = nn.Sequential(
            nn.Conv2d(
                3, list_channels[0], kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(list_channels[0], momentum=0.01, eps=1e-3),
            Swish(),
        )

        blocks = []
        counter = 0
        num_blocks = sum(list_num_repeats)
        for idx in range(7):

            num_channels = list_channels[idx]
            next_num_channels = list_channels[idx + 1]
            num_repeats = list_num_repeats[idx]
            expand_rate = expand_rates[idx]
            kernel_size = kernel_sizes[idx]
            stride = strides[idx]
            drop_rate = drop_connect_rate * counter / num_blocks

            name = "MBConv{}_{}".format(expand_rate, counter)
            blocks.append(
                (
                    name,
                    MBConv(
                        num_channels,
                        next_num_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        expand_rate=expand_rate,
                        se_rate=se_rate,
                        drop_connect_rate=drop_rate,
                    ),
                )
            )
            counter += 1
            for i in range(1, num_repeats):
                name = "MBConv{}_{}".format(expand_rate, counter)
                drop_rate = drop_connect_rate * counter / num_blocks
                blocks.append(
                    (
                        name,
                        MBConv(
                            next_num_channels,
                            next_num_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            expand_rate=expand_rate,
                            se_rate=se_rate,
                            drop_connect_rate=drop_rate,
                        ),
                    )
                )
                counter += 1

        self.blocks = nn.Sequential(OrderedDict(blocks))

        self.head = nn.Sequential(
            nn.Conv2d(list_channels[-2], list_channels[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(list_channels[-1], momentum=0.01, eps=1e-3),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(list_channels[-1], num_classes),
        )

        self.apply(init_weights)

    def forward(self, x):
        f = self.stem(x)
        f = self.blocks(f)
        y = self.head(f)
        return y




## === cell 2
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model_path = "efficientNet_best.pth"

best_model = EfficientNet(num_classes=5).to(device)

if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    state_dict = (
        ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    )
    if isinstance(state_dict, dict) and any(
        k.startswith("module.") for k in state_dict.keys()
    ):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    best_model.load_state_dict(state_dict, strict=True)
    print("Loaded existing checkpoint:", model_path)
else:
    print(
        "No external checkpoint found; will train EfficientNet from scratch on train.csv."
    )



## === cell 3
from torchvision.transforms import Compose, Resize
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms import (
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
)

from torch.utils.data import Subset
from PIL import Image
from PIL.Image import BICUBIC


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, root, path_list, targets=None, transform=None, extension=".png"):
        super().__init__()
        self.root = root
        self.path_list = list(path_list)
        self.targets = targets
        self.transform = transform
        self.extension = extension
        if targets is not None:
            assert len(self.path_list) == len(self.targets)
            self.targets = torch.LongTensor(list(targets))

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, path + self.extension)
        sample = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)

        if self.targets is not None:
            return sample, self.targets[index]
        else:
            return sample, torch.LongTensor([])

    def __len__(self):
        return len(self.path_list)


image_size = 224

norm = Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081])

train_transform = Compose(
    [
        Resize((image_size, image_size), BICUBIC),
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation(degrees=10),
        ToTensor(),
        norm,
    ]
)

test_transform = Compose(
    [
        Resize((image_size, image_size), BICUBIC),
        ToTensor(),
        norm,
    ]
)

df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)
df_sample = pd.read_csv(SAMPLE_SUB)

train_dataset_full = ImageDataset(
    root=TRAIN_IMG_DIR,
    path_list=df_train.id_code.values,
    targets=df_train.diagnosis.values,
    transform=train_transform,
)

val_dataset_full = ImageDataset(
    root=TRAIN_IMG_DIR,
    path_list=df_train.id_code.values,
    targets=df_train.diagnosis.values,
    transform=test_transform,  # deterministic eval transform
)

test_dataset = ImageDataset(
    root=TEST_IMG_DIR,
    path_list=df_test.id_code.values,
    transform=test_transform,
)

print("Train rows:", len(df_train), "Test rows:", len(df_test))



## === cell 4
from torch.utils.data import DataLoader

batch_size = 32  # safer for GPU memory; still fast enough
num_workers = min(4, (os.cpu_count() or 2))
print("num_workers:", num_workers)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
train_idx, val_idx = next(sss.split(df_train.id_code.values, df_train.diagnosis.values))

train_loader = DataLoader(
    Subset(train_dataset_full, train_idx),
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    drop_last=True,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    Subset(val_dataset_full, val_idx),
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size * 2,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
)

print(
    "Train batches:",
    len(train_loader),
    "Val batches:",
    len(val_loader),
    "Test batches:",
    len(test_loader),
)



## === cell 5
from tqdm import tqdm


def apply_thresholds(x, thr):
    thr = np.asarray(thr, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    return np.digitize(x, thr).astype(np.int64)


def fit_thresholds_by_coordinate_descent(y_true, y_cont, init_thr=None, n_iter=12):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_cont = np.asarray(y_cont, dtype=np.float32)

    if init_thr is None:
        thr = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
    else:
        thr = np.array(init_thr, dtype=np.float32)

    best_thr = thr.copy()
    best_k = cohen_kappa_score(
        y_true, apply_thresholds(y_cont, best_thr), weights="quadratic"
    )

    for _ in range(n_iter):
        improved = False
        for i in range(4):
            lo = -0.5 if i == 0 else best_thr[i - 1] + 1e-3
            hi = 4.5 if i == 3 else best_thr[i + 1] - 1e-3
            if not (lo < hi):
                continue
            grid = np.linspace(lo, hi, 41, dtype=np.float32)
            local_best_k = best_k
            local_best_t = best_thr[i]
            for t in grid:
                thr_try = best_thr.copy()
                thr_try[i] = t
                pred_try = apply_thresholds(y_cont, thr_try)
                k = cohen_kappa_score(y_true, pred_try, weights="quadratic")
                if k > local_best_k:
                    local_best_k = k
                    local_best_t = t
            if local_best_k > best_k:
                best_k = local_best_k
                best_thr[i] = local_best_t
                improved = True
        if not improved:
            break

    return best_thr, best_k


best_thresholds = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)

if not os.path.exists(model_path):
    best_model = best_model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(best_model.parameters(), lr=3e-4, weight_decay=1e-5)

    epochs = 3  # keep runtime bounded
    best_kappa = -1.0
    best_state = None
    best_thr_state = best_thresholds.copy()

    for epoch in range(1, epochs + 1):
        best_model.train()
        running_loss = 0.0
        for x, y in tqdm(
            train_loader, desc=f"Train epoch {epoch}/{epochs}", leave=False
        ):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = best_model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().cpu())

        best_model.eval()
        val_true = []
        val_cont = []
        with torch.no_grad():
            for x, y in tqdm(
                val_loader, desc=f"Val epoch {epoch}/{epochs}", leave=False
            ):
                x = x.to(device, non_blocking=True)
                y_pred1 = best_model(x)
                y_pred2 = best_model(x.flip(dims=(-1,)))
                probs = 0.5 * (F.softmax(y_pred1, dim=-1) + F.softmax(y_pred2, dim=-1))
                cont = (
                    probs * torch.arange(5, device=probs.device, dtype=probs.dtype)
                ).sum(dim=1)
                val_cont.append(cont.detach().cpu().numpy())
                val_true.append(y.numpy())

        val_true = np.concatenate(val_true)
        val_cont = np.concatenate(val_cont)

        thr, kappa_thr = fit_thresholds_by_coordinate_descent(
            val_true, val_cont, init_thr=best_thr_state
        )

        print(
            f"Epoch {epoch}: train_loss={running_loss/len(train_loader):.4f} val_qwk(argmax)={cohen_kappa_score(val_true, apply_thresholds(val_cont, [0.5,1.5,2.5,3.5]), weights='quadratic'):.4f} val_qwk(thr)={kappa_thr:.4f}"
        )

        if kappa_thr > best_kappa:
            best_kappa = kappa_thr
            best_thr_state = thr.copy()
            best_state = {
                k: v.detach().cpu().clone() for k, v in best_model.state_dict().items()
            }

    if best_state is not None:
        best_model.load_state_dict(best_state, strict=True)
        best_thresholds = best_thr_state.copy()
        torch.save(
            {"state_dict": best_model.state_dict(), "thresholds": best_thresholds},
            model_path,
        )
        print(
            f"Saved trained checkpoint to {model_path} with best val_qwk={best_kappa:.4f} thresholds={best_thresholds}"
        )

if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        best_model.load_state_dict(state_dict, strict=True)
        if "thresholds" in ckpt:
            best_thresholds = np.asarray(ckpt["thresholds"], dtype=np.float32)

all_cont = []
best_model = best_model.to(device)
best_model.eval()
with torch.no_grad():
    for x, _ in tqdm(test_loader, total=len(test_loader), desc="Infer"):
        x = x.to(device, non_blocking=True)
        y_pred1 = best_model(x)
        y_pred2 = best_model(x.flip(dims=(-1,)))
        probs = 0.5 * (F.softmax(y_pred1, dim=-1) + F.softmax(y_pred2, dim=-1))
        cont = (probs * torch.arange(5, device=probs.device, dtype=probs.dtype)).sum(
            dim=1
        )
        all_cont.append(cont.detach().cpu().numpy())

all_cont = np.concatenate(all_cont)
all_pred = apply_thresholds(all_cont, best_thresholds).tolist()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
UnpicklingError                           Traceback (most recent call last)
/tmp/ipykernel_55/928155773.py in <cell line: 0>()
    134 # Change: support loading thresholds from checkpoint (keeps same model; just stores extra metadata).
    135 if os.path.exists(model_path):
--> 136     ckpt = torch.load(model_path, map_location="cpu")
    137     if isinstance(ckpt, dict) and "state_dict" in ckpt:
    138         state_dict = ckpt["state_dict"]

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1468                         )
   1469                     except pickle.UnpicklingError as e:
-> 1470                         raise pickle.UnpicklingError(_get_wo_message(str(e))) from None
   1471                 return _load(
   1472                     opened_zipfile,

UnpicklingError: Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint. 
	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.load` with `weights_only` set to `False` will likely succeed, but it can result in arbitrary code execution. Do it only if you got the file from a trusted source.
	(2) Alternatively, to load with `weights_only=True` please check the recommended steps in the following error message.
	WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray._reconstruct was not an allowed global by default. Please use `torch.serialization.add_safe_globals([_reconstruct])` or the `torch.serialization.safe_globals([_reconstruct])` context manager to allowlist this global if you trust this class/function.

Check the documentation of torch.load to learn more about types accepted by default with weights_only https://pytorch.org/docs/stable/generated/torch.load.html.

## === cell 6
print("Pred count:", len(all_pred), " Test rows:", len(df_test))
if len(all_pred) != len(df_test):
    raise RuntimeError(f"Prediction length mismatch: {len(all_pred)} vs {len(df_test)}")

sub = df_sample[["id_code"]].copy()
pred_map = dict(zip(df_test["id_code"].values.tolist(), all_pred))
sub["diagnosis"] = sub["id_code"].map(pred_map)

missing = sub["diagnosis"].isna().sum()
if missing:
    raise RuntimeError(
        f"Missing predictions for {missing} ids; check id_code mapping/alignment."
    )

sub["diagnosis"] = sub["diagnosis"].astype(int)
sub = sub[["id_code", "diagnosis"]]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
print(sub.head())



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2879728569.py in <cell line: 0>()
----> 1 print("Pred count:", len(all_pred), " Test rows:", len(df_test))
      2 if len(all_pred) != len(df_test):
      3     raise RuntimeError(f"Prediction length mismatch: {len(all_pred)} vs {len(df_test)}")
      4 
      5 # Change: write submission in the exact same id_code order as sample_submission.csv to avoid 0.0 from misalignment.

NameError: name 'all_pred' is not defined

## === cell 7
ss = pd.read_csv(SAMPLE_SUB)
print(
    "id_code order mismatch count vs sample_submission:",
    int((sub.id_code.values != ss.id_code.values).sum()),
)
print(sub.dtypes)
print(sub["diagnosis"].value_counts().sort_index())



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3357153209.py in <cell line: 0>()
      2 print(
      3     "id_code order mismatch count vs sample_submission:",
----> 4     int((sub.id_code.values != ss.id_code.values).sum()),
      5 )
      6 print(sub.dtypes)

NameError: name 'sub' is not defined

## === cell 8
tr = pd.read_csv(TRAIN_CSV)
print(tr.head())
print(tr["diagnosis"].value_counts().sort_index())
