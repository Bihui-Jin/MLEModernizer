# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os, glob, subprocess, textwrap, sys, math, random

print(
    subprocess.check_output(["bash", "-lc", "ls -la ../input/ | head -n 200"]).decode(
        "utf-8"
    )
)

CANDIDATE_COMP_ROOTS = [
    "../input/aptos2019-blindness-detection",
    "/kaggle/input/aptos2019-blindness-detection",
    "../input",  # fallback (older kernels sometimes mount directly)
]
COMP_ROOT = None
for p in CANDIDATE_COMP_ROOTS:
    if os.path.exists(os.path.join(p, "train.csv")) and (
        os.path.exists(os.path.join(p, "train_images"))
        or os.path.exists(os.path.join(p, "train_images.zip"))
    ):
        COMP_ROOT = p
        break
if COMP_ROOT is None:
    raise FileNotFoundError(
        "Could not locate competition data root containing train.csv and images. "
        f"Tried: {CANDIDATE_COMP_ROOTS}"
    )
print("Using COMP_ROOT =", COMP_ROOT)

WORKING_DIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
print("WORKING_DIR =", WORKING_DIR)




## === cell 1
model_path = "efficientNet_best.pth"

candidates = []
candidates += glob.glob("../input/efficientnet*/efficientNet_*.pth")
if not candidates:
    candidates += glob.glob("../input/**/*.pth", recursive=True)

if candidates:
    preferred = [p for p in candidates if "efficientNet" in os.path.basename(p)]
    weights_path = preferred[0] if preferred else candidates[0]
    print(
        "Found external weights candidate; will validate compatibility:", weights_path
    )
    subprocess.check_call(["bash", "-lc", f"md5sum '{weights_path}' || true"])
else:
    weights_path = None
    print(
        "No external .pth weights found under ../input; will train and write:",
        model_path,
    )




## === cell 2
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict


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
        self.drop_connect_rate = torch.tensor(drop_connect_rate, requires_grad=False)

    def _drop_connect(self, x):
        keep_prob = 1.0 - self.drop_connect_rate
        drop_mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) + keep_prob
        drop_mask = drop_mask.type_as(x)
        drop_mask.floor_()
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




## === cell 3
import numpy as np
import pandas as pd
from PIL import Image
from PIL.Image import BICUBIC
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def seed_everything(seed=2020):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(2020)


def normalize_id(x):
    x = str(x).strip()
    if x.lower().endswith(".png"):
        x = x[:-4]
    if x.lower().endswith(".jpg"):
        x = x[:-4]
    if x.lower().endswith(".jpeg"):
        x = x[:-5]
    return x


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, path_list, targets=None, transform=None, extension=".png"):
        super().__init__()
        self.root = root
        self.path_list = [normalize_id(p) for p in list(path_list)]
        self.targets = targets
        self.transform = transform
        self.extension = extension
        if targets is not None:
            assert len(self.path_list) == len(self.targets)
            self.targets = torch.LongTensor(targets)

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, path + self.extension)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Missing image file: {img_path}")
        sample = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)

        if self.targets is not None:
            return sample, self.targets[index]
        else:
            return sample, torch.LongTensor([])

    def __len__(self):
        return len(self.path_list)


image_size = 256

test_transform = Compose(
    [
        Resize((image_size, image_size), BICUBIC),
        ToTensor(),
        Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081]),
    ]
)

train_transform = test_transform

df_train = pd.read_csv(os.path.join(COMP_ROOT, "train.csv"))
df_test = pd.read_csv(os.path.join(COMP_ROOT, "test.csv"))

df_train["id_code"] = df_train["id_code"].map(normalize_id)
df_test["id_code"] = df_test["id_code"].map(normalize_id)

print("train:", df_train.shape, "test:", df_test.shape)
print(df_train.head())




## === cell 4
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score

batch_size = 32
num_workers = min(4, os.cpu_count() or 0)
print("num_workers:", num_workers)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=2020)
train_idx, val_idx = next(
    sss.split(df_train["id_code"].values, df_train["diagnosis"].values)
)

df_tr = df_train.iloc[train_idx].reset_index(drop=True)
df_va = df_train.iloc[val_idx].reset_index(drop=True)

train_dataset = ImageDataset(
    root=os.path.join(COMP_ROOT, "train_images"),
    path_list=df_tr.id_code.values,
    targets=df_tr.diagnosis.values,
    transform=train_transform,
)
val_dataset = ImageDataset(
    root=os.path.join(COMP_ROOT, "train_images"),
    path_list=df_va.id_code.values,
    targets=df_va.diagnosis.values,
    transform=test_transform,
)
test_dataset = ImageDataset(
    root=os.path.join(COMP_ROOT, "test_images"),
    path_list=df_test.id_code.values,
    transform=test_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    drop_last=True,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
)


def qwk(y_true, y_pred):
    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def probs_to_expected_value(probs_np):
    classes = np.arange(probs_np.shape[1], dtype=np.float32)
    return (probs_np * classes[None, :]).sum(axis=1)


def apply_thresholds(x, thr):
    return np.digitize(x, bins=np.asarray(thr, dtype=np.float32), right=False)


def find_best_thresholds(x, y, init_thr=(0.5, 1.5, 2.5, 3.5), step=0.05, n_iter=6):
    thr = np.array(init_thr, dtype=np.float32)
    best_thr = thr.copy()
    best = -1e9
    for _ in range(n_iter):
        improved = False
        for k in range(4):
            grid = np.arange(
                best_thr[k] - step, best_thr[k] + step + 1e-9, step, dtype=np.float32
            )
            local_best = best
            local_thr = best_thr[k]
            for v in grid:
                tmp = best_thr.copy()
                tmp[k] = v
                tmp = np.sort(tmp)  # keep monotonic
                pred = apply_thresholds(x, tmp)
                score = qwk(y, pred)
                if score > local_best:
                    local_best = score
                    local_thr = v
            if local_best > best:
                best = local_best
                best_thr[k] = local_thr
                best_thr = np.sort(best_thr)
                improved = True
        if not improved:
            step *= 0.5
    return best_thr, best




## === cell 5
best_model = EfficientNet(num_classes=5).to(device)


def accuracy_from_logits(logits, y_true):
    y_pred = torch.argmax(logits, dim=1)
    return (y_pred == y_true).float().mean().item()


def _state_dict_compatible(model, state_dict):
    if not isinstance(state_dict, dict):
        return False, "state_dict is not a dict"
    msd = model.state_dict()
    missing = []
    mismatched = []
    for k, v in msd.items():
        if k not in state_dict:
            missing.append(k)
        else:
            if tuple(state_dict[k].shape) != tuple(v.shape):
                mismatched.append((k, tuple(state_dict[k].shape), tuple(v.shape)))
    unexpected = [k for k in state_dict.keys() if k not in msd]
    if missing or mismatched:
        return (
            False,
            f"missing={len(missing)} mismatched={len(mismatched)} unexpected={len(unexpected)}",
        )
    return True, f"ok unexpected={len(unexpected)}"


best_thr_from_training = None

if (
    "weights_path" in globals()
    and weights_path is not None
    and os.path.exists(weights_path)
):
    try:
        ext = torch.load(weights_path, map_location="cpu")
        if (
            isinstance(ext, dict)
            and "state_dict" in ext
            and isinstance(ext["state_dict"], dict)
        ):
            ext_sd = ext["state_dict"]
        else:
            ext_sd = ext
        ok, msg = _state_dict_compatible(best_model, ext_sd)
        print("External weights compatibility:", ok, msg)
        if ok:
            subprocess.check_call(
                ["bash", "-lc", f"cp '{weights_path}' '{model_path}'"]
            )
            print("Copied compatible external weights to", model_path)
        else:
            print("Will ignore incompatible external weights and train from scratch.")
    except Exception as e:
        print(
            "Failed to inspect external weights; will train from scratch. Error:",
            repr(e),
        )

if os.path.exists(model_path):
    state = torch.load(model_path, map_location="cpu")
    best_model.load_state_dict(state)
    best_model.to(device)
    print("Loaded weights from", model_path)
else:
    print("Training model (no compatible pretrained weights found).")
    optimizer = torch.optim.Adam(best_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_val_qwk = -1e9

    epochs = 14

    for epoch in range(1, epochs + 1):
        best_model.train()
        tr_losses = []
        tr_accs = []
        for x, y in tqdm(
            train_loader, desc=f"epoch {epoch}/{epochs} [train]", leave=False
        ):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = best_model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            tr_losses.append(loss.item())
            tr_accs.append(accuracy_from_logits(logits.detach(), y))

        best_model.eval()
        va_losses = []
        va_accs = []
        va_probs = []
        va_y = []
        with torch.no_grad():
            for x, y in tqdm(
                val_loader, desc=f"epoch {epoch}/{epochs} [val]", leave=False
            ):
                x = x.to(device, non_blocking=True)
                y_dev = y.to(device, non_blocking=True)

                logits = best_model(x)
                loss = criterion(logits, y_dev)
                va_losses.append(loss.item())
                va_accs.append(accuracy_from_logits(logits, y_dev))

                logits2 = best_model(x.flip(dims=(-1,)))
                probs = 0.5 * (F.softmax(logits, dim=-1) + F.softmax(logits2, dim=-1))
                va_probs.append(probs.detach().cpu().numpy())
                va_y.append(y.numpy())

        mean_tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        mean_va_loss = float(np.mean(va_losses)) if va_losses else float("nan")
        mean_tr_acc = float(np.mean(tr_accs)) if tr_accs else float("nan")
        mean_va_acc = float(np.mean(va_accs)) if va_accs else float("nan")

        va_probs = np.concatenate(va_probs, axis=0)
        va_y = np.concatenate(va_y, axis=0).astype(int)

        va_x = probs_to_expected_value(va_probs)
        thr, thr_qwk = find_best_thresholds(va_x, va_y)
        print(
            f"epoch {epoch}: train_loss={mean_tr_loss:.4f} train_acc={mean_tr_acc:.4f} "
            f"val_loss={mean_va_loss:.4f} val_acc={mean_va_acc:.4f} val_qwk={thr_qwk:.4f}"
        )

        if thr_qwk > best_val_qwk:
            best_val_qwk = thr_qwk
            best_thr_from_training = thr.copy()
            torch.save(best_model.state_dict(), model_path)
            print(
                "Saved new best weights to",
                model_path,
                "val_qwk=",
                best_val_qwk,
                "thr=",
                best_thr_from_training,
            )

    state = torch.load(model_path, map_location="cpu")
    best_model.load_state_dict(state)
    best_model.to(device)
    print("Reloaded best weights from", model_path)




## === cell 6
best_model.eval()
va_probs = []
va_y = []
with torch.no_grad():
    for x, y in tqdm(
        val_loader, total=len(val_loader), desc="val inference (for QWK thr)"
    ):
        x = x.to(device, non_blocking=True)
        logits1 = best_model(x)
        logits2 = best_model(x.flip(dims=(-1,)))
        probs = 0.5 * (F.softmax(logits1, dim=-1) + F.softmax(logits2, dim=-1))
        va_probs.append(probs.detach().cpu().numpy())
        va_y.append(y.numpy())

va_probs = np.concatenate(va_probs, axis=0)
va_y = np.concatenate(va_y, axis=0).astype(int)

va_pred_argmax = np.argmax(va_probs, axis=1).astype(int)
print("val QWK (argmax):", qwk(va_y, va_pred_argmax))

va_x = probs_to_expected_value(va_probs)

if best_thr_from_training is not None:
    best_thr = np.asarray(best_thr_from_training, dtype=np.float32)
    va_pred_thr = apply_thresholds(va_x, best_thr).astype(int)
    best_thr_qwk = qwk(va_y, va_pred_thr)
    print("val QWK (expected value + best saved thr):", best_thr_qwk, "thr:", best_thr)
else:
    best_thr, best_thr_qwk = find_best_thresholds(va_x, va_y)
    va_pred_thr = apply_thresholds(va_x, best_thr).astype(int)
    print(
        "val QWK (expected value + tuned thr):",
        qwk(va_y, va_pred_thr),
        "thr:",
        best_thr,
    )




## === cell 7
from tqdm import tqdm

all_pred = []
best_model.eval()
with torch.no_grad():
    for x, _ in tqdm(test_loader, total=len(test_loader), desc="inference"):
        x = x.to(device, non_blocking=True)
        y_pred1 = best_model(x)
        y_pred2 = best_model(x.flip(dims=(-1,)))
        curr_prob = 0.5 * (F.softmax(y_pred1, dim=-1) + F.softmax(y_pred2, dim=-1))
        curr_prob = curr_prob.detach().cpu().numpy()

        curr_x = probs_to_expected_value(curr_prob)
        curr_cls = apply_thresholds(curr_x, best_thr).astype(int)
        all_pred.extend(curr_cls.tolist())

print("num predictions:", len(all_pred), "expected:", len(df_test))




## === cell 8
sample_sub = pd.read_csv(os.path.join(COMP_ROOT, "sample_submission.csv"))
sample_sub["id_code"] = sample_sub["id_code"].map(normalize_id)

test_ids = df_test["id_code"].tolist()
if len(all_pred) != len(test_ids):
    raise ValueError(
        f"Prediction length {len(all_pred)} != test length {len(test_ids)}"
    )

pred_arr = np.asarray(all_pred, dtype=int)
if pred_arr.min() < 0 or pred_arr.max() > 4:
    raise ValueError(
        f"Predictions out of range [0,4]: min={pred_arr.min()} max={pred_arr.max()}"
    )

if set(sample_sub["id_code"].tolist()) != set(test_ids):
    only_in_sample = sorted(set(sample_sub["id_code"].tolist()) - set(test_ids))[:5]
    only_in_test = sorted(set(test_ids) - set(sample_sub["id_code"].tolist()))[:5]
    raise ValueError(
        "ID set mismatch between sample_submission.csv and test.csv after normalization. "
        f"only_in_sample(first5)={only_in_sample} only_in_test(first5)={only_in_test}"
    )

sub = pd.DataFrame({"id_code": test_ids, "diagnosis": pred_arr.astype(int)})
if sub["diagnosis"].isna().any():
    raise ValueError("NaN diagnoses in submission (unexpected).")
if sub.shape[0] != df_test.shape[0]:
    raise ValueError(f"Submission rows {sub.shape[0]} != test rows {df_test.shape[0]}")
if sub["id_code"].duplicated().any():
    dups = sub.loc[sub["id_code"].duplicated(), "id_code"].head(5).tolist()
    raise ValueError(f"Duplicate id_code in submission; first few: {dups}")

submission_path = os.path.join(WORKING_DIR, "submission.csv")
sub.to_csv(submission_path, index=False)

if os.path.abspath(submission_path) != os.path.abspath("submission.csv"):
    sub.to_csv("submission.csv", index=False)

if not os.path.exists(submission_path):
    raise FileNotFoundError(f"submission.csv was not created at: {submission_path}")
if os.path.getsize(submission_path) < 50:
    raise ValueError(
        f"submission.csv seems too small: {os.path.getsize(submission_path)} bytes"
    )

check = pd.read_csv(submission_path)
if list(check.columns) != ["id_code", "diagnosis"]:
    raise ValueError(f"Bad submission columns: {check.columns.tolist()}")
if check.shape[0] != df_test.shape[0]:
    raise ValueError(
        f"Bad submission row count: {check.shape[0]} expected {df_test.shape[0]}"
    )

print(check.head())
print("Wrote submission.csv with shape:", check.shape, "to:", submission_path)
print("Unique diagnoses:", np.unique(check["diagnosis"].values, return_counts=True))




## === cell 9
_ = sub.hist()




## === cell 10
tr = pd.read_csv(os.path.join(COMP_ROOT, "train.csv"))
_ = tr.hist()
