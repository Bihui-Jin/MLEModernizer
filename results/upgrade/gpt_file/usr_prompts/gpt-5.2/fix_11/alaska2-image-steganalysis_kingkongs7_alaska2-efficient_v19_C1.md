# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.10

# 3. Installed packages

albumentations==2.0.8
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
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Code solution

## === cell 0
import os
import random
import gc
import time
from glob import glob

import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
from tqdm.auto import tqdm
from sklearn import metrics

import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 1
seed = 42
print(f"setting everything to seed {seed}")
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

torch.backends.cudnn.benchmark = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    cv2.setNumThreads(max(1, os.cpu_count() or 1))
except Exception:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)



## === cell 2
data_dir = "../input/alaska2-image-steganalysis"

sample_size = int(
    os.environ.get("ALASKA2_SAMPLE_SIZE", "12000")
)  # per class; adjust via env var if needed
val_size = int(sample_size * 0.25)

train_fn, val_fn = [], []
train_labels, val_labels = [], []

folder_names = ["Cover/", "JMiPOD/", "JUNIWARD/", "UERD/"]  # label 0 1 2 3

for label, folder in enumerate(folder_names):
    train_filenames = sorted(glob(f"{data_dir}/{folder}/*.jpg")[:sample_size])
    np.random.shuffle(train_filenames)
    train_fn.extend(train_filenames[val_size:])
    train_labels.extend(np.zeros(len(train_filenames[val_size:])) + label)
    val_fn.extend(train_filenames[:val_size])
    val_labels.extend(np.zeros(len(train_filenames[:val_size])) + label)

assert len(train_labels) == len(train_fn), "wrong labels"
assert len(val_labels) == len(val_fn), "wrong labels"

train_df = pd.DataFrame(
    {"ImageFileName": train_fn, "Label": train_labels},
    columns=["ImageFileName", "Label"],
)
train_df["Label"] = train_df["Label"].astype(int)
val_df = pd.DataFrame(
    {"ImageFileName": val_fn, "Label": val_labels}, columns=["ImageFileName", "Label"]
)
val_df["Label"] = val_df["Label"].astype(int)

print(train_df.head())
_ = train_df.Label.hist()



## === cell 3
img_size = 512

BASE_PREPROCESS = A.Compose(
    [
        A.Resize(img_size, img_size, p=1),
        A.ToFloat(max_value=255.0),
    ],
    p=1,
)

TRAIN_STOCHASTIC = A.Compose(
    [
        A.VerticalFlip(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_range=(75, 100), p=0.5),
    ],
    p=1,
)

TO_TENSOR = ToTensorV2()
AUGMENTATIONS_TRAIN = None
AUGMENTATIONS_TEST = None

USE_BASE_CACHE = False

CACHE_DIR = os.environ.get("ALASKA2_CACHE_DIR", "")  # e.g. "/kaggle/working/base_cache"
CACHE_IN_RAM = os.environ.get("ALASKA2_CACHE_IN_RAM", "1") == "1"
if CACHE_DIR:
    os.makedirs(CACHE_DIR, exist_ok=True)


def _base_preprocess_cv2_rgb(
    im_rgb: np.ndarray, out_size: int = img_size
) -> np.ndarray:
    im = cv2.resize(im_rgb, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return im.astype(np.float32) * (1.0 / 255.0)


_base_cache_ram = {} if CACHE_IN_RAM else None


def _cache_key_from_path(fn: str) -> str:
    return os.path.basename(fn)


def _load_or_compute_base(fn: str) -> np.ndarray:
    key = _cache_key_from_path(fn)

    if _base_cache_ram is not None:
        arr = _base_cache_ram.get(key, None)
        if arr is not None:
            return arr

    cache_path = (
        os.path.join(CACHE_DIR, key.replace(".jpg", f"_{img_size}.npy"))
        if CACHE_DIR
        else ""
    )
    if cache_path and os.path.exists(cache_path):
        arr = np.load(cache_path, allow_pickle=False)
    else:
        im = cv2.imread(fn, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if im is None:
            raise FileNotFoundError(f"Could not read image: {fn}")
        im = im[:, :, ::-1]  # BGR -> RGB
        arr = _base_preprocess_cv2_rgb(im, img_size)
        if cache_path:
            np.save(cache_path, arr, allow_pickle=False)

    if _base_cache_ram is not None:
        _base_cache_ram[key] = arr
    return arr




## === cell 4
class Alaska2Dataset(Dataset):
    def __init__(self, df, augmentations=None):
        df = df.reset_index(drop=True)
        self.fns = df["ImageFileName"].to_numpy()
        self.labels = df["Label"].to_numpy(dtype=np.int64, copy=False)
        self.augment = augmentations

        self._imread_flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        fn = self.fns[idx]
        label = int(self.labels[idx])

        im = _load_or_compute_base(fn)  # float32 HWC in [0,1], RGB

        if self.augment is not None:
            im = self.augment(image=im)["image"]
        im = TO_TENSOR(image=im)["image"]  # CHW float32 tensor

        return im, label




## === cell 5
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
        self.model = torchvision.models.efficientnet_b0(weights=weights)
        self.model.classifier = nn.Identity()  # produce 1280-d embedding
        self.dense_output = nn.Linear(1280, 4)

    def forward(self, x):
        feat = self.model(x)  # [B, 1280]
        return self.dense_output(feat)




## === cell 6
batch_size = int(os.environ.get("ALASKA2_BATCH_SIZE", "32"))

cpu_cnt = os.cpu_count() or 1
num_workers = int(os.environ.get("ALASKA2_NUM_WORKERS", str(min(12, cpu_cnt))))

train_dataset = Alaska2Dataset(train_df, augmentations=TRAIN_STOCHASTIC)
valid_dataset = Alaska2Dataset(val_df, augmentations=None)


def _seed_worker(worker_id):
    worker_seed = (seed + worker_id) % (2**32 - 1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)

_loader_kwargs = dict(
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=(2 if num_workers > 0 else None),
)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=True,
    drop_last=True,
    worker_init_fn=_seed_worker,
    generator=g,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

valid_loader = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=batch_size * 2,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    worker_init_fn=_seed_worker,
    generator=g,
    **{k: v for k, v in _loader_kwargs.items() if v is not None},
)

model = Net().to(device)

if device == "cuda":
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("torch.compile enabled")
    except Exception as e:
        print("torch.compile not enabled:", repr(e))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)




## === cell 7
def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = np.array([0.0, 0.4, 1.0], dtype=np.float64)
    weights = np.array([2.0, 1.0], dtype=np.float64)

    fpr, tpr, _ = metrics.roc_curve(y_true, y_valid, pos_label=1)

    areas = tpr_thresholds[1:] - tpr_thresholds[:-1]
    normalization = float(np.dot(areas, weights))

    competition_metric = 0.0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)
        if not np.any(mask):
            continue

        fpr_m = fpr[mask]
        tpr_m = tpr[mask]

        x_padding = np.linspace(fpr_m[-1], 1.0, 100, dtype=fpr.dtype)
        x = np.concatenate([fpr_m, x_padding])
        y = np.concatenate([tpr_m, np.full_like(x_padding, y_max)])
        y = y - y_min

        competition_metric += float(metrics.auc(x, y)) * float(weight)

    return competition_metric / normalization




## === cell 8
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 2

train_loss, val_loss = [], []

if device == "cuda":
    model = model.to(memory_format=torch.channels_last)

tqdm_mininterval = float(os.environ.get("ALASKA2_TQDM_MININTERVAL", "5.0"))

for epoch in range(num_epochs):
    print("Epoch {}/{}".format(epoch, num_epochs - 1))
    print("-" * 10)
    model.train()
    running_loss = 0.0

    tk0 = tqdm(train_loader, total=int(len(train_loader)), mininterval=tqdm_mininterval)
    for inputs, labels in tk0:
        inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
        if device == "cuda":
            inputs = inputs.contiguous(memory_format=torch.channels_last)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        running_loss += loss_val
        tk0.set_postfix(loss=loss_val)

    epoch_loss = running_loss / max(1, len(train_loader))
    train_loss.append(epoch_loss)
    print("Training Loss: {:.8f}".format(epoch_loss))

    tk1 = tqdm(valid_loader, total=int(len(valid_loader)), mininterval=tqdm_mininterval)
    model.eval()
    running_loss = 0.0

    y_batches, preds_batches = [], []
    with torch.inference_mode():
        for inputs, labels in tk1:
            inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
            if device == "cuda":
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            y_batches.append(labels.detach().cpu().numpy().astype(np.int64, copy=False))
            preds_batches.append(F.softmax(outputs, 1).detach().cpu().numpy())

            loss_val = float(loss.item())
            running_loss += loss_val
            tk1.set_postfix(loss=loss_val)

        epoch_loss = running_loss / max(1, len(valid_loader))
        val_loss.append(epoch_loss)

        y = np.concatenate(y_batches, axis=0)
        preds = np.concatenate(preds_batches, axis=0)

        labels_pred = preds.argmax(1)
        acc = (labels_pred == y).mean() * 100.0

        stego_prob = 1.0 - preds[:, 0]

        y_bin = y.copy()
        y_bin[y_bin != 0] = 1
        auc_score = alaska_weighted_auc(y_bin, stego_prob)
        print(f"Val Loss: {epoch_loss:.3}, Weighted AUC:{auc_score:.3}, Acc: {acc:.3}")

    torch.save(
        model.state_dict(),
        f"epoch_{epoch}_val_loss_{epoch_loss:.3}_auc_{auc_score:.3}.pth",
    )



## === cell 9
if len(train_loss) > 0 or len(val_loss) > 0:
    plt.figure(figsize=(15, 7))
    plt.plot(train_loss, c="r")
    plt.plot(val_loss, c="b")
    plt.legend(["train_loss", "val_loss"])
    plt.title("Loss Plot")
    plt.show()




## === cell 10
class Alaska2TestDataset(Dataset):
    def __init__(self, df, augmentations=None):
        df = df.reset_index(drop=True)
        self.fns = df["ImageFileName"].to_numpy()
        self.augment = augmentations
        self._imread_flags = cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION

    def __len__(self):
        return self.fns.shape[0]

    def __getitem__(self, idx):
        fn = self.fns[idx]

        im = _load_or_compute_base(fn)  # float32 HWC in [0,1], RGB

        if self.augment is not None:
            im = self.augment(image=im)["image"]
        im = TO_TENSOR(image=im)["image"]
        return im


test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_df = pd.DataFrame(
    {"ImageFileName": list(test_filenames)}, columns=["ImageFileName"]
)

batch_size = int(os.environ.get("ALASKA2_TEST_BATCH_SIZE", "64"))
cpu_cnt = os.cpu_count() or 1
num_workers = int(os.environ.get("ALASKA2_TEST_NUM_WORKERS", str(min(12, cpu_cnt))))

test_dataset = Alaska2TestDataset(test_df, augmentations=None)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=2 if num_workers > 0 else None,
    worker_init_fn=_seed_worker,
    generator=g,
)

print("test images:", len(test_df))

model.eval()

preds = []
tk0 = tqdm(
    test_loader,
    total=len(test_loader),
    mininterval=float(os.environ.get("ALASKA2_TQDM_MININTERVAL", "5.0")),
)
with torch.inference_mode():
    for inputs in tk0:
        inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
        if device == "cuda":
            inputs = inputs.contiguous(memory_format=torch.channels_last)

        x = torch.stack([inputs, inputs.flip(2), inputs.flip(3)], dim=0)  # [3,B,C,H,W]
        x = x.flatten(0, 1)  # [3B,C,H,W]
        out = model(x).view(3, inputs.shape[0], -1)  # [3,B,4]
        outputs = (0.5 * out[0]) + (0.25 * out[1]) + (0.25 * out[2])

        preds.append(F.softmax(outputs, 1).cpu().numpy())

preds = np.concatenate(preds, axis=0)

stego_prob = 1.0 - preds[:, 0]

test_df["Id"] = test_df["ImageFileName"].map(os.path.basename)
test_df["Label"] = stego_prob.astype(np.float32)
sub_df = test_df.drop("ImageFileName", axis=1)

sample_path = os.path.join(data_dir, "sample_submission.csv")
if os.path.exists(sample_path):
    sample_sub = pd.read_csv(sample_path)
    sub_df = sample_sub[["Id"]].merge(sub_df, on="Id", how="left")
    sub_df["Label"] = sub_df["Label"].fillna(
        float(np.nanmedian(sub_df["Label"].values))
    )

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print("submission.csv columns:", list(sub_df.columns))
assert list(sub_df.columns) == ["Id", "Label"]
assert sub_df.shape[0] == 5000
assert sub_df["Id"].isna().sum() == 0
assert sub_df["Label"].isna().sum() == 0
