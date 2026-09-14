# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import sys, os

print("Python:", sys.version)
print("Kaggle working dir:", os.getcwd())



## === cell 1
import os
import random
import numpy as np
import pandas as pd
import timm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm



## === cell 2
path = "../input/cassava-leaf-disease-classification"
os.listdir(path)



## === cell 3
df = pd.read_csv(path + "/train.csv")



## === cell 4
df.head()



## === cell 5
df["path"] = path + "/train_images/" + df["image_id"].astype(str)
df = df.drop(columns=["image_id"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)



## === cell 6
df.head()



## === cell 7
from sklearn import model_selection

train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.label.values
)



## === cell 8
pass



## === cell 9
pass



## === cell 10
train_df = train_df.reset_index().drop(columns=["index"])
train_df.head()



## === cell 11
valid_df = valid_df.reset_index().drop(columns=["index"])
valid_df.head()



## === cell 12
pass



## === cell 13
pass



## === cell 14
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset

from sklearn.model_selection import KFold

import matplotlib.image as img



## === cell 15
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # must be False for true determinism

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // 2))
    torch.set_num_interop_threads(1)
except Exception:
    pass



## === cell 16
_CPU = os.cpu_count() or 2
DL_NUM_WORKERS = max(
    2, min(4, _CPU)
)  # was up to 8; 4 is typically faster/stabler on Kaggle
DL_PREFETCH_FACTOR = 4

try:
    from PIL import ImageFile

    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    pass



## === cell 17
from collections import OrderedDict
from io import BytesIO

try:
    from torchvision.io import read_image as _tv_read_image
except Exception:
    _tv_read_image = None

_GLOBAL_CACHE_BYTES_LIMIT = (
    1024 * 1024 * 1024
)  # 1GB per process upper bound (best-effort)


class _LRUBytesCache:
    def __init__(self, max_bytes=_GLOBAL_CACHE_BYTES_LIMIT):
        self.max_bytes = int(max_bytes)
        self._od = OrderedDict()
        self._nbytes = 0

    def get(self, k):
        v = self._od.get(k, None)
        if v is not None:
            self._od.move_to_end(k)
        return v

    def put(self, k, v: bytes):
        if v is None:
            return
        vb = len(v)
        old = self._od.pop(k, None)
        if old is not None:
            self._nbytes -= len(old)
        self._od[k] = v
        self._nbytes += vb
        while self._nbytes > self.max_bytes and len(self._od) > 1:
            _, ev = self._od.popitem(last=False)
            self._nbytes -= len(ev)


_BYTES_CACHE = _LRUBytesCache(max_bytes=_GLOBAL_CACHE_BYTES_LIMIT)




## === cell 18
class _EpochState:
    def __init__(self, initial_epoch: int = 0):
        self.epoch = initial_epoch


_EPOCH_STATE = _EpochState(0)


def set_dataset_epoch(epoch: int):
    _EPOCH_STATE.epoch = int(epoch)




## === cell 19
import torchvision.transforms.v2 as T2




## === cell 20
class CassavaDataset(Dataset):
    def __init__(
        self, dataframe, transform=None, bytes_cache=None, base_size=None, seed=42
    ):
        super().__init__()
        dfr = dataframe.reset_index(drop=True)
        self.paths = dfr["path"].tolist()
        self.labels = dfr["label"].astype(int).tolist()
        self.transform = transform
        self.bytes_cache = bytes_cache
        self.base_size = base_size  # kept for API compatibility; no longer used
        self.seed = int(seed)

    def __len__(self):
        return len(self.paths)

    def _load_image_tensor(self, path: str) -> torch.Tensor:
        if _tv_read_image is not None:
            return _tv_read_image(path)  # uint8 [C,H,W] RGB
        if self.bytes_cache is not None:
            b = self.bytes_cache.get(path)
            if b is None:
                with open(path, "rb") as f:
                    b = f.read()
                self.bytes_cache.put(path, b)
            im = Image.open(BytesIO(b)).convert("RGB")
        else:
            with open(path, "rb") as f:
                im = Image.open(f).convert("RGB")
        return T2.ToImage()(im)  # CHW uint8 tensor

    def __getitem__(self, index):
        path = self.paths[index]
        label = int(self.labels[index])
        image = self._load_image_tensor(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label




## === cell 21
import random




## === cell 22
class make_mask_image:
    def __init__(self, p, mask_size=50):
        self.p = p
        self.mask_size = mask_size

    def __call__(self, image):
        start_width, start_height = [], []
        if random.random() < self.p:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            for _ in range(10):
                start_width.append(random.randrange(0, width - self.mask_size))
                start_height.append(random.randrange(0, height - self.mask_size))
            for x, y in zip(start_width, start_height):
                draw.rectangle(
                    (x, y, x + self.mask_size, y + self.mask_size),
                    fill=(0, 0, 0),
                    outline=(0, 0, 0),
                )
        return image




## === cell 23
image_size = 512
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = T2.Compose(
    [
        T2.RandomHorizontalFlip(p=0.5),
        T2.RandomVerticalFlip(p=0.5),
        T2.RandomResizedCrop(size=(image_size, image_size)),
        T2.ToDtype(torch.float32, scale=True),
        T2.Normalize(mean=mean, std=std),
    ]
)

valid_transform = T2.Compose(
    [
        T2.Resize(size=(image_size, image_size)),
        T2.ToDtype(torch.float32, scale=True),
        T2.Normalize(mean=mean, std=std),
    ]
)



## === cell 24
_BASE_PIL_SIZE = None

dataset = CassavaDataset(
    train_df,
    train_transform,
    bytes_cache=_BYTES_CACHE,
    base_size=_BASE_PIL_SIZE,
    seed=SEED,
)



## === cell 25
pass



## === cell 26
import json

map_path = "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
with open(map_path, mode="r") as f:
    label_to_name = json.load(f)




## === cell 27
class Unnormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor




## === cell 28
unnorm = Unnormalize(mean, std)




## === cell 29
def display_img(img, unnorm=None, label=None):
    if unnorm is not None:
        img = unnorm(img)

    plt.imshow(img.permute(1, 2, 0))

    if label is not None:
        plt.title(label_to_name[str(label)])




## === cell 30
def display_batch(batch, unnorm=None):
    imgs, labels = batch

    if unnorm:
        unnorm_imgs = []
        for img in imgs:
            unnorm_imgs.append(unnorm(img))
        imgs = unnorm_imgs

    ig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(imgs, nrow=8).permute(1, 2, 0))




## === cell 31
pass



## === cell 32
pass



## === cell 33
import torch
import torch.nn as nn
import torch.nn.functional as F



## === cell 34
epoch = 3
batch_size = 16
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 35
resNet = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
resNet = resNet.to(device)



## === cell 36
ef_model = timm.create_model(
    "tf_efficientnet_b2_ns", pretrained=True, num_classes=num_classes
)
ef_model = ef_model.to(device)



## === cell 37
ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=0.0001)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=0.0001)
resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)
criterion = nn.CrossEntropyLoss()




## === cell 38
class CassavaEvalDataset(Dataset):
    def __init__(self, dataframe, transform, bytes_cache=None, base_size=None):
        super().__init__()
        dfr = dataframe.reset_index(drop=True)
        self.paths = dfr["path"].tolist()
        self.labels = dfr["label"].astype(int).tolist()
        self.transform = transform
        self.bytes_cache = bytes_cache
        self.base_size = base_size  # kept for API compatibility; unused

    def __len__(self):
        return len(self.paths)

    def _load_image_tensor(self, path: str) -> torch.Tensor:
        if _tv_read_image is not None:
            return _tv_read_image(path)
        if self.bytes_cache is not None:
            b = self.bytes_cache.get(path)
            if b is None:
                with open(path, "rb") as f:
                    b = f.read()
                self.bytes_cache.put(path, b)
            im = Image.open(BytesIO(b)).convert("RGB")
        else:
            with open(path, "rb") as f:
                im = Image.open(f).convert("RGB")
        return T2.ToImage()(im)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = int(self.labels[idx])
        im = self._load_image_tensor(p)
        t = self.transform(im)
        return t, y


_EVAL_LOADER_CACHE = {}


def _make_eval_loader(cache_key, df):
    ds = CassavaEvalDataset(
        df,
        valid_transform,
        bytes_cache=_BYTES_CACHE,
        base_size=_BASE_PIL_SIZE,
    )
    dl = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=DL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(DL_NUM_WORKERS > 0),
        prefetch_factor=DL_PREFETCH_FACTOR if DL_NUM_WORKERS > 0 else None,
    )
    return dl


def calc_correction(model, df):
    model.eval()
    key = ("valid", len(df))
    dl = _EVAL_LOADER_CACHE.get(key)
    if dl is None:
        dl = _make_eval_loader(key, df)
        _EVAL_LOADER_CACHE[key] = dl

    correct = 0
    total = 0
    pred_list = [0, 0, 0, 0, 0]

    with torch.no_grad():
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            out = model(xb)
            preds = out.argmax(1)
            correct += (preds == yb).sum().item()
            total += yb.numel()

            binc = torch.bincount(preds.detach().cpu(), minlength=5).tolist()
            for i in range(5):
                pred_list[i] += int(binc[i])

    return correct / max(1, total), pred_list




## === cell 39
from matplotlib import pyplot as plt


def plot_losses(epoch, title, train_losses, valid_losses):
    y = list(range(len(train_losses)))
    train_loss = plt.plot(y, train_losses)
    valid_loss = plt.plot(y, valid_losses)
    plt.title(title)
    plt.ylabel("loss")
    plt.legend(
        (train_loss[0], valid_loss[0]),
        ("train loss", "valid loss"),
    )
    plt.show()




## === cell 40
import time
import copy
from torch.utils.data import SubsetRandomSampler, SequentialSampler


def _seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)




## === cell 41
def _maybe_optimize_model(m: torch.nn.Module) -> torch.nn.Module:
    if device.type == "cuda":
        try:
            m = m.to(memory_format=torch.channels_last)
        except Exception:
            pass
    if not getattr(m, "_compiled", False):
        try:
            m = torch.compile(m, mode="reduce-overhead", fullgraph=False)
            setattr(m, "_compiled", True)
        except Exception:
            pass
    return m


resNet = _maybe_optimize_model(resNet)
ef_model = _maybe_optimize_model(ef_model)



## === cell 42
_FOLD_LOADER_CACHE = {}


def train_model(
    model, dataset, batch_size, optimizer, criterion, scheduler, epoch, model_title
):
    best_loss = float("inf")
    best_ckpt_path = model_title + ".best_tmp.pth"
    train_losses, valid_losses = [], []

    kf = KFold(n_splits=4, shuffle=True, random_state=SEED)

    g = torch.Generator()
    g.manual_seed(SEED)

    for fold, (train_index, valid_index) in enumerate(kf.split(range(len(dataset)))):
        print("fold: ", fold)

        cache_key = (id(dataset), fold, batch_size, DL_NUM_WORKERS)
        cached = _FOLD_LOADER_CACHE.get(cache_key)
        if cached is None:
            train_sampler = SubsetRandomSampler(train_index, generator=g)

            train_loader = DataLoader(
                dataset,
                batch_size,
                sampler=train_sampler,
                shuffle=False,
                num_workers=DL_NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=(DL_NUM_WORKERS > 0),
                prefetch_factor=DL_PREFETCH_FACTOR if DL_NUM_WORKERS > 0 else None,
                worker_init_fn=_seed_worker if DL_NUM_WORKERS > 0 else None,
            )

            valid_dataset = Subset(dataset, valid_index)
            valid_loader = DataLoader(
                valid_dataset,
                batch_size,
                shuffle=False,
                num_workers=DL_NUM_WORKERS,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=(DL_NUM_WORKERS > 0),
                prefetch_factor=DL_PREFETCH_FACTOR if DL_NUM_WORKERS > 0 else None,
                worker_init_fn=_seed_worker if DL_NUM_WORKERS > 0 else None,
            )
            _FOLD_LOADER_CACHE[cache_key] = (
                train_loader,
                valid_loader,
                len(train_index),
                len(valid_index),
            )
        else:
            train_loader, valid_loader, n_train, n_valid = cached

        if cached is None:
            n_train = len(train_index)
            n_valid = len(valid_index)

        for ep in range(1, epoch + 1):
            set_dataset_epoch(ep + fold * 100)

            epoch_start_time = time.time()
            acc = []
            train_loss_sum = 0.0
            valid_loss_sum = 0.0

            model.train()
            for data, target in train_loader:
                data = data.to(device, non_blocking=True)
                if device.type == "cuda":
                    try:
                        data = data.to(memory_format=torch.channels_last)
                    except Exception:
                        pass
                target = target.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                bs = data.size(0)
                train_loss_sum += loss.item() * bs

            train_loss = train_loss_sum / n_train
            train_losses.append(train_loss)

            model.eval()
            with torch.no_grad():
                for data, target in valid_loader:
                    data = data.to(device, non_blocking=True)
                    if device.type == "cuda":
                        try:
                            data = data.to(memory_format=torch.channels_last)
                        except Exception:
                            pass
                    target = target.to(device, non_blocking=True)
                    output = model(data)
                    pred = output.argmax(1) == target
                    acc.append((pred.float().mean()).item())
                    loss = criterion(output, target)
                    bs = data.size(0)
                    valid_loss_sum += loss.item() * bs

            if valid_loss_sum < best_loss:
                best_loss = valid_loss_sum
                torch.save(model.state_dict(), best_ckpt_path)

            scheduler.step()

            collection = sum(acc) / max(1, len(acc))
            valid_loss = valid_loss_sum / n_valid
            valid_losses.append(valid_loss)
            print(
                "Time: {:.3f}\t Epoch: {} \tTraining Loss: {:.3f} \tValidation Loss: {:.3f} \t Acc: {:.2f}".format(
                    time.time() - epoch_start_time,
                    ep,
                    train_loss,
                    valid_loss,
                    collection,
                )
            )

    if os.path.exists(best_ckpt_path):
        model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
        try:
            os.remove(best_ckpt_path)
        except Exception:
            pass

    torch.save(model.state_dict(), model_title)
    return model, train_losses, valid_losses




## === cell 43
def train_models(resNet, ef_model):
    model_title = "./res_model.pth"
    resNet, train_losses, valid_losses = train_model(
        resNet,
        dataset,
        batch_size,
        resNet_optimizer,
        criterion,
        resNet_scheduler,
        epoch,
        model_title,
    )
    print(calc_correction(resNet, valid_df))

    model_title = "./ef_model.pth"
    ef_model, train_losses, valid_losses = train_model(
        ef_model,
        dataset,
        batch_size,
        ef_optimizer,
        criterion,
        ef_scheduler,
        epoch,
        model_title,
    )
    print(calc_correction(ef_model, valid_df))




## === cell 44
if not (os.path.exists("./ef_model.pth") and os.path.exists("./res_model.pth")):
    train_models(resNet, ef_model)



## === cell 45
if os.path.exists("./ef_model.pth"):
    ef_model.load_state_dict(torch.load("./ef_model.pth", map_location=device))
if os.path.exists("./res_model.pth"):
    resNet.load_state_dict(torch.load("./res_model.pth", map_location=device))

resNet = _maybe_optimize_model(resNet)
ef_model = _maybe_optimize_model(ef_model)




## === cell 46
class CassaveClassifier(nn.Module):
    def __init__(self, model, ef_model):
        super().__init__()
        self.model = model
        self.ef_model = ef_model

    def forward(self, x):
        x1 = self.model(x)
        x2 = self.ef_model(x)
        return 0.5 * x1 + 0.5 * x2

    def test(self, x, rate):
        x1 = self.model(x)
        x2 = self.ef_model(x)
        p = rate * x1 + (1 - rate) * x2
        return p




## === cell 47
classifier = CassaveClassifier(resNet, ef_model)
classifier = classifier.to(device)
classifier = _maybe_optimize_model(classifier)




## === cell 48
def test_rate():
    classifier.eval()
    eval_ds = CassavaEvalDataset(
        valid_df,
        valid_transform,
        bytes_cache=_BYTES_CACHE,
        base_size=_BASE_PIL_SIZE,
    )
    eval_dl = DataLoader(
        eval_ds,
        batch_size=64,
        shuffle=False,
        num_workers=DL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(DL_NUM_WORKERS > 0),
        prefetch_factor=DL_PREFETCH_FACTOR if DL_NUM_WORKERS > 0 else None,
        worker_init_fn=_seed_worker if DL_NUM_WORKERS > 0 else None,
    )

    for rate in range(1, 10):
        count = 0
        total = 0
        pred_list = [0, 0, 0, 0, 0]
        with torch.no_grad():
            for xb, yb in eval_dl:
                xb = xb.to(device, non_blocking=True)
                if device.type == "cuda":
                    try:
                        xb = xb.to(memory_format=torch.channels_last)
                    except Exception:
                        pass
                yb = yb.to(device, non_blocking=True)
                out = classifier.test(xb, rate / 10).argmax(1)
                count += (out == yb).sum().item()
                total += yb.numel()
                binc = torch.bincount(out.detach().cpu(), minlength=5).tolist()
                for i in range(5):
                    pred_list[i] += int(binc[i])

        percent = count / max(1, total)
        print("rate: ", rate / 10)
        print("percent: ", percent)




## === cell 49
pass



## === cell 50
pass



## === cell 51
test_dir = "../input/cassava-leaf-disease-classification/test_images/"



## === cell 52
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)

image_ids = sample_sub["image_id"].tolist()
image_paths = [os.path.join(test_dir, img_id) for img_id in image_ids]

missing = [p for p in image_paths if not os.path.isfile(p)]
print("Missing test files:", len(missing))




## === cell 53
class CassavaTestDataset(Dataset):
    def __init__(self, image_paths, transform, bytes_cache=None, base_size=None):
        self.image_paths = list(image_paths)
        self.transform = transform
        self.bytes_cache = bytes_cache
        self.base_size = base_size  # kept for API compatibility; unused

    def __len__(self):
        return len(self.image_paths)

    def _load_image_tensor(self, path: str) -> torch.Tensor:
        if _tv_read_image is not None:
            return _tv_read_image(path)
        if self.bytes_cache is not None:
            b = self.bytes_cache.get(path)
            if b is None:
                with open(path, "rb") as f:
                    b = f.read()
                self.bytes_cache.put(path, b)
            im = Image.open(BytesIO(b)).convert("RGB")
        else:
            with open(path, "rb") as f:
                im = Image.open(f).convert("RGB")
        return T2.ToImage()(im)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        im = self._load_image_tensor(p)
        t = self.transform(im)
        return t


classifier.eval()
test_ds = CassavaTestDataset(
    image_paths,
    valid_transform,
    bytes_cache=_BYTES_CACHE,
    base_size=_BASE_PIL_SIZE,
)
test_dl = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=DL_NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(DL_NUM_WORKERS > 0),
    prefetch_factor=DL_PREFETCH_FACTOR if DL_NUM_WORKERS > 0 else None,
    worker_init_fn=_seed_worker if DL_NUM_WORKERS > 0 else None,
)

pred = []
with torch.no_grad():
    for xb in test_dl:
        xb = xb.to(device, non_blocking=True)
        if device.type == "cuda":
            try:
                xb = xb.to(memory_format=torch.channels_last)
            except Exception:
                pass
        out = classifier(xb).argmax(1)
        pred.extend(out.detach().cpu().tolist())



## === cell 54
pred[:10], len(pred), len(image_ids)



## === cell 55
sub = pd.DataFrame({"image_id": image_ids, "label": pred})
sub.head()



## === cell 56
sub.shape



## === cell 57
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
