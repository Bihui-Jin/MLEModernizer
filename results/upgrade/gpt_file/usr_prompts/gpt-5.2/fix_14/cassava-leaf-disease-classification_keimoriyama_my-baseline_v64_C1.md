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
import os
import pandas as pd
import timm
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm



## === cell 1
path = "../input/cassava-leaf-disease-classification"
os.listdir(path)[:10]



## === cell 2
df = pd.read_csv(path + "/train.csv")



## === cell 3
df.head()



## === cell 4
df["path"] = path + "/train_images/" + df["image_id"].astype(str)
df = df.drop(columns=["image_id"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)



## === cell 5
df.head()



## === cell 6
from sklearn import model_selection

train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.label.values
)



## === cell 7
train_df.label.value_counts().plot(kind="bar")



## === cell 8
valid_df.label.value_counts().plot(kind="bar")



## === cell 9
train_df = train_df.reset_index().drop(columns=["index"])
train_df.head()



## === cell 10
valid_df = valid_df.reset_index().drop(columns=["index"])
valid_df.head()



## === cell 11
im = Image.open(train_df["path"][0])



## === cell 12
im



## === cell 13
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import Subset

from sklearn.model_selection import KFold

import matplotlib.image as img



## === cell 14
pass



## === cell 15
from functools import lru_cache
import torch.utils.data as tud


def _cached_pil_rgb(path: str):
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


def _get_worker_cache(maxsize: int = 4096):
    info = tud.get_worker_info()
    if info is None:
        global _MAIN_PROC_PIL_CACHE
        try:
            return _MAIN_PROC_PIL_CACHE
        except NameError:
            _MAIN_PROC_PIL_CACHE = lru_cache(maxsize=maxsize)(_cached_pil_rgb)
            return _MAIN_PROC_PIL_CACHE
    ds = info.dataset
    cache = getattr(ds, "_pil_cache", None)
    if cache is None:
        cache = lru_cache(maxsize=maxsize)(_cached_pil_rgb)
        setattr(ds, "_pil_cache", cache)
    return cache




## === cell 16
class CassavaDataset(Dataset):
    def __init__(
        self, dataframe, transform=None, cache_images=True, cache_max_items=2048
    ):
        super().__init__()
        df = dataframe.reset_index(drop=True)
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform
        self.cache_images = bool(cache_images)
        self.cache_max_items = int(cache_max_items)

    def __len__(self):
        return len(self.paths)

    def _load_pil_rgb(self, path):
        if self.cache_images:
            cache = _get_worker_cache(maxsize=self.cache_max_items)
            return cache(path)
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")

    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        image = self._load_pil_rgb(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label




## === cell 17
import random




## === cell 18
class make_mask_image:
    def __init__(self, p, mask_size=50):
        self.p = p
        self.mask_size = mask_size

    def __call__(self, image):
        start_width, start_height = [], []
        if random.random() < self.p:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            for i in range(10):
                start_width.append(random.randrange(0, max(1, width - self.mask_size)))
                start_height.append(
                    random.randrange(0, max(1, height - self.mask_size))
                )
            for x, y in zip(start_width, start_height):
                draw.rectangle(
                    (x, y, x + self.mask_size, y + self.mask_size),
                    fill=(0, 0, 0),
                    outline=(0, 0, 0),
                )
        return image




## === cell 19
from torchvision.transforms import v2 as T

image_size = 512
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = T.Compose(
    [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomResizedCrop(image_size),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=mean, std=std),
    ]
)

valid_transform = T.Compose(
    [
        T.Resize((image_size + 32, image_size + 32)),
        T.CenterCrop((image_size, image_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=mean, std=std),
    ]
)



## === cell 20
train_dataset_full = CassavaDataset(
    train_df, train_transform, cache_images=True, cache_max_items=4096
)
valid_dataset_full = CassavaDataset(
    valid_df, valid_transform, cache_images=True, cache_max_items=4096
)



## === cell 21
pass



## === cell 22
import json

map_path = "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
with open(map_path, mode="r") as f:
    label_to_name = json.load(f)




## === cell 23
class Unnormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor = tensor.clone()
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor




## === cell 24
unnorm = Unnormalize(mean, std)




## === cell 25
def display_img(img, unnorm=None, label=None):
    if unnorm is not None:
        img = unnorm(img)

    plt.imshow(img.permute(1, 2, 0))

    if label is not None:
        plt.title(label_to_name[str(label)])




## === cell 26
def display_batch(batch, unnorm=None):
    imgs, labels = batch

    if unnorm:
        imgs = [unnorm(img) for img in imgs]

    ig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(imgs, nrow=8).permute(1, 2, 0))




## === cell 27
tensor, label = train_dataset_full[0]
display_img(tensor, unnorm, label)



## === cell 28
DO_VIS = False
if DO_VIS:
    loader = DataLoader(train_dataset_full, 16, shuffle=True, num_workers=2)
    display_batch(next(iter(loader)), unnorm=unnorm)



## === cell 29
import torch
import torch.nn as nn
import torch.nn.functional as F



## === cell 30
epoch = 3
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

torch.manual_seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass


def _default_num_workers():
    cpu = os.cpu_count() or 8
    if not torch.cuda.is_available():
        return max(2, min(8, cpu - 1))
    return max(4, min(12, cpu - 1))


DL_NUM_WORKERS = _default_num_workers()
DL_PREFETCH = 4
print("DataLoader num_workers:", DL_NUM_WORKERS, "prefetch_factor:", DL_PREFETCH)

if torch.cuda.is_available():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

batch_size = 32 if torch.cuda.is_available() else 16
print("batch_size:", batch_size)



## === cell 31
resNet = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
resNet = resNet.to(device)



## === cell 32
ef_model = timm.create_model(
    "tf_efficientnet_b2_ns", pretrained=True, num_classes=num_classes
)
ef_model = ef_model.to(device)



## === cell 33
ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=0.0001)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=0.0001)
resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)
criterion = nn.CrossEntropyLoss()




## === cell 34
def calc_correction(model, df=None, dataset=None, batch_size_eval=64, num_workers=None):
    model.eval()
    if num_workers is None:
        num_workers = DL_NUM_WORKERS

    if dataset is None:
        ds = CassavaDataset(
            df, transform=valid_transform, cache_images=True, cache_max_items=4096
        )
    else:
        ds = dataset

    loader = DataLoader(
        ds,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(DL_PREFETCH if num_workers > 0 else None),
    )

    use_channels_last = torch.cuda.is_available()
    correct = torch.zeros((), device=device, dtype=torch.long)
    pred_hist = torch.zeros((num_classes,), device=device, dtype=torch.long)

    with torch.inference_mode():
        for data, target in loader:
            if use_channels_last:
                data = data.contiguous(memory_format=torch.channels_last)
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pred = model(data).argmax(1)
            correct += (pred == target).sum()
            pred_hist += torch.bincount(pred, minlength=num_classes)

    percent = (correct.float() / len(ds)).item()
    pred_list = pred_hist.detach().cpu().tolist()
    return percent, pred_list




## === cell 35
from matplotlib import pyplot as plt


def plot_losses(epoch, title, train_losses, valid_losses):
    y = list(range(len(train_losses)))
    train_loss = plt.plot(y, train_losses)
    valid_loss = plt.plot(y, valid_losses)
    plt.title(title)
    plt.ylabel("loss")
    plt.legend((train_loss[0], valid_loss[0]), ("train loss", "valid loss"))
    plt.show()




## === cell 36
import time


def _maybe_compile(model: nn.Module):
    if not torch.cuda.is_available():
        return model
    if getattr(model, "_is_compiled", False):
        return model
    try:
        cm = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        setattr(cm, "_is_compiled", True)
        return cm
    except Exception:
        setattr(model, "_is_compiled", True)
        return model




## === cell 37
def train_model(
    model,
    train_dataset,
    valid_dataset,
    batch_size,
    optimizer,
    criterion,
    scheduler,
    epoch,
    model_title,
):
    best_model_state = None
    best_loss = float("inf")
    train_losses, valid_losses = [], []

    use_channels_last = torch.cuda.is_available()
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    model = _maybe_compile(model)

    train_loader = DataLoader(
        train_dataset,
        batch_size,
        shuffle=True,
        num_workers=DL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(DL_NUM_WORKERS > 0),
        prefetch_factor=(DL_PREFETCH if DL_NUM_WORKERS > 0 else None),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size,
        shuffle=False,
        num_workers=DL_NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(DL_NUM_WORKERS > 0),
        prefetch_factor=(DL_PREFETCH if DL_NUM_WORKERS > 0 else None),
    )

    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for ep in range(1, epoch + 1):
        epoch_start_time = time.time()
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for data, target in train_loader:
            if use_channels_last:
                data = data.contiguous(memory_format=torch.channels_last)
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(data)

        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for data, target in valid_loader:
                if use_channels_last:
                    data = data.contiguous(memory_format=torch.channels_last)
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    output = model(data)
                    loss = criterion(output, target)

                pred = output.argmax(1)
                correct += (pred == target).sum().item()
                total += target.numel()
                valid_loss += loss.item() * len(data)

        valid_loss_mean = valid_loss / len(valid_loader.dataset)
        if valid_loss_mean < best_loss:
            best_loss = valid_loss_mean
            best_model_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

        scheduler.step()

        collection = correct / max(1, total)
        valid_losses.append(valid_loss_mean)
        print(
            "Time: {:.3f}\t Epoch: {} \tTraining Loss: {:.3f} \tValidation Loss: {:.3f} \t Acc: {:.2f}".format(
                time.time() - epoch_start_time,
                ep,
                train_loss,
                valid_loss_mean,
                collection,
            )
        )

    if best_model_state is not None:
        torch.save(best_model_state, model_title)
        model.load_state_dict(best_model_state, strict=True)

    return model, train_losses, valid_losses




## === cell 38
def train_models(resNet, ef_model):
    model_title = "./res_model.pth"
    resNet, train_losses, valid_losses = train_model(
        resNet,
        train_dataset_full,
        valid_dataset_full,
        batch_size,
        resNet_optimizer,
        criterion,
        resNet_scheduler,
        epoch,
        model_title,
    )

    DO_PLOT = False
    if DO_PLOT:
        title = "resNet losses"
        plot_losses(epoch, title, train_losses, valid_losses)

    model_title = "./ef_model.pth"
    ef_model, train_losses, valid_losses = train_model(
        ef_model,
        train_dataset_full,
        valid_dataset_full,
        batch_size,
        ef_optimizer,
        criterion,
        ef_scheduler,
        epoch,
        model_title,
    )
    if DO_PLOT:
        title = "ef losses"
        plot_losses(epoch, title, train_losses, valid_losses)




## === cell 39
DO_TRAIN = True
if DO_TRAIN:
    train_models(resNet, ef_model)



## === cell 40
if os.path.exists("./ef_model.pth"):
    ef_model.load_state_dict(
        torch.load("./ef_model.pth", map_location=device), strict=True
    )
if os.path.exists("./res_model.pth"):
    resNet.load_state_dict(
        torch.load("./res_model.pth", map_location=device), strict=True
    )




## === cell 41
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




## === cell 42
classifier = CassaveClassifier(resNet, ef_model).to(device)




## === cell 43
def _batched_logits_two_models(
    model_a, model_b, df=None, dataset=None, batch_size_eval=64, num_workers=None
):
    if num_workers is None:
        num_workers = DL_NUM_WORKERS

    if dataset is None:
        ds = CassavaDataset(
            df, transform=valid_transform, cache_images=True, cache_max_items=4096
        )
    else:
        ds = dataset

    loader = DataLoader(
        ds,
        batch_size=batch_size_eval,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=(DL_PREFETCH if num_workers > 0 else None),
    )

    model_a.eval()
    model_b.eval()

    pin = torch.cuda.is_available()
    logits_a = torch.empty((len(ds), num_classes), dtype=torch.float32, pin_memory=pin)
    logits_b = torch.empty((len(ds), num_classes), dtype=torch.float32, pin_memory=pin)
    labels = torch.empty((len(ds),), dtype=torch.long, pin_memory=pin)

    use_channels_last = torch.cuda.is_available()
    use_amp = torch.cuda.is_available()

    offset = 0
    with torch.inference_mode():
        for data, target in loader:
            bs = data.size(0)
            if use_channels_last:
                data = data.contiguous(memory_format=torch.channels_last)
            data = data.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                la = model_a(data)
                lb = model_b(data)
            logits_a[offset : offset + bs].copy_(
                la.detach().float().cpu(), non_blocking=pin
            )
            logits_b[offset : offset + bs].copy_(
                lb.detach().float().cpu(), non_blocking=pin
            )
            labels[offset : offset + bs].copy_(target.to(torch.long), non_blocking=pin)
            offset += bs

    return logits_a, logits_b, labels


def find_best_rate_on_valid(classifier, valid_df=None, valid_dataset=None):
    logits_res, logits_ef, labels = _batched_logits_two_models(
        classifier.model,
        classifier.ef_model,
        df=valid_df,
        dataset=valid_dataset,
        batch_size_eval=64,
        num_workers=DL_NUM_WORKERS,
    )

    best_rate = 0.5
    best_acc = -1.0

    for rate_i in range(1, 10):
        rate = rate_i / 10.0
        p = rate * logits_res + (1.0 - rate) * logits_ef
        pred = p.argmax(1)
        acc = (pred == labels).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_rate = rate

    print("Best ensemble rate on valid_df:", best_rate, "valid acc:", best_acc)
    return best_rate, best_acc




## === cell 44
pass



## === cell 45
path = "../input/cassava-leaf-disease-classification/test_images/"



## === cell 46
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)
image_id = sample_sub["image_id"].tolist()
image_path = [os.path.join(path, fname) for fname in image_id]

missing = [p for p in image_path if not os.path.isfile(p)]
print("Expected test images:", len(image_id))
print("Missing files:", len(missing))
if len(missing) > 0:
    print("First missing example:", missing[0])



## === cell 47
best_rate, _ = find_best_rate_on_valid(classifier, valid_dataset=valid_dataset_full)




## === cell 48
class CassavaTestDataset(Dataset):
    def __init__(
        self, image_paths, transform=None, cache_images=True, cache_max_items=2048
    ):
        self.image_paths = list(image_paths)
        self.transform = transform
        self.cache_images = bool(cache_images)
        self.cache_max_items = int(cache_max_items)

    def __len__(self):
        return len(self.image_paths)

    def _load_pil_rgb(self, path):
        if self.cache_images:
            cache = _get_worker_cache(maxsize=self.cache_max_items)
            return cache(path)
        with open(path, "rb") as f:
            return Image.open(f).convert("RGB")

    def __getitem__(self, idx):
        pth = self.image_paths[idx]
        image = self._load_pil_rgb(pth)
        if self.transform is not None:
            image = self.transform(image)
        return image


classifier.eval()
test_ds = CassavaTestDataset(
    image_path, transform=valid_transform, cache_images=True, cache_max_items=4096
)
test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=DL_NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(DL_NUM_WORKERS > 0),
    prefetch_factor=(DL_PREFETCH if DL_NUM_WORKERS > 0 else None),
)

use_channels_last = torch.cuda.is_available()
use_amp = torch.cuda.is_available()
pred = []
with torch.inference_mode():
    for data in tqdm(test_loader, total=len(test_loader)):
        if use_channels_last:
            data = data.contiguous(memory_format=torch.channels_last)
        data = data.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            out = classifier.test(data, best_rate)
        pred.extend(out.argmax(1).detach().cpu().to(torch.int64).tolist())



## === cell 49
pred[:10], len(pred)



## === cell 50
sub = pd.DataFrame({"image_id": image_id, "label": pred})



## === cell 51
sub.head(), sub.shape



## === cell 52
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("Submission columns:", list(sub.columns))
