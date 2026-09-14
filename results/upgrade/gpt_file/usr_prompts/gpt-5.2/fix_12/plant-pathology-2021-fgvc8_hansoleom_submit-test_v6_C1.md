# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import copy

import pandas as pd
from PIL import Image

import torch
from torch import optim
from torch.optim.lr_scheduler import _LRScheduler
from torchvision import models

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = True  # faster for fixed-size inputs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

from torchvision.transforms import v2 as T

DATA_DIR = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

print("Train CSV:", TRAIN_CSV_PATH)
print("Train images dir exists:", os.path.isdir(TRAIN_IMG_DIR))
print("Test images dir exists:", os.path.isdir(TEST_IMG_DIR))



## === cell 1
transform_train = T.Compose(
    [
        T.RandomResizedCrop((640, 640)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
transform_valid = T.Compose(
    [
        T.Resize((640, 640)),
        T.CenterCrop((640, 640)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)



## === cell 2
df = pd.read_csv(TRAIN_CSV_PATH)

idxs = list(range(len(df)))
random.shuffle(idxs)
cnt = int(len(df) * 0.9)
train_idxs = idxs[:cnt]
valid_idxs = idxs[cnt:]

train_df = df.iloc[train_idxs].reset_index(drop=True)
valid_df = df.iloc[valid_idxs].reset_index(drop=True)

print("Train size:", len(train_df), "Valid size:", len(valid_df))



## === cell 3
try:
    from torchvision.io import read_image, ImageReadMode

    _HAS_TV_READ_IMAGE = True
except Exception:
    _HAS_TV_READ_IMAGE = False


class torchvision_Dataset(torch.utils.data.Dataset):
    """
    Speed optimization (correctness-preserving):
    - Cache *raw decoded uint8 RGB tensors* in RAM (not transformed tensors).
      This removes repeated JPEG decode + disk I/O across epochs while keeping
      identical per-epoch random augmentations because transforms still run on every __getitem__.

    Important stability note:
    - If you use cache_images=True with num_workers>0, each worker gets its own copy of the
      Dataset object and builds its own cache -> huge RAM usage and workers can be OOM-killed.
      Therefore, we will run with num_workers=0 when caching is enabled.
    """

    def __init__(
        self, data_root, df, label2idx, transforms=None, cache_images: bool = False
    ):
        self.image_path = data_root
        self.label2idx = label2idx
        self.transform = transforms
        self.cache_images = cache_images

        df = df.reset_index(drop=True)
        self._images = df["image"].tolist()
        self._labels = df["labels"].tolist()

        self._cache = {} if cache_images else None

    def __len__(self):
        return len(self._images)

    def _read_rgb(self, img_path: str):
        if _HAS_TV_READ_IMAGE:
            return read_image(img_path, mode=ImageReadMode.RGB)  # uint8 CHW
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            return T.ToImage()(im)

    def __getitem__(self, idx):
        image_name = self._images[idx]
        label_name = self._labels[idx]
        img_path = os.path.join(self.image_path, image_name)

        if self._cache is not None:
            img = self._cache.get(image_name)
            if img is None:
                img = self._read_rgb(img_path)
                self._cache[image_name] = img
        else:
            img = self._read_rgb(img_path)

        x = self.transform(img) if self.transform else img
        y = self.label2idx[label_name]
        return x, y




## === cell 4
all_labels = sorted(df["labels"].unique().tolist())
label2idx = {lab: i for i, lab in enumerate(all_labels)}
idx2label = {i: lab for lab, i in label2idx.items()}

NUM_CLASSES = len(label2idx)
print("Num classes:", NUM_CLASSES)

train_dataset = torchvision_Dataset(
    TRAIN_IMG_DIR, train_df, label2idx, transform_train, cache_images=True
)
valid_dataset = torchvision_Dataset(
    TRAIN_IMG_DIR, valid_df, label2idx, transform_valid, cache_images=True
)




## === cell 5
def _seed_worker(worker_id):
    worker_seed = 42 + worker_id
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


_cpu = os.cpu_count() or 2
num_workers_base = min(12, max(4, _cpu - 1))

g = torch.Generator()
g.manual_seed(42)

BASE_BATCH = 16
TARGET_EFFECTIVE_BATCH = 64  # preserve original effective batch via accumulation


def _pick_batch_size(device, base=BASE_BATCH):
    return base


def _make_loader(
    dataset,
    batch_size,
    shuffle,
    pin_memory,
    num_workers,
    generator=None,
):
    kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if num_workers > 0:
        kwargs.update(
            dict(
                persistent_workers=True,
                prefetch_factor=4,
                worker_init_fn=_seed_worker,
            )
        )
        if generator is not None and shuffle:
            kwargs["generator"] = generator
    return torch.utils.data.DataLoader(**kwargs)




## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

if device.type == "cuda":
    torch.cuda.empty_cache()

BATCH_SIZE = _pick_batch_size(device, base=BASE_BATCH)

GRAD_ACCUM_STEPS = max(1, TARGET_EFFECTIVE_BATCH // BATCH_SIZE)
if TARGET_EFFECTIVE_BATCH % BATCH_SIZE != 0:
    GRAD_ACCUM_STEPS = 1

print(
    "Batch size:",
    BATCH_SIZE,
    "Grad accumulation steps:",
    GRAD_ACCUM_STEPS,
    "Effective batch:",
    BATCH_SIZE * GRAD_ACCUM_STEPS,
)

pin_mem = device.type == "cuda"

train_num_workers = (
    0 if getattr(train_dataset, "cache_images", False) else num_workers_base
)
valid_num_workers = (
    0 if getattr(valid_dataset, "cache_images", False) else num_workers_base
)
print("num_workers (train/valid):", train_num_workers, "/", valid_num_workers)

train_dataloaders = _make_loader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=pin_mem,
    num_workers=train_num_workers,
    generator=g,
)
valid_dataloaders = _make_loader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=pin_mem,
    num_workers=valid_num_workers,
    generator=None,
)



## === cell 7
weights = None  # preserve original (no pretrained)
model_ft = models.efficientnet_b4(weights=weights)
in_features = model_ft.classifier[1].in_features
model_ft.classifier[1] = torch.nn.Linear(in_features, NUM_CLASSES)
model_ft = model_ft.to(device)

if device.type == "cuda":
    model_ft = model_ft.to(memory_format=torch.channels_last)

print("torch.compile disabled for runtime safety under 600s timeout")




## === cell 8
class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [
            base_lr
            * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)




## === cell 9
criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_ft, 30, eta_min=0, last_epoch=-1
)
exp_lr_scheduler = GradualWarmupScheduler(
    optimizer_ft, multiplier=100, total_epoch=3, after_scheduler=cosine_scheduler
)




## === cell 10
class CUDAPrefetcher:
    def __init__(self, loader, device, channels_last: bool = False):
        self.loader = loader
        self.device = device
        self.channels_last = channels_last and (device.type == "cuda")

    def __iter__(self):
        if self.device.type != "cuda":
            yield from self.loader
            return

        stream = torch.cuda.Stream()
        it = iter(self.loader)

        def _to_device(batch):
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                a, b = batch
                if torch.is_tensor(a):
                    a = a.to(self.device, non_blocking=True)
                    if self.channels_last:
                        a = a.to(memory_format=torch.channels_last)
                if torch.is_tensor(b):
                    b = b.to(self.device, non_blocking=True)
                return (a, b)
            return batch

        def _preload():
            try:
                batch = next(it)
            except StopIteration:
                return None
            with torch.cuda.stream(stream):
                batch = _to_device(batch)
            return batch

        next_batch = _preload()
        while next_batch is not None:
            torch.cuda.current_stream().wait_stream(stream)
            batch = next_batch
            next_batch = _preload()
            yield batch




## === cell 11
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    os.makedirs("outputs", exist_ok=True)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = -1

    use_amp = device.type == "cuda"
    autocast_ctx = torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    for epoch in range(num_epochs):
        model.train()

        running_loss = 0.0
        train_corrects_t = torch.zeros((), device=device, dtype=torch.long)
        train_data_cnt = 0

        optimizer.zero_grad(set_to_none=True)
        step_in_accum = 0

        train_iterable = CUDAPrefetcher(train_dataloaders, device, channels_last=True)

        for inputs, labels in train_iterable:
            if device.type != "cuda":
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

            with autocast_ctx(enabled=use_amp):
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                loss = criterion(outputs, labels) / GRAD_ACCUM_STEPS

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            bs = inputs.size(0)
            running_loss += loss.detach().float().item() * bs * GRAD_ACCUM_STEPS
            train_corrects_t += (preds == labels).sum()
            train_data_cnt += bs

            step_in_accum += 1
            if step_in_accum == GRAD_ACCUM_STEPS:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step_in_accum = 0

        if step_in_accum != 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        model.eval()
        valid_corrects_t = torch.zeros((), device=device, dtype=torch.long)

        valid_iterable = CUDAPrefetcher(valid_dataloaders, device, channels_last=True)

        with torch.inference_mode():
            for inputs, labels in valid_iterable:
                if device.type != "cuda":
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                with autocast_ctx(enabled=use_amp):
                    outputs = model(inputs)
                    preds = outputs.argmax(dim=1)

                valid_corrects_t += (preds == labels).sum()

        valid_corrects = int(valid_corrects_t.detach().cpu().item())
        epoch_acc = valid_corrects / len(valid_dataset)

        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, f"outputs/{best_epoch}.pth")
            print(f"best epoch : {best_epoch} (valid acc={best_acc:.5f})")

    return best_model_wts, best_epoch




## === cell 12
try:
    best_wts, best_epoch = train_model(
        model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25
    )
except RuntimeError as e:
    msg = str(e)
    if "DataLoader worker" in msg or "killed by signal" in msg:
        print(
            "Caught DataLoader worker failure; retrying with num_workers=0 for train/valid."
        )
        train_dataloaders = _make_loader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=pin_mem,
            num_workers=0,
            generator=g,
        )
        valid_dataloaders = _make_loader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            pin_memory=pin_mem,
            num_workers=0,
            generator=None,
        )
        best_wts, best_epoch = train_model(
            model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=25
        )
    else:
        raise

model_ft.load_state_dict(best_wts)
model_ft.to(device)



## === cell 13
print("Using trained model from this run. Best epoch:", best_epoch)



## === cell 14
print("Example labels:", list(idx2label.values())[:5])




## === cell 15
class TestDataset(torch.utils.data.Dataset):
    def __init__(
        self, data_root, image_names, transforms=None, cache_images: bool = False
    ):
        self.data_root = data_root
        self.image_names = list(image_names)
        self.transform = transforms
        self.cache_images = cache_images
        self._cache = {} if cache_images else None

    def __len__(self):
        return len(self.image_names)

    def _read_rgb(self, img_path: str):
        if _HAS_TV_READ_IMAGE:
            return read_image(img_path, mode=ImageReadMode.RGB)
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            return T.ToImage()(im)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.data_root, img_name)

        if self._cache is not None:
            img = self._cache.get(img_name)
            if img is None:
                img = self._read_rgb(img_path)
                self._cache[img_name] = img
        else:
            img = self._read_rgb(img_path)

        x = self.transform(img) if self.transform else img
        return img_name, x


sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_images = sample_sub["image"].tolist()

test_dataset = TestDataset(
    TEST_IMG_DIR, test_images, transform_valid, cache_images=False
)

test_num_workers = num_workers_base
test_loader = _make_loader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=pin_mem,
    num_workers=test_num_workers,
    generator=None,
)

use_amp = device.type == "cuda"
autocast_ctx = torch.cuda.amp.autocast if use_amp else torch.cpu.amp.autocast

submit = []
model_ft.eval()

test_iterable = CUDAPrefetcher(test_loader, device, channels_last=True)

with torch.inference_mode():
    for img_names, x in test_iterable:
        if device.type != "cuda":
            x = x.to(device, non_blocking=True)
        with autocast_ctx(enabled=use_amp):
            logits = model_ft(x)
        pred_idxs = torch.argmax(logits, dim=1).detach().cpu().tolist()
        submit.extend([[n, idx2label[i]] for n, i in zip(img_names, pred_idxs)])

submission = pd.DataFrame(submit, columns=["image", "labels"])
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv with shape:", submission.shape)
print(submission.head())
