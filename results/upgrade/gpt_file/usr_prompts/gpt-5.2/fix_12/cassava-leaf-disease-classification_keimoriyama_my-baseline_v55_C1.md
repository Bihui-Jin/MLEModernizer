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

No external packages required in the script and installed.

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
import random
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from PIL import Image, ImageDraw

from sklearn import model_selection

import timm

Image.MAX_IMAGE_PIXELS = None

_HAS_TORCH_COMPILE = hasattr(torch, "compile")

os.environ.setdefault("PYTHONHASHSEED", "42")
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass




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
train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.label.values
)




## === cell 6
try:
    ax = train_df.label.value_counts().plot(kind="bar", title="train label counts")
except Exception:
    pass




## === cell 7
try:
    ax = valid_df.label.value_counts().plot(kind="bar", title="valid label counts")
except Exception:
    pass




## === cell 8
train_df = train_df.reset_index().drop(columns=["index"])
train_df.head()




## === cell 9
valid_df = valid_df.reset_index().drop(columns=["index"])
valid_df.head()




## === cell 10
im = Image.open(train_df["path"][0])
im.size




## === cell 11
try:
    im
except Exception:
    pass




## === cell 12
import matplotlib.image as img  # noqa: F401




## === cell 13
pass




## === cell 14
class CassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None, cache_images=False):
        super().__init__()
        self.paths = dataframe["path"].values
        self.labels = dataframe["label"].values.astype(np.int64)
        self.transform = transform
        self.cache_images = bool(cache_images)
        self._cache = {} if self.cache_images else None

    def __len__(self):
        return len(self.paths)

    def _load_image(self, path_):
        if self._cache is None:
            return Image.open(path_).convert("RGB")
        im = self._cache.get(path_)
        if im is None:
            im = Image.open(path_).convert("RGB")
            self._cache[path_] = im.copy()
            im.close()
            im = self._cache[path_]
        return im.copy()

    def __getitem__(self, index):
        path_ = self.paths[index]
        label = int(self.labels[index])
        image = self._load_image(path_)
        if self.transform is not None:
            image = self.transform(image)
        return image, label




## === cell 15
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)




## === cell 16
class make_mask_image:
    def __init__(self, p, mask_size=50):
        self.p = p
        self.mask_size = mask_size

    def __call__(self, image):
        start_width, start_height = [], []
        if random.random() < self.p:
            draw = ImageDraw.Draw(image)
            width, height = image.size
            ms = min(self.mask_size, max(1, width - 1), max(1, height - 1))
            for _ in range(10):
                start_width.append(random.randrange(0, max(1, width - ms)))
                start_height.append(random.randrange(0, max(1, height - ms)))
            for x, y in zip(start_width, start_height):
                draw.rectangle(
                    (x, y, x + ms, y + ms), fill=(0, 0, 0), outline=(0, 0, 0)
                )
        return image




## === cell 17
image_size = 512
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(image_size),
        make_mask_image(p=0.3, mask_size=50),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

valid_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)




## === cell 18
pass




## === cell 19
train_dataset_full = CassavaDataset(
    train_df, transform=train_transform, cache_images=True
)
valid_dataset_full = CassavaDataset(
    valid_df, transform=valid_transform, cache_images=True
)




## === cell 20
pass




## === cell 21
epoch = 3
batch_size = 16
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device




## === cell 22
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

use_amp = torch.cuda.is_available()
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)




## === cell 23
resNet = timm.create_model("resnet50", pretrained=True)
resNet.fc = nn.Linear(resNet.fc.in_features, num_classes)
resNet = resNet.to(device)




## === cell 24
ef_model = timm.create_model("tf_efficientnet_b2_ns", pretrained=True)
ef_model.classifier = nn.Linear(ef_model.classifier.in_features, num_classes)
ef_model = ef_model.to(device)




## === cell 25
ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=0.0001)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=0.0001)
resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)

criterion = nn.CrossEntropyLoss()




## === cell 26
class _EvalDataset(Dataset):
    def __init__(self, dataframe, transform, cache_images=False):
        self.paths = dataframe["path"].values
        self.labels = dataframe["label"].values.astype(int)
        self.transform = transform
        self.cache_images = bool(cache_images)
        self._cache = {} if self.cache_images else None

    def __len__(self):
        return len(self.paths)

    def _load_image(self, path_):
        if self._cache is None:
            return Image.open(path_).convert("RGB")
        im = self._cache.get(path_)
        if im is None:
            im = Image.open(path_).convert("RGB")
            self._cache[path_] = im.copy()
            im.close()
            im = self._cache[path_]
        return im.copy()

    def __getitem__(self, idx):
        im = self._load_image(self.paths[idx])
        im = self.transform(im)
        return im, int(self.labels[idx])


def calc_correction(model, df_):
    model.eval()
    eval_ds = _EvalDataset(df_, valid_transform, cache_images=True)

    cpu = os.cpu_count() or 0
    nw = min(4, cpu) if cpu >= 2 else 0
    persistent = bool(nw > 0)

    eval_loader = DataLoader(
        eval_ds,
        batch_size=64,
        shuffle=False,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent,
        prefetch_factor=4 if nw > 0 else None,
    )
    correct = 0
    pred_counts = torch.zeros(5, dtype=torch.long)
    with torch.inference_mode():
        for data, target in eval_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if torch.cuda.is_available():
                data = data.contiguous(memory_format=torch.channels_last)
            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(data)
            preds = out.argmax(1)
            correct += (preds == target).sum().item()
            pred_counts += torch.bincount(preds.detach().cpu(), minlength=5)
    percent = correct / len(eval_ds)
    return percent, pred_counts.tolist()




## === cell 27
from matplotlib import pyplot as plt


def plot_losses(epoch_, title, train_losses, valid_losses):
    try:
        y = list(range(len(train_losses)))
        train_loss = plt.plot(y, train_losses)
        valid_loss = plt.plot(y, valid_losses)
        plt.title(title)
        plt.ylabel("loss")
        plt.legend((train_loss[0], valid_loss[0]), ("train loss", "valid loss"))
        plt.show()
    except Exception:
        pass




## === cell 28
def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _maybe_compile(model):
    if _HAS_TORCH_COMPILE and torch.cuda.is_available():
        try:
            return torch.compile(model, mode="reduce-overhead")
        except Exception:
            return model
    return model


def train_model(
    model,
    train_ds,
    valid_ds,
    batch_size_,
    optimizer,
    criterion_,
    scheduler,
    epoch_,
    model_title,
):
    train_losses, valid_losses = [], []

    cpu = os.cpu_count() or 0
    nw = min(4, cpu) if cpu >= 2 else 0
    persistent = bool(nw > 0)
    pin = torch.cuda.is_available()

    g = torch.Generator()
    g.manual_seed(42)

    dl_kwargs = dict(
        num_workers=nw,
        pin_memory=pin,
        persistent_workers=persistent,
        prefetch_factor=4 if nw > 0 else None,
        worker_init_fn=_seed_worker if nw > 0 else None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size_,
        shuffle=True,
        generator=g,
        **dl_kwargs,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size_,
        shuffle=False,
        **dl_kwargs,
    )

    if hasattr(scheduler, "last_epoch"):
        scheduler.last_epoch = -1
    if hasattr(scheduler, "_step_count"):
        scheduler._step_count = 0

    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)

    model = _maybe_compile(model)

    best_ckpt_path = model_title + ".best_tmp"
    if os.path.exists(best_ckpt_path):
        try:
            os.remove(best_ckpt_path)
        except Exception:
            pass

    best_loss_global = float("inf")

    for ep in range(1, epoch_ + 1):
        epoch_start_time = time.time()
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        for data, target in train_loader:
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            if torch.cuda.is_available():
                data = data.contiguous(memory_format=torch.channels_last)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(data)
                loss = criterion_(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * len(data)

        train_loss = train_loss / len(train_ds)
        train_losses.append(train_loss)

        correct = 0
        total = 0
        model.eval()
        with torch.inference_mode():
            for data, target in valid_loader:
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                if torch.cuda.is_available():
                    data = data.contiguous(memory_format=torch.channels_last)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    output = model(data)
                    loss = criterion_(output, target)
                preds = output.argmax(1)
                correct += (preds == target).sum().item()
                total += target.numel()
                valid_loss += loss.item() * len(data)

        avg_valid_loss = valid_loss / len(valid_ds)

        if avg_valid_loss < best_loss_global:
            best_loss_global = avg_valid_loss
            torch.save(model.state_dict(), best_ckpt_path)

        scheduler.step()

        collection = (correct / total) if total else 0.0
        valid_losses.append(avg_valid_loss)
        print(
            "Time: {:.3f}\t Epoch: {} \tTraining Loss: {:.3f} \tValidation Loss: {:.3f} \t Acc: {:.2f}".format(
                time.time() - epoch_start_time,
                ep,
                train_loss,
                avg_valid_loss,
                collection,
            )
        )

    if os.path.exists(best_ckpt_path):
        best_state = torch.load(best_ckpt_path, map_location="cpu")
        model.load_state_dict(best_state)
        torch.save(best_state, model_title)
        try:
            os.remove(best_ckpt_path)
        except Exception:
            pass
    else:
        torch.save(model.state_dict(), model_title)

    return model, train_losses, valid_losses




## === cell 29
def train_models():
    global resNet, ef_model
    res_ckpt = "./res_model.pth"
    ef_ckpt = "./ef_model.pth"

    if os.path.exists(res_ckpt):
        resNet.load_state_dict(torch.load(res_ckpt, map_location=device))
        resNet = resNet.to(device).eval()
        if torch.cuda.is_available():
            resNet = resNet.to(memory_format=torch.channels_last)
        print("Loaded existing resNet checkpoint; skipped training.")
    else:
        model_title = res_ckpt
        resNet_trained, train_losses, valid_losses = train_model(
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
        resNet = resNet_trained
        print("resNet valid correction:", calc_correction(resNet, valid_df))
        plot_losses(epoch, "resNet losses", train_losses, valid_losses)

    if os.path.exists(ef_ckpt):
        ef_model.load_state_dict(torch.load(ef_ckpt, map_location=device))
        ef_model = ef_model.to(device).eval()
        if torch.cuda.is_available():
            ef_model = ef_model.to(memory_format=torch.channels_last)
        print("Loaded existing efficientnet checkpoint; skipped training.")
    else:
        model_title = ef_ckpt
        ef_trained, train_losses, valid_losses = train_model(
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
        ef_model = ef_trained
        print("ef_model valid correction:", calc_correction(ef_model, valid_df))
        plot_losses(epoch, "ef losses", train_losses, valid_losses)




## === cell 30
train_models()




## === cell 31
if os.path.exists("./ef_model.pth"):
    ef_model.load_state_dict(torch.load("./ef_model.pth", map_location=device))
if os.path.exists("./res_model.pth"):
    resNet.load_state_dict(torch.load("./res_model.pth", map_location=device))

ef_model = ef_model.to(device).eval()
resNet = resNet.to(device).eval()
if torch.cuda.is_available():
    ef_model = ef_model.to(memory_format=torch.channels_last)
    resNet = resNet.to(memory_format=torch.channels_last)




## === cell 32
class CassaveClassifier(nn.Module):
    def __init__(self, model, ef_model):
        super().__init__()
        self.model = model
        self.ef_model = ef_model

    def forward(self, x):
        x1 = self.model(x)
        x2 = self.ef_model(x)
        return 0.3 * x1 + 0.7 * x2

    def test(self, x, rate):
        x1 = self.model(x)
        x2 = self.ef_model(x)
        p = rate * x1 + (1 - rate) * x2
        return p




## === cell 33
classifier = CassaveClassifier(resNet, ef_model).to(device).eval()
if torch.cuda.is_available():
    classifier = classifier.to(memory_format=torch.channels_last)

classifier = _maybe_compile(classifier)




## === cell 34
def test_rate():
    classifier.eval()
    for rate in range(1, 10):
        paths = valid_df["path"].values
        labels = valid_df["label"].values.astype(int)
        count = 0
        with torch.inference_mode():
            for image_path, image_label in zip(paths, labels):
                image = Image.open(image_path).convert("RGB")
                image = valid_transform(image).unsqueeze(0).to(device)
                if torch.cuda.is_available():
                    image = image.contiguous(memory_format=torch.channels_last)
                pred = classifier.test(image, rate / 10).argmax(1).item()
                if pred == int(image_label):
                    count += 1
        percent = count / len(paths)
        print("rate: ", rate / 10, "percent: ", percent)




## === cell 35
pass




## === cell 36
test_images_dir = "../input/cassava-leaf-disease-classification/test_images/"




## === cell 37
sample_sub = pd.read_csv(path + "/sample_submission.csv")
image_id = sample_sub["image_id"].tolist()
image_path = [os.path.join(test_images_dir, fn) for fn in image_id]

missing = [p for p in image_path if not os.path.isfile(p)]
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test image files. Example: {missing[0]}"
    )




## === cell 38
class _TestDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        im = Image.open(self.paths[idx]).convert("RGB")
        im = self.transform(im)
        return im


test_ds = _TestDataset(image_path, valid_transform)

cpu = os.cpu_count() or 0
nw = min(4, cpu) if cpu >= 2 else 0
persistent = bool(nw > 0)

test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=nw,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=persistent,
    prefetch_factor=4 if nw > 0 else None,
    worker_init_fn=_seed_worker if nw > 0 else None,
)

pred = []
classifier.eval()
with torch.inference_mode():
    for data in test_loader:
        data = data.to(device, non_blocking=True)
        if torch.cuda.is_available():
            data = data.contiguous(memory_format=torch.channels_last)
        with torch.cuda.amp.autocast(enabled=use_amp):
            out = classifier(data)
        preds = out.argmax(1).detach().cpu().tolist()
        pred.extend(int(p) for p in preds)

len(pred), len(image_id)




## === cell 39
pred[:10]




## === cell 40
sub = pd.DataFrame({"image_id": image_id, "label": np.asarray(pred, dtype=np.int64)})




## === cell 41
sub.head()




## === cell 42
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv columns:", list(sub.columns))
print(sub.iloc[:3])
