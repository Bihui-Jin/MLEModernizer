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
import os, subprocess, sys

wheel_path = "../input/pretrined-models/timm-0.3.3-py3-none-any.whl"
if os.path.exists(wheel_path):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-deps", wheel_path]
    )
else:
    print(f"Wheel not found at {wheel_path}; using the preinstalled timm instead.")



## === cell 1
import os
import pandas as pd
import timm



## === cell 2
path = "../input/cassava-leaf-disease-classification"
_ = path  # keep variable used later



## === cell 3
df = pd.read_csv(path + "/train.csv")



## === cell 4
_ = df.shape



## === cell 5
df["path"] = path + "/train_images/" + df["image_id"]
df = df.drop(columns=["image_id"])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)



## === cell 6
from sklearn import model_selection

train_df, valid_df = model_selection.train_test_split(
    df, test_size=0.2, random_state=42, stratify=df.label.values
)



## === cell 7
_ = train_df.label.value_counts()



## === cell 8
_ = valid_df.label.value_counts()



## === cell 9
train_df = train_df.reset_index().drop(columns=["index"])
_ = train_df.shape



## === cell 10
valid_df = valid_df.reset_index().drop(columns=["index"])
_ = valid_df.shape



## === cell 11
from PIL import Image, ImageDraw

_ = None



## === cell 12
_ = None



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
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True



## === cell 15
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.functional import to_pil_image


def _fast_open_as_pil_rgb(path_):
    try:
        t = read_image(path_, mode=ImageReadMode.RGB)
        return to_pil_image(t)
    except Exception:
        with Image.open(path_) as im:
            return im.convert("RGB")


class CassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        super().__init__()
        self.paths = dataframe["path"].to_numpy()
        self.labels = dataframe["label"].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path_ = self.paths[index]
        label = int(self.labels[index])
        image = _fast_open_as_pil_rgb(path_)
        if self.transform is not None:
            image = self.transform(image)
        return image, label




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
dataset = CassavaDataset(df, train_transform)



## === cell 19
import torch
import torch.nn as nn
import torch.nn.functional as F



## === cell 20
epoch = 3
batch_size = 16
num_classes = 5
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 21
torch.set_float32_matmul_precision("high")

resNet = timm.create_model("resnet50", pretrained=False)
resNet.fc = nn.Linear(resNet.fc.in_features, num_classes)

resnet_w_path = "../input/pretrined-models/models/models/pretrained_resNet.pth"
if os.path.exists(resnet_w_path):
    state = torch.load(resnet_w_path, map_location="cpu")
    resNet.load_state_dict(state, strict=False)
else:
    print(
        f"Pretrained weights not found: {resnet_w_path}. Falling back to timm pretrained weights."
    )
    resNet = timm.create_model("resnet50", pretrained=True)
    resNet.fc = nn.Linear(resNet.fc.in_features, num_classes)

if device.type == "cuda":
    resNet = resNet.to(memory_format=torch.channels_last)
resNet = resNet.to(device)



## === cell 22
ef_model = timm.create_model("tf_efficientnet_b2_ns", pretrained=False)
ef_model.classifier = nn.Linear(ef_model.classifier.in_features, num_classes)

ef_w_path = "../input/pretrined-models/models/models/pretrained_ef_model.pth"
if os.path.exists(ef_w_path):
    state = torch.load(ef_w_path, map_location="cpu")
    ef_model.load_state_dict(state, strict=False)
else:
    print(
        f"Pretrained weights not found: {ef_w_path}. Falling back to timm pretrained weights."
    )
    ef_model = timm.create_model("tf_efficientnet_b2_ns", pretrained=True)
    ef_model.classifier = nn.Linear(ef_model.classifier.in_features, num_classes)

if device.type == "cuda":
    ef_model = ef_model.to(memory_format=torch.channels_last)
ef_model = ef_model.to(device)



## === cell 23
ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=0.0001)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=0.0001)
resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)
criterion = nn.CrossEntropyLoss()




## === cell 24
class CassavaInferenceDataset(Dataset):
    def __init__(self, dataframe, transform):
        self.paths = dataframe["path"].to_numpy()
        self.labels = dataframe["label"].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = int(self.labels[idx])
        im = _fast_open_as_pil_rgb(p)
        im = self.transform(im)
        return im, y


def _best_num_workers():
    cpu = os.cpu_count() or 2
    return 4 if cpu >= 8 else 2


def calc_correction(model, df_):
    model.eval()
    infer_ds = CassavaInferenceDataset(df_, valid_transform)
    num_workers = _best_num_workers()
    pin = torch.cuda.is_available()
    loader = DataLoader(
        infer_ds,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    correct = 0
    total = 0
    pred_hist = torch.zeros(5, dtype=torch.long)
    with torch.inference_mode():
        for x, y in loader:
            if device.type == "cuda":
                x = x.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
            else:
                x = x.to(device)
            y = (
                y.to(device, non_blocking=True)
                if device.type == "cuda"
                else y.to(device)
            )
            p = model(x).argmax(1)
            total += y.numel()
            correct += (p == y).sum().item()
            pred_hist += torch.bincount(p.detach().cpu(), minlength=5)
    percent = correct / total if total else 0.0
    return percent, pred_hist.tolist()




## === cell 25
from matplotlib import pyplot as plt


def plot_losses(epoch, title, train_losses, valid_losses):
    y = list(range(len(train_losses)))
    train_loss = plt.plot(y, train_losses)
    valid_loss = plt.plot(y, valid_losses)
    plt.title(title)
    plt.ylabel("loss")
    plt.legend((train_loss[0], valid_loss[0]), ("train loss", "valid loss"))
    plt.show()




## === cell 26
import time


def train_model(
    model, dataset, batch_size, optimizer, criterion, scheduler, epoch, model_title
):
    best_model_state = None
    best_loss = float("inf")
    train_losses, valid_losses = [], []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    num_workers = _best_num_workers()
    pin = torch.cuda.is_available()
    persist = True if num_workers > 0 else False

    for fold, (train_index, valid_index) in enumerate(kf.split(range(len(dataset)))):
        print("fold: ", fold)
        train_dataset = Subset(dataset, train_index)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=persist,
            prefetch_factor=2 if num_workers > 0 else None,
        )
        valid_dataset = Subset(dataset, valid_index)
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=persist,
            prefetch_factor=2 if num_workers > 0 else None,
        )

        for ep in range(1, epoch + 1):
            epoch_start_time = time.time()
            acc = []
            train_loss = 0.0
            valid_loss = 0.0

            model.train()
            for data, target in train_loader:
                if device.type == "cuda":
                    data = data.to(device, non_blocking=True).to(
                        memory_format=torch.channels_last
                    )
                    target = target.to(device, non_blocking=True)
                else:
                    data = data.to(device)
                    target = target.to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(data)
            train_loss = train_loss / len(train_loader.dataset)
            train_losses.append(train_loss)

            model.eval()
            with torch.no_grad():
                for data, target in valid_loader:
                    if device.type == "cuda":
                        data = data.to(device, non_blocking=True).to(
                            memory_format=torch.channels_last
                        )
                        target = target.to(device, non_blocking=True)
                    else:
                        data = data.to(device)
                        target = target.to(device)
                    output = model(data)
                    pred = output.argmax(1) == target
                    acc.append(pred.float().mean().item())
                    loss = criterion(output, target)
                    valid_loss += loss.item() * len(data)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

            scheduler.step()

            collection = sum(acc) / len(acc) if len(acc) else 0.0
            valid_loss = valid_loss / len(valid_loader.dataset)
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

    if best_model_state is not None:
        torch.save(best_model_state, model_title)
        model.load_state_dict(best_model_state)
    else:
        torch.save(model.state_dict(), model_title)

    return model, train_losses, valid_losses




## === cell 27
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
    return resNet, ef_model




## === cell 28
fine_ef_path = "../input/pretrined-models/models/models/fine_tuned_ef_model.pth"
fine_res_path = "../input/pretrined-models/models/models/res_model.pth"

local_ef_path = "./ef_model.pth"
local_res_path = "./res_model.pth"

have_finetuned = os.path.exists(fine_ef_path) and os.path.exists(fine_res_path)

if have_finetuned:
    print("Found fine-tuned weights; loading and skipping training.")
    ef_model.load_state_dict(
        torch.load(fine_ef_path, map_location=device), strict=False
    )
    resNet.load_state_dict(torch.load(fine_res_path, map_location=device), strict=False)
else:
    print(
        "Fine-tuned weights not found; training models to produce a valid submission."
    )
    resNet, ef_model = train_models(resNet, ef_model)




## === cell 29
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




## === cell 30
classifier = CassaveClassifier(resNet, ef_model)
classifier = classifier.to(device)




## === cell 31
def test_rate():
    infer_ds = CassavaInferenceDataset(valid_df, valid_transform)
    num_workers = _best_num_workers()
    pin = torch.cuda.is_available()
    loader = DataLoader(
        infer_ds,
        batch_size=64,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    classifier.eval()
    with torch.inference_mode():
        for rate in range(1, 10):
            correct = 0
            total = 0
            for x, y in loader:
                if device.type == "cuda":
                    x = x.to(device, non_blocking=True).to(
                        memory_format=torch.channels_last
                    )
                    y = y.to(device, non_blocking=True)
                else:
                    x = x.to(device)
                    y = y.to(device)
                pred = classifier.test(x, rate / 10).argmax(1)
                total += y.numel()
                correct += (pred == y).sum().item()
            percent = correct / total if total else 0.0
            print("rate: ", rate / 10)
            print("percent: ", percent)




## === cell 32
path = "../input/cassava-leaf-disease-classification/test_images/"



## === cell 33
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_sub_path)

test_dir = path
image_id = sample_sub["image_id"].tolist()
image_path = [os.path.join(test_dir, iid) for iid in image_id]

len(image_id), len(image_path), image_id[:3]




## === cell 34
class CassavaTestDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        im = _fast_open_as_pil_rgb(p)
        im = self.transform(im)
        return im


classifier.eval()
test_ds = CassavaTestDataset(image_path, valid_transform)
num_workers = _best_num_workers()
pin = torch.cuda.is_available()
test_loader = DataLoader(
    test_ds,
    batch_size=64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=True if num_workers > 0 else False,
    prefetch_factor=2 if num_workers > 0 else None,
)

pred = []
with torch.inference_mode():
    for x in test_loader:
        if device.type == "cuda":
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        else:
            x = x.to(device)
        p = classifier(x).argmax(1).detach().cpu().tolist()
        pred.extend([int(v) for v in p])

len(pred), pred[:10]



## === cell 35
_ = pred[:20]



## === cell 36
sub = pd.DataFrame({"image_id": image_id, "label": pred})
sub.head()



## === cell 37
sub



## === cell 38
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
