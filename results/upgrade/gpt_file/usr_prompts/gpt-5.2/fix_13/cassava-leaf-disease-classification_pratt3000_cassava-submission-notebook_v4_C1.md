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
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.models as models
import torchvision.transforms as T

from torchvision.io import ImageReadMode

torch.manual_seed(42)
np.random.seed(42)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True  # fixed image size => faster

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
except Exception:
    pass

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

assert os.path.exists(TEST_DIR), f"Missing test directory: {TEST_DIR}"
assert os.path.exists(TRAIN_DIR), f"Missing train directory: {TRAIN_DIR}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample_submission.csv: {SAMPLE_SUB_PATH}"
assert os.path.exists(TRAIN_CSV_PATH), f"Missing train.csv: {TRAIN_CSV_PATH}"

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass



## === cell 1
from torchvision.io import decode_jpeg


def get_image_tensor(path: str) -> torch.Tensor:
    with open(path, "rb") as f:
        data = f.read()
    buf = torch.frombuffer(data, dtype=torch.uint8)
    img = decode_jpeg(buf, mode=ImageReadMode.RGB)
    return img  # CHW uint8


class GetDataset(Dataset):
    def __init__(self, df, data_root, transforms=None, output_label=True):
        super().__init__()
        df = df.reset_index(drop=True)
        self.image_ids = df["image_id"].tolist()
        self.paths = [os.path.join(data_root, x) for x in self.image_ids]
        self.labels = (
            df["label"].astype(int).tolist()
            if output_label and "label" in df.columns
            else None
        )
        self.transforms = transforms
        self.output_label = output_label

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        img = get_image_tensor(self.paths[index])  # CHW uint8
        if self.transforms:
            img = self.transforms(img)
        if self.output_label:
            return img, int(self.labels[index])
        else:
            return img




## === cell 2
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for x in self.dl:
            yield to_device(x, self.device)

    def __len__(self):
        return len(self.dl)


device = get_device()
device




## === cell 3
def accuracy(out, labels):
    _, preds = torch.max(out, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {"val_loss": loss.detach(), "val_acc": acc}

    def validation_epoch_end(self, outputs):
        batch_loss = [x["val_loss"] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x["val_acc"] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {"val_loss": epoch_loss.item(), "val_acc": epoch_acc.item()}

    def epoch_end(self, epoch, epochs, result):
        print(
            "Epoch: [{}/{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch,
                epochs,
                result["train_loss"],
                result["val_loss"],
                result["val_acc"],
            )
        )




## === cell 4
class Classifier(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = models.wide_resnet101_2(pretrained=True)
        number_of_features = self.network.fc.in_features
        self.network.fc = nn.Linear(number_of_features, 5)

    def forward(self, xb):
        return self.network(xb)

    def freeze(self):
        for param in self.network.parameters():
            param.requires_grad = False
        for param in self.network.fc.parameters():
            param.requires_grad = True

    def unfreeze(self):
        for param in self.network.parameters():
            param.requires_grad = True




## === cell 5
IMG_SIZE = 224
IMG_SHAPE = (IMG_SIZE, IMG_SIZE)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

try:
    import torchvision.transforms.v2 as Tv2

    train_transforms = Tv2.Compose(
        [
            Tv2.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            Tv2.RandomHorizontalFlip(p=0.5),
            Tv2.RandomApply(
                [
                    Tv2.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                    )
                ],
                p=0.5,
            ),
            Tv2.RandomRotation(degrees=10),
            Tv2.ToDtype(torch.float32, scale=True),
            Tv2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    valid_transforms = Tv2.Compose(
        [
            Tv2.Resize(int(IMG_SIZE * 256 / 224), antialias=True),
            Tv2.CenterCrop(IMG_SIZE),
            Tv2.ToDtype(torch.float32, scale=True),
            Tv2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    test_transforms_a = valid_transforms
    test_transforms_b = Tv2.Compose(
        [
            Tv2.Resize(int(IMG_SIZE * 288 / 224), antialias=True),
            Tv2.CenterCrop(IMG_SIZE),
            Tv2.ToDtype(torch.float32, scale=True),
            Tv2.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
except Exception:
    train_transforms = T.Compose(
        [
            T.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)],
                p=0.5,
            ),
            T.RandomRotation(degrees=10),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    valid_transforms = T.Compose(
        [
            T.Resize(int(IMG_SIZE * 256 / 224)),
            T.CenterCrop(IMG_SIZE),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    test_transforms_a = valid_transforms
    test_transforms_b = T.Compose(
        [
            T.Resize(int(IMG_SIZE * 288 / 224)),
            T.CenterCrop(IMG_SIZE),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

train_df = pd.read_csv(TRAIN_CSV_PATH)
assert "image_id" in train_df.columns and "label" in train_df.columns
train_df["label"] = train_df["label"].astype(int)

perm = np.random.RandomState(42).permutation(len(train_df))
val_size = int(0.1 * len(train_df))
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

trn_ds = GetDataset(
    trn_df[["image_id", "label"]],
    TRAIN_DIR,
    transforms=train_transforms,
    output_label=True,
)
val_ds = GetDataset(
    val_df[["image_id", "label"]],
    TRAIN_DIR,
    transforms=valid_transforms,
    output_label=True,
)

BATCH_SIZE = 64

_cpu = os.cpu_count() or 2

num_workers = min(4, max(2, _cpu // 2))
persistent_workers = True if num_workers > 0 else False

g = torch.Generator()
g.manual_seed(42)


def collate_train(batch):
    imgs, labels = zip(*batch)
    x = torch.stack(imgs, 0).contiguous(memory_format=torch.channels_last)
    y = torch.as_tensor(labels, dtype=torch.long)
    return x, y


trn_loader = DataLoader(
    trn_ds,
    batch_size=BATCH_SIZE,
    num_workers=num_workers,
    shuffle=True,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=persistent_workers,
    prefetch_factor=4 if num_workers > 0 else None,
    generator=g,
    collate_fn=collate_train,
    drop_last=False,
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=persistent_workers,
    prefetch_factor=4 if num_workers > 0 else None,
    collate_fn=collate_train,
    drop_last=False,
)

trn_loader = DeviceDataLoader(trn_loader, device)
val_loader = DeviceDataLoader(val_loader, device)

model = Classifier()
model.freeze()
model = model.to(device).to(memory_format=torch.channels_last)

_DO_COMPILE = False
if _DO_COMPILE and hasattr(torch, "compile") and torch.cuda.is_available():
    try:
        model = torch.compile(model, mode="max-autotune")
    except Exception:
        pass




## === cell 6
def evaluate(model, val_loader):
    model.eval()
    loss_sum = 0.0
    acc_sum = 0.0
    n_batches = 0
    with torch.inference_mode():
        for images, labels in val_loader:
            out = model(images)
            loss = F.cross_entropy(out, labels)
            preds = out.argmax(dim=1)
            acc = (preds == labels).float().mean()
            loss_sum += float(loss)
            acc_sum += float(acc)
            n_batches += 1
    return {"val_loss": loss_sum / n_batches, "val_acc": acc_sum / n_batches}


def fit_fc(epochs, lr, model, train_loader, val_loader):
    optimizer = torch.optim.Adam(
        model.network.fc.parameters(), lr=lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for images, labels in train_loader:
            out = model(images)
            loss = F.cross_entropy(out, labels)
            train_loss_sum += float(loss.detach())
            n_batches += 1

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        result = evaluate(model, val_loader)
        result["train_loss"] = train_loss_sum / n_batches
        model.epoch_end(epoch, epochs, result)

    return model


EPOCHS = 16
LR = 3e-4
model = fit_fc(EPOCHS, LR, model, trn_loader, val_loader)



## === cell 7
test_csv = pd.read_csv(SAMPLE_SUB_PATH)
assert "image_id" in test_csv.columns and "label" in test_csv.columns


class CachedDecodeTestDataset(Dataset):
    def __init__(self, df, data_root, transforms_a, transforms_b):
        super().__init__()
        df = df.reset_index(drop=True)
        self.image_ids = df["image_id"].tolist()
        self.paths = [os.path.join(data_root, x) for x in self.image_ids]
        self.ta = transforms_a
        self.tb = transforms_b
        self._cache = {}

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]
        img = self._cache.get(image_id)
        if img is None:
            img = get_image_tensor(self.paths[index])  # CHW uint8
            self._cache[image_id] = img
        a = self.ta(img) if self.ta else img
        b = self.tb(img) if self.tb else img
        return a, b


test_ds = CachedDecodeTestDataset(
    test_csv[["image_id"]].copy(),
    TEST_DIR,
    transforms_a=test_transforms_a,
    transforms_b=test_transforms_b,
)


def collate_test_pair(batch):
    a, b = zip(*batch)
    xa = torch.stack(a, 0).contiguous(memory_format=torch.channels_last)
    xb = torch.stack(b, 0).contiguous(memory_format=torch.channels_last)
    return xa, xb


test_loader = DataLoader(
    test_ds,
    batch_size=256,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True if num_workers > 0 else False,
    prefetch_factor=4 if num_workers > 0 else None,
    collate_fn=collate_test_pair,
    drop_last=False,
)
test_loader = DeviceDataLoader(test_loader, device)

len(test_csv), len(test_ds)




## === cell 8
def inference_tta_hflip_pair(model, test_loader, device):
    model.to(device)
    model.eval()

    probs_a = []
    probs_b = []
    with torch.inference_mode():
        for images_a, images_b in test_loader:
            bs = images_a.shape[0]
            x = torch.cat(
                [
                    images_a,
                    torch.flip(images_a, dims=[3]),
                    images_b,
                    torch.flip(images_b, dims=[3]),
                ],
                dim=0,
            )
            logits = model(x).softmax(1)  # (4*bs, 5)
            p1a, p2a, p1b, p2b = logits.split(bs, dim=0)
            probs_a.append(((p1a + p2a) * 0.5).detach().cpu().numpy())
            probs_b.append(((p1b + p2b) * 0.5).detach().cpu().numpy())

    probs_a = np.concatenate(probs_a, axis=0)
    probs_b = np.concatenate(probs_b, axis=0)
    print("predictions shape A:", probs_a.shape, "B:", probs_b.shape)
    return probs_a, probs_b


pred_a, pred_b = inference_tta_hflip_pair(model, test_loader, device)
predictions = (pred_a + pred_b) * 0.5

assert predictions.shape[0] == len(test_csv), (predictions.shape, len(test_csv))

test_csv["label"] = predictions.argmax(axis=1).astype(int)
submission_path = "./submission.csv"
test_csv[["image_id", "label"]].to_csv(submission_path, index=False)

print(test_csv.head())
print("Wrote:", submission_path, "rows:", len(test_csv))
test_csv
