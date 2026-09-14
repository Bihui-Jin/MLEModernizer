# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Label artwork images with significant attributes.

## Metric
Micro averaged F1 score.

## Submission Format
```
id,attribute_ids
00011f01965f141f5d1eea6592fa9862,0 1 2
00014abc91ed3e4bf1663fde8136fe80,0 1 2
0002e2054e303badc1a33463f6fb7973,0 1 2
```

## Dataset
Multiple modalities can be expected and the camera sources are unknown. The photographs are often centered for objects, and in the case where the museum artifact is an entire room, the images are scenic in nature.

Each object is annotated by a single annotator without a verification step. You should consider these annotations noisy.

The filename of each image is its `id`.

- **train.csv** gives the `attribute_ids` for the train images in **/train**
- **/test** contains the test images. You must predict the `attribute_ids` for these images.
- **sample_submission.csv** contains a submission in the correct format
- **labels.csv** provides descriptions of the attributes

# 2. Python version

3.8

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
scipy==1.15.3
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
            description.md (81 lines)
            labels.csv (3475 lines)
            labels.csv.zip (28.4 kB)
            sample_submission.csv (21319 lines)
            sample_submission.csv.zip (426.2 kB)
            test.zip (3.8 GB)
            train.csv (120802 lines)
            train.csv.zip (3.2 MB)
            train.zip (21.4 GB)
            imet-2020-fgvc7/
                description.md (81 lines)
                labels.csv (3475 lines)
                ... and 7 other files
                imet-2020-fgvc7/
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
            test/
                c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                ... and 21316 other files
                test/
            train/
                cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                ... and 120799 other files
                train/
        input/
            description.md (81 lines)
            labels.csv (3475 lines)
            labels.csv.zip (28.4 kB)
            sample_submission.csv (21319 lines)
            sample_submission.csv.zip (426.2 kB)
            test.zip (3.8 GB)
            train.csv (120802 lines)
            train.csv.zip (3.2 MB)
            train.zip (21.4 GB)
            imet-2020-fgvc7/
                description.md (81 lines)
                labels.csv (3475 lines)
                ... and 7 other files
                imet-2020-fgvc7/
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
            test/
                c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                ... and 21316 other files
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
            train/
                cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                ... and 120799 other files
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
        working/
            imet-2020-fgvc7/
                description.md (81 lines)
                labels.csv (3475 lines)
                ... and 7 other files
                imet-2020-fgvc7/
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
```

-> data/imet-2020-fgvc7/labels.csv has 3474 rows and 2 columns.
The columns are: attribute_id, attribute_name

-> data/imet-2020-fgvc7/sample_submission.csv has 21318 rows and 2 columns.
The columns are: id, attribute_ids

-> data/imet-2020-fgvc7/train.csv has 120801 rows and 2 columns.
The columns are: id, attribute_ids

-> data/labels.csv has 3474 rows and 2 columns.
The columns are: attribute_id, attribute_name

-> data/sample_submission.csv has 21318 rows and 2 columns.
The columns are: id, attribute_ids

-> data/train.csv has 120801 rows and 2 columns.
The columns are: id, attribute_ids

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, random, time
from contextlib import contextmanager

import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import torchvision.models as M
from functools import partial
from tqdm import tqdm
import sklearn.metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f"[{name}] start")
    yield
    LOGGER.info(f"[{name}] done in {time.time() - t0:.0f} s.")


def init_logger(log_file="train.log"):
    from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler

    log_format = "%(asctime)s %(levelname)s %(message)s"

    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter(log_format))

    file_handler = FileHandler(log_file)
    file_handler.setFormatter(Formatter(log_format))

    logger = getLogger("Herbarium")
    logger.setLevel(DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


LOG_FILE = "train.log"
LOGGER = init_logger(LOG_FILE)


def seed_torch(seed=777):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


SEED = 777
seed_torch(SEED)




## === cell 1
submission = pd.read_csv("../input/imet-2020-fgvc7/sample_submission.csv")




## === cell 2
N_CLASSES = 3474


class TrainDataset(Dataset):
    def __init__(self, df, labels, transform=None):
        self.transform = transform
        self.image_paths = [
            f"../input/imet-2020-fgvc7/train/{fid}.png" for fid in df["id"].values
        ]
        targets_list = []
        for lbl in labels.values:
            idx = [int(cls) for cls in lbl.split()] if lbl else []
            targets_list.append(idx)

        self.targets = torch.zeros(len(df), N_CLASSES, dtype=torch.float32)
        for i, idxs in enumerate(targets_list):
            if idxs:  # non‑empty list
                self.targets[i, idxs] = 1.0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        target = self.targets[idx]
        return image, target


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.transform = transform
        self.image_paths = [
            f"../input/imet-2020-fgvc7/test/{fid}.png" for fid in df["id"].values
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image




## === cell 3
from albumentations import Compose, RandomResizedCrop, Resize, Normalize
from albumentations.pytorch import ToTensorV2

HEIGHT = 224
WIDTH = 224
batch_size = 64  # unchanged batch size


def get_transforms(*, data):
    assert data in ("train", "valid")
    if data == "train":
        return Compose(
            [
                RandomResizedCrop(
                    size=(HEIGHT, WIDTH),
                    scale=(0.8, 1.0),
                ),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:  # valid
        return Compose(
            [
                Resize(height=HEIGHT, width=WIDTH),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


test_dataset = TestDataset(submission, transform=get_transforms(data="valid"))
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)




## === cell 4
class AvgPool(nn.Module):
    def forward(self, x):
        return F.avg_pool2d(x, x.shape[2:])


class ResNet(nn.Module):
    def __init__(
        self, num_classes, pretrained=False, net_cls=M.resnet50, dropout=False
    ):
        super().__init__()
        self.net = net_cls(pretrained=pretrained)
        self.net.avgpool = AvgPool()
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.fc.parameters()

    def forward(self, x):
        return self.net(x)


class DenseNet(nn.Module):
    def __init__(self, num_classes, pretrained=False, net_cls=M.densenet121):
        super().__init__()
        self.net = net_cls(pretrained=pretrained)
        self.avg_pool = AvgPool()
        self.net.classifier = nn.Linear(self.net.classifier.in_features, num_classes)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = F.relu(out, inplace=True)
        out = self.avg_pool(out).view(out.size(0), -1)
        out = self.net.classifier(out)
        return out


class ShuffleNet(nn.Module):
    def __init__(
        self, num_classes, pretrained=False, net_cls=M.shufflenet_v2_x1_0, dropout=False
    ):
        super().__init__()
        self.net = net_cls(pretrained=pretrained)
        if dropout:
            self.net.fc = nn.Sequential(
                nn.Dropout(),
                nn.Linear(self.net.fc.in_features, num_classes),
            )
        else:
            self.net.fc = nn.Linear(self.net.fc.in_features, num_classes)

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        return self.net(x)


class SqueezeNet(nn.Module):
    def __init__(
        self, num_classes, pretrained=False, net_cls=M.squeezenet1_0, dropout=False
    ):
        super().__init__()
        self.net = net_cls(pretrained=pretrained)

        fin_conv = nn.Conv2d(512, num_classes, kernel_size=1)
        self.net.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            fin_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        out = self.net.features(x)
        out = self.net.classifier(out)
        out = out.view(-1, N_CLASSES)
        return out


class MobileNet(nn.Module):
    def __init__(
        self, num_classes, pretrained=False, net_cls=M.mobilenet_v2, dropout=False
    ):
        super().__init__()
        self.net = net_cls(pretrained=pretrained)

        self.net.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.net.last_channel, num_classes),
        )

    def fresh_params(self):
        return self.net.classifier.parameters()

    def forward(self, x):
        return self.net(x)


resnet50 = partial(ResNet, net_cls=M.resnet50)
densenet121 = partial(DenseNet, net_cls=M.densenet121)
shufflenet = partial(ShuffleNet, net_cls=M.shufflenet_v2_x1_0)
squeezenet = partial(SqueezeNet, net_cls=M.squeezenet1_0)
mobilenet = partial(MobileNet, net_cls=M.mobilenet_v2)




## === cell 5
model = densenet121(num_classes=N_CLASSES, pretrained=True)
model.to(device)

train_df = pd.read_csv("../input/imet-2020-fgvc7/train.csv")

VAL_SIZE = 2000
val_df = (
    train_df.sample(VAL_SIZE, random_state=SEED)
    if len(train_df) > VAL_SIZE
    else train_df
)

train_subset = train_df.drop(val_df.index).reset_index(drop=True)

train_dataset = TrainDataset(
    train_subset, train_subset["attribute_ids"], transform=get_transforms(data="train")
)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

val_dataset = TrainDataset(
    val_df, val_df["attribute_ids"], transform=get_transforms(data="valid")
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
)

criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

model.train()
for epoch in range(4):  # reduced epochs to meet time limit
    epoch_loss = 0.0
    for images, targets in tqdm(train_loader, desc=f"training epoch {epoch+1}"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, targets)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item()
    LOGGER.info(
        f"Epoch {epoch+1} finished – avg loss {epoch_loss/len(train_loader):.4f}"
    )

model.eval()
val_preds = []
with torch.no_grad():
    for images, _ in tqdm(val_loader, desc="validation"):
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outs = torch.sigmoid(model(images)).cpu().numpy()
        val_preds.append(outs)
val_preds = np.concatenate(val_preds)

true_mat = torch.zeros((len(val_df), N_CLASSES), dtype=torch.int32).numpy()
for i, lbl in enumerate(val_df["attribute_ids"].values):
    for cls in lbl.split():
        true_mat[i, int(cls)] = 1

best_thr = 0.0
best_f1 = 0.0
for thr in np.arange(0.05, 0.61, 0.02):  # coarser search reduces compute
    pred_bin = (val_preds > thr).astype(int)
    f1 = sklearn.metrics.f1_score(true_mat, pred_bin, average="micro")
    if f1 > best_f1:
        best_f1 = f1
        best_thr = thr
LOGGER.info(
    f"Best threshold determined on validation: {best_thr:.2f} (micro‑F1={best_f1:.4f})"
)




## === cell 6
with timer("inference"):
    model.eval()
    preds = []
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    for i, images in tk0:
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                y_preds = model(images)
            preds.append(torch.sigmoid(y_preds).cpu().numpy())




## === cell 7
threshold = best_thr  # calibrated threshold from validation
predictions = np.concatenate(preds) > threshold

for i, row in enumerate(predictions):
    ids = np.nonzero(row)[0]
    submission.iloc[i].attribute_ids = " ".join([str(x) for x in ids])

submission.to_csv("submission.csv", index=False)
submission.head()
