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

# 5. Target score

0.5395899547270685

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00346) has done: 'I replace the missing EfficientNet import with a fallback that uses the already‑defined ResNet50 model, simplify the weight‑loading function, and adjust the validation transforms so they output images of the expected size (128×128). These changes eliminate the import error, define `model` properly, and ensure the inference loop runs, producing a valid `submission.csv` that can be scored.'
- What this solution (achieved 0.0033) has done: 'I enable ImageNet pretrained weights for the ResNet50 model (setting `pretrained=True`) so the network provides meaningful feature representations instead of random ones, which should raise the micro‑F1 score toward the target. I also raise the prediction threshold slightly from 0.10 to 0.20 to reduce excessive false positives, helping precision and thus improving the F1 metric while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import random
import time
import logging
from contextlib import contextmanager
from functools import partial

import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torchvision.models as M

import albumentations as A
from albumentations import Compose, RandomResizedCrop, Resize, Normalize
from albumentations.pytorch import ToTensorV2

import sklearn.metrics
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 777
seed_torch(SEED)

submission = pd.read_csv("../input/imet-2020-fgvc7/sample_submission.csv")



## === cell 1
N_CLASSES = 3474


class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["id"].values[idx]
        file_path = f"../input/imet-2020-fgvc7/train/{file_name}.png"
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        label = self.df["attribute_ids"].values[idx]
        target = torch.zeros(N_CLASSES, dtype=torch.float32)
        for cls in label.split():
            target[int(cls)] = 1.0
        return image, target


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df["id"].values[idx]
        file_path = f"../input/imet-2020-fgvc7/test/{file_name}.png"
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image




## === cell 2
HEIGHT = 128
WIDTH = 128


def get_transforms(*, data):
    assert data in ("train", "valid")
    if data == "train":
        return Compose(
            [
                RandomResizedCrop(HEIGHT, WIDTH),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:  # valid / test
        return Compose(
            [
                Resize(HEIGHT, WIDTH),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )




## === cell 3
batch_size = 128

train_df = pd.read_csv("../input/imet-2020-fgvc7/train.csv")
val_frac = 0.1
val_size = int(len(train_df) * val_frac)
train_part = train_df.iloc[:-val_size].reset_index(drop=True)
val_part = train_df.iloc[-val_size:].reset_index(drop=True)

train_dataset = TrainDataset(train_part, transform=get_transforms(data="train"))
val_dataset = TrainDataset(val_part, transform=get_transforms(data="valid"))
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

test_dataset = TestDataset(submission, transform=get_transforms(data="valid"))
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     66             schema_kwargs["strict"] = strict
---> 67             config = schema_cls(**schema_kwargs)
     68             validated_kwargs = config.model_dump()

/usr/local/lib/python3.11/dist-packages/pydantic/main.py in __init__(self, **data)
    249         __tracebackhide__ = True
--> 250         validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
    251         if self is not validated_self:

ValidationError: 2 validation errors for InitSchema
scale
  Input should be a valid tuple [type=tuple_type, input_value=128, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=128, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1579740653.py in <cell line: 0>()
      7 val_part = train_df.iloc[-val_size:].reset_index(drop=True)
      8 
----> 9 train_dataset = TrainDataset(train_part, transform=get_transforms(data="train"))
     10 val_dataset = TrainDataset(val_part, transform=get_transforms(data="valid"))
     11 train_loader = DataLoader(

/tmp/ipykernel_55/3319997832.py in get_transforms(data)
      8         return Compose(
      9             [
---> 10                 RandomResizedCrop(HEIGHT, WIDTH),
     11                 Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
     12                 ToTensorV2(),

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in custom_init(self, *args, **kwargs)
    103                 full_kwargs, param_names, strict = cls._process_init_parameters(original_init, args, kwargs)
    104 
--> 105                 validated_kwargs = cls._validate_parameters(
    106                     dct["InitSchema"],
    107                     full_kwargs,

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     69             validated_kwargs.pop("strict", None)
     70         except ValidationError as e:
---> 71             raise ValueError(str(e)) from e
     72         except Exception as e:
     73             if strict:

ValueError: 2 validation errors for InitSchema
scale
  Input should be a valid tuple [type=tuple_type, input_value=128, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=128, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

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


resnet50 = partial(ResNet, net_cls=M.resnet50)




## === cell 5
def load_pretrained_weights2(
    model, model_name, weights_path=None, load_fc=True, advprop=False
):
    """Placeholder that does nothing when EfficientNet is unavailable."""
    LOGGER.info(
        f"Skipped loading pretrained weights for {model_name} (package not installed)."
    )
    return




## === cell 6
model = resnet50(num_classes=N_CLASSES, pretrained=True)
model.to(device)



## === cell 7
criterion = nn.BCEWithLogitsLoss(reduction="none")
optimizer = Adam(model.fresh_params(), lr=1e-3)
scheduler = ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=1, verbose=True
)



## === cell 8
epochs = 3
best_val_f1 = 0.0
best_threshold = 0.20
best_state = model.state_dict()  # fallback in case no improvement

for epoch in range(epochs):
    model.train()
    train_losses = []
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())

    model.eval()
    val_targets = []
    val_preds = []
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu().numpy()
            val_preds.append(probs)
            val_targets.append(targets.cpu().numpy())
    val_preds = np.concatenate(val_preds)
    val_targets = np.concatenate(val_targets)

    thresholds = np.arange(0.10, 0.51, 0.05)
    best_thr = best_threshold
    best_f1 = 0.0
    for thr in thresholds:
        pred_bin = val_preds > thr
        f1 = sklearn.metrics.f1_score(val_targets, pred_bin, average="micro")
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    LOGGER.info(
        f"Epoch {epoch+1}: Train loss {np.mean(train_losses):.4f}, Val F1 {best_f1:.4f} at thr {best_thr:.2f}"
    )

    if best_f1 > best_val_f1:
        best_val_f1 = best_f1
        best_threshold = best_thr
        best_state = model.state_dict()

    scheduler.step(best_val_f1)

model.load_state_dict(best_state)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/807438738.py in <cell line: 0>()
      7     model.train()
      8     train_losses = []
----> 9     for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"):
     10         images = images.to(device, non_blocking=True)
     11         targets = targets.to(device, non_blocking=True)

NameError: name 'train_loader' is not defined

## === cell 9
with timer("inference"):
    model.eval()
    preds = []
    for i, images in tqdm(
        enumerate(test_loader), total=len(test_loader), desc="Inference"
    ):
        images = images.to(device, non_blocking=True)
        with torch.no_grad():
            y_preds = model(images)
        preds.append(torch.sigmoid(y_preds).cpu().numpy())



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2255649873.py in <cell line: 0>()
      3     preds = []
      4     for i, images in tqdm(
----> 5         enumerate(test_loader), total=len(test_loader), desc="Inference"
      6     ):
      7         images = images.to(device, non_blocking=True)

NameError: name 'test_loader' is not defined

## === cell 10
threshold = best_threshold
predictions = np.concatenate(preds) > threshold

for i, row in enumerate(predictions):
    ids = np.nonzero(row)[0]
    submission.at[i, "attribute_ids"] = " ".join(map(str, ids))

submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2341341224.py in <cell line: 0>()
      1 threshold = best_threshold
----> 2 predictions = np.concatenate(preds) > threshold
      3 
      4 for i, row in enumerate(predictions):
      5     ids = np.nonzero(row)[0]

ValueError: need at least one array to concatenate
