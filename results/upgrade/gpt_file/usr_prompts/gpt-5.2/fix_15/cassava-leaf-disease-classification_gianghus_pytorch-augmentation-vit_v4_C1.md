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
seaborn==0.12.2
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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os



## === cell 1
train_path = "../input/cassava-leaf-disease-classification/train_images/"
test_path = "../input/cassava-leaf-disease-classification/test_images/"



## === cell 2
df_train = pd.read_csv(
    "../input/cassava-leaf-disease-classification/train.csv",
    dtype={"image_id": "string", "label": "int64"},
)

test_csv_path = "../input/cassava-leaf-disease-classification/test.csv"
if os.path.exists(test_csv_path):
    df_test = pd.read_csv(test_csv_path, dtype={"image_id": "string"})
else:
    test_ids = []
    with os.scandir(test_path) as it:
        for e in it:
            if e.is_file():
                n = e.name
                if n.endswith(".jpg") or n.endswith(".JPG"):
                    test_ids.append(n)
    test_ids.sort()
    df_test = pd.DataFrame({"image_id": pd.Series(test_ids, dtype="string")})



## === cell 3
_ = df_train.iloc[:0]



## === cell 4
import importlib

if importlib.util.find_spec("timm") is None:
    raise ImportError("timm is required but not found in this environment.")



## === cell 5
import sys
import warnings
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import cv2
from tqdm.auto import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from sklearn import model_selection

import struct
from typing import Optional, Dict, Tuple, List

warnings.simplefilter("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(False)

torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    cv2.setNumThreads(0)
except Exception:
    pass

_HAS_TV_JPEG = False



## === cell 6
df_train.shape



## === cell 7
pass



## === cell 8
df_train["kfold"] = -1
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
kf = model_selection.StratifiedKFold(n_splits=5, shuffle=False)
for f, (t_, v_) in enumerate(kf.split(X=df_train, y=df_train.label.values)):
    df_train.loc[v_, "kfold"] = f
print(df_train["kfold"].value_counts())

fold_dfs = {}
for fold in range(5):
    fold_dfs[fold] = {
        "train": df_train[df_train["kfold"] != fold].reset_index(drop=True),
        "valid": df_train[df_train["kfold"] == fold].reset_index(drop=True),
    }



## === cell 9
image_size = 384
epochs = 10
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




## === cell 10
class Augments:
    """Contains Train, Validation and Testing Augments"""

    train_augments = A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.8, 1.0),
                ratio=(0.75, 1.3333333333),
                p=1.0,
            ),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.Rotate(limit=45, p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.5, p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            A.CoarseDropout(p=0.5),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )

    valid_augments = A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
    )




## === cell 11
class EfficientNetModel(nn.Module):
    def __init__(self, num_classes=5, model_name="efficientnet_b7", pretrained=True):
        super(EfficientNetModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        if hasattr(self.model, "classifier"):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_classes)
        elif hasattr(self.model, "fc"):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class VITModel(nn.Module):
    def __init__(
        self, num_classes=5, model_name="vit_base_patch16_384", pretrained=True
    ):
        super(VITModel, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.head.in_features
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)




## === cell 12
class CustomDataset(Dataset):
    def __init__(
        self,
        df,
        num_classes=5,
        is_train=True,
        augments=None,
        image_size=image_size,
        folder_path=train_path,
        cache_images=False,
        image_store=None,  # dict: image_id -> RGB uint8 ndarray
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.num_classes = num_classes
        self.is_train = is_train
        self.augments = augments
        self.image_size = image_size
        self.folder_path = folder_path

        self._image_ids = self.df["image_id"].astype(str).values
        self._has_label = self.is_train and ("label" in self.df.columns)
        self._labels = self.df["label"].values if self._has_label else None

        self.cache_images = bool(cache_images)
        self._cache = {} if self.cache_images else None

        self.image_store = image_store

    @staticmethod
    def _read_rgb_fast(img_path: str) -> np.ndarray:
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(img)

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, idx):
        image_id = self._image_ids[idx]

        img = None
        if self.image_store is not None:
            img = self.image_store.get(image_id, None)

        if img is None:
            img_path = os.path.join(self.folder_path, image_id)
            if self.cache_images:
                img = self._cache.get(image_id)
                if img is None:
                    img = self._read_rgb_fast(img_path)
                    self._cache[image_id] = img
            else:
                img = self._read_rgb_fast(img_path)

        if self.augments:
            img = self.augments(image=img)["image"]

        if self._has_label:
            label = int(self._labels[idx])
            return img, label
        return img




## === cell 13
def train_one_cycle(model, dataloader, loss_fn, optim, scaler):
    model.train()

    run_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device, non_blocking=True).contiguous(
            memory_format=torch.channels_last
        )
        labels = labels.to(device, non_blocking=True).long()

        optim.zero_grad(set_to_none=True)
        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(inputs)
            train_loss = loss_fn(outputs, labels)

        scaler.scale(train_loss).backward()
        scaler.step(optim)
        scaler.update()

        run_loss += float(train_loss.item())

        with torch.no_grad():
            preds = torch.argmax(outputs, 1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

    acc = correct / max(1, total)
    print(f"Training Accuracy: {acc:.3f}")
    floss = run_loss / len(dataloader)
    return (acc, floss)


def valid_one_cycle(model, dataloader, loss_fn):
    model.eval()
    with torch.no_grad():
        run_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            labels = labels.to(device, non_blocking=True).long()

            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
                valid_loss = loss_fn(outputs, labels)

            run_loss += float(valid_loss.item())

            preds = torch.argmax(outputs, 1)
            correct += int((preds == labels).sum().item())
            total += int(labels.numel())

        acc = correct / max(1, total)
        print(f"Valid Accuracy: {acc:.3f}")
        floss = run_loss / len(dataloader)
    return (acc, floss, model)




## === cell 14
def plot_results(train_acc, valid_acc, train_loss, valid_loss, nb_epochs):
    return




## === cell 15
def _seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)




## === cell 16
def _build_image_store(df, folder_path: str, resize_hw=(image_size, image_size)):
    image_ids = df["image_id"].astype(str).tolist()
    store: Dict[str, np.ndarray] = {}
    for image_id in image_ids:
        img_path = os.path.join(folder_path, image_id)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        store[image_id] = np.ascontiguousarray(img)
    return store




## === cell 17
def _maybe_compile(model: nn.Module) -> nn.Module:
    if hasattr(torch, "compile"):
        try:
            return torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except Exception:
            return model
    return model




## === cell 18
def run(fold):
    train_fold = fold_dfs[fold]["train"]
    valid_fold = fold_dfs[fold]["valid"]

    train_store = _build_image_store(
        train_fold.iloc[:0], train_path, resize_hw=(image_size, image_size)
    )  # empty, no cost
    valid_store = _build_image_store(
        valid_fold, train_path, resize_hw=(image_size, image_size)
    )

    train_set = CustomDataset(
        df=train_fold,
        augments=Augments.train_augments,
        is_train=True,
        folder_path=train_path,
        cache_images=False,
        image_store=None,
    )
    valid_set = CustomDataset(
        df=valid_fold,
        augments=Augments.valid_augments,
        is_train=True,
        folder_path=train_path,
        cache_images=False,  # store already caches decode; no need for per-worker dict cache
        image_store=valid_store,
    )

    cpu = os.cpu_count() or 4
    nw = min(8, max(2, cpu // 2))

    g = torch.Generator()
    g.manual_seed(SEED)

    pin = torch.cuda.is_available()

    train = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,
        drop_last=False,
        num_workers=nw,
        persistent_workers=True if nw > 0 else False,
        prefetch_factor=4 if nw > 0 else None,
        worker_init_fn=_seed_worker if nw > 0 else None,
        generator=g,
    )
    valid = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin,
        num_workers=nw,
        persistent_workers=True if nw > 0 else False,
        prefetch_factor=4 if nw > 0 else None,
        worker_init_fn=_seed_worker if nw > 0 else None,
        generator=g,
    )

    model = VITModel(num_classes=5, model_name="vit_base_patch16_384").to(device)
    model = model.to(memory_format=torch.channels_last)
    model = _maybe_compile(model)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)
    loss_fn = nn.CrossEntropyLoss().to(device)

    scaler = GradScaler(enabled=torch.cuda.is_available())

    train_accs = []
    valid_accs = []
    train_losses = []
    valid_losses = []
    best_acc = 0.0
    best_path = f"vit_base_p16_384_fold_{fold}_model.pth"

    for epoch in range(epochs):
        print(f"{'-'*20} EPOCH: {epoch}/{epochs} {'-'*20}")

        current_train_acc, current_train_loss = train_one_cycle(
            model=model, dataloader=train, loss_fn=loss_fn, optim=optim, scaler=scaler
        )
        train_accs.append(current_train_acc)
        train_losses.append(current_train_loss)

        current_val_acc, current_val_loss, op_model = valid_one_cycle(
            model=model, dataloader=valid, loss_fn=loss_fn
        )
        valid_accs.append(current_val_acc)
        valid_losses.append(current_val_loss)

        if best_acc < current_val_acc:
            best_acc = current_val_acc
            print("Saving Model for this epoch...")
            torch.save(op_model.state_dict(), best_path)

    del train_set, valid_set, train, valid, train_store, valid_store
    torch.cuda.empty_cache()
    print(f"Best Accuracy of {fold} fold: {best_acc:.3f}")
    return best_path




## === cell 19
best_model_path = None
for fold in range(1):
    best_model_path = run(fold)



## === cell 20
model = VITModel(num_classes=5, model_name="vit_base_patch16_384").to(device)

if best_model_path is not None and os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))

model = model.to(memory_format=torch.channels_last)
model = _maybe_compile(model)




## === cell 21
def show_predictions(model, data_loader):
    preds = []
    model = model.eval()
    with torch.inference_mode():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                inputs, _ = batch
            else:
                inputs = batch
            inputs = inputs.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(inputs)
            pred = torch.argmax(outputs, 1).detach().cpu().numpy()
            preds.extend(pred.tolist())
    return preds




## === cell 22
test_store = _build_image_store(df_test, test_path, resize_hw=(image_size, image_size))

test_set = CustomDataset(
    df=df_test,
    augments=Augments.valid_augments,
    folder_path=test_path,
    is_train=False,
    cache_images=False,
    image_store=test_store,
)

cpu = os.cpu_count() or 4
nw = min(8, max(2, cpu // 2))
g = torch.Generator()
g.manual_seed(SEED)

test = DataLoader(
    test_set,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=nw,
    persistent_workers=True if nw > 0 else False,
    prefetch_factor=4 if nw > 0 else None,
    worker_init_fn=_seed_worker if nw > 0 else None,
    generator=g,
)



## === cell 23
predictions = show_predictions(model, test)
len(predictions), df_test.shape



## === cell 24
df_test = df_test.copy()
df_test["label"] = predictions
df_test.head()



## === cell 25
sub = df_test[["image_id", "label"]].copy()
sub["label"] = sub["label"].astype(int)

assert (
    len(sub) == len(predictions) == len(df_test)
), f"Length mismatch: sub={len(sub)} preds={len(predictions)} df_test={len(df_test)}"

sub.to_csv("submission.csv", index=None)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
