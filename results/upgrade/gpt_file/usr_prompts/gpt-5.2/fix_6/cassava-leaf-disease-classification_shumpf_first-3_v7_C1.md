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
pillow==11.3.0
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
from torchvision.models import resnext50_32x4d as resnext
from torchvision.models import vgg16
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import os
from tqdm import tqdm

from matplotlib import cm
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.model_selection import train_test_split

print("gpu,", torch.cuda.is_available())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

try:
    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass

try:
    torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
except Exception:
    pass




## === cell 1
class LeafNet(nn.Module):
    def __init__(self, hidden_size=1280, num_cls=5):
        super(LeafNet, self).__init__()
        self.model_base = resnext()  # architecture unchanged
        self.linear1 = nn.Linear(1000, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_cls)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.loss = nn.CrossEntropyLoss()


    def forward(self, inputs):
        img, labels = inputs
        x = self.model_base(img)
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        x = self.linear3(x)
        ls = self.loss(x, labels)
        return ls

    def inference(self, inputs):
        img = inputs
        x = self.model_base(img)
        x = self.dropout(self.relu(self.linear1(x)))
        x = self.dropout(self.relu(self.linear2(x)))
        x = self.linear3(x)
        x = torch.argmax(x, dim=1)
        return x




## === cell 2
import albumentations as A
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    RandomRotate90,
    Transpose,
    ShiftScaleRotate,
    HueSaturationValue,
    RandomResizedCrop,
    RandomBrightnessContrast,
    OneOf,
    Compose,
    Normalize,
    CoarseDropout,
    CenterCrop,
    Resize,
)
from albumentations.pytorch import ToTensorV2

_TRAIN_CSV_CACHE = {}


def _read_train_csv(csv_path: str) -> pd.DataFrame:
    df = _TRAIN_CSV_CACHE.get(csv_path)
    if df is None:
        df = pd.read_csv(csv_path)
        _TRAIN_CSV_CACHE[csv_path] = df
    return df


def _imread_rgb_fast(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is not None:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    try:
        return np.array(Image.open(path).convert("RGB"))
    except Exception:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


class Leafdataset(Dataset):
    def __init__(self, path, mode_train=False, num_cls=5):
        self.mode_train = mode_train

        image_base = os.path.join(path, "train_images/")
        csv_path = os.path.join(path, "train.csv")
        info = _read_train_csv(csv_path)

        labels = info["label"].astype(int).to_numpy()
        img_names = info["image_id"].to_numpy()

        imgs = [os.path.join(image_base, n) for n in img_names.tolist()]

        split = int(0.8 * len(imgs))
        if self.mode_train:
            self.imgs = imgs[:split]
            self.labels = labels[:split]
        else:
            self.imgs = imgs[split:]
            self.labels = labels[split:]

        self.preprocess = A.Compose(
            [
                CenterCrop(256, 256, p=1.0),
                Resize(256, 256),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ]
        )

        self.augmentation = A.Compose(
            [
                RandomResizedCrop(size=(256, 256)),
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                HueSaturationValue(
                    hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
                ),
                RandomBrightnessContrast(
                    brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
                ),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                CoarseDropout(p=0.5),
                ToTensorV2(p=1.0),
            ]
        )
        print(len(self.labels), len(self.imgs))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img = _imread_rgb_fast(img_name)

        if self.mode_train:
            img = self.augmentation(image=img)["image"].float()
        else:
            img = self.preprocess(image=img)["image"].float()

        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return img, label


class Leafdataset_val(Dataset):
    def __init__(self, path, num_cls=5):
        import glob

        image_base = os.path.join(path, "test_images/")
        self.imgs = sorted(glob.glob(os.path.join(image_base, "*.jpg")))

        self.preprocess = A.Compose(
            [
                CenterCrop(256, 256, p=1.0),
                Resize(256, 256),
                Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
                ToTensorV2(p=1.0),
            ]
        )
        print(len(self.imgs))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img = _imread_rgb_fast(img_name)

        img = self.preprocess(image=img)["image"].float()
        return os.path.basename(img_name), img




## === cell 3
def validate_test(model, dataset_test):
    model.eval()
    predictions = []
    ids = []
    for i, (img_name, img) in enumerate(dataset_test):
        img = img.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model.inference(img).cpu().detach().item()
        predictions.append(pred)
        ids.append(img_name)
    sub = pd.DataFrame({"image_id": ids, "label": predictions})
    sub.to_csv("./submission.csv", index=False)


def validate(model, holdout_loader, device):
    model.eval()
    num_corr = 0
    num_total = 0
    with torch.inference_mode():
        for imgs, labels in holdout_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred = model.inference(imgs)
            num_corr += int((pred == labels).sum().item())
            num_total += int(labels.numel())
    print("accuracy is", num_corr * 1.0 / max(1, num_total), num_corr, "/", num_total)




## === cell 4
def inference(model, test_loader, device):
    model.to(device)
    model.eval()

    predictions = []
    ids = []
    with torch.inference_mode():
        for img_name, img in tqdm(test_loader, total=len(test_loader)):
            batch_ids = list(img_name)
            img = img.to(device, non_blocking=True)
            pred = model.inference(img).cpu().numpy().tolist()
            ids.extend(batch_ids)
            predictions.extend(pred)

    sub = pd.DataFrame({"image_id": ids, "label": predictions})
    sub.to_csv("./submission.csv", index=False)
    return sub.head()




## === cell 5
lr = 1e-4
batch_size = 50
num_epochs = 10
path = "../input/cassava-leaf-disease-classification/"

dataset_train = Leafdataset(path, mode_train=True)
dataset_test = Leafdataset(path, mode_train=False)
dataset_val = Leafdataset_val(path)

cpu = os.cpu_count() or 2
num_workers = min(8, max(2, cpu // 2))
pin = torch.cuda.is_available()


def _seed_worker(worker_id):
    wseed = (seed + worker_id) % (2**32 - 1)
    np.random.seed(wseed)
    torch.manual_seed(wseed)


g = torch.Generator()
g.manual_seed(seed)

train_loader = DataLoader(
    dataset_train,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    generator=g,
)

holdout_loader = DataLoader(
    dataset_test,
    batch_size=64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

model = LeafNet().float().to(device)

if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="max-autotune")
    except Exception as e:
        print("torch.compile unavailable/fallback:", repr(e))

optimizer = optim.Adam(model.parameters(), lr=lr)

for epoch in range(num_epochs):
    model.train()
    running = 0.0
    n = 0
    for imgs, labels in tqdm(train_loader, desc=f"train epoch {epoch+1}/{num_epochs}"):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = model((imgs, labels))
        loss.backward()
        optimizer.step()

        running += float(loss.detach().cpu().item()) * imgs.size(0)
        n += imgs.size(0)

    print(f"epoch {epoch+1}: train_loss={running/max(1,n):.4f}")
    validate(model, holdout_loader, device)

test_loader = DataLoader(
    dataset_val,
    batch_size=64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)

_ = inference(model, test_loader, device)

print("Wrote ./submission.csv")
print(pd.read_csv("./submission.csv").head())
print("submission shape:", pd.read_csv("./submission.csv").shape)
assert os.path.exists("./submission.csv") and "./submission.csv".endswith(".csv")
