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
import warnings
import random
import pandas as pd
import cv2

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from torchvision import models

warnings.filterwarnings("ignore")

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

torch.backends.cudnn.benchmark = True




## === cell 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




## === cell 2
def get_tta(image_size, p=0.5):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    augs = A.Compose(
        [
            A.Resize(600, 600),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=p),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=p),
            A.OneOf(
                [
                    A.MotionBlur(p=0.3),
                    A.MedianBlur(blur_limit=3, p=0.3),
                    A.Blur(blur_limit=3, p=0.3),
                ],
                p=p,
            ),
            A.OneOf(
                [
                    A.OpticalDistortion(p=0.3),
                    A.GridDistortion(p=0.3),
                    A.PiecewiseAffine(p=0.3),
                ],
                p=p,
            ),
            A.OneOf(
                [
                    A.GaussNoise(p=0.5),
                ],
                p=p,
            ),
            A.Normalize(mean=imagenet_mean, std=imagenet_std),
            ToTensorV2(),
        ]
    )
    return augs




## === cell 3
class CassavaLeafDataset(Dataset):
    def __init__(self, root_dir, transforms, csv_path):
        self.root_dir = root_dir
        self.transform = transforms

        df = pd.read_csv(csv_path)
        if "image_id" not in df.columns:
            raise ValueError(f"CSV at {csv_path} must contain 'image_id' column.")
        self.image_ids = df["image_id"].astype(str).tolist()
        self.img_paths = [
            os.path.join(self.root_dir, img_id) for img_id in self.image_ids
        ]

    def __len__(self):
        return len(self.img_paths)

    def _read_rgb(self, img_path):
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = self._read_rgb(img_path)
        image = self.transform(image=image)["image"]
        return image




## === cell 4
DATA_DIR = "../input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

test_transforms = get_tta(546)

_BATCH_SIZE = 128 if torch.cuda.is_available() else 64
_PIN = device.type == "cuda"

_NUM_WORKERS = min(8, max(2, (os.cpu_count() or 2) // 2))


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


test_ds = CassavaLeafDataset(TEST_IMG_DIR, test_transforms, SAMPLE_SUB_PATH)
print("Test size:", len(test_ds), "batch:", _BATCH_SIZE, "workers:", _NUM_WORKERS)




## === cell 5
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(in_features=1280, out_features=5, bias=True)

ckpt_path = "../input/cassava-mobilenetv2/my-mobile.pth"
loaded = False
if os.path.exists(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    loaded = True
else:
    try:
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=5, bias=True)
        loaded = True
    except Exception as e:
        print(
            "Could not load torchvision pretrained weights; proceeding with random init. Error:",
            repr(e),
        )

model.to(device)
if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

if device.type == "cuda":
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass

model.eval()
print("Weights loaded:", loaded)




## === cell 6
df = pd.read_csv(SAMPLE_SUB_PATH)
df.shape[0]




## === cell 7
res_cpu = torch.zeros((df.shape[0], 5), device="cpu")
epochs = 10

with torch.inference_mode():
    n = df.shape[0]
    for ep in range(epochs):
        g = torch.Generator()
        g.manual_seed(0 + ep)

        loader = DataLoader(
            test_ds,
            batch_size=_BATCH_SIZE,
            shuffle=False,
            num_workers=_NUM_WORKERS,
            pin_memory=_PIN,
            drop_last=False,
            worker_init_fn=_seed_worker,
            generator=g,
            persistent_workers=(_NUM_WORKERS > 0),
            prefetch_factor=2 if _NUM_WORKERS > 0 else None,
        )

        offset = 0
        for images in loader:
            bs = images.shape[0]
            if device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last).to(
                    device, non_blocking=True
                )
            else:
                images = images.to(device, non_blocking=True)

            outputs = model(images)
            res_cpu[offset : offset + bs] += outputs.to("cpu", non_blocking=True)
            offset += bs

res_cpu = res_cpu / epochs
pred = torch.argmax(res_cpu, dim=1)
pred[:10], pred.shape




## === cell 8
labels = pred.detach().cpu().numpy().astype(int)
sub = df[["image_id"]].copy()
sub["label"] = labels
sub.to_csv("./submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
