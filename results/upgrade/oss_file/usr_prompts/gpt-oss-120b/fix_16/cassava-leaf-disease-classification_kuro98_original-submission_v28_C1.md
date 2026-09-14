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

3.12

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

# 5. Target score

0.8925657298277425

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.22085) has done: 'Implemented a robust image listing that skips directories and non‑JPEG files, preventing the `IsADirectoryError`. This ensures the DataLoader only processes actual image files, leading to a correctly sized prediction list and a valid `submission.csv` matching the required number of rows.'
- What this solution (achieved 0.78999) has done: 'The changes enable mixed‑precision training and inference with torch.cuda.amp, which greatly speeds up GPU computation while keeping the model architecture, epoch count and data pipeline unchanged. DataLoaders now use `persistent_workers=True` to avoid the overhead of repeatedly spawning worker processes. All modifications are limited to performance‑related settings and preserve the original training/validation logic and final predictions.'
- What this solution (achieved 0.74477) has done: 'I speed up the pipeline by (1) allowing cuDNN to use the fastest kernels, (2) increasing the batch size and worker count, (3) simplifying the image preprocessing (center‑crop directly to the target size instead of a costly 600 → 224 resize), and (4) enabling TF‑32 matrix math for faster GPU ops. These changes keep the model, loss, optimizer, epoch count, and overall training logic unchanged, so the results remain equivalent while the runtime drops well below the 600‑second limit.'

# 9. Code solution

## === cell 0
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3878653887.py in <cell line: 0>()
      1 # Enable CuDNN benchmark for faster convolution kernels and set high‑precision matmul.
----> 2 torch.backends.cudnn.benchmark = True
      3 torch.set_float32_matmul_precision("high")
      4 
      5 

NameError: name 'torch' is not defined

## === cell 1
class TrainCassavaDataset(VisionDataset):
    """Dataset for training/validation that pre‑loads images into RAM using parallel I/O."""

    def __init__(self, dataframe, img_dir, transform=None):
        super().__init__(root=img_dir)
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

        from concurrent.futures import ThreadPoolExecutor

        file_paths = [os.path.join(self.root, fname) for fname in self.df["image_id"]]

        def load(p):
            return read_image(p)  # (3, H, W), uint8

        with ThreadPoolExecutor() as executor:
            self.images = list(executor.map(load, file_paths))

        self.labels = torch.tensor(
            self.df["label"].astype(int).values, dtype=torch.long
        )

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx].item())
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.df)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/593544259.py in <cell line: 0>()
----> 1 class TrainCassavaDataset(VisionDataset):
      2     """Dataset for training/validation that pre‑loads images into RAM using parallel I/O."""
      3 
      4     def __init__(self, dataframe, img_dir, transform=None):
      5         super().__init__(root=img_dir)

NameError: name 'VisionDataset' is not defined

## === cell 2
class TestCassavaDataset(Dataset):
    """Dataset for inference that pre‑loads images into RAM using parallel I/O."""

    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.filenames = sorted(
            [
                f
                for f in os.listdir(img_dir)
                if f.lower().endswith(".jpg")
                and os.path.isfile(os.path.join(img_dir, f))
            ]
        )
        from concurrent.futures import ThreadPoolExecutor

        file_paths = [os.path.join(self.img_dir, f) for f in self.filenames]

        def load(p):
            return read_image(p).float() / 255.0

        with ThreadPoolExecutor() as executor:
            self.images = list(executor.map(load, file_paths))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform:
            img = self.transform(img)
        filename = self.filenames[idx]
        return img, filename




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4069675283.py in <cell line: 0>()
----> 1 class TestCassavaDataset(Dataset):
      2     """Dataset for inference that pre‑loads images into RAM using parallel I/O."""
      3 
      4     def __init__(self, img_dir, transform=None):
      5         self.img_dir = img_dir

NameError: name 'Dataset' is not defined

## === cell 3
import os
import pandas as pd
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import VisionDataset
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode, v2
from torchvision import models
from torchvision.models import ViT_B_16_Weights
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

img_size = 224
batch_size = 64  # larger batch reduces number of optimizer steps per epoch
num_workers = 0  # disabled after pre‑loading images to RAM
num_classes = 5
tta = False

model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

if isinstance(model.heads, torch.nn.Sequential):
    in_features = model.heads[-1].in_features
else:
    in_features = model.heads.in_features
model.heads = torch.nn.Linear(in_features, num_classes)

model = model.to(device)
model = torch.compile(model, mode="reduce-overhead")  # JIT‑compile for speed
