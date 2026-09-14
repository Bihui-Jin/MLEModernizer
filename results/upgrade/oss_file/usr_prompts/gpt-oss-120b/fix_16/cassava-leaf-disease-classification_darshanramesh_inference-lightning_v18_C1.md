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

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
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

0.8612873980054397

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11547) has done: 'I fixed the import errors, added a proper dataset class, implemented the missing training steps in the Lightning model, removed the unused SWA checkpoint loading, and ensured the inference loop uses the trained model to write a correctly‑formatted `submission.csv`. This resolves all runtime errors and produces a valid submission file, while keeping the original architecture (ResNet‑50 backbone with a 512‑dim embedding and a linear classifier) unchanged.'
- What this solution (achieved 0.11547) has done: 'I fixed the Trainer initialization to match the current PyTorch Lightning API (removed the deprecated gpus and progress_bar_refresh_rate arguments and added accelerator and devices), increased training epochs slightly to give the model a chance to learn, and streamlined the inference loop by moving the model and tensors to the proper device once rather than each iteration. These changes resolve the runtime error, allow proper training, and should raise the validation accuracy toward the target score while still producing a correctly‑formatted submission.csv file.'

# 9. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms

import pytorch_lightning as pl
from torchmetrics.functional import accuracy

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True  # ensure cuDNN speedups
torch.set_float32_matmul_precision("high")




## === cell 1
class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        """
        df: DataFrame with columns ['image_id', 'label']
        img_dir: directory containing the images
        transform: torchvision/albumentations transform applied to PIL image
        """
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.loc[idx, "image_id"]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)  # Tensor after transforms
        else:
            image = transforms.ToTensor()(image)
        label = int(self.df.loc[idx, "label"])
        return image, label




## === cell 2
class LitModel(pl.LightningModule):
    def __init__(self, n_cls=5, pretrained=False, lr=1e-3):
        super().__init__()
        self.n_cls = n_cls
        self.lr = lr
        self.backbone = torchvision.models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(2048, 512, bias=False), nn.BatchNorm1d(512)
        )
        self.logits = nn.Linear(512, self.n_cls)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        emb = self.backbone(x)  # (B,512)
        logits = self.logits(emb)  # (B, n_cls)
        return logits

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        logits = self(imgs)
        loss = self.criterion(logits, targets)
        acc = accuracy(logits, targets, task="multiclass", num_classes=self.n_cls)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        imgs, targets = batch
        logits = self(imgs)
        loss = self.criterion(logits, targets)
        acc = accuracy(logits, targets, task="multiclass", num_classes=self.n_cls)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer




## === cell 3
BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUBMIT = os.path.join(BASE_PATH, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_CSV)

train_split, val_split = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["label"],
    random_state=42,
)

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = CassavaDataset(train_split, TRAIN_IMG_DIR, transform=train_transform)
val_dataset = CassavaDataset(val_split, TRAIN_IMG_DIR, transform=val_transform)

num_workers = min(12, os.cpu_count() or 4)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

val_images = []
val_labels = []
for img, lbl in val_dataset:
    val_images.append(img)
    val_labels.append(lbl)
val_tensor_dataset = torch.utils.data.TensorDataset(
    torch.stack(val_images), torch.tensor(val_labels)
)

val_loader = DataLoader(
    val_tensor_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0,  # no extra workers needed for in‑memory data
    pin_memory=True,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    412             try:
--> 413                 return self._range.index(new_key)
    414             except ValueError as err:

ValueError: 1873 is not in range

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_56/1535519193.py in <cell line: 0>()
     50 val_images = []
     51 val_labels = []
---> 52 for img, lbl in val_dataset:
     53     val_images.append(img)
     54     val_labels.append(lbl)

/tmp/ipykernel_56/3291112614.py in __getitem__(self, idx)
     14 
     15     def __getitem__(self, idx):
---> 16         img_name = self.df.loc[idx, "image_id"]
     17         img_path = os.path.join(self.img_dir, img_name)
     18         image = Image.open(img_path).convert("RGB")

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1181             key = tuple(com.apply_if_callable(x, self.obj) for x in key)
   1182             if self._is_scalar_access(key):
-> 1183                 return self.obj._get_value(*key, takeable=self._takeable)
   1184             return self._getitem_tuple(key)
   1185         else:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _get_value(self, index, col, takeable)
   4219             #  results if our categories are integers that dont match our codes
   4220             # IntervalIndex: IntervalTree has no get_loc
-> 4221             row = self.index.get_loc(index)
   4222             return series._values[row]
   4223 

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    413                 return self._range.index(new_key)
    414             except ValueError as err:
--> 415                 raise KeyError(key) from err
    416         if isinstance(key, Hashable):
    417             raise KeyError(key)

KeyError: 1873

## === cell 4
pl.seed_everything(42)

model = LitModel(n_cls=5, pretrained=True, lr=1e-3)

model = torch.compile(model)

use_gpu = torch.cuda.is_available()
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="gpu" if use_gpu else "cpu",
    devices=1,
    precision=16 if use_gpu else 32,  # mixed precision only on GPU
    enable_progress_bar=False,
    logger=False,
    enable_checkpointing=False,
)

trainer.fit(model, train_loader, val_loader)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/3238465782.py in <cell line: 0>()
     16 )
     17 
---> 18 trainer.fit(model, train_loader, val_loader)
     19 
     20 

NameError: name 'val_loader' is not defined

## === cell 5
sample_submission_df = pd.read_csv(SAMPLE_SUBMIT)


class TestCassavaDataset(Dataset):
    def __init__(self, img_ids, img_dir, transform):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_name = self.img_ids[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, img_name


test_dataset = TestCassavaDataset(
    sample_submission_df["image_id"].tolist(),
    TEST_IMG_DIR,
    transform=val_transform,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=2,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()
model.freeze()  # ensure no grads

predictions = []
image_ids = []

with torch.no_grad():
    for imgs, ids in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds.tolist())
        image_ids.extend(ids)

assert (
    len(predictions) == len(image_ids) == len(sample_submission_df)
), "Length mismatch"




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_56/3336746541.py in <cell line: 0>()
     29 
     30 # Reduced workers for test loader – reading each file once, no need for many processes
---> 31 test_loader = DataLoader(
     32     test_dataset,
     33     batch_size=256,

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __init__(self, dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, generator, prefetch_factor, persistent_workers, pin_memory_device, in_order)
    268 
    269         if num_workers == 0 and prefetch_factor is not None:
--> 270             raise ValueError(
    271                 "prefetch_factor option could only be specified in multiprocessing."
    272                 "let num_workers > 0 to enable multiprocessing, otherwise set prefetch_factor to None."

ValueError: prefetch_factor option could only be specified in multiprocessing.let num_workers > 0 to enable multiprocessing, otherwise set prefetch_factor to None.

## === cell 6
submission = pd.DataFrame({"image_id": image_ids, "label": predictions})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, rows: {len(submission)}")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1903460677.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image_id": image_ids, "label": predictions})
      2 
      3 submission_path = "submission.csv"
      4 submission.to_csv(submission_path, index=False)
      5 print(f"Submission written to {submission_path}, rows: {len(submission)}")

NameError: name 'image_ids' is not defined
