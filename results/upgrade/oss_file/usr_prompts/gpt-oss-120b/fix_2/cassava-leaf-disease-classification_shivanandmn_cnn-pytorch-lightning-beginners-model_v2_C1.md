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

2.7

# 3. Installed packages

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

0.6272287700211544

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pytorch_lightning as pl
import pandas as pd
import cv2
import os
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy

IMG_SIZE = 64
PATH = "../input/cassava-leaf-disease-classification/train_images/"
CLASSES = 5




## === cell 1
class CassavaModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.cnv = nn.Conv2d(3, 128, 5, 4)
        self.rel = nn.ReLU()
        self.bn = nn.BatchNorm2d(128)
        self.mxpool = nn.MaxPool2d(4)
        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(1152, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, CLASSES)
        self.accuracy = Accuracy()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.bn(self.rel(self.cnv(x)))
        out = self.flat(self.mxpool(out))
        out = self.rel(self.fc1(out))
        out = self.rel(self.fc2(out))
        out = self.fc3(out)
        return out

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        out = self(x)
        loss = self.loss_fn(out, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["x"], batch["y"]
        out = self(x)
        loss = self.loss_fn(out, y)
        probs = nn.Softmax(dim=1)(out)
        preds = torch.argmax(probs, dim=1)
        acc = self.accuracy(preds, y)
        self.log("valid_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss




## === cell 2
class CassavaDataset(Dataset):
    def __init__(self, path, image_ids, labels, image_size):
        self.image_ids = image_ids
        self.labels = labels
        self.path = path
        self.image_size = image_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = str(self.image_ids[idx])
        label = self.labels[idx]
        img_path = os.path.join(self.path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0  # normalize
        img = torch.from_numpy(img).permute(2, 0, 1)  # channel‑first
        return {"x": img, "y": torch.tensor(label, dtype=torch.long)}




## === cell 3
class CassavaLightDataset(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        df = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
        x_train, x_val, y_train, y_val = train_test_split(
            df["image_id"].values, df["label"].values, test_size=0.1, random_state=42
        )
        self.train_dataset = CassavaDataset(PATH, x_train, y_train, IMG_SIZE)
        self.val_dataset = CassavaDataset(PATH, x_val, y_val, IMG_SIZE)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )




## === cell 4
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor="valid_loss",
    dirpath=".",
    filename="model-{epoch:02d}-{valid_loss:.2f}",
    save_top_k=3,
    mode="min",
)

model = CassavaModel()
datamodule = CassavaLightDataset()
trainer = pl.Trainer(
    max_epochs=6,
    callbacks=[checkpoint_callback],
    accelerator="cpu",
    enable_progress_bar=True,
    logger=False,
)
trainer.fit(model, datamodule=datamodule)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/431536897.py in <cell line: 0>()
      7 )
      8 
----> 9 model = CassavaModel()
     10 datamodule = CassavaLightDataset()
     11 trainer = pl.Trainer(

/tmp/ipykernel_55/3838605922.py in __init__(self)
     10         self.fc2 = nn.Linear(64, 64)
     11         self.fc3 = nn.Linear(64, CLASSES)
---> 12         self.accuracy = Accuracy()
     13         self.loss_fn = nn.CrossEntropyLoss()
     14 

TypeError: Accuracy.__new__() missing 1 required positional argument: 'task'

## === cell 5
best_ckpt_path = checkpoint_callback.best_model_path
best_model = CassavaModel.load_from_checkpoint(best_ckpt_path)
best_model.eval()
best_model.freeze()

TEST_PATH = "../input/cassava-leaf-disease-classification/test_images/"


class CassavaTestDataset(Dataset):
    def __init__(self, path, image_ids, image_size):
        self.image_ids = image_ids
        self.path = path
        self.image_size = image_size

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = str(self.image_ids[idx])
        img_path = os.path.join(self.path, img_name)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        return {"x": img}


sample = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_dataset = CassavaTestDataset(TEST_PATH, sample["image_id"].values, IMG_SIZE)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

preds = []
for batch in test_loader:
    logits = best_model(batch["x"])
    probs = nn.Softmax(dim=1)(logits)
    pred = torch.argmax(probs, dim=1).cpu().numpy()
    preds.append(pred.item())

sample["label"] = preds
sample[["image_id", "label"]].to_csv("submission.csv", index=False)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_55/3325753452.py in <cell line: 0>()
      1 # Load the best checkpoint found during training
      2 best_ckpt_path = checkpoint_callback.best_model_path
----> 3 best_model = CassavaModel.load_from_checkpoint(best_ckpt_path)
      4 best_model.eval()
      5 best_model.freeze()

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/utilities/model_helpers.py in wrapper(*args, **kwargs)
    123                     " Please call it on the class type and make sure the return value is used."
    124                 )
--> 125             return self.method(cls, *args, **kwargs)
    126 
    127         return wrapper

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/core/module.py in load_from_checkpoint(cls, checkpoint_path, map_location, hparams_file, strict, **kwargs)
   1660 
   1661         """
-> 1662         loaded = _load_from_checkpoint(
   1663             cls,
   1664             checkpoint_path,

/usr/local/lib/python3.11/dist-packages/pytorch_lightning/core/saving.py in _load_from_checkpoint(cls, checkpoint_path, map_location, hparams_file, strict, **kwargs)
     61     map_location = map_location or _default_map_location
     62     with pl_legacy_patch():
---> 63         checkpoint = pl_load(checkpoint_path, map_location=map_location)
     64 
     65     # convert legacy checkpoints to the new format

/usr/local/lib/python3.11/dist-packages/lightning_fabric/utilities/cloud_io.py in _load(path_or_url, map_location, weights_only)
     58         )
     59     fs = get_filesystem(path_or_url)
---> 60     with fs.open(path_or_url, "rb") as f:
     61         return torch.load(
     62             f,

/usr/local/lib/python3.11/dist-packages/fsspec/spec.py in open(self, path, mode, block_size, cache_options, compression, **kwargs)
   1347         else:
   1348             ac = kwargs.pop("autocommit", not self._intrans)
-> 1349             f = self._open(
   1350                 path,
   1351                 mode=mode,

/usr/local/lib/python3.11/dist-packages/fsspec/implementations/local.py in _open(self, path, mode, block_size, **kwargs)
    208         if self.auto_mkdir and "w" in mode:
    209             self.makedirs(self._parent(path), exist_ok=True)
--> 210         return LocalFileOpener(path, mode, fs=self, **kwargs)
    211 
    212     def touch(self, path, truncate=True, **kwargs):

/usr/local/lib/python3.11/dist-packages/fsspec/implementations/local.py in __init__(self, path, mode, autocommit, fs, compression, **kwargs)
    385         self.compression = get_compression(path, compression)
    386         self.blocksize = io.DEFAULT_BUFFER_SIZE
--> 387         self._open()
    388 
    389     def _open(self):

/usr/local/lib/python3.11/dist-packages/fsspec/implementations/local.py in _open(self)
    390         if self.f is None or self.f.closed:
    391             if self.autocommit or "w" not in self.mode:
--> 392                 self.f = open(self.path, mode=self.mode)
    393                 if self.compression:
    394                     compress = compr[self.compression]

IsADirectoryError: [Errno 21] Is a directory: '/kaggle/working'
