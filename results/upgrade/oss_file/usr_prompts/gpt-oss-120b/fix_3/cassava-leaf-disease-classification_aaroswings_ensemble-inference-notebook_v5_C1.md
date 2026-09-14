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

0.8773043215472952

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import io
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

from albumentations import Compose, Normalize, RandomResizedCrop
from albumentations.pytorch.transforms import ToTensorV2

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



## === cell 1
batch_size = 32
valid_input_size = 600
train_img_path = Path("input/cassava-leaf-disease-classification/train_images")
test_img_path = Path("input/cassava-leaf-disease-classification/test_images")
train_csv_path = Path("input/cassava-leaf-disease-classification/train.csv")




## === cell 2
def build_model(num_classes=5):
    model = models.resnet34(pretrained=True)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


base_model = build_model(num_classes=5).to(device)



## === cell 3
train_tfms = Compose(
    [
        RandomResizedCrop(
            (valid_input_size, valid_input_size), always_apply=True, scale=(0.8, 1.0)
        ),
        Normalize(),
        ToTensorV2(),
    ]
)


class ImageDataset(Dataset):
    def __init__(self, csv_path, img_root, tfms):
        self.df = pd.read_csv(csv_path)
        self.img_root = Path(img_root)
        self.tfms = tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_root / row["image_id"]
        img = np.array(Image.open(img_path).convert("RGB"))
        img = self.tfms(image=img)["image"]
        label = int(row["label"])
        return img, label


train_dataset = ImageDataset(train_csv_path, train_img_path, train_tfms)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1462245229.py in <cell line: 0>()
     28 
     29 
---> 30 train_dataset = ImageDataset(train_csv_path, train_img_path, train_tfms)
     31 

/tmp/ipykernel_55/1462245229.py in __init__(self, csv_path, img_root, tfms)
     12 class ImageDataset(Dataset):
     13     def __init__(self, csv_path, img_root, tfms):
---> 14         self.df = pd.read_csv(csv_path)
     15         self.img_root = Path(img_root)
     16         self.tfms = tfms

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'input/cassava-leaf-disease-classification/train.csv'

## === cell 4
val_len = int(0.1 * len(train_dataset))
train_len = len(train_dataset) - val_len
train_subset, val_subset = random_split(train_dataset, [train_len, val_len])

train_loader = DataLoader(
    train_subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_subset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/327567226.py in <cell line: 0>()
      1 # simple 90/10 split for quick validation
----> 2 val_len = int(0.1 * len(train_dataset))
      3 train_len = len(train_dataset) - val_len
      4 train_subset, val_subset = random_split(train_dataset, [train_len, val_len])
      5 

NameError: name 'train_dataset' is not defined

## === cell 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(base_model.parameters(), lr=1e-4)

base_model.train()
epochs = 3  # keep lightweight
for epoch in range(epochs):
    running_loss = 0.0
    for imgs, targets in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = base_model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / train_len
    base_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = base_model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    val_acc = correct / total if total > 0 else 0
    print(
        f"Epoch {epoch+1}/{epochs} - Train loss: {epoch_loss:.4f} - Val acc: {val_acc:.4f}"
    )
    base_model.train()

base_model.eval()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1485653187.py in <cell line: 0>()
      6 for epoch in range(epochs):
      7     running_loss = 0.0
----> 8     for imgs, targets in train_loader:
      9         imgs = imgs.to(device, non_blocking=True)
     10         targets = targets.to(device, non_blocking=True)

NameError: name 'train_loader' is not defined

## === cell 6
test_tfms = Compose(
    [
        RandomResizedCrop(
            (valid_input_size, valid_input_size), always_apply=True, scale=(0.8, 1.0)
        ),
        Normalize(),
        ToTensorV2(),
    ]
)


class TestDataset(Dataset):
    def __init__(self, path, tfms):
        self.data = [
            (file.name, open(file, "rb").read())
            for file in Path(path).iterdir()
            if file.is_file()
        ]
        self.tfms = tfms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename, img_bytes = self.data[idx]
        img = np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        img = self.tfms(image=img)["image"]
        return filename, img


test_ds = TestDataset(test_img_path, test_tfms)
test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2, drop_last=False)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2071206592.py in <cell line: 0>()
     29 
     30 
---> 31 test_ds = TestDataset(test_img_path, test_tfms)
     32 test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=2, drop_last=False)
     33 

/tmp/ipykernel_55/2071206592.py in __init__(self, path, tfms)
     12 class TestDataset(Dataset):
     13     def __init__(self, path, tfms):
---> 14         self.data = [
     15             (file.name, open(file, "rb").read())
     16             for file in Path(path).iterdir()

/tmp/ipykernel_55/2071206592.py in <listcomp>(.0)
     12 class TestDataset(Dataset):
     13     def __init__(self, path, tfms):
---> 14         self.data = [
     15             (file.name, open(file, "rb").read())
     16             for file in Path(path).iterdir()

/usr/lib/python3.11/pathlib.py in iterdir(self)
    929         result for the special paths '.' and '..'.
    930         """
--> 931         for name in os.listdir(self):
    932             yield self._make_child_relpath(name)
    933 

FileNotFoundError: [Errno 2] No such file or directory: 'input/cassava-leaf-disease-classification/test_images'

## === cell 7
class EnsemblePredictor:
    def __init__(self, models):
        self.models = models

    def predict_on_loader(self, loader):
        predictions = []
        filenames = []
        for batch_files, batch_imgs in loader:
            batch_imgs = batch_imgs.to(device)
            cum_logits = None
            for model in self.models:
                with torch.no_grad():
                    logits = model(batch_imgs)
                cum_logits = logits if cum_logits is None else cum_logits + logits
            batch_preds = torch.argmax(cum_logits, dim=1).cpu().numpy()
            predictions.extend(batch_preds.tolist())
            filenames.extend(batch_files)
        return predictions, filenames




## === cell 8
predictor = EnsemblePredictor([base_model])
predictions, filenames = predictor.predict_on_loader(test_loader)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3647091436.py in <cell line: 0>()
      1 predictor = EnsemblePredictor([base_model])
----> 2 predictions, filenames = predictor.predict_on_loader(test_loader)
      3 

NameError: name 'test_loader' is not defined

## === cell 9
submission_path = Path("submission.csv")
with submission_path.open("w") as submission:
    submission.write("image_id,label\n")
    for filename, pred in zip(filenames, predictions):
        submission.write(f"{filename},{int(pred)}\n")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2006774258.py in <cell line: 0>()
      2 with submission_path.open("w") as submission:
      3     submission.write("image_id,label\n")
----> 4     for filename, pred in zip(filenames, predictions):
      5         submission.write(f"{filename},{int(pred)}\n")
      6 

NameError: name 'filenames' is not defined

## === cell 10
print(
    f"Submission written to {submission_path}, lines:",
    sum(1 for _ in submission_path.open()),
)

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
