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
seaborn==0.12.2
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

# 5. Target score

0.8434572378362043

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms, models
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/tensorboard/compat/__init__.py in tf()
     41     try:
---> 42         from tensorboard.compat import notf  # noqa: F401
     43     except ImportError:

ImportError: cannot import name 'notf' from 'tensorboard.compat' (/usr/local/lib/python3.11/dist-packages/tensorboard/compat/__init__.py)

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAINING = False
WEIGHT_FILE = "../input/ramki-cassava-weights/weight-at-epoch-61-acc-0.85794.pth"

SEED = 42
N_EPOCHS = 100
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 0.001
NUM_CLASSES = 5

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)




## === cell 2
def seed_everything(seed: int):
    """set deterministic seeds for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(SEED)



## === cell 3
DATA_ROOT = "./input/cassava-leaf-disease-classification/"

train_path = os.path.join(DATA_ROOT, "train_images/")
test_path = os.path.join(DATA_ROOT, "test_images/")

train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
sample = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

print("train shape:", train_csv.shape, "sample shape:", sample.shape)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2564205366.py in <cell line: 0>()
      5 test_path = os.path.join(DATA_ROOT, "test_images/")
      6 
----> 7 train_csv = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
      8 sample = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
      9 

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

FileNotFoundError: [Errno 2] No such file or directory: './input/cassava-leaf-disease-classification/train.csv'

## === cell 4
class CasavaDataset(Dataset):
    def __init__(self, dataframe, transforms=None, test=False):
        self.df = dataframe.reset_index(drop=True)
        self.transforms = transforms
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df.iloc[idx].label if not self.test else -1
        img_id = self.df.iloc[idx].image_id
        img_path = (
            os.path.join(train_path, img_id)
            if not self.test
            else os.path.join(test_path, img_id)
        )
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transforms:
            image = self.transforms(image=image)["image"]
        return image, label




## === cell 5
transforms_train = A.Compose(
    [
        A.RandomResizedCrop(height=300, width=300, p=1.0),
        A.Rotate(limit=20, p=0.5),
        A.Flip(p=0.5),
        A.Transpose(p=0.5),
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ]
)

transforms_valid = A.Compose(
    [
        A.Resize(height=IMG_SIZE, width=IMG_SIZE, p=1.0),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ]
)



## --- ERROR in cell 5, traceback:
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

ValidationError: 1 validation error for InitSchema
size
  Field required [type=missing, input_value={'p': 1.0, 'scale': (0.08...: None, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1000341325.py in <cell line: 0>()
      2 transforms_train = A.Compose(
      3     [
----> 4         A.RandomResizedCrop(height=300, width=300, p=1.0),
      5         A.Rotate(limit=20, p=0.5),
      6         A.Flip(p=0.5),

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

ValueError: 1 validation error for InitSchema
size
  Field required [type=missing, input_value={'p': 1.0, 'scale': (0.08...: None, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

## === cell 6
model_name = "efficientnet_b7"
if TRAINING:
    model = models.efficientnet_b7(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    print("model created with ImageNet weights")
else:
    model = models.efficientnet_b7(pretrained=False)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    state_dict = torch.load(WEIGHT_FILE, map_location=device)
    model.load_state_dict(state_dict)
    print("model loaded from weight file")

model = model.to(device)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1364988847.py in <cell line: 0>()
      9     model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
     10     # load provided checkpoint (it matches the architecture)
---> 11     state_dict = torch.load(WEIGHT_FILE, map_location=device)
     12     model.load_state_dict(state_dict)
     13     print("model loaded from weight file")

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '../input/ramki-cassava-weights/weight-at-epoch-61-acc-0.85794.pth'

## === cell 7
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def show_metrics(model, dataloader, criterion):
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()
    all_preds, all_labels = [], []
    tk = tqdm(dataloader, total=len(dataloader), leave=False)
    for imgs, labels in tk:
        imgs = imgs.to(device)
        labels = labels.to(device).long()
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        accs.update((preds == labels).sum().item() / imgs.size(0), imgs.size(0))
        losses.update(loss.item(), imgs.size(0))
        tk.set_postfix(loss=losses.avg, acc=accs.avg)

    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.show()
    print(
        classification_report(
            all_labels, all_preds, target_names=["CBB", "CBSD", "CGM", "CMD", "Healthy"]
        )
    )
    return losses.avg, all_preds, all_labels


def train_model(model, dataloader, criterion, optimizer):
    model.train()
    losses = AverageMeter()
    accs = AverageMeter()
    tk = tqdm(dataloader, total=len(dataloader), leave=False)
    for imgs, labels in tk:
        imgs = imgs.to(device)
        labels = labels.to(device).long()
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        accs.update((preds == labels).sum().item() / imgs.size(0), imgs.size(0))
        losses.update(loss.item(), imgs.size(0))
        tk.set_postfix(loss=losses.avg, acc=accs.avg)
    return losses.avg, accs.avg


def test_model(model, dataloader, criterion):
    model.eval()
    losses = AverageMeter()
    accs = AverageMeter()
    with torch.no_grad():
        tk = tqdm(dataloader, total=len(dataloader), leave=False)
        for imgs, labels in tk:
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            accs.update(
                (outputs.argmax(1) == labels).sum().item() / imgs.size(0), imgs.size(0)
            )
            losses.update(loss.item(), imgs.size(0))
            tk.set_postfix(loss=losses.avg, acc=accs.avg)
    return losses.avg, accs.avg, loss




## === cell 8
X_train, X_valid = train_test_split(
    train_csv, test_size=0.25, random_state=SEED, stratify=train_csv["label"]
)

train_dataset = CasavaDataset(X_train, transforms=transforms_train, test=False)
valid_dataset = CasavaDataset(X_valid, transforms=transforms_valid, test=False)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3617115693.py in <cell line: 0>()
      1 # split for optional training (kept for completeness)
      2 X_train, X_valid = train_test_split(
----> 3     train_csv, test_size=0.25, random_state=SEED, stratify=train_csv["label"]
      4 )
      5 

NameError: name 'train_csv' is not defined

## === cell 9
test_dataset = CasavaDataset(sample, transforms=transforms_valid, test=True)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)

model.eval()
test_preds = []
with torch.no_grad():
    for imgs, _ in tqdm(test_loader, leave=False):
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1).cpu().numpy()
        test_preds.extend(preds.tolist())

sample["label"] = test_preds
submission_path = "submission.csv"
sample.to_csv(submission_path, index=False)
print(f"submission written to {submission_path}")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3252176453.py in <cell line: 0>()
      1 # inference on test set and submission creation
----> 2 test_dataset = CasavaDataset(sample, transforms=transforms_valid, test=True)
      3 test_loader = DataLoader(
      4     test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
      5 )

NameError: name 'sample' is not defined

## === cell 10
print(sample.head())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3794781747.py in <cell line: 0>()
      1 # optional: display a few rows of the submission to verify
----> 2 print(sample.head())

NameError: name 'sample' is not defined
