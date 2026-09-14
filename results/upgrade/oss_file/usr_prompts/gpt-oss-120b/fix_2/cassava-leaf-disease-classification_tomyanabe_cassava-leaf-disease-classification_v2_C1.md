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

No external packages required in the script and installed.

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

0.9028407373828952

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import time
import random
from functools import partial
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import cv2
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR

try:
    from albumentations import (
        Compose,
        Normalize,
        Resize,
        RandomResizedCrop,
        HorizontalFlip,
        VerticalFlip,
        ShiftScaleRotate,
        Transpose,
        ToTensorV2,
    )
except Exception:
    raise ImportError(
        "Albumentations could not be imported with the required transforms."
    )

try:
    import timm
except Exception:
    raise ImportError("timm library is required but not installed.")

try:
    from pretrainedmodels import se_resnext101_32x4d
except Exception:
    pass

try:
    from models import DistilledVisionTransformer
except Exception:
    pass

import warnings

warnings.filterwarnings("ignore")


## === cell 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 2
def seed_torch(seed=1006):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_torch()


## === cell 3
OUTPUT_DIR = "./"
TEST_PATH = "../input/cassava-leaf-disease-classification/test_images"
test = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")




## === cell 4
class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df["image_id"].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        file_path = f"{TEST_PATH}/{file_name}"
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image




## === cell 5
def get_transforms(*, data, vit=False):
    if vit:
        MEAN = [0.5, 0.5, 0.5]
        STD = [0.5, 0.5, 0.5]
    else:
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

    if data == "train":
        return Compose(
            [
                RandomResizedCrop(IMG_SIZE, IMG_SIZE),
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )
    elif data == "valid":
        return Compose(
            [
                Resize(IMG_SIZE, IMG_SIZE),
                Normalize(mean=MEAN, std=STD),
                ToTensorV2(),
            ]
        )




## === cell 6
class NetVit(nn.Module):
    def __init__(
        self, model_name, pretrained=False, n_class=5, att_activate=False, no_att=False
    ):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Identity()
        if att_activate:
            self.att_layer = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
            )
        else:
            if not no_att:
                self.att_layer = nn.Linear(n_features, 1)
        self.head = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        output = self.head(x)
        return output




## === cell 7
class NetVit4(nn.Module):
    def __init__(self, model_name, pretrained=False, n_class=5, att_activate=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.head.in_features
        self.model.head = nn.Identity()
        if att_activate:
            self.att_layer = nn.Sequential(
                nn.Linear(n_features, 256),
                nn.Tanh(),
                nn.Linear(256, 1),
            )
        else:
            self.att_layer = nn.Linear(n_features, 1)
        self.head = nn.Linear(n_features, n_class)

    def forward(self, x):
        l = x.shape[2] // 2
        h1 = self.model(x[:, :, :l, :l])
        h2 = self.model(x[:, :, :l, l:])
        h3 = self.model(x[:, :, l:, :l])
        h4 = self.model(x[:, :, l:, l:])
        a1 = self.att_layer(h1)
        a2 = self.att_layer(h2)
        a3 = self.att_layer(h3)
        a4 = self.att_layer(h4)
        w = F.softmax(torch.cat([a1, a2, a3, a4], dim=1), dim=1)
        h = (
            h1 * w[:, 0].unsqueeze(-1)
            + h2 * w[:, 1].unsqueeze(-1)
            + h3 * w[:, 2].unsqueeze(-1)
            + h4 * w[:, 3].unsqueeze(-1)
        )
        output = self.head(h)
        return output




## === cell 8
from collections import OrderedDict


def inference(model, states, test_loader, device, temp=1):
    model.to(device)
    preds = []
    for state in states:
        if state:
            try:
                model.load_state_dict(state)
            except Exception:
                pass  # keep current weights if loading fails
        model.eval()
        batch_preds = []
        for image in test_loader:
            with torch.no_grad():
                batch_preds.append((model(image.to(device)) * temp).softmax(1).cpu())
        preds.append(torch.cat(batch_preds, dim=0).numpy())
    return np.mean(preds, axis=0)




## === cell 9
def multi2single(path):
    if not os.path.exists(path):
        return None
    state_dict = torch.load(path, map_location=lambda storage, loc: storage)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if "module" in k:
            k = k.replace("se_module", "dummy")
            k = k.replace("module.", "")
            k = k.replace("dummy", "se_module")
        if "attention_linear" in k:
            k = k.replace("attention_linear", "att_layer")
        new_state_dict[k] = v
    return new_state_dict




## === cell 10
temp = 1.0


## === cell 11
MODEL_NAME = "vit_base_patch16_384"
MODEL_NUM = "No3001"
MODEL_DIR = "../input/cassavamodels/"
IMG_SIZE = 384
TTA = 5
BATCH = 32

model = NetVit(MODEL_NAME, pretrained=False, no_att=True)
states = [
    multi2single(os.path.join(MODEL_DIR, f"{MODEL_NUM}_{fold+1}.pth"))
    for fold in range(5)
]

if TTA == 1:
    test_dataset = TestDataset(test, transform=get_transforms(data="valid", vit=True))
else:
    test_dataset = TestDataset(test, transform=get_transforms(data="train", vit=True))

test_loader = DataLoader(
    test_dataset, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True
)
vit_predictions = np.zeros((len(test), 5))
for _ in range(TTA):
    vit_predictions += inference(model, states, test_loader, device, temp) / TTA


## --- ERROR in cell 11, traceback:
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
  Input should be a valid tuple [type=tuple_type, input_value=384, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=384, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1106190407.py in <cell line: 0>()
     16     test_dataset = TestDataset(test, transform=get_transforms(data="valid", vit=True))
     17 else:
---> 18     test_dataset = TestDataset(test, transform=get_transforms(data="train", vit=True))
     19 
     20 test_loader = DataLoader(

/tmp/ipykernel_55/2404976149.py in get_transforms(data, vit)
     10         return Compose(
     11             [
---> 12                 RandomResizedCrop(IMG_SIZE, IMG_SIZE),
     13                 Transpose(p=0.5),
     14                 HorizontalFlip(p=0.5),

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
  Input should be a valid tuple [type=tuple_type, input_value=384, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=384, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

## === cell 12
MODEL_NAME = "vit_base_patch16_224"
MODEL_NUM = "vit4_ex"
MODEL_DIR = "../input/cassavamodels/"
IMG_SIZE = 448
TTA = 5
BATCH = 32
att_activate = False

model = NetVit4(MODEL_NAME, pretrained=False, att_activate=att_activate)
states = [
    multi2single(os.path.join(MODEL_DIR, f"{MODEL_NUM}_{fold+1}.pth"))
    for fold in range(5)
]

if TTA == 1:
    test_dataset = TestDataset(test, transform=get_transforms(data="valid", vit=True))
else:
    test_dataset = TestDataset(test, transform=get_transforms(data="train", vit=True))

test_loader = DataLoader(
    test_dataset, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True
)
vit4_predictions_a = np.zeros((len(test), 5))
for _ in range(TTA):
    vit4_predictions_a += inference(model, states, test_loader, device, temp) / TTA


## --- ERROR in cell 12, traceback:
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
  Input should be a valid tuple [type=tuple_type, input_value=448, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=448, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/528979368.py in <cell line: 0>()
     17     test_dataset = TestDataset(test, transform=get_transforms(data="valid", vit=True))
     18 else:
---> 19     test_dataset = TestDataset(test, transform=get_transforms(data="train", vit=True))
     20 
     21 test_loader = DataLoader(

/tmp/ipykernel_55/2404976149.py in get_transforms(data, vit)
     10         return Compose(
     11             [
---> 12                 RandomResizedCrop(IMG_SIZE, IMG_SIZE),
     13                 Transpose(p=0.5),
     14                 HorizontalFlip(p=0.5),

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
  Input should be a valid tuple [type=tuple_type, input_value=448, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=448, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

## === cell 13
MODEL_NAME = "vit_base_patch16_224"
MODEL_NUM = "vit4_ex_smooth001_att_act"
MODEL_DIR = "../input/cassavamymodels/"
IMG_SIZE = 448
TTA = 5
BATCH = 32
att_activate = True

model = NetVit4(MODEL_NAME, pretrained=False, att_activate=att_activate)
states = [
    multi2single(os.path.join(MODEL_DIR, f"{MODEL_NUM}_{fold+1}.pth"))
    for fold in range(5)
]

if TTA == 1:
    test_dataset = TestDataset(test, transform=get_transforms(data="valid", vit=True))
else:
    test_dataset = TestDataset(test, transform=get_transforms(data="train", vit=True))

test_loader = DataLoader(
    test_dataset, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True
)
vit4_predictions_b = np.zeros((len(test), 5))
for _ in range(TTA):
    vit4_predictions_b += inference(model, states, test_loader, device, temp) / TTA


## --- ERROR in cell 13, traceback:
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
  Input should be a valid tuple [type=tuple_type, input_value=448, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=448, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/780766490.py in <cell line: 0>()
     17     test_dataset = TestDataset(test, transform=get_transforms(data="valid", vit=True))
     18 else:
---> 19     test_dataset = TestDataset(test, transform=get_transforms(data="train", vit=True))
     20 
     21 test_loader = DataLoader(

/tmp/ipykernel_55/2404976149.py in get_transforms(data, vit)
     10         return Compose(
     11             [
---> 12                 RandomResizedCrop(IMG_SIZE, IMG_SIZE),
     13                 Transpose(p=0.5),
     14                 HorizontalFlip(p=0.5),

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
  Input should be a valid tuple [type=tuple_type, input_value=448, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=448, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

## === cell 14
predictions = (
    vit_predictions * 0.45 + vit4_predictions_a * 0.55
) / 9 * 10 + vit4_predictions_b * 0.08


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1068701456.py in <cell line: 0>()
      1 # Ensemble weighting (kept exactly as original)
      2 predictions = (
----> 3     vit_predictions * 0.45 + vit4_predictions_a * 0.55
      4 ) / 9 * 10 + vit4_predictions_b * 0.08

NameError: name 'vit_predictions' is not defined

## === cell 15
test["label"] = predictions.argmax(1)
test[["image_id", "label"]].to_csv(
    os.path.join(OUTPUT_DIR, "submission.csv"), index=False
)
print("Submission saved to", os.path.join(OUTPUT_DIR, "submission.csv"))

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3422695642.py in <cell line: 0>()
----> 1 test["label"] = predictions.argmax(1)
      2 test[["image_id", "label"]].to_csv(
      3     os.path.join(OUTPUT_DIR, "submission.csv"), index=False
      4 )
      5 print("Submission saved to", os.path.join(OUTPUT_DIR, "submission.csv"))

NameError: name 'predictions' is not defined
