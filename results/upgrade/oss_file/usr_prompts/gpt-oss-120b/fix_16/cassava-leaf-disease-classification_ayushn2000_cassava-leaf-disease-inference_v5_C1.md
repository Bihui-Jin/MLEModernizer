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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 5. Target score

0.819431852523421

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'The fix changes the model initialization to avoid requesting unavailable pretrained weights (setting `pretrained=False`). This allows the checkpoint weights to be loaded correctly, so inference runs and `tst_preds` is defined for the final submission step.'
- What this solution (achieved 0.11024) has done: 'Implemented fixes:
- Removed premature call to `seed_everything` and placed it after its definition.
- Added a call to `seed_everything` with the configured seed.
- Set `pretrained=False` when creating the model to avoid missing‑weight errors.
- Minor comment clean‑ups.'
- What this solution (achieved 0.11024) has done: 'The fix changes the model initialization to use `pretrained=False`, avoiding the runtime error caused by missing pretrained weights for the specified architecture. With this correction the inference loop runs, `tst_preds` is correctly populated, and the final cell can create a proper `submission.csv` containing the required `image_id` and `label` columns.'
- What this solution (achieved 0.11024) has done: 'Implemented a minimal fix: changed the model initialization to use `pretrained=False` (the requested architecture has no pretrained weights), preventing the RuntimeError and allowing the inference loop to run. This ensures `tst_preds` is defined, so the final cell can create a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.11024) has done: 'The fix changes the model initialization to use `pretrained=False`, preventing the RuntimeError caused by missing ImageNet weights for the requested architecture. With the model now loading correctly, the inference loop runs, `tst_preds` is defined, and a proper `submission.csv` containing `image_id` and `label` columns is written.'

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import albumentations.pytorch as Apy

from glob import glob
import os
import time
import random
import cv2
import warnings
import timm

from tqdm import tqdm
from datetime import datetime
from skimage import io
from sklearn.model_selection import StratifiedKFold

import torch
import torchvision
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler



## === cell 1
config = {
    "fold_num": 5,
    "seed": 719,
    "model_arch": "vit_base_resnet50d_224",
    "img_size": 224,
    "resize_to": 224,
    "epochs": 3,
    "train_bs": 32,
    "valid_bs": 32,
    "lr": 1e-4,
    "num_workers": 4,
    "accum_iter": 1,
    "verbose_step": 1,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "tta": 1,
    "used_epochs": [0, 1, 2],
    "weights": [1, 1, 1, 1],
}


## === cell 2
submission = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)




## === cell 3
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_img(path):
    im_bgr = cv2.imread(path)
    if im_bgr is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {path}")
    return im_bgr[:, :, ::-1]


seed_everything(config["seed"])




## === cell 4
class CassavaDataset(Dataset):
    def __init__(self, df, data_root, transforms=None, output_label=True):
        self.resize_image = torchvision.transforms.Resize(
            size=(config["resize_to"], config["resize_to"])
        )
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        if self.output_label:
            target = self.df.iloc[index]["label"]
        path = f"{self.data_root}/{self.df.iloc[index]['image_id']}"
        img = get_img(path)
        if self.transforms:
            img = self.transforms(image=img)["image"]
        if self.output_label:
            return img, target
        else:
            return img




## === cell 5
def get_train_transforms():
    return A.Compose(
        [
            A.RandomResizedCrop(
                height=config["img_size"],
                width=config["img_size"],
                scale=(0.8, 1.0),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            Apy.ToTensorV2(),
        ]
    )


def get_inference_transforms():
    return A.Compose(
        [
            A.Resize(height=config["img_size"], width=config["img_size"], p=1.0),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
            ),
            Apy.ToTensorV2(),
        ]
    )




## === cell 6
class CassvaImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            model_arch, pretrained=pretrained, num_classes=n_class
        )

    def forward(self, x):
        return self.model(x)




## === cell 7
train_df = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
train_dir = "../input/cassava-leaf-disease-classification/train_images"

train_dataset = CassavaDataset(
    train_df,
    train_dir,
    transforms=get_train_transforms(),
    output_label=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config["train_bs"],
    shuffle=True,
    num_workers=config["num_workers"],
    pin_memory=True,
)

device = torch.device(config["device"])
model = CassvaImgClassifier(config["model_arch"], n_class=5, pretrained=True).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])
scaler = GradScaler()


## --- ERROR in cell 7, traceback:
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
  Field required [type=missing, input_value={'scale': (0.8, 1.0), 'p'...: None, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_57/2649196882.py in <cell line: 0>()
      7     train_df,
      8     train_dir,
----> 9     transforms=get_train_transforms(),
     10     output_label=True,
     11 )

/tmp/ipykernel_57/2075490093.py in get_train_transforms()
      2     return A.Compose(
      3         [
----> 4             A.RandomResizedCrop(
      5                 height=config["img_size"],
      6                 width=config["img_size"],

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
  Field required [type=missing, input_value={'scale': (0.8, 1.0), 'p'...: None, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

## === cell 8
model.train()
for epoch in range(config["epochs"]):
    epoch_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
    for imgs, targets in pbar:
        imgs = imgs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()
        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        pbar.set_postfix(loss=epoch_loss / (pbar.n + 1))


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/3959301316.py in <cell line: 0>()
      1 # Simple training loop (few epochs)
----> 2 model.train()
      3 for epoch in range(config["epochs"]):
      4     epoch_loss = 0.0
      5     pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")

NameError: name 'model' is not defined

## === cell 9
test_dir = "../input/cassava-leaf-disease-classification/test_images"
test_files = [
    f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
test_df = pd.DataFrame({"image_id": test_files})

test_dataset = CassavaDataset(
    test_df,
    test_dir,
    transforms=get_inference_transforms(),
    output_label=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=config["valid_bs"],
    shuffle=False,
    num_workers=config["num_workers"],
    pin_memory=True,
)

model.eval()
all_preds = []
with torch.no_grad():
    for _ in range(config["tta"]):
        preds_epoch = []
        for imgs in tqdm(test_loader, desc="Inference"):
            imgs = imgs.to(device, dtype=torch.float)
            with autocast():
                logits = model(imgs)
                probs = torch.softmax(logits, dim=1)
            preds_epoch.append(probs.detach().cpu().numpy())
        preds_epoch = np.concatenate(preds_epoch, axis=0)
        all_preds.append(preds_epoch)

tst_preds = np.mean(all_preds, axis=0)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/2313775436.py in <cell line: 0>()
     21 )
     22 
---> 23 model.eval()
     24 all_preds = []
     25 with torch.no_grad():

NameError: name 'model' is not defined

## === cell 10
test_df["label"] = np.argmax(tst_preds, axis=1)
test_df.to_csv("submission.csv", index=False)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_57/2159542246.py in <cell line: 0>()
----> 1 test_df["label"] = np.argmax(tst_preds, axis=1)
      2 test_df.to_csv("submission.csv", index=False)

NameError: name 'tst_preds' is not defined
