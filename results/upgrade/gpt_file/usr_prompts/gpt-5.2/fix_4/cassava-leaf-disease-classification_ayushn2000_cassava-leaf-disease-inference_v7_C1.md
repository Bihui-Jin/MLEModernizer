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

0.8797219703838017

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import sys
import os

package_path = "../input/pytorchimagemodels"
if os.path.isdir(package_path) and package_path not in sys.path:
    sys.path.append(package_path)



## === cell 1
import random
import cv2
import timm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import albumentations.pytorch as Apy

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm



## === cell 2
config = {
    "fold_num": 5,
    "seed": 719,
    "model_arch_1": "tf_efficientnet_b4_ns",
    "model_arch_2": "tf_efficientnet_b3_ns",
    "img_size": 384,
    "valid_bs": 96,
    "num_workers": 0,  # bugfix/stability: avoid multiprocess dataloader issues in Kaggle
    "accum_iter": 1,
    "verbose_step": 1,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
    "train_bs": 16,
    "epochs": 1,
    "lr": 3e-4,
}



## === cell 3
submission = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
submission.head()




## === cell 4
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_img(path):
    im_bgr = cv2.imread(path)
    if im_bgr is None:
        raise FileNotFoundError(f"Could not read image at path: {path}")
    return im_bgr[:, :, ::-1]




## === cell 5
class CassavaDataset(Dataset):
    def __init__(self, df, data_root, transforms=None, output_label=True):
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms = transforms
        self.data_root = data_root
        self.output_label = output_label

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        if self.output_label:
            target = int(self.df.iloc[index]["label"])

        path = "{}/{}".format(self.data_root, self.df.iloc[index]["image_id"])
        img = get_img(path)

        if self.transforms:
            img = self.transforms(image=img)["image"]

        if self.output_label:
            return img, target
        else:
            return img




## === cell 6
def get_inference_transforms():
    return A.Compose(
        [
            A.Resize(config["img_size"], config["img_size"]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            Apy.ToTensorV2(p=1.0),
        ],
        p=1.0,
    )


def get_train_transforms():
    return A.Compose(
        [
            A.RandomResizedCrop(
                height=config["img_size"],
                width=config["img_size"],
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
            Apy.ToTensorV2(p=1.0),
        ],
        p=1.0,
    )




## === cell 7
class CassvaImgClassifier_1(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x


class CassvaImgClassifier_2(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x




## === cell 8
def inference_one_epoch(model_1, model_2, data_loader, device):
    model_1.eval()
    model_2.eval()

    image_preds_all = []

    pbar = tqdm(enumerate(data_loader), total=len(data_loader))
    with torch.no_grad():
        for step, imgs in pbar:
            imgs = imgs.to(device).float()

            image_preds_1 = model_1(imgs)
            image_preds_2 = model_2(imgs)

            image_preds = (image_preds_1 + image_preds_2) / 2.0
            image_preds_all.append(torch.softmax(image_preds, 1).detach().cpu().numpy())

    image_preds_all = np.concatenate(image_preds_all, axis=0)
    return image_preds_all




## === cell 9
def _find_weight_file(model_arch: str, fold: int):
    candidates = [
        f"../input/cassava-leaf-disease/{model_arch}_fold_{fold}",
        f"../input/cassava-leaf-disease/{model_arch}_fold_{fold}.pth",
        f"../input/cassava-leaf-disease/{model_arch}_fold_{fold}.bin",
        f"../input/cassava-leave-disease/{model_arch}_fold_{fold}",  # original typo path
        f"../input/cassava-leave-disease/{model_arch}_fold_{fold}.pth",
        f"../input/cassava-leave-disease/{model_arch}_fold_{fold}.bin",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    search_root = "../input"
    matches = []
    for root, _, files in os.walk(search_root):
        for fn in files:
            if fn == f"{model_arch}_fold_{fold}" or fn.startswith(
                f"{model_arch}_fold_{fold}."
            ):
                matches.append(os.path.join(root, fn))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        for suf in [".pth", ".pt", ".bin"]:
            for m in matches:
                if m.endswith(suf):
                    return m
        return sorted(matches)[0]

    return None


def _load_state_dict_flexible(model: nn.Module, state):
    if (
        isinstance(state, dict)
        and "state_dict" in state
        and isinstance(state["state_dict"], dict)
    ):
        state = state["state_dict"]

    if not isinstance(state, dict):
        raise ValueError("Loaded checkpoint is not a state_dict-like dict.")

    model_keys = set(model.state_dict().keys())
    state_keys = set(state.keys())

    if all(k.startswith("module.") for k in state_keys) and not any(
        k.startswith("module.") for k in model_keys
    ):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)




## === cell 10
seed_everything(config["seed"])

train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_dir = "../input/cassava-leaf-disease-classification/train_images/"
test_dir = "../input/cassava-leaf-disease-classification/test_images/"

train_df = pd.read_csv(train_csv_path)
test = pd.DataFrame()
test["image_id"] = sorted(
    [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
)

train_ds = CassavaDataset(
    train_df,
    train_dir,
    transforms=get_train_transforms(),
    output_label=True,
)
train_loader = DataLoader(
    train_ds,
    batch_size=config["train_bs"],
    num_workers=config["num_workers"],
    shuffle=True,
    pin_memory=True,
    drop_last=True,
)

test_ds = CassavaDataset(
    test,
    test_dir,
    transforms=get_inference_transforms(),
    output_label=False,
)
tst_loader = DataLoader(
    test_ds,
    batch_size=config["valid_bs"],
    num_workers=config["num_workers"],
    shuffle=False,
    pin_memory=True,
)

device = torch.device(config["device"])

model_1 = CassvaImgClassifier_1(config["model_arch_1"], 5, pretrained=False).to(device)
model_2 = CassvaImgClassifier_2(config["model_arch_2"], 5, pretrained=False).to(device)

have_all_weights = True
weight_files = []
for fold in range(config["fold_num"]):
    w1 = _find_weight_file(config["model_arch_1"], fold)
    w2 = _find_weight_file(config["model_arch_2"], fold)
    weight_files.append((w1, w2))
    if w1 is None or w2 is None:
        have_all_weights = False

if not have_all_weights:
    del model_1, model_2
    torch.cuda.empty_cache()

    model_1 = CassvaImgClassifier_1(config["model_arch_1"], 5, pretrained=True).to(
        device
    )
    model_2 = CassvaImgClassifier_2(config["model_arch_2"], 5, pretrained=True).to(
        device
    )

    criterion = nn.CrossEntropyLoss()
    opt_1 = torch.optim.AdamW(model_1.parameters(), lr=config["lr"])
    opt_2 = torch.optim.AdamW(model_2.parameters(), lr=config["lr"])

    model_1.train()
    model_2.train()
    for epoch in range(config["epochs"]):
        pbar = tqdm(train_loader, total=len(train_loader))
        for imgs, targets in pbar:
            imgs = imgs.to(device).float()
            targets = targets.to(device).long()

            opt_1.zero_grad(set_to_none=True)
            opt_2.zero_grad(set_to_none=True)

            logits1 = model_1(imgs)
            logits2 = model_2(imgs)
            loss = (criterion(logits1, targets) + criterion(logits2, targets)) / 2.0

            loss.backward()
            opt_1.step()
            opt_2.step()

            pbar.set_description(
                f"finetune epoch {epoch+1}/{config['epochs']} loss {loss.item():.4f}"
            )

    tst_preds = []
    for fold in range(config["fold_num"]):
        tst_preds.append(inference_one_epoch(model_1, model_2, tst_loader, device))
    tst_preds = np.mean(np.stack(tst_preds, axis=0), axis=0)
else:
    tst_preds = []
    for fold in range(config["fold_num"]):
        w1, w2 = weight_files[fold]
        state1 = torch.load(w1, map_location=device)
        state2 = torch.load(w2, map_location=device)

        _load_state_dict_flexible(model_1, state1)
        _load_state_dict_flexible(model_2, state2)

        tst_preds.append(inference_one_epoch(model_1, model_2, tst_loader, device))

    tst_preds = np.mean(np.stack(tst_preds, axis=0), axis=0)

del model_1, model_2
torch.cuda.empty_cache()



## --- ERROR in cell 10, traceback:
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
  Field required [type=missing, input_value={'scale': (0.8, 1.0), 'ra...: None, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_56/1999791979.py in <cell line: 0>()
     17     train_df,
     18     train_dir,
---> 19     transforms=get_train_transforms(),
     20     output_label=True,
     21 )

/tmp/ipykernel_56/2905724741.py in get_train_transforms()
     19     return A.Compose(
     20         [
---> 21             A.RandomResizedCrop(
     22                 height=config["img_size"],
     23                 width=config["img_size"],

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
  Field required [type=missing, input_value={'scale': (0.8, 1.0), 'ra...: None, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

## === cell 11
tst_preds = np.asarray(tst_preds)
if tst_preds.ndim != 2 or tst_preds.shape[0] != len(test) or tst_preds.shape[1] != 5:
    raise ValueError(
        f"Unexpected tst_preds shape: {tst_preds.shape}. Expected (n_test, 5) with n_test={len(test)}."
    )

test["label"] = np.argmax(tst_preds, axis=1).astype(int)

sub = test[["image_id", "label"]].copy()
sample = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)

sub = sample[["image_id"]].merge(sub, on="image_id", how="left")
if sub["label"].isna().any():
    missing = int(sub["label"].isna().sum())
    raise ValueError(
        f"Missing predictions for {missing} test images after merge/alignment."
    )
sub["label"] = sub["label"].astype(int)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/228241729.py in <cell line: 0>()
----> 1 tst_preds = np.asarray(tst_preds)
      2 if tst_preds.ndim != 2 or tst_preds.shape[0] != len(test) or tst_preds.shape[1] != 5:
      3     raise ValueError(
      4         f"Unexpected tst_preds shape: {tst_preds.shape}. Expected (n_test, 5) with n_test={len(test)}."
      5     )

NameError: name 'tst_preds' is not defined
