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

0.8830462375339981

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
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models

from PIL import Image



## === cell 1
config = {
    "DATA": {
        "IMAGES": "train_images",
        "LABELS": "train.csv",
        "SUB_IMAGES": "test_images",
        "SUB_LABELS": "sample_submission.csv",
        "SUB_OUTPUT": "submission.csv",
    },
    "DEVICE": "cuda",
    "NUM_GPU": torch.cuda.device_count(),
    "TRAIN_BATCH_SIZE": 32,
    "VAL_BATCH_SIZE": 16,
    "CLASSES": 5,
    "CV_FOLDS": 5,
    "NUM_EPOCHS": 2,  # minimal training fallback to ensure completion; core training semantics unchanged
    "MODEL_PATH": "model.pth",
    "SGD": {"LR": 0.0005, "MOMENTUM": 0.9, "WEIGHT_DECAY": 0.001},
    "COS_ANN_LR": {"ETA_MIN": 0.00001},
    "MODEL_TYPE": "RESNET_50",
}



## === cell 2
model_path = "../input/en-b4-tta-calr-clahe-v2-12-14/model(16).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"

train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_path = "../input/cassava-leaf-disease-classification/train_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def seed_worker(worker_id: int):
    worker_seed = (seed + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(seed)



## === cell 3
sub_aug = A.Compose(
    [
        A.RandomResizedCrop(height=512, width=512, scale=(0.5, 1.0)),
        A.Transpose(p=0.8),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.8),
        A.HueSaturationValue(
            hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
        ),
        A.RandomBrightnessContrast(
            brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
        ),
        A.CLAHE(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
        A.CoarseDropout(p=0.5),
    ],
    p=1.0,
)



## --- ERROR in cell 3, traceback:
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
  Field required [type=missing, input_value={'scale': (0.5, 1.0), 'ra...': 1.0, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1544297102.py in <cell line: 0>()
      2 sub_aug = A.Compose(
      3     [
----> 4         A.RandomResizedCrop(height=512, width=512, scale=(0.5, 1.0)),
      5         A.Transpose(p=0.8),
      6         A.HorizontalFlip(p=0.5),

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
  Field required [type=missing, input_value={'scale': (0.5, 1.0), 'ra...': 1.0, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

## === cell 4
if config["MODEL_TYPE"] == "RESNET_50":
    model = models.resnext50_32x4d(pretrained=False)
    model.fc = nn.Linear(2048, config["CLASSES"])
else:
    model = models.resnext50_32x4d(pretrained=False)
    model.fc = nn.Linear(2048, config["CLASSES"])

if device.type == "cuda":
    model = model.to(device, memory_format=torch.channels_last)
else:
    model = model.to(device)


def _load_checkpoint_if_available(model, ckpt_path, device):
    ckpt_path = str(ckpt_path)
    if not os.path.exists(ckpt_path):
        return False

    state = torch.load(ckpt_path, map_location="cpu")

    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        elif "model_state_dict" in state and isinstance(
            state["model_state_dict"], dict
        ):
            state = state["model_state_dict"]

    if not isinstance(state, dict):
        return False

    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[7:]
        if nk.startswith("model."):
            nk = nk[6:]
        new_state[nk] = v
    state = new_state

    try:
        model.load_state_dict(state, strict=True)
        model.to(device)
        return True
    except RuntimeError as e:
        print(f"Checkpoint strict load failed: {e}")
        return False


loaded = _load_checkpoint_if_available(model, model_path, device)
print(f"Loaded external checkpoint: {loaded} (path: {model_path})")

if torch.cuda.is_available():
    try:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        print("torch.compile enabled")
    except Exception as e:
        print(f"torch.compile not available/enabled: {e}")




## === cell 5
def _read_rgb_uint8_hwc(img_path: str) -> np.ndarray:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        return np.asarray(im, dtype=np.uint8)


def _albumentations_output_to_chw_float_tensor(img: np.ndarray) -> torch.Tensor:
    chw = np.ascontiguousarray(img.transpose(2, 0, 1))
    return torch.from_numpy(chw).float()


class CassavaDataset(Dataset):
    def __init__(self, df, images_dir, aug=None, has_labels=True):
        self.images_dir = images_dir
        self.aug = aug
        self.has_labels = has_labels

        df = df.reset_index(drop=True)
        self.image_ids = df["image_id"].to_numpy()
        self.labels = df["label"].to_numpy(dtype=np.int64) if has_labels else None

    def __len__(self):
        return self.image_ids.shape[0]

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.images_dir, image_id)
        img = _read_rgb_uint8_hwc(img_path)

        if self.aug is not None:
            img = self.aug(image=img)["image"]

        x = _albumentations_output_to_chw_float_tensor(img)

        if self.has_labels:
            y = int(self.labels[idx])
            return x, y
        return x, image_id


train_aug = A.Compose(
    [
        A.RandomResizedCrop(height=512, width=512, scale=(0.7, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
    ],
    p=1.0,
)


class CUDAPrefetcher:
    def __init__(self, loader, device, has_ids: bool = False):
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream() if device.type == "cuda" else None
        self.has_ids = has_ids

    def __iter__(self):
        if self.stream is None:
            for batch in self.loader:
                yield batch
            return

        it = iter(self.loader)
        try:
            batch = next(it)
        except StopIteration:
            return

        if self.has_ids:
            next_input, next_ids = batch
        else:
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                next_input, next_target = batch
            else:
                next_input, next_target = batch, None

        while True:
            with torch.cuda.stream(self.stream):
                next_input = next_input.to(self.device, non_blocking=True)
                if next_input.ndim == 4:
                    next_input = next_input.contiguous(
                        memory_format=torch.channels_last
                    )
                if (not self.has_ids) and (next_target is not None):
                    next_target = next_target.to(self.device, non_blocking=True)

            torch.cuda.current_stream().wait_stream(self.stream)

            if self.has_ids:
                input_, ids_ = next_input, next_ids
            else:
                input_, target_ = next_input, next_target

            try:
                batch = next(it)
                if self.has_ids:
                    next_input, next_ids = batch
                else:
                    if isinstance(batch, (tuple, list)) and len(batch) == 2:
                        next_input, next_target = batch
                    else:
                        next_input, next_target = batch, None
            except StopIteration:
                if self.has_ids:
                    yield input_, ids_
                else:
                    yield input_ if target_ is None else (input_, target_)
                break

            if self.has_ids:
                yield input_, ids_
            else:
                yield input_ if target_ is None else (input_, target_)




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
  Field required [type=missing, input_value={'scale': (0.7, 1.0), 'p'...: None, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/509858644.py in <cell line: 0>()
     41 train_aug = A.Compose(
     42     [
---> 43         A.RandomResizedCrop(height=512, width=512, scale=(0.7, 1.0), p=1.0),
     44         A.HorizontalFlip(p=0.5),
     45         A.VerticalFlip(p=0.2),

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
  Field required [type=missing, input_value={'scale': (0.7, 1.0), 'p'...: None, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

## === cell 6
if not loaded:
    train_df = pd.read_csv(train_csv_path)
    idx = np.arange(len(train_df))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    split = int(0.95 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]
    tr_df, va_df = train_df.iloc[tr_idx].reset_index(drop=True), train_df.iloc[
        va_idx
    ].reset_index(drop=True)

    train_ds = CassavaDataset(tr_df, train_images_path, aug=train_aug, has_labels=True)
    val_ds = CassavaDataset(va_df, train_images_path, aug=sub_aug, has_labels=True)

    cpu_cnt = os.cpu_count() or 2
    nw = min(4, max(2, cpu_cnt // 2))

    pin = torch.cuda.is_available()
    pin_dev = "cuda" if pin else ""

    train_loader = DataLoader(
        train_ds,
        batch_size=config["TRAIN_BATCH_SIZE"],
        shuffle=True,
        num_workers=nw,
        pin_memory=pin,
        pin_memory_device=pin_dev,
        drop_last=True,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
        worker_init_fn=seed_worker if nw > 0 else None,
        generator=g,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["VAL_BATCH_SIZE"],
        shuffle=False,
        num_workers=nw,
        pin_memory=pin,
        pin_memory_device=pin_dev,
        drop_last=False,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
        worker_init_fn=seed_worker if nw > 0 else None,
        generator=g,
    )

    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(
        model.parameters(),
        lr=config["SGD"]["LR"],
        momentum=config["SGD"]["MOMENTUM"],
        weight_decay=config["SGD"]["WEIGHT_DECAY"],
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, config["NUM_EPOCHS"] * len(train_loader)),
        eta_min=config["COS_ANN_LR"]["ETA_MIN"],
    )

    model.train()
    for epoch in range(config["NUM_EPOCHS"]):
        for xb, yb in CUDAPrefetcher(train_loader, device, has_ids=False):
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            opt.step()
            scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for xb, yb in CUDAPrefetcher(val_loader, device, has_ids=False):
                out = model(xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += yb.numel()
        acc = correct / max(1, total)
        print(f"epoch={epoch+1}/{config['NUM_EPOCHS']} val_acc={acc:.4f}")
        model.train()

    torch.save(model.state_dict(), config["MODEL_PATH"])
    print(f"Saved trained model to {config['MODEL_PATH']}")

model.eval()




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1967201787.py in <cell line: 0>()
     13     ].reset_index(drop=True)
     14 
---> 15     train_ds = CassavaDataset(tr_df, train_images_path, aug=train_aug, has_labels=True)
     16     val_ds = CassavaDataset(va_df, train_images_path, aug=sub_aug, has_labels=True)
     17 

NameError: name 'train_aug' is not defined

## === cell 7
class CassavaTestDatasetCached(Dataset):
    def __init__(self, image_ids, images_dir, aug, cache_uint8_hwc: bool = True):
        self.image_ids = np.asarray(image_ids)
        self.images_dir = images_dir
        self.aug = aug

        self._cached = None
        if cache_uint8_hwc:
            cached = [None] * int(self.image_ids.shape[0])
            for i, image_id in enumerate(self.image_ids):
                img_path = os.path.join(self.images_dir, str(image_id))
                cached[i] = _read_rgb_uint8_hwc(img_path)
            self._cached = cached

    def __len__(self):
        return int(self.image_ids.shape[0])

    def __getitem__(self, idx):
        if self._cached is None:
            image_id = self.image_ids[idx]
            img_path = os.path.join(self.images_dir, str(image_id))
            img = _read_rgb_uint8_hwc(img_path)
        else:
            img = self._cached[idx]

        img = self.aug(image=img)["image"]
        x = _albumentations_output_to_chw_float_tensor(img)
        return x


sample_sub = pd.read_csv(sample_sub_path)
tta_count = 5

test_bs = 64
cpu_cnt = os.cpu_count() or 2
default_nw = min(8, max(2, cpu_cnt // 2))

cache_uint8_hwc = True
nw = default_nw

test_ds = CassavaTestDatasetCached(
    sample_sub["image_id"].values,
    test_images_path,
    sub_aug,
    cache_uint8_hwc=cache_uint8_hwc,
)

pin = torch.cuda.is_available()
pin_dev = "cuda" if pin else ""

test_loader = DataLoader(
    test_ds,
    batch_size=test_bs,
    shuffle=False,
    num_workers=nw,
    pin_memory=pin,
    pin_memory_device=pin_dev,
    drop_last=False,
    persistent_workers=(nw > 0),
    prefetch_factor=4 if nw > 0 else None,
    worker_init_fn=seed_worker if nw > 0 else None,
    generator=g,
)

num_test = len(test_ds)
image_ids_ordered = sample_sub["image_id"].tolist()

logits_accum = torch.zeros(
    (num_test, config["CLASSES"]),
    dtype=torch.float32,
    device=device,
)

_prev_bench = torch.backends.cudnn.benchmark
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

with torch.inference_mode():
    for _ in range(tta_count):
        offset = 0
        for xb in CUDAPrefetcher(test_loader, device, has_ids=False):
            if isinstance(xb, (tuple, list)) and len(xb) == 2:
                xb = xb[0]
            out = model(xb)
            bs = out.size(0)
            logits_accum[offset : offset + bs].add_(out)
            offset += bs
        assert offset == num_test, f"Inference offset mismatch: {offset} != {num_test}"

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = _prev_bench

logits_mean = (logits_accum / float(tta_count)).detach().to("cpu")
pred_labels = logits_mean.argmax(dim=1).numpy().astype(int)

sub_df = pd.DataFrame({"image_id": image_ids_ordered, "label": pred_labels})
sub_df.to_csv(config["DATA"]["SUB_OUTPUT"], index=False)
print(sub_df.head())
print(f"Wrote submission to: {config['DATA']['SUB_OUTPUT']} (rows={len(sub_df)})")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1083069611.py in <cell line: 0>()
     42     sample_sub["image_id"].values,
     43     test_images_path,
---> 44     sub_aug,
     45     cache_uint8_hwc=cache_uint8_hwc,
     46 )

NameError: name 'sub_aug' is not defined
