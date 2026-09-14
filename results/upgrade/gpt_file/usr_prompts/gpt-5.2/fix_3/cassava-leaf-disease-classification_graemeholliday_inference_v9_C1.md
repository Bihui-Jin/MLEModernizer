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

0.8927168328800241

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import time
import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import cv2

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from tqdm import tqdm

from torchvision import models



## === cell 1
image_size = 380



## === cell 2
config = dict(
    seed=22,
    experiment_name="baseline",
    test_location="../input/cassava-leaf-disease-classification/test_images",
    checkpoint_path="../input/saved-models",
    checkpoint="baselinebest.pt",
    model="efficientnet-b4",
    epochs=10,
    batch_size=32,
    workers=8,
    inference_augmentations=[
        dict(
            name="RandomResizedCrop", params=dict(height=image_size, width=image_size)
        ),
        dict(
            name="HorizontalFlip",
            params=dict(
                always_apply=False,
                p=0.5,
            ),
        ),
        dict(name="Transpose", params=dict(p=0.5)),
        dict(name="VerticalFlip", params=dict(always_apply=False, p=0.5)),
        dict(
            name="HueSaturationValue",
            params=dict(
                hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5
            ),
        ),
        dict(
            name="RandomBrightnessContrast",
            params=dict(
                brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5
            ),
        ),
        dict(
            name="Normalize",
            params=dict(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                max_pixel_value=255.0,
                p=1.0,
            ),
        ),
        dict(name="CoarseDropout", params=dict(p=0.5)),
        dict(name="Cutout", params=dict(p=0.5)),
    ],
)




## === cell 3
def _resolve_path(path: str) -> str:
    if os.path.exists(path):
        return path
    candidates = [
        path,
        path.replace("../input", "/kaggle/input"),
        path.replace("../input", "/kaggle/data/input"),
        path.replace("../input", "/kaggle/data"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return path


sample_path = _resolve_path(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_path)
test = sample_sub[["image_id"]].copy()
test.head()




## === cell 4
def seed(seed=22):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed(config["seed"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 5
def _strip_prefix_if_present(
    state_dict: Dict[str, torch.Tensor], prefix: str
) -> Dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in state_dict.keys()):
        return state_dict
    return {
        k[len(prefix) :] if k.startswith(prefix) else k: v
        for k, v in state_dict.items()
    }


def _find_checkpoint_file(checkpoint_name: str, base_dir: Optional[str] = None) -> str:
    search_roots = []
    if base_dir is not None:
        search_roots.append(_resolve_path(base_dir))
    search_roots += [
        _resolve_path("../input"),
        "/kaggle/input",
        "/kaggle/data/input",
        "/kaggle/data",
        ".",
    ]
    seen = set()
    for root in search_roots:
        if not root or root in seen:
            continue
        seen.add(root)
        direct = os.path.join(root, checkpoint_name)
        if os.path.exists(direct):
            return direct
        if os.path.exists(root) and os.path.isdir(root):
            for dirpath, _, filenames in os.walk(root):
                if checkpoint_name in filenames:
                    return os.path.join(dirpath, checkpoint_name)
    return os.path.join(base_dir or "", checkpoint_name)


def load_model():
    if config["model"] != "efficientnet-b4":
        raise ValueError(
            f"Only efficientnet-b4 is supported by this script, got {config['model']}"
        )

    model = models.efficientnet_b4(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 5)

    ckpt_path = _find_checkpoint_file(config["checkpoint"], config["checkpoint_path"])
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: tried to locate '{config['checkpoint']}' starting from "
            f"'{config['checkpoint_path']}'. Final attempted path: {ckpt_path}"
        )

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    state = (
        checkpoint.get("model", checkpoint)
        if isinstance(checkpoint, dict)
        else checkpoint
    )
    state = _strip_prefix_if_present(state, "module.")
    state = _strip_prefix_if_present(state, "model.")

    missing, unexpected = model.load_state_dict(state, strict=False)

    print("Loading model from checkpoint:", ckpt_path)
    if isinstance(checkpoint, dict):
        if "epoch" in checkpoint:
            print("Epoch", int(checkpoint["epoch"]))
        if "train_loss" in checkpoint:
            print("Train loss", checkpoint["train_loss"])
        if "val_loss" in checkpoint:
            print("Validation loss", checkpoint["val_loss"])
        if "metrics" in checkpoint:
            print("Accuracy", checkpoint["metrics"])
        if "lr" in checkpoint:
            print("Learning rate", checkpoint["lr"])

    if missing:
        print(
            f"Warning: missing keys when loading checkpoint (showing up to 20): {missing[:20]}"
        )
    if unexpected:
        print(
            f"Warning: unexpected keys when loading checkpoint (showing up to 20): {unexpected[:20]}"
        )

    return model.to(device)




## === cell 6
def get_transforms():
    transforms: List[A.BasicTransform] = []
    for item in config["inference_augmentations"]:
        name = item["name"]
        params = dict(item["params"])

        try:
            if name == "RandomResizedCrop":
                pass

            if name == "Cutout":
                name = "CoarseDropout"
                params.setdefault("num_holes_range", (1, 8))
                params.setdefault(
                    "hole_height_range", (int(image_size * 0.05), int(image_size * 0.2))
                )
                params.setdefault(
                    "hole_width_range", (int(image_size * 0.05), int(image_size * 0.2))
                )
                params.setdefault("fill", 0)

            if name == "CoarseDropout":
                params.setdefault("num_holes_range", (1, 8))
                params.setdefault(
                    "hole_height_range", (int(image_size * 0.05), int(image_size * 0.2))
                )
                params.setdefault(
                    "hole_width_range", (int(image_size * 0.05), int(image_size * 0.2))
                )
                params.setdefault("fill", 0)

            transforms.append(getattr(A, name)(**params))
        except Exception as e:
            print(
                f"Warning: failed to build transform {name} with params={params}. Error: {e}"
            )
            transforms = [
                A.Resize(image_size, image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0,
                    p=1.0,
                ),
            ]
            break

    return A.Compose(transforms)




## === cell 7
class CassavaDataset(Dataset):
    def __init__(self, images, transforms):
        self.images = images
        self.transforms = transforms

    def __getitem__(self, n):
        image_path = os.path.join(
            _resolve_path(config["test_location"]), self.images[n]
        )
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        image = np.moveaxis(image, -1, 0)
        image = torch.as_tensor(image, dtype=torch.float32)
        return image

    def __len__(self):
        return len(self.images)




## === cell 8
def get_dataloader():
    transforms = get_transforms()
    test_data = np.array(test["image_id"])

    data = CassavaDataset(test_data, transforms)

    num_workers = int(config["workers"])
    if device.type == "cpu":
        num_workers = min(num_workers, 2)

    dataloader = DataLoader(
        data,
        shuffle=False,
        batch_size=config["batch_size"],
        pin_memory=(device.type == "cuda"),
        num_workers=num_workers,
    )
    return dataloader




## === cell 9
def infer(model, dataloader):
    print("Running inference...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device, non_blocking=(device.type == "cuda"))
            batch_hat = model(batch)
            predictions.append(batch_hat.detach().cpu())

    return torch.cat(predictions, dim=0)




## === cell 10
if torch.cuda.is_available():
    torch.cuda.empty_cache()

dataloader = get_dataloader()
model = load_model()
predictions = None
print("Inferring experiment", config["experiment_name"])

for epoch in range(int(config["epochs"])):
    print("Epoch:", epoch)
    start_time = time.time()

    if epoch == 0:
        predictions = infer(model, dataloader)
    else:
        predictions += infer(model, dataloader)

    print("Time:", time.time() - start_time)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

predictions /= float(config["epochs"])
results = predictions.numpy()
test["label"] = np.argmax(results, axis=-1).astype(int)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2812916740.py in <cell line: 0>()
      4 
      5 dataloader = get_dataloader()
----> 6 model = load_model()
      7 predictions = None
      8 print("Inferring experiment", config["experiment_name"])

/tmp/ipykernel_55/2710476591.py in load_model()
     52     ckpt_path = _find_checkpoint_file(config["checkpoint"], config["checkpoint_path"])
     53     if not os.path.exists(ckpt_path):
---> 54         raise FileNotFoundError(
     55             f"Checkpoint not found: tried to locate '{config['checkpoint']}' starting from "
     56             f"'{config['checkpoint_path']}'. Final attempted path: {ckpt_path}"

FileNotFoundError: Checkpoint not found: tried to locate 'baselinebest.pt' starting from '../input/saved-models'. Final attempted path: ../input/saved-models/baselinebest.pt

## === cell 11
submission = sample_sub[["image_id"]].merge(
    test[["image_id", "label"]], on="image_id", how="left"
)
if submission["label"].isna().any():
    missing_ids = (
        submission.loc[submission["label"].isna(), "image_id"].head(10).tolist()
    )
    raise RuntimeError(
        f"Missing predictions for some test images (showing up to 10): {missing_ids}"
    )

submission = submission[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/751614047.py in <cell line: 0>()
      1 # Ensure exact submission format/length/order matches sample_submission.
      2 submission = sample_sub[["image_id"]].merge(
----> 3     test[["image_id", "label"]], on="image_id", how="left"
      4 )
      5 if submission["label"].isna().any():

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['label'] not in index"
