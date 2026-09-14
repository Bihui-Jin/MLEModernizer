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

3.13

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

0.1548806285886975

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import Dataset, DataLoader

import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2



## === cell 1
num_tta = 5



## === cell 2
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"



## === cell 3
test_df = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_df.head()



## === cell 4
efficientnet_transforms = A.Compose(
    [
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
        A.Resize(384, 384),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 5
class CassavaTestDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image at: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            image = torch.tensor(image, dtype=torch.float32)

        return image, img_name




## === cell 6
test_dataset = CassavaTestDataset(
    test_df, test_image_dir, transform=efficientnet_transforms
)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)



## === cell 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 8


def find_first_existing(paths):
    for p in paths:
        if p and Path(p).exists():
            return str(p)
    return None


def find_by_pattern(root, pattern):
    root = Path(root)
    matches = list(root.rglob(pattern))
    matches = [m for m in matches if m.is_file()]
    return str(matches[0]) if matches else None


resnet_weight = find_first_existing(
    [
        "/kaggle/input/casava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth",
        "/kaggle/input/cassava-aug/pytorch/default/1/cassava_leaf_best_model_fine_aug.pth",  # keep duplicate harmless
    ]
)
if resnet_weight is None:
    resnet_weight = find_by_pattern(
        "/kaggle/input", "*cassava*_aug*.pth"
    ) or find_by_pattern("/kaggle/input", "*resnet*.pth")

eff5_weight = find_first_existing(
    [
        "/kaggle/input/eff-5/pytorch/default/1/Eff_best5.pth",
    ]
)
if eff5_weight is None:
    eff5_weight = find_by_pattern(
        "/kaggle/input", "*Eff_best5*.pth"
    ) or find_by_pattern("/kaggle/input", "*best5*.pth")

eff7_weight = find_first_existing(
    [
        "/kaggle/input/eff-7/pytorch/default/1/Eff_best7.pth",
    ]
)
if eff7_weight is None:
    eff7_weight = find_by_pattern(
        "/kaggle/input", "*Eff_best7*.pth"
    ) or find_by_pattern("/kaggle/input", "*best7*.pth")

eff6_weight = find_first_existing(
    [
        "/kaggle/input/eff-6/pytorch/default/1/Eff_best6.pth",
    ]
)
if eff6_weight is None:
    eff6_weight = find_by_pattern(
        "/kaggle/input", "*Eff_best6*.pth"
    ) or find_by_pattern("/kaggle/input", "*best6*.pth")

print("Resolved weights:")
print("  resnet :", resnet_weight)
print("  eff5   :", eff5_weight)
print("  eff7   :", eff7_weight)
print("  eff6   :", eff6_weight)

if any(w is None for w in [resnet_weight, eff5_weight, eff7_weight, eff6_weight]):
    raise FileNotFoundError(
        "One or more model weight files could not be found under /kaggle/input. "
        "Please ensure the datasets (casava-aug, eff-5, eff-7, eff-6) are attached."
    )



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/615089766.py in <cell line: 0>()
     66 
     67 if any(w is None for w in [resnet_weight, eff5_weight, eff7_weight, eff6_weight]):
---> 68     raise FileNotFoundError(
     69         "One or more model weight files could not be found under /kaggle/input. "
     70         "Please ensure the datasets (casava-aug, eff-5, eff-7, eff-6) are attached."

FileNotFoundError: One or more model weight files could not be found under /kaggle/input. Please ensure the datasets (casava-aug, eff-5, eff-7, eff-6) are attached.

## === cell 9
resnet_model = models.resnet50(pretrained=False)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)

resnet_model.load_state_dict(torch.load(resnet_weight, map_location=device))
resnet_model = resnet_model.to(device)
resnet_model.eval()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _check_seekable(f)
    850     try:
--> 851         f.seek(f.tell())
    852         return True

AttributeError: 'NoneType' object has no attribute 'seek'

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3146473359.py in <cell line: 0>()
      5 
      6 # Change: use map_location=device for stability across CPU/GPU.
----> 7 resnet_model.load_state_dict(torch.load(resnet_weight, map_location=device))
      8 resnet_model = resnet_model.to(device)
      9 resnet_model.eval()

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    754             return _open_buffer_writer(name_or_buffer)
    755         elif "r" in mode:
--> 756             return _open_buffer_reader(name_or_buffer)
    757         else:
    758             raise RuntimeError(f"Expected 'r' or 'w' in mode but got {mode}")

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, buffer)
    739     def __init__(self, buffer):
    740         super().__init__(buffer)
--> 741         _check_seekable(buffer)
    742 
    743 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _check_seekable(f)
    852         return True
    853     except (io.UnsupportedOperation, AttributeError) as e:
--> 854         raise_err_msg(["seek", "tell"], e)
    855     return False
    856 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in raise_err_msg(patterns, e)
    845                     + " try to load from it instead."
    846                 )
--> 847                 raise type(e)(msg)
    848         raise e
    849 

AttributeError: 'NoneType' object has no attribute 'seek'. You can only torch.load from a file that is seekable. Please pre-load the data into a buffer like io.BytesIO and try to load from it instead.

## === cell 10
efficientnet_model_1 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_1.classifier[1].in_features
efficientnet_model_1.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)
efficientnet_model_1.load_state_dict(torch.load(eff5_weight, map_location=device))
efficientnet_model_1 = efficientnet_model_1.to(device)
efficientnet_model_1.eval()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _check_seekable(f)
    850     try:
--> 851         f.seek(f.tell())
    852         return True

AttributeError: 'NoneType' object has no attribute 'seek'

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3834115597.py in <cell line: 0>()
      5     nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
      6 )
----> 7 efficientnet_model_1.load_state_dict(torch.load(eff5_weight, map_location=device))
      8 efficientnet_model_1 = efficientnet_model_1.to(device)
      9 efficientnet_model_1.eval()

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    754             return _open_buffer_writer(name_or_buffer)
    755         elif "r" in mode:
--> 756             return _open_buffer_reader(name_or_buffer)
    757         else:
    758             raise RuntimeError(f"Expected 'r' or 'w' in mode but got {mode}")

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, buffer)
    739     def __init__(self, buffer):
    740         super().__init__(buffer)
--> 741         _check_seekable(buffer)
    742 
    743 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _check_seekable(f)
    852         return True
    853     except (io.UnsupportedOperation, AttributeError) as e:
--> 854         raise_err_msg(["seek", "tell"], e)
    855     return False
    856 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in raise_err_msg(patterns, e)
    845                     + " try to load from it instead."
    846                 )
--> 847                 raise type(e)(msg)
    848         raise e
    849 

AttributeError: 'NoneType' object has no attribute 'seek'. You can only torch.load from a file that is seekable. Please pre-load the data into a buffer like io.BytesIO and try to load from it instead.

## === cell 11
efficientnet_model_7 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_7.classifier[1].in_features
efficientnet_model_7.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)
efficientnet_model_7.load_state_dict(torch.load(eff7_weight, map_location=device))
efficientnet_model_7 = efficientnet_model_7.to(device)
efficientnet_model_7.eval()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _check_seekable(f)
    850     try:
--> 851         f.seek(f.tell())
    852         return True

AttributeError: 'NoneType' object has no attribute 'seek'

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3062749716.py in <cell line: 0>()
      4     nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
      5 )
----> 6 efficientnet_model_7.load_state_dict(torch.load(eff7_weight, map_location=device))
      7 efficientnet_model_7 = efficientnet_model_7.to(device)
      8 efficientnet_model_7.eval()

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    754             return _open_buffer_writer(name_or_buffer)
    755         elif "r" in mode:
--> 756             return _open_buffer_reader(name_or_buffer)
    757         else:
    758             raise RuntimeError(f"Expected 'r' or 'w' in mode but got {mode}")

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, buffer)
    739     def __init__(self, buffer):
    740         super().__init__(buffer)
--> 741         _check_seekable(buffer)
    742 
    743 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _check_seekable(f)
    852         return True
    853     except (io.UnsupportedOperation, AttributeError) as e:
--> 854         raise_err_msg(["seek", "tell"], e)
    855     return False
    856 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in raise_err_msg(patterns, e)
    845                     + " try to load from it instead."
    846                 )
--> 847                 raise type(e)(msg)
    848         raise e
    849 

AttributeError: 'NoneType' object has no attribute 'seek'. You can only torch.load from a file that is seekable. Please pre-load the data into a buffer like io.BytesIO and try to load from it instead.

## === cell 12
efficientnet_model_8 = models.efficientnet_v2_s(weights=None)
num_features_efficientnet = efficientnet_model_8.classifier[1].in_features
efficientnet_model_8.classifier = nn.Sequential(
    nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
)
efficientnet_model_8.load_state_dict(torch.load(eff6_weight, map_location=device))
efficientnet_model_8 = efficientnet_model_8.to(device)
efficientnet_model_8.eval()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _check_seekable(f)
    850     try:
--> 851         f.seek(f.tell())
    852         return True

AttributeError: 'NoneType' object has no attribute 'seek'

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/4092567729.py in <cell line: 0>()
      4     nn.Dropout(p=0.8), nn.Linear(num_features_efficientnet, 5)
      5 )
----> 6 efficientnet_model_8.load_state_dict(torch.load(eff6_weight, map_location=device))
      7 efficientnet_model_8 = efficientnet_model_8.to(device)
      8 efficientnet_model_8.eval()

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    754             return _open_buffer_writer(name_or_buffer)
    755         elif "r" in mode:
--> 756             return _open_buffer_reader(name_or_buffer)
    757         else:
    758             raise RuntimeError(f"Expected 'r' or 'w' in mode but got {mode}")

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, buffer)
    739     def __init__(self, buffer):
    740         super().__init__(buffer)
--> 741         _check_seekable(buffer)
    742 
    743 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _check_seekable(f)
    852         return True
    853     except (io.UnsupportedOperation, AttributeError) as e:
--> 854         raise_err_msg(["seek", "tell"], e)
    855     return False
    856 

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in raise_err_msg(patterns, e)
    845                     + " try to load from it instead."
    846                 )
--> 847                 raise type(e)(msg)
    848         raise e
    849 

AttributeError: 'NoneType' object has no attribute 'seek'. You can only torch.load from a file that is seekable. Please pre-load the data into a buffer like io.BytesIO and try to load from it instead.

## === cell 13
weight_efficientnet = 0.7
weight_resnet = 0.3

w_e1 = weight_efficientnet / 3.0
w_e7 = weight_efficientnet / 3.0
w_e8 = weight_efficientnet / 3.0
w_r = weight_resnet

ensemble_predictions = []
image_names = []

with torch.no_grad():
    for images, img_names in test_loader:
        images = images.to(device)

        p_r = torch.softmax(resnet_model(images), dim=1)
        p_e1 = torch.softmax(efficientnet_model_1(images), dim=1)
        p_e7 = torch.softmax(efficientnet_model_7(images), dim=1)
        p_e8 = torch.softmax(efficientnet_model_8(images), dim=1)

        p_ens = (w_r * p_r) + (w_e1 * p_e1) + (w_e7 * p_e7) + (w_e8 * p_e8)
        preds = torch.argmax(p_ens, dim=1).detach().cpu().numpy().tolist()

        ensemble_predictions.extend(preds)
        image_names.extend(list(img_names))



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/29531311.py in <cell line: 0>()
     18 
     19         # Softmax probabilities for calibrated averaging
---> 20         p_r = torch.softmax(resnet_model(images), dim=1)
     21         p_e1 = torch.softmax(efficientnet_model_1(images), dim=1)
     22         p_e7 = torch.softmax(efficientnet_model_7(images), dim=1)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py in forward(self, x)
    283 
    284     def forward(self, x: Tensor) -> Tensor:
--> 285         return self._forward_impl(x)
    286 
    287 

/usr/local/lib/python3.11/dist-packages/torchvision/models/resnet.py in _forward_impl(self, x)
    266     def _forward_impl(self, x: Tensor) -> Tensor:
    267         # See note [TorchScript super()]
--> 268         x = self.conv1(x)
    269         x = self.bn1(x)
    270         x = self.relu(x)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in forward(self, input)
    552 
    553     def forward(self, input: Tensor) -> Tensor:
--> 554         return self._conv_forward(input, self.weight, self.bias)
    555 
    556 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
    547                 self.groups,
    548             )
--> 549         return F.conv2d(
    550             input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    551         )

RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same

## === cell 14
pred_map = dict(zip(image_names, ensemble_predictions))

submission_df = test_df.copy()
submission_df["label"] = submission_df["image_id"].map(pred_map).astype(int)

if submission_df["label"].isna().any():
    missing = (
        submission_df.loc[submission_df["label"].isna(), "image_id"].head(5).tolist()
    )
    raise RuntimeError(f"Missing predictions for some image_ids, e.g.: {missing}")

submission_df.to_csv("submission.csv", index=False)
print("Submission file saved as 'submission.csv'")
print(submission_df.head())

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
IntCastingNaNError                        Traceback (most recent call last)
/tmp/ipykernel_55/1376858787.py in <cell line: 0>()
      3 
      4 submission_df = test_df.copy()
----> 5 submission_df["label"] = submission_df["image_id"].map(pred_map).astype(int)
      6 
      7 # Safety check: no missing predictions

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in astype(self, dtype, copy, errors)
   6641         else:
   6642             # else, only a single dtype is given
-> 6643             new_data = self._mgr.astype(dtype=dtype, copy=copy, errors=errors)
   6644             res = self._constructor_from_mgr(new_data, axes=new_data.axes)
   6645             return res.__finalize__(self, method="astype")

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in astype(self, dtype, copy, errors)
    428             copy = False
    429 
--> 430         return self.apply(
    431             "astype",
    432             dtype=dtype,

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/managers.py in apply(self, f, align_keys, **kwargs)
    361                 applied = b.apply(f, **kwargs)
    362             else:
--> 363                 applied = getattr(b, f)(**kwargs)
    364             result_blocks = extend_blocks(applied, result_blocks)
    365 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/blocks.py in astype(self, dtype, copy, errors, using_cow, squeeze)
    756             values = values[0, :]  # type: ignore[call-overload]
    757 
--> 758         new_values = astype_array_safe(values, dtype, copy=copy, errors=errors)
    759 
    760         new_values = maybe_coerce_values(new_values)

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array_safe(values, dtype, copy, errors)
    235 
    236     try:
--> 237         new_values = astype_array(values, dtype, copy=copy)
    238     except (ValueError, TypeError):
    239         # e.g. _astype_nansafe can fail on object-dtype of strings

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in astype_array(values, dtype, copy)
    180 
    181     else:
--> 182         values = _astype_nansafe(values, dtype, copy=copy)
    183 
    184     # in pandas we don't store numpy str dtypes, so convert to object

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_nansafe(arr, dtype, copy, skipna)
     99 
    100     elif np.issubdtype(arr.dtype, np.floating) and dtype.kind in "iu":
--> 101         return _astype_float_to_int_nansafe(arr, dtype, copy)
    102 
    103     elif arr.dtype == object:

/usr/local/lib/python3.11/dist-packages/pandas/core/dtypes/astype.py in _astype_float_to_int_nansafe(values, dtype, copy)
    143     """
    144     if not np.isfinite(values).all():
--> 145         raise IntCastingNaNError(
    146             "Cannot convert non-finite values (NA or inf) to integer"
    147         )

IntCastingNaNError: Cannot convert non-finite values (NA or inf) to integer
