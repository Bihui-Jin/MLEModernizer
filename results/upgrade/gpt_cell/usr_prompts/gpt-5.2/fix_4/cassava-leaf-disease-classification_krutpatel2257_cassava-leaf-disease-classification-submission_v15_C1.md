# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.9

# 2. Installed packages

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

# 3. Data file paths

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

# 4. Code solution

## === cell 0
import os

import albumentations
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


## === cell 1
model_path = "../input/rn-tta-calr-ft-ofasf/model(11).pth"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"


## === cell 2
model = models.resnext50_32x4d(pretrained=False)
model.fc = nn.Linear(2048, 5)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if not os.path.exists(model_path):
    candidate_paths = [
        os.path.join(os.path.dirname(sample_sub_path), "model.pth"),
        os.path.join(os.path.dirname(sample_sub_path), "model(11).pth"),
        os.path.join(os.path.dirname(sample_sub_path), "model.pt"),
        os.path.join(os.path.dirname(sample_sub_path), "model(11).pt"),
    ]
    found = None
    for p in candidate_paths:
        if os.path.exists(p):
            found = p
            break
    if found is None:
        for root_dir in (
            "../input",
            "../kaggle/input",
            "/kaggle/input",
            "/kaggle/data",
            "input",
            "data",
            "/kaggle/working",
        ):
            if os.path.isdir(root_dir):
                for root, _, files in os.walk(root_dir):
                    for f in files:
                        lf = f.lower()
                        if lf.endswith(".pth") or lf.endswith(".pt"):
                            found = os.path.join(root, f)
                            break
                    if found is not None:
                        break
            if found is not None:
                break
    if found is not None:
        model_path = found
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(
            f"WARNING: Model weights not found. Proceeding with randomly initialized weights. "
            f"Configured model_path='{model_path}'."
        )
else:
    model.load_state_dict(torch.load(model_path, map_location=device))

model.eval()


## === cell 3
sub_aug = albumentations.Compose([
            albumentations.RandomResizedCrop(256, 256),
            albumentations.Transpose(p=0.5),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.ShiftScaleRotate(p=0.5),
            albumentations.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
            albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1,0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5
            ),
            albumentations.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0
            ),
            albumentations.CoarseDropout(p=0.5),
            albumentations.Cutout(p=0.5)], p=1.)


## --- ERROR in cell 3, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mValidationError[0m                           Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py[0m in [0;36m_validate_parameters[0;34m(schema_cls, full_kwargs, param_names, strict)[0m
[1;32m     66[0m             [0mschema_kwargs[0m[0;34m[[0m[0;34m"strict"[0m[0;34m][0m [0;34m=[0m [0mstrict[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 67[0;31m             [0mconfig[0m [0;34m=[0m [0mschema_cls[0m[0;34m([0m[0;34m**[0m[0mschema_kwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     68[0m             [0mvalidated_kwargs[0m [0;34m=[0m [0mconfig[0m[0;34m.[0m[0mmodel_dump[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/pydantic/main.py[0m in [0;36m__init__[0;34m(self, **data)[0m
[1;32m    249[0m         [0m__tracebackhide__[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 250[0;31m         [0mvalidated_self[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m__pydantic_validator__[0m[0;34m.[0m[0mvalidate_python[0m[0;34m([0m[0mdata[0m[0;34m,[0m [0mself_instance[0m[0;34m=[0m[0mself[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    251[0m         [0;32mif[0m [0mself[0m [0;32mis[0m [0;32mnot[0m [0mvalidated_self[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mValidationError[0m: 2 validation errors for InitSchema
scale
  Input should be a valid tuple [type=tuple_type, input_value=256, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=256, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

The above exception was the direct cause of the following exception:

[0;31mValueError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/875322520.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m sub_aug = albumentations.Compose([
[0;32m----> 2[0;31m             [0malbumentations[0m[0;34m.[0m[0mRandomResizedCrop[0m[0;34m([0m[0;36m256[0m[0;34m,[0m [0;36m256[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      3[0m             [0malbumentations[0m[0;34m.[0m[0mTranspose[0m[0;34m([0m[0mp[0m[0;34m=[0m[0;36m0.5[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      4[0m             [0malbumentations[0m[0;34m.[0m[0mHorizontalFlip[0m[0;34m([0m[0mp[0m[0;34m=[0m[0;36m0.5[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m             [0malbumentations[0m[0;34m.[0m[0mVerticalFlip[0m[0;34m([0m[0mp[0m[0;34m=[0m[0;36m0.5[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py[0m in [0;36mcustom_init[0;34m(self, *args, **kwargs)[0m
[1;32m    103[0m                 [0mfull_kwargs[0m[0;34m,[0m [0mparam_names[0m[0;34m,[0m [0mstrict[0m [0;34m=[0m [0mcls[0m[0;34m.[0m[0m_process_init_parameters[0m[0;34m([0m[0moriginal_init[0m[0;34m,[0m [0margs[0m[0;34m,[0m [0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    104[0m [0;34m[0m[0m
[0;32m--> 105[0;31m                 validated_kwargs = cls._validate_parameters(
[0m[1;32m    106[0m                     [0mdct[0m[0;34m[[0m[0;34m"InitSchema"[0m[0;34m][0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    107[0m                     [0mfull_kwargs[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py[0m in [0;36m_validate_parameters[0;34m(schema_cls, full_kwargs, param_names, strict)[0m
[1;32m     69[0m             [0mvalidated_kwargs[0m[0;34m.[0m[0mpop[0m[0;34m([0m[0;34m"strict"[0m[0;34m,[0m [0;32mNone[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     70[0m         [0;32mexcept[0m [0mValidationError[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 71[0;31m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0mstr[0m[0;34m([0m[0me[0m[0;34m)[0m[0;34m)[0m [0;32mfrom[0m [0me[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     72[0m         [0;32mexcept[0m [0mException[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m     73[0m             [0;32mif[0m [0mstrict[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;31mValueError[0m: 2 validation errors for InitSchema
scale
  Input should be a valid tuple [type=tuple_type, input_value=256, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=256, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

## === cell 4
sample_sub = pd.read_csv(sample_sub_path)

predictions = []
for _, sample_row in sample_sub.iterrows():
    
    image_pred = 0
    for j in range(10):
        image = np.array(Image.open(os.path.join(test_images_path, sample_row.image_id)))
        image = sub_aug(image=image)["image"]
        image = transforms.ToTensor()(np.array(image))
        image = image.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        outputs = model(image.unsqueeze(0))
        
        image_pred += outputs
    image_pred /= 10
    _, pred_label = torch.max(image_pred, 1)
        
    predictions.append([sample_row.image_id, pred_label.item()])

sub_df = pd.DataFrame(predictions,columns=['image_id', 'label'])
sub_df.to_csv('submission.csv', index=False)
print(sub_df.head())
