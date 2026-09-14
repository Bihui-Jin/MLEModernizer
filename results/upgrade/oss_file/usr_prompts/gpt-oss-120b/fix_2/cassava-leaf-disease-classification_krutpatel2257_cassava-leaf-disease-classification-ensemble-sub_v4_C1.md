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
scipy==1.15.3
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

0.8952855847688124

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

import albumentations
import numpy as np
import pandas as pd
from PIL import Image
from scipy.special import softmax

import torch
import torch.nn as nn
from torchvision import models, transforms




## === cell 1
resnet_model_path = "../input/rn-wc-tta-calr-clahe-cutmix/model(24).pth"
effnet_model_path = "../input/en-b4-tta-calr-clahe-v3/eff_epoch_11.pth"

sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
test_images_path = "../input/cassava-leaf-disease-classification/test_images"




## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet_model = models.resnext50_32x4d(pretrained=True)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 5)
resnet_model = resnet_model.to(device)
if os.path.exists(resnet_model_path):
    resnet_model.load_state_dict(torch.load(resnet_model_path, map_location=device))
resnet_model.eval()

effnet_model = models.efficientnet_b4(pretrained=True)
effnet_model.classifier[1] = nn.Linear(effnet_model.classifier[1].in_features, 5)
effnet_model = effnet_model.to(device)
if os.path.exists(effnet_model_path):
    effnet_model.load_state_dict(torch.load(effnet_model_path, map_location=device))
effnet_model.eval()




## === cell 3
sub_aug = albumentations.Compose(
    [
        albumentations.RandomResizedCrop(height=512, width=512, scale=(0.5, 1.0)),
        albumentations.Transpose(p=0.5),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.ShiftScaleRotate(p=0.8),
        albumentations.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0,
        ),
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
/tmp/ipykernel_55/879022221.py in <cell line: 0>()
      2 sub_aug = albumentations.Compose(
      3     [
----> 4         albumentations.RandomResizedCrop(height=512, width=512, scale=(0.5, 1.0)),
      5         albumentations.Transpose(p=0.5),
      6         albumentations.HorizontalFlip(p=0.5),

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
sample_sub = pd.read_csv(sample_sub_path)
tta_count = 10

predictions = []
to_tensor = transforms.ToTensor()

with torch.no_grad():
    for _, sample_row in sample_sub.iterrows():
        image_pred = 0.0
        img_path = os.path.join(test_images_path, sample_row.image_id)
        raw_image = np.array(Image.open(img_path).convert("RGB"))
        for _ in range(tta_count):
            aug_image = sub_aug(image=raw_image)["image"]
            tensor_image = to_tensor(aug_image).to(device)

            resnet_out = resnet_model(tensor_image.unsqueeze(0))
            image_pred += resnet_out.cpu().numpy() * 0.6

            effnet_out = effnet_model(tensor_image.unsqueeze(0))
            image_pred += effnet_out.cpu().numpy() * 0.4

        image_pred /= tta_count
        predictions.append(image_pred.squeeze())

predictions = softmax(np.stack(predictions), axis=1).argmax(axis=1)

sub_df = pd.DataFrame({"image_id": sample_sub.image_id, "label": predictions})
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1388519859.py in <cell line: 0>()
     14         for _ in range(tta_count):
     15             # Apply augmentation
---> 16             aug_image = sub_aug(image=raw_image)["image"]
     17             # Convert to tensor and move to device
     18             tensor_image = to_tensor(aug_image).to(device)

NameError: name 'sub_aug' is not defined
