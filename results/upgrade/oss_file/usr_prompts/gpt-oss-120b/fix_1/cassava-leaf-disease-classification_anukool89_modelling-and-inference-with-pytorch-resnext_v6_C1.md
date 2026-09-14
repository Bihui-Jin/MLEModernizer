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

0.0507706255666364

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
import albumentations as albu
import matplotlib.pyplot as plt
import json
import seaborn as sns
import cv2
import albumentations as albu
import numpy as np

## === cell 1
BASE_DIR="../input/cassava-leaf-disease-classification/"
TRAIN_IMAGES_DIR=os.path.join(BASE_DIR,'train_images')

train_df=pd.read_csv(os.path.join(BASE_DIR,'train.csv'))

## === cell 2
train_df.head()

## === cell 3
print("Count of training images {0}".format(len(os.listdir(TRAIN_IMAGES_DIR))))

## === cell 4
with open(f'{BASE_DIR}/label_num_to_disease_map.json', 'r') as f:
    name_mapping = json.load(f)
    
name_mapping = {int(k): v for k, v in name_mapping.items()}
train_df["class_id"]=train_df["label"].map(name_mapping)

## === cell 5
name_mapping

## === cell 6
len(train_df)

## === cell 7
sns.countplot(y=train_df['label'].map(name_mapping),orient='v')
plt.title("Target Distribution")

## === cell 9
def visualize_images(image_ids,labels):
    plt.figure(figsize=(16,12))
    
    for ind,(image_id,label) in enumerate(zip(image_ids,labels)):
        plt.subplot(3,3,ind+1)
        
        image=cv2.imread(os.path.join(TRAIN_IMAGES_DIR,image_id))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        plt.imshow(image)
        plt.title(f"Class: {label}",fontsize=12)
        
        plt.axis("off")
    plt.show()
    

def plot_augmentation(image_id,transform):
    plt.figure(figsize=(16,4))
    
    img=cv2.imread(os.path.join(TRAIN_IMAGES_DIR,image_id))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.axis("off")
    
    plt.subplot(1,3,2)
    x=transform(image=img)["image"]
    plt.imshow(x)
    plt.axis("off")
    
    plt.subplot(1,3,3)
    x=transform(image=img)["image"]
    plt.imshow(x)
    plt.axis("off")
    
    plt.show()
    
    
def visualize(images, transform):
    """
    Plot images and their transformations
    """
    fig = plt.figure(figsize=(32, 16))
    
    for i, im in enumerate(images):
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        plt.imshow(im)
        
    for i, im in enumerate(images):
        ax = fig.add_subplot(2, 5, i + 6, xticks=[], yticks=[])
        plt.imshow(transform(image=im)['image'])

## === cell 10
tm_df=train_df.sample(9)
image_ids=tm_df["image_id"].values
labels=tm_df["class_id"].values

visualize_images(image_ids,labels)

## === cell 11
train_df[train_df.label==2]

## === cell 13
tmp_df = train_df[train_df["label"] == 0]
print(f"Total train images for class 0: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(9)
image_ids = tmp_df["image_id"].values
labels = tmp_df["class_id"].values

visualize_images(image_ids, labels)

## === cell 15
tmp_df = train_df[train_df["label"] == 1]
print(f"Total train images for class 1: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(9)
image_ids = tmp_df["image_id"].values
labels = tmp_df["class_id"].values

visualize_images(image_ids, labels)

## === cell 17
tmp_df = train_df[train_df["label"] == 2]
print(f"Total train images for class 2: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(9)
image_ids = tmp_df["image_id"].values
labels = tmp_df["class_id"].values

visualize_images(image_ids, labels)

## === cell 19
tmp_df = train_df[train_df["label"] == 3]
print(f"Total train images for class 3: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(9)
image_ids = tmp_df["image_id"].values
labels = tmp_df["class_id"].values

visualize_images(image_ids, labels)

## === cell 21
tmp_df = train_df[train_df["label"] == 4]
print(f"Total train images for class 4: {tmp_df.shape[0]}")

tmp_df = tmp_df.sample(9)
image_ids = tmp_df["image_id"].values
labels = tmp_df["class_id"].values

visualize_images(image_ids, labels)

## === cell 23
transform_all = albu.Compose(
    [
        albu.RandomResizedCrop(256, 256),
        albu.Transpose(p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(p=0.5),
    ]
)

shiftScaleRotate=albu.ShiftScaleRotate(p=0.5)

cutout=albu.Cutout(max_h_size=64, max_w_size=64, p=1)



## --- ERROR in cell 23, traceback:
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
  Input should be a valid tuple [type=tuple_type, input_value=256, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=256, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1836917885.py in <cell line: 0>()
      1 transform_all = albu.Compose(
      2     [
----> 3         albu.RandomResizedCrop(256, 256),
      4         albu.Transpose(p=0.5),
      5         albu.HorizontalFlip(p=0.5),

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
  Input should be a valid tuple [type=tuple_type, input_value=256, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=256, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

## === cell 24
plot_augmentation("100042118.jpg", transform_all)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2425248336.py in <cell line: 0>()
----> 1 plot_augmentation("100042118.jpg", transform_all)

NameError: name 'transform_all' is not defined

## === cell 25
plot_augmentation("1003442061.jpg", cutout)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/94069276.py in <cell line: 0>()
----> 1 plot_augmentation("1003442061.jpg", cutout)

NameError: name 'cutout' is not defined

## === cell 26
plot_augmentation("1003442061.jpg", transform_all)


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4229483338.py in <cell line: 0>()
----> 1 plot_augmentation("1003442061.jpg", transform_all)

NameError: name 'transform_all' is not defined

## === cell 27
train_augs = albu.Compose([
    albu.RandomResizedCrop(height=256, width=256, p=1.0),
    albu.Flip(),
    albu.RandomBrightnessContrast(),
    albu.ShiftScaleRotate(),
    albu.Normalize(),
])

## --- ERROR in cell 27, traceback:
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
/tmp/ipykernel_11/3405897254.py in <cell line: 0>()
      1 train_augs = albu.Compose([
----> 2     albu.RandomResizedCrop(height=256, width=256, p=1.0),
      3     albu.Flip(),
      4     albu.RandomBrightnessContrast(),
      5     albu.ShiftScaleRotate(),

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

## === cell 28
plot_augmentation("1003442061.jpg", train_augs)

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1302047588.py in <cell line: 0>()
----> 1 plot_augmentation("1003442061.jpg", train_augs)

NameError: name 'train_augs' is not defined

## === cell 29
train_aug1=albu.Compose([
            albu.RandomResizedCrop(256,256),
            albu.Transpose(p=0.5),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.5),
            albu.ShiftScaleRotate(p=0.5),
            albu.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            albu.CoarseDropout(p=0.5),
            albu.Cutout(p=0.5),
        ], p=1.)

## --- ERROR in cell 29, traceback:
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
  Input should be a valid tuple [type=tuple_type, input_value=256, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=256, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4283702809.py in <cell line: 0>()
      1 train_aug1=albu.Compose([
----> 2             albu.RandomResizedCrop(256,256),
      3             albu.Transpose(p=0.5),
      4             albu.HorizontalFlip(p=0.5),
      5             albu.VerticalFlip(p=0.5),

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
  Input should be a valid tuple [type=tuple_type, input_value=256, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=256, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

## === cell 30
plot_augmentation("1003442061.jpg", train_aug1)

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3274913511.py in <cell line: 0>()
----> 1 plot_augmentation("1003442061.jpg", train_aug1)

NameError: name 'train_aug1' is not defined

## === cell 33
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold, train_test_split
from albumentations.pytorch import ToTensorV2

import datetime

## === cell 34

class CassavaDataset(Dataset):
    def __init__(self,df:pd.DataFrame,imfolder:str,train:bool = True, transforms=None):
        self.df=df
        self.imfolder=imfolder
        self.train=train
        self.transforms=transforms
        
    def __getitem__(self,index):
        im_path=os.path.join(self.imfolder,self.df.iloc[index]['image_id'])
        x=cv2.imread(im_path,cv2.IMREAD_COLOR)
        x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
        
        if(self.transforms):
            x=self.transforms(image=x)['image']
        
        if(self.train):
            y=self.df.iloc[index]['label']
            return x,y
        else:
            return x
        
    def __len__(self):
        return len(self.df)

## === cell 35
train_augs = albu.Compose([
    albu.RandomResizedCrop(height=256, width=256, p=1.0),
    albu.Flip(),
    albu.RandomBrightnessContrast(),
    albu.ShiftScaleRotate(),
    albu.Normalize(),
    ToTensorV2(),
])

valid_augs = albu.Compose([
    albu.Resize(height=256, width=256, p=1.0),
    albu.Normalize(),
    ToTensorV2(),
])

## --- ERROR in cell 35, traceback:
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
/tmp/ipykernel_11/3176339723.py in <cell line: 0>()
      1 train_augs = albu.Compose([
----> 2     albu.RandomResizedCrop(height=256, width=256, p=1.0),
      3     albu.Flip(),
      4     albu.RandomBrightnessContrast(),
      5     albu.ShiftScaleRotate(),

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

## === cell 36
train, valid = train_test_split(
    train_df, 
    test_size=0.1, 
    random_state=42,
    stratify=train_df.label.values
)


train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)

train_image_paths = [os.path.join(TRAIN_IMAGES_DIR, x) for x in train.image_id.values]

valid_image_paths = [os.path.join(TRAIN_IMAGES_DIR, x) for x in valid.image_id.values]

train_targets = train.label.values

valid_targets = valid.label.values

## === cell 37
train_dataset=CassavaDataset(
    df=train,
    imfolder=TRAIN_IMAGES_DIR,
    train=True,
    transforms=train_augs
)

valid_dataset=CassavaDataset(
    df=valid,
    imfolder=TRAIN_IMAGES_DIR,
    train=True,
    transforms=valid_augs
)

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1718819178.py in <cell line: 0>()
      3     imfolder=TRAIN_IMAGES_DIR,
      4     train=True,
----> 5     transforms=train_augs
      6 )
      7 

NameError: name 'train_augs' is not defined

## === cell 38
def plot_image(img_dict):
    image_tensor = img_dict[0]
    target = img_dict[1]
    print(target)
    plt.figure(figsize=(10, 10))
    image = image_tensor.permute(1, 2, 0) 
    plt.imshow(image)

## === cell 39
plot_image(train_dataset[5])

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2068629312.py in <cell line: 0>()
----> 1 plot_image(train_dataset[5])

NameError: name 'train_dataset' is not defined

## === cell 40
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    num_workers=4,
    shuffle=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=64,
    num_workers=4,
    shuffle=False,
)


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2433549529.py in <cell line: 0>()
      1 train_loader = DataLoader(
----> 2     train_dataset,
      3     batch_size=64,
      4     num_workers=4,
      5     shuffle=True,

NameError: name 'train_dataset' is not defined

## === cell 41
def train_model(n_epochs, model, criterion, optimizer, train_loader,device):
    for epoch in range(1, n_epochs+1):
        loss_train = 0.0
        for batch, labels in train_loader:
            batch,labels=batch.to(device),labels.to(device)
            outputs = model(batch)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()

            loss_train += loss.item()
        if epoch == 1 or epoch % 10 == 0:
            print('{} Epoch {} ,Training Loss {}'.format(
                datetime.datetime.now(), epoch, loss_train/len(train_loader)))

            
def validate(model, train_loader, valid_loader,device):
    for name, loader in [('train', train_loader), ('valid', valid_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs,labels=imgs.to(device),labels.to(device)
                model=model.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())
        print("Accuracy {} {:.2f}".format(name, correct/total))

## === cell 42
model=models.resnet18()#,nn.Linear(512,5))
model.fc=nn.Linear(512,5)

model.fc=nn.Linear(512,5)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
criterion = nn.CrossEntropyLoss()

## === cell 43
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## === cell 44


model.load_state_dict(torch.load('../input/cassava-model-resnet18-1/cassava_model_resnet50_1.pt'))

## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/107475100.py in <cell line: 0>()
     10 # )
     11 
---> 12 model.load_state_dict(torch.load('../input/cassava-model-resnet18-1/cassava_model_resnet50_1.pt'))

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/cassava-model-resnet18-1/cassava_model_resnet50_1.pt'

## === cell 47
validate(model, train_loader, valid_loader,device)

## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3949759546.py in <cell line: 0>()
----> 1 validate(model, train_loader, valid_loader,device)

NameError: name 'train_loader' is not defined

## === cell 49
test_df = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")
image_path = "../input/cassava-leaf-disease-classification/test_images/"
test_targets = test_df.label.values


test_aug = albu.Compose([
            albu.CenterCrop(256, 256, p=1.),
            albu.Resize(256, 256),
            albu.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225], 
                max_pixel_value=255.0, 
                p=1.0),
            ToTensorV2()], p=1.)

test_dataset=CassavaDataset(
    df=test_df,
    imfolder=image_path,
    train=False,
    transforms=test_aug
)

test_loader =  DataLoader(
        test_dataset,
        batch_size=4,
        num_workers=4,
        shuffle=False,
)

## === cell 50
predictions=[]

for imgs in test_loader:

    imgs = imgs.to(device)
    with torch.no_grad():
        model=model.to(device)
        outputs = model(imgs)
        _, predicted = torch.max(outputs, dim=1)
        predicted=predicted.to('cpu')
        predictions.append(predicted)


## === cell 51
test_df['label'] = np.concatenate(predictions)


## === cell 52
test_df.to_csv('submission.csv', index=False)
