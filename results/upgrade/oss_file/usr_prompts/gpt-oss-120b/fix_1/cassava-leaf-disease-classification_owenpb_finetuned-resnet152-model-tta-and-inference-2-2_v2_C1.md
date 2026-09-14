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

3.12

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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

0.8884859474161378

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchinfo import summary

from torchvision.models import resnet152, ResNet152_Weights

import albumentations 
from albumentations.pytorch.transforms import ToTensorV2

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold

import os
import copy
import glob
import json
import random
import pathlib
from PIL import Image
import pickle 


BASE_PATH = '/kaggle/input/cassava-leaf-disease-classification/'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Device: {DEVICE}')

## === cell 5
loaded_model_0 = torch.load('/kaggle/input/best-model-tuned-resnet152-10-epochs-all-folds/best_model_tuned_resnet152_10_epochs_fold_0.pt')
loaded_weights_0 = torch.load('/kaggle/input/best-model-tuned-resnet152-10-epochs-all-folds/best_model_tuned_weights_resnet152_10_epochs_fold_0.pt')

loaded_model_1 = torch.load('/kaggle/input/best-model-tuned-resnet152-10-epochs-all-folds/best_model_tuned_resnet152_10_epochs_fold_1.pt')
loaded_weights_1 = torch.load('/kaggle/input/best-model-tuned-resnet152-10-epochs-all-folds/best_model_tuned_weights_resnet152_10_epochs_fold_1.pt')

loaded_model_2 = torch.load('/kaggle/input/best-model-tuned-resnet152-10-epochs-all-folds/best_model_tuned_resnet152_10_epochs_fold_2.pt')
loaded_weights_2 = torch.load('/kaggle/input/best-model-tuned-resnet152-10-epochs-all-folds/best_model_tuned_weights_resnet152_10_epochs_fold_2.pt')

loaded_model_3 = torch.load('/kaggle/input/best-model-tuned-resnet152-10-epochs-all-folds/best_model_tuned_resnet152_10_epochs_fold_3.pt')
loaded_weights_3 = torch.load('/kaggle/input/best-model-tuned-resnet152-10-epochs-all-folds/best_model_tuned_weights_resnet152_10_epochs_fold_3.pt')

loaded_model_4 = torch.load('/kaggle/input/best-model-tuned-resnet152-10-epochs-all-folds/best_model_tuned_resnet152_10_epochs_fold_4.pt')
loaded_weights_4 = torch.load('/kaggle/input/best-model-tuned-resnet152-10-epochs-all-folds/best_model_tuned_weights_resnet152_10_epochs_fold_4.pt')

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/884559426.py in <cell line: 0>()
----> 1 loaded_model_0 = torch.load('/kaggle/input/best-model-tuned-resnet152-10-epochs-all-folds/best_model_tuned_resnet152_10_epochs_fold_0.pt')
      2 loaded_weights_0 = torch.load('/kaggle/input/best-model-tuned-resnet152-10-epochs-all-folds/best_model_tuned_weights_resnet152_10_epochs_fold_0.pt')
      3 
      4 loaded_model_1 = torch.load('/kaggle/input/best-model-tuned-resnet152-10-epochs-all-folds/best_model_tuned_resnet152_10_epochs_fold_1.pt')
      5 loaded_weights_1 = torch.load('/kaggle/input/best-model-tuned-resnet152-10-epochs-all-folds/best_model_tuned_weights_resnet152_10_epochs_fold_1.pt')

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/best-model-tuned-resnet152-10-epochs-all-folds/best_model_tuned_resnet152_10_epochs_fold_0.pt'

## === cell 6
loaded_model_0.load_state_dict(loaded_weights_0)
loaded_model_0.eval()

loaded_model_1.load_state_dict(loaded_weights_1)
loaded_model_1.eval()

loaded_model_2.load_state_dict(loaded_weights_2)
loaded_model_2.eval()

loaded_model_3.load_state_dict(loaded_weights_3)
loaded_model_3.eval()

loaded_model_4.load_state_dict(loaded_weights_4)
loaded_model_4.eval()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/950300743.py in <cell line: 0>()
----> 1 loaded_model_0.load_state_dict(loaded_weights_0)
      2 loaded_model_0.eval()
      3 
      4 loaded_model_1.load_state_dict(loaded_weights_1)
      5 loaded_model_1.eval()

NameError: name 'loaded_model_0' is not defined

## === cell 7
models = [loaded_model_0, loaded_model_1, loaded_model_2, loaded_model_3, loaded_model_4]

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3825398486.py in <cell line: 0>()
----> 1 models = [loaded_model_0, loaded_model_1, loaded_model_2, loaded_model_3, loaded_model_4]

NameError: name 'loaded_model_0' is not defined

## === cell 9
test_images = glob.glob('../input/cassava-leaf-disease-classification/test_images/*.jpg')
df_test = pd.DataFrame(test_images, columns = ['path'])
df_test['label'] = -1

## === cell 11
width = 512
height = 512

test_transforms = albumentations.Compose([
    albumentations.CenterCrop(width, height, p=1.0),
    albumentations.Resize(width, height),
    albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

tta_transforms = albumentations.Compose([
    
    albumentations.RandomResizedCrop(width, height),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.ShiftScaleRotate(p=0.5),
    albumentations.HueSaturationValue(
                hue_shift_limit=0.2, 
                sat_shift_limit=0.2, 
                val_shift_limit=0.2, 
                p=0.5
            ),
    albumentations.RandomBrightnessContrast(
                brightness_limit=(-0.1, 0.1), 
                contrast_limit=(-0.1, 0.1), 
                p=0.5),
    albumentations.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
    
])

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
  Input should be a valid tuple [type=tuple_type, input_value=512, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=512, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2773643985.py in <cell line: 0>()
     11 tta_transforms = albumentations.Compose([
     12 
---> 13     albumentations.RandomResizedCrop(width, height),
     14     albumentations.HorizontalFlip(p=0.5),
     15     albumentations.Transpose(p=0.5),

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
  Input should be a valid tuple [type=tuple_type, input_value=512, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=512, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

## === cell 12
class TestDataset(Dataset):
    
    def __init__(self, image_ids, labels, transform=None):
        
        self.transform = transform
        self.image_ids = image_ids
        self.labels = labels
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        
        img = Image.open(self.image_ids[index])
        img = np.array(img)
        label = torch.tensor(self.labels[index], dtype=torch.long)
        
        if self.transform:
            return self.transform(image=img)['image'], label 
        else:
            return img, label 

## === cell 13
test_dataset = TestDataset(image_ids=df_test.path, labels=df_test.label, transform=test_transforms)
tta_dataset = TestDataset(image_ids=df_test.path, labels=df_test.label, transform=tta_transforms)

test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
tta_dl = DataLoader(tta_dataset, batch_size=1, shuffle=False)

test_set_size = len(test_dataset)
tta_num = 5

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3070962305.py in <cell line: 0>()
      1 test_dataset = TestDataset(image_ids=df_test.path, labels=df_test.label, transform=test_transforms)
----> 2 tta_dataset = TestDataset(image_ids=df_test.path, labels=df_test.label, transform=tta_transforms)
      3 
      4 test_dl = DataLoader(test_dataset, batch_size=1, shuffle=False)
      5 tta_dl = DataLoader(tta_dataset, batch_size=1, shuffle=False)

NameError: name 'tta_transforms' is not defined

## === cell 14
def get_model_probabilities(model):

    final_probabilities = np.zeros(shape=(test_set_size, num_classes))

    probabilities = np.zeros(shape=(test_set_size, num_classes))

    count = 0

    with torch.no_grad():
        for image, label in test_dl:

            image = image.to(DEVICE)
            label = label.to(DEVICE)

            logits = model(image)

            probs = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()

            probabilities[count] = probs            

            count += 1

    final_probabilities += probabilities


    for i in range(tta_num):

        probabilities = np.zeros(shape=(test_set_size, num_classes))

        count = 0

        with torch.no_grad():
            for image, label in tta_dl:

                image = image.to(DEVICE)
                label = label.to(DEVICE)

                logits = model(image)

                probs = torch.nn.functional.softmax(logits, dim=1).detach().cpu().numpy()

                probabilities[count] = probs            

                count += 1


        final_probabilities += probabilities

    final_probabilities /= (tta_num + 1)
    
    return final_probabilities


## === cell 16
num_models = len(models)
num_classes = 5

ensemble_probabilities = np.zeros(shape=(test_set_size, num_classes))

for model in models:
    
    model_probabilities = get_model_probabilities(model)
    
    ensemble_probabilities += model_probabilities
    
ensemble_probabilities /= num_models

ensemble_predictions = ensemble_probabilities.argmax(axis=1)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1086943848.py in <cell line: 0>()
----> 1 num_models = len(models)
      2 num_classes = 5
      3 
      4 ensemble_probabilities = np.zeros(shape=(test_set_size, num_classes))
      5 

NameError: name 'models' is not defined

## === cell 18
df_test.label = ensemble_predictions

final_test_submission = df_test
final_test_submission['image_id'] = final_test_submission.path.str.split('/').str[-1]
final_test_submission['label'] = ensemble_predictions
final_test_csv = final_test_submission[['image_id', 'label']]

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4044257387.py in <cell line: 0>()
----> 1 df_test.label = ensemble_predictions
      2 
      3 final_test_submission = df_test
      4 final_test_submission['image_id'] = final_test_submission.path.str.split('/').str[-1]
      5 final_test_submission['label'] = ensemble_predictions

NameError: name 'ensemble_predictions' is not defined

## === cell 20
df_test

## === cell 21
final_test_csv.to_csv('submission.csv', index=False)
print('Submission csv file created!')

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2400557957.py in <cell line: 0>()
----> 1 final_test_csv.to_csv('submission.csv', index=False)
      2 print('Submission csv file created!')

NameError: name 'final_test_csv' is not defined
