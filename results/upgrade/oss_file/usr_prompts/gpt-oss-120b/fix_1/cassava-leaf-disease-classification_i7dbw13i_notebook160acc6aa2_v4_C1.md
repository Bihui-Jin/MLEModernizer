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

3.11

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
protobuf==6.33.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
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
transformers==4.53.3

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

0.8396796615291628

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
import random
import json
import gc
import cv2
import pandas as pd
import numpy as np

from tqdm import tqdm
from PIL import Image
from sklearn.metrics import accuracy_score
from functools import partial
from albumentations import (Compose, OneOf, Normalize, Resize, RandomResizedCrop, RandomCrop, CenterCrop, 
                            HorizontalFlip, VerticalFlip, Rotate, ShiftScaleRotate, Transpose)
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform
import tensorflow as tf
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
from transformers import ViTForImageClassification,ViTConfig,AutoModel
from transformers import ViTModel

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "/kaggle/input/cassava-leaf-disease-classification/"
image_path = path+"test_images/"

IMAGE_SIZE = (512,512)
submission_df = pd.DataFrame(columns={"image_id","label"})
submission_df["image_id"] = os.listdir(image_path)
submission_df["label"] = 0

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3631183280.py in <cell line: 0>()
      3 
      4 IMAGE_SIZE = (512,512)
----> 5 submission_df = pd.DataFrame(columns={"image_id","label"})
      6 submission_df["image_id"] = os.listdir(image_path)
      7 submission_df["label"] = 0

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    742             raise ValueError("index cannot be a set")
    743         if isinstance(columns, set):
--> 744             raise ValueError("columns cannot be a set")
    745 
    746         if copy is None:

ValueError: columns cannot be a set

## === cell 2
onlykeras = False
        

used_models_pytorch = {"resnext": [f'../input/models/resnext50_32x4d_fold{fold}_best.pth' for fold in [1]]}

used_models_pytorch = {"vit": f'../input/model-vit/original_save_pretrained'}

used_models_keras = {}

stacked_mean = False

## === cell 4
class CustomResNext(nn.Module):
        def __init__(self, model_name='resnext50_32x4d', pretrained=False):
            super().__init__()
            self.model = timm.create_model(model_name, pretrained=pretrained)
            n_features = self.model.fc.in_features
            self.model.fc = nn.Linear(n_features, 5)

        def forward(self, x):
            x = self.model(x)
            return x

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['image_path_id'].values
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image

if "resnext" in used_models_pytorch:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_transforms():
        return Compose([Resize(512, 512),
                        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ToTensorV2()])

    def inference(model, states, test_loader, device):
        model.to(device)

        probabilities = []
        for i, (images) in enumerate(test_loader):
            images = images.to(device)
            avg_preds = []
            for state in states:
                model.load_state_dict(state['model'])
                model.eval()
                with torch.no_grad():
                    y_preds = model(images)
                avg_preds.append(y_preds.softmax(1).to('cpu').numpy())
            avg_preds = np.mean(avg_preds, axis=0)
            probabilities.append(avg_preds)
        return np.concatenate(probabilities)
    

    predictions_resnext = pd.DataFrame(columns={"image_id"})
    predictions_resnext["image_id"] = submission_df["image_id"].values
    predictions_resnext['image_path_id'] = image_path + predictions_resnext['image_id'].astype(str)

    model = CustomResNext('resnext50_32x4d', pretrained=False)
    states = [torch.load(f) for f in used_models_pytorch["resnext"]]

    test_dataset = TestDataset(predictions_resnext, transform=get_transforms())
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    predictions = inference(model, states, test_loader, device)

    predictions_resnext['resnext'] = [np.squeeze(p) for p in predictions]
    predictions_resnext = predictions_resnext.drop(["image_path_id"], axis=1)

    torch.cuda.empty_cache()
    try:
        del(model)
        del(states)
    except:
        pass
    gc.collect()

## === cell 6
if "vit" in used_models_pytorch:
    
    IMG_SIZE = 224
    BATCH_SIZE = 16
    num_classes = 5

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
        
    class LeafDataset(torch.utils.data.Dataset):
    
        def __init__(self, df, data_path, mode='train', transforms=None):
            super().__init__()
            self.df_data = df.values
            self.data_path = data_path
            self.transforms = transforms 
            self.mode = mode
            self.data_dir = 'train_images' if mode == 'train' else 'test_images'

        def __len__(self):
            return len(self.df_data)

        def __getitem__(self, index):
            img_name = self.df_data[index][0]
            img_path = os.path.join(self.data_path, self.data_dir, img_name)
            img = Image.open(img_path).convert('RGB')

            if self.transforms is not None:
                img = self.transforms(img)

            return img

    transforms_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    def predict(model, test_dataset):
        preds = []
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE)

        for test_images in tqdm(test_dataloader):
            test_images = test_images.to(device)
            model.eval()
            with torch.no_grad():
                output = model(test_images)
            preds.extend(output.logits.data.softmax(1).cpu().data.numpy())

        return preds    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    predictions_vit = pd.DataFrame(columns={"image_id"})
    predictions_vit["image_id"] = submission_df["image_id"].values

    model = ViTForImageClassification.from_pretrained(used_models_pytorch["vit"], num_labels = num_classes)
    model.to(device)
    
    test_dataset = LeafDataset(df=predictions_vit, data_path=path, mode='test', transforms=transforms_val)

    predictions_raw_vit = predict(model, test_dataset)

    predictions_vit['vit'] = [np.squeeze(p) for p in predictions_raw_vit]
    print(predictions_vit)
    
    torch.cuda.empty_cache()
    try:
        del(model)
    except:
        pass

    gc.collect()

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3485827594.py in <cell line: 0>()
     52     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     53 
---> 54     predictions_vit = pd.DataFrame(columns={"image_id"})
     55     predictions_vit["image_id"] = submission_df["image_id"].values
     56 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    742             raise ValueError("index cannot be a set")
    743         if isinstance(columns, set):
--> 744             raise ValueError("columns cannot be a set")
    745 
    746         if copy is None:

ValueError: columns cannot be a set

## === cell 10
    
submission_df["label"] = 0

if "resnext" in used_models_pytorch:
    submission_df = submission_df.merge(predictions_resnext, on="image_id")
    
if "efficientnetb3" in used_models_pytorch:
    submission_df = submission_df.merge(predictions_cutmix, on="image_id")
    
if "vit" in used_models_pytorch:
    submission_df = submission_df.merge(predictions_vit, on="image_id")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4073046658.py in <cell line: 0>()
      5 #     submission_df = submission_df.merge(tmp, right_index=True, left_index=True)
      6 
----> 7 submission_df["label"] = 0
      8 
      9 if "resnext" in used_models_pytorch:

NameError: name 'submission_df' is not defined

## === cell 11
if stacked_mean:
    submission_df["stage_1"] = submission_df.apply(lambda row: [np.mean(e) for e in zip(row["vit2020"], row["resnext"])], axis=1)
    submission_df["label"] = submission_df.apply(lambda row: np.argmax(
        [np.sum(e) for e in zip(row["mobilenet"],row["stage_1"], row["efficientnetb4"])]), axis=1)        
else:
    submission_df["label"] = submission_df.apply(lambda row: np.argmax(
        [np.sum(e) for e in zip(*[row[m] for m in list(used_models_pytorch.keys())+list(used_models_keras.keys())])]), axis=1)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3171132018.py in <cell line: 0>()
      4         [np.sum(e) for e in zip(row["mobilenet"],row["stage_1"], row["efficientnetb4"])]), axis=1)        
      5 else:
----> 6     submission_df["label"] = submission_df.apply(lambda row: np.argmax(
      7         [np.sum(e) for e in zip(*[row[m] for m in list(used_models_pytorch.keys())+list(used_models_keras.keys())])]), axis=1)

NameError: name 'submission_df' is not defined

## === cell 12
submission_df.head(1)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1818596873.py in <cell line: 0>()
----> 1 submission_df.head(1)

NameError: name 'submission_df' is not defined

## === cell 13
submission_df[["image_id","label"]].to_csv("submission.csv", index=False)
!head submission.csv

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2037983141.py in <cell line: 0>()
----> 1 submission_df[["image_id","label"]].to_csv("submission.csv", index=False)
      2 get_ipython().system('head submission.csv')

NameError: name 'submission_df' is not defined
