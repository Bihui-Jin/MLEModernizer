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

0.1403747355696585

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_path = '../input/cassava-leaf-disease-classification/train_images/'
test_path = '../input/cassava-leaf-disease-classification/test_images/'

## === cell 2
df_train = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
df_test = pd.read_csv('../input/cassava-leaf-disease-classification/sample_submission.csv')

## === cell 3
df_train.head()

## === cell 4
!pip install timm

## === cell 5
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import torchvision
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from albumentations import *
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
warnings.simplefilter('ignore')

## === cell 6
df_train.shape

## === cell 7
import glob
import os
train_list = glob.glob(os.path.join(train_path, '*')) 

## === cell 8
plt.figure(figsize=(10, 10))
for i in range(3*3):
    plt.subplot(3,3,i+1)
    img = cv2.imread(train_list[i])
    img = img[:,:,::-1]
    plt.imshow(img)
    plt.title(df_train[df_train['image_id'] == train_list[i].split('/')[-1] ]['label'].values[0])
    plt.xlabel(img.shape)
plt.show()

## === cell 9
from sklearn import model_selection
df_train['kfold'] = -1
df_train = df_train.sample(frac=1).reset_index(drop=True)
kf = model_selection.StratifiedKFold(n_splits=5)
for f, (t_, v_) in enumerate(kf.split(X=df_train, y=df_train.label.values)):
  df_train.loc[v_, 'kfold'] = f
print(df_train['kfold'].value_counts())
for fold in range(5):
  train_fold = df_train[df_train['kfold'] != fold]
  valid_fold = df_train[df_train['kfold'] == fold]
  train_fold.to_csv(f'fold_{fold}_train.csv',index=None)
  valid_fold.to_csv(f'fold_{fold}_valid.csv',index=None)

## === cell 10
image_size = 384
epochs = 10
batch_size = 16
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

## === cell 11
class Augments:
  """Contains Train, Validation and Testing Augments"""
  train_augments = Compose([
                            Resize(image_size,image_size),
                            RandomResizedCrop(image_size, image_size),
                            Transpose(p=0.5),
                            HorizontalFlip(p=0.5),
                            VerticalFlip(p=0.5),
                            ShiftScaleRotate(p=0.5),
                            Rotate(limit=45, p=0.5),
                            HueSaturationValue(hue_shift_limit=0.2,sat_shift_limit=0.2,val_shift_limit=0.5),
                            RandomBrightnessContrast(brightness_limit=(-0.1,0.1),contrast_limit=(-0.1,0.1),p=0.5),
                            Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225],max_pixel_value=225.0,p=1.0),
                            CoarseDropout(p=0.5),
                            Cutout(p=0.5),
                            ToTensorV2(p=1.0)
  ],p=1.0
  )

  valid_augments = Compose([
                            CenterCrop(image_size,image_size,p=1.),
                            Resize(image_size,image_size),
                            Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224, 0.225], max_pixel_value=225.0,p=1.0),
                            ToTensorV2(p=1.0)
  ],p=1.0)

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
  Input should be a valid tuple [type=tuple_type, input_value=384, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=384, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3474067259.py in <cell line: 0>()
----> 1 class Augments:
      2   """Contains Train, Validation and Testing Augments"""
      3   train_augments = Compose([
      4                             Resize(image_size,image_size),
      5                             RandomResizedCrop(image_size, image_size),

/tmp/ipykernel_11/3474067259.py in Augments()
      3   train_augments = Compose([
      4                             Resize(image_size,image_size),
----> 5                             RandomResizedCrop(image_size, image_size),
      6                             Transpose(p=0.5),
      7                             HorizontalFlip(p=0.5),

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
  Input should be a valid tuple [type=tuple_type, input_value=384, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type
size
  Input should be a valid tuple [type=tuple_type, input_value=384, input_type=int]
    For further information visit https://errors.pydantic.dev/2.12/v/tuple_type

## === cell 12
class EfficientNetModel(nn.Module):
  def __init__(self, num_classes = 5, model_name = 'efficientnet_b7',pretrained = True):
    super(EfficientNetModel, self).__init__()
    self.model = timm.create_model(model_name, pretrained=pretrained)
    self.model.fc = nn.Linear(self.model.classifier.in_features, num_classes)

  def forward(self, x):
    x = self.model(x)
    return x

class VITModel(nn.Module):
  def __init__(self, num_classes = 5, model_name = 'vit_base_patch16_384', pretrained = True):
    super(VITModel, self).__init__()
    self.model = timm.create_model(model_name, pretrained= pretrained)
    self.model.fc = nn.Linear(self.model.head.in_features, num_classes)
  
  def forward(self, x):
    x = self.model(x)
    return x

## === cell 13
class CustomDataset(Dataset):
  def __init__(self, df, num_classes = 5, is_train = True, augments=None, 
               image_size = image_size,folder_path = train_path):
    super().__init__()
    self.df = df.sample(frac = 1).reset_index(drop=True)
    self.num_classes = num_classes
    self.is_train = is_train
    self.augments = augments
    self.image_size = image_size
    self.folder_path = folder_path

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    img_path = os.path.join(self.folder_path,self.df['image_id'][idx])
    img = cv2.imread(img_path)
    img = img[ :, :, ::-1]
    if self.augments:
        img = self.augments(image=img)['image']
    if self.is_train:
        label = self.df['label'][idx]
        return img, label
    return img

## === cell 14
def train_one_cycle(model, dataloader, loss_fn, optim):
    model.train() #train mode
    train_prog = tqdm(dataloader, total = len(dataloader))
    
    all_labels = []
    all_preds = []
    
    run_loss = 0.0
    scaler = GradScaler()
    
    for inputs, labels in train_prog:
        inputs = inputs.to(device).float()
        labels = labels.to(device).long()
        with autocast():
            outputs = model(inputs)
            train_loss = loss_fn(outputs, labels)
            scaler.scale(train_loss).backward()
            
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()
            
            run_loss += train_loss
            
            preds = torch.argmax(outputs, 1).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            
            all_preds += [preds]
            all_labels += [labels]
        train_pbar = f'loss: {train_loss.item():.3f}'
        train_prog.set_description(desc = train_pbar)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    acc = (all_preds == all_labels).mean()
    print(f'Training Accuracy: {acc:.3f}')
    floss = run_loss / len(dataloader)
    del all_preds, all_labels, run_loss
    return (acc, floss)
            
            
def valid_one_cycle(model, dataloader, loss_fn):
    model.eval() #eval model
    valid_prog = tqdm(dataloader, total = len(dataloader))
    with torch.no_grad():
        all_labels = []
        all_preds = []

        run_loss = 0.0
        scaler = GradScaler()

        for inputs, labels in valid_prog:
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            outputs = model(inputs)
            valid_loss = loss_fn(outputs, labels)
            run_loss += valid_loss.item()

            preds = torch.argmax(outputs, 1).detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            all_preds += [preds]
            all_labels += [labels]
            valid_pbar = f'loss: {valid_loss.item():.3f}'
            valid_prog.set_description(desc = valid_pbar)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        acc = (all_preds == all_labels).mean()
        print(f'Valid Accuracy: {acc:.3f}')
        floss = run_loss / len(dataloader)
        del all_preds, all_labels, run_loss
    return (acc, floss, model)

## === cell 15
def plot_results(train_acc, valid_acc, train_loss, valid_loss, nb_epochs):
    epochs = [i for i in range(nb_epochs)]
    
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(20, 10)
    
    ax[0].plot(epochs, train_acc, 'go-', label='Training Accuracy')
    ax[0].plot(epochs, valid_acc, 'ro-', label='Validation Accuracy')
    ax[0].set_title('Training & Validation Accuracy')
    ax[0].legend()
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    
    ax[1].plot(epochs, train_loss, 'go-', label='Training Loss')
    ax[1].plot(epochs, valid_loss, 'ro-', label='Validation Loss')
    ax[1].set_title('Training & Validation Loss')
    ax[1].legend()
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss')
    
    plt.show()

## === cell 16
def run(fold):
  train_fold = pd.read_csv(f"./fold_{fold}_train.csv")
  valid_fold = pd.read_csv(f"./fold_{fold}_valid.csv")

  train_set = CustomDataset(df=train_fold, augments=Augments.train_augments)
  valid_set = CustomDataset(df=valid_fold, augments=Augments.valid_augments)

  train = DataLoader(train_set,batch_size=batch_size,shuffle=True,pin_memory=False,drop_last=False,num_workers=8)

  valid = DataLoader(valid_set,batch_size=batch_size,shuffle=False,pin_memory=False,num_workers=8)

  model = VITModel(num_classes=5, model_name='vit_base_patch16_384').to(device)
  optim = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-6)
  loss_fn = nn.CrossEntropyLoss().to(device)

  train_accs = []
  valid_accs = []
  train_losses = []
  valid_losses = []
  best_acc = 0.0
    

  for epoch in range(epochs):
      print(f"{'-'*20} EPOCH: {epoch}/{epochs} {'-'*20}")

      current_train_acc, current_train_loss = train_one_cycle(model = model, dataloader = train, loss_fn=loss_fn, optim =optim)
      train_accs.append(current_train_acc)
      train_losses.append(current_train_loss)

      current_val_acc, current_val_loss, op_model = valid_one_cycle(model = model, dataloader = valid,loss_fn = loss_fn)
      valid_accs.append(current_val_acc)
      valid_losses.append(current_val_loss)

      torch.cuda.empty_cache()
      
      if best_acc < current_val_acc:
            print(f"Saving Model for this epoch...")
            torch.save(op_model.state_dict(), f"vit_base_p16_384_fold_{fold}_model.pth")
      
  del train_set, valid_set, train, valid #, scaler
  torch.cuda.empty_cache()
  print(f'Best Accuracy of {fold} fold: {best_acc:.3f}')  


## === cell 17
for fold in range(1):
  run(fold)

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2705882548.py in <cell line: 0>()
      1 for fold in range(1):
----> 2   run(fold)

/tmp/ipykernel_11/3286194291.py in run(fold)
      3   valid_fold = pd.read_csv(f"./fold_{fold}_valid.csv")
      4 
----> 5   train_set = CustomDataset(df=train_fold, augments=Augments.train_augments)
      6   valid_set = CustomDataset(df=valid_fold, augments=Augments.valid_augments)
      7 

NameError: name 'Augments' is not defined

## === cell 18
model = VITModel(num_classes=5, model_name='vit_base_patch16_384').to(device)

## === cell 19
model.load_state_dict(torch.load('../input/model-kfold0/vit_base_p16_384_fold_0_model.pth'))

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1827437687.py in <cell line: 0>()
----> 1 model.load_state_dict(torch.load('../input/model-kfold0/vit_base_p16_384_fold_0_model.pth'))

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

FileNotFoundError: [Errno 2] No such file or directory: '../input/model-kfold0/vit_base_p16_384_fold_0_model.pth'

## === cell 20
def show_predictions(model, data_loader):
    preds = []
    model = model.eval()
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(data_loader):
          inputs = inputs.to(device)
          labels = labels.to(device)

          outputs = model(inputs)
          _, pred = torch.max(outputs, 1)
          pred = pred.cpu().numpy()
          preds.append(pred[0])
    return preds

## === cell 21
test_set = CustomDataset(df=df_test, augments=Augments.valid_augments,folder_path = test_path)
test = DataLoader(test_set,batch_size=batch_size,shuffle=False,pin_memory=False,num_workers=8)

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3827321883.py in <cell line: 0>()
----> 1 test_set = CustomDataset(df=df_test, augments=Augments.valid_augments,folder_path = test_path)
      2 test = DataLoader(test_set,batch_size=batch_size,shuffle=False,pin_memory=False,num_workers=8)

NameError: name 'Augments' is not defined

## === cell 22
predictions = show_predictions(model, test)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3174848406.py in <cell line: 0>()
----> 1 predictions = show_predictions(model, test)

NameError: name 'test' is not defined

## === cell 23
predictions

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/943844562.py in <cell line: 0>()
----> 1 predictions

NameError: name 'predictions' is not defined

## === cell 24
df_test.head()

## === cell 25
df_test['label'] = predictions

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1620297965.py in <cell line: 0>()
----> 1 df_test['label'] = predictions

NameError: name 'predictions' is not defined

## === cell 26
df_test.to_csv("submission.csv",index=None)

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same length as the answers.
