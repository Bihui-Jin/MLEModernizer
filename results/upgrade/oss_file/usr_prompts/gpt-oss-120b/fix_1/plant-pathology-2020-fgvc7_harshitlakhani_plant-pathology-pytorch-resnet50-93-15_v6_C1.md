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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.91943

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import time
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, imshow, axis
from matplotlib.image import imread

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from collections import OrderedDict


import cv2
import torch
from torch import optim
import torchvision
import torch.nn as nn
import torch.utils.data as Data
from torchvision import models,transforms
import time

import albumentations as A
from albumentations.pytorch import ToTensorV2


## === cell 1
root = "/kaggle/input/plant-pathology-2020-fgvc7/"
train = pd.read_csv(os.path.join(root,"train.csv"))
test = pd.read_csv(os.path.join(root,"test.csv"))
submission = pd.read_csv(os.path.join(root,"sample_submission.csv"))
images = os.path.join(root,"images")


## === cell 2
diseases = dict()

for column in ["healthy","multiple_diseases","rust","scab"]:
    counts = pd.DataFrame(train[column].value_counts())
    diseases[column] = counts.iloc[1,0]
    
fig, (ax1, ax2) = plt.subplots(2, 1,figsize=(15,25))
ax1.bar(diseases.keys(),diseases.values(), color=["#6666ff"])
ax1.set_title('Bar Chart', fontsize=18)

ax2.pie(diseases.values(),labels = diseases.keys(), colors=["#6666ff","#4da6ff","#1ac6ff","#c44dff"], autopct='%1.1f%%')
ax2.set_title('Pie Chart', fontsize=18)
ax2.axis('equal') 

plt.show()


## === cell 3
def ShowImages(images, typ):
    fig = figure(figsize=(16,12))
    number_of_images = len(images)
    for i in range(number_of_images):
        a=fig.add_subplot(1,number_of_images,i+1)
        a.set_title(typ, fontsize = 10)
        image = imread(os.path.join(root,"images",images[i]))
        imshow(image)
        axis('off')
        
col=["healthy","multiple_diseases","rust","scab"]
print("Row's are in order of", col)

for column in col:
    images = (train[train[column].apply(lambda x: x == 1)]["image_id"].sample(4).values) + ".jpg"
    ShowImages(images, column)


## === cell 4
def get_path(image):
    return os.path.join(root,"images",image + ".jpg")

train_data = train.copy()
train_data["image_path"] = train_data["image_id"].apply(get_path)
train_labels = train.loc[:, "healthy":"scab"]

test_data = test.copy()
test_data["image_path"] = test_data["image_id"].apply(get_path)
test_paths = test_data["image_path"]

train_paths, valid_paths, train_labels, valid_labels = train_test_split(train_data["image_path"], train_labels, test_size = 0.2, random_state=23, stratify = train_labels)

train_paths.reset_index(drop=True,inplace=True)
train_labels.reset_index(drop=True,inplace=True)
valid_paths.reset_index(drop=True,inplace=True)
valid_labels.reset_index(drop=True,inplace=True)


## === cell 5
mytransform = {
    "train": A.Compose([
    A.RandomResizedCrop(height=256, width=256, p=1.0),
    A.Flip(),
    A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
    ]),
    "validation": A.Compose([
    A.RandomResizedCrop(height=256, width=256, p=1.0),
    A.Normalize(p=1.0),
    ToTensorV2(p=1.0),
    ]),
}


class ImageDataset(Data.Dataset):
    def __init__(self, images_path, labels = None , test=False, transform=None):
        self.images_path = images_path
        self.test = test
        if self.test == False:
            self.labels = labels
            
        self.images_transform = transform

    def __getitem__(self, index):
        if self.test == False:
            labels = torch.tensor(np.argmax(self.labels.iloc[index, :]))
        
        image = cv2.imread(self.images_path[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_transformed = self.images_transform(image=image)
        
        if self.test ==False:
            return image_transformed["image"], labels
        return image_transformed["image"]

    def __len__(self):
        return self.images_path.shape[0]


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
  Field required [type=missing, input_value={'p': 1.0, 'scale': (0.08...: None, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3226048746.py in <cell line: 0>()
      1 mytransform = {
      2     "train": A.Compose([
----> 3     A.RandomResizedCrop(height=256, width=256, p=1.0),
      4     A.Flip(),
      5     A.ShiftScaleRotate(rotate_limit=1.0, p=0.8),

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

## === cell 6
def train_function(model, loader):
    
    running_loss = 0
    preds_for_acc = []
    labels_for_acc = []
    
    progress = tqdm(loader, desc="Training")
    
    for _, (images,labels) in enumerate(progress):
        
        images, labels = images.to(device), labels.to(device)
        model.train()
        
        optimizer.zero_grad()
        predictions = model(images)
        loss = loss_function(predictions, labels)
        loss.backward()
        optimizer.step()


        running_loss += loss.item()*labels.shape[0]
        labels_for_acc = np.concatenate((labels_for_acc,labels.cpu().detach().numpy()), axis=0)
        preds_for_acc = np.concatenate((preds_for_acc,np.argmax(predictions.cpu().detach().numpy(), axis=1)), axis=0)


    accuracy = accuracy_score(labels_for_acc, preds_for_acc)

    return running_loss/TRAIN_SIZE, accuracy


## === cell 7
def valid_function(model, loader):
    running_loss = 0
    preds_for_acc = []
    labels_for_acc = []
    
    progress = tqdm(loader, desc="Validation")
    
    for _, (images,labels) in enumerate(progress):
        
        images, labels = images.to(device), labels.to(device)
        
        
        with torch.no_grad():
            model.eval()
            predictions = model(images)
        loss = loss_function(predictions, labels)

        running_loss += loss.item()*labels.shape[0]
        labels_for_acc = np.concatenate((labels_for_acc,labels.cpu().detach().numpy()), axis=0)
        preds_for_acc = np.concatenate((preds_for_acc,np.argmax(predictions.cpu().detach().numpy(), axis=1)), axis=0)


    accuracy = accuracy_score(labels_for_acc, preds_for_acc)
    conf_matrix = confusion_matrix(labels_for_acc, preds_for_acc)

    return running_loss/VALID_SIZE, accuracy, conf_matrix


## === cell 8
BATCH_SIZE = 64 #4
NUM_EPOCHS = 15 #10
device = "cuda"
TRAIN_SIZE = train_labels.shape[0]
VALID_SIZE = valid_labels.shape[0]
learning_rate = 5e-5


## === cell 9
train_images = ImageDataset(images_path=train_paths, labels=train_labels, transform=mytransform["train"])
train_loader = Data.DataLoader(train_images, shuffle=True, batch_size = BATCH_SIZE)

valid_images = ImageDataset(images_path=valid_paths, labels=valid_labels, transform=mytransform["validation"])
valid_loader = Data.DataLoader(valid_images, shuffle=False, batch_size = BATCH_SIZE)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2640360844.py in <cell line: 0>()
----> 1 train_images = ImageDataset(images_path=train_paths, labels=train_labels, transform=mytransform["train"])
      2 train_loader = Data.DataLoader(train_images, shuffle=True, batch_size = BATCH_SIZE)
      3 
      4 valid_images = ImageDataset(images_path=valid_paths, labels=valid_labels, transform=mytransform["validation"])
      5 valid_loader = Data.DataLoader(valid_images, shuffle=False, batch_size = BATCH_SIZE)

NameError: name 'ImageDataset' is not defined

## === cell 10
model = models.resnet50(pretrained = True)

for param in model.parameters():
    param.require_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(num_ftrs,512,bias=True),
                          nn.ReLU(),
                          nn.Dropout(p=0.3),
                          nn.Linear(512,4, bias = True))

model.to(device)
loss_function = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


## === cell 11
train_loss = []
valid_loss = []
train_acc = []
val_acc = []


## === cell 12
for epoch in range(NUM_EPOCHS):
    
    tl, ta = train_function(model, loader = train_loader)
    vl, va, conf_mat = valid_function(model, loader = valid_loader)
    train_loss.append(tl)
    valid_loss.append(vl)
    train_acc.append(ta)
    val_acc.append(va)
    
    printstr = 'Epoch: '+ str(epoch) + ', Train loss: ' + str(tl) + ', Val loss: ' + str(vl) + ', Train acc: ' + str(ta) + ', Val acc: ' + str(va)
    tqdm.write(printstr)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3649797586.py in <cell line: 0>()
      1 for epoch in range(NUM_EPOCHS):
      2 
----> 3     tl, ta = train_function(model, loader = train_loader)
      4     vl, va, conf_mat = valid_function(model, loader = valid_loader)
      5     train_loss.append(tl)

NameError: name 'train_loader' is not defined

## === cell 13
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure()
plt.ylim(0,1.5)
sns.lineplot(list(range(len(train_loss))), train_loss)
sns.lineplot(list(range(len(valid_loss))), valid_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train','Val'])


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1512846841.py in <cell line: 0>()
      4 plt.figure()
      5 plt.ylim(0,1.5)
----> 6 sns.lineplot(list(range(len(train_loss))), train_loss)
      7 sns.lineplot(list(range(len(valid_loss))), valid_loss)
      8 plt.xlabel('Epoch')

TypeError: lineplot() takes from 0 to 1 positional arguments but 2 were given

## === cell 14
plt.figure()
sns.lineplot(list(range(len(train_acc))), train_acc)
sns.lineplot(list(range(len(val_acc))), val_acc)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Val'])


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2235636413.py in <cell line: 0>()
      1 plt.figure()
----> 2 sns.lineplot(list(range(len(train_acc))), train_acc)
      3 sns.lineplot(list(range(len(val_acc))), val_acc)
      4 plt.xlabel('Epoch')
      5 plt.ylabel('Accuracy')

TypeError: lineplot() takes from 0 to 1 positional arguments but 2 were given

## === cell 15
def sigmoid(X):
    return 1/(1+np.exp(-X))

def test_function(model, loader):
    preds_for_output = np.zeros((1,4))
    
    progress = tqdm(loader, desc="Testing")
    with torch.no_grad():
        for _, images in enumerate(progress):
            images = images.to(device)
            model.eval()
            predictions = model(images)
            preds_for_output = np.concatenate((preds_for_output, sigmoid(predictions.cpu().detach().numpy())), axis=0)
        preds_for_output = np.delete(preds_for_output, 0, 0)
    return preds_for_output


## === cell 16
test_images = ImageDataset(images_path=test_paths, test=True, transform=mytransform["validation"])
test_loader = Data.DataLoader(test_images, shuffle=False, batch_size = BATCH_SIZE)

predictions = test_function(model, test_loader)

submission[["healthy","multiple_diseases","rust","scab"]]  = predictions
submission.to_csv("submission_2.csv", index=False)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3415137225.py in <cell line: 0>()
----> 1 test_images = ImageDataset(images_path=test_paths, test=True, transform=mytransform["validation"])
      2 test_loader = Data.DataLoader(test_images, shuffle=False, batch_size = BATCH_SIZE)
      3 
      4 predictions = test_function(model, test_loader)
      5 

NameError: name 'ImageDataset' is not defined
