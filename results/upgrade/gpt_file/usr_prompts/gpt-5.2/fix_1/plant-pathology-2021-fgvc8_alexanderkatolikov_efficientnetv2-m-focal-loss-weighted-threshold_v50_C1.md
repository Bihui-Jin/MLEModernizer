# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.7630470914127425

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np 
import pandas as pd
from PIL import Image
import os
import time
import copy

import sys
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


## === cell 1
import sys

sys.path.append('../input/torchcontrib/contrib-master/')
import torchcontrib
from torchcontrib.optim import SWA
from torch.optim.swa_utils import AveragedModel, SWALR

## === cell 2
BATCH = 6
EPOCHS = 10

WEIGHT_DECAY = 0.000
LR = 0.000001
IM_SIZE = 728

trainnum = 14800
valnum = 3700
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAIN_DIR = '../input/plant-pathology-2021-fgvc8/train_images'
TEST_DIR = '../input/plant-pathology-2021-fgvc8/test_images/'

## === cell 3
train_df = pd.read_csv('../input/plant-pathology-2021-fgvc8/train.csv') 
labeled_train_df = pd.read_csv('../input/train-labeled/train.csv') 
train_df

## === cell 4
train_df['labels'].value_counts()

## === cell 5
NUM_CL = len(train_df['labels'].value_counts())
NUM_CL


## === cell 6
from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(train_df['labels'])
train_df['label_id'] = le.transform(train_df['labels'])
print(train_df)


## === cell 7
my_dict = {}
i=0
for i in range(NUM_CL):
    equality = labeled_train_df.loc[labeled_train_df['label_id'] == i].values[0][1] != train_df.loc[train_df['label_id'] == i].values[0][1]
    if equality == True:
        for l in range(NUM_CL):
            equality = labeled_train_df.loc[labeled_train_df['label_id'] == i].values[0][1] == train_df.loc[train_df['label_id'] == l].values[0][1]
            if equality == True:
                my_dict[l] = i
train_df=train_df.replace({"label_id":my_dict})
print(train_df)

## === cell 8
class_map = dict(sorted(train_df[['label_id', 'labels']].values.tolist()))


## === cell 9
tr_df = train_df[:trainnum]
print(len(tr_df))
X_Train, Y_Train = tr_df['image'].values, tr_df['label_id'].values

## === cell 10

Transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    

## === cell 11
Transformval = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.CenterCrop(IM_SIZE*0.8),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

## === cell 12
class GetData(Dataset):
    def __init__(self, Dir, FNames, Labels, Transform):
        self.dir = Dir
        self.fnames = FNames
        self.transform = Transform
        self.labels = Labels 
    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):       
        x = Image.open(os.path.join(self.dir, self.fnames[index]))
    
        if "train" in self.dir:             
            return self.transform(x), self.labels[index]
        elif "test" in self.dir:            
            return self.transform(x), self.fnames[index]

## === cell 13
trainset = GetData(TRAIN_DIR, X_Train, Y_Train, Transform)
trainloader = DataLoader(trainset, batch_size=BATCH, shuffle=True)

## === cell 14
val_df = train_df[-valnum:]
print(len(val_df))
X_val, Y_val = val_df['image'].values, val_df['label_id'].values
valset = GetData(TRAIN_DIR, X_val, Y_val, Transformval)
valloader = DataLoader(valset, batch_size=BATCH, shuffle=True)


## === cell 15
next(iter(trainloader))[0].shape

## === cell 16
def PREDS(l):
    word = train_df.loc[train_df['label_id'] == l].values[0][1]#finds label's name
    words = word.split(' ')
    
    return words
        

## === cell 17
def metrics(preds,labels,tp,fn,fp):
    a = preds.tolist()
    b = labels.tolist()
    for i in range(len(preds)):
        tp += len(list(set(PREDS(a[i])) & set(PREDS(b[i])))) 
        fn += len(PREDS(b[i])) - len(list(set(PREDS(a[i])) & set(PREDS(b[i])))) 
        fp += len(PREDS(a[i])) - len(list(set(PREDS(a[i])) & set(PREDS(b[i]))))
        
    return tp, fn, fp

## === cell 18
model = torchvision.models.resnext101_32x8d()
model.fc = nn.Linear(2048, NUM_CL, bias=True)
model.load_state_dict(torch.load(os.path.join('../input/resnet-model/ResNext16.pth')))

model.cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay = WEIGHT_DECAY )


## === cell 19
testnum = 3700
test_df = train_df[-testnum:]
X_test, Y_test = test_df['image'].values, test_df['label_id'].values
testset = GetData(TRAIN_DIR, X_test, Y_test, Transformval)
testloader = DataLoader(testset, batch_size=BATCH, shuffle=True)

## === cell 21
X_Test = [name for name in (os.listdir(TEST_DIR))] 

## === cell 22
testset = GetData(TEST_DIR, X_Test, None, Transformval)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

## === cell 23
%%time

s_ls = []

with torch.no_grad():
    model.eval()
    for image, fname in testloader: 
        image = image.to(DEVICE)         
        logits = model(image)
        ps = torch.exp(logits)
        _, top_class = ps.topk(1, dim=1)
        
        for pred in top_class:
            s_ls.append([fname[0], pred.item()])

## === cell 24
pred_df = pd.DataFrame.from_records(s_ls, columns=['image', 'label_id'])
pred_df

## === cell 25
pred_df['labels'] = pred_df['label_id'].map(class_map)
pred_df

## === cell 26
sub = pred_df[['image', 'labels']]
sub.head()

## === cell 27
sub.to_csv("submission.csv", index=False)
