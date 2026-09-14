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

0.8705046841946207

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install ../input/efficientnetpytorch-install/dist/efficientnet_pytorch-0.7.0.tar

## === cell 1
!ls -lrt ../input/ramki-cassava-weights/


## === cell 2
!ls ../input/cassava-leaf-disease-classification

## === cell 4
TRAINING = False
WEIGHT_BASE_PATH = '../input/ramki-cassava-weights/'

WEIGHT_FILES = [
    WEIGHT_BASE_PATH+'fold-0-weight-at-epoch-42-acc-0.87126.pth',
    WEIGHT_BASE_PATH+'fold-1-loss-weight-at-epoch-33-loss-0.48675.pth',
    WEIGHT_BASE_PATH+'fold-2-weight-at-epoch-47-acc-0.86212.pth',
    
]

## === cell 5
import numpy as np 
import pandas as pd 
import os
from PIL import Image, ImageFilter
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import *

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from torchvision import models
import time
from tqdm import tqdm
import random
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import classification_report
import torchvision.models as models



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/tensorboard/compat/__init__.py in tf()
     41     try:
---> 42         from tensorboard.compat import notf  # noqa: F401
     43     except ImportError:

ImportError: cannot import name 'notf' from 'tensorboard.compat' (/usr/local/lib/python3.11/dist-packages/tensorboard/compat/__init__.py)

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
from efficientnet_pytorch import EfficientNet
from albumentations.pytorch import ToTensorV2
from albumentations import Rotate 
import albumentations as A


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/796341250.py in <cell line: 0>()
----> 1 from efficientnet_pytorch import EfficientNet
      2 from albumentations.pytorch import ToTensorV2
      3 from albumentations import Rotate
      4 import albumentations as A

ModuleNotFoundError: No module named 'efficientnet_pytorch'

## === cell 7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device

## === cell 8
if torch.cuda.is_available():
    print(torch.cuda.device_count())

## === cell 9
logs='logs/fold-class-'

## === cell 10
SEED = 42
N_FOLDS = 5
N_EPOCHS = 50
BATCH_SIZE = 16
IMG_SIZE = 224
LR = 0.001
NUM_CLASSES = 5


## === cell 11
def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


seed_everything(SEED)

## === cell 12
base_path = '../input/cassava-leaf-disease-classification/'


## === cell 13
train_path =base_path + 'train_images/'
test_path = base_path + 'test_images/'

train_csv = pd.read_csv(base_path + 'train.csv')
sample = pd.read_csv(base_path + 'sample_submission.csv')

## === cell 14
train_csv.head()

## === cell 15

train_csv_disease = train_csv.label.map({0:"Cassava Bacterial Blight (CBB)",
1:"Cassava Brown Streak Disease (CBSD)",
2:"Cassava Green Mottle (CGM)",
3:"Cassava Mosaic Disease (CMD)",
4:"Healthy"})
diseases = train_csv_disease.value_counts()

## === cell 19
class CasavaDataset(Dataset):
    
    def __init__(self, dataframe, transforms=None, test=False):
        self.df = dataframe
        self.transforms = transforms
        self.test = test
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        
        label = self.df.iloc[idx].label
        p = self.df.iloc[idx].image_id
        
        if self.test == False:
            p_path = train_path + p
        else:
            p_path = test_path + p
            
        image =  Image.open(p_path).convert('RGB')
        image = np.array(image)
        
        if self.transforms:
            transformed = self.transforms(image=image)
            image = transformed['image']
    
        
        return image, label

## === cell 20


transforms_train = A.Compose([
        A.RandomResizedCrop(IMG_SIZE, IMG_SIZE),
        A.Transpose(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        A.pytorch.ToTensorV2(),
    ], p=1.0)

transforms_valid = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
    A.pytorch.ToTensorV2(),
])



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2786498527.py in <cell line: 0>()
      8 
      9 
---> 10 transforms_train = A.Compose([
     11         A.RandomResizedCrop(IMG_SIZE, IMG_SIZE),
     12         A.Transpose(p=0.5),

NameError: name 'A' is not defined

## === cell 21
folds = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


## === cell 22
train_csv.shape

## === cell 25
def build_model():
    
    model_name = 'efficientnet-b7'

    if TRAINING:
        model = EfficientNet.from_pretrained(model_name, num_classes=5) 
        print('model created with imagenet weights')
    else:
        model = EfficientNet.from_name(model_name, num_classes=5) 
        print('model loaded from weight file')
    return model.to(device)

## === cell 26
trainset      = CasavaDataset(train_csv, transforms=transforms_train, test=False)
train_loader  = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

testset      = CasavaDataset(sample, transforms=transforms_valid, test=True)
test_loader  = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/564013970.py in <cell line: 0>()
----> 1 trainset      = CasavaDataset(train_csv, transforms=transforms_train, test=False)
      2 train_loader  = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
      3 
      4 testset      = CasavaDataset(sample, transforms=transforms_valid, test=True)
      5 test_loader  = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

NameError: name 'transforms_train' is not defined

## === cell 27
BATCH_SIZE

## === cell 28
class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

## === cell 29
def train_model(model, epoch, dataloader_train, criterion, optimizer ):
    model.train() 
    
    losses = AverageMeter()
    accs = AverageMeter()
    tk = tqdm(dataloader_train, total=len(dataloader_train), position=0, leave=True)
    for idx, (imgs, labels) in enumerate(tk):
        
        imgs_train, labels_train = imgs.to(device), labels.to(device).long()
        output_train = model(imgs_train)

        loss = criterion(output_train, labels_train)
        
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step() 
        predicted_classes = output_train.argmax(1)
        correctly_identified_sum = (predicted_classes==labels_train).sum().item()
        number_of_images = imgs_train.size(0)
                                    
        accs.update(correctly_identified_sum/number_of_images, number_of_images)
        
        losses.update(loss.item(), number_of_images)

        tk.set_postfix(loss=losses.avg,acc=accs.avg)
        
    return losses.avg, accs.avg


def test_model(model, dataloader_valid, criterion):    
    model.eval()
    
    losses = AverageMeter()
    accs = AverageMeter()
    
    with torch.no_grad():
        tk = tqdm(dataloader_valid, total=len(dataloader_valid), position=0, leave=True)
        for idx, (imgs, labels) in enumerate(tk):
            imgs_valid, labels_valid = imgs.to(device), labels.to(device).long()
            output_valid = model(imgs_valid)
            
            loss = criterion(output_valid, labels_valid)

            losses.update(loss.item(), imgs_valid.size(0))
            accs.update((output_valid.argmax(1)==labels_valid).sum().item()/imgs_valid.size(0),imgs_valid.size(0))
            
            tk.set_postfix(loss=losses.avg,acc=accs.avg)

            
    return losses.avg,accs.avg , loss

## === cell 30
X = train_csv.iloc[:,:-1]
y =  train_csv.iloc[:,-1:]

## === cell 32
if TRAINING:

    model = None
    
    for i_fold, (train_idx, valid_idx) in enumerate(folds.split(X,y)):
        print("Fold {}/{}".format(i_fold + 1, N_FOLDS))
        
        writer = SummaryWriter(logs+str(i_fold))
        
        train = train_csv.iloc[train_idx]
        train.reset_index(drop=True, inplace=True) 

        valid = train_csv.iloc[valid_idx]
        valid.reset_index(drop=True, inplace=True)

        

        unique_labels, nSamples = np.unique(train.iloc[:,-1:], return_counts =True)

        normedWeights = [1 - (x / sum(nSamples)) for x in nSamples]
        normedWeights = torch.FloatTensor(normedWeights).to(device)


        dataset_train = CasavaDataset( train, transforms=transforms_train)
        dataset_valid = CasavaDataset( valid, transforms=transforms_valid)

        dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, num_workers=4, shuffle=True)
        dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE, num_workers=4, shuffle=False)
        
        if model is not None:
            del model
            
        torch.cuda.empty_cache() 
        
        model = build_model()

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0)
        criterion = nn.CrossEntropyLoss(weight=normedWeights)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, \
                                                               patience=1, verbose=True, min_lr=1e-5)

        best_acc = 0
        best_loss = 100

        for epoch in range(N_EPOCHS):
            train_loss, train_acc = train_model(model, epoch, dataloader_train, criterion, optimizer)
            val_loss, val_acc, loss = test_model(model, dataloader_valid, criterion)

            writer.add_scalar('training acc', train_acc, epoch+1)
            writer.add_scalar('training loss', train_loss, epoch+1)

            writer.add_scalar('validation loss', val_loss, epoch+1)
            writer.add_scalar('validation Acc', val_acc, epoch+1)

            writer.flush()

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), 'folds-weight/fold-{}-loss-weight-at-epoch-{}-loss-{:.5}.pth'.format(i_fold, epoch, best_loss))


            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), 'folds-weight/fold-{}-weight-at-epoch-{}-acc-{:.5}.pth'.format(i_fold, epoch, best_acc))

            print('current_val_acc:', val_acc, 'best_val_acc:', best_acc)
        writer.close()

    show_metrics(model, dataloader_train, criterion )


## === cell 33

def show_metrics(model, dataloader, criterion ):
    
    
    accs = AverageMeter()
    
    complete_outputs = []
    complete_labels = []
    image_ids = []
    
    probs = []
    tk = tqdm(dataloader, total=len(dataloader), position=0, leave=True)
    for idx, (imgs, labels) in enumerate(tk):
        
        avg_preds = []
        imgs_train, labels_train = imgs.to(device), labels.to(device).long()
        
        for weight_file in WEIGHT_FILES:
            checkpoint = torch.load(weight_file, map_location=device)
            model.load_state_dict(checkpoint) 
            torch.cuda.empty_cache() 
            model.eval()
            with torch.no_grad():
                output_train = model(imgs_train)
                avg_preds.append(output_train.softmax(1).to('cpu').numpy())
        avg_preds = np.mean(avg_preds, axis=0)
        
        probs.append(avg_preds)

        
        
        predicted_classes = avg_preds.argmax(1)
        predicted_classes = torch.tensor(predicted_classes).to(device)
        complete_outputs.append(predicted_classes)
        complete_labels.append(labels_train)
        
        correctly_identified_sum = (predicted_classes==labels_train).sum().item()
        number_of_images = imgs_train.size(0)
                                    
        accs.update(correctly_identified_sum / number_of_images, number_of_images)

        tk.set_postfix(acc=accs.avg)
    print(len(probs))
            

       
                    
                    
   
    return  complete_outputs, complete_labels, image_ids

## === cell 34
model = build_model()

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1335155156.py in <cell line: 0>()
----> 1 model = build_model()

/tmp/ipykernel_11/1248616538.py in build_model()
      7         print('model created with imagenet weights')
      8     else:
----> 9         model = EfficientNet.from_name(model_name, num_classes=5)
     10 #         weights_file = '../input/ramki-cassava-weights/weight.pt'
     11 #         model.load_state_dict(torch.load(weights_file))

NameError: name 'EfficientNet' is not defined

## === cell 35
if TRAINING:
    criterion = nn.CrossEntropyLoss()
    complete_outputs, complete_labels, image_ids = show_metrics(model, train_loader, criterion )
    

## === cell 37
test_pred = []

model.eval()

with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader, position=0, leave=True)):
        images, _ = data
        images = images.to(device)
        avg_preds=[]
        
        for weight_file in WEIGHT_FILES:
            checkpoint = torch.load(weight_file, map_location=device)
            model.load_state_dict(checkpoint) 
            torch.cuda.empty_cache() 
            model.eval()
            with torch.no_grad():
                output = model(images)
                avg_preds.append(output.softmax(1).to('cpu').numpy())
        pred = np.mean(avg_preds, axis=0)
        pred = pred.argmax(1)
        


        test_pred.extend(pred)

sample.label = test_pred
sample.to_csv('submission.csv',index=False)

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1362572241.py in <cell line: 0>()
      1 test_pred = []
      2 
----> 3 model.eval()
      4 
      5 with torch.no_grad():

NameError: name 'model' is not defined

## === cell 38
sample

## === cell 39
!cat submission.csv
