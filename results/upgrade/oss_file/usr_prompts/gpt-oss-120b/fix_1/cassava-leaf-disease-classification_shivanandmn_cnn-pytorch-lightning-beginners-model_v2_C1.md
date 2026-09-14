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

2.7

# 3. Installed packages

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

0.6272287700211544

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pytorch_lightning as pl
import pandas as pd
import cv2
import os 
from torch import nn
from torch.utils.data import Dataset ,DataLoader
import numpy as np
import torch
from sklearn.model_selection import train_test_split 

IMG_SIZE = 64
PATH = "../input/cassava-leaf-disease-classification/train_images/"
CLASSES = 5

## === cell 2
class CassavaModel(pl.LightningModule):
    def __init__(self):
      super().__init__()
      self.cnv = nn.Conv2d(3,128,5,4)
      self.rel = nn.ReLU()
      self.bn = nn.BatchNorm2d(128)
      self.mxpool = nn.MaxPool2d(4)
      self.flat = nn.Flatten()
      self.fc1 = nn.Linear(1152,64)
      self.fc2 = nn.Linear(64,64)
      self.fc3 = nn.Linear(64,CLASSES)
      self.softmax = nn.Softmax()
      self.accuracy = pl.metrics.Accuracy()

    def forward(self,x):
      out = self.bn(self.rel(self.cnv(x)))
      out = self.flat(self.mxpool(out))
      out = self.rel(self.fc1(out))
      out = self.rel(self.fc2(out))
      out = self.fc3(out)
      return out

    def loss_fn(self,out,target):
      return nn.CrossEntropyLoss()(out.view(-1,CLASSES),target)
    
    def configure_optimizers(self):
      LR = 1e-3
      optimizer = torch.optim.AdamW(self.parameters(),lr=LR)
      return optimizer

    def training_step(self,batch,batch_idx):
      x,y = batch["x"],batch["y"]
      img = x.view(-1,3,IMG_SIZE,IMG_SIZE)
      label = y.view(-1)
      out = self(img)
      loss = self.loss_fn(out,label)
      self.log('train_loss', loss)
      return loss       

    def validation_step(self,batch,batch_idx):
      x,y = batch["x"],batch["y"]
      img = x.view(-1,3,IMG_SIZE,IMG_SIZE)
      label = y.view(-1)
      out = self(img)
      loss = self.loss_fn(out,label)
      out = nn.Softmax(-1)(out) 
      logits = torch.argmax(out,dim=1)
      accu = self.accuracy(logits, label)        
      self.log('valid_loss', loss)
      self.log('train_acc_step', accu)
      return loss, accu


## === cell 4
class CassavaDataset(Dataset):
    def __init__(self,path,image_ids,labels,image_size):
        self.image_ids = image_ids
        self.labels = labels
        self.path = path
        self.image_size = image_size

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self,item):
      image_ids = str(self.image_ids[item])
      labels = self.labels[item]
      img_file = cv2.imread(self.path+image_ids)
      img = cv2.resize(img_file,(self.image_size,self.image_size))
      img = img.astype(np.float64)

      return {
            "x":torch.tensor(img,dtype=torch.float),
            "y":torch.tensor(labels,dtype=torch.long),
        } 

## === cell 6

class CassavaLightDataset(pl.LightningDataModule):
    def __init__(self,batch_size=64):
      super().__init__()
      self.batch_size = batch_size
    
    def setup(self,stage=None):
      dfx = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
      xtrain, xval, ytrain, yval = train_test_split(dfx["image_id"].values,
                                                      dfx.label.values,
                                                      test_size = 0.1)
      self.train_dataset = CassavaDataset(PATH,xtrain,ytrain,IMG_SIZE)
      self.validation_dataset = CassavaDataset(PATH,xval,yval,IMG_SIZE)

    def train_dataloader(self):
      train_loader = DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True)
      return train_loader
    def val_dataloader(self):
      valid_loader = DataLoader(self.validation_dataset,
                            batch_size=self.batch_size,
                            shuffle=False)       
      return valid_loader


## === cell 8
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='valid_loss',
    dirpath='./',
    filename='models-{epoch:02d}-{valid_loss:.2f}',
    save_top_k=3,
    mode='min') 

mod = CassavaModel()
dx = CassavaLightDataset()
trainer = pl.Trainer(gpus=-1,max_epochs=6,callbacks=[checkpoint_callback])
trainer.fit(model=mod,datamodule=dx) 

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/521432978.py in <cell line: 0>()
      6     mode='min') 
      7 
----> 8 mod = CassavaModel()
      9 dx = CassavaLightDataset()
     10 trainer = pl.Trainer(gpus=-1,max_epochs=6,callbacks=[checkpoint_callback])

/tmp/ipykernel_11/1986898943.py in __init__(self)
     12       self.fc3 = nn.Linear(64,CLASSES)
     13       self.softmax = nn.Softmax()
---> 14       self.accuracy = pl.metrics.Accuracy()
     15 
     16     def forward(self,x):

AttributeError: module 'pytorch_lightning' has no attribute 'metrics'

## === cell 10
BEST_MODEL_PATH = "../input/cassava-leaf-disease-clf-model/models-epoch03-valid_loss1.05.ckpt" #checkpoint_callback.best_model_path
pretrained_model = CassavaModel().load_from_checkpoint(BEST_MODEL_PATH)
pretrained_model.eval()
pretrained_model.freeze()

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2926155374.py in <cell line: 0>()
      1 BEST_MODEL_PATH = "../input/cassava-leaf-disease-clf-model/models-epoch03-valid_loss1.05.ckpt" #checkpoint_callback.best_model_path
----> 2 pretrained_model = CassavaModel().load_from_checkpoint(BEST_MODEL_PATH)
      3 pretrained_model.eval()
      4 pretrained_model.freeze()

/tmp/ipykernel_11/1986898943.py in __init__(self)
     12       self.fc3 = nn.Linear(64,CLASSES)
     13       self.softmax = nn.Softmax()
---> 14       self.accuracy = pl.metrics.Accuracy()
     15 
     16     def forward(self,x):

AttributeError: module 'pytorch_lightning' has no attribute 'metrics'

## === cell 12
TEST_FILE_PATH = "../input/cassava-leaf-disease-classification/test_images/"
class CassavaTestDataset(Dataset):
    def __init__(self,path,image_ids,image_size):
        self.image_ids = image_ids
        self.path = path
        self.image_size = image_size

    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self,item):
      image_ids = str(self.image_ids[item])
      img_file = cv2.imread(self.path+image_ids)
      img = cv2.resize(img_file,(self.image_size,self.image_size))
      img = img.astype(np.float64)

      return {
            "x":torch.tensor(img,dtype=torch.float),
        } 
sample = pd.read_csv("../input/cassava-leaf-disease-classification/sample_submission.csv")
test_dataset = CassavaTestDataset(TEST_FILE_PATH,sample.image_id,IMG_SIZE)
test_loader = DataLoader(test_dataset,
                      batch_size=1,
                      shuffle=False)
fin_y = []
for data in test_loader:
  y_hat = pretrained_model(data["x"].view(-1,3,IMG_SIZE,IMG_SIZE))
  y_hat = nn.Softmax(dim=-1)(y_hat)
  y_hat = torch.argmax(y_hat,dim=1)
  fin_y.append(y_hat.cpu().detach().numpy())
sample["label"] = np.array(fin_y).reshape(-1)
sample[["image_id","label"]].to_csv("submission.csv",index=False)
sample.head()

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3876288574.py in <cell line: 0>()
     25 fin_y = []
     26 for data in test_loader:
---> 27   y_hat = pretrained_model(data["x"].view(-1,3,IMG_SIZE,IMG_SIZE))
     28   y_hat = nn.Softmax(dim=-1)(y_hat)
     29   y_hat = torch.argmax(y_hat,dim=1)

NameError: name 'pretrained_model' is not defined
