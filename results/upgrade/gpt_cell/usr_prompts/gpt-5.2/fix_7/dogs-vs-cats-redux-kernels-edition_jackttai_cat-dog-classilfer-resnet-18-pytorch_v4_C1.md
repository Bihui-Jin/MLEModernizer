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
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 4. Code solution

## === cell 0
!unzip ../input/dogs-vs-cats-redux-kernels-edition/train.zip
!unzip ../input/dogs-vs-cats-redux-kernels-edition/test.zip


## === cell 1

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

import torchmetrics as metrics

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

plt.style.use("ggplot")
import pandas as pd
import random
import time
import os
import zipfile
from PIL import Image
import numpy as np
import zipfile


## === cell 2
test_list = [str(i) + ".jpg" for i in range(1, 12501)]


def _first_existing_dir(candidates):
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "Could not find train/test directories. Tried: " + ", ".join(candidates)
    )


train_dir = _first_existing_dir(
    [
        "./train",
        "./dogs-vs-cats-redux-kernels-edition/train",
        "../input/dogs-vs-cats-redux-kernels-edition/train",
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train",
    ]
)

test_dir = _first_existing_dir(
    [
        "./test",
        "./dogs-vs-cats-redux-kernels-edition/test",
        "../input/dogs-vs-cats-redux-kernels-edition/test",
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test",
    ]
)

train_df = pd.DataFrame(os.listdir(train_dir), columns=["filename"])
test_df = pd.DataFrame(test_list, columns=["filename"])

train_df["label"] = train_df.filename.str[:3]
train_df["label"] = train_df["label"].map({"dog": 1, "cat": 0})

train_df["filename"] = train_df["filename"].apply(lambda x: os.path.join(train_dir, x))
test_df["filename"] = test_df["filename"].apply(lambda x: os.path.join(test_dir, x))

"""
Use only 2000 images for testing first, if the model is running well without any error, then change back to full dataset
"""
TRAIN_SAMPLES = train_df.shape[0]

train_df = train_df.sample(TRAIN_SAMPLES)

train_df, val_df, _, _ = train_test_split(
    train_df, train_df, test_size=0.04, random_state=42
)

train_df.head()


## === cell 3
test_df.tail()


## === cell 4
print('Training set images: {}, Validation set image: {}'.format(train_df.shape[0], val_df.shape[0]))


## === cell 5
def show_6_photos(dataframe):
    n = min(6, len(dataframe))
    if n == 0:
        return

    sample_df = dataframe.sample(n)
    paths = sample_df.filename.tolist()

    valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    valid_paths = [
        p for p in paths if os.path.isfile(p) and str(p).lower().endswith(valid_ext)
    ]

    if len(valid_paths) == 0:
        return

    for path in valid_paths:
        img = plt.imread(path)
        plt.subplots(figsize=(3, 3))
        plt.imshow(img)
        plt.show()


show_6_photos(train_df)


## === cell 6
data_transforms = {
    'train':transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val':transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}


## === cell 7
class image_set(Dataset):
  def __init__(self,dataframe,transform=None,test=False):
    self.dataframe = dataframe
    self.transform = transform
    self.test = test 

  def __getitem__(self,index):
    x = self.dataframe.iloc[index,0]
    x = Image.open(x)
    if self.transform:
      x = self.transform(x)
    if self.test==True:
      return x
    else:
      y = self.dataframe.iloc[index,1]
      return x,np.array([y])

  def __len__(self):
    return self.dataframe.shape[0]


## === cell 8

train_set = image_set(train_df,transform=data_transforms['train'])
val_set = image_set(val_df,transform=data_transforms['val'])
test_set = image_set(test_df,transform=data_transforms['val'],test=True)

BATCH_SIZE=32

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,num_workers=4)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True,num_workers=4)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,num_workers=4)


## === cell 9
device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')


## === cell 10
def train_model(model, cost_function,optimizer,num_epochs=5):

  train_losses = []
  val_losses = []
  train_acc = []
  val_acc=[]

  train_acc_object = metrics.Accuracy(compute_on_step=False)
  val_acc_object = metrics.Accuracy(compute_on_step=False)

  for epoch in range(num_epochs):
    """
    On Epoch start
    """
    print('-'*20)
    print('Start training {}/{}'.format(epoch+1,num_epochs))
    print('-'*20)
    train_acc_object.reset()
    val_acc_object.reset()
    
    """
    Start Training model
    """
    model.train()
    epoch_losses = []
    for x,y in train_loader:
      optimizer.zero_grad()
    
      x,y = x.to(device),y.to(device)
      outputs = model(x)
        
      loss = cost_function(outputs,y.type_as(outputs))
      epoch_losses.append(loss.item())

      loss.backward()
      optimizer.step()
    
      train_acc_object(outputs.cpu(), y.type_as(outputs).cpu())

    """
    Counting Validation loss
    """
    model.eval()
    epoch_val_losses = []
    for x,y in val_loader:
      x,y  = x.to(device),y.to(device)
      outputs = model(x)
      loss = cost_function(outputs,y.type_as(outputs))
      epoch_val_losses.append(loss.item())
      val_acc_object(outputs.cpu(), y.type_as(outputs).cpu())
        
    """
    On epoch ends
    """
    train_losses.append(np.mean(epoch_losses))
    val_losses.append(np.mean(epoch_val_losses))
    
    epoch_t_acc = train_acc_object.compute()
    epoch_v_acc = val_acc_object.compute()
    train_acc.append(epoch_t_acc)
    val_acc.append(epoch_v_acc)
    
    print('loss:{:.3f}, acc:{:.3f}, val_loss:{:.3f}, val_acc:{:.3f}'.format(np.mean(epoch_losses),
                                                                             epoch_t_acc,
                                                                             np.mean(epoch_val_losses),
                                                                             epoch_v_acc))

  print('Finish training.')
  return train_losses, val_losses, train_acc, val_acc


## === cell 11
class net(nn.Module):
    def __init__(self,resnet):
        super(net,self).__init__()
        self.resnet = resnet
        self.linear1 = nn.Linear(1000,512)
        self.linear2 = nn.Linear(512,1)
    
    def forward(self,x):
        x = F.relu(self.resnet(x))
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = torch.sigmoid(x)
        return x


## === cell 12
res = models.resnet18(pretrained=True)
for param in res.parameters():
    param.requires_grad=False

model_final = net(resnet=res)

model_final= model_final.to(device)

cost_function = nn.BCELoss()  

optimizer_ft = optim.Adam([param for param in model_final.parameters() if param.requires_grad],lr=0.009)


EPOCHS=10


## === cell 13
def train_model(model, cost_function, optimizer, num_epochs=5):

    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []

    train_acc_object = metrics.Accuracy(task="binary").cpu()
    val_acc_object = metrics.Accuracy(task="binary").cpu()

    for epoch in range(num_epochs):
        """
        On Epoch start
        """
        print("-" * 20)
        print("Start training {}/{}".format(epoch + 1, num_epochs))
        print("-" * 20)
        train_acc_object.reset()
        val_acc_object.reset()

        """
    Start Training model
    """
        model.train()
        epoch_losses = []
        for x, y in train_loader:
            optimizer.zero_grad()

            x, y = x.to(device), y.to(device)
            outputs = model(x)

            loss = cost_function(outputs, y.type_as(outputs))
            epoch_losses.append(loss.item())

            loss.backward()
            optimizer.step()

            train_acc_object(
                outputs.detach().cpu().squeeze(1),
                y.detach().type_as(outputs).cpu().squeeze(1),
            )

        """
    Counting Validation loss
    """
        model.eval()
        epoch_val_losses = []
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = cost_function(outputs, y.type_as(outputs))
            epoch_val_losses.append(loss.item())
            val_acc_object(
                outputs.detach().cpu().squeeze(1),
                y.detach().type_as(outputs).cpu().squeeze(1),
            )

        """
    On epoch ends
    """
        train_losses.append(np.mean(epoch_losses))
        val_losses.append(np.mean(epoch_val_losses))

        epoch_t_acc = train_acc_object.compute()
        epoch_v_acc = val_acc_object.compute()
        train_acc.append(epoch_t_acc)
        val_acc.append(epoch_v_acc)

        print(
            "loss:{:.3f}, acc:{:.3f}, val_loss:{:.3f}, val_acc:{:.3f}".format(
                np.mean(epoch_losses),
                epoch_t_acc,
                np.mean(epoch_val_losses),
                epoch_v_acc,
            )
        )

    print("Finish training.")
    return train_losses, val_losses, train_acc, val_acc


_valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
for _ds in (train_set, val_set):
    if hasattr(_ds, "dataframe") and "filename" in _ds.dataframe.columns:
        _mask = _ds.dataframe["filename"].apply(
            lambda p: os.path.isfile(p) and str(p).lower().endswith(_valid_ext)
        )
        _ds.dataframe = _ds.dataframe.loc[_mask].reset_index(drop=True)

train_losses, val_losses, train_acc, val_acc = train_model(
    model=model_final,
    cost_function=cost_function,
    optimizer=optimizer_ft,
    num_epochs=EPOCHS,
)


## --- ERROR in cell 13, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mZeroDivisionError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/404843625.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     90[0m         [0m_ds[0m[0;34m.[0m[0mdataframe[0m [0;34m=[0m [0m_ds[0m[0;34m.[0m[0mdataframe[0m[0;34m.[0m[0mloc[0m[0;34m[[0m[0m_mask[0m[0;34m][0m[0;34m.[0m[0mreset_index[0m[0;34m([0m[0mdrop[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     91[0m [0;34m[0m[0m
[0;32m---> 92[0;31m train_losses, val_losses, train_acc, val_acc = train_model(
[0m[1;32m     93[0m     [0mmodel[0m[0;34m=[0m[0mmodel_final[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m     94[0m     [0mcost_function[0m[0;34m=[0m[0mcost_function[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/404843625.py[0m in [0;36mtrain_model[0;34m(model, cost_function, optimizer, num_epochs)[0m
[1;32m     24[0m         [0mmodel[0m[0;34m.[0m[0mtrain[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     25[0m         [0mepoch_losses[0m [0;34m=[0m [0;34m[[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 26[0;31m         [0;32mfor[0m [0mx[0m[0;34m,[0m [0my[0m [0;32min[0m [0mtrain_loader[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     27[0m             [0moptimizer[0m[0;34m.[0m[0mzero_grad[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     28[0m [0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m__iter__[0;34m(self)[0m
[1;32m    489[0m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_iterator[0m[0;34m[0m[0;34m[0m[0m
[1;32m    490[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 491[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_get_iterator[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    492[0m [0;34m[0m[0m
[1;32m    493[0m     [0;34m@[0m[0mproperty[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m_get_iterator[0;34m(self)[0m
[1;32m    420[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    421[0m             [0mself[0m[0;34m.[0m[0mcheck_worker_number_rationality[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 422[0;31m             [0;32mreturn[0m [0m_MultiProcessingDataLoaderIter[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    423[0m [0;34m[0m[0m
[1;32m    424[0m     [0;34m@[0m[0mproperty[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m__init__[0;34m(self, loader)[0m
[1;32m   1197[0m         [0m_utils[0m[0;34m.[0m[0msignal_handling[0m[0;34m.[0m[0m_set_SIGCHLD_handler[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1198[0m         [0mself[0m[0;34m.[0m[0m_worker_pids_set[0m [0;34m=[0m [0;32mTrue[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1199[0;31m         [0mself[0m[0;34m.[0m[0m_reset[0m[0;34m([0m[0mloader[0m[0;34m,[0m [0mfirst_iter[0m[0;34m=[0m[0;32mTrue[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1200[0m [0;34m[0m[0m
[1;32m   1201[0m     [0;32mdef[0m [0m_reset[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mloader[0m[0;34m,[0m [0mfirst_iter[0m[0;34m=[0m[0;32mFalse[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m_reset[0;34m(self, loader, first_iter)[0m
[1;32m   1234[0m         [0;31m# prime the prefetch loop[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1235[0m         [0;32mfor[0m [0m_[0m [0;32min[0m [0mrange[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_prefetch_factor[0m [0;34m*[0m [0mself[0m[0;34m.[0m[0m_num_workers[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1236[0;31m             [0mself[0m[0;34m.[0m[0m_try_put_index[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1237[0m [0;34m[0m[0m
[1;32m   1238[0m     [0;32mdef[0m [0m_try_get_data[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mtimeout[0m[0;34m=[0m[0m_utils[0m[0;34m.[0m[0mMP_STATUS_CHECK_INTERVAL[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m_try_put_index[0;34m(self)[0m
[1;32m   1484[0m [0;34m[0m[0m
[1;32m   1485[0m         [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m-> 1486[0;31m             [0mindex[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0m_next_index[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m   1487[0m         [0;32mexcept[0m [0mStopIteration[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m   1488[0m             [0;32mreturn[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py[0m in [0;36m_next_index[0;34m(self)[0m
[1;32m    696[0m [0;34m[0m[0m
[1;32m    697[0m     [0;32mdef[0m [0m_next_index[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 698[0;31m         [0;32mreturn[0m [0mnext[0m[0;34m([0m[0mself[0m[0;34m.[0m[0m_sampler_iter[0m[0;34m)[0m  [0;31m# may raise StopIteration[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    699[0m [0;34m[0m[0m
[1;32m    700[0m     [0;32mdef[0m [0m_next_data[0m[0;34m([0m[0mself[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/sampler.py[0m in [0;36m__iter__[0;34m(self)[0m
[1;32m    342[0m                 [0;32myield[0m [0;34m[[0m[0;34m*[0m[0mbatch_droplast[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m    343[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 344[0;31m             [0mbatch[0m [0;34m=[0m [0;34m[[0m[0;34m*[0m[0mitertools[0m[0;34m.[0m[0mislice[0m[0;34m([0m[0msampler_iter[0m[0;34m,[0m [0mself[0m[0;34m.[0m[0mbatch_size[0m[0;34m)[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    345[0m             [0;32mwhile[0m [0mbatch[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    346[0m                 [0;32myield[0m [0mbatch[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torch/utils/data/sampler.py[0m in [0;36m__iter__[0;34m(self)[0m
[1;32m    195[0m             ).tolist()
[1;32m    196[0m         [0;32melse[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 197[0;31m             [0;32mfor[0m [0m_[0m [0;32min[0m [0mrange[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mnum_samples[0m [0;34m//[0m [0mn[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    198[0m                 [0;32myield[0m [0;32mfrom[0m [0mtorch[0m[0;34m.[0m[0mrandperm[0m[0;34m([0m[0mn[0m[0;34m,[0m [0mgenerator[0m[0;34m=[0m[0mgenerator[0m[0;34m)[0m[0;34m.[0m[0mtolist[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    199[0m             yield from torch.randperm(n, generator=generator).tolist()[

[0;31mZeroDivisionError[0m: integer division or modulo by zero

## === cell 14
def plot_result(train_losses, val_losses, train_acc, val_acc):
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(7,6))
    
    ax1.plot(train_losses,label='train_losses')
    ax1.plot(val_losses, label='val_losses')
    
    ax2.plot(train_acc, label='train_acc', color='brown')
    ax2.plot(val_acc,label='val_acc', color='pink')
    
    ax1.legend()
    ax2.legend()
    plt.show()
