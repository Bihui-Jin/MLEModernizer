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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

0.7137396121883658

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install ../input/efficientnet/EfficientNet-PyTorch -f ./ --no-index

## === cell 1
from PIL import Image
from tqdm import tqdm
import copy 
import pandas as pd
from torchvision import transforms, models
import torchvision
from torch import optim
from torch.optim import lr_scheduler
import torch
import os

from efficientnet_pytorch import EfficientNet

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1839634808.py in <cell line: 0>()
     10 import os
     11 
---> 12 from efficientnet_pytorch import EfficientNet

ModuleNotFoundError: No module named 'efficientnet_pytorch'

## === cell 2
transform_train = transforms.Compose([
        transforms.RandomResizedCrop(640),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
transform_valid = transforms.Compose([
        transforms.Resize(640),
        transforms.CenterCrop(640),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

## === cell 3
import random
with open('../input/plant-pathology-2021-fgvc8/train.csv', 'r') as f:
    csv = f.readlines()[1:]
    for _ in range(5):
        random.shuffle(csv)
    cnt = int(len(csv)*0.9)
    train_csv = csv[:cnt]
    valid_csv = csv[cnt:]

## === cell 4
class torchvision_Dataset(torch.utils.data.Dataset): 
    def __init__(self, data_root, csv, transforms=None):
        self.data = csv
        self.image_path = data_root
        label = {i.split(',')[1] for i in self.data}
        self.label = {label:idx for idx, label in enumerate(label)}
        self.transform = transforms
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx): 
        image_name, label_name = self.data[idx].split(',')
        img = Image.open(os.path.join(self.image_path, image_name))
        if self.transform:
            x = self.transform(img)
        
        return x, self.label[label_name]

## === cell 5
train_dataset = torchvision_Dataset('train_images', train_csv, transform_train)
valid_dataset = torchvision_Dataset('train_images', valid_csv, transform_valid)

## === cell 6
train_dataloaders = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)
valid_dataloaders = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=1)

## === cell 7
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

## === cell 8
model_ft = EfficientNet.from_name('efficientnet-b4', num_classes=12)
model_ft.to(device)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/348404067.py in <cell line: 0>()
      1 # model_ft = models.resnet34(pretrained=False)
----> 2 model_ft = EfficientNet.from_name('efficientnet-b4', num_classes=12)
      3 model_ft.to(device)

NameError: name 'EfficientNet' is not defined

## === cell 9
from torch.optim.lr_scheduler import _LRScheduler
class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

## === cell 10
criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, 30, eta_min=0, last_epoch=-1)
exp_lr_scheduler = GradualWarmupScheduler(optimizer_ft, multiplier=100, total_epoch=3, after_scheduler=cosine_scheduler)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1128319976.py in <cell line: 0>()
      1 criterion = torch.nn.CrossEntropyLoss()
----> 2 optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
      3 cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, 30, eta_min=0, last_epoch=-1)
      4 exp_lr_scheduler = GradualWarmupScheduler(optimizer_ft, multiplier=100, total_epoch=3, after_scheduler=cosine_scheduler)

NameError: name 'model_ft' is not defined

## === cell 11
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        train_corrects = 0
        train_data_cnt = 0
        train_progress_bar = tqdm(train_dataloaders) 
        for inputs, labels in train_progress_bar:
            model.train()
            
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            train_data_cnt += inputs.size(0)
            train_progress_bar.set_description(f" Epoch[{epoch+1}/{num_epochs}] train : runing_Loss {running_loss / train_data_cnt:.5f}, train_acc {train_corrects / train_data_cnt:.5f}")
        scheduler.step()
 
        valid_corrects = 0
        valid_data_cnt = 0
        valid_progress_bar = tqdm(valid_dataloaders)
        for inputs, labels in valid_progress_bar:    
            model.eval()
           
            inputs = inputs.to(device)
            labels = labels.to(device)
           
            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                
            valid_corrects += torch.sum(preds == labels.data)
            valid_data_cnt += inputs.size(0)
            valid_progress_bar.set_description(f" Epoch[{epoch+1}/{num_epochs}] valid : valid_acc {valid_corrects / valid_data_cnt}")
            
        epoch_acc = valid_corrects / valid_dataset.__len__()
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model_ft.state_dict(), f"outputs/{best_epoch}.pth")
            print(f"best epoch : {best_epoch}")
    best_model_wts = copy.deepcopy(model.state_dict())        
    return best_model_wts

## === cell 13
model_ft.load_state_dict(torch.load('../input/best-model/28.pth'))
model_ft.to(device)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1216793423.py in <cell line: 0>()
----> 1 model_ft.load_state_dict(torch.load('../input/best-model/28.pth'))
      2 model_ft.to(device)

NameError: name 'model_ft' is not defined

## === cell 14
label2idx = {'frog_eye_leaf_spot\n': 0, 'rust frog_eye_leaf_spot\n': 1, 'healthy\n': 2, 'scab frog_eye_leaf_spot complex\n': 3, 'rust\n': 4, 'powdery_mildew complex\n': 5, 'frog_eye_leaf_spot complex\n': 6, 'powdery_mildew\n': 7, 'rust complex\n': 8, 'scab frog_eye_leaf_spot\n': 9, 'scab\n': 10, 'complex\n': 11}
idx2label = {label2idx[label]:label.strip() for label in label2idx}

## === cell 15
from glob import glob
import csv
img_paths = glob("../input/plant-pathology-2021-fgvc8/test_images/*")

submit =[] 

for img_path in img_paths:
    model_ft.eval()
    img = Image.open(img_path)
    img = transform_valid(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        pred = model_ft(img.cuda())
    _, top_one = torch.max(pred, 1)
    img_path = img_path.split('/')[-1]
    
    submit.append([img_path,idx2label[int(top_one)]])
    
submission = pd.DataFrame(submit, columns=["image", "labels"])
submission.to_csv("/kaggle/working/submission.csv", index=False)

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2615481710.py in <cell line: 0>()
      6 
      7 for img_path in img_paths:
----> 8     model_ft.eval()
      9     img = Image.open(img_path)
     10     img = transform_valid(img)

NameError: name 'model_ft' is not defined
