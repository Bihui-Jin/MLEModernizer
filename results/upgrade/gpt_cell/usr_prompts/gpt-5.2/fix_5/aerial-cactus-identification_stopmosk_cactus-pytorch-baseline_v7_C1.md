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

3.8

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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
from os.path import join
import shutil

from tqdm import tqdm   # Progress bar

import torch
import torchvision
import torch.nn.functional as T
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset


random.seed(6)
np.random.seed(6)
torch.manual_seed(6)
torch.cuda.manual_seed(6)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

root_dir = ''
input_dir = '../input/aerial-cactus-identification'


## === cell 1
train_val_labels = pd.read_csv(join(input_dir, 'train.csv'))
train_val_labels.head()


## === cell 2
plt.figure(figsize=(3,3))
plt.title('Labels distribution')
sns.countplot(train_val_labels['has_cactus']);


## === cell 3
labels = ['no_cactus', 'has_cactus']

train_dir = join(root_dir, 'train')
val_dir = join(root_dir, 'val')
test_dir = join(root_dir, 'test')

for label in labels:
    os.makedirs(join(train_dir, label), exist_ok=True)
    os.makedirs(join(val_dir, label), exist_ok=True)


## === cell 4

source_dir = join(input_dir, 'train', 'train')

for i, filename in enumerate(tqdm(os.listdir(source_dir))):


    is_cactus = int(train_val_labels.loc[train_val_labels['id'] == filename]['has_cactus'])

    if i % 7 == 0:   #if i % 7 == 0:   
        shutil.copy(join(source_dir, filename), join(val_dir, labels[is_cactus], filename))
    else:
        shutil.copy(join(source_dir, filename), join(train_dir, labels[is_cactus], filename))


## === cell 5
def show_sample_images(dataloader, batch_size, images_from_batch=0, denormalize=False, classes=None):
    if denormalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
    else:
        mean = np.array([0., 0., 0.])
        std = np.array([1., 1., 1.])
    
    if images_from_batch == 0 or images_from_batch > batch_size:
            images_from_batch = batch_size
        
    for images, labels in dataloader:
        plt.figure(figsize=(20, (batch_size // 20 + 1) * 3))

        cols = 12
        rows = batch_size // cols + 1
        for i in range(images_from_batch):
            image = images[i].permute(1, 2, 0).numpy() * std + mean   # Размерность RGB в конец
            plt.subplot(rows, cols, i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image.clip(0, 1))
            if classes is not None:
                plt.xlabel(classes[labels[i].numpy()])
        plt.show()
        
        break


## === cell 6
batch_size = 500

train_dir = os.path.abspath(join(root_dir, "train"))
val_dir = os.path.abspath(join(root_dir, "val"))

expected_classes = ["no_cactus", "has_cactus"]

use_imagefolder = True
for base in [train_dir, val_dir]:
    for cls in expected_classes:
        cls_dir = join(base, cls)
        if (not os.path.isdir(cls_dir)) or (len(os.listdir(cls_dir)) == 0):
            use_imagefolder = False
            break
    if not use_imagefolder:
        break

classes = ["No", "Cactus"]

train_transforms1 = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_transforms2 = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_transforms3 = transforms.Compose(
    [
        transforms.RandomVerticalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_transforms4 = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=1),
        transforms.RandomVerticalFlip(p=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if use_imagefolder:
    tds1 = torchvision.datasets.ImageFolder(train_dir, train_transforms1)
    tds2 = torchvision.datasets.ImageFolder(train_dir, train_transforms2)
    tds3 = torchvision.datasets.ImageFolder(train_dir, train_transforms3)
    tds4 = torchvision.datasets.ImageFolder(train_dir, train_transforms4)

    train_dataset = torch.utils.data.ConcatDataset([tds1, tds2, tds3, tds4])
    val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)
else:
    from PIL import Image

    raw_train_dir = join(input_dir, "train")
    if not os.path.isdir(raw_train_dir):
        raw_train_dir = join(input_dir, "train", "train")

    train_val_df = train_val_labels.copy()
    train_val_df = train_val_df.sort_values("id").reset_index(drop=True)

    val_mask = (np.arange(len(train_val_df)) % 7) == 0
    train_df = train_val_df.loc[~val_mask].reset_index(drop=True)
    val_df = train_val_df.loc[val_mask].reset_index(drop=True)

    class CactusCSVDataset(Dataset):
        def __init__(self, df, img_dir, transform=None):
            self.df = df
            self.img_dir = img_dir
            self.transform = transform

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            img_path = join(self.img_dir, row["id"])
            img = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            label = int(row["has_cactus"])
            return img, label

    ds1 = CactusCSVDataset(train_df, raw_train_dir, train_transforms1)
    ds2 = CactusCSVDataset(train_df, raw_train_dir, train_transforms2)
    ds3 = CactusCSVDataset(train_df, raw_train_dir, train_transforms3)
    ds4 = CactusCSVDataset(train_df, raw_train_dir, train_transforms4)
    train_dataset = torch.utils.data.ConcatDataset([ds1, ds2, ds3, ds4])

    val_dataset = CactusCSVDataset(val_df, raw_train_dir, val_transforms)

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0
)
val_dataloader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0
)


for images, labels in train_dataloader:
    print(images.size())
    print(labels.size())
    break


## === cell 7
show_sample_images(train_dataloader, batch_size, 72, denormalize=True)


## === cell 8
print(f'Batch size: {batch_size}')
print(f'Train batches: {len(train_dataloader)}, Train samples: {len(train_dataset)}')
print(f'Val batches:   {len(val_dataloader)}, Val samples:    {len(val_dataset)}')


## === cell 9
train_batch_loss_history = []
train_batch_accuracy_history = []

train_loss_history = []
train_accuracy_history = []

val_loss_history = []
val_accuracy_history = []

def validate(model, loss, optimizer):
        
    dataloader = val_dataloader
    model.eval()   # Set model to evaluate mode

    sum_loss = 0.
    sum_accuracy = 0.

    for inputs, labels in dataloader:
        inputs = inputs.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            preds = model(inputs)
            loss_value = loss(preds, labels)
            preds_class = preds.argmax(dim=1)

        sum_loss += loss_value.item()
        sum_accuracy += (preds_class == labels.data).float().mean().cpu().numpy().item()

    val_loss = sum_loss / len(dataloader)
    val_accuracy = sum_accuracy / len(dataloader)

    val_loss_history.append(val_loss)
    val_accuracy_history.append(val_accuracy)
    
    print(f'Validation accuracy {val_accuracy * 100:.2f} %, loss {val_loss:.4f}')

    model.train()  # Вернули как было


def train_model(model, loss, optimizer, scheduler, num_epochs):
        
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}: ', end='')

        dataloader = train_dataloader
        model.train()  # Set model to training mode

        sum_loss = 0.
        sum_accuracy = 0.

        for inputs, labels in dataloader:   #tqdm(dataloader):
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()

            with torch.set_grad_enabled(True):
                preds = model(inputs)
                loss_value = loss(preds, labels)
                preds_class = preds.argmax(dim=1)

                loss_value.backward()
                optimizer.step()

            batch_loss = loss_value.item()
            batch_accuracy = (preds_class == labels.data).float().mean().cpu().numpy().item()

            sum_loss += batch_loss
            sum_accuracy += batch_accuracy
            
            train_batch_loss_history.append(batch_loss)
            train_batch_accuracy_history.append(batch_accuracy)
            
        epoch_loss = sum_loss / len(dataloader)
        epoch_acc = sum_accuracy / len(dataloader)

        train_loss_history.append(epoch_loss)
        train_accuracy_history.append(epoch_acc)
        scheduler.step()

        validate(model, loss, optimizer)
        
    return model


## === cell 10
model = models.resnet50(pretrained=True)


model.fc = torch.nn.Linear(model.fc.in_features, 2)

model = model.cuda()

loss = torch.nn.CrossEntropyLoss() #weight=torch.FloatTensor([1, 1]).cuda())
optimizer = torch.optim.Adam(model.parameters())#, lr=1.0e-3, weight_decay=0.01, amsgrad=True)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.33)


## === cell 11
print(f'Batch size: {batch_size}\nBatches: {len(train_dataloader)}\nAll elements: {len(train_dataset)}')


## === cell 12
epochs = 7

train_model(model, loss, optimizer, scheduler, num_epochs=epochs);


## === cell 13
plt.figure(figsize=(20,10))
    
plt.subplot(1, 3, 1)
plt.plot(train_batch_loss_history, label='Train Batch Loss')
plt.plot(train_batch_accuracy_history, label='Train Batch Accuracy')
plt.legend();

plt.subplot(1, 3, 2)
plt.plot(train_accuracy_history, label='Train accuracy')
plt.plot(val_accuracy_history, label='Val accuracy')
plt.legend();
    
plt.subplot(1, 3, 3)
plt.plot(train_loss_history, label='Train Loss')
plt.plot(val_loss_history, label='Val Loss')
plt.legend();


## === cell 14
os.makedirs(join(root_dir, 'test'), exist_ok=True)

test_dir = join(root_dir, 'test')

os.makedirs(join(test_dir, 'unknown'), exist_ok=True)

source_dir = join(input_dir, 'test', 'test')

for i, filename in enumerate(tqdm(sorted(os.listdir(source_dir)))):
    shutil.copy(join(source_dir, filename), join(test_dir, 'unknown', filename))
    
    if i < 10:
        print(filename)


## === cell 15
test_transforms = val_transforms

test_dataset = torchvision.datasets.ImageFolder(test_dir, test_transforms)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

for images, labels in test_dataloader:
    print(images.size())
    print(labels.size())
    break


## --- ERROR in cell 15, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mFileNotFoundError[0m                         Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/553012567.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0mtest_transforms[0m [0;34m=[0m [0mval_transforms[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;34m[0m[0m
[0;32m----> 3[0;31m [0mtest_dataset[0m [0;34m=[0m [0mtorchvision[0m[0;34m.[0m[0mdatasets[0m[0;34m.[0m[0mImageFolder[0m[0;34m([0m[0mtest_dir[0m[0;34m,[0m [0mtest_transforms[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m [0mtest_dataloader[0m [0;34m=[0m [0mDataLoader[0m[0;34m([0m[0mtest_dataset[0m[0;34m,[0m [0mbatch_size[0m[0;34m=[0m[0mbatch_size[0m[0;34m,[0m [0mshuffle[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m [0mpin_memory[0m[0;34m=[0m[0;32mTrue[0m[0;34m,[0m [0mnum_workers[0m[0;34m=[0m[0;36m0[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py[0m in [0;36m__init__[0;34m(self, root, transform, target_transform, loader, is_valid_file, allow_empty)[0m
[1;32m    326[0m         [0mallow_empty[0m[0;34m:[0m [0mbool[0m [0;34m=[0m [0;32mFalse[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    327[0m     ):
[0;32m--> 328[0;31m         super().__init__(
[0m[1;32m    329[0m             [0mroot[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    330[0m             [0mloader[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py[0m in [0;36m__init__[0;34m(self, root, loader, extensions, transform, target_transform, is_valid_file, allow_empty)[0m
[1;32m    148[0m         [0msuper[0m[0;34m([0m[0;34m)[0m[0;34m.[0m[0m__init__[0m[0;34m([0m[0mroot[0m[0;34m,[0m [0mtransform[0m[0;34m=[0m[0mtransform[0m[0;34m,[0m [0mtarget_transform[0m[0;34m=[0m[0mtarget_transform[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    149[0m         [0mclasses[0m[0;34m,[0m [0mclass_to_idx[0m [0;34m=[0m [0mself[0m[0;34m.[0m[0mfind_classes[0m[0;34m([0m[0mself[0m[0;34m.[0m[0mroot[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 150[0;31m         samples = self.make_dataset(
[0m[1;32m    151[0m             [0mself[0m[0;34m.[0m[0mroot[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    152[0m             [0mclass_to_idx[0m[0;34m=[0m[0mclass_to_idx[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py[0m in [0;36mmake_dataset[0;34m(directory, class_to_idx, extensions, is_valid_file, allow_empty)[0m
[1;32m    201[0m             [0;31m# is potentially overridden and thus could have a different logic.[0m[0;34m[0m[0;34m[0m[0m
[1;32m    202[0m             [0;32mraise[0m [0mValueError[0m[0;34m([0m[0;34m"The class_to_idx parameter cannot be None."[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 203[0;31m         return make_dataset(
[0m[1;32m    204[0m             [0mdirectory[0m[0;34m,[0m [0mclass_to_idx[0m[0;34m,[0m [0mextensions[0m[0;34m=[0m[0mextensions[0m[0;34m,[0m [0mis_valid_file[0m[0;34m=[0m[0mis_valid_file[0m[0;34m,[0m [0mallow_empty[0m[0;34m=[0m[0mallow_empty[0m[0;34m[0m[0;34m[0m[0m
[1;32m    205[0m         )

[0;32m/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py[0m in [0;36mmake_dataset[0;34m(directory, class_to_idx, extensions, is_valid_file, allow_empty)[0m
[1;32m    102[0m         [0;32mif[0m [0mextensions[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    103[0m             [0mmsg[0m [0;34m+=[0m [0;34mf"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 104[0;31m         [0;32mraise[0m [0mFileNotFoundError[0m[0;34m([0m[0mmsg[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    105[0m [0;34m[0m[0m
[1;32m    106[0m     [0;32mreturn[0m [0minstances[0m[0;34m[0m[0;34m[0m[0m

[0;31mFileNotFoundError[0m: Found no valid file for the classes unknown. Supported extensions are: .jpg, .jpeg, .png, .ppm, .bmp, .pgm, .tif, .tiff, .webp

## === cell 16
show_sample_images(test_dataloader, batch_size, 12, denormalize=True, classes=['Unknown'])
