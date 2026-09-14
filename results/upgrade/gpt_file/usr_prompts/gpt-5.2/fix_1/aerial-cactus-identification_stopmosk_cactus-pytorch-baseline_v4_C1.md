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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.8

# 3. Installed packages

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

# 4. Data file paths

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

# 5. Target score

0.995

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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

    if i % 10 != 0:   # Пропускаем 9/10 всех фото
        continue

    is_cactus = int(train_val_labels.loc[train_val_labels['id'] == filename]['has_cactus'])

    if i % 70 == 0:   
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
batch_size = 250

train_dir = join(root_dir, 'train')
val_dir = join(root_dir, 'val')

classes = ['No', 'Cactus']

train_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms)
val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)


for images, labels in train_dataloader:
    print(images.size())
    print(labels.size())
    break


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2475949969.py in <cell line: 0>()
     20 
     21 
---> 22 train_dataset = torchvision.datasets.ImageFolder(train_dir, train_transforms)
     23 val_dataset = torchvision.datasets.ImageFolder(val_dir, val_transforms)
     24 

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in __init__(self, root, transform, target_transform, loader, is_valid_file, allow_empty)
    326         allow_empty: bool = False,
    327     ):
--> 328         super().__init__(
    329             root,
    330             loader,

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in __init__(self, root, loader, extensions, transform, target_transform, is_valid_file, allow_empty)
    148         super().__init__(root, transform=transform, target_transform=target_transform)
    149         classes, class_to_idx = self.find_classes(self.root)
--> 150         samples = self.make_dataset(
    151             self.root,
    152             class_to_idx=class_to_idx,

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in make_dataset(directory, class_to_idx, extensions, is_valid_file, allow_empty)
    201             # is potentially overridden and thus could have a different logic.
    202             raise ValueError("The class_to_idx parameter cannot be None.")
--> 203         return make_dataset(
    204             directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file, allow_empty=allow_empty
    205         )

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in make_dataset(directory, class_to_idx, extensions, is_valid_file, allow_empty)
    102         if extensions is not None:
    103             msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
--> 104         raise FileNotFoundError(msg)
    105 
    106     return instances

FileNotFoundError: Found no valid file for the classes has_cactus, no_cactus. Supported extensions are: .jpg, .jpeg, .png, .ppm, .bmp, .pgm, .tif, .tiff, .webp

## === cell 7
show_sample_images(train_dataloader, batch_size, 72, denormalize=True)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3724011563.py in <cell line: 0>()
----> 1 show_sample_images(train_dataloader, batch_size, 72, denormalize=True)

NameError: name 'train_dataloader' is not defined

## === cell 8
print(f'Batch size: {batch_size}')
print(f'Train batches: {len(train_dataloader)}, Train samples: {len(train_dataset)}')
print(f'Val batches:   {len(val_dataloader)}, Val samples:    {len(val_dataset)}')


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2677534514.py in <cell line: 0>()
      1 print(f'Batch size: {batch_size}')
----> 2 print(f'Train batches: {len(train_dataloader)}, Train samples: {len(train_dataset)}')
      3 print(f'Val batches:   {len(val_dataloader)}, Val samples:    {len(val_dataset)}')

NameError: name 'train_dataloader' is not defined

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
        print(f'\nEpoch {epoch}/{num_epochs-1}: \n', end='')

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

        print('\n End epoch: ', end='')
        validate(model, loss, optimizer)
        
    return model


## === cell 10
model = models.mobilenet_v2(pretrained=True)


model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)

model = model.cuda()

loss = torch.nn.CrossEntropyLoss() #weight=torch.FloatTensor([1, 1]).cuda())
optimizer = torch.optim.Adam(model.parameters())#, lr=1.0e-3, weight_decay=0.01, amsgrad=True)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)  


## === cell 11
print(f'Batch size: {batch_size}\nBatches: {len(train_dataloader)}\nAll elements: {len(train_dataset)}')


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/306320188.py in <cell line: 0>()
----> 1 print(f'Batch size: {batch_size}\nBatches: {len(train_dataloader)}\nAll elements: {len(train_dataset)}')

NameError: name 'train_dataloader' is not defined

## === cell 12
epochs = int(np.sqrt(batch_size)) + 5  # Automatically choose epoch number

train_model(model, loss, optimizer, scheduler, num_epochs=epochs);


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3831950517.py in <cell line: 0>()
      1 epochs = int(np.sqrt(batch_size)) + 5  # Automatically choose epoch number
      2 
----> 3 train_model(model, loss, optimizer, scheduler, num_epochs=epochs);

/tmp/ipykernel_11/2983487495.py in train_model(model, loss, optimizer, scheduler, num_epochs)
     46         print(f'\nEpoch {epoch}/{num_epochs-1}: \n', end='')
     47 
---> 48         dataloader = train_dataloader
     49         model.train()  # Set model to training mode
     50 

NameError: name 'train_dataloader' is not defined

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
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/553012567.py in <cell line: 0>()
      1 test_transforms = val_transforms
      2 
----> 3 test_dataset = torchvision.datasets.ImageFolder(test_dir, test_transforms)
      4 
      5 test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in __init__(self, root, transform, target_transform, loader, is_valid_file, allow_empty)
    326         allow_empty: bool = False,
    327     ):
--> 328         super().__init__(
    329             root,
    330             loader,

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in __init__(self, root, loader, extensions, transform, target_transform, is_valid_file, allow_empty)
    148         super().__init__(root, transform=transform, target_transform=target_transform)
    149         classes, class_to_idx = self.find_classes(self.root)
--> 150         samples = self.make_dataset(
    151             self.root,
    152             class_to_idx=class_to_idx,

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in make_dataset(directory, class_to_idx, extensions, is_valid_file, allow_empty)
    201             # is potentially overridden and thus could have a different logic.
    202             raise ValueError("The class_to_idx parameter cannot be None.")
--> 203         return make_dataset(
    204             directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file, allow_empty=allow_empty
    205         )

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in make_dataset(directory, class_to_idx, extensions, is_valid_file, allow_empty)
    102         if extensions is not None:
    103             msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
--> 104         raise FileNotFoundError(msg)
    105 
    106     return instances

FileNotFoundError: Found no valid file for the classes unknown. Supported extensions are: .jpg, .jpeg, .png, .ppm, .bmp, .pgm, .tif, .tiff, .webp

## === cell 16
show_sample_images(test_dataloader, batch_size, 12, denormalize=True, classes=['Unknown'])


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3715065085.py in <cell line: 0>()
----> 1 show_sample_images(test_dataloader, batch_size, 12, denormalize=True, classes=['Unknown'])

NameError: name 'test_dataloader' is not defined

## === cell 17
model.eval()

test_predictions = []

i = 1
for images, labels in test_dataloader:
    images = images.cuda()
    with torch.set_grad_enabled(False):
        preds = model(images)
    test_predictions.append(T.softmax(preds, dim=1)[:,1].data.cpu().numpy())
    print(f'\r{i}/{len(test_dataloader)}', end='')
    i += 1
    
test_predictions = np.concatenate(test_predictions)  
test_predictions = (test_predictions >= 0.5).astype('int')


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3640887259.py in <cell line: 0>()
      4 
      5 i = 1
----> 6 for images, labels in test_dataloader:
      7     images = images.cuda()
      8     with torch.set_grad_enabled(False):

NameError: name 'test_dataloader' is not defined

## === cell 18
test_files = next(iter(os.walk(join(test_dir, 'unknown'))))[2]

print(sorted(test_files)[:10])


## === cell 19
submission_df = pd.DataFrame.from_dict({'id': sorted(test_files), 'has_cactus': test_predictions})
submission_df.set_index('id', inplace=True)
submission_df.head(25)


## === cell 20
submission_df.to_csv('submission.csv')


## === cell 21
!rm -rf train val test


## --- ERROR in outputing the csv:
Invalid submission: Submission and answers should have the same number of rows
