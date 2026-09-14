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

3.7

# 3. Installed packages

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

0.8545

# 6. Current score

0.9601

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import seaborn as sns
import os

import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image


## === cell 1
train_data_directory = "../input/train/train"
test_data_directory = "../input/test/test"


## === cell 2
train_csv_df = pd.read_csv("../input/train.csv")
train_csv_df.head()


## === cell 3
train_csv_df.shape


## === cell 4
train_csv_df.isna().sum()


## === cell 5
train_csv_df.has_cactus.value_counts()


## === cell 6
plt.style.use("seaborn")
train_csv_df.has_cactus.value_counts().plot(kind = "barh")
plt.ylabel("Classes")
plt.xlabel("Number of catus in each class")
plt.show()


## === cell 7
test_data_df = pd.read_csv("../input/sample_submission.csv")


## === cell 8
test_data_df.head()


## === cell 9
test_data_df.shape


## === cell 10
from sklearn.model_selection import train_test_split


## === cell 11
train_df, validation_df = train_test_split(train_csv_df, stratify=train_csv_df.has_cactus, test_size=0.2)


## === cell 12
train_df.shape


## === cell 13
validation_df.shape


## === cell 14
class AerialCatcusClassification(Dataset):
    def __init__(self,file_data,root_dir,transform=None):
        self.transform = transform
        self.file_data = file_data.values
        self.data_root = root_dir 
            
    def __len__(self):
        return len(self.file_data)
    
    def __getitem__(self, index):
        img_name, label = self.file_data[index]
        img_data = self.pil_loader(os.path.join(self.data_root, img_name))
        if self.transform:
            img_data = self.transform(img_data)
        return img_data, label
          
    def pil_loader(self,path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


## === cell 15
train_data = AerialCatcusClassification(file_data=train_df, root_dir=train_data_directory, transform = transforms.Compose([transforms.ToTensor()]))
validation_data = AerialCatcusClassification(file_data=validation_df, root_dir=train_data_directory, transform = transforms.Compose([transforms.ToTensor()]))

test_data = AerialCatcusClassification(file_data=test_data_df, root_dir=test_data_directory,transform = transforms.Compose([transforms.ToTensor()]))


## === cell 16
train_loader = torch.utils.data.DataLoader(train_data, batch_size=5, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=5, shuffle=True)


## === cell 17

def imshow(img, title):
    
    npimg = img.numpy()
    plt.figure(figsize = (20, 20))
    plt.axis("off")
    
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title, fontsize=15)
    plt.show()


## === cell 18

def show_batch_images(dataloader):
    
    images, labels = next(iter(dataloader))
    img = torchvision.utils.make_grid(images)
    
    imshow(img, title = [str(x.item()) for x in labels])


## === cell 19

for i in range(4):
    show_batch_images(train_loader)


## === cell 20
transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


## === cell 21
train_data = AerialCatcusClassification(file_data=train_df, root_dir=train_data_directory, transform = transform_train)
validation_data = AerialCatcusClassification(file_data=validation_df, root_dir=train_data_directory, transform = transform_train)

test_data = AerialCatcusClassification(file_data=test_data_df, root_dir=test_data_directory,transform = transform_test)


## === cell 22
train_loader = torch.utils.data.DataLoader(train_data, batch_size=10, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=10, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=10, shuffle=False)


## === cell 23
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
import copy


## === cell 24
num_classes = 2
training_batchsize = 10


## === cell 25

dataiter = iter(train_loader)
images, labels = dataiter.next()

print(images.shape)

print(images[1].shape)

print(labels)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3276592587.py in <cell line: 0>()
      2 
      3 dataiter = iter(train_loader)
----> 4 images, labels = dataiter.next()
      5 
      6 #shape of images bunch

AttributeError: '_SingleProcessDataLoaderIter' object has no attribute 'next'

## === cell 26

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


## === cell 27
vgg = models.vgg19_bn(pretrained = True)


## === cell 28
print("Number of trainable parameters: ", sum(p.numel() for p in vgg.parameters() if p.requires_grad))


## === cell 29
for name, child in vgg.named_children():
    print(name)


## === cell 30
for param in vgg.parameters():
    param.requires_grad = False


## === cell 31
final_in_features = vgg.classifier[6].in_features

vgg.classifier[6] = nn.Linear(final_in_features, num_classes)


## === cell 32
for param in vgg.parameters():
    if param.requires_grad:
        print(param.shape)


## === cell 33
criterion = nn.CrossEntropyLoss()
vgg = vgg.to(device) #push to gpu

optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, vgg.parameters()), lr=0.001, amsgrad=True, weight_decay=1e-4)


## === cell 34

def evaluation(dataloader, model):
    total, correct = 0, 0
    for data in dataloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        total += labels.size(0)

        correct += (pred == labels).sum().item()

    return 100 * correct/total


## === cell 35
def train_model(model, criterion, optimizer, num_epochs=25):
    
    since = time.time()
    train_acc, validation_acc = [], []
    best_loss = 1000
    best_model_wts = copy.deepcopy(model.state_dict())

    n_iter = np.ceil(len(train_loader.dataset)/training_batchsize)

    for epoch in range(num_epochs):
        running_trainloss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_trainloss += loss.item()

            del inputs, labels, outputs
            torch.cuda.empty_cache()

        else:
            running_trainloss = running_trainloss / len(train_loader.dataset)

            validation_accuracy = 0
            training_accuracy = 0
            
            model.eval()
            with torch.no_grad():
                validation_accuracy = evaluation(validation_loader, model)
                training_accuracy = evaluation(train_loader, model)
                
            if running_trainloss < best_loss:
                best_loss = running_trainloss
                best_model_wts = copy.deepcopy(model.state_dict())

            model.train()

            print("Epoch: {}/{}.. ".format(epoch+1, num_epochs), "Training Loss: {:.3f}.. ".format(running_trainloss),
                  "Training Accuracy % : {:.3f}.. ".format(training_accuracy),
                  "Validation Accuracy % : {:.3f}..".format(validation_accuracy))

            train_acc.append(training_accuracy)
            validation_acc.append(validation_accuracy)
    
    time_elapsed = time.time() - since        
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best training loss: {:4f}'.format(best_loss))
    
    torch.save(model.state_dict(best_model_wts),"saved.pth")
    print("best model saved")
    
    model.load_state_dict(best_model_wts)
    
    return model, train_acc, validation_acc


## === cell 36
best_model, train_acc, validation_acc = train_model(vgg, criterion, optimizer_ft,1)


## === cell 37
plt.plot(train_acc, label='Training Accuracy')
plt.plot(validation_acc, label='Validation Accuracy')
plt.legend(frameon=False)
plt.show()


## === cell 38
validation_acc_bestmodel = evaluation(validation_loader, best_model)
print("validation accuracy %: ", validation_acc_bestmodel)


## === cell 39

dataiter = iter(test_loader)
images, labels = dataiter.next()

print(images.shape)

print(images[1].shape)

print(labels[1])


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1746129929.py in <cell line: 0>()
      2 
      3 dataiter = iter(test_loader)
----> 4 images, labels = dataiter.next()
      5 
      6 #shape of images bunch

AttributeError: '_SingleProcessDataLoaderIter' object has no attribute 'next'

## === cell 40
def testdata_sumission(dataloader, model):
    since = time.time()
    predict = []
    
    model.eval()  # Set model to evaluate mode
    for data in dataloader:
        inputs, labels = data

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, pred = torch.max(outputs.data, 1)
        
        pred = pred.cpu().numpy()
        print(pred)
        for i in pred:
            predict.append(i)
      
    time_elapsed = time.time() - since
    print('Run complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    return predict


## === cell 41
test_data_df['has_cactus'] = testdata_sumission(test_loader, best_model)

test_data_df.to_csv('submission.csv', index= False)
