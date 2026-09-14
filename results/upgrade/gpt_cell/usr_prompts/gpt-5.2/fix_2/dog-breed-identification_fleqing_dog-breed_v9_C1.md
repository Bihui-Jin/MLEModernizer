# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.12

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
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
from torchvision.transforms import ToPILImage
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts,ExponentialLR
from sklearn.model_selection import train_test_split,KFold
from PIL import Image
import cv2
import albumentations
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from mpl_toolkits.axes_grid1 import ImageGrid


## === cell 2
train_data = pd.read_csv('/kaggle/input/dog-breed-identification/labels.csv')
labels = sorted(list(set(list(train_data['breed']))))
labels_num = []
for i in range(len(train_data)):
    labels_num.append(labels.index(train_data['breed'][i]))
train_data['number'] = labels_num
train_data.shape


## === cell 3
file_names = sorted(os.listdir('/kaggle/input/dog-breed-identification/test'))
file_names = [name[:-4] for name in file_names]
test_data = pd.DataFrame({'id':file_names})
test_data.head()


## === cell 4
transforms_train =transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
transforms_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


## === cell 5
class Dog_Breed(Dataset):
    def __init__(self, train_csv, transform = None, test = False):
        super().__init__()
        self.train_csv = train_csv
        self.image_path = list(self.train_csv['id']) #图像所在地址记录
        self.test = test
        if not self.test:
            self.label_nums = list(self.train_csv['number']) #图像的标号记录
        self.transform = transform
    def __getitem__(self, idx):
        '''
        idx : 所需要获取的图像的索引
        return : image， label
        '''
        if self.test:
            image = Image.open(os.path.join("/kaggle/input/dog-breed-identification/test", self.image_path[idx]+".jpg")).convert('RGB')
        else:
            image = Image.open(os.path.join("/kaggle/input/dog-breed-identification/train", self.image_path[idx]+".jpg")).convert('RGB')
        
        if(self.transform != None):
            image = self.transform(image)
        if not self.test:
            label = self.label_nums[idx]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.image_path)


## === cell 6
def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
device = get_device()
device


## === cell 7
train, valid = train_test_split(train_data, test_size=0.2, random_state=i)
trainset = Dog_Breed(train, transform = transforms_train)
validset = Dog_Breed(valid, transform = transforms_test)


## === cell 8
train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, drop_last=False)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=32, shuffle=False, drop_last=False)


## === cell 9
dataiter = iter(train_loader)
images, label = next(dataiter)
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 4))

for i, ax in enumerate(axes):
    image = images[i].numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    ax.imshow(image)
    ax.set_title(f"Label: {labels[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()


## === cell 10
class MyResNet50(nn.Module):
    def __init__(self, num_classes=120):
        super(MyResNet50, self).__init__()
        self.net = models.resnet50(pretrained=True)
        for param in self.net.parameters():
            param.requires_grad = False
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)
        

    def forward(self, x):
        x = self.net(x)

        return x


## === cell 11
class EfficientNetCustom(nn.Module):
    def __init__(self, num_classes=120):
        super(EfficientNetCustom, self).__init__()
        self.net = EfficientNet.from_pretrained('efficientnet-b0')  # 可以选择不同的 EfficientNet 版本
        for param in self.net.parameters():
            param.requires_grad = False
        in_features = self.net._fc.in_features
        self.net._fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)


## === cell 12
def train_model(model,train_loader, valid_loader,loss,optimizer,epoch,device = torch.device("cuda:0")):
    net = model.to(device)
    best_epoch = 0
    best_score = 0.0
    best_model_state = None
    early_stopping_round = 3
    losses = []

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min = 1e-4)
    for i in range(epoch):
        acc = 0
        loss_sum = 0
        net.train()
        for x, y in tqdm(train_loader):
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            loss_temp = loss(y_hat, y)
            loss_sum += loss_temp
            loss_temp.backward()
            optimizer.step()
            acc += torch.sum(y_hat.argmax(dim=1) == y)
        scheduler.step()
        losses.append(loss_sum.cpu().detach().numpy() / len(train_loader))
        print( "epoch: ", i, "loss=", loss_sum.item()/(len(train_loader)*train_loader.batch_size), "训练集准确度=",(acc/(len(train_loader)*train_loader.batch_size)).item(),end="")

        test_acc = 0
        net.eval()
        for x, y in tqdm(valid_loader):
            x = x.to(device)
            y = y.to(device)
            y_hat = net(x)
            test_acc += torch.sum(y_hat.argmax(dim=1) == y)
        print("验证集准确度", (test_acc / (len(valid_loader)*valid_loader.batch_size)).item())
        if test_acc > best_score:
            best_model_state = copy.deepcopy(net.state_dict())
            best_score = test_acc
            best_epoch = i
            print('best epoch save!')
        if i - best_epoch >= early_stopping_round:
            break

    net.load_state_dict(best_model_state)
    
    testset = Dog_Breed(test_data, transform = transforms_test,test = True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, drop_last=False)

    predictions = []
    with torch.no_grad():
        for x in tqdm(test_loader):
            x = x.to(device)
            y_hat = net(x)
            predict = list(y_hat.cpu().detach().numpy())
            predictions.extend(predict)
    predictions = np.array(predictions)
    prediction = torch.tensor(predictions).reshape(-1, 120)
    return prediction


## === cell 13
learn_rate = 0.001
momentum = 0.9 
epoch = 15


## === cell 14
kfold = KFold(n_splits=5, shuffle=True, random_state=2021)

_test_dir = "/kaggle/input/dog-breed-identification/test"
_existing_test_stems = {
    fn[:-4]
    for fn in os.listdir(_test_dir)
    if fn.lower().endswith(".jpg") and len(fn) > 4
}
test_data = test_data[test_data["id"].isin(_existing_test_stems)].reset_index(drop=True)

all_predictions_sum = torch.zeros((len(test_data), 120), dtype=torch.float32)
for train_index, val_index in kfold.split(train_data):
    model = MyResNet50()
    train, valid = train_data.iloc[train_index], train_data.iloc[val_index]
    trainset = Dog_Breed(train, transform=transforms_train)
    validset = Dog_Breed(valid, transform=transforms_test)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, drop_last=False
    )
    valid_loader = torch.utils.data.DataLoader(
        validset, batch_size=32, shuffle=False, drop_last=False
    )

    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.net.fc.parameters(), lr=learn_rate, weight_decay=1e-5)

    prediction = train_model(
        model, train_loader, valid_loader, loss, optimizer, epoch, device
    )

    all_predictions_sum += prediction * 0.2
result = pd.DataFrame(all_predictions_sum.numpy(), columns=labels)
result = pd.concat([test_data, result], axis=1)
result.to_csv("dog_breed.csv", index=False)


## === cell 16
df = pd.read_csv('/kaggle/working/dog_breed.csv')
data = df.iloc[:,1:].values
tensor_data = torch.tensor(data,dtype=torch.float32)
probability = F.softmax(tensor_data,dim=1)
df.iloc[:,1:] = probability.numpy()
df.to_csv('dog_breed.csv',index=False)
