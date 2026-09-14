# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.12

# 3. Installed packages

fastai==2.8.5
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

# 5. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os
import zipfile
from fastai.vision.all import *
from PIL import Image
import pandas as pd
import random
import shutil
from torchvision.transforms import ToTensor


## === cell 2
train_file_path = "/kaggle/input/aerial-cactus-identification/train.zip"
image_dir = "/kaggle/working/"

with zipfile.ZipFile(train_file_path, "r") as zip_ref:
    zip_ref.extractall(image_dir)

train_dir = "/kaggle/working/train"
if not os.path.isdir(train_dir):
    alt_train_dir = "/kaggle/working/aerial-cactus-identification/train"
    if os.path.isdir(alt_train_dir):
        train_dir = alt_train_dir

train_list = os.listdir(train_dir)


## === cell 3
len(train_list)


## === cell 4
for i, file_name in enumerate(train_list[:10]):
    print(f"{i+1}: {file_name}")


## === cell 5
test_file_path = "/kaggle/input/aerial-cactus-identification/test.zip"

with zipfile.ZipFile(test_file_path, "r") as zip_ref:
    zip_ref.extractall(image_dir)

test_dir = "/kaggle/working/test"
if not os.path.isdir(test_dir):
    alt_test_dir = "/kaggle/working/aerial-cactus-identification/test"
    if os.path.isdir(alt_test_dir):
        test_dir = alt_test_dir

test_list = os.listdir(test_dir)


## === cell 6
len(test_list)


## === cell 7
for i, file_name in enumerate(test_list[:10]):
    print(f"{i+1}: {file_name}")


## === cell 8
train_image_file_path = get_image_files(train_dir)

if len(train_image_file_path) == 0:
    raise FileNotFoundError(
        f"No training images found in train_dir={train_dir!r}. Check extraction paths."
    )

train_image_file_path[0]


## === cell 9
im = Image.open(train_image_file_path[0])
im


## === cell 10
test_image_file_path = get_image_files(test_dir)

if len(test_image_file_path) == 0:
    raise FileNotFoundError(
        f"No test images found in test_dir={test_dir!r}. Check extraction paths."
    )

test_image_file_path[0]


## === cell 11
im2 = Image.open(test_image_file_path[0])
im2


## === cell 12
train_csv = pd.read_csv('/kaggle/input/aerial-cactus-identification/train.csv')
test_csv = pd.read_csv('/kaggle/input/aerial-cactus-identification/sample_submission.csv')


## === cell 13
train_csv.head()


## === cell 14
test_csv.head()


## === cell 15
train_csv[train_csv['has_cactus']==1]


## === cell 16
train_csv[train_csv['has_cactus']==0]


## === cell 18
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, path, df, transform=None):  
        self.path = path
        self.df = df
        self.transform = transform      
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        img_id = self.df.iloc[i, 0]
        
        img = Image.open(self.path + img_id).convert('RGB')
        label = self.df.iloc[i, 1]
        
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


## === cell 22
from sklearn.model_selection import train_test_split

train, valid = train_test_split(train_csv, test_size=0.1, stratify = train_csv['has_cactus'])


## === cell 23
train.iloc[0, 0]


## === cell 24
train.iloc[0, 1]


## === cell 25
len(train), len(valid)


## === cell 26
valid['has_cactus'].value_counts()


## === cell 27
print('/kaggle/working/train/' + train.iloc[0, 0])


## === cell 32
train_ds, valid_ds = CustomDataset(path='/kaggle/working/train/', df=train, transform=ToTensor()), CustomDataset(path='/kaggle/working/train/', df=valid, transform=ToTensor())


## === cell 33
len(train_ds)


## === cell 34
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, path, df, transform=None):
        self.path = path
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img_id = str(self.df.iloc[i, 0])

        base = self.path
        if not os.path.isdir(base):
            alt_base = os.path.join(
                os.path.dirname(base.rstrip("/")),
                "aerial-cactus-identification",
                os.path.basename(base.rstrip("/")),
            )
            if os.path.isdir(alt_base):
                base = alt_base

        img_path = os.path.join(base, img_id)
        img = Image.open(img_path).convert("RGB")
        label = self.df.iloc[i, 1]

        if self.transform:
            img = self.transform(img)

        return img, label


## === cell 35
train_ds = CustomDataset(path="/kaggle/working/train/", df=train, transform=ToTensor())
valid_ds = CustomDataset(path="/kaggle/working/train/", df=valid, transform=ToTensor())

x, y = train_ds[0]
x.shape, y


## === cell 36
test_ds = CustomDataset('/kaggle/working/test/', test_csv, transform=ToTensor())


## === cell 37
z, k = test_ds[0]
z.shape, k


## === cell 38
def collate(idxs, ds):
    xb, yb = zip(*[ds[i] for i in idxs])
    return torch.stack(xb), torch.tensor([y for y in yb], dtype=torch.int64)


## === cell 39
x, y = collate([1, 2], train_ds)
x.shape, y


## === cell 40
class DataLoader:
    def __init__(self, ds, bs=64, shuffle=False, n_workers=1):
        self.ds, self.bs, self.shuffle, self.n_workers = ds, bs, shuffle, n_workers
        
    def __len__(self): return (len(self.ds)-1)//self.bs+1
    
    def __iter__(self):
        idxs = L.range(self.ds)
        if self.shuffle: idxs = idxs.shuffle()
        chunks = [idxs[n: n+self.bs] for n in range(0, len(self.ds), self.bs)]
        with ProcessPoolExecutor(self.n_workers) as ex:
            yield from ex.map(collate, chunks, ds=self.ds)


## === cell 41
n_workers = min(16, defaults.cpus)
train_dl = DataLoader(train_ds, bs=64, shuffle=True, n_workers = n_workers)
valid_dl = DataLoader(valid_ds, bs=64, shuffle=False, n_workers = n_workers)

xb, yb = first(train_dl)


## === cell 42
test_dl = DataLoader(test_ds, bs=64, shuffle=False, n_workers=n_workers)


## === cell 43
zb, kb = first(test_dl)
zb, kb


## === cell 44
zb.shape, kb.shape


## === cell 45
xb, yb


## === cell 46
xb.shape, yb.shape


## === cell 47
import torch.nn as nn #신경망 모듈
import torch.nn.functional as F #신경망 모듈에서 자주 사용되는 함수 


## === cell 48
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=2),
                                   nn.ReLU(), 
                                   nn.MaxPool2d(kernel_size=2))
        self.layer2 = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2))
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.fc = nn.Linear(in_features=32*4*4, out_features=2)
        
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.avg_pool(x)
        x = x.view(-1, 32*4*4) #평탄화
        x = self.fc(x)
        
        return x


## === cell 49
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')


## === cell 50
device


## === cell 51
model = Model().to(device)


## === cell 52
loss_fn = nn.CrossEntropyLoss()


## === cell 53
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)


## === cell 54
len(train_dl)


## === cell 55
epochs = 9

for epoch in range(epochs):
    epoch_loss = 0
    
    for images, labels in train_dl:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        pred = model(images)
        
        loss = loss_fn(pred, labels)
        
        epoch_loss += loss.item() #역전파 수행
        loss.backward()
        
        optimizer.step()
        
    print(f'에폭 [{epoch+1}/{epochs}] - 손실값: {epoch_loss/len(train_dl):.4f}') 


## === cell 56
from sklearn.metrics import roc_auc_score

true_list = []
preds_list = []


## === cell 57
model.eval()

with torch.no_grad(): #성능 검증 시에는 기울기 계산을 비활성화
    for images, labels in valid_dl:
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images)
        preds = torch.softmax(output.cpu(), dim=1)[:, 1] #예측 확률
        true = labels.cpu() #실젯값
        
        preds_list.extend(preds)
        true_list.extend(true)

        
print(f'검증 데이터 ROC AUC: {roc_auc_score(true_list, preds_list):.4f}')


## === cell 58
model.eval()
preds = []

with torch.no_grad():
    for images, _ in test_dl:
        images = images.to(device)
        
        outputs = model(images)
        
        preds_part = torch.softmax(outputs.cpu(), dim=1)[:, 1].tolist()
        
        preds.extend(preds_part)


## === cell 59
test_csv['has_cactus'] = preds
test_csv.to_csv('submission.csv', index=False)
