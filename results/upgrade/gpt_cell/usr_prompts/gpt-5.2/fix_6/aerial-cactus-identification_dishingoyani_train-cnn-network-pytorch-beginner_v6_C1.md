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

0.9871

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
%matplotlib inline
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt
from matplotlib import image as img

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.optim as optim

from sklearn.model_selection import train_test_split




## === cell 1
data_dir = "../input/aerial-cactus-identification"            # main dir
train_dir = data_dir + "/train/train"                         # train images folder
test_dir = data_dir + "/test/test"                            # test images folder


## === cell 2
data = pd.read_csv(data_dir + "/train.csv")
data.head()                                                   # test csv


## === cell 3
data["has_cactus"].value_counts()


## === cell 4
plt.hist(data.has_cactus,align='mid',bins=4)
plt.xlim(-1, 2)
plt.xlabel("0: has not cactus and 1: has cactus")
plt.ylabel("Frequency")
plt.title("Number of Images has cactus {1} and without cactus")


## === cell 5
class cactData(Dataset):
    def __init__(self, split_data, data_root = './', transform = None):
        super().__init__()
        self.df = split_data.values                       # set dataframe
        self.data_root = data_root                        # images path
        self.transform = transform                        # transform

    def __len__(self):
        return len(self.df)                               # return total length of dataframe 
    
    def __getitem__(self, index):
        img_name,label = self.df[index]                   # get image id and label from csv
        img_path = os.path.join(self.data_root, img_name) # set image path
        image = img.imread(img_path)                      # read image from given image path
        if self.transform is not None:                    # transform image if transform available 
            image = self.transform(image)
        return image, label                               # return image and image label


## === cell 6
train, valid = train_test_split(data, stratify=data.has_cactus, test_size = 0.2)


## === cell 7
train_transf = transforms.Compose([transforms.ToPILImage(),
                                   transforms.ToTensor()])

valid_transf = transforms.Compose([transforms.ToPILImage(),
                                  transforms.ToTensor()])


## === cell 8
batch_size = 128


## === cell 9
train_data = cactData(train, train_dir, train_transf)
valid_data = cactData(valid, train_dir, valid_transf)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size//2, shuffle=False)


## === cell 10
data_dir = "../input/aerial-cactus-identification"  # main dir
train_dir = os.path.join(data_dir, "train")  # train images folder
test_dir = os.path.join(data_dir, "test")  # test images folder


## === cell 11
class Net(nn.Module): 
    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 10, 3, padding=1)
        self.conv2 = nn.Conv2d(10, 20, 3, padding=1)
        
        self.fc1 = nn.Linear(1280, 640)
        self.fc2 = nn.Linear(640, 2)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        
        return x


## === cell 12
num_epochs = 20
learning_rate = 0.01
momentum = 0.9

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device


## === cell 13
model = Net().to(device)
print(model)


## === cell 14
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=momentum)


## === cell 15
_candidates = [
    os.path.join(train_dir, "train", "train"),  # deeper nesting
    os.path.join(train_dir, "train"),  # single nesting
    train_dir,  # already points to images folder
]

_sample_img_name = (
    str(train.iloc[0]["id"]) if len(train) > 0 else str(data.iloc[0]["id"])
)

_actual_train_img_dir = None
for _c in _candidates:
    if os.path.isdir(_c) and os.path.isfile(os.path.join(_c, _sample_img_name)):
        _actual_train_img_dir = _c
        break

if _actual_train_img_dir is None:
    raise FileNotFoundError(
        "Could not find training images directory containing the expected .jpg files. "
        f"Tried: {', '.join(_candidates)} (looked for '{_sample_img_name}')"
    )

train_data = cactData(train, _actual_train_img_dir, train_transf)
valid_data = cactData(valid, _actual_train_img_dir, valid_transf)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size // 2, shuffle=False)

valid_loss_min = np.Inf

for epoch in range(1, num_epochs + 1):
    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)

    model.eval()
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item() * data.size(0)

    train_loss = train_loss / len(train_loader.sampler)
    valid_loss = valid_loss / len(valid_loader.sampler)

    print(
        "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
            epoch, train_loss, valid_loss
        )
    )

    if valid_loss <= valid_loss_min:
        print(
            "Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...".format(
                valid_loss_min, valid_loss
            )
        )
        torch.save(model.state_dict(), "CNN_model.pt")
        valid_loss_min = valid_loss


## === cell 16
model.load_state_dict(torch.load('CNN_model.pt'))


## === cell 17
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for data, target in valid_loader:
        data = data.to(device)
        target = target.to(device)
        
        outputs = model(data)
        
        _, predicted = torch.max(outputs.data, 1)
        
        total += target.size(0)
        correct += (predicted == target).sum().item()
    print('Test Accuracy: {} %'.format(100 * correct / total))


## === cell 18
df_submission = pd.read_csv(data_dir + "/sample_submission.csv")

test_data = cactData(df_submission, test_dir, transform=valid_transf)
test_loader = DataLoader(dataset = test_data, batch_size=batch_size, shuffle=False)


## === cell 19
model.eval()

pred_label = []
for data, target in test_loader:
    data, target = data.cuda(), target.cuda()
    
    output = model(data)
    
    prob = output[:,1].detach().cpu().numpy()
    for p in prob:
        pred_label.append(p)

df_submission['has_cactus'] = pred_label
df_submission.to_csv('submission.csv', index=False)
