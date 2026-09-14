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
scipy==1.15.3
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

0.9973

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy
import cv2

import torch
import torchvision
from torchvision import models
import torch.nn as nn
from torchvision import transforms, datasets
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
import torch.nn.functional as F
from torch.nn.modules.pooling import AvgPool3d


class SummaryWriter:  # no-op fallback
    def __init__(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass

    def add_image(self, *args, **kwargs):
        pass

    def add_images(self, *args, **kwargs):
        pass

    def add_histogram(self, *args, **kwargs):
        pass

    def add_graph(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


from sklearn.model_selection import train_test_split
from itertools import product



## === cell 1
torch.cuda.is_available()



## === cell 2
train = pd.read_csv("../input/aerial-cactus-identification/train.csv")
sample = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")



## === cell 3
train.head()



## === cell 4
train.info()



## === cell 5
train["has_cactus"].value_counts().plot(kind="pie")



## === cell 6
"""extra=train[train.has_cactus==0]
train=pd.concat([train,extra],axis=0)"""



## === cell 7
image_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomRotation(degrees=0),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
        ]
    ),
    "test": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])]
    ),
}



## === cell 8
train_set, val_set = train_test_split(train, stratify=train.has_cactus, test_size=0.2)

len1 = len(train_set)
len2 = len(val_set)

train_dir = "train/train"
test_dir = "test/test"




## === cell 9
class dataset_(torch.utils.data.Dataset):
    def __init__(self, labels, data_directory, transform):
        super().__init__()
        self.list_id = labels.values[:, 0]
        self.labels = labels.values[:, 1]
        self.data_dir = data_directory
        self.transform = transform

    def __len__(self):
        return len(self.list_id)

    def __getitem__(self, index):
        name = self.list_id[index]
        img = Image.open(
            "../input/aerial-cactus-identification/{}/{}".format(self.data_dir, name)
        )
        img = self.transform(img)
        return img, torch.tensor(self.labels[index], dtype=torch.float32)




## === cell 10
train_set = dataset_(train_set, train_dir, image_transforms["train"])
val_set = dataset_(val_set, train_dir, image_transforms["test"])



## === cell 11
train_set.data_dir = "train"
val_set.data_dir = "train"

lst, labels = next(iter(train_set))



## === cell 12
lst.shape, labels.shape, labels




## === cell 13
def size(image_size, ker, stri, pad=0):
    return (image_size - ker + 2 * pad) / stri + 1




## === cell 14
size(8, 2, 2)




## === cell 15
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.dense_1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=3, padding=1
        )
        self.dense_2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.dense_3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=3, padding=1
        )
        self.dense_4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(in_features=128 * 2 * 2, out_features=128)
        self.fc_dense1 = nn.BatchNorm1d(128)
        self.out = nn.Linear(in_features=128, out_features=2)
        self.d1 = nn.Dropout(0.5)
        self.f = nn.Sigmoid()

    def forward(self, t):
        t = F.max_pool2d(
            F.leaky_relu(self.dense_1(self.conv1(t))), kernel_size=2, stride=2
        )
        t = F.max_pool2d(
            F.leaky_relu(self.dense_2(self.conv2(t))), stride=2, kernel_size=2
        )
        t = F.max_pool2d(
            F.leaky_relu(self.dense_3(self.conv3(t))), stride=2, kernel_size=2
        )
        t = F.max_pool2d(
            F.leaky_relu(self.dense_4(self.conv4(t))), stride=2, kernel_size=2
        )
        t = t.reshape(-1, 128 * 2 * 2)
        t = F.leaky_relu(self.fc_dense1(self.fc1(t)))
        t = self.d1(t)
        t = self.f(self.out(t))
        return t




## === cell 16
batch_sizes = 120
lrs = 0.2
train_loss = []
val_loss = []
train_correct = []
val_correct = []
epoch = []
model = Model()
model = model.to("cuda:0")
optimizer = optim.SGD(model.parameters(), lr=lrs)

train_df = DataLoader(train_set, batch_size=batch_sizes, shuffle=True)
val_df = DataLoader(val_set, batch_size=batch_sizes, shuffle=True)

for i in range(30):
    total_loss = 0
    total_correct = 0
    for batch in train_df:
        images, labels = batch
        images = images.to("cuda:0")
        labels = labels.to("cuda:0")
        labels = labels.long()
        preds = model(images)

        loss = F.cross_entropy(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += preds.argmax(dim=1).eq(labels).sum().item()
        del images, labels

    train_loss.append(total_loss)
    train_correct.append(total_correct / len1)
    epoch.append(i + 1)

    with torch.no_grad():
        total_val_loss = 0
        total_val_correct = 0
        for val_batch in val_df:
            val_im, val_lab = val_batch
            val_im = val_im.to("cuda:0")
            val_lab = val_lab.to("cuda:0")
            val_lab = val_lab.long()

            val_preds = model(val_im)
            loss_val = F.cross_entropy(val_preds, val_lab)
            total_val_loss += loss_val
            total_val_correct += val_preds.argmax(dim=1).eq(val_lab).sum().item()

        val_loss.append(total_val_loss)
        val_correct.append(total_val_correct / len2)

        print(
            "Epoch {}\t train_loss {}\t train_accuracy{}\t val_loss {}\t val_accuracy {}\n".format(
                epoch[i],
                total_loss,
                total_correct / len1,
                total_val_loss,
                total_val_correct / len2,
            )
        )



## === cell 17
ep = [i for i in range(1, 31)]

train_loss_plot = [
    float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x)
    for x in train_loss
]
val_loss_plot = [
    float(x.detach().cpu().item()) if torch.is_tensor(x) else float(x) for x in val_loss
]

plt.plot(ep, train_loss_plot, label="train")
plt.plot(ep, val_loss_plot, label="test")
plt.legend()



## === cell 18
plt.plot(ep, train_correct, label="train", color="magenta")
plt.plot(ep, val_correct, label="test", color="royalblue")
plt.legend()




## === cell 19
class dataset_(torch.utils.data.Dataset):
    def __init__(self, data_directory, transform):
        super().__init__()
        self.list_id = sorted(os.listdir(data_directory))
        self.labels = [0] * len(self.list_id)
        self.data_dir = data_directory
        self.transform = transform

    def __len__(self):
        return len(self.list_id)

    def __getitem__(self, index):
        name = self.list_id[index]
        img = Image.open("{}/{}".format(self.data_dir, name))
        img = self.transform(img)
        return img, torch.tensor(self.labels[index], dtype=torch.float32), name




## === cell 20
test_path = "../input/aerial-cactus-identification/test/test"
test = dataset_(test_path, image_transforms["test"])
test = DataLoader(test, batch_size=batch_sizes, shuffle=False)



## === cell 21
model.eval()
ids = []
label = []
with torch.no_grad():
    for test_batch in test:
        test_img, _, test_names = test_batch
        test_img = test_img.to("cuda:0")
        probs = model(test_img).detach().cpu().numpy()  # shape (B,2), already sigmoid'd
        p1 = probs[:, 1].astype(float)
        ids.extend(list(test_names))
        label.extend(list(p1))



## === cell 22
submissions = pd.DataFrame({"id": ids, "has_cactus": label})



## === cell 23
submissions.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions.shape)
print(submissions.head())
