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

0.9952

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

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

from sklearn.model_selection import train_test_split
from itertools import product

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device



## === cell 1
torch.cuda.is_available()



## === cell 2
train = pd.read_csv("../input/aerial-cactus-identification/train.csv")
sample = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")

train.shape, sample.shape



## === cell 3
train.head()



## === cell 4
train.info()



## === cell 5
train["has_cactus"].value_counts().plot(kind="pie")



## === cell 6
extra = train[train.has_cactus == 0]
train = pd.concat([train, extra], axis=0).reset_index(drop=True)
train["has_cactus"].value_counts()



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
train_set, val_set = train_test_split(
    train, stratify=train.has_cactus, test_size=0.2, random_state=42
)

len1 = len(train_set)
len2 = len(val_set)

train_dir = "train"
test_dir = "test"

len1, len2, train_dir, test_dir




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
        img_path = os.path.join(
            "../input/aerial-cactus-identification", self.data_dir, name
        )
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(self.labels[index], dtype=torch.float32)




## === cell 10
train_set = dataset_(train_set, train_dir, image_transforms["train"])
val_set = dataset_(val_set, train_dir, image_transforms["test"])



## === cell 11
lst, labels = train_set[0]
lst.shape, labels




## === cell 12
def size(image_size, ker, stri, pad=0):
    return (image_size - ker + 2 * pad) / stri + 1


size(8, 2, 2)




## === cell 13
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

        t = self.out(t)  # logits
        return t




## === cell 14
batch_sizes = 128
lrs = 0.2
train_loss = []
val_loss = []
train_correct = []
val_correct = []
epoch = []

model = Model().to(device)
optimizer = optim.SGD(model.parameters(), lr=lrs)

train_df = DataLoader(
    train_set,
    batch_size=batch_sizes,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
val_df = DataLoader(
    val_set,
    batch_size=batch_sizes,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

for i in range(21):
    model.train()
    total_loss = 0.0
    total_correct = 0

    for images, labels in train_df:
        images = images.to(device)
        labels = labels.to(device).long()

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

    model.eval()
    with torch.no_grad():
        total_val_loss = 0.0
        total_val_correct = 0

        for val_im, val_lab in val_df:
            val_im = val_im.to(device)
            val_lab = val_lab.to(device).long()

            val_preds = model(val_im)
            loss_val = F.cross_entropy(val_preds, val_lab)

            total_val_loss += loss_val.item()
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



## === cell 15
ep = [i for i in range(1, len(train_loss) + 1)]
plt.plot(ep, train_loss, label="train")
plt.plot(ep, val_loss, label="val")
plt.legend()
plt.show()



## === cell 16
plt.plot(ep, train_correct, label="train", color="magenta")
plt.plot(ep, val_correct, label="val", color="royalblue")
plt.legend()
plt.show()




## === cell 17
class dataset_test(torch.utils.data.Dataset):
    def __init__(self, data_directory, transform):
        super().__init__()
        self.data_dir = data_directory
        self.transform = transform
        self.list_id = sorted(os.listdir(self.data_dir))
        self.labels = [0] * len(self.list_id)

    def __len__(self):
        return len(self.list_id)

    def __getitem__(self, index):
        name = self.list_id[index]
        img_path = os.path.join(self.data_dir, name)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(self.labels[index], dtype=torch.float32), name




## === cell 18
test_path = "../input/aerial-cactus-identification/test"
test_ds = dataset_test(test_path, image_transforms["test"])
test_loader = DataLoader(
    test_ds,
    batch_size=batch_sizes,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

pred_rows = []
model.eval()
with torch.no_grad():
    for test_img, _, names in test_loader:
        test_img = test_img.to(device)
        logits = model(test_img)
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # P(class=1)
        for n, p in zip(names, probs):
            pred_rows.append((n, float(p)))

pred_df = pd.DataFrame(pred_rows, columns=["id", "has_cactus"])

submission = sample[["id"]].merge(pred_df, on="id", how="left")
submission["has_cactus"] = submission["has_cactus"].fillna(0.5).clip(0.0, 1.0)

submission.shape, submission.head()



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/3168611697.py in <cell line: 0>()
     12 model.eval()
     13 with torch.no_grad():
---> 14     for test_img, _, names in test_loader:
     15         test_img = test_img.to(device)
     16         logits = model(test_img)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

IsADirectoryError: Caught IsADirectoryError in DataLoader worker process 1.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_11/1884741881.py", line 16, in __getitem__
    img = Image.open(img_path).convert("RGB")
          ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3513, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IsADirectoryError: [Errno 21] Is a directory: '../input/aerial-cactus-identification/test/test'


## === cell 19
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(submission))
print(submission.columns.tolist())
print(submission.head())

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/961202423.py in <cell line: 0>()
      1 # BUGFIX: Kaggle expects a .csv with correct row count and filename is arbitrary but must be .csv.
----> 2 submission.to_csv("submission.csv", index=False)
      3 print("Wrote submission.csv with rows:", len(submission))
      4 print(submission.columns.tolist())
      5 print(submission.head())

NameError: name 'submission' is not defined
