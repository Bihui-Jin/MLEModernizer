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

3.11

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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

0.9835

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import torch
import random
import numpy as np
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
import pandas as pd

data_path = "/kaggle/input/aerial-cactus-identification/"

labels = pd.read_csv(data_path + "train.csv")
submission = pd.read_csv(data_path + "sample_submission.csv")



## === cell 2
from zipfile import ZipFile

work_dir = "/kaggle/working/aerial_cactus_data"
os.makedirs(work_dir, exist_ok=True)

train_dir = os.path.join(work_dir, "train")
test_dir = os.path.join(work_dir, "test")


def _find_dir_with_images(root, target_dir_name):
    """
    Find a directory named `target_dir_name` under `root` that contains .jpg files.
    This handles zip structures like:
      work_dir/train/*.jpg
      work_dir/aerial-cactus-identification/train/*.jpg
      work_dir/aerial-cactus-identification/aerial-cactus-identification/train/*.jpg
    """
    candidates = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == target_dir_name:
            jpgs = [f for f in filenames if f.lower().endswith(".jpg")]
            if len(jpgs) > 0:
                candidates.append((dirpath, len(jpgs)))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]


existing_train = _find_dir_with_images(work_dir, "train")
existing_test = _find_dir_with_images(work_dir, "test")

if existing_train is None:
    with ZipFile(data_path + "train.zip") as zipper:
        zipper.extractall(work_dir)

if existing_test is None:
    with ZipFile(data_path + "test.zip") as zipper:
        zipper.extractall(work_dir)

resolved_train_dir = _find_dir_with_images(work_dir, "train")
resolved_test_dir = _find_dir_with_images(work_dir, "test")

if resolved_train_dir is None or resolved_test_dir is None:
    raise FileNotFoundError(
        "Could not locate extracted train/test image directories with .jpg files under "
        f"{work_dir}. train_dir={resolved_train_dir}, test_dir={resolved_test_dir}"
    )

train_dir = resolved_train_dir
test_dir = resolved_test_dir

print(
    "Resolved train_dir:",
    train_dir,
    "num_files:",
    len([f for f in os.listdir(train_dir) if f.lower().endswith(".jpg")]),
)
print(
    "Resolved test_dir :",
    test_dir,
    "num_files:",
    len([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]),
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1806031641.py in <cell line: 0>()
     47 
     48 if resolved_train_dir is None or resolved_test_dir is None:
---> 49     raise FileNotFoundError(
     50         "Could not locate extracted train/test image directories with .jpg files under "
     51         f"{work_dir}. train_dir={resolved_train_dir}, test_dir={resolved_test_dir}"

FileNotFoundError: Could not locate extracted train/test image directories with .jpg files under /kaggle/working/aerial_cactus_data. train_dir=None, test_dir=None

## === cell 3
from sklearn.model_selection import train_test_split

train, valid = train_test_split(
    labels,
    test_size=0.1,  # train:valid = 9:1
    stratify=labels["has_cactus"],  # keep class ratio
    random_state=50,
)



## === cell 4
print("Number of train data:", len(train))
print("Number of valid data:", len(valid))



## === cell 5
import cv2
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, df, img_dir="./", transform=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

        self.has_label = "has_cactus" in self.df.columns and self.df.shape[1] >= 2

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_id)

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at path: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        if self.has_label:
            label = int(self.df.iloc[idx, 1])
        else:
            label = 0

        return image, label




## === cell 6
from torchvision import transforms

transform = transforms.ToTensor()



## === cell 7
dataset_train = ImageDataset(df=train, img_dir=train_dir, transform=transform)
dataset_valid = ImageDataset(df=valid, img_dir=train_dir, transform=transform)



## === cell 8
from torch.utils.data import DataLoader

loader_train = DataLoader(dataset=dataset_train, batch_size=32, shuffle=True)
loader_valid = DataLoader(dataset=dataset_valid, batch_size=32, shuffle=False)



## === cell 9
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=2
        )

        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)

        self.fc = nn.Linear(in_features=64 * 4 * 4, out_features=2)

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = self.avg_pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        return x




## === cell 10
model = Model().to(device)



## === cell 11
model



## === cell 12
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)



## === cell 13
epochs = 10

for epoch in range(epochs):
    epoch_loss = 0.0

    model.train()
    for images, y in loader_train:
        images = images.to(device)
        y = torch.as_tensor(y, dtype=torch.long, device=device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, y)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"epoch [{epoch+1}/{epochs}] - loss: {epoch_loss/len(loader_train):.4f}")



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/579887166.py in <cell line: 0>()
      5 
      6     model.train()
----> 7     for images, y in loader_train:
      8         images = images.to(device)
      9         y = torch.as_tensor(y, dtype=torch.long, device=device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_11/3011800194.py in __getitem__(self, idx)
     22         image = cv2.imread(img_path)
     23         if image is None:
---> 24             raise FileNotFoundError(f"Failed to read image at path: {img_path}")
     25 
     26         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

FileNotFoundError: Failed to read image at path: /kaggle/working/aerial_cactus_data/train/564f88e2e335a2f86376199f549fe85d.jpg

## === cell 14
from sklearn.metrics import roc_auc_score

true_list = []
preds_list = []

model.eval()
with torch.no_grad():
    for images, y in loader_valid:
        images = images.to(device)
        y = torch.as_tensor(y, dtype=torch.long)

        outputs = model(images)
        preds = torch.softmax(outputs.cpu(), dim=1)[:, 1]
        true_list.extend(y.numpy().tolist())
        preds_list.extend(preds.numpy().tolist())

print(f"ROC AUC of validation data : {roc_auc_score(true_list, preds_list):.4f}")



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3145281293.py in <cell line: 0>()
      6 model.eval()
      7 with torch.no_grad():
----> 8     for images, y in loader_valid:
      9         images = images.to(device)
     10         y = torch.as_tensor(y, dtype=torch.long)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_11/3011800194.py in __getitem__(self, idx)
     22         image = cv2.imread(img_path)
     23         if image is None:
---> 24             raise FileNotFoundError(f"Failed to read image at path: {img_path}")
     25 
     26         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

FileNotFoundError: Failed to read image at path: /kaggle/working/aerial_cactus_data/train/0eb5c8e2ece103d136cfd51ccc891c62.jpg

## === cell 15
dataset_test = ImageDataset(df=submission, img_dir=test_dir, transform=transform)
loader_test = DataLoader(dataset=dataset_test, batch_size=32, shuffle=False)



## === cell 16
model.eval()

preds = []
with torch.no_grad():
    for images, _ in loader_test:
        images = images.to(device)
        outputs = model(images)
        preds_part = torch.softmax(outputs.cpu(), dim=1)[:, 1].tolist()
        preds.extend(preds_part)

len(preds), len(submission)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1140104286.py in <cell line: 0>()
      3 preds = []
      4 with torch.no_grad():
----> 5     for images, _ in loader_test:
      6         images = images.to(device)
      7         outputs = model(images)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_11/3011800194.py in __getitem__(self, idx)
     22         image = cv2.imread(img_path)
     23         if image is None:
---> 24             raise FileNotFoundError(f"Failed to read image at path: {img_path}")
     25 
     26         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

FileNotFoundError: Failed to read image at path: /kaggle/working/aerial_cactus_data/test/09034a34de0e2015a8a28dfe18f423f6.jpg

## === cell 17
preds[:5]



## === cell 18
if len(preds) != len(submission):
    raise ValueError(
        f"Pred length ({len(preds)}) does not match submission rows ({len(submission)})."
    )

submission["has_cactus"] = preds
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/428728180.py in <cell line: 0>()
      1 if len(preds) != len(submission):
----> 2     raise ValueError(
      3         f"Pred length ({len(preds)}) does not match submission rows ({len(submission)})."
      4     )
      5 

ValueError: Pred length (0) does not match submission rows (3325).

## === cell 19
import shutil

print("Done.")
