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

0.9944

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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image

import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None

from sklearn.model_selection import train_test_split




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/tensorboard/compat/__init__.py in tf()
     41     try:
---> 42         from tensorboard.compat import notf  # noqa: F401
     43     except ImportError:

ImportError: cannot import name 'notf' from 'tensorboard.compat' (/usr/local/lib/python3.11/dist-packages/tensorboard/compat/__init__.py)

During handling of the above exception, another exception occurred:

AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "../input/aerial-cactus-identification"

train_csv = os.path.join(BASE_PATH, "train.csv")
sample_submission = os.path.join(BASE_PATH, "sample_submission.csv")

train_df_full = pd.read_csv(train_csv)
sample_df = pd.read_csv(sample_submission)




## === cell 3
image_transforms = {
    "train": transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2]),
        ]
    ),
    "test": transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.2, 0.2, 0.2])]
    ),
}




## === cell 4
train_df, val_df = train_test_split(
    train_df_full, stratify=train_df_full["has_cactus"], test_size=0.2, random_state=42
)

train_dir = os.path.join(BASE_PATH, "train")
test_dir = os.path.join(BASE_PATH, "test")




## === cell 5
class CactusDataset(Dataset):
    """Dataset for training / validation."""

    def __init__(self, df, data_dir, transform):
        """
        df: DataFrame with columns ['id', 'has_cactus']
        data_dir: directory that holds the image files
        transform: torchvision transforms
        """
        self.ids = df["id"].values
        self.labels = df["has_cactus"].values.astype(np.float32)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = self.ids[idx]
        img_path = os.path.join(self.data_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img, label




## === cell 6
batch_size = 128
train_dataset = CactusDataset(train_df, train_dir, image_transforms["train"])
val_dataset = CactusDataset(val_df, train_dir, image_transforms["test"])

train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)




## === cell 7
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 2 * 2, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)

        self.out = nn.Linear(128, 2)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.leaky_relu(self.bn2(self.conv2(x))), 2)
        x = F.max_pool2d(F.leaky_relu(self.bn3(self.conv3(x))), 2)
        x = F.max_pool2d(F.leaky_relu(self.bn4(self.conv4(x))), 2)
        x = x.view(x.size(0), -1)  # flatten
        x = F.leaky_relu(self.bn_fc(self.fc1(x)))
        x = self.dropout(x)
        logits = self.out(x)  # shape (batch, 2)
        return logits




## === cell 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
num_epochs = 15

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []




## === cell 9
for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    correct = 0

    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device).long()  # CrossEntropy expects long

        optimizer.zero_grad()
        logits = model(imgs)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()

    train_losses.append(epoch_loss / len(train_dataset))
    train_accuracies.append(correct / len(train_dataset))

    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            targets = targets.to(device).long()
            logits = model(imgs)
            loss = F.cross_entropy(logits, targets)
            val_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            val_correct += (preds == targets).sum().item()

    val_losses.append(val_loss / len(val_dataset))
    val_accuracies.append(val_correct / len(val_dataset))

    print(
        f"Epoch {epoch:02d} | "
        f"Train loss {train_losses[-1]:.4f} | Train acc {train_accuracies[-1]:.4f} | "
        f"Val loss {val_losses[-1]:.4f} | Val acc {val_accuracies[-1]:.4f}"
    )




## === cell 11
class TestDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.ids = sorted(os.listdir(data_dir))  # keep deterministic order
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_name = self.ids[idx]
        img_path = os.path.join(self.data_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, img_name




## === cell 12
test_dataset = TestDataset(test_dir, image_transforms["test"])
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
)




## === cell 13
model.eval()
predictions = []

with torch.no_grad():
    for imgs, img_names in test_loader:
        imgs = imgs.to(device)
        logits = model(imgs)  # (batch, 2)
        probs = torch.softmax(logits, dim=1)[:, 1]  # probability of class 1
        probs = probs.cpu().numpy()
        for name, p in zip(img_names, probs):
            predictions.append((name, float(p)))




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_54/52560707.py in <cell line: 0>()
      3 
      4 with torch.no_grad():
----> 5     for imgs, img_names in test_loader:
      6         imgs = imgs.to(device)
      7         logits = model(imgs)  # (batch, 2)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1453                 data = self._task_info.pop(self._rcvd_idx)[1]
   1454                 self._rcvd_idx += 1
-> 1455                 return self._process_data(data)
   1456 
   1457             assert not self._shutdown and self._tasks_outstanding > 0

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
  File "/tmp/ipykernel_54/1296226955.py", line 14, in __getitem__
    img = Image.open(img_path).convert("RGB")
          ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/PIL/Image.py", line 3513, in open
    fp = builtins.open(filename, "rb")
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
IsADirectoryError: [Errno 21] Is a directory: '../input/aerial-cactus-identification/test/test'


## === cell 14
sub_df = pd.DataFrame(predictions, columns=["id", "has_cactus"])
sub_df = sub_df.set_index("id").loc[sample_df["id"]].reset_index()

submission_path = "submission.csv"
sub_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with {len(sub_df)} rows.")

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_54/2913170909.py in <cell line: 0>()
      2 sub_df = pd.DataFrame(predictions, columns=["id", "has_cactus"])
      3 # Ensure ordering matches sample submission (optional)
----> 4 sub_df = sub_df.set_index("id").loc[sample_df["id"]].reset_index()
      5 
      6 # Write to CSV with correct filename

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1418                     raise ValueError("Cannot index with multidimensional key")
   1419 
-> 1420                 return self._getitem_iterable(key, axis=axis)
   1421 
   1422             # nested tuple slicing

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_iterable(self, key, axis)
   1358 
   1359         # A collection of keys
-> 1360         keyarr, indexer = self._get_listlike_indexer(key, axis)
   1361         return self.obj._reindex_with_indexers(
   1362             {axis: [keyarr, indexer]}, copy=True, allow_dups=True

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _get_listlike_indexer(self, key, axis)
   1556         axis_name = self.obj._get_axis_name(axis)
   1557 
-> 1558         keyarr, indexer = ax._get_indexer_strict(key, axis_name)
   1559 
   1560         return keyarr, indexer

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['fc9ff5384d669488775b9414d0a6d056.jpg', 'faa2325b8863647759bcc3aaa997d42f.jpg', 'f8c13ac12a157cff3526e690acb7c747.jpg', 'f7fc39475887c1f43e3964fcbdbd28b7.jpg', 'fe7283fbeae6e177d3a7195ad04194c8.jpg', 'ff1d1fb5ea11766976d640d2f03112cb.jpg', 'f90be5e5693143979fb400cecc134da4.jpg', 'f8eea4be2f46b528580b39126a7af8a2.jpg', 'f72d69f9a0f281d607beea8887c858aa.jpg', 'fc518358f5714ed0024dac558ea37ce0.jpg', 'ff9170268788ec65add31cd3ba8f92d0.jpg', 'f8988f4497f2c77d20848cafc4c4d0e8.jpg', 'f7d91277923e6302b9c7baa62919fd8d.jpg', 'f7402180b19e1fe0ceb5f05c934f3de7.jpg', 'f8bb2f6d792c94fff0da8bc5ad0af4c6.jpg', 'fad56c392fd77469ebd42dcdf5860372.jpg', 'fe0cc314f3f944c69d2b156d0b5bde36.jpg', 'fbd51e7dbb31cee4b2e11ee20cd133ef.jpg', 'f964be8da498ec4e98155a331367f689.jpg', 'fa1737647c5d5448ffc61a8754869718.jpg', 'fa419239d84612100f14d1995ee89460.jpg', 'f8c0a1c6be093c9519d7727240dac1a3.jpg', 'f9e2e3793272f02e39edb9cad5908097.jpg', 'f7f528247bb83b7e4c2da042a513d192.jpg', 'f84aea0abfe86c1b5581cb72a00f07f6.jpg', 'fe1dea46072dbb3c0a734d81f8dddcca.jpg', 'fdd57c8a3c9c597e00b4b77063228845.jpg', 'fed0dd8c55d00450e3764e06e5de20b3.jpg', 'fb7ac0f8dc940e1783cc040cb9fb07e4.jpg', 'faaf3d6a475538dce5102ae27169bccb.jpg', 'febdcba6db739c5625c835a7903bfbd2.jpg', 'fc9c1d26bc0dd913f123eb2aa493be99.jpg', 'f74d8743d2f204459de5104a823f6b2d.jpg', 'fda9a27561a06f32a3a4779686a04ebf.jpg', 'fe05e3ee2539542cf8903d4078e149c1.jpg', 'f98e5056ceaf74f2d38e973fd9c1d2de.jpg', 'fa4411b08a8efff184ad4e71abbac8a5.jpg', 'fcb2382a9cc6f542b7a5f3f3bd6643ee.jpg', 'fcf9701cd7516a03586e27a814a6298c.jpg', 'ff5d58dc290f1aea81ba970e13fec2e4.jpg', 'fa6d2ac34760c16857673a7d9893c26f.jpg', 'fe6e1310e257d70c2d6b43860b97b727.jpg', 'fa68e2f4276a7a98254a12508dcd4fc4.jpg', 'ff9da08031f447c2fff7766461e7c2dd.jpg', 'f7b72b7d34ce6d45ea9dd39f8724b254.jpg', 'f7e69cf623c5d56ccf97f4b29b27542c.jpg', 'fa66410cdfd2d783d37259c94c91ef92.jpg', 'f972952c6b12ca201576b7629d68abb9.jpg', 'ffded2566b32d90c8fc9fe116ce1ba6f.jpg', 'fa4f01648221978a9b541292fc56d7a9.jpg', 'fa3851e9fed24ea8227fde7d7b4fb8ee.jpg', 'f935e2a659456efc097b313b788bf96e.jpg', 'f852db0e5fcded7e80be3a8b3993a66d.jpg', 'fadbb7b2788f493f873ffd1051df5a75.jpg', 'f80cecf09a79c46bb7dde8820949c257.jpg', 'f8eb52e677503af437059d79b30f2b5e.jpg', 'f797513ac834e90933b4715ebaf4a036.jpg', 'fea1a0158685999512b6740eec728b67.jpg', 'fceadb4256d97f83935f647d4681b7f2.jpg', 'fe47e3ee358f9811ca0c6acf94da14e1.jpg', 'ffeafc3bd716a83bc93014fdb0ef53fa.jpg', 'fc2411e8fb3624250c3c944f53288f5c.jpg', 'f949913facf5104c5638064f916f0951.jpg', 'f8f82bbdda7797ac488055c815e228b0.jpg', 'f9fad90e5d9b9aef793795b1e4c4048f.jpg', 'fbcf0e04e5b0c1e171addce8dfd05fbe.jpg', 'fd18704c4ae2f56c6663028cfd21a074.jpg', 'f8dc18499bfaf63b51e8753bcca5d6d4.jpg', 'f829072d501e71cdb87ef6d01538155b.jpg', 'fe66386d537b74dfb34a1130fbf449af.jpg', 'f8996fd180eff590653a572a11a3fdd6.jpg', 'fa677deff72193791cf83b1072e46042.jpg', 'fbe7e3e00eb7ecdef514685fa488599f.jpg', 'ff1c431c7626fb1474b9f2d329b73efd.jpg', 'f6ced4ff4a69689fc693c0cb5775f309.jpg', 'fc9bd6085c176e10f5d8f8c3c09b801b.jpg', 'fd3003b28cf7c755556e8967d9737df2.jpg', 'f9d9f3b30938aa0827d9ccd7001f11b4.jpg', 'f8035e48f5325a100aa16ca0c9109473.jpg', 'f6f7a1b0d29056d73c0db9f018c8ad69.jpg', 'fe8d5be6bcf70fe6eb6a978501073930.jpg', 'fece53159e0bd3d975fc977d63193e02.jpg', 'fe70200913c77c1c60ebbdb5ad56c717.jpg', 'ff2e711e4aa5b239e19a04b35ffad866.jpg', 'fbfdd1026908929e68c7af75dad86075.jpg', 'fa03c6f0290dff9b54845e93c1593ba4.jpg', 'ffcd13809e02dcda609dc47202f1fd0e.jpg', 'ffb3b4cf5fca27d68a599a8ad8c26ec6.jpg', 'f6e15117bc8fdf53d81e045adfcf1c7f.jpg', 'f872e73609fba46c5933b59475bba5cd.jpg', 'fb4c675da9c6def8ccd03c237d25429f.jpg', 'f7092cd665aaacca617dfd5d3e7e74df.jpg', 'fa8c263e0789c8f6c33a013c1c5f4380.jpg', 'fc5d9ce43a6452aded30a6511dfb87c7.jpg', 'ff034cde9f7f7afdc350b48f4ded19bf.jpg', 'f9146266c416dcb08d31ce69b1a2a666.jpg', 'fb3b8321139c3239554f9e1e43577b44.jpg', 'faf736b507f747a0362653805647e41c.jpg', 'fefb9242121004a9e18f52f4260b9d21.jpg', 'f8be95326a912688ebd53f418b718222.jpg', 'f87f70d4b86b68e41d7350356be7fe19.jpg', 'fd8033ab2c4d18d4916ae5a069b40692.jpg', 'f72471758944fa0eb9745ad23dd2ebca.jpg', 'fb569ff2f35a6732b427af3d0e89d660.jpg', 'f7c1b69a6f6cfbd9f176c60ebc11a280.jpg', 'f95699320cdf107aa74c01862fa4372e.jpg', 'f82c768ebb91bc88e76821aea2696baa.jpg', 'fa825ee1eaf6d73715cb2558743080ac.jpg', 'f9b9e68bd7d5ad4f509a0219f20d42a1.jpg', 'ff9b5f66dfa0f316b167084f0108c189.jpg', 'fee9ec0a3cfc4511ad33a44b2a44a5ba.jpg', 'ff6cfd649d25b31ff9219eb89756667c.jpg', 'fe484a0dc0feba254060b898d193d117.jpg', 'f939efe441da93005b27817b2f4ea5c6.jpg', 'fffd9e9b990eba07c836745d8aef1a3a.jpg', 'fbb058e700614038e04b48aa16703964.jpg', 'fd98445d7c523c87b080563c35db1895.jpg', 'ff9ab24ffb1968d9ddf19f737fa40958.jpg', 'f8b3097d93e9408f5f96bcce396c056f.jpg', 'fc4c0cdcfc753e852330f0a8e3473461.jpg', 'f6e801012f615d70aa6b613369e65c0c.jpg', 'f814d8da32403db51f3012c725da25fc.jpg', 'faf087f586c918b3efcc4d69ba5ed259.jpg', 'fe8eb3887f640174697382ed6ab9bb50.jpg', 'fbf74e29c817774a096f292caff9118b.jpg'] not in index"
