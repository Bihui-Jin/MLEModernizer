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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

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
scipy==1.15.3
seaborn==0.12.2
sentence-transformers==4.1.0
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
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.50635

# 6. Current score

0.829

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.829) has done: 'I fix the import/runtime blockers so the notebook runs end-to-end and writes a valid `submission.csv` with the exact required columns. Specifically: replace the broken `transformers.AdamW` import (use `torch.optim.AdamW` which you already use), fix Albumentations `Compose`/transforms imports, and make the split stratification valid for multi-label data by stratifying on the single-class label derived from `argmax`. I also make device selection robust (use CUDA if available) and remove the incorrect `/5` ensembling divisor so predictions aren’t artificially shrunk, which should improve ROC AUC compared to the current broken output. Finally, I ensure the test DataLoader isn’t shuffled so submission rows align with `test.csv`.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd

import torch
import torch.utils.data as Data
import torch.nn as nn
from torchvision import models

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.special import softmax
import cv2

from transformers import get_cosine_schedule_with_warmup
from tqdm.notebook import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
import gc

torch.manual_seed(27)
np.random.seed(27)



## === cell 1
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
im_healthy = plt.imread("../input/plant-pathology-2020-fgvc7/images/Train_2.jpg")
im_multi = plt.imread("../input/plant-pathology-2020-fgvc7/images/Train_1.jpg")
im_rust = plt.imread("../input/plant-pathology-2020-fgvc7/images/Train_3.jpg")
im_scab = plt.imread("../input/plant-pathology-2020-fgvc7/images/Train_0.jpg")

plt.subplot(2, 2, 1)
plt.imshow(im_healthy)
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(im_multi)
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(im_rust)
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(im_scab)
plt.axis("off")

plt.show()



## === cell 2
Img_folder = "/kaggle/input/plant-pathology-2020-fgvc7/images/"


def get_path_of_img(filename):
    return Img_folder + filename + ".jpg"


train = pd.read_csv("../input/plant-pathology-2020-fgvc7/train.csv")
test = pd.read_csv("../input/plant-pathology-2020-fgvc7/test.csv")

train.head()



## === cell 3
train["image_path"] = train["image_id"].apply(get_path_of_img)
test["image_path"] = test["image_id"].apply(get_path_of_img)



## === cell 4
from sklearn.model_selection import train_test_split

train_targets = train.loc[:, "healthy":"scab"]
train_paths = train.image_path
test_paths = test.image_path



## === cell 5
stratify_labels = train_targets.values.argmax(1)

train_paths, valid_paths, train_targets, valid_targets = train_test_split(
    train_paths,
    train_targets,
    test_size=0.2,
    random_state=27,
    shuffle=True,
    stratify=stratify_labels,
)



## === cell 6
train_paths.head()



## === cell 7
train_targets.head()



## === cell 8
train_targets.head()



## === cell 9
train_paths.head()



## === cell 10
pass



## === cell 11
img_scab = plt.imread(train_paths.iloc[3])
plt.subplot(1, 1, 1)
plt.imshow(img_scab)
plt.axis("off")
plt.show()




## === cell 12
class Leaf_Dataset(Data.Dataset):
    def __init__(self, image_paths, labels=None, test=False, train=True):
        self.paths = image_paths.reset_index(drop=True)
        self.test = test
        if self.test is False:
            self.labels = labels.reset_index(drop=True)
        self.train = train

        self.train_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(rotate_limit=25.0, p=0.7),
                A.RandomBrightnessContrast(
                    p=0.7, brightness_limit=0.2, contrast_limit=0.2
                ),
                A.OneOf(
                    [
                        A.Sharpen(p=1.0),
                        A.Blur(p=1.0),
                    ],
                    p=0.5,
                ),
                A.Resize(height=224, width=224),
            ]
        )

        self.test_transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(rotate_limit=25.0, p=0.7),
                A.Resize(height=1365, width=1365),
            ]
        )

        self.default_transform = A.Compose(
            [
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    always_apply=True,
                ),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return self.paths.shape[0]

    def __getitem__(self, item):
        img = cv2.imread(self.paths.iloc[item])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.test is False:
            label = torch.tensor(
                int(np.argmax(self.labels.iloc[item].values)), dtype=torch.long
            )

        if self.train is True:
            img = self.train_transform(image=img)["image"]
            img = self.default_transform(image=img)["image"]
        elif self.test is True:
            img = self.test_transform(image=img)["image"]
            img = self.default_transform(image=img)["image"]
        else:
            img = self.default_transform(image=img)["image"]

        if self.test is False:
            return img, label
        return img




## === cell 13
def train_fn(net, loader):
    running_loss = 0.0
    model_predictions = []
    accuracy_labels = []
    pbar = tqdm(total=len(loader), desc="Training")
    net.train()

    for _, (images, labels) in enumerate(loader):
        images, labels = images.to(CFG.device), labels.to(CFG.device)

        optimizer.zero_grad()
        predictions = net(images)
        loss = loss_fn(predictions, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item() * labels.shape[0]
        accuracy_labels = np.concatenate(
            (accuracy_labels, labels.detach().cpu().numpy()), 0
        )
        model_predictions = np.concatenate(
            (model_predictions, np.argmax(predictions.detach().cpu().numpy(), 1)), 0
        )
        pbar.update()

    accuracy = accuracy_score(accuracy_labels, model_predictions)
    pbar.close()
    return running_loss / CFG.train_size, accuracy


def valid_fn(net, loader):
    running_loss = 0.0
    model_predictions = []
    accuracy_labels = []
    pbar = tqdm(total=len(loader), desc="Validation")
    net.eval()

    with torch.no_grad():
        for _, (images, labels) in enumerate(loader):
            images, labels = images.to(CFG.device), labels.to(CFG.device)

            predictions = net(images)
            loss = loss_fn(predictions, labels)

            running_loss += loss.item() * labels.shape[0]
            accuracy_labels = np.concatenate(
                (accuracy_labels, labels.detach().cpu().numpy()), 0
            )
            model_predictions = np.concatenate(
                (model_predictions, np.argmax(predictions.detach().cpu().numpy(), 1)), 0
            )
            pbar.update()

    accuracy = accuracy_score(accuracy_labels, model_predictions)
    conf_matrix = confusion_matrix(accuracy_labels, model_predictions)
    pbar.close()
    return running_loss / CFG.valid_size, accuracy, conf_matrix


def test_fn(net, loader):
    preds_for_output = []
    net.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(loader), desc="Test")
        for _, images in enumerate(loader):
            images = images.to(CFG.device)
            predictions = net(images)
            preds_for_output.append(predictions.detach().cpu().numpy())
            pbar.update()
    pbar.close()
    return np.concatenate(preds_for_output, axis=0)




## === cell 14
class CFG:
    batch_size = 8
    num_epochs = 30
    train_size = train_targets.shape[0]
    valid_size = valid_targets.shape[0]
    model_name = "ResNet18"
    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Fix: avoid crash if CUDA unavailable
    lr = 8e-4




## === cell 15
train_targets.reset_index(drop=True, inplace=True)
train_paths.reset_index(drop=True, inplace=True)
valid_targets.reset_index(drop=True, inplace=True)
valid_paths.reset_index(drop=True, inplace=True)



## === cell 16
len(train_paths)



## === cell 17
train_dataset = Leaf_Dataset(train_paths, labels=train_targets, test=False, train=True)
train_loader = Data.DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=CFG.batch_size,
    num_workers=2,
    pin_memory=True,
)

valid_dataset = Leaf_Dataset(valid_paths, labels=valid_targets, train=False, test=False)
valid_loader = Data.DataLoader(
    valid_dataset,
    shuffle=False,
    batch_size=CFG.batch_size,
    num_workers=2,
    pin_memory=True,
)

test_dataset = Leaf_Dataset(test_paths, labels=None, test=True, train=False)
test_loader = Data.DataLoader(
    test_dataset,
    shuffle=False,
    batch_size=CFG.batch_size,
    num_workers=2,
    pin_memory=True,
)



## === cell 18
len(train_dataset)



## === cell 19
train_paths.head()



## === cell 20
from torchvision.models import resnet18

model = resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 1000, bias=True),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1000, 4, bias=True),
)
model.to(CFG.device)

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=0.001)

num_train_steps = (len(train_dataset) / CFG.batch_size) * CFG.num_epochs
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=(len(train_dataset) / CFG.batch_size) * 5,
    num_training_steps=num_train_steps,
)
loss_fn = torch.nn.CrossEntropyLoss()



## === cell 21
train_loss = []
valid_loss = []
train_acc = []
valid_acc = []



## === cell 22
len(train_loader)



## === cell 23
pass



## === cell 24
pass



## === cell 25
pass



## === cell 26
best_valid_loss = float("inf")
patience = 3
trigger_times = 0

for epoch in range(6):
    tl, ta = train_fn(model, loader=train_loader)
    vl, va, conf_matrix = valid_fn(model, loader=valid_loader)

    train_loss.append(tl)
    valid_loss.append(vl)
    train_acc.append(ta)
    valid_acc.append(va)

    if vl < best_valid_loss:
        best_valid_loss = vl
        torch.save(model.state_dict(), "best_model.pt")
        trigger_times = 0
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1732580866.py in <cell line: 0>()
      6 for epoch in range(6):
      7     tl, ta = train_fn(model, loader=train_loader)
----> 8     vl, va, conf_matrix = valid_fn(model, loader=valid_loader)
      9 
     10     train_loss.append(tl)

/tmp/ipykernel_11/3096385864.py in valid_fn(net, loader)
     38 
     39     with torch.no_grad():
---> 40         for _, (images, labels) in enumerate(loader):
     41             images, labels = images.to(CFG.device), labels.to(CFG.device)
     42 

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

RuntimeError: Caught RuntimeError in DataLoader worker process 1.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 398, in default_collate
    return collate(batch, collate_fn_map=default_collate_fn_map)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 211, in collate
    return [
           ^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 212, in <listcomp>
    collate(samples, collate_fn_map=collate_fn_map)
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 155, in collate
    return collate_fn_map[elem_type](batch, collate_fn_map=collate_fn_map)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 272, in collate_tensor_fn
    return torch.stack(batch, 0, out=out)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: stack expects each tensor to be equal size, but got [3, 1365, 2048] at entry 0 and [3, 2048, 1365] at entry 5


## === cell 27
plt.figure()
plt.ylim(0, 1.5)
sns.lineplot(x=list(range(len(train_loss))), y=train_loss, label="Train Loss")
sns.lineplot(x=list(range(len(valid_loss))), y=valid_loss, label="Valid Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()



## === cell 28
if os.path.exists("best_model.pt"):
    model.load_state_dict(torch.load("best_model.pt", map_location=CFG.device))

out = test_fn(model, test_loader)
output = pd.DataFrame(
    softmax(out, 1), columns=["healthy", "multiple_diseases", "rust", "scab"]
)

sub_eff2 = output.copy()



## === cell 29
sub2 = sub_eff2.copy()
sub2["image_id"] = test.image_id.values
sub2 = sub2[["image_id", "healthy", "multiple_diseases", "rust", "scab"]]
sub2.to_csv("submission_efficientnet2.csv", index=False)



## === cell 30
sub = sub_eff2.copy()
sub["image_id"] = test.image_id.values
sub = sub[["image_id", "healthy", "multiple_diseases", "rust", "scab"]]
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
