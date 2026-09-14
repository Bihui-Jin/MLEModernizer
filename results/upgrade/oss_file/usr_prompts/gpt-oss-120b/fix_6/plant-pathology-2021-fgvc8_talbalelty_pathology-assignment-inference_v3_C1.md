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
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.738670360110804

# 6. Current score

0.24762

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.29148) has done: 'I fix the missing model checkpoint by loading a pretrained ResNet‑50 and adjusting its final layer when the checkpoint file is unavailable. I also replace the deprecated `DataFrame.append` with `pd.concat` to build the submission DataFrame correctly. These changes remove the runtime errors and ensure a valid `submission.csv` is written, while keeping the original modeling approach unchanged.'
- What this solution (achieved 0.23784) has done: 'I add a lightweight training phase before generating predictions. The model architecture stays the same, but the network is now fine‑tuned on the training set (with a small train/validation split and a few epochs). This should raise the mean F1‑score from the current ~0.29 toward the target while keeping the original inference pipeline unchanged.'
- What this solution (achieved 0.20291) has done: 'I fixed the crash that prevented training by correcting label handling in the `PlantDataSet` class. The original code tried to call `.astype` on a Python list, causing an AttributeError inside the DataLoader workers. Now the labels are extracted as a pandas Series, converted to a NumPy array and then to a `torch.float32` tensor, which works for both training and validation loaders. This change restores the training loop, allowing the model to learn and the final F1‑score to move toward the target while keeping all other logic unchanged.'
- What this solution (achieved 0.22713) has done: 'I fixed the label‑tensor conversion error by dropping the original string‑based `labels` column when building the label tensors in the dataset, and I extended training to ten epochs so the model can learn enough to raise the F1 score toward the target. These changes keep the original architecture and workflow intact while ensuring the script runs end‑to‑end and outputs a valid `submission.csv`.'
- What this solution (achieved 0.24762) has done: 'I fixed the label conversion error by explicitly casting the label series to a NumPy float array before creating the tensor, and changed the evaluation threshold to 0.6 (matching the submission rule) to improve F1. I also extended training to 15 epochs for a modest boost while keeping the core model unchanged.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import glob
import matplotlib.pyplot as plt
import gc
import albumentations as A
import torchmetrics
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim



## === cell 1
torch.cuda.empty_cache()
gc.collect()
DEBUG = False
DIMENTION = (256, 171)  # image size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_data = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
test_images_path = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_images_names = glob.glob(test_images_path + "*.jpg")

train_data["labels"] = train_data["labels"].apply(lambda string: string.split(" "))
s = list(train_data["labels"])
mlb = MultiLabelBinarizer()
train_labels = pd.DataFrame(
    mlb.fit_transform(s), columns=mlb.classes_, index=train_data.index
)  # one‑hot encoding
labels_size = len(train_labels.columns)

train_df = train_data.copy()
for col in train_labels.columns:
    train_df[col] = train_labels[col].values




## === cell 2
class PlantDataSet(Dataset):
    def __init__(self, dataset, images_path, transform=None):
        super(PlantDataSet, self).__init__()
        self.dataset = dataset
        self.images_path = images_path
        self.transform = transform

    def __getitem__(self, idx):
        if self.images_path is not None:
            image = cv2.imread(self.images_path + self.dataset.image[idx])
            label_series = self.dataset.loc[
                idx, self.dataset.columns.difference(["image", "labels"])
            ]
            labels = torch.tensor(
                label_series.to_numpy(dtype=np.float32), dtype=torch.float32
            )
        else:
            image = cv2.imread(self.dataset[idx])
            labels = torch.tensor([], dtype=torch.float32)  # no labels for test data

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, DIMENTION)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, labels

    def __len__(self):
        return len(self.dataset)




## === cell 3
transform = A.Compose(
    [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
)

test_dataset = PlantDataSet(test_images_names, None, transform)
BS = 30
plants_test_data_loader = DataLoader(dataset=test_dataset, batch_size=BS, shuffle=False)




## === cell 4
def test(test_dataloader, model):
    predictions = None
    model.eval()
    model = model.to(device)
    with torch.no_grad():
        for i, (images, _) in enumerate(test_dataloader):
            images = images.float().to(device)
            output = model(images)
            probabilities = torch.sigmoid(output)
            if i == 0:
                predictions = probabilities.detach().cpu().numpy()
            else:
                predictions = np.concatenate(
                    (predictions, probabilities.detach().cpu().numpy()), axis=0
                )
            del images
            torch.cuda.empty_cache()
            gc.collect()
    return np.array(predictions)




## === cell 5
def create_submission(test_images_path, predictions):
    submission_df = pd.DataFrame(columns=["image", "labels"])
    rows = []
    for image_name, prediction in zip(test_images_path, predictions):
        name = image_name.split("/")[-1]
        arr = [
            label for pred, label in zip(prediction, train_labels.columns) if pred > 0.6
        ]
        if len(arr) == 0:
            arr = ["healthy"]
        prediction_labels = " ".join(arr)
        rows.append({"image": name, "labels": prediction_labels})
    submission_df = pd.concat([submission_df, pd.DataFrame(rows)], ignore_index=True)
    submission_df.to_csv("submission.csv", index=False)




## === cell 6
train_split, val_split = train_test_split(train_df, test_size=0.2, random_state=42)

train_dataset = PlantDataSet(
    train_split, "/kaggle/input/plant-pathology-2021-fgvc8/train_images/", transform
)
val_dataset = PlantDataSet(
    val_split, "/kaggle/input/plant-pathology-2021-fgvc8/train_images/", transform
)

train_loader = DataLoader(
    train_dataset, batch_size=BS, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BS, shuffle=False, num_workers=2, pin_memory=True
)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images = images.float().to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, loader, threshold=0.6):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.float().to(device)
            labels = labels.float()
            outputs = model(images)
            probs = torch.sigmoid(outputs).cpu()
            all_preds.append(probs)
            all_labels.append(labels)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_labels)
    preds_bin = (preds > threshold).int()
    f1 = torchmetrics.F1Score(num_classes=labels_size, average="samples")
    return f1(preds_bin, targets.int()).item()




## === cell 7
try:
    resnet50 = models.resnet50(pretrained=False, num_classes=labels_size)
    resnet50.load_state_dict(
        torch.load("../input/resnet50-final/resnet50_final.pth", map_location=device)
    )
except FileNotFoundError:
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = nn.Linear(resnet50.fc.in_features, labels_size)

resnet50 = resnet50.to(device)



## === cell 8
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(resnet50.parameters(), lr=1e-4)

EPOCHS = 15  # a few more epochs for better learning
for epoch in range(1, EPOCHS + 1):
    train_loss = train_one_epoch(resnet50, train_loader, criterion, optimizer)
    val_f1 = evaluate(resnet50, val_loader, threshold=0.6)
    print(
        f"Epoch {epoch}/{EPOCHS} - Train loss: {train_loss:.4f} - Val F1: {val_f1:.4f}"
    )



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/18077304.py in <cell line: 0>()
      4 EPOCHS = 15  # a few more epochs for better learning
      5 for epoch in range(1, EPOCHS + 1):
----> 6     train_loss = train_one_epoch(resnet50, train_loader, criterion, optimizer)
      7     val_f1 = evaluate(resnet50, val_loader, threshold=0.6)
      8     print(

/tmp/ipykernel_55/647552361.py in train_one_epoch(model, loader, criterion, optimizer)
     19     model.train()
     20     running_loss = 0.0
---> 21     for images, labels in loader:
     22         images = images.float().to(device)
     23         labels = labels.float().to(device)

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

KeyError: Caught KeyError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py", line 3805, in get_loc
    return self._engine.get_loc(casted_key)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 2606, in pandas._libs.hashtable.Int64HashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 2630, in pandas._libs.hashtable.Int64HashTable.get_item
KeyError: 2619

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/158975588.py", line 10, in __getitem__
    image = cv2.imread(self.images_path + self.dataset.image[idx])
                                          ~~~~~~~~~~~~~~~~~~^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pandas/core/series.py", line 1121, in __getitem__
    return self._get_value(key)
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pandas/core/series.py", line 1237, in _get_value
    loc = self.index.get_loc(label)
          ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py", line 3812, in get_loc
    raise KeyError(key) from err
KeyError: 2619


## === cell 9
test_predictions = test(plants_test_data_loader, resnet50)
create_submission(test_images_names, test_predictions)
