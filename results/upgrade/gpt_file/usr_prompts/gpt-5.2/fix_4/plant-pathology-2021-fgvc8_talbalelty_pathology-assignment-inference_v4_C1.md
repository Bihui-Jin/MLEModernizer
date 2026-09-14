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

0.6398707294552166

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.24507) has done: 'I (1) fix the missing weights path by falling back to a built-in torchvision pretrained ResNet50 and adapting the final layer to the competition’s label count so inference can run end-to-end. I (2) fix the pandas 2.x `DataFrame.append` crash by building rows in a list and creating the submission DataFrame once. I (3) make the test dataset robust by ensuring it returns a dummy label tensor and normalizes types consistently. These are execution/stability fixes; they should also produce a reasonable baseline score without changing the model’s core (ResNet50 + sigmoid + thresholded multilabel output).'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import glob
import matplotlib.pyplot as plt
import gc
import albumentations as A
import torchmetrics  # noqa: F401
import seaborn as sns  # noqa: F401
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torchvision.models as models
import os
import random



## === cell 1
torch.cuda.empty_cache()
gc.collect()
DEBUG = False

DIMENSION_HW = (256, 171)  # (H, W)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

train_data = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
test_images_path = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_images_names = sorted(glob.glob(test_images_path + "*.jpg"))

train_data["labels"] = train_data["labels"].apply(lambda string: string.split(" "))
s = list(train_data["labels"])
mlb = MultiLabelBinarizer()
train_labels = pd.DataFrame(
    mlb.fit_transform(s), columns=mlb.classes_, index=train_data.index
)
labels_size = len(train_labels.columns)

train_df = pd.concat([train_data[["image"]], train_labels], axis=1)




## === cell 2
class PlantDataSet(Dataset):
    def __init__(self, dataset, images_path, transform=None):
        super(PlantDataSet, self).__init__()
        self.dataset = dataset
        self.images_path = images_path
        self.transform = transform

    def __getitem__(self, idx):
        if self.images_path is not None:
            img_path = os.path.join(self.images_path, self.dataset.image.iloc[idx])
            image = cv2.imread(img_path)
            labels = torch.tensor(
                self.dataset.loc[
                    self.dataset.index[idx], self.dataset.columns != "image"
                ].tolist(),
                dtype=torch.float32,
            )
        else:
            img_path = self.dataset[idx]
            image = cv2.imread(img_path)
            labels = torch.zeros((0,), dtype=torch.float32)

        if image is None:
            raise FileNotFoundError(f"Failed to read image at path: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h, w = DIMENSION_HW
        image = cv2.resize(image, (w, h))

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, labels

    def __len__(self):
        return len(self.dataset)




## === cell 3
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

valid_transform = A.Compose(
    [A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
)

test_transform = valid_transform

test_dataset = PlantDataSet(test_images_names, None, test_transform)
BS = 32
plants_test_data_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BS,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




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

            probs_np = probabilities.detach().cpu().numpy()
            if predictions is None:
                predictions = probs_np
            else:
                predictions = np.concatenate((predictions, probs_np), axis=0)

            del images, output, probabilities
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    return np.array(predictions)




## === cell 5
def create_submission(test_images_path, predictions, threshold=0.75):
    rows = []
    for image_name, prediction in zip(test_images_path, predictions):
        fname = os.path.basename(image_name)
        arr = [
            label
            for pred, label in zip(prediction, train_labels.columns)
            if pred > threshold
        ]
        if len(arr) == 0:
            arr = ["healthy"]
        prediction_labels = " ".join(arr)
        rows.append((fname, prediction_labels))

    submission_df = pd.DataFrame(rows, columns=["image", "labels"])
    submission_df.to_csv("submission.csv", index=False)
    return submission_df




## === cell 6
try:
    weights_path = "../input/resnet50-final/resnet50_final.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(weights_path)
    resnet50 = models.resnet50(pretrained=False, num_classes=labels_size)
    resnet50.load_state_dict(torch.load(weights_path, map_location="cpu"))
except Exception:
    resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    in_features = resnet50.fc.in_features
    resnet50.fc = torch.nn.Linear(in_features, labels_size)

resnet50 = resnet50.to(device)

for name, p in resnet50.named_parameters():
    p.requires_grad = False
for p in resnet50.fc.parameters():
    p.requires_grad = True

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.15, random_state=seed, shuffle=True
)
train_split = train_df.iloc[train_idx].reset_index(drop=True)
val_split = train_df.iloc[val_idx].reset_index(drop=True)

train_images_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/"
train_ds = PlantDataSet(train_split, train_images_dir, train_transform)
val_ds = PlantDataSet(val_split, train_images_dir, valid_transform)

train_loader = DataLoader(
    train_ds,
    batch_size=BS,
    shuffle=True,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)
val_loader = DataLoader(
    val_ds,
    batch_size=BS,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(resnet50.fc.parameters(), lr=2e-3, weight_decay=1e-4)


def train_head(model, train_loader, epochs=2):
    model.train()
    for ep in range(epochs):
        running = 0.0
        n = 0
        for images, targets in train_loader:
            images = images.float().to(device)
            targets = targets.float().to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running += loss.item() * images.size(0)
            n += images.size(0)

            del images, targets, logits, loss
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print(f"epoch {ep+1}/{epochs} train_loss={running/max(n,1):.4f}")


def collect_val_probs(model, val_loader):
    model.eval()
    probs_list = []
    y_list = []
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.float().to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            probs_list.append(probs)
            y_list.append(targets.numpy())
            del images, targets, logits
    return np.vstack(probs_list), np.vstack(y_list)


train_head(resnet50, train_loader, epochs=2)

val_probs, val_true = collect_val_probs(resnet50, val_loader)


def macro_f1_from_probs(y_true, y_prob, thr):
    y_pred = (y_prob > thr).astype(np.int32)
    eps = 1e-12
    tp = (y_pred * y_true).sum(axis=0)
    fp = (y_pred * (1 - y_true)).sum(axis=0)
    fn = ((1 - y_pred) * y_true).sum(axis=0)
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    return float(np.mean(f1))


threshold_grid = np.arange(0.20, 0.81, 0.05)
best_thr = 0.75
best_f1 = -1.0
for thr in threshold_grid:
    f1 = macro_f1_from_probs(val_true, val_probs, thr)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = float(thr)

print(f"Chosen threshold={best_thr:.2f} (val mean-F1 proxy={best_f1:.4f})")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3133759610.py in <cell line: 0>()
     86 
     87 
---> 88 train_head(resnet50, train_loader, epochs=2)
     89 
     90 val_probs, val_true = collect_val_probs(resnet50, val_loader)

/tmp/ipykernel_55/3133759610.py in train_head(model, train_loader, epochs)
     51         running = 0.0
     52         n = 0
---> 53         for images, targets in train_loader:
     54             images = images.float().to(device)
     55             targets = targets.float().to(device)

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

RuntimeError: Caught RuntimeError in DataLoader worker process 0.
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
RuntimeError: stack expects each tensor to be equal size, but got [3, 256, 171] at entry 0 and [3, 171, 256] at entry 15


## === cell 7
test_predictions = test(plants_test_data_loader, resnet50)
print(test_predictions.shape, test_predictions[:2])
submission_df = create_submission(
    test_images_names, test_predictions, threshold=best_thr
)
print(submission_df.head())
print("Wrote submission.csv with", len(submission_df), "rows")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1448198570.py in <cell line: 0>()
      2 print(test_predictions.shape, test_predictions[:2])
      3 submission_df = create_submission(
----> 4     test_images_names, test_predictions, threshold=best_thr
      5 )
      6 print(submission_df.head())

NameError: name 'best_thr' is not defined
