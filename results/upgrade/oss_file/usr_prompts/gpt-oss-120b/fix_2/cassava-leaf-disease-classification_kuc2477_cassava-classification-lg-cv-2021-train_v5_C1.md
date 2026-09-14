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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
imageio==2.37.0
imageio-ffmpeg==0.6.0
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.6128739800543971

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os.path as osp

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import imageio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



## === cell 1
ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_DIR = f"{ROOT}/train_images/"
TRAIN_CSV = f"{ROOT}/train.csv"
TEST_DIR = f"{ROOT}/test_images/"
TEST_CSV = f"{ROOT}/sample_submission.csv"
BEST_MODEL_PATH = "/kaggle/working/model.pt"




## === cell 2
class CassavaDataset(Dataset):
    def __init__(self, split, transform=None):
        assert split in ("train", "val", "test")
        self.split = split
        self.transform = transform
        if split in ("train", "val"):
            csv = pd.read_csv(TRAIN_CSV)
            train_df, val_df = train_test_split(
                csv, test_size=0.1, random_state=0, stratify=csv["label"]
            )
            self.df = train_df if split == "train" else val_df
        else:
            self.df = pd.read_csv(TEST_CSV)
        self.df = self.df.reset_index(drop=True)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, i: int):
        base_dir = TRAIN_DIR if self.split in ("train", "val") else TEST_DIR
        img_path = osp.join(base_dir, self.df.loc[i, "image_id"])
        x = imageio.imread(img_path)
        if self.transform:
            x = self.transform(x)
        if self.split in ("train", "val"):
            y = int(self.df.loc[i, "label"])
            y = torch.tensor(y, dtype=torch.long)
            return x, y
        else:
            return x, -1  # placeholder label for test




## === cell 3
INPUT_SIZE = 128
TRANSFORMS = {
    "train": transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
        ]
    ),
    "test": transforms.Compose(
        [
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
        ]
    ),
}




## === cell 4
class CassavaClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        CH_LAST = 128
        NUM_CLASSES = 5
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, CH_LAST, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(
            CH_LAST * (INPUT_SIZE // (2**3)) ** 2,
            NUM_CLASSES,
        )

    def forward(self, x):
        h = self.pool(F.relu(self.conv1(x)))
        h = self.pool(F.relu(self.conv2(h)))
        h = self.pool(F.relu(self.conv3(h)))
        return self.fc(h.flatten(1))




## === cell 5
EPOCHS = 5  # a few more epochs to approach the target score
BATCH_SIZE = 64
MOMENTUM = 0.99
LR = 1e-4
WD = 1e-4
TRAIN_AND_SAVE = True



## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CassavaClassifier().to(device)
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)

train_dataset = CassavaDataset("train", transform=TRANSFORMS["train"])
val_dataset = CassavaDataset("val", transform=TRANSFORMS["val"])



## === cell 7
best_val_accuracy = 0.0

for e in range(EPOCHS if TRAIN_AND_SAVE else 0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    model.train()
    total, correct = 0, 0
    for x, y in tqdm(train_loader, desc=f"Training (Epoch {e})"):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, pred = torch.max(logits, dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()

    print("=== Train accuracy:", correct / total)

    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for x, y in tqdm(val_loader, desc=f"Validating (Epoch {e})"):
            x, y = x.to(device), y.to(device)
            logits = model(x)
            _, pred = torch.max(logits, dim=1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    val_accuracy = correct / total
    print("=== Val accuracy:", val_accuracy)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        print(f"* Saving best model (val acc {val_accuracy:.4f})")
        torch.save(model.state_dict(), BEST_MODEL_PATH)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2109991754.py in <cell line: 0>()
     20     model.train()
     21     total, correct = 0, 0
---> 22     for x, y in tqdm(train_loader, desc=f"Training (Epoch {e})"):
     23         x, y = x.to(device), y.to(device)
     24         logits = model(x)

/usr/local/lib/python3.11/dist-packages/tqdm/notebook.py in __iter__(self)
    248         try:
    249             it = super().__iter__()
--> 250             for obj in it:
    251                 # return super(tqdm...) will not catch exception
    252                 yield obj

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

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

/tmp/ipykernel_55/4243846501.py in __getitem__(self, i)
     22         x = imageio.imread(img_path)
     23         if self.transform:
---> 24             x = self.transform(x)
     25         if self.split in ("train", "val"):
     26             y = int(self.df.loc[i, "label"])

/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py in __call__(self, img)
     93     def __call__(self, img):
     94         for t in self.transforms:
---> 95             img = t(img)
     96         return img
     97 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py in forward(self, img)
    352             PIL Image or Tensor: Rescaled image.
    353         """
--> 354         return F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
    355 
    356     def __repr__(self) -> str:

/usr/local/lib/python3.11/dist-packages/torchvision/transforms/functional.py in resize(img, size, interpolation, max_size, antialias)
    463             )
    464 
--> 465     _, image_height, image_width = get_dimensions(img)
    466     if isinstance(size, int):
    467         size = [size]

/usr/local/lib/python3.11/dist-packages/torchvision/transforms/functional.py in get_dimensions(img)
     78         return F_t.get_dimensions(img)
     79 
---> 80     return F_pil.get_dimensions(img)
     81 
     82 

/usr/local/lib/python3.11/dist-packages/torchvision/transforms/_functional_pil.py in get_dimensions(img)
     29         width, height = img.size
     30         return [channels, height, width]
---> 31     raise TypeError(f"Unexpected type {type(img)}")
     32 
     33 

TypeError: Unexpected type <class 'numpy.ndarray'>

## === cell 8
model = CassavaClassifier().to(device)
if osp.exists(BEST_MODEL_PATH):
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
model.eval()

test_dataset = CassavaDataset("test", transform=TRANSFORMS["test"])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

test_csv = pd.read_csv(TEST_CSV)

pred_labels = []
with torch.no_grad():
    for x, _ in tqdm(test_loader, desc="Predicting test"):
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(logits, dim=1)
        pred_labels.extend(preds.cpu().numpy().tolist())

test_csv["label"] = pred_labels
submission_path = "submission.csv"
test_csv[["image_id", "label"]].to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2709583554.py in <cell line: 0>()
     12 pred_labels = []
     13 with torch.no_grad():
---> 14     for x, _ in tqdm(test_loader, desc="Predicting test"):
     15         x = x.to(device)
     16         logits = model(x)

/usr/local/lib/python3.11/dist-packages/tqdm/notebook.py in __iter__(self)
    248         try:
    249             it = super().__iter__()
--> 250             for obj in it:
    251                 # return super(tqdm...) will not catch exception
    252                 yield obj

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

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

/tmp/ipykernel_55/4243846501.py in __getitem__(self, i)
     22         x = imageio.imread(img_path)
     23         if self.transform:
---> 24             x = self.transform(x)
     25         if self.split in ("train", "val"):
     26             y = int(self.df.loc[i, "label"])

/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py in __call__(self, img)
     93     def __call__(self, img):
     94         for t in self.transforms:
---> 95             img = t(img)
     96         return img
     97 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torchvision/transforms/transforms.py in forward(self, img)
    352             PIL Image or Tensor: Rescaled image.
    353         """
--> 354         return F.resize(img, self.size, self.interpolation, self.max_size, self.antialias)
    355 
    356     def __repr__(self) -> str:

/usr/local/lib/python3.11/dist-packages/torchvision/transforms/functional.py in resize(img, size, interpolation, max_size, antialias)
    463             )
    464 
--> 465     _, image_height, image_width = get_dimensions(img)
    466     if isinstance(size, int):
    467         size = [size]

/usr/local/lib/python3.11/dist-packages/torchvision/transforms/functional.py in get_dimensions(img)
     78         return F_t.get_dimensions(img)
     79 
---> 80     return F_pil.get_dimensions(img)
     81 
     82 

/usr/local/lib/python3.11/dist-packages/torchvision/transforms/_functional_pil.py in get_dimensions(img)
     29         width, height = img.size
     30         return [channels, height, width]
---> 31     raise TypeError(f"Unexpected type {type(img)}")
     32 
     33 

TypeError: Unexpected type <class 'numpy.ndarray'>
