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

3.10

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

0.8055303717135086

# 6. Current score

0.11472

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.11472) has done: 'The fix adds a fallback when the checkpoint file is missing, changes the model’s output layer to 5 classes (the correct number for the competition), switches the model to the “semi” mode so a pretrained backbone is used, and implements a minimal training loop that fine‑tunes the last layer on the training data before making predictions. This ensures the script runs end‑to‑end, creates a valid `submission.csv`, and yields sensible predictions without altering the core architecture.'

# 9. Code solution

## === cell 0
import os
import re
import cv2
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from time import time
from torch import nn, optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
from torch.nn.utils import weight_norm as WN
from torchvision import models, transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder



## === cell 1
le = LabelEncoder()

SEED = 42
SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRANSFORM_FINAL = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.43032, 0.49672, 0.31341], [0.20665, 0.21170, 0.18763]),
    ]
)




## === cell 2
def breaker(num: int = 50, char: str = "*") -> None:
    print("\n" + num * char + "\n")


def load_image(path: str, size: int) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2RGB)
    image = cv2.resize(src=image, dsize=(size, size), interpolation=cv2.INTER_AREA)
    return image


def show_image(image: np.ndarray, cmap: str = "gnuplot2") -> None:
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.axis("off")
    plt.show()




## === cell 3
class DS(Dataset):
    def __init__(
        self,
        base_path: str,
        filenames: np.ndarray,
        labels: np.ndarray = None,
        mode: str = "train",
        transform=None,
    ):

        assert (
            re.match(r"^train$", mode, re.IGNORECASE)
            or re.match(r"^valid$", mode, re.IGNORECASE)
            or re.match(r"^test$", mode, re.IGNORECASE)
        ), "Invalid Mode"

        self.mode = mode
        self.base_path = base_path
        self.filenames = filenames
        self.transform = transform

        if re.match(r"^train$", mode, re.IGNORECASE) or re.match(
            r"^valid$", mode, re.IGNORECASE
        ):
            self.labels = labels

    def __len__(self):
        return self.filenames.shape[0]

    def __getitem__(self, idx):
        image = load_image(os.path.join(self.base_path, self.filenames[idx]), SIZE)
        if re.match(r"^train$", self.mode, re.IGNORECASE) or re.match(
            r"^valid$", self.mode, re.IGNORECASE
        ):
            return self.transform(image), torch.LongTensor(self.labels[idx])
        else:
            return self.transform(image)




## === cell 4
class CNN(nn.Module):
    def __init__(self, mode: str, model_name: str):

        super(CNN, self).__init__()

        self.model_name = model_name
        self.mode = mode

        if re.match(r"^vgg$", self.model_name, re.IGNORECASE):
            if re.match(r"^full$", self.mode, re.IGNORECASE):
                self.model = models.vgg16_bn(pretrained=False, progress=True)
                self.model.classifier[-1] = nn.Linear(
                    in_features=self.model.classifier[-1].in_features, out_features=5
                )
            elif re.match(r"^semi$", self.mode, re.IGNORECASE) or re.match(
                r"^final$", self.mode, re.IGNORECASE
            ):
                self.model = models.vgg16_bn(pretrained=True, progress=True)
                self.freeze()
                self.model.classifier[-1] = nn.Linear(
                    in_features=self.model.classifier[-1].in_features, out_features=5
                )

        elif re.match(r"^resnet$", self.model_name, re.IGNORECASE):
            if re.match(r"^full$", self.mode, re.IGNORECASE):
                self.model = models.resnet50(pretrained=False, progress=True)
                self.model.fc = nn.Linear(
                    in_features=self.model.fc.in_features, out_features=5
                )
            elif re.match(r"^semi$", self.mode, re.IGNORECASE) or re.match(
                r"^final$", self.mode, re.IGNORECASE
            ):
                self.model = models.resnet50(pretrained=True, progress=True)
                self.freeze()
                self.model.fc = nn.Linear(
                    in_features=self.model.fc.in_features, out_features=5
                )

        elif re.match(r"^densenet$", self.model_name, re.IGNORECASE):
            if re.match(r"^full$", self.mode, re.IGNORECASE):
                self.model = models.densenet169(pretrained=False, progress=True)
                self.model.classifier = nn.Linear(
                    in_features=self.model.classifier.in_features, out_features=5
                )
            elif re.match(r"^semi$", self.mode, re.IGNORECASE) or re.match(
                r"^final$", self.mode, re.IGNORECASE
            ):
                self.model = models.densenet169(pretrained=True, progress=True)
                self.freeze()
                self.model.classifier = nn.Linear(
                    in_features=self.model.classifier.in_features, out_features=5
                )

        elif re.match(r"^mobilenet$", self.model_name, re.IGNORECASE):
            if re.match(r"^full$", self.mode, re.IGNORECASE):
                self.model = models.mobilenet_v3_small(pretrained=False, progress=True)
                self.model.classifier[-1] = nn.Linear(
                    in_features=self.model.classifier[-1].in_features, out_features=5
                )
            elif re.match(r"^semi$", self.mode, re.IGNORECASE) or re.match(
                r"^final$", self.mode, re.IGNORECASE
            ):
                self.model = models.mobilenet_v3_small(pretrained=True, progress=True)
                self.freeze()
                self.model.classifier[-1] = nn.Linear(
                    in_features=self.model.classifier[-1].in_features, out_features=5
                )

    def freeze(self):
        for params in self.parameters():
            params.requires_grad = False

        if re.match(r"^vgg$", self.model_name, re.IGNORECASE):
            if re.match(r"^semi$", self.mode, re.IGNORECASE):
                for names, params in self.named_parameters():
                    if (
                        re.match(r".*features.3[4-9].*", names, re.IGNORECASE)
                        or re.match(r".*features.4[0-9].*", names, re.IGNORECASE)
                        or re.match(r".*classifier.*", names, re.IGNORECASE)
                    ):
                        params.requires_grad = True

        elif re.match(r"^resnet$", self.model_name, re.IGNORECASE):
            if re.match(r"^semi$", self.mode, re.IGNORECASE):
                for names, params in self.named_parameters():
                    if re.match(r".*layer4.*", names, re.IGNORECASE):
                        params.requires_grad = True

        elif re.match(r"^densenet$", self.model_name, re.IGNORECASE):
            if re.match(r"^semi$", self.mode, re.IGNORECASE):
                for names, params in self.named_parameters():
                    if re.match(r".*denseblock4.*", names, re.IGNORECASE) or re.match(
                        r".*norm5.*", names, re.IGNORECASE
                    ):
                        params.requires_grad = True

        elif re.match(r"^mobilenet$", self.model_name, re.IGNORECASE):
            if re.match(r"^semi$", self.mode, re.IGNORECASE):
                for names, params in self.named_parameters():
                    if (
                        re.match(r".*features.9.*", names, re.IGNORECASE)
                        or re.match(r".*features.1[0-2].*", names, re.IGNORECASE)
                        or re.match(r".*classifier.*", names, re.IGNORECASE)
                    ):
                        params.requires_grad = True

    def get_optimizer(self, lr=1e-3, wd=0.0):
        params = [p for p in self.parameters() if p.requires_grad]
        return optim.Adam(params, lr=lr, weight_decay=wd)

    def get_plateau_scheduler(self, optimizer=None, patience=5, eps=1e-8):
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer, patience=patience, eps=eps
        )

    def forward(self, x):
        return nn.LogSoftmax(dim=1)(self.model(x))




## === cell 5
def predict(model=None, dataloader=None, path=None) -> np.ndarray:
    if path is not None and os.path.exists(path):
        checkpoint = torch.load(path, map_location=DEVICE)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(f"Warning: checkpoint {path} not found – using current model weights.")
    model.to(DEVICE)
    model.eval()

    y_pred = torch.zeros(1, 1).to(DEVICE)

    for X in dataloader:
        X = X.to(DEVICE)
        with torch.no_grad():
            output = torch.argmax(torch.exp(model(X)), dim=1)
        y_pred = torch.cat((y_pred, output.view(-1, 1)), dim=0)

    return y_pred[1:].detach().cpu().numpy()




## === cell 6
DEBUG: bool = False

if DEBUG:
    n_splits = 3
    epochs = 2
    mode = "semi"
    model_name = "mobilenet"
else:
    n_splits = 5  # reduced for faster training while still providing a split
    epochs = 5  # modest number of epochs to keep runtime low
    mode = "semi"
    model_name = "resnet"

batch_size = 128
lr = 1e-4
wd = 1e-5
early_stopping = 5
patience = None
eps = None



## === cell 7
train_df = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
train_filenames = train_df["image_id"].values
train_labels = train_df["label"].values.astype(np.int64)

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
train_idx, valid_idx = next(skf.split(train_filenames, train_labels))

train_set = DS(
    "../input/cassava-leaf-disease-classification/train_images",
    train_filenames[train_idx],
    train_labels[train_idx],
    mode="train",
    transform=TRANSFORM,
)

valid_set = DS(
    "../input/cassava-leaf-disease-classification/train_images",
    train_filenames[valid_idx],
    train_labels[valid_idx],
    mode="valid",
    transform=TRANSFORM_FINAL,
)

train_loader = DL(
    train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
)
valid_loader = DL(
    valid_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)



## === cell 8
model = CNN(mode=mode, model_name=model_name).to(DEVICE)
criterion = nn.NLLLoss()
optimizer = model.get_optimizer(lr=lr, wd=wd)

best_val_loss = np.inf
best_state_path = "best_state.pt"

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE).squeeze()
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * X_batch.size(0)
    epoch_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in valid_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE).squeeze()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item() * X_batch.size(0)
    val_loss /= len(valid_loader.dataset)

    print(
        f"Epoch {epoch+1}/{epochs} - Train loss: {epoch_loss:.4f} - Val loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({"model_state_dict": model.state_dict()}, best_state_path)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/618271257.py in <cell line: 0>()
     10     model.train()
     11     epoch_loss = 0.0
---> 12     for X_batch, y_batch in train_loader:
     13         X_batch = X_batch.to(DEVICE)
     14         y_batch = y_batch.to(DEVICE).squeeze()

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
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/collate.py", line 271, in collate_tensor_fn
    out = elem.new(storage).resize_(len(batch), *list(elem.size()))
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Trying to resize storage that is not resizable


## === cell 9
ss_df = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
ts_filenames = ss_df.image_id.copy().values

ts_data_setup = DS(
    "../input/cassava-leaf-disease-classification/test_images",
    ts_filenames,
    None,
    "test",
    TRANSFORM,
)
ts_data = DL(
    ts_data_setup, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
)

y_pred = predict(model=model, dataloader=ts_data, path=best_state_path)

ss_df["label"] = y_pred.astype("uint8")
ss_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
