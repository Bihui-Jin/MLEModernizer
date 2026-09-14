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

3.8

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
plotly==5.24.1
plotly-express==0.4.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
timm==1.0.19
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

0.9533226297898088

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, time, random

import numpy as np
import pandas as pd

import albumentations as A
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim

from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import warnings

warnings.filterwarnings("ignore")



## === cell 1
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGE_INPUT = DIR_INPUT

SEED = 42
N_FOLDS = 5
N_EPOCHS = 15
BATCH_SIZE = 8
IMAGE_SIZE = (409, 273)  # kept (unused in original)
LR = 1e-4  # conservative default; only used if we need to train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)
device




## === cell 2
def resolve_image_path(image_id: str) -> str:
    p1 = os.path.join(IMAGE_INPUT, "images_409_273", f"{image_id}.jpg")
    p2 = os.path.join(IMAGE_INPUT, "images", f"{image_id}.jpg")
    if os.path.exists(p1):
        return p1
    if os.path.exists(p2):
        return p2
    p3 = os.path.join(IMAGE_INPUT, "images", image_id)
    if os.path.exists(p3):
        return p3
    raise FileNotFoundError(
        f"Image not found for image_id={image_id}. Tried: {p1} and {p2}"
    )




## === cell 3
class PlantDataset(Dataset):
    def __init__(self, df, transforms=None, test_set=False):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.test_set = test_set
        if self.transforms is None:
            self.transforms = A.Compose([ToTensorV2(p=1.0)])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        image_src = resolve_image_path(image_id)
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"cv2.imread failed for: {image_src}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transforms(image=image)
        image = transformed["image"]

        if not self.test_set:
            labels = self.df.loc[
                idx, ["healthy", "multiple_diseases", "rust", "scab"]
            ].values.astype(np.float32)
            labels = torch.from_numpy(labels)
            return image, labels
        else:
            return image




## === cell 4
def trim_network_at_index(network, index=-1):
    assert index < 0, f"Param index must be negative. Received {index}"
    return nn.Sequential(*list(network.children())[:index])


class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT
        )
        in_features = self.backbone.fc.in_features
        self.backbone = trim_network_at_index(self.backbone, -1)
        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.logit(x)
        return x




## === cell 5
transforms_valid = A.Compose(
    [
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ]
)

transforms_train = A.Compose(
    [
        A.RandomResizedCrop(size=(273, 409), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ]
)

train_df = pd.read_csv(os.path.join(DIR_INPUT, "train.csv"))
test_df = pd.read_csv(os.path.join(DIR_INPUT, "test.csv"))

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

dataset_test = PlantDataset(df=test_df, test_set=True, transforms=transforms_valid)
testloader = DataLoader(
    dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## === cell 6
def predict_probs(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            images = batch.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.detach().cpu().numpy())
    return np.concatenate(all_probs, axis=0)




## === cell 7
EXTERNAL_MODEL_DIR = "/kaggle/input/plant-pathology-2020-training"


def external_fold_checkpoint_path(i_fold: int) -> str:
    return os.path.join(EXTERNAL_MODEL_DIR, f"modelF{i_fold}.pth")


def can_use_external_models(n_folds: int) -> bool:
    return os.path.isdir(EXTERNAL_MODEL_DIR) and all(
        os.path.exists(external_fold_checkpoint_path(i)) for i in range(n_folds)
    )


def load_fold_model_from_path(model_path: str) -> nn.Module:
    model = PlantModel(num_classes=4).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    return model




## === cell 8
def train_one_fold(fold: int, trn_idx: np.ndarray, val_idx: np.ndarray) -> str:
    trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
    val_df = train_df.iloc[val_idx].reset_index(drop=True)

    ds_tr = PlantDataset(trn_df, transforms=transforms_train, test_set=False)
    ds_va = PlantDataset(val_df, transforms=transforms_valid, test_set=False)

    dl_tr = DataLoader(
        ds_tr,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    model = PlantModel(num_classes=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_path = os.path.join("/kaggle/working", f"modelF{fold}.pth")

    for epoch in range(N_EPOCHS):
        model.train()
        for images, labels in dl_tr:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_logits = []
        val_labels = []
        with torch.no_grad():
            for images, labels in dl_va:
                images = images.to(device)
                logits = model(images)
                val_logits.append(logits.detach().cpu().numpy())
                val_labels.append(labels.detach().cpu().numpy())
        val_logits = np.concatenate(val_logits, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))

        try:
            fold_auc = roc_auc_score(val_labels, val_probs, average=None)
            fold_auc = float(np.mean(fold_auc))
        except Exception:
            fold_auc = -1.0

        if fold_auc > best_auc:
            best_auc = fold_auc
            torch.save(model.state_dict(), best_path)

    return best_path




## === cell 9
start = time.perf_counter()

test_probs_folds = []

if can_use_external_models(N_FOLDS):
    for i_fold in range(N_FOLDS):
        mp = external_fold_checkpoint_path(i_fold)
        model = load_fold_model_from_path(mp)
        test_probs_folds.append(predict_probs(model, testloader))
else:
    strat_labels = train_df[TARGET_COLS].values.argmax(axis=1)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    for i_fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, strat_labels)):
        best_path = train_one_fold(i_fold, trn_idx, val_idx)
        model = load_fold_model_from_path(best_path)
        test_probs_folds.append(predict_probs(model, testloader))

test_probs_folds = np.stack(test_probs_folds, axis=0)  # (n_folds, n_test, 4)
test_probs_mean = test_probs_folds.mean(axis=0)  # (n_test, 4)

print(
    f"Finished inference (and training if needed) in {(time.perf_counter() - start):.2f} seconds"
)
test_probs_mean.shape



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2898845840.py in <cell line: 0>()
     13 
     14     for i_fold, (trn_idx, val_idx) in enumerate(skf.split(train_df, strat_labels)):
---> 15         best_path = train_one_fold(i_fold, trn_idx, val_idx)
     16         model = load_fold_model_from_path(best_path)
     17         test_probs_folds.append(predict_probs(model, testloader))

/tmp/ipykernel_55/3682101901.py in train_one_fold(fold, trn_idx, val_idx)
     45         val_labels = []
     46         with torch.no_grad():
---> 47             for images, labels in dl_va:
     48                 images = images.to(device)
     49                 logits = model(images)

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
RuntimeError: stack expects each tensor to be equal size, but got [3, 1365, 2048] at entry 0 and [3, 2048, 1365] at entry 1


## === cell 10
submission_df = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))
submission_df[TARGET_COLS] = test_probs_mean.astype(np.float32)
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)

submission_df.head()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1456639640.py in <cell line: 0>()
      1 submission_df = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))
----> 2 submission_df[TARGET_COLS] = test_probs_mean.astype(np.float32)
      3 submission_path = "submission.csv"
      4 submission_df.to_csv(submission_path, index=False)
      5 

NameError: name 'test_probs_mean' is not defined

## === cell 11
assert submission_path.endswith(".csv")
assert submission_df.shape[0] == test_df.shape[0]
assert list(submission_df.columns) == ["image_id"] + TARGET_COLS
assert np.isfinite(submission_df[TARGET_COLS].values).all()
submission_path

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1754611530.py in <cell line: 0>()
----> 1 assert submission_path.endswith(".csv")
      2 assert submission_df.shape[0] == test_df.shape[0]
      3 assert list(submission_df.columns) == ["image_id"] + TARGET_COLS
      4 assert np.isfinite(submission_df[TARGET_COLS].values).all()
      5 submission_path

NameError: name 'submission_path' is not defined
