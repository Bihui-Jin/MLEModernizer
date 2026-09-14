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

0.9694673783378412

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
import torch.optim as optim

import timm

from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader

from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold

import warnings

warnings.filterwarnings("ignore")



## === cell 1
SEED = 42


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 2
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"


def resolve_images_dir():
    candidates = [
        os.path.join(DIR_INPUT, "images"),
        "/kaggle/input/images",
        "/kaggle/input/plant-pathology-2020-fgvc7/images",
    ]
    for c in candidates:
        if os.path.isdir(c) and len(os.listdir(c)) > 0:
            return c
    for root in ["/kaggle/input", "/kaggle/data"]:
        for dirpath, dirnames, filenames in os.walk(root):
            if os.path.basename(dirpath) == "images" and any(
                f.lower().endswith(".jpg") for f in filenames
            ):
                return dirpath
    raise FileNotFoundError(
        "Could not locate images directory under /kaggle/input or /kaggle/data."
    )


IMAGES_DIR = resolve_images_dir()
IMAGES_DIR



## === cell 3
N_FOLDS = 5
N_EPOCHS = 15
BATCH_SIZE = 8
NUM_CLASSES = 4

train_df = pd.read_csv(os.path.join(DIR_INPUT, "train.csv"))
test_df = pd.read_csv(os.path.join(DIR_INPUT, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
assert all(c in train_df.columns for c in ["image_id"] + target_cols)
assert list(sample_sub.columns) == ["image_id"] + target_cols

train_df["stratify_label"] = train_df[target_cols].values.argmax(1)

train_df.shape, test_df.shape, IMAGES_DIR




## === cell 4
class PlantDataset(Dataset):
    def __init__(self, df, images_dir, transforms=None, test_set=False):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transforms = transforms
        self.test_set = test_set
        if self.transforms is None:
            self.transforms = A.Compose([ToTensorV2(p=1.0)])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        image_src = os.path.join(self.images_dir, f"{image_id}.jpg")
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_src}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transforms(image=image)
        image = transformed["image"]

        if not self.test_set:
            labels = self.df.loc[idx, target_cols].values.astype(np.float32)
            labels = torch.from_numpy(labels)
            labels = labels.unsqueeze(-1)
            return image, labels
        else:
            return image




## === cell 5
def trim_network_at_index(network, index=-1):
    assert index < 0, f"Param index must be negative. Received {index}"
    return nn.Sequential(*list(network.children())[:index])


class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = timm.create_model("resnest269e", pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone = trim_network_at_index(self.backbone, -1)
        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        x = self.logit(x)
        return x




## === cell 6
transforms_train = A.Compose(
    [
        A.RandomResizedCrop(
            height=273, width=409, scale=(0.85, 1.0), ratio=(0.75, 1.3333), p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ]
)

transforms_valid = A.Compose(
    [
        A.Resize(height=273, width=409, p=1.0),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ]
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValidationError                           Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     66             schema_kwargs["strict"] = strict
---> 67             config = schema_cls(**schema_kwargs)
     68             validated_kwargs = config.model_dump()

/usr/local/lib/python3.11/dist-packages/pydantic/main.py in __init__(self, **data)
    249         __tracebackhide__ = True
--> 250         validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
    251         if self is not validated_self:

ValidationError: 1 validation error for InitSchema
size
  Field required [type=missing, input_value={'scale': (0.85, 1.0), 'r...: None, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

The above exception was the direct cause of the following exception:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2140988634.py in <cell line: 0>()
      2 transforms_train = A.Compose(
      3     [
----> 4         A.RandomResizedCrop(
      5             height=273, width=409, scale=(0.85, 1.0), ratio=(0.75, 1.3333), p=1.0
      6         ),

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in custom_init(self, *args, **kwargs)
    103                 full_kwargs, param_names, strict = cls._process_init_parameters(original_init, args, kwargs)
    104 
--> 105                 validated_kwargs = cls._validate_parameters(
    106                     dct["InitSchema"],
    107                     full_kwargs,

/usr/local/lib/python3.11/dist-packages/albumentations/core/validation.py in _validate_parameters(schema_cls, full_kwargs, param_names, strict)
     69             validated_kwargs.pop("strict", None)
     70         except ValidationError as e:
---> 71             raise ValueError(str(e)) from e
     72         except Exception as e:
     73             if strict:

ValueError: 1 validation error for InitSchema
size
  Field required [type=missing, input_value={'scale': (0.85, 1.0), 'r...: None, 'strict': False}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.12/v/missing

## === cell 7
def train_one_fold(fold, tr_idx, va_idx, out_dir="/kaggle/working"):
    tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
    va_df = train_df.iloc[va_idx].reset_index(drop=True)

    ds_tr = PlantDataset(
        tr_df, images_dir=IMAGES_DIR, transforms=transforms_train, test_set=False
    )
    ds_va = PlantDataset(
        va_df, images_dir=IMAGES_DIR, transforms=transforms_valid, test_set=False
    )

    dl_tr = DataLoader(
        ds_tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    dl_va = DataLoader(
        ds_va, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    model = PlantModel(num_classes=NUM_CLASSES).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-4)

    best_va_loss = float("inf")
    best_path = os.path.join(out_dir, f"modelF{fold}.pth")

    for epoch in range(N_EPOCHS):
        model.train()
        tr_loss = 0.0
        for xb, yb in dl_tr:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True).squeeze(-1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            tr_loss += loss.item() * xb.size(0)

        tr_loss /= len(ds_tr)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_va:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True).squeeze(-1)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_loss += loss.item() * xb.size(0)
        va_loss /= len(ds_va)

        if va_loss < best_va_loss:
            best_va_loss = va_loss
            torch.save(model.state_dict(), best_path)

        print(
            f"Fold {fold} | Epoch {epoch+1}/{N_EPOCHS} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | best_val={best_va_loss:.4f}"
        )

    return best_path




## === cell 8
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_paths = []

start = time.perf_counter()
for fold, (tr_idx, va_idx) in enumerate(
    skf.split(train_df, train_df["stratify_label"].values)
):
    path = os.path.join("/kaggle/working", f"modelF{fold}.pth")
    if not os.path.exists(path):
        fold_paths.append(
            train_one_fold(fold, tr_idx, va_idx, out_dir="/kaggle/working")
        )
    else:
        fold_paths.append(path)

print(f"Finished training in {(time.perf_counter() - start):.2f} seconds")
fold_paths



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/890028945.py in <cell line: 0>()
     10     if not os.path.exists(path):
     11         fold_paths.append(
---> 12             train_one_fold(fold, tr_idx, va_idx, out_dir="/kaggle/working")
     13         )
     14     else:

/tmp/ipykernel_55/3414186470.py in train_one_fold(fold, tr_idx, va_idx, out_dir)
      6 
      7     ds_tr = PlantDataset(
----> 8         tr_df, images_dir=IMAGES_DIR, transforms=transforms_train, test_set=False
      9     )
     10     ds_va = PlantDataset(

NameError: name 'transforms_train' is not defined

## === cell 9
dataset_test = PlantDataset(
    df=test_df, images_dir=IMAGES_DIR, test_set=True, transforms=transforms_valid
)
testloader = DataLoader(
    dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/169678223.py in <cell line: 0>()
      1 dataset_test = PlantDataset(
----> 2     df=test_df, images_dir=IMAGES_DIR, test_set=True, transforms=transforms_valid
      3 )
      4 testloader = DataLoader(
      5     dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True

NameError: name 'transforms_valid' is not defined

## === cell 10
def test_model(model_path, testloader):
    model = PlantModel(num_classes=NUM_CLASSES)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    test_probs = []
    with torch.no_grad():
        for batch in tqdm(testloader, total=len(testloader), leave=False):
            xb = batch.to(device, non_blocking=True)
            logits = model(xb)
            probs = F.softmax(logits, dim=1)
            test_probs.append(probs.detach().cpu().numpy())

    return np.concatenate(test_probs, axis=0)




## === cell 11
test_probs_folds = []
start = time.perf_counter()
for fold in range(N_FOLDS):
    model_path = os.path.join("/kaggle/working", f"modelF{fold}.pth")
    test_probs_fold = test_model(model_path, testloader)
    test_probs_folds.append(test_probs_fold)

print(f"Finished inference in {(time.perf_counter() - start):.2f} seconds")
np.array(test_probs_folds).shape



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/294660801.py in <cell line: 0>()
      4 for fold in range(N_FOLDS):
      5     model_path = os.path.join("/kaggle/working", f"modelF{fold}.pth")
----> 6     test_probs_fold = test_model(model_path, testloader)
      7     test_probs_folds.append(test_probs_fold)
      8 

NameError: name 'testloader' is not defined

## === cell 12
test_probs_mean = np.mean(np.stack(test_probs_folds, axis=0), axis=0)
test_probs_mean.shape



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1780338611.py in <cell line: 0>()
      1 # Fix: correct fold-averaging. Original zip(*test_probs) was wrong and broke submission assignment.
----> 2 test_probs_mean = np.mean(np.stack(test_probs_folds, axis=0), axis=0)
      3 test_probs_mean.shape
      4 

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    443     arrays = [asanyarray(arr) for arr in arrays]
    444     if not arrays:
--> 445         raise ValueError('need at least one array to stack')
    446 
    447     shapes = {arr.shape for arr in arrays}

ValueError: need at least one array to stack

## === cell 13
submission_df = sample_sub.copy()
submission_df[target_cols] = test_probs_mean.astype(np.float32)

if not submission_df["image_id"].equals(test_df["image_id"]):
    submission_df = submission_df.merge(
        test_df[["image_id"]], on="image_id", how="right"
    )

submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
submission_df.head(), submission_path



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/764358217.py in <cell line: 0>()
      1 # Build submission with correct columns/order and valid probabilities
      2 submission_df = sample_sub.copy()
----> 3 submission_df[target_cols] = test_probs_mean.astype(np.float32)
      4 
      5 # Safety: ensure alignment by image_id order in sample_submission/test.csv

NameError: name 'test_probs_mean' is not defined

## === cell 14
assert os.path.exists("submission.csv")
sub_check = pd.read_csv("submission.csv")
assert list(sub_check.columns) == ["image_id"] + target_cols
assert len(sub_check) == len(test_df)
sub_check.describe(include="all")

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/300733285.py in <cell line: 0>()
      1 # Quick validation of submission format
----> 2 assert os.path.exists("submission.csv")
      3 sub_check = pd.read_csv("submission.csv")
      4 assert list(sub_check.columns) == ["image_id"] + target_cols
      5 assert len(sub_check) == len(test_df)

AssertionError:
