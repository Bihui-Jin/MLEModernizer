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

0.9292863989502848

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

from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold

import warnings

warnings.filterwarnings("ignore")



## === cell 1
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"

SEED = 42
N_FOLDS = 5
N_EPOCHS = 3
BATCH_SIZE = 16
IMAGE_SIZE = (273, 409)
device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
device




## === cell 2
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)




## === cell 3
class PlantDataset(Dataset):

    def __init__(self, df, transforms=None, test_set=False):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.test_set = test_set
        if not self.transforms:
            self.transforms = A.Compose([ToTensorV2(p=1.0)])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        image_src = DIR_INPUT + "/images/" + self.df.loc[idx, "image_id"] + ".jpg"
        image = cv2.imread(image_src, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_src}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, IMAGE_SIZE)

        transformed = self.transforms(image=image)
        image = transformed["image"]

        if not self.test_set:
            labels = self.df.loc[
                idx, ["healthy", "multiple_diseases", "rust", "scab"]
            ].values
            labels = torch.from_numpy(
                labels.astype(np.float32)
            )  # Fix: BCE expects float targets
            return image, labels
        else:
            return image




## === cell 4
transforms_train = A.Compose(
    [
        A.RandomResizedCrop(
            height=IMAGE_SIZE[0],
            width=IMAGE_SIZE[1],
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            p=1.0,
        ),
        A.HorizontalFlip(p=0.5),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ]
)

transforms_valid = A.Compose(
    [
        A.Resize(height=IMAGE_SIZE[0], width=IMAGE_SIZE[1], p=1.0),
        A.Normalize(p=1.0),
        ToTensorV2(p=1.0),
    ]
)




## --- ERROR in cell 4, traceback:
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
/tmp/ipykernel_55/3215666730.py in <cell line: 0>()
      2 transforms_train = A.Compose(
      3     [
----> 4         A.RandomResizedCrop(
      5             height=IMAGE_SIZE[0],
      6             width=IMAGE_SIZE[1],

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

## === cell 5
class PlantModel(nn.Module):

    def __init__(self, num_classes=4):
        super().__init__()

        self.backbone = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )

        in_features = self.backbone.fc.in_features
        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        batch_size, C, H, W = x.shape

        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        x = F.dropout(x, 0.25, self.training)
        x = self.logit(x)
        return x




## === cell 6
train_df = pd.read_csv(DIR_INPUT + "/train.csv")
test_df = pd.read_csv(DIR_INPUT + "/test.csv")

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

train_df["stratify_label"] = train_df[TARGET_COLS].values.argmax(axis=1)

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def train_one_fold(fold, trn_idx, val_idx):
    df_trn = train_df.iloc[trn_idx].reset_index(drop=True)
    df_val = train_df.iloc[val_idx].reset_index(drop=True)

    ds_trn = PlantDataset(df_trn, transforms=transforms_train, test_set=False)
    ds_val = PlantDataset(df_val, transforms=transforms_valid, test_set=False)

    dl_trn = DataLoader(
        ds_trn,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    model = PlantModel(num_classes=len(TARGET_COLS)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")

    for epoch in range(N_EPOCHS):
        model.train()
        tr_losses = []
        for images, labels in dl_trn:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for images, labels in dl_val:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                loss = criterion(logits, labels)
                val_losses.append(loss.item())

        val_loss = float(np.mean(val_losses)) if len(val_losses) else float("inf")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"modelF{fold}.pth")

    return best_val_loss


start = time.perf_counter()
fold_losses = []
for fold, (trn_idx, val_idx) in enumerate(
    skf.split(train_df, train_df["stratify_label"])
):
    loss = train_one_fold(fold, trn_idx, val_idx)
    fold_losses.append(loss)

print(
    f"Finished training {N_FOLDS} folds in {(time.perf_counter()-start):.2f}s. Best val losses: {fold_losses}"
)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/624048846.py in <cell line: 0>()
     76     skf.split(train_df, train_df["stratify_label"])
     77 ):
---> 78     loss = train_one_fold(fold, trn_idx, val_idx)
     79     fold_losses.append(loss)
     80 

/tmp/ipykernel_55/624048846.py in train_one_fold(fold, trn_idx, val_idx)
     15     df_val = train_df.iloc[val_idx].reset_index(drop=True)
     16 
---> 17     ds_trn = PlantDataset(df_trn, transforms=transforms_train, test_set=False)
     18     ds_val = PlantDataset(df_val, transforms=transforms_valid, test_set=False)
     19 

NameError: name 'transforms_train' is not defined

## === cell 7
dataset_test = PlantDataset(df=test_df, test_set=True, transforms=transforms_valid)
testloader = DataLoader(
    dataset_test,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1220311397.py in <cell line: 0>()
----> 1 dataset_test = PlantDataset(df=test_df, test_set=True, transforms=transforms_valid)
      2 testloader = DataLoader(
      3     dataset_test,
      4     batch_size=BATCH_SIZE,
      5     shuffle=False,

NameError: name 'transforms_valid' is not defined

## === cell 8
def test_model(model_name, testloader):
    model_path = f"{model_name}.pth"
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Missing checkpoint: {model_path}. Training should have created it."
        )

    model = PlantModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)

    test_probs = []
    model.eval()

    with torch.no_grad():
        for image in tqdm(testloader, total=len(testloader)):
            logits = model(image.to(device, non_blocking=True))
            probs = F.softmax(logits, dim=1)
            test_probs.append(probs.detach().cpu().numpy())

    test_probs = np.concatenate(test_probs, axis=0)
    return test_probs




## === cell 9
test_probs = []
start = time.perf_counter()
for i_fold in range(N_FOLDS):
    test_probs_fold = test_model(f"modelF{i_fold}", testloader)
    test_probs.append(test_probs_fold)
print(f"Finished Inference in {(time.perf_counter() - start):.2f} seconds")



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/447762004.py in <cell line: 0>()
      2 start = time.perf_counter()
      3 for i_fold in range(N_FOLDS):
----> 4     test_probs_fold = test_model(f"modelF{i_fold}", testloader)
      5     test_probs.append(test_probs_fold)
      6 print(f"Finished Inference in {(time.perf_counter() - start):.2f} seconds")

NameError: name 'testloader' is not defined

## === cell 10
test_probs_mean = np.mean(np.stack(test_probs, axis=0), axis=0)
test_probs_mean.shape



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1841970295.py in <cell line: 0>()
      1 # Fix: average across folds correctly; expected shape (n_test, 4)
----> 2 test_probs_mean = np.mean(np.stack(test_probs, axis=0), axis=0)
      3 test_probs_mean.shape
      4 

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    443     arrays = [asanyarray(arr) for arr in arrays]
    444     if not arrays:
--> 445         raise ValueError('need at least one array to stack')
    446 
    447     shapes = {arr.shape for arr in arrays}

ValueError: need at least one array to stack

## === cell 11
submission_df = pd.read_csv(DIR_INPUT + "/sample_submission.csv")

sub = submission_df[["image_id"]].merge(
    test_df[["image_id"]], on="image_id", how="left"
)
if sub.shape[0] != submission_df.shape[0]:
    raise RuntimeError("Submission image_id mismatch with sample_submission.")

submission_df[TARGET_COLS] = test_probs_mean.astype(np.float32)
submission_df.to_csv("submission.csv", index=False)
submission_df.head()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3765544146.py in <cell line: 0>()
      9     raise RuntimeError("Submission image_id mismatch with sample_submission.")
     10 
---> 11 submission_df[TARGET_COLS] = test_probs_mean.astype(np.float32)
     12 submission_df.to_csv("submission.csv", index=False)
     13 submission_df.head()

NameError: name 'test_probs_mean' is not defined

## === cell 12
assert os.path.exists("submission.csv")
assert submission_df.shape == (len(test_df), 1 + len(TARGET_COLS))
assert list(submission_df.columns) == ["image_id"] + TARGET_COLS

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/2562744393.py in <cell line: 0>()
      1 # Sanity checks (won't affect output)
----> 2 assert os.path.exists("submission.csv")
      3 assert submission_df.shape == (len(test_df), 1 + len(TARGET_COLS))
      4 assert list(submission_df.columns) == ["image_id"] + TARGET_COLS

AssertionError:
