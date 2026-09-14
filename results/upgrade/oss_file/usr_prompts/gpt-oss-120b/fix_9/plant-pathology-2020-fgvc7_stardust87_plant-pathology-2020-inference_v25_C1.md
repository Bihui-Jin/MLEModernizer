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

0.9599834514754224

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, time, shutil, warnings

warnings.filterwarnings("ignore")

import numpy as np, pandas as pd
import albumentations as A
import cv2

import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import timm
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from tqdm.notebook import tqdm

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True  # use TF32 where possible for speed




## === cell 1
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"
IMAGE_INPUT = os.path.join(DIR_INPUT, "images")  # corrected path
SEED = 42
N_FOLDS = 1  # reduced folds to cut training time
N_EPOCHS = 4  # reduced epochs while keeping the same training loop
BATCH_SIZE = 8
IMAGE_SIZE = (409, 273)  # width, height

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




## === cell 2
IMAGE_CACHE = {}


def load_and_cache_image(img_id):
    """Load an image, resize it once, and store it in IMAGE_CACHE."""
    if img_id in IMAGE_CACHE:
        return IMAGE_CACHE[img_id]
    img_name = f"{img_id}.jpg"
    img_path = os.path.join(IMAGE_INPUT, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:  # fallback for missing files
        img = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.uint8)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)
    IMAGE_CACHE[img_id] = img
    return img


class PlantDataset(Dataset):
    """
    Caches all images in RAM at initialization to avoid repeated disk I/O.
    Images are resized during caching, and labels are pre‑converted to tensors
    to eliminate per‑sample pandas look‑ups.
    """

    def __init__(self, df, transforms=None, test_set=False):
        self.df = df.reset_index(drop=True)
        self.test_set = test_set
        self.transforms = transforms
        if self.transforms is None:
            self.transforms = A.Compose(
                [A.Normalize(p=1.0), A.pytorch.transforms.ToTensorV2(p=1.0)]
            )
        self.images = [load_and_cache_image(img_id) for img_id in self.df["image_id"]]
        if not self.test_set:
            self.labels = torch.from_numpy(
                self.df[["healthy", "multiple_diseases", "rust", "scab"]].values.astype(
                    np.float32
                )
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.images[idx]

        transformed = self.transforms(image=image)
        image = transformed["image"]

        if not self.test_set:
            labels = self.labels[idx]
            return image, labels
        else:
            return image




## === cell 3
def trim_network_at_index(network, index=-1):
    assert index < 0, f"Param index must be negative. Received {index}"
    return nn.Sequential(*list(network.children())[:index])




## === cell 4
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




## === cell 5
transforms_train = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.Normalize(p=1.0),
        A.pytorch.transforms.ToTensorV2(p=1.0),
    ]
)

transforms_valid = A.Compose(
    [
        A.Normalize(p=1.0),
        A.pytorch.transforms.ToTensorV2(p=1.0),
    ]
)




## === cell 6
train_df = pd.read_csv(os.path.join(DIR_INPUT, "train.csv"))
test_df = pd.read_csv(os.path.join(DIR_INPUT, "test.csv"))

test_dataset = PlantDataset(df=test_df, test_set=True, transforms=transforms_valid)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,  # avoid duplicate caching across workers
    pin_memory=True,
    persistent_workers=False,
)




## === cell 7
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
    print(f"\n=== Fold {fold} ===")
    df_tr = train_df.iloc[train_idx].reset_index(drop=True)
    df_val = train_df.iloc[val_idx].reset_index(drop=True)

    train_dataset = PlantDataset(df=df_tr, transforms=transforms_train, test_set=False)
    val_dataset = PlantDataset(df=df_val, transforms=transforms_valid, test_set=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # caching already in RAM
        pin_memory=True,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    model = PlantModel().to(device)
    model = torch.compile(model, mode="max-autotune")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    scaler = torch.cuda.amp.GradScaler()

    best_val_auc = 0.0
    for epoch in range(N_EPOCHS):
        model.train()
        epoch_loss = 0.0
        for images, targets in tqdm(
            train_loader, leave=False, desc=f"Train epoch {epoch+1}"
        ):
            images, targets = images.to(device, non_blocking=True), targets.to(
                device, non_blocking=True
            )

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
        model.eval()
        all_targets, all_preds = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device, non_blocking=True)
                with torch.cuda.amp.autocast():
                    logits = model(images)
                preds = torch.sigmoid(logits).cpu().numpy()
                all_preds.append(preds)
                all_targets.append(targets.numpy())
        val_preds = np.concatenate(all_preds, axis=0)
        val_targets = np.concatenate(all_targets, axis=0)
        aucs = []
        for i in range(4):
            try:
                auc = roc_auc_score(val_targets[:, i], val_preds[:, i])
            except ValueError:
                auc = 0.5
            aucs.append(auc)
        mean_auc = np.mean(aucs)
        print(
            f"Epoch {epoch+1}/{N_EPOCHS} - loss: {epoch_loss/len(train_loader):.4f} - val AUC: {mean_auc:.4f}"
        )
        if mean_auc > best_val_auc:
            best_val_auc = mean_auc
            torch.save(model.state_dict(), f"modelF{fold}.pth")
    print(f"Best val AUC for fold {fold}: {best_val_auc:.4f}")




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/591127731.py in <cell line: 0>()
----> 1 kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
      2 for fold, (train_idx, val_idx) in enumerate(kf.split(train_df)):
      3     print(f"\n=== Fold {fold} ===")
      4     df_tr = train_df.iloc[train_idx].reset_index(drop=True)
      5     df_val = train_df.iloc[val_idx].reset_index(drop=True)

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in __init__(self, n_splits, shuffle, random_state)
    449 
    450     def __init__(self, n_splits=5, *, shuffle=False, random_state=None):
--> 451         super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    452 
    453     def _iter_test_indices(self, X, y=None, groups=None):

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in __init__(self, n_splits, shuffle, random_state)
    296 
    297         if n_splits <= 1:
--> 298             raise ValueError(
    299                 "k-fold cross-validation requires at least one"
    300                 " train/test split by setting n_splits=2 or more,"

ValueError: k-fold cross-validation requires at least one train/test split by setting n_splits=2 or more, got n_splits=1.

## === cell 8
def test_model(model_name, testloader):
    model_path = f"./{model_name}.pth"
    model = PlantModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model = torch.compile(model, mode="max-autotune")
    model.eval()

    probs = []
    with torch.no_grad():
        for images in tqdm(testloader, leave=False, desc="Inference"):
            images = images.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                logits = model(images)
            prob = torch.sigmoid(logits).cpu().numpy()
            probs.append(prob)
    probs = np.concatenate(probs, axis=0)
    return probs




## === cell 9
all_fold_probs = []
for fold in range(N_FOLDS):
    probs_fold = test_model(f"modelF{fold}", test_loader)
    all_fold_probs.append(probs_fold)

test_probs_mean = np.mean(np.stack(all_fold_probs, axis=0), axis=0)  # (n_samples, 4)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2101227971.py in <cell line: 0>()
      1 all_fold_probs = []
      2 for fold in range(N_FOLDS):
----> 3     probs_fold = test_model(f"modelF{fold}", test_loader)
      4     all_fold_probs.append(probs_fold)
      5 

/tmp/ipykernel_55/2402436615.py in test_model(model_name, testloader)
      2     model_path = f"./{model_name}.pth"
      3     model = PlantModel()
----> 4     model.load_state_dict(torch.load(model_path, map_location=device))
      5     model = model.to(device)
      6     model = torch.compile(model, mode="max-autotune")

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: './modelF0.pth'

## === cell 10
submission_df = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))
submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = test_probs_mean
submission_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1964392488.py in <cell line: 0>()
      1 submission_df = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))
----> 2 submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = test_probs_mean
      3 submission_df.to_csv("submission.csv", index=False)
      4 print("Submission saved to submission.csv")

NameError: name 'test_probs_mean' is not defined
