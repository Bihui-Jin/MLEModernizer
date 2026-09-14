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
import os, time
import numpy as np, pandas as pd
import albumentations as A
import cv2
import torch, torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split



## === cell 1
DIR_INPUT = "/kaggle/input/plant-pathology-2020-fgvc7"
SEED = 42
N_EPOCHS = 3
BATCH_SIZE = 16
IMAGE_SIZE = (273, 409)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)




## === cell 2
class PlantDataset(Dataset):
    def __init__(self, df, transforms=None, test_set=False):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.test_set = test_set
        if self.transforms is None:
            self.transforms = A.Compose([ToTensorV2(p=1.0)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(
            DIR_INPUT, "images", f"{self.df.loc[idx, 'image_id']}.jpg"
        )
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMAGE_SIZE)
        transformed = self.transforms(image=img)
        img = transformed["image"]
        if not self.test_set:
            labels = self.df.loc[
                idx, ["healthy", "multiple_diseases", "rust", "scab"]
            ].values
            labels = torch.from_numpy(labels.astype(np.float32))
            return img, labels
        else:
            return img




## === cell 3
class PlantModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = torchvision.models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.logit = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = F.dropout(x, p=0.25, training=self.training)
        return self.logit(x)




## === cell 4
train_df = pd.read_csv(os.path.join(DIR_INPUT, "train.csv"))
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=SEED)

train_transforms = A.Compose(
    [
        A.Resize(*IMAGE_SIZE),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(),
        ToTensorV2(),
    ]
)

val_transforms = A.Compose([A.Resize(*IMAGE_SIZE), A.Normalize(), ToTensorV2()])

train_dataset = PlantDataset(train_df, transforms=train_transforms, test_set=False)
val_dataset = PlantDataset(val_df, transforms=val_transforms, test_set=False)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)


## === cell 5
model = PlantModel().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

best_val_auc = 0.0
for epoch in range(N_EPOCHS):
    model.train()
    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
    model.eval()
    all_targets, all_preds = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())
    val_pred = np.concatenate(all_preds)
    val_true = np.concatenate(all_targets)
    aucs = []
    for i in range(val_true.shape[1]):
        try:
            auc = roc_auc_score(val_true[:, i], val_pred[:, i])
        except ValueError:
            auc = 0.5
        aucs.append(auc)
    val_auc = np.mean(aucs)
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), "model.pth")
    print(f"Epoch {epoch+1}/{N_EPOCHS} - Val AUC: {val_auc:.4f}")


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2333917684.py in <cell line: 0>()
     30     for i in range(val_true.shape[1]):
     31         try:
---> 32             auc = roc_auc_score(val_true[:, i], val_pred[:, i])
     33         except ValueError:
     34             auc = 0.5

NameError: name 'roc_auc_score' is not defined

## === cell 6
test_transforms = A.Compose([A.Resize(*IMAGE_SIZE), A.Normalize(), ToTensorV2()])
test_df = pd.read_csv(os.path.join(DIR_INPUT, "test.csv"))
test_dataset = PlantDataset(test_df, transforms=test_transforms, test_set=True)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
)


## === cell 7
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()
test_probs = []
with torch.no_grad():
    for imgs in test_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy()
        test_probs.append(probs)
test_probs = np.concatenate(test_probs, axis=0)  # shape (num_test, 4)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3031316857.py in <cell line: 0>()
      1 # Inference
----> 2 model.load_state_dict(torch.load("model.pth", map_location=device))
      3 model.eval()
      4 test_probs = []
      5 with torch.no_grad():

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

FileNotFoundError: [Errno 2] No such file or directory: 'model.pth'

## === cell 8
submission = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))
submission.iloc[:, 1:] = test_probs
submission.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3556623868.py in <cell line: 0>()
      1 # Build submission
      2 submission = pd.read_csv(os.path.join(DIR_INPUT, "sample_submission.csv"))
----> 3 submission.iloc[:, 1:] = test_probs
      4 submission.to_csv("submission.csv", index=False)
      5 print("Submission written to submission.csv")

NameError: name 'test_probs' is not defined

## === cell 9
submission.head()
