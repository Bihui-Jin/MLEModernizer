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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

0.9429711711064372

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import random
import gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score



## === cell 1
DATA_DIR = Path("../input/plant-pathology-2020-fgvc7")
CLASS_NAMES = np.array(["healthy", "multiple_diseases", "rust", "scab"])
BATCH_SIZE = 8
IMAGE_SIZE = (512, 512)
TEST_SPLIT = 0.2
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)



## === cell 2
device = "cuda" if torch.cuda.is_available() else "cpu"




## === cell 3
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        backbone = torchvision.models.densenet201(pretrained=True)
        self.features = backbone.features
        self.relu = backbone.relu
        self.norm5 = backbone.norm5
        self.pool = backbone.pool
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(1024, 4)  # 4 target columns

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.norm5(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits  # raw logits for BCEWithLogitsLoss




## === cell 4
model = MyModel().to(device)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1253378788.py in <cell line: 0>()
----> 1 model = MyModel().to(device)
      2 

/tmp/ipykernel_55/991234383.py in __init__(self)
      6         # Remove original classifier
      7         self.features = backbone.features
----> 8         self.relu = backbone.relu
      9         self.norm5 = backbone.norm5
     10         self.pool = backbone.pool

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in __getattr__(self, name)
   1926             if name in modules:
   1927                 return modules[name]
-> 1928         raise AttributeError(
   1929             f"'{type(self).__name__}' object has no attribute '{name}'"
   1930         )

AttributeError: 'DenseNet' object has no attribute 'relu'

## === cell 5
ckpt_path = Path("../input/plant-pathology-models/acc_96_size_512.pth")
if ckpt_path.is_file():
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Checkpoint loaded.")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
else:
    print("No checkpoint found – using pretrained backbone.")




## === cell 6
class PlantPathologyDataset(Dataset):
    def __init__(self, root, df, transform=None, preload=False, is_test=False):
        self.root = root
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.preload = preload
        self.is_test = is_test
        if preload:
            self.images = []
            for idx in range(len(self.df)):
                img_path = (
                    self.root / "images" / (self.df.loc[idx, "image_id"] + ".jpg")
                )
                img = Image.open(str(img_path)).convert("RGB")
                img = img.resize(IMAGE_SIZE)
                self.images.append(img)
        else:
            self.images = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.images is None:
            img_path = self.root / "images" / (self.df.loc[idx, "image_id"] + ".jpg")
            img = Image.open(str(img_path)).convert("RGB")
            img = img.resize(IMAGE_SIZE)
        else:
            img = self.images[idx]

        if self.transform:
            img = self.transform(img)

        if self.is_test:
            return self.df.loc[idx, "image_id"], img
        else:
            targets = self.df.loc[idx, CLASS_NAMES].values.astype(np.float32)
            return img, torch.from_numpy(targets)




## === cell 7
train_df = pd.read_csv(DATA_DIR / "train.csv")
train_df[CLASS_NAMES] = train_df[CLASS_NAMES].astype(float)

train_df, val_df = train_test_split(
    train_df,
    test_size=TEST_SPLIT,
    random_state=RANDOM_SEED,
    stratify=train_df[CLASS_NAMES].values.argmax(axis=1),
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = PlantPathologyDataset(
    DATA_DIR, train_df, transform=transform, preload=False, is_test=False
)
val_dataset = PlantPathologyDataset(
    DATA_DIR, val_df, transform=transform, preload=False, is_test=False
)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3567747505.py in <cell line: 0>()
     32 
     33 criterion = nn.BCEWithLogitsLoss()
---> 34 optimizer = optim.Adam(model.parameters(), lr=1e-4)
     35 

NameError: name 'model' is not defined

## === cell 8
TARGET_AUROC = 0.9429711711064372
MAX_EPOCHS = 10
best_val_auc = 0.0

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    running_loss = 0.0
    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_targets.append(targets)

    val_logits = torch.cat(all_logits).numpy()
    val_targets = torch.cat(all_targets).numpy()
    val_probs = 1 / (1 + np.exp(-val_logits))  # sigmoid

    aucs = []
    for i in range(len(CLASS_NAMES)):
        try:
            auc = roc_auc_score(val_targets[:, i], val_probs[:, i])
        except ValueError:
            auc = np.nan  # class missing in batch
        aucs.append(auc)
    mean_auc = np.nanmean(aucs)

    print(
        f"Epoch {epoch:02d} | Train Loss: {epoch_loss:.4f} | Val Mean AUC: {mean_auc:.4f}"
    )

    if mean_auc > best_val_auc:
        best_val_auc = mean_auc
        torch.save(model.state_dict(), "best_model.pth")

    if best_val_auc >= TARGET_AUROC:
        print(f"Target AUC reached ({best_val_auc:.4f}); stopping training.")
        break

model.load_state_dict(torch.load("best_model.pth", map_location=device))



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2030407856.py in <cell line: 0>()
      5 
      6 for epoch in range(1, MAX_EPOCHS + 1):
----> 7     model.train()
      8     running_loss = 0.0
      9     for imgs, targets in train_loader:

NameError: name 'model' is not defined

## === cell 9
model.eval()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/490022066.py in <cell line: 0>()
----> 1 model.eval()
      2 

NameError: name 'model' is not defined

## === cell 10
test_df = pd.read_csv(DATA_DIR / "test.csv")
submission_dataset = PlantPathologyDataset(
    DATA_DIR, test_df, transform=transforms.ToTensor(), preload=True, is_test=True
)
submission_loader = DataLoader(
    submission_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1
)



## === cell 11
image_ids = []
preds = []
with torch.no_grad():
    for batch_ids, batch_imgs in submission_loader:
        batch_imgs = batch_imgs.to(device)
        logits = model(batch_imgs)
        probs = torch.sigmoid(logits).cpu().numpy()
        image_ids.extend(batch_ids)
        preds.append(probs)

preds = np.concatenate(preds, axis=0)  # shape (N, 4)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3579048422.py in <cell line: 0>()
      4     for batch_ids, batch_imgs in submission_loader:
      5         batch_imgs = batch_imgs.to(device)
----> 6         logits = model(batch_imgs)
      7         probs = torch.sigmoid(logits).cpu().numpy()
      8         image_ids.extend(batch_ids)

NameError: name 'model' is not defined

## === cell 12
sub = pd.DataFrame(
    {
        "image_id": image_ids,
        "healthy": preds[:, 0],
        "multiple_diseases": preds[:, 1],
        "rust": preds[:, 2],
        "scab": preds[:, 3],
    }
)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3006822458.py in <cell line: 0>()
      2     {
      3         "image_id": image_ids,
----> 4         "healthy": preds[:, 0],
      5         "multiple_diseases": preds[:, 1],
      6         "rust": preds[:, 2],

TypeError: list indices must be integers or slices, not tuple

## === cell 13
sub.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")
sub.head()

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2057268762.py in <cell line: 0>()
----> 1 sub.to_csv("submission.csv", index=False)
      2 print("Submission file saved as submission.csv")
      3 sub.head()

NameError: name 'sub' is not defined
