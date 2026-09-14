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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.10

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
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.8274918088498221

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random, gc
import numpy as np, pandas as pd, cv2
from glob import glob
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

seed = 42
print(f"setting everything to seed {seed}")
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## === cell 1
data_dir = "../input/alaska2-image-steganalysis"
sample_size = 75000
val_ratio = 0.25
img_size = 512
batch_size = 32
num_workers = 2
num_epochs = 2  # short training to get a usable model
device = "cuda" if torch.cuda.is_available() else "cpu"

folder_names = ["Cover/", "JMiPOD/", "JUNIWARD/", "UERD/"]  # label 0‑3
train_fn, val_fn = [], []
train_labels, val_labels = [], []

for label, folder in enumerate(folder_names):
    files = sorted(glob(f"{data_dir}/{folder}/*.jpg"))[:sample_size]
    np.random.shuffle(files)
    split = int(len(files) * val_ratio)
    val_fn.extend(files[:split])
    val_labels.extend([label] * len(files[:split]))
    train_fn.extend(files[split:])
    train_labels.extend([label] * len(files[split:]))

train_df = pd.DataFrame({"ImageFileName": train_fn, "Label": train_labels})
val_df = pd.DataFrame({"ImageFileName": val_fn, "Label": val_labels})



## === cell 2
train_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
    ]
)

test_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((img_size, img_size)),
        torchvision.transforms.ToTensor(),
    ]
)


class Alaska2Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn = self.df.loc[idx, "ImageFileName"]
        label = self.df.loc[idx, "Label"]
        img = cv2.imread(fn)[:, :, ::-1]  # BGR to RGB
        if self.transform:
            img = self.transform(img)
        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}


class Alaska2TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn = self.df.loc[idx, "ImageFileName"]
        img = cv2.imread(fn)[:, :, ::-1]
        if self.transform:
            img = self.transform(img)
        return {"image": img, "filename": fn}




## === cell 3
from efficientnet_pytorch import EfficientNet


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained("efficientnet-b0")
        self.head = nn.Linear(1280, 4)  # 4 classes

    def forward(self, x):
        feats = self.backbone.extract_features(x)
        pooled = F.adaptive_avg_pool2d(feats, 1).view(x.size(0), -1)
        return self.head(pooled)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_55/2643483625.py in <cell line: 0>()
----> 1 from efficientnet_pytorch import EfficientNet
      2 
      3 
      4 class Net(nn.Module):
      5     def __init__(self):

ModuleNotFoundError: No module named 'efficientnet_pytorch'

## === cell 4
train_dataset = Alaska2Dataset(train_df, transform=train_transform)
val_dataset = Alaska2Dataset(val_df, transform=test_transform)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

ckpt_path = "../input/alaska/epoch_14_val_loss_6.63_auc_0.824.pth"
if os.path.isfile(ckpt_path):
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded checkpoint")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3644964480.py in <cell line: 0>()
     19 
     20 # model, loss, optimizer
---> 21 model = Net().to(device)
     22 criterion = nn.CrossEntropyLoss()
     23 optimizer = optim.AdamW(model.parameters(), lr=1e-4)

NameError: name 'Net' is not defined

## === cell 5
def alaska_weighted_auc(y_true, y_score):
    tpr_thr = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
    seg_areas = np.diff(tpr_thr)
    norm = np.dot(seg_areas, weights)

    metric = 0.0
    for i, w in enumerate(weights):
        lo, hi = tpr_thr[i], tpr_thr[i + 1]
        mask = (tpr > lo) & (tpr <= hi)
        if not np.any(mask):
            continue
        fpr_seg = np.concatenate([fpr[mask], [1.0]])
        tpr_seg = np.concatenate([tpr[mask], [hi]])
        tpr_adj = tpr_seg - lo
        seg_auc = metrics.auc(fpr_seg, tpr_adj)
        metric += seg_auc * w
    return metric / norm




## === cell 6
train_losses, val_losses = [], []
best_auc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
        imgs = batch["image"].to(device)
        lbls = batch["label"].to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_train_loss)

    model.eval()
    val_running_loss = 0.0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device)
            out = model(imgs)
            loss = criterion(out, lbls)
            val_running_loss += loss.item() * imgs.size(0)

            probs = F.softmax(out, dim=1).cpu().numpy()
            pred_labels = probs.argmax(axis=1)
            binary_true = (lbls.cpu().numpy() != 0).astype(int)
            binary_score = np.where(
                pred_labels == 0, probs[:, 0], probs[:, 1:].sum(axis=1)
            )
            all_labels.extend(binary_true)
            all_preds.extend(binary_score)

    epoch_val_loss = val_running_loss / len(val_loader.dataset)
    val_losses.append(epoch_val_loss)

    auc = alaska_weighted_auc(np.array(all_labels), np.array(all_preds))
    print(
        f"Epoch {epoch+1}: TrainLoss={epoch_train_loss:.4f} ValLoss={epoch_val_loss:.4f} WeightedAUC={auc:.4f}"
    )

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved new best model")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/121992939.py in <cell line: 0>()
      4 
      5 for epoch in range(num_epochs):
----> 6     model.train()
      7     running_loss = 0.0
      8     for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):

NameError: name 'model' is not defined

## === cell 7
test_files = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_df = pd.DataFrame({"ImageFileName": test_files})
test_dataset = Alaska2TestDataset(test_df, transform=test_transform)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True
)

if os.path.isfile("best_model.pth"):
    model.load_state_dict(torch.load("best_model.pth", map_location=device))

model.eval()
test_preds = []
filenames = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test inference"):
        imgs = batch["image"].to(device)
        outs = model(imgs)
        probs = F.softmax(outs, dim=1).cpu().numpy()
        pred_labels = probs.argmax(axis=1)
        scores = np.where(pred_labels == 0, probs[:, 0], probs[:, 1:].sum(axis=1))
        test_preds.extend(scores)
        filenames.extend(batch["filename"])

submission = pd.DataFrame(
    {"Id": [os.path.basename(p) for p in filenames], "Label": test_preds}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
print(submission.head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/862987215.py in <cell line: 0>()
     11     model.load_state_dict(torch.load("best_model.pth", map_location=device))
     12 
---> 13 model.eval()
     14 test_preds = []
     15 filenames = []

NameError: name 'model' is not defined
