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

0.8097877448852011

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random, gc
import numpy as np, pandas as pd
import cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt
from glob import glob

from albumentations import (
    Compose,
    HorizontalFlip,
    VerticalFlip,
    Resize,
    JpegCompression,
    ToFloat,
)
from albumentations.pytorch import ToTensorV2

from efficientnet_pytorch import EfficientNet

seed = 42
print(f"setting everything to seed {seed}")
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/1604390593.py in <cell line: 0>()
     10 
     11 # Albumentations (already installed; no version downgrade)
---> 12 from albumentations import (
     13     Compose,
     14     HorizontalFlip,

ImportError: cannot import name 'JpegCompression' from 'albumentations' (/usr/local/lib/python3.11/dist-packages/albumentations/__init__.py)

## === cell 1
default_dir = "/kaggle/input/alaska2-image-steganalysis"
data_dir = (
    default_dir if os.path.isdir(default_dir) else "../input/alaska2-image-steganalysis"
)

sample_size = 75000  # number of images taken from each folder
val_ratio = 0.25
val_size = int(sample_size * val_ratio)

train_fn, val_fn = [], []
train_labels, val_labels = [], []

folder_names = ["Cover/", "JMiPOD/", "JUNIWARD/", "UERD/"]  # label 0‑3
for label, folder in enumerate(folder_names):
    all_files = sorted(glob(f"{data_dir}/{folder}/*.jpg"))[:sample_size]
    np.random.shuffle(all_files)
    train_files = all_files[val_size:]
    val_files = all_files[:val_size]

    train_fn.extend(train_files)
    train_labels.extend([label] * len(train_files))

    val_fn.extend(val_files)
    val_labels.extend([label] * len(val_files))

assert len(train_fn) == len(train_labels)
assert len(val_fn) == len(val_labels)

train_df = pd.DataFrame({"ImageFileName": train_fn, "Label": train_labels})
val_df = pd.DataFrame({"ImageFileName": val_fn, "Label": val_labels})

print("Training set size:", len(train_df))
print("Validation set size:", len(val_df))




## === cell 2
class Alaska2Dataset(Dataset):
    """Dataset used for training / validation."""

    def __init__(self, df, augmentations=None):
        self.df = df.reset_index(drop=True)
        self.augment = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fn, label = row["ImageFileName"], int(row["Label"])
        img = cv2.imread(fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augment:
            img = self.augment(image=img)

        return img, label


img_size = 512
AUGMENTATIONS_TRAIN = Compose(
    [
        Resize(img_size, img_size, p=1.0),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        JpegCompression(quality_lower=75, quality_upper=100, p=0.5),
        ToFloat(max_value=255),
        ToTensorV2(),
    ]
)

AUGMENTATIONS_TEST = Compose(
    [Resize(img_size, img_size, p=1.0), ToFloat(max_value=255), ToTensorV2()]
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1872549023.py in <cell line: 0>()
     27         VerticalFlip(p=0.5),
     28         HorizontalFlip(p=0.5),
---> 29         JpegCompression(quality_lower=75, quality_upper=100, p=0.5),
     30         ToFloat(max_value=255),
     31         ToTensorV2(),

NameError: name 'JpegCompression' is not defined

## === cell 3
temp_df = train_df.sample(64).reset_index(drop=True)
vis_dataset = Alaska2Dataset(temp_df, augmentations=AUGMENTATIONS_TEST)
vis_loader = DataLoader(vis_dataset, batch_size=64, shuffle=False, num_workers=0)

imgs, lbls = next(iter(vis_loader))
imgs = imgs["image"].permute(0, 2, 3, 1).cpu().numpy()
grid_w, grid_h = 16, 4
fig, axs = plt.subplots(grid_h, grid_w, figsize=(grid_w + 1, grid_h + 1))
for i, (im, lab) in enumerate(zip(imgs, lbls)):
    ax = axs[i // grid_w, i % grid_w]
    ax.imshow(im)
    ax.set_title(str(lab.item()))
    ax.axis("off")
plt.suptitle("0:COVER, 1:JMiPOD, 2:JUNIWARD, 3:UERD")
plt.show()
del imgs, lbls, vis_loader
gc.collect()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1663242927.py in <cell line: 0>()
      1 # Quick visual check (optional but harmless)
      2 temp_df = train_df.sample(64).reset_index(drop=True)
----> 3 vis_dataset = Alaska2Dataset(temp_df, augmentations=AUGMENTATIONS_TEST)
      4 vis_loader = DataLoader(vis_dataset, batch_size=64, shuffle=False, num_workers=0)
      5 

NameError: name 'AUGMENTATIONS_TEST' is not defined

## === cell 4
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = EfficientNet.from_pretrained("efficientnet-b0")
        self.classifier = nn.Linear(1280, 4)  # 4 classes

    def forward(self, x):
        feats = self.backbone.extract_features(x)
        pooled = F.adaptive_avg_pool2d(feats, 1).reshape(x.size(0), -1)
        return self.classifier(pooled)




## === cell 5
batch_size = 8
num_workers = 4

train_dataset = Alaska2Dataset(train_df, augmentations=AUGMENTATIONS_TRAIN)
val_dataset = Alaska2Dataset(val_df, augmentations=AUGMENTATIONS_TEST)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size * 2,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4261024612.py in <cell line: 0>()
      2 num_workers = 4
      3 
----> 4 train_dataset = Alaska2Dataset(train_df, augmentations=AUGMENTATIONS_TRAIN)
      5 val_dataset = Alaska2Dataset(val_df, augmentations=AUGMENTATIONS_TEST)
      6 

NameError: name 'AUGMENTATIONS_TRAIN' is not defined

## === cell 6
def alaska_weighted_auc(y_true, y_score):
    """Weighted AUC as defined by the competition."""
    tpr_thr = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)

    areas = np.diff(tpr_thr)
    norm = np.dot(areas, weights)
    metric = 0.0

    for i, w in enumerate(weights):
        lo, hi = tpr_thr[i], tpr_thr[i + 1]
        mask = (tpr > lo) & (tpr <= hi)

        fpr_ext = np.concatenate([fpr[mask], [1.0]])
        tpr_ext = np.concatenate([tpr[mask], [hi]])

        tpr_shifted = tpr_ext - lo
        seg_auc = metrics.auc(fpr_ext, tpr_shifted)
        metric += seg_auc * w

    return metric / norm




## === cell 7
criterion = nn.CrossEntropyLoss()
num_epochs = 2  # a small number of epochs to keep runtime low
train_losses, val_losses = [], []

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc="Train", leave=False):
        imgs, targets = batch
        imgs = imgs["image"].to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_train_loss = running_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    model.eval()
    val_running = 0.0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Valid", leave=False):
            imgs, targets = batch
            imgs = imgs["image"].to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.long)

            outputs = model(imgs)
            loss = criterion(outputs, targets)
            val_running += loss.item()

            probs = F.softmax(outputs, dim=1).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(targets.cpu().numpy())

    epoch_val_loss = val_running / len(val_loader)
    val_losses.append(epoch_val_loss)

    preds = np.concatenate(all_preds)
    true_labels = np.concatenate(all_labels)
    binary_true = (true_labels != 0).astype(int)  # 0 = clean, 1 = stego
    stego_score = preds[:, 1:].sum(axis=1)
    auc = alaska_weighted_auc(binary_true, stego_score)

    print(
        f"Train loss: {epoch_train_loss:.4f} | Val loss: {epoch_val_loss:.4f} | Weighted AUC: {auc:.4f}"
    )

    torch.save(
        model.state_dict(),
        f"epoch_{epoch+1}_val_loss_{epoch_val_loss:.3f}_auc_{auc:.3f}.pth",
    )



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4057498077.py in <cell line: 0>()
      5 for epoch in range(num_epochs):
      6     print(f"Epoch {epoch+1}/{num_epochs}")
----> 7     model.train()
      8     running_loss = 0.0
      9     for batch in tqdm(train_loader, desc="Train", leave=False):

NameError: name 'model' is not defined

## === cell 8
plt.figure(figsize=(12, 5))
plt.plot(train_losses, label="Train loss", color="r")
plt.plot(val_losses, label="Valid loss", color="b")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over epochs")
plt.legend()
plt.show()




## === cell 9
class Alaska2TestDataset(Dataset):
    """Dataset for inference on the test split."""

    def __init__(self, df, augmentations=None):
        self.df = df.reset_index(drop=True)
        self.augment = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn = self.df.iloc[idx]["ImageFileName"]
        img = cv2.imread(fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.augment:
            img = self.augment(image=img)

        return img


test_files = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_df = pd.DataFrame({"ImageFileName": test_files})

test_dataset = Alaska2TestDataset(test_df, augmentations=AUGMENTATIONS_TEST)
test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1563848811.py in <cell line: 0>()
     23 test_df = pd.DataFrame({"ImageFileName": test_files})
     24 
---> 25 test_dataset = Alaska2TestDataset(test_df, augmentations=AUGMENTATIONS_TEST)
     26 test_loader = DataLoader(
     27     test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True

NameError: name 'AUGMENTATIONS_TEST' is not defined

## === cell 10
model.eval()
test_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test inference"):
        imgs = batch["image"].to(device, dtype=torch.float)

        img_hflip = torch.flip(imgs, dims=[3])  # horizontal flip
        img_vflip = torch.flip(imgs, dims=[2])  # vertical flip

        out = model(imgs)
        out += model(img_hflip) * 0.25
        out += model(img_vflip) * 0.25
        out += model(torch.flip(imgs, dims=[2, 3])) * 0.5  # both flips

        probs = F.softmax(out, dim=1).cpu().numpy()
        test_preds.append(probs)

test_preds = np.concatenate(test_preds)
clean_prob = test_preds[:, 0]
stego_score = test_preds[:, 1:].sum(axis=1)
final_score = np.where(stego_score > clean_prob, stego_score, clean_prob)

test_df["Id"] = test_df["ImageFileName"].apply(lambda x: os.path.basename(x))
test_df["Label"] = final_score
submission = test_df[["Id", "Label"]]
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(submission.head())

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3019849120.py in <cell line: 0>()
----> 1 model.eval()
      2 test_preds = []
      3 
      4 with torch.no_grad():
      5     for batch in tqdm(test_loader, desc="Test inference"):

NameError: name 'model' is not defined
