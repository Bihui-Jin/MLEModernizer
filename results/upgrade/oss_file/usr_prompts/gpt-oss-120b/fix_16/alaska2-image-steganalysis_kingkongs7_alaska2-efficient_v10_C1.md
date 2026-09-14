# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.5748

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.58227) has done: 'The fix ensures the script correctly locates the dataset directories, uses a realistic image size for ResNet‑18, and adjusts training parameters so the loaders are populated and a proper CSV with matching row counts is written. These changes resolve the DataLoader “num_samples=0” error, prevent the submission length mismatch, and modestly improve model training while keeping the core logic unchanged.'
- What this solution (achieved 0.58318) has done: 'The changes lower the training set size (sample_size = 5000 per class) to keep the same training loop, model, and augmentation logic while drastically cutting I/O and compute time. A fixed random seed is added for reproducibility, and the DataLoader workers are capped to the available CPU count. These tweaks preserve the exact architecture, loss, optimizer, and evaluation metric, ensuring identical semantics with a runtime that fits the 600 s limit.'
- What this solution (achieved 0.5748) has done: 'I increase the training set size and use a slightly larger validation split to give the model more data while keeping the same architecture. At inference I add a simple test‑time augmentation (horizontal flip) and average the stego probabilities, which usually boosts the weighted AUC without altering the core training loop.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

data_dir = os.getenv("DATA_DIR", "/kaggle/input/alaska2-image-steganalysis")
sample_size = 8000  # ↑ more training images per class
val_ratio = 0.20  # slightly larger validation split
img_size = 224
batch_size = 32
num_workers = min(12, os.cpu_count() or 4)
num_epochs = 12
device = "cuda" if torch.cuda.is_available() else "cpu"

folder_names = ["Cover/", "JMiPOD/", "JUNIWARD/", "UERD/"]  # labels 0‑3

train_files, val_files = [], []
train_labels, val_labels = [], []

for label, folder in enumerate(folder_names):
    files = sorted(glob.glob(os.path.join(data_dir, folder, "*.jpg")))
    files = files[: min(sample_size, len(files))]
    random.shuffle(files)
    split = int(len(files) * val_ratio)
    val_files.extend(files[:split])
    val_labels.extend([label] * len(files[:split]))
    train_files.extend(files[split:])
    train_labels.extend([label] * len(files[split:]))

train_df = pd.DataFrame({"ImageFileName": train_files, "Label": train_labels})
val_df = pd.DataFrame({"ImageFileName": val_files, "Label": val_labels})

train_transform = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.RandomCrop(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


class Alaska2Dataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.iloc[idx]["ImageFileName"]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        if self.is_test:
            return {"image": img, "filename": os.path.basename(path)}
        else:
            label = int(self.df.iloc[idx]["Label"])
            return {
                "image": img,
                "label": torch.tensor(label, dtype=torch.long),
                "filename": os.path.basename(path),
            }


train_dataset = Alaska2Dataset(train_df, transform=train_transform, is_test=False)
val_dataset = Alaska2Dataset(val_df, transform=train_transform, is_test=False)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)




## === cell 1
def alaska_weighted_auc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    area = 0.0
    for i in range(1, len(tpr)):
        tpr_mid = (tpr[i] + tpr[i - 1]) / 2.0
        w = 2.0 if tpr_mid <= 0.4 else 1.0
        segment = (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2.0
        area += w * segment
    return area / 1.4  # normalization factor (2*0.4 + 1*0.6 = 1.4)


model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 4)
model = torch.compile(model)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=2, verbose=True
)




## === cell 2
scaler = torch.cuda.amp.GradScaler()
best_auc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
        imgs = batch["image"].to(device, non_blocking=True)
        lbls = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(imgs)
            loss = criterion(out, lbls)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)

    epoch_train_loss = running_loss / len(train_loader.dataset)

    model.eval()
    val_running_loss = 0.0
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
            imgs = batch["image"].to(device, non_blocking=True)
            lbls = batch["label"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
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
    auc = alaska_weighted_auc(np.array(all_labels), np.array(all_preds))
    print(
        f"Epoch {epoch+1}: TrainLoss={epoch_train_loss:.4f} "
        f"ValLoss={epoch_val_loss:.4f} WeightedAUC={auc:.4f}"
    )

    scheduler.step(epoch_val_loss)

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), "best_model.pth")
        print("Saved new best model")




## === cell 3
test_files = sorted(glob.glob(os.path.join(data_dir, "Test", "*.jpg")))
test_df = pd.DataFrame({"ImageFileName": test_files})
test_dataset = Alaska2Dataset(test_df, transform=test_transform, is_test=True)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

ids, scores = [], []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test Inference"):
        imgs = batch["image"].to(device, non_blocking=True)

        probs_orig = torch.softmax(model(imgs), dim=1)[:, 1:].sum(dim=1)

        imgs_flipped = torch.flip(imgs, dims=[-1])  # flip width dimension
        probs_flip = torch.softmax(model(imgs_flipped), dim=1)[:, 1:].sum(dim=1)

        probs = (probs_orig + probs_flip) / 2.0

        ids.extend(batch["filename"])
        scores.extend(probs.cpu().numpy().tolist())

submission = pd.DataFrame({"Id": ids, "Label": scores})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
