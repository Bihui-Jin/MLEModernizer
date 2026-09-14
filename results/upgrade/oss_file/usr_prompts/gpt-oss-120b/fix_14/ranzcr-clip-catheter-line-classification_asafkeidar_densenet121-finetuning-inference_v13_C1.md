# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

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
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, math, time, torch.multiprocessing
import numpy as np, pandas as pd
import torch, torch.nn as nn, torchvision, torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_auc_score

torch.multiprocessing.set_sharing_strategy("file_system")

np.random.seed(42)
torch.manual_seed(42)

torch.set_num_threads(os.cpu_count() or 1)

torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True  # already set, kept for completeness

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 1
ROOT_DIR = "/kaggle/input/ranzcr-clip-catheter-line-classification/test"
TRAIN_ROOT = "/kaggle/input/ranzcr-clip-catheter-line-classification/train"
SAMPLE_SUB_PATH = (
    "/kaggle/input/ranzcr-clip-catheter-line-classification/sample_submission.csv"
)
TRAIN_CSV_PATH = "/kaggle/input/ranzcr-clip-catheter-line-classification/train.csv"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
train_df = pd.read_csv(TRAIN_CSV_PATH)

TARGET_COLUMNS = [col for col in sample_sub.columns if col != "StudyInstanceUID"]

DEBUG = False
if DEBUG:
    BATCH_SIZE = 4
    EPOCHS = 2
else:
    BATCH_SIZE = 256 if torch.cuda.is_available() else 64
    EPOCHS = 3  # unchanged logic
IMG_SIZE = 256



## === cell 2
import torchvision.io as io
import torchvision.transforms.functional as TF


class RanzcrClipTrainDataset(Dataset):
    """Dataset returning (image tensor, multi‑label vector) for training/validation."""

    def __init__(self, df, root_dir):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.ids = self.df["StudyInstanceUID"].values
        self.labels = self.df[TARGET_COLUMNS].values.astype(np.float32)

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.ids)

    def _load_image(self, uid):
        img_path = os.path.join(self.root_dir, f"{uid}.jpg")
        img = io.read_image(img_path)  # shape: (3, H, W), uint8
        img = img.float() / 255.0  # to float in [0,1]
        img = TF.resize(img, [IMG_SIZE, IMG_SIZE])
        img = (img - self.mean) / self.std  # normalize
        return img

    def __getitem__(self, idx):
        uid = self.ids[idx]
        image = self._load_image(uid)
        label = torch.from_numpy(self.labels[idx])
        return image, label




## === cell 3
common_transforms = None  # not used; transforms are applied inside the dataset



## === cell 4
full_dataset = RanzcrClipTrainDataset(train_df, TRAIN_ROOT)
val_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

num_workers = min(16, os.cpu_count() or 1)
pin_mem = torch.cuda.is_available()  # pin_memory helps when using GPU

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_mem,
    prefetch_factor=4,  # fetch more batches ahead to keep GPU fed
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_mem,
    prefetch_factor=4,
    persistent_workers=True,
)




## === cell 5
class CustomPretrainedModel(nn.Module):
    def __init__(self, num_targets):
        super().__init__()
        self.backbone = torchvision.models.densenet121(pretrained=True)
        n_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(n_features, num_targets)

    def forward(self, x):
        return self.backbone(x)


model = CustomPretrainedModel(num_targets=len(TARGET_COLUMNS)).to(device)

model = torch.compile(model)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

use_amp = torch.cuda.is_available()
if use_amp:
    scaler = torch.cuda.amp.GradScaler()
else:
    scaler = None  # placeholder; not used on CPU



## === cell 6
print("Starting training...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for imgs, labs in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labs = labs.to(device, non_blocking=True)

        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labs)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * imgs.size(0)
    epoch_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch}/{EPOCHS} – Train loss: {epoch_loss:.4f}")


def evaluate_auc(loader, net):
    net.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs = imgs.to(device, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = net(imgs)
                    probs = torch.sigmoid(logits).cpu().numpy()
            else:
                logits = net(imgs)
                probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(labs.numpy())
    preds = np.vstack(all_preds)
    labels = np.vstack(all_labels)
    aucs = []
    for i in range(labels.shape[1]):
        try:
            aucs.append(roc_auc_score(labels[:, i], preds[:, i]))
        except ValueError:
            aucs.append(np.nan)
    return np.nanmean(aucs)


val_auc = evaluate_auc(val_loader, model)
print(f"Final validation macro AUC: {val_auc:.4f}")

print("Training completed.")




## === cell 7
class RanzcrClipTestDataset(Dataset):
    def __init__(self, ids, root_dir):
        self.ids = ids
        self.root_dir = root_dir
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.ids)

    def _load_image(self, uid):
        img_path = os.path.join(self.root_dir, f"{uid}.jpg")
        img = io.read_image(img_path).float() / 255.0
        img = TF.resize(img, [IMG_SIZE, IMG_SIZE])
        img = (img - self.mean) / self.std
        return img

    def __getitem__(self, idx):
        uid = self.ids[idx]
        return self._load_image(uid)


test_dataset = RanzcrClipTestDataset(
    ids=sample_sub["StudyInstanceUID"].values,
    root_dir=ROOT_DIR,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_mem,
    prefetch_factor=4,
    persistent_workers=True,
)
print(f"Test loader prepared, total batches: {len(test_loader)}")




## === cell 8
def inference(net, loader):
    net.eval()
    all_probs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = net(batch)
                    probs = torch.sigmoid(logits)
            else:
                logits = net(batch)
                probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu().numpy())
    return np.vstack(all_probs)


y_prob = inference(model, test_loader)
print(f"Inference completed, shape of probabilities: {y_prob.shape}")



## === cell 9
submission_df = pd.DataFrame(
    data=np.concatenate(
        [sample_sub["StudyInstanceUID"].values.reshape(-1, 1), y_prob], axis=1
    ),
    columns=sample_sub.columns,
)
submission_path = "./submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
