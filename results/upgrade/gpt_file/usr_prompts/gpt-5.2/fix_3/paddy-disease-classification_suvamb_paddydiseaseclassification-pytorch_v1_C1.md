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
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.13

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Target score

0.8790322580645161

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
import timm

BASE_CANDIDATES = [
    Path("/kaggle/input/paddy-disease-classification"),
    Path("/kaggle/input/paddy-disease-classification/paddy-disease-classification"),
    Path("/kaggle/data/paddy-disease-classification"),
    Path("/kaggle/data/paddy-disease-classification/paddy-disease-classification"),
]


def find_comp_root(candidates):
    for base in candidates:
        if not base.exists():
            continue
        if (
            (base / "train_images").exists()
            and (base / "test_images").exists()
            and (base / "sample_submission.csv").exists()
        ):
            return base
        nested = base / "paddy-disease-classification"
        if (
            (nested / "train_images").exists()
            and (nested / "test_images").exists()
            and (nested / "sample_submission.csv").exists()
        ):
            return nested
    fallback = Path("/kaggle/input/paddy-disease-classification")
    return fallback


path = find_comp_root(BASE_CANDIDATES)

trn_path = path / "train_images"
tst_path = path / "test_images"
sample_path = path / "sample_submission.csv"

if not trn_path.exists():
    raise FileNotFoundError(f"train_images directory not found at: {trn_path}")
if not tst_path.exists():
    raise FileNotFoundError(f"test_images directory not found at: {tst_path}")
if not sample_path.exists():
    raise FileNotFoundError(f"sample_submission.csv not found at: {sample_path}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

NUM_WORKERS = 2
print("Using data root:", path)
print("Train dir:", trn_path)
print("Test dir :", tst_path)



## === cell 1
train_tfms = transforms.Compose(
    [
        transforms.Resize((480, 480)),
        transforms.RandomResizedCrop(128, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

valid_tfms = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 2
full_ds = datasets.ImageFolder(trn_path, transform=train_tfms)

n_val = int(0.2 * len(full_ds))
n_train = len(full_ds) - n_val
train_ds, valid_ds = random_split(
    full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
)

valid_ds.dataset.transform = valid_tfms

train_dl = DataLoader(
    train_ds, batch_size=64, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
)
valid_dl = DataLoader(
    valid_ds, batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
)

print(
    f"Train images: {len(train_ds)}, Valid images: {len(valid_ds)}, Classes: {len(full_ds.classes)}"
)
print("Classes:", full_ds.classes)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3063940856.py in <cell line: 0>()
----> 1 full_ds = datasets.ImageFolder(trn_path, transform=train_tfms)
      2 
      3 n_val = int(0.2 * len(full_ds))
      4 n_train = len(full_ds) - n_val
      5 train_ds, valid_ds = random_split(

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in __init__(self, root, transform, target_transform, loader, is_valid_file, allow_empty)
    326         allow_empty: bool = False,
    327     ):
--> 328         super().__init__(
    329             root,
    330             loader,

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in __init__(self, root, loader, extensions, transform, target_transform, is_valid_file, allow_empty)
    148         super().__init__(root, transform=transform, target_transform=target_transform)
    149         classes, class_to_idx = self.find_classes(self.root)
--> 150         samples = self.make_dataset(
    151             self.root,
    152             class_to_idx=class_to_idx,

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in make_dataset(directory, class_to_idx, extensions, is_valid_file, allow_empty)
    201             # is potentially overridden and thus could have a different logic.
    202             raise ValueError("The class_to_idx parameter cannot be None.")
--> 203         return make_dataset(
    204             directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file, allow_empty=allow_empty
    205         )

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in make_dataset(directory, class_to_idx, extensions, is_valid_file, allow_empty)
    102         if extensions is not None:
    103             msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
--> 104         raise FileNotFoundError(msg)
    105 
    106     return instances

FileNotFoundError: Found no valid file for the classes train_images. Supported extensions are: .jpg, .jpeg, .png, .ppm, .bmp, .pgm, .tif, .tiff, .webp

## === cell 3
model = timm.create_model(
    "resnet26d", pretrained=True, num_classes=len(full_ds.classes)
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train_one_epoch(model, dl, optimizer, criterion):
    model.train()
    total_loss, correct = 0.0, 0
    for xb, yb in dl:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += preds.argmax(dim=1).eq(yb).sum().item()
    return total_loss / len(dl.dataset), correct / len(dl.dataset)


def validate(model, dl, criterion):
    model.eval()
    total_loss, correct = 0.0, 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * xb.size(0)
            correct += preds.argmax(dim=1).eq(yb).sum().item()
    return total_loss / len(dl.dataset), correct / len(dl.dataset)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/77338923.py in <cell line: 0>()
      1 model = timm.create_model(
----> 2     "resnet26d", pretrained=True, num_classes=len(full_ds.classes)
      3 )
      4 model = model.to(device)
      5 

NameError: name 'full_ds' is not defined

## === cell 4
epochs = 3
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, criterion)
    val_loss, val_acc = validate(model, valid_dl, criterion)
    print(
        f"Epoch {epoch+1}/{epochs}: "
        f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
        f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
    )



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1668851415.py in <cell line: 0>()
      1 epochs = 3
      2 for epoch in range(epochs):
----> 3     train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, criterion)
      4     val_loss, val_acc = validate(model, valid_dl, criterion)
      5     print(

NameError: name 'train_one_epoch' is not defined

## === cell 5
test_tfms = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class TestImagesDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = list(files)
        self.transform = transform
        self.loader = datasets.folder.default_loader

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fp = self.files[idx]
        img = self.loader(fp)
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


test_files = sorted(glob(str(tst_path / "*.jpg")))
if len(test_files) == 0:
    raise FileNotFoundError(f"No .jpg files found in: {tst_path}")

test_ds = TestImagesDataset(test_files, transform=test_tfms)
test_dl = DataLoader(
    test_ds, batch_size=64, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
)

print(f"Test images: {len(test_ds)}")



## === cell 6
model.eval()
all_preds = []
with torch.no_grad():
    for xb, _ in test_dl:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu()
        all_preds.append(preds)

idxs = torch.cat(all_preds).numpy()

class_names = full_ds.classes
mapping = dict(enumerate(class_names))
pred_labels = pd.Series(idxs).map(mapping)

ss = pd.read_csv(sample_path)
pred_df = pd.DataFrame(
    {"image_id": [Path(f).name for f in test_files], "label": pred_labels.values}
)

merged = ss[["image_id"]].merge(pred_df, on="image_id", how="left")
if merged["label"].isna().any():
    missing = merged.loc[merged["label"].isna(), "image_id"].head(5).tolist()
    raise RuntimeError(f"Missing predictions for some test images, e.g.: {missing}")

submission_path = Path("submission.csv")
merged.to_csv(submission_path, index=False)

print("Wrote:", submission_path.resolve())
print(merged.head())
print("Submission shape:", merged.shape)
print("Label counts:\n", merged["label"].value_counts().head(10))

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2206501184.py in <cell line: 0>()
----> 1 model.eval()
      2 all_preds = []
      3 with torch.no_grad():
      4     for xb, _ in test_dl:
      5         xb = xb.to(device, non_blocking=True)

NameError: name 'model' is not defined
