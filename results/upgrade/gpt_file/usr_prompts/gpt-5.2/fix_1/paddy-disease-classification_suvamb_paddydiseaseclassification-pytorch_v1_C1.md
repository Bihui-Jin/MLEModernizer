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

## === cell 1
!pip install torch_lr_finder

## === cell 3
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import timm
from pathlib import Path
from torch_lr_finder import LRFinder

path = Path("/kaggle/input/paddy-disease-classification")
trn_path = path / "train_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

## === cell 5
train_tfms = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.RandomResizedCrop(128, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

valid_tfms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

## === cell 7
full_ds = datasets.ImageFolder(trn_path, transform=train_tfms)
n_val = int(0.2 * len(full_ds))
n_train = len(full_ds) - n_val
train_ds, valid_ds = random_split(full_ds, [n_train, n_val])
valid_ds.dataset.transform = valid_tfms

train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
valid_dl = DataLoader(valid_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_10/4066410065.py in <cell line: 0>()
----> 1 full_ds = datasets.ImageFolder(trn_path, transform=train_tfms)
      2 n_val = int(0.2 * len(full_ds))
      3 n_train = len(full_ds) - n_val
      4 train_ds, valid_ds = random_split(full_ds, [n_train, n_val])
      5 valid_ds.dataset.transform = valid_tfms

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

## === cell 9
model = timm.create_model("resnet26d", pretrained=True, num_classes=len(full_ds.classes))
model = model.to(device) # fp16 training

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/96235513.py in <cell line: 0>()
----> 1 model = timm.create_model("resnet26d", pretrained=True, num_classes=len(full_ds.classes))
      2 model = model.to(device) # fp16 training

NameError: name 'full_ds' is not defined

## === cell 11
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch(model, dl, optimizer, criterion):
    model.train()
    total_loss, correct = 0, 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        correct += preds.argmax(dim=1).eq(yb).sum().item()
    return total_loss / len(dl.dataset), correct / len(dl.dataset)

def validate(model, dl, criterion):
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * xb.size(0)
            correct += preds.argmax(dim=1).eq(yb).sum().item()
    return total_loss / len(dl.dataset), correct / len(dl.dataset)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/4265025388.py in <cell line: 0>()
      1 criterion = nn.CrossEntropyLoss()
----> 2 optimizer = optim.Adam(model.parameters(), lr=1e-3)
      3 
      4 def train_one_epoch(model, dl, optimizer, criterion):
      5     model.train()

NameError: name 'model' is not defined

## === cell 13
def find_lr(model, train_dl, criterion, optimizer):
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(train_dl, end_lr=1, num_iter=100)
    lr_finder.plot()  # This will plot loss vs lr
    lr_finder.reset()

## === cell 15
print("Running learning rate finder...")
find_lr(model, train_dl, criterion, optimizer)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3125381792.py in <cell line: 0>()
      1 print("Running learning rate finder...")
----> 2 find_lr(model, train_dl, criterion, optimizer)
      3 
      4 

NameError: name 'model' is not defined

## === cell 16
epochs = 3
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, criterion)
    val_loss, val_acc = validate(model, valid_dl, criterion)
    print(f"Epoch {epoch+1}/{epochs}: "
          f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1143562695.py in <cell line: 0>()
      1 epochs = 3
      2 for epoch in range(epochs):
----> 3     train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, criterion)
      4     val_loss, val_acc = validate(model, valid_dl, criterion)
      5     print(f"Epoch {epoch+1}/{epochs}: "

NameError: name 'train_one_epoch' is not defined

## === cell 18
tst_path = path/"test_images"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_tfms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


## === cell 19
test_ds = datasets.ImageFolder(
    root=path, 
    transform=test_tfms,
    loader=datasets.folder.default_loader
)


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_10/2889774324.py in <cell line: 0>()
----> 1 test_ds = datasets.ImageFolder(
      2     root=path,
      3     transform=test_tfms,
      4     loader=datasets.folder.default_loader
      5 )

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

FileNotFoundError: Found no valid file for the classes paddy-disease-classification. Supported extensions are: .jpg, .jpeg, .png, .ppm, .bmp, .pgm, .tif, .tiff, .webp

## === cell 20
from glob import glob
test_files = sorted(glob(str(tst_path/"*.jpg")))
test_ds.samples = [(f, 0) for f in test_files]
test_ds.targets = [0] * len(test_files)

test_dl = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3143788556.py in <cell line: 0>()
      1 from glob import glob
      2 test_files = sorted(glob(str(tst_path/"*.jpg")))
----> 3 test_ds.samples = [(f, 0) for f in test_files]
      4 test_ds.targets = [0] * len(test_files)
      5 

NameError: name 'test_ds' is not defined

## === cell 21
model.eval()
all_preds = []
with torch.no_grad():
    for xb, _ in test_dl:
        xb = xb.to(device)
        preds = model(xb)
        probs = torch.softmax(preds, dim=1)
        all_preds.append(probs.argmax(dim=1).cpu())

idxs = torch.cat(all_preds)   # predicted class indices


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2588906587.py in <cell line: 0>()
----> 1 model.eval()
      2 all_preds = []
      3 with torch.no_grad():
      4     for xb, _ in test_dl:
      5         xb = xb.to(device)

NameError: name 'model' is not defined

## === cell 22
import pandas as pd
class_names = train_dl.dataset.dataset.classes  # from ImageFolder train dataset
mapping = dict(enumerate(class_names))

results = pd.Series(idxs.numpy(), name="label").map(mapping)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1198657610.py in <cell line: 0>()
      1 # Map indices back
      2 import pandas as pd
----> 3 class_names = train_dl.dataset.dataset.classes  # from ImageFolder train dataset
      4 mapping = dict(enumerate(class_names))
      5 

NameError: name 'train_dl' is not defined

## === cell 23
ss = pd.read_csv(path/"sample_submission.csv")
ss['label'] = results.values
ss.to_csv("submission.csv", index=False)

!head submission.csv


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3742375042.py in <cell line: 0>()
      1 ss = pd.read_csv(path/"sample_submission.csv")
----> 2 ss['label'] = results.values
      3 ss.to_csv("submission.csv", index=False)
      4 
      5 get_ipython().system('head submission.csv')

NameError: name 'results' is not defined

## === cell 27
""" 
train_tfms = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.RandomResizedCrop(128, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),  # Added random rotation
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Added color jitter
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
"""

## === cell 30
""" 
# Add this import at the top
from torch.optim.lr_scheduler import StepLR

# In the main training loop, after initializing the optimizer:

scheduler = StepLR(optimizer, step_size=2, gamma=0.1)  # Initialize the scheduler

# Modify the training loop to step the scheduler
for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch(model, train_dl, optimizer, criterion)
    val_loss, val_acc = validate(model, valid_dl, criterion)
    print(f"Epoch {epoch+1}/{epochs}: "
          f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
          f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
    scheduler.step()  # Step the scheduler after each epoch
""" 

## === cell 33
""" 
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)

# At the end of the training loop, call the save_model function:
save_model(model, 'trained_model.pth')  # Save the model state
"""

## === cell 36
"""
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=len(full_ds.classes))
"""

## === cell 39
""" 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Modify the validate function to return predictions and true labels:

def validate(model, dl, criterion):
    model.eval()
    total_loss, correct = 0, 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * xb.size(0)
            correct += preds.argmax(dim=1).eq(yb).sum().item()
            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
    return total_loss / len(dl.dataset), correct / len(dl.dataset), all_labels, all_preds

# In the training loop, after validation:
val_loss, val_acc, val_labels, val_preds = validate(model, valid_dl, criterion)
plot_confusion_matrix(val_labels, val_preds, classes=full_ds.classes) 
"""
