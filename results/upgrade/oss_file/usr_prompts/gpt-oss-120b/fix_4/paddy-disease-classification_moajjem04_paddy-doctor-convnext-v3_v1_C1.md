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

3.10

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
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

0.5933179723502304

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
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets, models
from torchmetrics import Accuracy
from tqdm.notebook import tqdm
from PIL import Image




## === cell 1
def seed_everything(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 2
possible_roots = [
    Path("/kaggle/input/paddy-disease-classification"),
    Path.cwd() / "input" / "paddy-disease-classification",
    Path.cwd() / "data" / "paddy-disease-classification",
]


def locate_data_root(candidate_paths):
    """Return a Path that contains both train_images and test_images."""
    for p in candidate_paths:
        if not p.exists():
            continue
        if (p / "train_images").is_dir() and (p / "test_images").is_dir():
            return p
        for sub in p.iterdir():
            if (
                sub.is_dir()
                and (sub / "train_images").is_dir()
                and (sub / "test_images").is_dir()
            ):
                return sub
    return None


data_root = locate_data_root(possible_roots)
if data_root is None:
    raise FileNotFoundError(
        "Could not locate paddy-disease-classification data directory."
    )

train_root = data_root / "train_images"
test_root = data_root / "test_images"

train_tf = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

full_dataset = datasets.ImageFolder(root=train_root, transform=train_tf)

train_len = int(0.9 * len(full_dataset))
val_len = len(full_dataset) - train_len
train_ds, val_ds = random_split(
    full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
)
val_ds.dataset.transform = val_tf



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3120252735.py in <cell line: 0>()
     52 )
     53 
---> 54 full_dataset = datasets.ImageFolder(root=train_root, transform=train_tf)
     55 
     56 train_len = int(0.9 * len(full_dataset))

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
train_loader = DataLoader(
    train_ds, batch_size=64, shuffle=True, pin_memory=True, num_workers=0
)
val_loader = DataLoader(
    val_ds, batch_size=64, shuffle=False, pin_memory=True, num_workers=0
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2151971358.py in <cell line: 0>()
      1 train_loader = DataLoader(
----> 2     train_ds, batch_size=64, shuffle=True, pin_memory=True, num_workers=0
      3 )
      4 val_loader = DataLoader(
      5     val_ds, batch_size=64, shuffle=False, pin_memory=True, num_workers=0

NameError: name 'train_ds' is not defined

## === cell 4
model = models.convnext_tiny(pretrained=True)
model.classifier[2] = nn.Linear(in_features=768, out_features=10)



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()
accuracy_metric = Accuracy(task="multiclass", num_classes=10).to(device)




## === cell 6
def run_step(data, target, training: bool):
    if training:
        optimizer.zero_grad()
    outputs = model(data)  # raw logits
    loss = criterion(outputs, target)
    if training:
        loss.backward()
        optimizer.step()
    return outputs, loss




## === cell 7
n_epochs = 3
best_val_loss = float("inf")
best_acc = 0.0
patience = 10
no_improve = 0

for epoch in tqdm(range(n_epochs), desc="Epochs"):
    model.train()
    train_loss, train_steps = 0.0, 0
    for imgs, lbls in tqdm(train_loader, desc="Training", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        outputs, loss = run_step(imgs, lbls, training=True)
        train_loss += loss.item() * imgs.size(0)
        preds = torch.argmax(outputs, dim=1)
        accuracy_metric.update(preds, lbls)
        train_steps += 1
    train_loss /= len(train_loader.dataset)
    train_acc = accuracy_metric.compute().item()
    accuracy_metric.reset()

    model.eval()
    val_loss, val_steps = 0.0, 0
    with torch.no_grad():
        for imgs, lbls in tqdm(val_loader, desc="Validation", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            outputs, loss = run_step(imgs, lbls, training=False)
            val_loss += loss.item() * imgs.size(0)
            preds = torch.argmax(outputs, dim=1)
            accuracy_metric.update(preds, lbls)
            val_steps += 1
    val_loss /= len(val_loader.dataset)
    val_acc = accuracy_metric.compute().item()
    accuracy_metric.reset()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pt")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            break

    print(
        f"Epoch {epoch+1:02d} | "
        f"Train Loss {train_loss:.4f} Acc {train_acc:.4f} | "
        f"Val   Loss {val_loss:.4f} Acc {val_acc:.4f} | "
        f"Best Val Acc {best_acc:.4f}"
    )



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3927114804.py in <cell line: 0>()
      8     model.train()
      9     train_loss, train_steps = 0.0, 0
---> 10     for imgs, lbls in tqdm(train_loader, desc="Training", leave=False):
     11         imgs = imgs.to(device, non_blocking=True)
     12         lbls = lbls.to(device, non_blocking=True)

NameError: name 'train_loader' is not defined

## === cell 8
class_dict = {idx: class_name for class_name, idx in full_dataset.class_to_idx.items()}



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2389271195.py in <cell line: 0>()
      1 # Mapping from class index to original label string
----> 2 class_dict = {idx: class_name for class_name, idx in full_dataset.class_to_idx.items()}
      3 

NameError: name 'full_dataset' is not defined

## === cell 9
sample_df = pd.read_csv(data_root / "sample_submission.csv")
sample_df.head()



## === cell 10
if os.path.exists("best_model.pt"):
    model.load_state_dict(torch.load("best_model.pt", map_location=device))
    model.eval()
else:
    print("Warning: best_model.pt not found – using the current model state.")



## === cell 11
test_tf = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_files = sorted([f for f in os.listdir(test_root) if f.lower().endswith(".jpg")])



## === cell 12
results = []
print("Running inference on test set...")
for fname in tqdm(test_files):
    img_path = os.path.join(test_root, fname)
    img = Image.open(img_path).convert("RGB")
    img_tensor = test_tf(img).unsqueeze(0).to(device)  # add batch dim
    with torch.no_grad():
        logits = model(img_tensor)
    pred_idx = torch.argmax(logits, dim=1).item()
    pred_label = class_dict[pred_idx]
    results.append([fname, pred_label])

result_df = pd.DataFrame(results, columns=["image_id", "label"])
assert len(result_df) == len(test_files), "Row count mismatch!"



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2827742913.py in <cell line: 0>()
      8         logits = model(img_tensor)
      9     pred_idx = torch.argmax(logits, dim=1).item()
---> 10     pred_label = class_dict[pred_idx]
     11     results.append([fname, pred_label])
     12 

NameError: name 'class_dict' is not defined

## === cell 13
submission_path = "submission.csv"
result_df.to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path} ({len(result_df)} rows).")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/936275018.py in <cell line: 0>()
      1 submission_path = "submission.csv"
----> 2 result_df.to_csv(submission_path, index=False)
      3 print(f"Submission file saved to {submission_path} ({len(result_df)} rows).")

NameError: name 'result_df' is not defined
