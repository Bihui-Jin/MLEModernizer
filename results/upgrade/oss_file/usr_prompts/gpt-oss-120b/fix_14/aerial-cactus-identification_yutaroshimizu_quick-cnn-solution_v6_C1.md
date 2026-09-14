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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.12

# 3. Installed packages

geopandas==0.14.4
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.648

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

_DEFAULT_DATA_ROOT = os.path.join("input", "aerial-cactus-identification")


def _find_file(filename, start_dir="."):
    """Search recursively for *filename* starting at *start_dir*."""
    for root, _, files in os.walk(start_dir):
        if filename in files:
            return os.path.join(root, filename)
    raise FileNotFoundError(f"{filename} not found under {start_dir}")


if os.path.isdir(_DEFAULT_DATA_ROOT):
    DATA_ROOT = _DEFAULT_DATA_ROOT
else:
    DATA_ROOT = "."  # fallback – will be searched for needed files

train_csv_path = os.path.join(DATA_ROOT, "train.csv")
sample_sub_path = os.path.join(DATA_ROOT, "sample_submission.csv")

if not os.path.isfile(train_csv_path):
    train_csv_path = _find_file("train.csv", ".")
if not os.path.isfile(sample_sub_path):
    sample_sub_path = _find_file("sample_submission.csv", ".")


def _contains_jpgs(dir_path):
    """Return True if *dir_path* has at least one .jpg file."""
    if not os.path.isdir(dir_path):
        return False
    return any(f.lower().endswith(".jpg") for f in os.listdir(dir_path))


def _find_deep_image_dir(root):
    """
    Walk *root* depth‑first and return the first sub‑directory that contains .jpg files.
    If *root* itself contains .jpg files, it is returned.
    """
    if _contains_jpgs(root):
        return root
    for cur_root, _, files in os.walk(root):
        if any(f.lower().endswith(".jpg") for f in files):
            return cur_root
    return root


train_dir_candidate = os.path.join(DATA_ROOT, "train")
test_dir_candidate = os.path.join(DATA_ROOT, "test")

train_dir = (
    _find_deep_image_dir(train_dir_candidate)
    if os.path.isdir(train_dir_candidate)
    else _find_deep_image_dir(DATA_ROOT)
)
test_dir = (
    _find_deep_image_dir(test_dir_candidate)
    if os.path.isdir(test_dir_candidate)
    else _find_deep_image_dir(DATA_ROOT)
)

print(f"Using train images from: {train_dir}")
print(f"Using test images from:  {test_dir}")




## === cell 1
class CustomDataset(Dataset):
    """Loads images given a directory and a dataframe with an 'id' column."""

    def __init__(self, img_path, df, transform=None):
        self.img_path = img_path
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _find_image_path(self, img_id):
        candidate = os.path.join(self.img_path, img_id)
        if os.path.isfile(candidate):
            return candidate
        for root, _, files in os.walk(self.img_path):
            if img_id in files:
                return os.path.join(root, img_id)
        raise FileNotFoundError(f"Image file {img_id} not found in {self.img_path}")

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, "id"]
        img_full_path = self._find_image_path(img_id)
        img = Image.open(img_full_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if "has_cactus" in self.df.columns:
            label = float(self.df.loc[idx, "has_cactus"])
        else:
            label = -1.0
        return img, label




## === cell 2
BATCH_SIZE = 256
NUM_EPOCHS = 5
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

train_split, val_split = train_test_split(
    train_df, test_size=0.2, stratify=train_df["has_cactus"], random_state=42
)

train_dataset = CustomDataset(train_dir, train_split, transform=transform)
val_dataset = CustomDataset(train_dir, val_split, transform=transform)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
)


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


model = SimpleCNN().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_losses = []
    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
    avg_loss = sum(epoch_losses) / len(epoch_losses)

    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy().ravel()
            all_probs.extend(probs)
            all_targets.extend(labels.numpy())
    val_auc = roc_auc_score(all_targets, all_probs)
    print(f"Epoch {epoch}/{NUM_EPOCHS} - loss: {avg_loss:.4f} - Val AUC: {val_auc:.4f}")

test_dataset = CustomDataset(test_dir, sample_sub, transform=transform)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True
)

model.eval()
test_probs = []
with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()
        test_probs.extend(probs)

submission = pd.DataFrame({"id": sample_sub["id"], "has_cactus": test_probs})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1744032827.py in <cell line: 0>()
     60     model.train()
     61     epoch_losses = []
---> 62     for imgs, labels in train_loader:
     63         imgs = imgs.to(DEVICE)
     64         labels = labels.to(DEVICE).unsqueeze(1)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_55/1739433771.py in __getitem__(self, idx)
     21     def __getitem__(self, idx):
     22         img_id = self.df.loc[idx, "id"]
---> 23         img_full_path = self._find_image_path(img_id)
     24         img = Image.open(img_full_path).convert("RGB")
     25         if self.transform:

/tmp/ipykernel_55/1739433771.py in _find_image_path(self, img_id)
     17             if img_id in files:
     18                 return os.path.join(root, img_id)
---> 19         raise FileNotFoundError(f"Image file {img_id} not found in {self.img_path}")
     20 
     21     def __getitem__(self, idx):

FileNotFoundError: Image file 9e03c8d90e4ba953d1ace7ec5117487c.jpg not found in ./aerial-cactus-identification/test
