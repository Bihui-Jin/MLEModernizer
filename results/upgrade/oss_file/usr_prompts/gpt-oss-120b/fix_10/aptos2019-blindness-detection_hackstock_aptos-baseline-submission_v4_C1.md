# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
import pandas as pd
import random
from PIL import Image
import os

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

torch.backends.cudnn.benchmark = True  # keep for faster GPU kernels




## === cell 1
checkpoint_path = "../input/checkpoint4/checkpoint_epoch_4.pt"
try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print("Checkpoint loaded.")
except FileNotFoundError:
    checkpoint = None
    print("Checkpoint not found – will initialise a new model.")




## === cell 2
num_classes = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

if checkpoint is not None and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model weights loaded from checkpoint.")
else:
    print("Using randomly initialise​d model (pre‑trained backbone).")

model = model.to(device)




## === cell 3
class TrainDataset(torch.utils.data.Dataset):
    """
    Loads images lazily (on‑the‑fly) to keep RAM usage low.
    The transformation pipeline remains exactly the same as before.
    """

    def __init__(self, csv_path, img_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = self.df["diagnosis"].astype(int).tolist()
        self.id_codes = self.df["id_code"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.id_codes[idx]}.png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]




## === cell 4
train_csv = "../input/aptos2019-blindness-detection/train.csv"
train_img_dir = "../input/aptos2019-blindness-detection/train_images"

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

from sklearn.model_selection import train_test_split

full_df = pd.read_csv(train_csv)
train_idx, val_idx = train_test_split(
    full_df.index, test_size=0.1, stratify=full_df["diagnosis"], random_state=42
)

train_df = full_df.loc[train_idx].reset_index(drop=True)
val_df = full_df.loc[val_idx].reset_index(drop=True)

train_df.to_csv("tmp_train.csv", index=False)
val_df.to_csv("tmp_val.csv", index=False)

train_dataset = TrainDataset("tmp_train.csv", train_img_dir, transform=train_transform)
val_dataset = TrainDataset("tmp_val.csv", train_img_dir, transform=val_transform)

num_workers = 0

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, num_workers=num_workers, pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=False, num_workers=num_workers, pin_memory=True
)




## === cell 5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(
                device, non_blocking=True
            )
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0


epochs = 8
for epoch in range(1, epochs + 1):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    val_acc = evaluate(val_loader)
    print(f"Epoch {epoch}/{epochs} – Loss: {epoch_loss:.4f} – Val Acc: {val_acc:.4f}")

torch.save({"model_state_dict": model.state_dict()}, "trained_checkpoint.pt")
print("Training complete and checkpoint saved.")




## === cell 6
class TestDataset(torch.utils.data.Dataset):
    """
    Pre‑loads all test images into memory with the given transform.
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = sorted(
            [
                f
                for f in os.listdir(self.root_dir)
                if os.path.isfile(os.path.join(self.root_dir, f))
                and f.lower().endswith(".png")
            ]
        )
        self.images = []
        for fname in self.filenames:
            path = os.path.join(self.root_dir, fname)
            image = Image.open(path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            self.images.append(image)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        return self.images[idx], self.filenames[idx]




## === cell 7
test_data_dir = "../input/aptos2019-blindness-detection/test_images"
test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_dataset = TestDataset(test_data_dir, test_transform)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=20, shuffle=False, num_workers=0, pin_memory=True
)




## === cell 8
model.eval()
id_codes = []
diags = []

with torch.no_grad():
    for imgs, files in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        outputs = model(imgs)  # raw logits
        probs = F.softmax(outputs, dim=1)  # convert to probabilities
        class_range = torch.arange(num_classes, device=device, dtype=torch.float)
        expected = torch.sum(probs * class_range, dim=1)  # expected rating
        preds = torch.clamp(torch.round(expected), 0, num_classes - 1).long()
        for fname, pred in zip(files, preds):
            id_codes.append(fname.replace(".png", ""))
            diags.append(int(pred.item()))

df = pd.DataFrame({"id_code": id_codes, "diagnosis": diags})

test_ids = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")["id_code"]
df = df.set_index("id_code").reindex(test_ids).reset_index()
df["diagnosis"] = df["diagnosis"].fillna(0).astype(int)

submission_path = "./submission.csv"
df.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")




## === cell 9
print(df.head())

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same id_codes as answers
