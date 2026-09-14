# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os, copy, random, csv
from PIL import Image
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
from torchvision.models import efficientnet_b0
from torch.utils.data import Dataset, DataLoader
from glob import glob  # added for test file discovery



## === cell 1
transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(480),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
transform_valid = transforms.Compose(
    [
        transforms.Resize(512),
        transforms.CenterCrop(480),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)



## === cell 2
base_path = "../input/plant-pathology-2021-fgvc8"

csv_path = os.path.join(base_path, "train.csv")
with open(csv_path, "r") as f:
    lines = f.readlines()[1:]  # skip header
random.shuffle(lines)
split_idx = int(len(lines) * 0.9)
train_csv = lines[:split_idx]
valid_csv = lines[split_idx:]




## === cell 3
class PlantPathologyDataset(Dataset):
    def __init__(self, img_root, csv_lines, transform=None):
        self.img_root = img_root
        self.records = []
        self.labels_set = set()
        for line in csv_lines:
            img_name, label_str = line.strip().split(",")
            label = label_str.split(" ")[0]
            self.records.append((img_name, label))
            self.labels_set.add(label)
        self.label2idx = {lbl: idx for idx, lbl in enumerate(sorted(self.labels_set))}
        self.idx2label = {idx: lbl for lbl, idx in self.label2idx.items()}
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        img_name, label = self.records[idx]
        img_path = os.path.join(self.img_root, img_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label_idx = self.label2idx[label]
        return img, label_idx




## === cell 4
train_dataset = PlantPathologyDataset(
    img_root=os.path.join(base_path, "train_images"),
    csv_lines=train_csv,
    transform=transform_train,
)
valid_dataset = PlantPathologyDataset(
    img_root=os.path.join(base_path, "train_images"),
    csv_lines=valid_csv,
    transform=transform_valid,
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=1)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=1)



## === cell 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)



## === cell 6
num_classes = len(train_dataset.label2idx)
model_ft = efficientnet_b0(pretrained=True)
in_features = model_ft.classifier[1].in_features
model_ft.classifier[1] = nn.Linear(in_features, num_classes)
model_ft = model_ft.to(device)



## === cell 7
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=5)




## === cell 8
def train_model(model, criterion, optimizer, scheduler, num_epochs=3):
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        corrects = 0
        total = 0
        prog_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [train]")
        for inputs, labels in prog_bar:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            corrects += torch.sum(preds == labels).item()
            total += inputs.size(0)
            prog_bar.set_postfix(loss=running_loss / total, acc=corrects / total)

        scheduler.step()

        model.eval()
        val_corrects = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in tqdm(
                valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [val]"
            ):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                val_total += inputs.size(0)

        val_acc = val_corrects / val_total
        print(f"Validation Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            os.makedirs("outputs", exist_ok=True)
            torch.save(model.state_dict(), f"outputs/best_epoch_{epoch}.pth")
    return best_weights




## === cell 9
best_state = train_model(model_ft, criterion, optimizer_ft, scheduler, num_epochs=3)
model_ft.load_state_dict(best_state)
model_ft.eval()



## === cell 10
idx2label = train_dataset.idx2label



## === cell 11
test_paths = glob(os.path.join(base_path, "test_images", "*"))
submission_rows = []

for img_path in tqdm(test_paths, desc="Predict"):
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform_valid(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model_ft(img_tensor)
        _, pred_idx = torch.max(logits, 1)
    pred_label = idx2label[pred_idx.item()]
    img_name = os.path.basename(img_path)
    submission_rows.append([img_name, pred_label])

submission_df = pd.DataFrame(submission_rows, columns=["image", "labels"])
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Submission saved to:", submission_path)
