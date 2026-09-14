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
from PIL import Image
from tqdm import tqdm
import copy
import pandas as pd
import os
import torch
from torch import optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torchvision.models import efficientnet_b4

DATA_ROOT = "../input/plant-pathology-2021-fgvc8"
TRAIN_IMG_ROOT = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_ROOT = os.path.join(DATA_ROOT, "test_images")




## === cell 1
transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
transform_valid = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)




## === cell 2
df = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
cnt = int(len(df) * 0.9)
train_df = df.iloc[:cnt]
valid_df = df.iloc[cnt:]




## === cell 3
unique_labels = sorted({lbl.split(" ")[0] for lbl in df["labels"]})
label2idx = {lbl: idx for idx, lbl in enumerate(unique_labels)}
idx2label = {idx: lbl for lbl, idx in label2idx.items()}




## === cell 4
class TorchvisionDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, dataframe, label2idx, transforms=None):
        self.data_root = data_root
        self.df = dataframe
        self.label2idx = label2idx
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row["image"]
        label_name = row["labels"].split(" ")[0].strip()
        img_path = os.path.join(self.data_root, image_name)
        img = Image.open(img_path).convert("RGB")
        if self.transforms:
            img = self.transforms(img)
        label = self.label2idx[label_name]
        return img, label


train_dataset = TorchvisionDataset(
    TRAIN_IMG_ROOT, train_df, label2idx, transforms=transform_train
)
valid_dataset = TorchvisionDataset(
    TRAIN_IMG_ROOT, valid_df, label2idx, transforms=transform_valid
)




## === cell 5
train_dataloaders = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=0
)
valid_dataloaders = torch.utils.data.DataLoader(
    valid_dataset, batch_size=16, shuffle=False, num_workers=0
)




## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 7
model_ft = efficientnet_b4(weights="DEFAULT")
num_ftrs = model_ft.classifier[1].in_features
model_ft.classifier[1] = torch.nn.Linear(num_ftrs, len(label2idx))
model_ft = model_ft.to(device)




## === cell 8
criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
exp_lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=5, eta_min=0)




## === cell 9
def train_model(model, criterion, optimizer, scheduler, num_epochs=3):
    os.makedirs("outputs", exist_ok=True)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_corrects = 0
        train_cnt = 0
        prog_bar = tqdm(train_dataloaders, desc=f"Epoch {epoch+1}/{num_epochs} [train]")
        for inputs, labels in prog_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels).item()
            train_cnt += inputs.size(0)
            prog_bar.set_postfix(
                loss=running_loss / train_cnt, acc=train_corrects / train_cnt
            )

        scheduler.step()

        model.eval()
        val_corrects = 0
        val_cnt = 0
        val_bar = tqdm(valid_dataloaders, desc=f"Epoch {epoch+1}/{num_epochs} [valid]")
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels).item()
                val_cnt += inputs.size(0)
        epoch_acc = val_corrects / val_cnt
        print(f"Validation accuracy: {epoch_acc:.4f}")
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, f"outputs/best_model.pth")
            print(f"Best model saved at epoch {epoch+1}")
    model.load_state_dict(best_model_wts)
    return model




## === cell 10
model_ft = train_model(
    model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=3
)




## === cell 11
from glob import glob

test_paths = glob(os.path.join(TEST_IMG_ROOT, "*.jpg"))
submission_rows = []
model_ft.eval()
with torch.no_grad():
    for img_path in tqdm(test_paths, desc="Predict"):
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform_valid(img).unsqueeze(0).to(device)
        preds = model_ft(img_tensor)
        _, top_one = torch.max(preds, 1)
        label_str = idx2label[int(top_one)]
        submission_rows.append([os.path.basename(img_path), label_str])

submission = pd.DataFrame(submission_rows, columns=["image", "labels"])
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission saved to /kaggle/working/submission.csv")
