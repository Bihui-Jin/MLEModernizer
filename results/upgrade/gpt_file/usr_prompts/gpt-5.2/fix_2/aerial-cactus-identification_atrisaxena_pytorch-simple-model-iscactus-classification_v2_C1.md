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

0.8371

# 6. Current score

0.9392

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.9392) has done: 'I fix the runtime errors by (1) removing notebook-only magics, (2) correcting the input paths to the actual Kaggle dataset folder layout, and (3) replacing the deprecated `transforms.Scale` with `transforms.Resize`. To ensure the submission matches the competition metric (ROC AUC), I keep your ResNet50 + CrossEntropy training logic but change inference to output `P(has_cactus=1)` via softmax instead of hard class labels (this is necessary for a valid probabilistic submission and should improve AUC). I also make the dataset return `long` labels for training and a dummy label for test, and switch `tqdm_notebook` to `tqdm` so it runs in scripts. Finally, the code always write `submission.csv` with columns `id,has_cactus`.'

# 9. Code solution

## === cell 0
from __future__ import print_function, division

import os
import time
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image

import matplotlib.pyplot as plt

plt.ion()  # interactive mode (harmless in scripts)
multiGPU = False

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)



## === cell 1
BASE_PATH = "/kaggle/input/aerial-cactus-identification"
TRAIN_IMG_PATH = os.path.join(BASE_PATH, "train")
TEST_IMG_PATH = os.path.join(BASE_PATH, "test")
LABELS_CSV_PATH = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")

assert os.path.exists(TRAIN_IMG_PATH), f"Missing train images at {TRAIN_IMG_PATH}"
assert os.path.exists(TEST_IMG_PATH), f"Missing test images at {TEST_IMG_PATH}"
assert os.path.exists(LABELS_CSV_PATH), f"Missing labels at {LABELS_CSV_PATH}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Missing sample submission at {SAMPLE_SUB_PATH}"




## === cell 2
class CactusDataset(Dataset):
    """Cactus identification dataset."""

    def __init__(self, img_dir, dataframe, transform=None, has_labels=True):
        """
        Args:
            img_dir (string): Directory with all the images.
            dataframe (pd.DataFrame): Dataframe with at least column 'id'.
            transform (callable, optional): Optional transform to be applied.
            has_labels (bool): If False, returns dummy label 0 (for test set).
        """
        self.labels_frame = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = has_labels

        if "id" not in self.labels_frame.columns:
            raise ValueError("Dataframe must contain 'id' column.")
        if self.has_labels and "has_cactus" not in self.labels_frame.columns:
            raise ValueError("Training/val dataframe must contain 'has_cactus' column.")

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.labels_frame.loc[idx, "id"])
        image = Image.open(img_name).convert("RGB")

        if self.has_labels:
            label = int(self.labels_frame.loc[idx, "has_cactus"])
        else:
            label = 0  # dummy

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)




## === cell 3
dframe = pd.read_csv(LABELS_CSV_PATH)

cut = int(len(dframe) * 0.95)
train, val = np.split(dframe, [cut], axis=0)
val = val.reset_index(drop=True)

train_ds_raw = CactusDataset(TRAIN_IMG_PATH, train, transform=None, has_labels=True)
idx = 1
plt.imshow(np.array(train_ds_raw[idx][0]))
print(int(train_ds_raw[idx][1].item()))
print("Raw image size:", train_ds_raw[idx][0].size)



## === cell 4
data_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 5
train_ds = CactusDataset(
    TRAIN_IMG_PATH, train, transform=data_transform, has_labels=True
)
val_ds = CactusDataset(TRAIN_IMG_PATH, val, transform=data_transform, has_labels=True)
datasets = {"train": train_ds, "val": val_ds}

idx = 29
print(int(train_ds[idx][1].item()))
print("Transformed image shape:", train_ds[idx][0].shape)



## === cell 6
trainloader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
valloader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
dataloaders = {"train": trainloader, "val": valloader}



## === cell 7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device




## === cell 8
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = {
        k: v.detach().cpu().clone() for k, v in model.state_dict().items()
    }
    best_acc = 0.0

    for epoch in tqdm(range(num_epochs), desc="epochs"):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        for phase in ["train", "val"]:
            since_epoch = time.time()

            if phase == "train":
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=phase, leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / len(datasets[phase])
            epoch_acc = running_corrects.double().item() / len(datasets[phase])

            time_elapsed_epoch = time.time() - since_epoch
            print(
                "{} Loss: {:.4f} Acc: {:.4f} in {:.0f}m {:.0f}s".format(
                    phase,
                    epoch_loss,
                    epoch_acc,
                    time_elapsed_epoch // 60,
                    time_elapsed_epoch % 60,
                )
            )

            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }

        print()

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    model.load_state_dict(best_model_wts)
    return model




## === cell 9
model_ft = models.resnet50(pretrained=True)
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, 120)

if torch.cuda.device_count() > 1 and multiGPU:
    print("Using", torch.cuda.device_count(), "GPUs!")
    model_ft = nn.DataParallel(model_ft)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)



## === cell 10
model_ft = train_model(
    model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=10
)



## === cell 11
submission_df = pd.read_csv(SAMPLE_SUB_PATH)

tdata_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

submission_ds = CactusDataset(
    TEST_IMG_PATH, submission_df, transform=tdata_transform, has_labels=False
)
sub_loader = DataLoader(submission_ds, batch_size=64, shuffle=False, num_workers=0)


@torch.no_grad()
def test_submission_proba(model):
    model.eval()
    probs = []
    for inputs, _ in tqdm(sub_loader, desc="inference"):
        inputs = inputs.to(device)
        outputs = model(inputs)
        p = torch.softmax(outputs, dim=1)[:, 1]  # P(class=1)
        probs.extend(p.detach().cpu().numpy().tolist())
    return probs




## === cell 12
sub = pd.read_csv(SAMPLE_SUB_PATH)
sub["has_cactus"] = test_submission_proba(model_ft)
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
