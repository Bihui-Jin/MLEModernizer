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

0.995

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms

torch.backends.cudnn.benchmark = True

possible_base_dirs = [
    Path("./input/aerial-cactus-identification"),
    Path("./aerial-cactus-identification"),
    Path("../input/aerial-cactus-identification"),
    Path("../aerial-cactus-identification"),
]
BASE_DIR = next((p for p in possible_base_dirs if p.is_dir()), None)
if BASE_DIR is None:
    raise FileNotFoundError(
        "Could not locate the 'aerial-cactus-identification' data folder."
    )

DATA_DIR = BASE_DIR
TRAIN_ZIP_DIR = DATA_DIR / "train.zip"
TEST_ZIP_DIR = DATA_DIR / "test.zip"
SAMPLE_SUBMIS = DATA_DIR / "sample_submission.csv"
ANNOTATIONS_DIR = DATA_DIR / "train.csv"

TRAIN_DIR_DEFAULT = Path("./train")
TEST_DIR_DEFAULT = Path("./test")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_LABELS = 2
N_EPOCHS = 60
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MOMENTUM = 0.9
LABELS_MAP = {0: "No Cactus", 1: "Cactus"}


def init_weights(layer):
    if isinstance(layer, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            layer.bias.data.fill_(0.01)


def display_data(data, n=10, classes=None):
    fig, ax = plt.subplots(1, n, figsize=(15, 3))
    indices = np.random.randint(0, len(data), size=n)
    for i, j in enumerate(indices):
        ax[i].imshow(np.transpose(data[j][0], (1, 2, 0)))
        ax[i].axis("off")
        if classes:
            ax[i].set_title(classes[data[j][1]])


def train_epoch(
    model, dataloader, lr=LEARNING_RATE, optimizer=None, loss_fn=nn.CrossEntropyLoss()
):
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    total_loss, accuracy, count = 0.0, 0, 0
    for X, y in dataloader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(X)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        predicted = torch.max(out, 1)[1]
        accuracy += (predicted == y).sum().item()
        count += y.size(0)
    return total_loss / count, accuracy / count


def validate(model, dataloader, loss_fn=nn.CrossEntropyLoss()):
    model.eval()
    total_loss, accuracy, count = 0.0, 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            out = model(X)
            loss = loss_fn(out, y)
            total_loss += loss.item() * y.size(0)
            predicted = torch.max(out, 1)[1]
            accuracy += (predicted == y).sum().item()
            count += y.size(0)
    return total_loss / count, accuracy / count


def train(
    model,
    train_loader,
    valid_loader=None,
    optimizer=None,
    lr=LEARNING_RATE,
    epochs=N_EPOCHS,
    loss_fn=nn.CrossEntropyLoss(),
    scheduler=None,
):
    optimizer = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "train_accuracy": []}
    if valid_loader is not None:
        history["validation_loss"] = []
        history["validation_accuracy"] = []
    for epoch in range(epochs):
        tl, ta = train_epoch(
            model, train_loader, lr=lr, optimizer=optimizer, loss_fn=loss_fn
        )
        history["train_loss"].append(tl)
        history["train_accuracy"].append(ta)
        if valid_loader is not None:
            vl, va = validate(model, valid_loader, loss_fn=loss_fn)
            print(
                f"Epoch {epoch:2}, Train Acc = {ta:.3f}, Val Acc = {va:.3f}, "
                f"Train Loss = {tl:.3f}, Val Loss = {vl:.3f}"
            )
            history["validation_loss"].append(vl)
            history["validation_accuracy"].append(va)
        else:
            print(f"Epoch {epoch:2}, Train Acc = {ta:.3f}, Train Loss = {tl:.3f}")
        if scheduler is not None:
            scheduler.step()
    return history


def plot_history(history, validation=False):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")
    plt.plot(history["train_accuracy"], label="Training")
    if validation:
        plt.plot(history["validation_accuracy"], label="Validation")
    plt.legend()
    plt.subplot(122)
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.plot(history["train_loss"], label="Training")
    if validation:
        plt.plot(history["validation_loss"], label="Validation")
    plt.legend()


def submission(dataset, model):
    model.eval()
    results = []
    with torch.no_grad():
        for datapoint in dataset:
            img_tensor = datapoint[0][None, ...].to(DEVICE)
            out = model(img_tensor)
            prob = float(torch.exp(out)[0][1])  # probability of class 1
            results.append([datapoint[1], prob])
    df = pd.DataFrame(results, columns=["id", "has_cactus"])
    df = df.set_index("id")
    df = df.sort_index()
    Path("submission.csv").write_text(df.to_csv())
    return df




## === cell 1
if not TRAIN_DIR_DEFAULT.is_dir():
    if TRAIN_ZIP_DIR.is_file():
        with zipfile.ZipFile(TRAIN_ZIP_DIR, "r") as zip_ref:
            zip_ref.extractall(".")
if not TEST_DIR_DEFAULT.is_dir():
    if TEST_ZIP_DIR.is_file():
        with zipfile.ZipFile(TEST_ZIP_DIR, "r") as zip_ref:
            zip_ref.extractall(".")

possible_train_paths = [
    TRAIN_DIR_DEFAULT,
    Path(".") / "aerial-cactus-identification" / "train",
]
possible_test_paths = [
    TEST_DIR_DEFAULT,
    Path(".") / "aerial-cactus-identification" / "test",
]

TRAIN_DIR = next((p for p in possible_train_paths if p.is_dir()), None)
TEST_DIR = next((p for p in possible_test_paths if p.is_dir()), None)

if TRAIN_DIR is None or TEST_DIR is None:
    raise FileNotFoundError(
        "Could not locate train or test directories after extraction."
    )




## === cell 2
class ACIDataset(Dataset):
    """
    Loads all images into memory once (as tensors) to avoid repeated disk I/O.
    This does not change any model logic; transforms are applied during loading,
    which is deterministic.
    """

    def __init__(
        self, img_dir, annotations_file=None, transform=None, target_transform=None
    ):
        self.img_dir = Path(img_dir)
        self.is_labeled = annotations_file is not None
        self.transform = transform
        self.target_transform = target_transform

        if self.is_labeled:
            df = pd.read_csv(annotations_file)
            self.ids = df["id"].tolist()
            self.labels = df["has_cactus"].astype(int).tolist()
        else:
            self.ids = [p.name for p in sorted(self.img_dir.iterdir())]
            self.labels = [None] * len(self.ids)  # placeholder

        self.tensors = []
        for img_id in self.ids:
            img_path = self.img_dir / img_id
            try:
                image = io.imread(img_path)  # shape (H, W, C)
            except Exception:
                image = np.zeros((32, 32, 3), dtype=np.uint8)
            if self.transform:
                image = self.transform(image)  # now a torch.FloatTensor
            self.tensors.append(image)

        if self.is_labeled:
            self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_tensor = self.tensors[idx]
        if self.is_labeled:
            label = self.labels[idx]
            if self.target_transform:
                label = self.target_transform(label)
            return img_tensor, label
        else:
            return img_tensor, self.ids[idx]


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

data = ACIDataset(
    img_dir=TRAIN_DIR, annotations_file=ANNOTATIONS_DIR, transform=transform
)

test_data = ACIDataset(img_dir=TEST_DIR, transform=transform)

train_len = len(data) * 8 // 10
val_len = len(data) - train_len
train_data, val_data = random_split(data, [train_len, val_len])

NUM_WORKERS = min(4, os.cpu_count() or 1)

train_dl = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
valid_dl = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
all_dl = DataLoader(
    data,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)


def test_collate(batch):
    imgs = torch.stack([item[0] for item in batch])
    ids = [item[1] for item in batch]
    return imgs, ids


test_dl = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    collate_fn=test_collate,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)




## === cell 3
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3)
        self.fc = nn.Linear(in_features=20 * 6 * 6, out_features=2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 20 * 6 * 6)
        x = F.log_softmax(self.fc(x), dim=1)
        return x




## === cell 4
model_ = Net().to(DEVICE)
model_.apply(init_weights)



## === cell 5
optimizer = torch.optim.Adam(model_.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
history = train(
    model_,
    train_dl,
    valid_loader=valid_dl,
    optimizer=optimizer,
    lr=LEARNING_RATE,
    epochs=N_EPOCHS,
    loss_fn=nn.CrossEntropyLoss(),
    scheduler=scheduler,
)



## === cell 6
plot_history(history, validation=True)




## === cell 7
def create_submission(dataloader, model, device=DEVICE):
    model.eval()
    rows = []
    with torch.no_grad():
        for imgs, ids in dataloader:
            imgs = imgs.to(device)
            log_probs = model(imgs)
            probs = torch.exp(log_probs)[:, 1]  # probability of class 1
            probs_np = probs.cpu().numpy()
            for img_id, prob in zip(ids, probs_np):
                rows.append([img_id, float(prob)])
    df = pd.DataFrame(rows, columns=["id", "has_cactus"])
    df = df.set_index("id")
    df = df.sort_index()
    Path("submission.csv").write_text(df.to_csv())
    return df


create_submission(test_dl, model_)
