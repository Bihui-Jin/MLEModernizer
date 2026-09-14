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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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

0.95

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.72723) has done: 'The changes fix the test image loading path, ensure images are normalized to float tensors, keep tensors on the same device during loss computation, run inference without gradients, and write the raw probabilities (not rounded) to the submission file so the row count matches the test set. These fixes eliminate the runtime errors and produce a valid `submission.csv`, while modestly improving the AUC score toward the target.'

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

print(os.listdir("../input"))




## === cell 1
import cv2




## === cell 2
from PIL import Image
from matplotlib import pyplot as plt




## === cell 3
import torch




## === cell 4
BASE_PATH = "../input/aerial-cactus-identification/"
if not os.path.isdir(BASE_PATH):
    BASE_PATH = "../input/"
DATA_DIR = BASE_PATH
TRAIN_DIR = os.path.join(DATA_DIR, "train/")
NAMES_DIR = os.path.join(DATA_DIR, "train.csv")




## === cell 5
TEST_DIR = os.path.join(DATA_DIR, "test/")




## === cell 6
train_names = pd.read_csv(NAMES_DIR)
train_names.shape




## === cell 7
len(os.listdir(TRAIN_DIR))




## === cell 8
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms




## === cell 9
def show_image(index):
    image = Image.open(TRAIN_DIR + str(train_names["id"][index]))
    plt.imshow(image)
    print(train_names["has_cactus"][index])


show_image(20)




## === cell 10
class HasCactusDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.img_names = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        for idx in range(len(self.img_names)):
            img_path = os.path.join(self.root_dir, self.img_names.iloc[idx, 0])
            img = cv2.imread(img_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            self.images.append(img_tensor)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.img_names.iloc[idx, 1], dtype=torch.float32)
        return image, label




## === cell 11
train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

train_full_dataset = HasCactusDataset(
    csv_file=NAMES_DIR, root_dir=TRAIN_DIR, transform=train_transform
)
val_full_dataset = HasCactusDataset(
    csv_file=NAMES_DIR, root_dir=TRAIN_DIR, transform=val_transform
)

full_dataset = train_full_dataset




## === cell 12
from sklearn.model_selection import train_test_split

indices = list(range(len(train_full_dataset)))
train_idx, val_idx = train_test_split(
    indices,
    test_size=0.10,
    stratify=train_names["has_cactus"],
    random_state=42,
)

train_dataset = torch.utils.data.Subset(train_full_dataset, train_idx)
val_dataset = torch.utils.data.Subset(val_full_dataset, val_idx)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)




## === cell 13
plt.imshow(full_dataset[19][0].permute(1, 2, 0).numpy())




## === cell 14
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(10, 18, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(18, 25, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(25, 30, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(30, 30, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(30, 40, 5)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(40 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = x.view(-1, 40 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # raw logits
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = Net().to(device)




## === cell 15
import torch.optim as optim

pos = train_names["has_cactus"].sum()
neg = len(train_names) - pos
pos_weight = torch.tensor(neg / pos, dtype=torch.float32, device=device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(
    net.parameters(), lr=1e-4
)  # lower LR for more stable convergence

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)




## === cell 16
from torchmetrics import AUROC




## === cell 17
num_epochs = 150  # extended epochs for better convergence
best_val_auc = 0.0
best_state_dict = net.state_dict()  # ensure a fallback state

for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        inputs = images.to(device, non_blocking=True)
        targets = labels.view(-1, 1).to(device, non_blocking=True)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)

    net.eval()
    val_probs = []
    val_targets = []
    with torch.no_grad():
        for images, labels in val_loader:
            inputs = images.to(device, non_blocking=True)
            logits = net(inputs)
            probs = torch.sigmoid(logits)  # convert to probabilities for AUROC
            val_probs.extend(probs.squeeze().cpu().numpy())
            val_targets.extend(labels.numpy())
    val_auc = AUROC(task="binary")(
        torch.tensor(val_probs), torch.tensor(val_targets)
    ).item()

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state_dict = net.state_dict()

    print(
        f"Epoch {epoch+1}/{num_epochs} - loss: {avg_loss:.4f} - val AUC: {val_auc:.4f} - lr: {optimizer.param_groups[0]['lr']:.6f}"
    )
    scheduler.step()  # update learning rate

net.load_state_dict(best_state_dict)
print(f"Best validation AUC: {best_val_auc:.4f}")




## === cell 18
data = next(iter(train_loader))




## === cell 19
class TestSet(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.img_paths = []
        for entry in sorted(os.listdir(root_dir)):
            entry_path = os.path.join(root_dir, entry)
            if os.path.isdir(entry_path):
                for file in sorted(os.listdir(entry_path)):
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.img_paths.append(os.path.join(entry_path, file))
            else:
                if entry.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.img_paths.append(entry_path)
        self.images = []
        for p in self.img_paths:
            img = cv2.imread(p)
            if img is None:
                raise FileNotFoundError(f"Test image not found: {p}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            self.images.append(img_tensor)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        img_name = os.path.basename(self.img_paths[idx])
        return image, img_name




## === cell 20
testSet = TestSet(root_dir=TEST_DIR, transform=None)




## === cell 21
testloader = DataLoader(
    testSet,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)  # keep order matching ids




## === cell 22
tstdata = next(iter(testloader))




## === cell 23
tstdata[0].shape




## === cell 24
plt.imshow(tstdata[0][0].permute(1, 2, 0).numpy())




## === cell 25
net.eval()
tst_outputs = []
tst_names = []
with torch.no_grad():
    for image, name in testloader:
        inputs = image.to(device, non_blocking=True)
        logits = net(inputs)
        prob = torch.sigmoid(logits).cpu().item()
        flipped_h = torch.flip(inputs, dims=[3])
        prob_h = torch.sigmoid(net(flipped_h)).cpu().item()
        flipped_v = torch.flip(inputs, dims=[2])
        prob_v = torch.sigmoid(net(flipped_v)).cpu().item()
        avg_prob = (prob + prob_h + prob_v) / 3.0
        tst_outputs.append(avg_prob)
        tst_names.append(name)




## === cell 26
len(tst_outputs)




## === cell 27
tst_outputs[:5]




## === cell 28
len(tst_names)




## === cell 29
my_submission = pd.DataFrame({"id": tst_names, "has_cactus": tst_outputs})
my_submission.to_csv("submission.csv", index=False)




## === cell 30
print("Submission file written with", len(my_submission), "rows.")
