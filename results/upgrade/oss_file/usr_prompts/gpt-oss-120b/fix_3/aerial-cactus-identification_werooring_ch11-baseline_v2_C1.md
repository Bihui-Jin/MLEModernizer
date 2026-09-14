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
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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

# 4. Data file paths

```
/
    kaggle/
        data/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        input/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> working/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> working/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

# 5. Target score

0.9843

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import os

data_path = "/kaggle/input/aerial-cactus-identification/"

labels = pd.read_csv(os.path.join(data_path, "train.csv"))
submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))




## === cell 1
from zipfile import ZipFile

with ZipFile(os.path.join(data_path, "train.zip")) as zipper:
    zipper.extractall()
with ZipFile(os.path.join(data_path, "test.zip")) as zipper:
    zipper.extractall()




## === cell 2
import torch
import random
import numpy as np

seed = 10
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False




## === cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device




## === cell 4
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(
    labels, test_size=0.1, stratify=labels["has_cactus"], random_state=10
)




## === cell 5
print("훈련 데이터 개수:", len(train_df))
print("검증 데이터 개수:", len(valid_df))




## === cell 6
import cv2
from torch.utils.data import Dataset


def resolve_image_dir(base_dir):
    """
    Return the deepest existing directory that contains the images.
    Handles cases like train/train or train/train/train after unzip.
    """
    nested = os.path.join(base_dir, base_dir)
    if os.path.isdir(nested):
        return nested
    deeper = os.path.join(nested, base_dir)
    if os.path.isdir(deeper):
        return deeper
    return base_dir


class ImageDataset(Dataset):
    def __init__(self, df, img_dir="./", transform=None):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.img_dir = resolve_image_dir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_id)
        if not os.path.isfile(img_path):
            alt_path = os.path.join(self.img_dir, self.img_dir, img_id)
            if os.path.isfile(alt_path):
                img_path = alt_path
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.df.iloc[idx, 1])
        if self.transform is not None:
            image = self.transform(image)
        return image, label




## === cell 7
from torchvision import transforms

transform = transforms.ToTensor()




## === cell 8
dataset_train = ImageDataset(df=train_df, img_dir="train", transform=transform)
dataset_valid = ImageDataset(df=valid_df, img_dir="train", transform=transform)




## === cell 9
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)




## === cell 10
from torch.utils.data import DataLoader

loader_train = DataLoader(
    dataset=dataset_train,
    batch_size=32,
    shuffle=True,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=0,
)
loader_valid = DataLoader(
    dataset=dataset_valid,
    batch_size=32,
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=0,
)




## === cell 11
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=2
        )
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.fc = nn.Linear(in_features=64 * 4 * 4, out_features=2)

    def forward(self, x):
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = self.avg_pool(x)
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc(x)
        return x




## === cell 12
model = Model().to(device)




## === cell 13
criterion = nn.CrossEntropyLoss()




## === cell 14
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)




## === cell 15
import math

print("Steps per epoch:", math.ceil(len(train_df) / 32))




## === cell 16
print("Number of batches in loader_train:", len(loader_train))




## === cell 17
epochs = 20  # extended training for better AUC
for epoch in range(epochs):
    epoch_loss = 0.0
    model.train()
    for images, labels in loader_train:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f"에폭 [{epoch+1}/{epochs}] - 손실값: {epoch_loss/len(loader_train):.4f}")




## === cell 18
from sklearn.metrics import roc_auc_score

true_list = []
preds_list = []

model.eval()
with torch.no_grad():
    for images, labels in loader_valid:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        true = labels.cpu().numpy()
        preds_list.extend(preds)
        true_list.extend(true)

print(f"검증 데이터 ROC AUC : {roc_auc_score(true_list, preds_list):.4f}")




## === cell 19
dataset_test = ImageDataset(df=submission, img_dir="test", transform=transform)
loader_test = DataLoader(
    dataset=dataset_test,
    batch_size=32,
    shuffle=False,
    worker_init_fn=seed_worker,
    generator=g,
    num_workers=0,
)




## === cell 20
model.eval()
preds = []
with torch.no_grad():
    for images, _ in loader_test:
        images = images.to(device)
        outputs = model(images)
        preds_part = torch.softmax(outputs, dim=1)[:, 1].cpu().tolist()
        preds.extend(preds_part)




## === cell 21
submission["has_cactus"] = preds
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with", len(preds), "rows.")




## === cell 22
import shutil

for folder in ["./train", "./test"]:
    if os.path.isdir(folder):
        shutil.rmtree(folder)
