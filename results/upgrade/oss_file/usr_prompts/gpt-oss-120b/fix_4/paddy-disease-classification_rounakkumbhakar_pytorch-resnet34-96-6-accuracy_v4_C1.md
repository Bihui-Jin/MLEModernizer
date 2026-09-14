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

3.11

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

0.93894

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Resize, CenterCrop, Normalize, Pad
import glob
import random
from PIL import Image
import matplotlib.pyplot as plt



## === cell 1
SEED = 42


def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(SEED)



## === cell 2
train_val_images = glob.glob(
    "/kaggle/input/paddy-disease-classification/train_images/*/*"
)
test_images = [
    p
    for p in glob.glob("/kaggle/input/paddy-disease-classification/test_images/*")
    if os.path.isfile(p) and p.lower().endswith(".jpg")
]


def train_val_split(images_list, train_size):
    n = int(len(images_list) * train_size)
    return images_list[:n], images_list[n:]


random.shuffle(train_val_images)
train_images, val_images = train_val_split(train_val_images, train_size=0.9)
print(len(train_images), len(val_images), len(test_images))



## === cell 3
transform = Compose(
    [
        CenterCrop(446),
        Resize((256, 256)),
        Pad(11),
        ToTensor(),
        Normalize([0.4965, 0.5858, 0.2238], [0.2222, 0.2218, 0.1968]),
    ]
)



## === cell 4
target_names = [
    os.path.basename(p)
    for p in glob.glob("/kaggle/input/paddy-disease-classification/train_images/*")
    if os.path.isdir(p)
]
target_names.sort()
print("Classes:", target_names)



## === cell 5
target_2_int = {name: idx for idx, name in enumerate(target_names)}
int_2_target = {idx: name for name, idx in target_2_int.items()}
print("Mapping example:", list(target_2_int.items())[:3])




## === cell 6
class ImageDataset(Dataset):
    def __init__(self, img_paths, transform=None, train=True):
        self.img_paths = img_paths
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.train:
            class_name = img_path.split("/")[-2]
            label = target_2_int[class_name]
            return image, label
        else:
            image_id = img_path.split("/")[-1]
            return image, image_id




## === cell 7
training_data = ImageDataset(train_images, transform, train=True)
val_data = ImageDataset(val_images, transform, train=True)
test_data = ImageDataset(test_images, transform, train=False)



## === cell 8
train_dataloader = DataLoader(training_data, batch_size=16, shuffle=True, num_workers=2)
val_dataloader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=2)
test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2)



## === cell 9
for x, y in train_dataloader:
    print("Batch image shape:", x.shape, "Batch label shape:", y.shape)
    break



## === cell 10
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



## === cell 11
set_seed(SEED)
model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
model.fc = nn.Linear(512, len(target_names))
model = model.to(device)



## === cell 12
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)




## === cell 13
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss_val, cur = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss_val:.6f}  [{cur}/{size}]")




## === cell 14
def evaluate(dataloader, model, loss_fn, mode="Val"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0.0, 0.0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    acc = correct / size
    print(f"{mode} - Accuracy: {100*acc:.1f}%, Avg loss: {test_loss:.6f}")




## === cell 15
epochs = 5
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train(train_dataloader, model, loss_fn, optimizer)
    evaluate(val_dataloader, model, loss_fn, mode="Val")
print("Training completed.")



## === cell 16
num_test = len(test_dataloader.dataset)
test_labels = np.empty(num_test, dtype=object)
test_ids = np.empty(num_test, dtype=object)

model.eval()
with torch.no_grad():
    for i, (X, img_id) in enumerate(test_dataloader):
        X = X.to(device)
        preds = model(X)
        pred_idx = torch.argmax(preds, dim=1).item()
        test_labels[i] = int_2_target[pred_idx]
        test_ids[i] = img_id  # already filename like "12345.jpg"



## === cell 17
submission = pd.DataFrame({"image_id": test_ids, "label": test_labels})
print(submission.head())



## === cell 18
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
