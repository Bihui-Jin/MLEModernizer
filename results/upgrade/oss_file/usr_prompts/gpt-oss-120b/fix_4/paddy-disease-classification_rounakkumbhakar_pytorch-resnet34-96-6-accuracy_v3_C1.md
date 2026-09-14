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

0.96082

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, glob, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    CenterCrop,
    Resize,
    ToTensor,
    Normalize,
    RandomHorizontalFlip,
)
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid



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
train_val_images = [
    p
    for p in glob.glob("/kaggle/input/paddy-disease-classification/train_images/*/*")
    if p.lower().endswith(".jpg")
]
test_images = [
    p for p in glob.glob("/kaggle/input/paddy-disease-classification/test_images/*.jpg")
]


def train_val_split(images_list, train_size):
    n = int(len(images_list) * train_size)
    train_list = images_list[:n]
    val_list = images_list[n:]
    return train_list, val_list


random.shuffle(train_val_images)
train_images, val_images = train_val_split(train_val_images, train_size=0.9)

print(len(train_images), len(val_images), len(test_images))



## === cell 3
train_transform = Compose(
    [
        CenterCrop(446),
        Resize((256, 256)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.4965, 0.5858, 0.2238], [0.2222, 0.2218, 0.1968]),
    ]
)
test_transform = Compose(
    [
        CenterCrop(446),
        Resize((256, 256)),
        ToTensor(),
        Normalize([0.4965, 0.5858, 0.2238], [0.2222, 0.2218, 0.1968]),
    ]
)



## === cell 4
target_names = [
    os.path.basename(p)
    for p in glob.glob("/kaggle/input/paddy-disease-classification/train_images/*")
    if os.path.isdir(p) and not os.path.basename(p).startswith(".")
]
target_names.sort()
target_2_int = {name: idx for idx, name in enumerate(target_names)}
int_2_target = {idx: name for name, idx in target_2_int.items()}




## === cell 5
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
            label_name = img_path.split("/")[-2]
            label = target_2_int[label_name]
            return image, label
        else:
            image_id = os.path.basename(img_path)
            return image, image_id




## === cell 6
training_data = ImageDataset(train_images, transform=train_transform, train=True)
val_data = ImageDataset(val_images, transform=test_transform, train=True)
test_data = ImageDataset(test_images, transform=test_transform, train=False)



## === cell 7
train_loader = DataLoader(training_data, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=0)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0)



## === cell 8
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")



## === cell 9
set_seed(SEED)
model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True)
model.fc = nn.Linear(model.fc.in_features, len(target_names))  # 10 classes
model = model.to(device)



## === cell 10
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)




## === cell 11
def train_one_epoch(dataloader, model, loss_fn, optimizer):
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
            cur = (batch + 1) * len(X)
            print(f"loss: {loss.item():>7f}  [{cur:>5d}/{size:>5d}]")


def evaluate(dataloader, model, loss_fn):
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
    correct /= size
    print(f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    return correct  # return accuracy for checkpointing


best_val_acc = 0.0
best_model_path = "best_resnet18.pth"

epochs = 15  # modestly increased epochs
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n{'-'*30}")
    train_one_epoch(train_loader, model, loss_fn, optimizer)
    val_acc = evaluate(val_loader, model, loss_fn)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"--> New best model saved (val acc: {best_val_acc*100:.2f}%)")
    scheduler.step()
print("Training complete")



## === cell 12
model.load_state_dict(torch.load(best_model_path, map_location=device))
model.eval()
predictions = []
image_ids = []

with torch.no_grad():
    for X, img_id in test_loader:
        X = X.to(device)
        logits = model(X)
        pred_idx = torch.argmax(logits, dim=1).item()
        predictions.append(int_2_target[pred_idx])
        image_ids.append(img_id)



## === cell 13
sub_df = pd.DataFrame({"image_id": image_ids, "label": predictions})
print("Submission shape:", sub_df.shape)
assert sub_df.shape[0] == len(test_images), "Row count mismatch!"



## === cell 14
sub_df.to_csv("submission.csv", index=False)
print("submission.csv written")
