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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.13

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
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

0.0929363700647763

# 6. Current score

0.04897

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

- What this solution (achieved 0.04897) has done: 'The fix locates the actual train and test image folders (handling the extra nested directory created by the zip extraction) and falls back to using the original input directory when the zip isn’t extracted. This ensures `train_list` and `test_list` are populated, allowing the data loaders, training loop, inference, and submission steps to run without NameError exceptions. No core model logic is changed.'

# 9. Code solution

## === cell 0
import os, glob, zipfile, copy
from tqdm import tqdm
import numpy as np, pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch, torch.nn as nn, torch.nn.functional as F, torch.utils.data as data
from torchvision import models, transforms



## === cell 1
base_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"

train_zip_path = os.path.join(base_dir, "train.zip")
test_zip_path = os.path.join(base_dir, "test.zip")

data_dir = "/kaggle/working/data"
os.makedirs(data_dir, exist_ok=True)

if os.path.isfile(train_zip_path):
    with zipfile.ZipFile(train_zip_path) as z:
        z.extractall(data_dir)
if os.path.isfile(test_zip_path):
    with zipfile.ZipFile(test_zip_path) as z:
        z.extractall(data_dir)


def locate_dir(*candidates):
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(f"None of the candidate directories exist: {candidates}")


train_dir = locate_dir(
    os.path.join(data_dir, "train"),
    os.path.join(data_dir, "dogs-vs-cats-redux-kernels-edition", "train"),
    os.path.join(base_dir, "train"),
)

test_dir = locate_dir(
    os.path.join(data_dir, "test"),
    os.path.join(data_dir, "dogs-vs-cats-redux-kernels-edition", "test"),
    os.path.join(base_dir, "test"),
)



## === cell 2
train_list = glob.glob(os.path.join(train_dir, "**", "*.jpg"), recursive=True)
test_list = glob.glob(os.path.join(test_dir, "**", "*.jpg"), recursive=True)

if len(train_list) == 0:
    raise RuntimeError(f"No training images found in {train_dir}")
if len(test_list) == 0:
    raise RuntimeError(f"No test images found in {test_dir}")

train_list, val_list = train_test_split(train_list, test_size=0.2, random_state=42)

print(f"Train images: {len(train_list)}")
print(f"Validation images: {len(val_list)}")
print(f"Test images: {len(test_list)}")



## === cell 3
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

train_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)




## === cell 4
class CatDogDataset(data.Dataset):
    def __init__(self, file_list, transform=None):
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img) if self.transform else img

        label_str = os.path.basename(img_path).split(".")[0]
        label = 1 if label_str == "dog" else 0
        return img, label




## === cell 5
batch_size = 32
lr = 0.001
epochs = 3  # small for quick execution

train_dataset = CatDogDataset(train_list, transform=train_transforms)
val_dataset = CatDogDataset(val_list, transform=val_transforms)

train_loader = data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
)
val_loader = data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
)



## === cell 6
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model.fc = nn.Linear(model.fc.in_features, 2)

for name, param in model.named_parameters():
    param.requires_grad = name.startswith("layer4")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

criterion = nn.CrossEntropyLoss()
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(trainable_params, lr=lr)



## === cell 7
best_accuracy = 0.0
best_state = copy.deepcopy(model.state_dict())
model_path = "/kaggle/working/model.pth"

for epoch in range(epochs):
    model.train()
    epoch_loss, epoch_acc = 0.0, 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += (outputs.argmax(dim=1) == labels).float().mean().item()
    epoch_loss /= len(train_loader)
    epoch_acc /= len(train_loader)

    model.eval()
    val_loss, val_acc = 0.0, 0.0
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc=f"Val Epoch {epoch+1}"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_acc += (outputs.argmax(dim=1) == labels).float().mean().item()
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)

    print(
        f"Epoch {epoch+1}/{epochs} - "
        f"train_loss: {epoch_loss:.4f} train_acc: {epoch_acc:.4f} | "
        f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f}"
    )

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        best_state = copy.deepcopy(model.state_dict())

torch.save(best_state, model_path)



## === cell 8
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

id_list, pred_list = [], []

with torch.no_grad():
    for img_path in tqdm(test_list, desc="Inference"):
        img = Image.open(img_path).convert("RGB")
        img_tensor = val_transforms(img).unsqueeze(0).to(device)
        logits = model(img_tensor)
        prob = F.softmax(logits, dim=1)[0, 1].item()  # probability of class "dog"
        img_id = int(os.path.basename(img_path).split(".")[0])
        id_list.append(img_id)
        pred_list.append(prob)



## === cell 9
sample_sub_path = os.path.join(base_dir, "sample_submission.csv")
submission = pd.read_csv(sample_sub_path)

submission.set_index("id", inplace=True)
submission.loc[id_list, "label"] = pred_list
submission.reset_index(inplace=True)

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission saved to {submission_path}")
print(submission.head())
