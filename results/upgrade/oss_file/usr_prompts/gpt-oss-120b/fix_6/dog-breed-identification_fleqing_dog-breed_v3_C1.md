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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.12

# 3. Installed packages

albumentations==2.0.8
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
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.43748

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import copy
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.transforms import ToPILImage
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from PIL import Image

from torch.cuda.amp import autocast, GradScaler

try:
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, hard))
except Exception:
    pass  # on platforms where resource is unavailable, ignore

NUM_WORKERS = min(4, os.cpu_count() or 1)  # safe upper bound for Kaggle CPUs




## === cell 1
base_path = "/kaggle/input/dog-breed-identification"

train_dir_candidates = [
    os.path.join(base_path, "train"),
    os.path.join(base_path, "dog-breed-identification", "train"),
]
train_dir = next((p for p in train_dir_candidates if os.path.isdir(p)), None)
if train_dir is None:
    raise FileNotFoundError("Training image folder not found in expected locations.")

test_dir_candidates = [
    os.path.join(base_path, "test"),
    os.path.join(base_path, "dog-breed-identification", "test"),
]
test_dir = next((p for p in test_dir_candidates if os.path.isdir(p)), None)
if test_dir is None:
    raise FileNotFoundError("Test image folder not found in expected locations.")

test_ids = sorted(
    [fname[:-4] for fname in os.listdir(test_dir) if fname.lower().endswith(".jpg")]
)
test_data = pd.DataFrame({"id": test_ids})
test_data.head()




## === cell 2
transforms_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
transforms_test = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)




## === cell 3
class Dog_Breed(Dataset):
    """
    PyTorch Dataset for dog breed images.
    Handles both training/validation and test mode.
    """

    def __init__(self, df, transform=None, test=False):
        self.df = df
        self.ids = list(df["id"])
        self.test = test
        self.transform = transform

        base_path = "/kaggle/input/dog-breed-identification"
        train_dir_candidates = [
            os.path.join(base_path, "train"),
            os.path.join(base_path, "dog-breed-identification", "train"),
        ]
        test_dir_candidates = [
            os.path.join(base_path, "test"),
            os.path.join(base_path, "dog-breed-identification", "test"),
        ]
        self.train_dir = next(
            (p for p in train_dir_candidates if os.path.isdir(p)), None
        )
        self.test_dir = next((p for p in test_dir_candidates if os.path.isdir(p)), None)

        if not self.test and self.train_dir is None:
            raise FileNotFoundError("Training images directory not found.")
        if self.test and self.test_dir is None:
            raise FileNotFoundError("Test images directory not found.")

        if not self.test:
            self.labels = list(df["number"])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(
            self.test_dir if self.test else self.train_dir, img_id + ".jpg"
        )
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.test:
            return image
        else:
            label = self.labels[idx]
            return image, label




## === cell 4
def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


device = get_device()
print(f"Using device: {device}")

if device == "cuda":
    torch.backends.cudnn.benchmark = True




## === cell 5
train_data = pd.read_csv("/kaggle/input/dog-breed-identification/labels.csv")
labels = sorted(train_data["breed"].unique())
label_to_idx = {label: idx for idx, label in enumerate(labels)}
train_data["number"] = train_data["breed"].map(label_to_idx)

train_df, valid_df = train_test_split(
    train_data, test_size=0.2, random_state=42, stratify=train_data["number"]
)
trainset = Dog_Breed(train_df, transform=transforms_train, test=False)
validset = Dog_Breed(valid_df, transform=transforms_test, test=False)




## === cell 6
train_loader = DataLoader(
    trainset,
    batch_size=32,
    shuffle=True,
    drop_last=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
valid_loader = DataLoader(
    validset,
    batch_size=32,
    shuffle=False,
    drop_last=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)




## === cell 7
class MyResNet50(nn.Module):
    def __init__(self, num_classes=120):
        super(MyResNet50, self).__init__()
        self.net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        for param in self.net.parameters():
            param.requires_grad = False
        in_features = self.net.fc.in_features
        self.net.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.net(x)




## === cell 8
def train_model(model, train_loader, valid_loader, loss_fn, optimizer, epochs, device):
    net = model.to(device)
    best_epoch = 0
    best_score = -1.0  # higher validation accuracy is better
    best_state = None
    early_stop_rounds = 10
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-4)

    scaler = GradScaler() if device == "cuda" else None

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            optimizer.zero_grad()
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast(enabled=(device == "cuda")):
                outputs = net(imgs)
                loss = loss_fn(outputs, targets)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()
        train_acc = correct / total
        print(
            f"epoch {epoch}: loss={epoch_loss/len(train_loader):.4f}, train_acc={train_acc:.4f}",
            end=" ",
        )

        net.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, targets in tqdm(valid_loader, desc=f"Epoch {epoch} [val]"):
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast(enabled=(device == "cuda")):
                    outputs = net(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_acc = val_correct / val_total
        print(f"val_acc={val_acc:.4f}")

        if val_acc > best_score:
            best_score = val_acc
            best_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch
            print(">>> New best model saved")

        if epoch - best_epoch >= early_stop_rounds:
            print("Early stopping triggered")
            break

    net.load_state_dict(best_state)

    testset = Dog_Breed(test_data, transform=transforms_test, test=True)
    test_loader = DataLoader(
        testset,
        batch_size=64,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    all_logits = []
    net.eval()
    with torch.no_grad():
        for imgs in tqdm(test_loader, desc="Test inference"):
            imgs = imgs.to(device, non_blocking=True)
            with autocast(enabled=(device == "cuda")):
                logits = net(imgs)
            all_logits.append(logits.cpu())

    logits_tensor = torch.cat(all_logits, dim=0)  # shape (num_test, num_classes)
    probs = F.softmax(logits_tensor, dim=1)  # convert to probabilities
    return probs




## === cell 9
learn_rate = 0.001
epochs = 30




## === cell 10
model = MyResNet50()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.net.fc.parameters(), lr=learn_rate, weight_decay=1e-5)

test_probabilities = train_model(
    model, train_loader, valid_loader, criterion, optimizer, epochs, device
)

submission = pd.DataFrame(test_probabilities.numpy(), columns=labels)
submission.insert(0, "id", test_data["id"])
submission_path = "/kaggle/working/dog_breed.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
