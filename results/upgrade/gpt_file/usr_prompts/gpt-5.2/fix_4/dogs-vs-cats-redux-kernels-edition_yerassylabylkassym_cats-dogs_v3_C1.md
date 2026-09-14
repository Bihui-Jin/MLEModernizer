# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import random
import glob
import re

import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:3]:
        print(os.path.join(dirname, filename))



## === cell 1
from zipfile import ZipFile

with ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip", "r") as z:
    z.extractall("/kaggle/working")
with ZipFile("/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip", "r") as z:
    z.extractall("/kaggle/working")

print(
    "Extraction done. Top-level /kaggle/working contents:",
    sorted(os.listdir("/kaggle/working"))[:20],
)



## === cell 2
device = "cuda" if torch.cuda.is_available() else "cpu"
device



## === cell 3
from PIL import Image



def _list_jpgs(d):
    return glob.glob(os.path.join(d, "*.jpg"))


def _is_numeric_stem_jpg(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem.isdigit()


def _has_numeric_jpgs(d, min_count=50):
    jpgs = _list_jpgs(d)
    if len(jpgs) < min_count:
        return False
    return sum(_is_numeric_stem_jpg(p) for p in jpgs) > (0.8 * len(jpgs))


def _has_trainstyle_jpgs(d, min_count=200):
    jpgs = _list_jpgs(d)
    if len(jpgs) < min_count:
        return False
    bn = [os.path.basename(p).lower() for p in jpgs[:200]]
    return any("cat." in b for b in bn) and any("dog." in b for b in bn)


def _find_dir_by_predicate(preferred_candidates, predicate):
    for p in preferred_candidates:
        if os.path.isdir(p) and predicate(p):
            return p
    for root, dirs, files in os.walk("/kaggle/working"):
        if any(f.lower().endswith(".jpg") for f in files):
            if predicate(root):
                return root
    return None


train_candidates = [
    "/kaggle/working/train/train",
    "/kaggle/working/train",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train/train",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/train",
]
test_candidates = [
    "/kaggle/working/test/test",
    "/kaggle/working/test",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/test/test",
    "/kaggle/working/dogs-vs-cats-redux-kernels-edition/test",
]

train_folder_path = _find_dir_by_predicate(train_candidates, _has_trainstyle_jpgs)
test_folder_path = _find_dir_by_predicate(test_candidates, _has_numeric_jpgs)

if train_folder_path is None:
    raise FileNotFoundError(
        "Could not find extracted training JPG directory under /kaggle/working."
    )
if test_folder_path is None:
    raise FileNotFoundError(
        "Could not find extracted test JPG directory under /kaggle/working."
    )

print("Using train_folder_path:", train_folder_path)
print("Using test_folder_path:", test_folder_path)

train_paths = sorted(glob.glob(os.path.join(train_folder_path, "*.jpg")))
test_paths = sorted(glob.glob(os.path.join(test_folder_path, "*.jpg")))

if len(train_paths) == 0:
    raise FileNotFoundError(f"No training JPGs found in: {train_folder_path}")
if len(test_paths) == 0:
    raise FileNotFoundError(f"No test JPGs found in: {test_folder_path}")


def load_image_as_tensor(path, size=(150, 150)):
    img = Image.open(path).convert("RGB")
    img = img.resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1)  # C,H,W
    return t


def parse_test_id(path: str) -> int:
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.isdigit():
        return int(stem)
    m = re.findall(r"\d+", stem)
    if not m:
        raise ValueError(f"Could not parse numeric test id from filename: {path}")
    return int(m[-1])


train_labels = [
    1 if "dog." in os.path.basename(p).lower() else 0 for p in train_paths
]  # 1=dog, 0=cat
test_ids = [parse_test_id(p) for p in test_paths]

print("Discovered train/test files:", len(train_paths), len(test_paths))
print("First 5 test ids:", test_ids[:5])



## === cell 4
len(train_paths), len(test_paths)



## === cell 5
train_labels.count(0), train_labels.count(1)



## === cell 6
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

paths = np.array(train_paths)
labels = np.array(train_labels, dtype=np.int64)

X_train_paths, X_val_paths, y_train_np, y_val_np = train_test_split(
    paths, labels, test_size=0.25, random_state=42, stratify=labels
)

print("Train class counts:", np.bincount(y_train_np))


class CatsAndDogsPaths(Dataset):
    def __init__(self, paths, labels=None, size=(150, 150)):
        self.paths = list(paths)
        self.labels = None if labels is None else list(labels)
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = load_image_as_tensor(self.paths[idx], size=self.size)
        if self.labels is None:
            return img
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def collate_stack_train(batch):
    imgs, labels = zip(*batch)
    return torch.stack(list(imgs), dim=0), torch.stack(list(labels), dim=0)


def collate_stack_test(batch):
    return torch.stack(list(batch), dim=0)


batch_size = 2
training_data = CatsAndDogsPaths(X_train_paths, y_train_np)
val_data = CatsAndDogsPaths(X_val_paths, y_val_np)
test_data = CatsAndDogsPaths(test_paths, labels=None)

train_dataloader = DataLoader(
    training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_stack_train
)
val_dataloader = DataLoader(
    val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_stack_train
)
test_dataloader = DataLoader(
    test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_stack_test
)



## === cell 7
from torch import nn


class CNN(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


with torch.no_grad():
    dummy = torch.zeros(1, 3, 150, 150, device=device)
    dummy_out = nn.Sequential(
        nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(6),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(16),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
    ).to(device)(dummy)
    inferred_in_features = int(np.prod(dummy_out.shape[1:]))

model = CNN(inferred_in_features).to(device)



## === cell 8
model.train()

epochs = 7
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(epochs):
    print(f"Epoch: {epoch + 1}")
    running_loss = 0.0
    correct = 0
    total = 0

    for i, dt in enumerate(train_dataloader):
        inputs, labels = dt
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        model.train()

        pred = model(inputs)
        loss = loss_fn(pred, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item())

        predicted = torch.argmax(pred, dim=1)
        correct += int((predicted == labels).sum().item())
        total += int(labels.size(0))

    print(f"Accuracy: {correct / max(total, 1):.2%}")
    last_loss = running_loss / 1000.0
    print(f"Loss: {last_loss} \n ___________\n")



## === cell 9
model.eval()
val_correct = 0
val_total = 0

with torch.no_grad():
    for inputs, labels in val_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        predicted = torch.argmax(outputs, dim=1)

        val_correct += int((predicted == labels).sum().item())
        val_total += int(labels.size(0))

print(f"Test Accuracy: {val_correct / max(val_total, 1):.2%}")



## === cell 10
xb = next(iter(test_dataloader))
xb.shape



## === cell 11
model.eval()

probs_dog = []
with torch.no_grad():
    for xb in test_dataloader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy().tolist()
        probs_dog.extend(prob)

probs_dog = np.clip(np.array(probs_dog, dtype=np.float64), 1e-6, 1 - 1e-6)
len(probs_dog), len(test_ids)



## === cell 12
sample_path = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
submission = pd.read_csv(sample_path)

pred_df = pd.DataFrame({"id": test_ids, "label": probs_dog})
pred_df = pred_df.groupby("id", as_index=False)["label"].mean()

submission = submission[["id"]].merge(pred_df, on="id", how="left")

if submission["label"].isna().any():
    missing = int(submission["label"].isna().sum())
    missing_ids = submission.loc[submission["label"].isna(), "id"].head(10).tolist()
    raise ValueError(
        f"Missing predictions for {missing} ids; examples: {missing_ids}. Check test file parsing."
    )

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv saved to:", os.path.abspath("submission.csv"))
