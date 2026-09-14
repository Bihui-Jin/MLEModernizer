# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.12

# 3. Installed packages



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

9.18462

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re
import random
import zipfile
import shutil
from glob import glob

import numpy as np
import pandas as pd

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")




## === cell 1
def _find_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


train_zip = _find_existing(
    [
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train.zip",
        "/kaggle/data/dogs-vs-cats-redux-kernels-edition/train.zip",
        "/kaggle/input/train.zip",
        "/kaggle/data/train.zip",
    ]
)
test_zip = _find_existing(
    [
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test.zip",
        "/kaggle/data/dogs-vs-cats-redux-kernels-edition/test.zip",
        "/kaggle/input/test.zip",
        "/kaggle/data/test.zip",
    ]
)

if train_zip is None or test_zip is None:
    raise FileNotFoundError(
        "Could not locate train.zip and/or test.zip in expected Kaggle input paths."
    )

work_dir = "/kaggle/working"
os.makedirs(work_dir, exist_ok=True)


def extract_if_needed(zip_path, dest_dir, expected_jpg_glob_anywhere: str = "**/*.jpg"):
    if (
        len(glob(os.path.join(dest_dir, expected_jpg_glob_anywhere), recursive=True))
        > 0
    ):
        return
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest_dir)


extract_if_needed(train_zip, work_dir, "**/*.jpg")
extract_if_needed(test_zip, work_dir, "**/*.jpg")

print(
    "After extraction, top-level in /kaggle/working:",
    sorted([p for p in os.listdir(work_dir) if not p.startswith(".")])[:50],
)


def find_image_dir(root: str, want: str) -> str:
    candidates = [
        os.path.join(root, want),
        os.path.join(root, want, want),
        os.path.join(root, "dogs-vs-cats-redux-kernels-edition", want),
        os.path.join(root, "dogs-vs-cats-redux-kernels-edition", want, want),
        os.path.join(root, "dogs-vs-cats-redux-kernels-edition", want, want, want),
        os.path.join(root, want, want, want),
    ]
    for c in candidates:
        if os.path.isdir(c) and len(glob(os.path.join(c, "*.jpg"))) > 0:
            return c

    hits = glob(os.path.join(root, "**", want, "*.jpg"), recursive=True)
    if hits:
        return os.path.dirname(hits[0])

    if want in ("train", "test"):
        hits2 = glob(os.path.join(root, "**", want, want, "*.jpg"), recursive=True)
        if hits2:
            return os.path.dirname(hits2[0])

    raise FileNotFoundError(
        f"Could not find extracted '{want}' jpg directory under {root}"
    )


original_train_dir = find_image_dir(work_dir, "train")
original_test_dir = find_image_dir(work_dir, "test")
print("Resolved original_train_dir:", original_train_dir)
print("Resolved original_test_dir:", original_test_dir)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1694738744.py in <cell line: 0>()
     84 
     85 
---> 86 original_train_dir = find_image_dir(work_dir, "train")
     87 original_test_dir = find_image_dir(work_dir, "test")
     88 print("Resolved original_train_dir:", original_train_dir)

/tmp/ipykernel_11/1694738744.py in find_image_dir(root, want)
     79             return os.path.dirname(hits2[0])
     80 
---> 81     raise FileNotFoundError(
     82         f"Could not find extracted '{want}' jpg directory under {root}"
     83     )

FileNotFoundError: Could not find extracted 'train' jpg directory under /kaggle/working

## === cell 2
prepared_root = os.path.join(work_dir, "prepared")
train_dir = os.path.join(prepared_root, "train")
valid_dir = os.path.join(prepared_root, "valid")

cats_train = os.path.join(train_dir, "cat")
dogs_train = os.path.join(train_dir, "dog")
cats_valid = os.path.join(valid_dir, "cat")
dogs_valid = os.path.join(valid_dir, "dog")

for d in [cats_train, dogs_train, cats_valid, dogs_valid]:
    os.makedirs(d, exist_ok=True)

train_files = [f for f in os.listdir(original_train_dir) if f.lower().endswith(".jpg")]
cat_files = sorted([f for f in train_files if f.startswith("cat.")])
dog_files = sorted([f for f in train_files if f.startswith("dog.")])

if len(cat_files) == 0 or len(dog_files) == 0:
    raise RuntimeError(
        f"Did not find expected cat./dog. files in {original_train_dir}. Found {len(train_files)} jpgs."
    )

rng = random.Random(SEED)
rng.shuffle(cat_files)
rng.shuffle(dog_files)


def split_and_copy(files, dst_train, dst_valid, valid_frac=0.1):
    n_valid = int(len(files) * valid_frac)
    valid_set = set(files[:n_valid])
    for f in files:
        src = os.path.join(original_train_dir, f)
        if f in valid_set:
            dst = os.path.join(dst_valid, f)
        else:
            dst = os.path.join(dst_train, f)
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    return n_valid, len(files) - n_valid


cat_n_valid, cat_n_train = split_and_copy(
    cat_files, cats_train, cats_valid, valid_frac=0.1
)
dog_n_valid, dog_n_train = split_and_copy(
    dog_files, dogs_train, dogs_valid, valid_frac=0.1
)

print(
    f"Prepared dataset: cat train/valid={cat_n_train}/{cat_n_valid}, dog train/valid={dog_n_train}/{dog_n_valid}"
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1655640716.py in <cell line: 0>()
     11     os.makedirs(d, exist_ok=True)
     12 
---> 13 train_files = [f for f in os.listdir(original_train_dir) if f.lower().endswith(".jpg")]
     14 cat_files = sorted([f for f in train_files if f.startswith("cat.")])
     15 dog_files = sorted([f for f in train_files if f.startswith("dog.")])

NameError: name 'original_train_dir' is not defined

## === cell 3
img_tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

train_dataset = ImageFolder(root=train_dir, transform=img_tfms)
valid_dataset = ImageFolder(root=valid_dir, transform=img_tfms)

print("Class to idx:", train_dataset.class_to_idx)
print("Train size:", len(train_dataset), "Valid size:", len(valid_dataset))

train_dl = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=(device == "cuda"),
)
valid_dl = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=(device == "cuda"),
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3825692161.py in <cell line: 0>()
      1 img_tfms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
      2 
----> 3 train_dataset = ImageFolder(root=train_dir, transform=img_tfms)
      4 valid_dataset = ImageFolder(root=valid_dir, transform=img_tfms)
      5 

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in __init__(self, root, transform, target_transform, loader, is_valid_file, allow_empty)
    326         allow_empty: bool = False,
    327     ):
--> 328         super().__init__(
    329             root,
    330             loader,

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in __init__(self, root, loader, extensions, transform, target_transform, is_valid_file, allow_empty)
    148         super().__init__(root, transform=transform, target_transform=target_transform)
    149         classes, class_to_idx = self.find_classes(self.root)
--> 150         samples = self.make_dataset(
    151             self.root,
    152             class_to_idx=class_to_idx,

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in make_dataset(directory, class_to_idx, extensions, is_valid_file, allow_empty)
    201             # is potentially overridden and thus could have a different logic.
    202             raise ValueError("The class_to_idx parameter cannot be None.")
--> 203         return make_dataset(
    204             directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file, allow_empty=allow_empty
    205         )

/usr/local/lib/python3.11/dist-packages/torchvision/datasets/folder.py in make_dataset(directory, class_to_idx, extensions, is_valid_file, allow_empty)
    102         if extensions is not None:
    103             msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
--> 104         raise FileNotFoundError(msg)
    105 
    106     return instances

FileNotFoundError: Found no valid file for the classes cat, dog. Supported extensions are: .jpg, .jpeg, .png, .ppm, .bmp, .pgm, .tif, .tiff, .webp

## === cell 4
class mynet(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.convnet = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, 1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 12 * 12, 2),
        )

    def forward(self, x):
        x = self.convnet(x)
        return x


model = mynet().to(device)



## === cell 5
from torch.optim import SGD

loss_fn = nn.CrossEntropyLoss()
opt = SGD(model.parameters(), lr=1e-3)

epochs = 2
train_losses, test_losses = [], []
train_accs, test_accs = [], []

for epoch in range(epochs):
    train_loss = 0.0
    train_acc = 0.0

    model.train()
    for batch, (x, y) in enumerate(train_dl):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()

        train_loss += loss.item()
        y_pred_class = torch.argmax(torch.softmax(pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(pred)

    avg_train_loss = train_loss / len(train_dl)
    avg_train_acc = train_acc / len(train_dl)
    train_losses.append(avg_train_loss)
    train_accs.append(avg_train_acc)
    print(
        f"Epoch: {epoch} train loss: {avg_train_loss:.4f} train acc: {avg_train_acc:.4f}"
    )

    model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for batch, (x, y) in enumerate(valid_dl):
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            y_pred_class_test = torch.argmax(torch.softmax(pred, dim=1), dim=1)
            test_acc += (y_pred_class_test == y).sum().item() / len(pred)

    avg_test_loss = test_loss / len(valid_dl)
    avg_test_acc = test_acc / len(valid_dl)
    test_losses.append(avg_test_loss)
    test_accs.append(avg_test_acc)
    print(f"Epoch: {epoch} test loss: {avg_test_loss:.4f} test acc: {avg_test_acc:.4f}")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1362653853.py in <cell line: 0>()
     13 
     14     model.train()
---> 15     for batch, (x, y) in enumerate(train_dl):
     16         x, y = x.to(device), y.to(device)
     17         pred = model(x)

NameError: name 'train_dl' is not defined

## === cell 6
try:
    weights = models.ResNet18_Weights.DEFAULT
except Exception:
    weights = None

tl_model = models.resnet18(weights=weights)
for param in tl_model.parameters():
    param.requires_grad = False

num_classes = 2
tl_model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
input_tolinear = tl_model.fc.in_features
tl_model.fc = nn.Linear(input_tolinear, num_classes)
tl_model = tl_model.to(device)



## === cell 7
loss_fn = nn.CrossEntropyLoss()
opt = SGD([p for p in tl_model.parameters() if p.requires_grad], lr=1e-3)

epochs = 10
train_losses, test_losses = [], []
train_accs, test_accs = [], []

for epoch in range(epochs):
    train_loss = 0.0
    train_acc = 0.0

    tl_model.train()
    for batch, (x, y) in enumerate(train_dl):
        x, y = x.to(device), y.to(device)
        pred = tl_model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        opt.zero_grad()

        train_loss += loss.item()
        y_pred_class = torch.argmax(torch.softmax(pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(pred)

    avg_train_loss = train_loss / len(train_dl)
    avg_train_acc = train_acc / len(train_dl)
    train_losses.append(avg_train_loss)
    train_accs.append(avg_train_acc)
    print(
        f"TL Epoch: {epoch} train loss: {avg_train_loss:.4f} train acc: {avg_train_acc:.4f}"
    )

    tl_model.eval()
    test_loss = 0.0
    test_acc = 0.0
    with torch.no_grad():
        for batch, (x, y) in enumerate(valid_dl):
            x, y = x.to(device), y.to(device)
            pred = tl_model(x)
            loss = loss_fn(pred, y)
            test_loss += loss.item()
            y_pred_class_test = torch.argmax(torch.softmax(pred, dim=1), dim=1)
            test_acc += (y_pred_class_test == y).sum().item() / len(pred)

    avg_test_loss = test_loss / len(valid_dl)
    avg_test_acc = test_acc / len(valid_dl)
    test_losses.append(avg_test_loss)
    test_accs.append(avg_test_acc)
    print(
        f"TL Epoch: {epoch} test loss: {avg_test_loss:.4f} test acc: {avg_test_acc:.4f}"
    )



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3748223604.py in <cell line: 0>()
     11 
     12     tl_model.train()
---> 13     for batch, (x, y) in enumerate(train_dl):
     14         x, y = x.to(device), y.to(device)
     15         pred = tl_model(x)

NameError: name 'train_dl' is not defined

## === cell 8
test_dir = original_test_dir

sample_sub_path = _find_existing(
    [
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv",
        "/kaggle/data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)
if sample_sub_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected Kaggle input paths."
    )

sample_sub = pd.read_csv(sample_sub_path)
expected_ids = sample_sub["id"].astype(str).tolist()
expected_id_set = set(expected_ids)
print("Sample submission rows:", len(sample_sub))

custom_trans = transforms.Compose([transforms.Resize((224, 224))])


def transform_image(image_path: str) -> torch.Tensor:
    img = torchvision.io.read_image(str(image_path)).float() / 255.0
    if img.ndim == 3 and img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    elif img.ndim == 3 and img.shape[0] == 4:
        img = img[:3]
    img = custom_trans(img)
    return img


dogs_idx = train_dataset.class_to_idx.get("dog", 1)


def predict_proba_dog(image_tensor: torch.Tensor, model: nn.Module) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=1)[0]
        return float(probs[dogs_idx].clamp(1e-7, 1 - 1e-7).item())


pred_by_id = {}
test_files = [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
for f in test_files:
    img_id = os.path.splitext(f)[0]  # e.g. "1"
    if img_id not in expected_id_set:
        continue
    p = predict_proba_dog(transform_image(os.path.join(test_dir, f)), tl_model)
    pred_by_id[img_id] = p

missing = [i for i in expected_ids if i not in pred_by_id]
if missing:
    print(f"Warning: {len(missing)} ids missing from predictions; filling with 0.5")
    for i in missing:
        pred_by_id[i] = 0.5

submission = pd.DataFrame(
    {
        "id": sample_sub["id"],
        "label": [pred_by_id[str(i)] for i in sample_sub["id"].astype(str).tolist()],
    }
)

out_path = os.path.join(work_dir, "submission.csv")
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
print("Label min/max:", submission["label"].min(), submission["label"].max())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/566323931.py in <cell line: 0>()
----> 1 test_dir = original_test_dir
      2 
      3 sample_sub_path = _find_existing(
      4     [
      5         "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv",

NameError: name 'original_test_dir' is not defined
