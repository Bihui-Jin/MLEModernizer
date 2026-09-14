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

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
        input/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
            test/
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
            train/
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
        working/
            dogs-vs-cats-redux-kernels-edition/
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                ... and 2 other files
                test/
                    1332.jpg (22.2 kB)
                    617.jpg (22.9 kB)
                    ... and 2498 other files
                train/
                    dog.10425.jpg (33.5 kB)
                    cat.2462.jpg (18.8 kB)
                    ... and 22498 other files
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
Here is some information about the columns:
id (int64) has range: 1.00 - 2000.00, 0 nan values
label (float64) has 1 unique values: [0.5]

# 5. Code solution

## === cell 0
import glob
import numpy as np
import random
import torch
import pandas as pd
import os
from pathlib import Path
from PIL import Image




## === cell 1
class Config:
    train_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/train"
    test_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/test"
    sample_sub_path = (
        "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
    )
    seed = 2025
    img_size = 224
    batch_size = 64
    epochs = 5  # main training epochs
    extra_epochs = 2  # short fine‑tuning on all data
    lr = 1e-4


cfg = Config()
torch.manual_seed(cfg.seed)
random.seed(cfg.seed)
np.random.seed(cfg.seed)



## === cell 2
test_paths = glob.glob(os.path.join(cfg.test_dir, "**", "*.jpg"), recursive=True)
image_ids = [int(os.path.basename(p).split(".")[0]) for p in test_paths]

required_ids = pd.read_csv(cfg.sample_sub_path)["id"].tolist()

train_all_paths = glob.glob(os.path.join(cfg.train_dir, "*.jpg"))
train_cat_paths = [p for p in train_all_paths if os.path.basename(p).startswith("cat")]
train_dog_paths = [p for p in train_all_paths if os.path.basename(p).startswith("dog")]

dog_ratio = len(train_dog_paths) / max(1, len(train_cat_paths) + len(train_dog_paths))
fallback_pred = np.full(len(test_paths), dog_ratio, dtype=float)



## === cell 3
try:
    import torchvision
    from torchvision import transforms, models
    from torch.utils.data import Dataset, DataLoader
    from torch import nn, optim

    class CatsDogsDataset(Dataset):
        def __init__(self, cat_paths, dog_paths, transform):
            self.paths = cat_paths + dog_paths
            self.labels = [0] * len(cat_paths) + [1] * len(dog_paths)
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            img_path = self.paths[idx]
            img = Image.open(img_path).convert("RGB")
            img = self.transform(img)
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    all_paths = train_cat_paths + train_dog_paths
    random.shuffle(all_paths)
    split_idx = int(0.9 * len(all_paths))
    train_paths = all_paths[:split_idx]
    val_paths = all_paths[split_idx:]

    def split_paths(paths):
        cat = [p for p in paths if os.path.basename(p).startswith("cat")]
        dog = [p for p in paths if os.path.basename(p).startswith("dog")]
        return cat, dog

    train_cat, train_dog = split_paths(train_paths)
    val_cat, val_dog = split_paths(val_paths)

    train_dataset = CatsDogsDataset(train_cat, train_dog, train_transform)
    val_dataset = CatsDogsDataset(val_cat, val_dog, val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    try:
        resnet_weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=resnet_weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 1)

    with torch.no_grad():
        model.fc.bias.fill_(torch.logit(torch.tensor(dog_ratio, dtype=torch.float32)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=False
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        running_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / max(1, len(train_loader.dataset))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                logits = model(imgs).squeeze(1)
                loss = criterion(logits, lbls)
                val_loss += loss.item() * imgs.size(0)
        val_loss /= max(1, len(val_loader.dataset))
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{cfg.epochs} - Train loss: {epoch_loss:.4f} - Val loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    if cfg.extra_epochs > 0:
        all_cat = train_cat + val_cat
        all_dog = train_dog + val_dog
        ft_dataset = CatsDogsDataset(all_cat, all_dog, train_transform)
        ft_loader = DataLoader(
            ft_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0
        )
        model.train()
        for epoch in range(cfg.extra_epochs):
            ft_loss = 0.0
            for imgs, lbls in ft_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                logits = model(imgs).squeeze(1)
                loss = criterion(logits, lbls)
                loss.backward()
                optimizer.step()
                ft_loss += loss.item() * imgs.size(0)
            ft_loss /= max(1, len(ft_loader.dataset))
            print(f"Fine‑tune epoch {epoch+1}/{cfg.extra_epochs} - Loss: {ft_loss:.4f}")

    test_transform = transforms.Compose(
        [
            transforms.Resize((cfg.img_size, cfg.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    model.eval()
    sigmoid = nn.Sigmoid()
    test_preds = []
    with torch.no_grad():
        for p in test_paths:
            img = Image.open(p).convert("RGB")
            img = test_transform(img).unsqueeze(0).to(device)
            logit = model(img).squeeze(1)
            pred = sigmoid(logit).item()
            test_preds.append(pred)

    pred_dict = dict(zip(image_ids, test_preds))
    sorted_vals = [pred_dict.get(i, dog_ratio) for i in required_ids]

except Exception as e:
    print(f"Model path not usable ({e}), falling back to constant dog ratio.")
    sorted_vals = [dog_ratio] * len(required_ids)



## === cell 4
submission = pd.DataFrame({"id": required_ids, "label": sorted_vals})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
