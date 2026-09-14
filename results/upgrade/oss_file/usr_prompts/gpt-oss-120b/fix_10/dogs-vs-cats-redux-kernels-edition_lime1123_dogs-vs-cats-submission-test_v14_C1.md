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
import os
import zipfile
import glob
import torch
import pandas as pd

torch.set_num_threads(8)  # use more threads than default to speed up CPU tensor ops


def unzip_if_needed(zip_path, target_dir):
    if not os.path.isdir(target_dir):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(os.path.dirname(target_dir))


base_input = "/kaggle/input/dogs-vs-cats-redux-kernels-edition"
train_zip = os.path.join(base_input, "train.zip")
test_zip = os.path.join(base_input, "test.zip")
unzip_if_needed(train_zip, "/kaggle/working/train")
unzip_if_needed(test_zip, "/kaggle/working/test")




## === cell 1
class Config:
    base_dir = "/kaggle/working/dogs-vs-cats-redux-kernels-edition"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    batch_size = 256
    num_workers = 4
    device = "cpu"
    epochs = 15  # increased epochs for better training
    lr = 5e-4
    optimizer = torch.optim.AdamW
    warmup_epochs = 0
    size = (64, 64)


cfg = Config()

torch.manual_seed(42)
import random, numpy as np

random.seed(42)
np.random.seed(42)




## === cell 2
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr
            * 0.5
            * (
                1.0
                + torch.cos(
                    torch.tensor(self.last_epoch / self.total_epochs * 3.1415926535)
                ).item()
            )
            for base_lr in self.base_lrs
        ]




## === cell 3
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image, ImageOps

train_imgs = []
train_labels = []
for label_dir, label in [("cat", 0), ("dog", 1)]:
    dir_path = os.path.join(cfg.train_dir, label_dir)
    for fp in glob.glob(os.path.join(dir_path, "*.jpg")):
        img = Image.open(fp).convert("RGB")
        img = img.resize(cfg.size)
        img_np = np.array(img).astype(np.float32) / 255.0  # HWC, 0‑1
        img_np = np.transpose(img_np, (2, 0, 1))  # CHW
        train_imgs.append(img_np)
        train_labels.append(label)

        img_flipped = ImageOps.mirror(img)
        img_fl_np = np.array(img_flipped).astype(np.float32) / 255.0
        img_fl_np = np.transpose(img_fl_np, (2, 0, 1))
        train_imgs.append(img_fl_np)
        train_labels.append(label)

train_imgs_tensor = torch.from_numpy(np.stack(train_imgs))  # (N,3,64,64)
train_labels_tensor = torch.from_numpy(np.array(train_labels, dtype=np.float32))

train_dataset = TensorDataset(train_imgs_tensor, train_labels_tensor)
train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,  # faster host‑to‑device copies (even on CPU)
)


class SimpleCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = torch.nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)




## === cell 4
model = SimpleCNN().to(cfg.device)

try:
    model = torch.compile(model)
except Exception:
    pass  # torch.compile may not be available; continue without compilation

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = cfg.optimizer(model.parameters(), lr=cfg.lr)

scheduler = WarmupCosineAnnealingLR(optimizer, cfg.warmup_epochs, cfg.epochs)

model.train()
for epoch in range(cfg.epochs):
    for imgs, labels in train_loader:
        imgs = imgs.to(cfg.device)
        labels = labels.to(cfg.device).unsqueeze(1)
        logits = model(imgs)
        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()  # update LR after each epoch

test_paths = glob.glob(os.path.join(cfg.test_dir, "**", "*.jpg"), recursive=True)
image_ids = [os.path.basename(p).split(".")[0] for p in test_paths]

test_imgs = []
for fp in test_paths:
    img = Image.open(fp).convert("RGB")
    img = img.resize(cfg.size)
    img_np = np.array(img).astype(np.float32) / 255.0
    img_np = np.transpose(img_np, (2, 0, 1))
    test_imgs.append(img_np)

test_imgs_tensor = torch.from_numpy(np.stack(test_imgs))
test_dataset = TensorDataset(test_imgs_tensor)
test_loader = DataLoader(
    test_dataset,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,  # faster copies during inference
)

model.eval()
all_preds = []
with torch.no_grad():
    for (imgs,) in test_loader:
        imgs = imgs.to(cfg.device)
        logits = model(imgs)
        probs = torch.sigmoid(logits).squeeze(1)
        all_preds.append(probs.cpu())
outputs_tensor = torch.cat(all_preds)
outputs = outputs_tensor




## === cell 5
sample_sub_path = (
    "/kaggle/input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv"
)
sample_sub = pd.read_csv(sample_sub_path, dtype={"id": str})
pred_dict = dict(zip(image_ids, outputs.tolist()))
submission = sample_sub.copy()
submission["label"] = submission["id"].map(pred_dict).fillna(0.5)
submission = submission.sort_values("id")
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path}")
print(submission.head())
