# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
from PIL import Image
from tqdm import tqdm
import copy
import pandas as pd
import os
import torch
import torchvision
from torchvision import transforms
from torch import optim
from torch.optim import lr_scheduler

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True  # faster FP32 matmuls on recent GPUs
torch.backends.cuda.matmul.allow_tf32 = True  # same for CUDA matmuls
torch.set_float32_matmul_precision("high")  # use highest‑throughput precision

try:
    from efficientnet_pytorch import EfficientNet  # type: ignore
except Exception:
    EfficientNet = None  # will fall back to torchvision model later




## === cell 1
transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(640),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
transform_valid = transforms.Compose(
    [
        transforms.Resize(640),
        transforms.CenterCrop(640),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)




## === cell 2
csv_path = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"
train_df = pd.read_csv(csv_path)
lines = train_df.apply(lambda row: f"{row['image']},{row['labels']}", axis=1).tolist()
import random

for _ in range(5):
    random.shuffle(lines)

cnt = int(len(lines) * 0.9)
train_csv = lines[:cnt]
valid_csv = lines[cnt:]




## === cell 3
all_labels = set()
for line in lines:
    _img, lbls = line.split(",", 1)
    primary = lbls.strip().split()[0]
    all_labels.add(primary)

label2idx = {lbl: idx for idx, lbl in enumerate(sorted(all_labels))}
idx2label = {idx: lbl for lbl, idx in label2idx.items()}




## === cell 4
class torchvision_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, csv_lines, transform=None):
        self.csv = csv_lines
        self.root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        line = self.csv[idx].strip()
        image_name, label_name = line.split(",", 1)
        img_path = os.path.join(self.root, image_name)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        primary_label = label_name.strip().split()[0]
        label_idx = label2idx[primary_label]
        return img, label_idx




## === cell 5
worker_cnt = min(16, os.cpu_count() or 1)

DATA_ROOT = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
train_dataset = torchvision_Dataset(DATA_ROOT, train_csv, transform_train)
valid_dataset = torchvision_Dataset(DATA_ROOT, valid_csv, transform_valid)

train_dataloaders = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=worker_cnt,
    pin_memory=True,
    persistent_workers=True,
)
valid_dataloaders = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=worker_cnt,
    pin_memory=True,
    persistent_workers=True,
)




## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")




## === cell 7
if EfficientNet is not None:
    try:
        model_ft = EfficientNet.from_name("efficientnet-b4", num_classes=len(label2idx))
    except Exception:
        model_ft = None
else:
    model_ft = None

if model_ft is None:
    model_ft = torchvision.models.efficientnet_b4(weights="IMAGENET1K_V1")
    in_features = model_ft.classifier[1].in_features
    model_ft.classifier[1] = torch.nn.Linear(in_features, len(label2idx))

for param in model_ft.parameters():
    param.requires_grad = False
for param in model_ft.classifier[1].parameters():
    param.requires_grad = True

model_ft = model_ft.to(device)

model_ft = torch.compile(model_ft, mode="reduce-overhead")




## === cell 8
criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(
    filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.001, momentum=0.9
)
cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_ft, T_max=30, eta_min=0
)


class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [
                        base_lr * self.multiplier for base_lr in self.base_lrs
                    ]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [
            base_lr
            * ((self.multiplier - 1.0) * self.last_epoch / self.total_epoch + 1.0)
            for base_lr in self.base_lrs
        ]

    def step(self, epoch=None, metrics=None):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epoch)
        else:
            return super().step(epoch)


exp_lr_scheduler = GradualWarmupScheduler(
    optimizer_ft, multiplier=100, total_epoch=3, after_scheduler=cosine_scheduler
)




## === cell 9
def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    scaler = torch.cuda.amp.GradScaler()  # mixed‑precision scaler

    os.makedirs("outputs", exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_corrects = 0
        train_cnt = 0
        for inputs, labels in tqdm(
            train_dataloaders, desc=f"Epoch {epoch+1}/{num_epochs} - Train"
        ):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data)
            train_cnt += inputs.size(0)

        scheduler.step()
        model.eval()
        valid_corrects = 0
        valid_cnt = 0
        for inputs, labels in tqdm(
            valid_dataloaders, desc=f"Epoch {epoch+1}/{num_epochs} - Valid"
        ):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
            valid_corrects += torch.sum(preds == labels.data)
            valid_cnt += inputs.size(0)

        epoch_acc = valid_corrects.double() / valid_cnt
        print(f"Epoch {epoch+1} validation accuracy: {epoch_acc:.4f}")
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f"outputs/best_epoch_{epoch}.pth")
    model.load_state_dict(best_model_wts)
    return model




## === cell 10
model_ft = train_model(
    model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=2
)




## === cell 11
ckpt_path = "/kaggle/input/best-model/28.pth"
if os.path.exists(ckpt_path):
    model_ft.load_state_dict(torch.load(ckpt_path, map_location=device))
    model_ft.to(device)
else:
    print("Checkpoint not found – using the trained model above.")




## === cell 12
from glob import glob

test_img_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"
img_paths = sorted(glob(os.path.join(test_img_dir, "*.jpg")))
submission_rows = []


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        filename = os.path.basename(p)
        return img, filename


def test_collate(batch):
    imgs = torch.stack([item[0] for item in batch])
    fnames = [item[1] for item in batch]
    return imgs, fnames


test_dataset = TestDataset(img_paths, transform_valid)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=worker_cnt,
    pin_memory=True,
    collate_fn=test_collate,
    persistent_workers=True,
)

model_ft.eval()
with torch.no_grad():
    for imgs, filenames in tqdm(test_loader, desc="Generating predictions"):
        imgs = imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = model_ft(imgs)
        _, pred_idxs = torch.max(outputs, 1)
        for pred_idx, fname in zip(pred_idxs, filenames):
            pred_label = idx2label[pred_idx.item()]
            submission_rows.append([fname, pred_label])

submission_df = pd.DataFrame(submission_rows, columns=["image", "labels"])
output_path = "/kaggle/working/submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")
