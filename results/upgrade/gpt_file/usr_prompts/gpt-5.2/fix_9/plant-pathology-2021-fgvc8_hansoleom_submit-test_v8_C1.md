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

# 5. Target score

0.2896768236380446

# 6. Current score

0.39132

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.3378) has done: 'I fix the crash in inference by filtering the test image glob to include only actual files (excluding the nested `test_images/` directory that causes `IsADirectoryError`). I also make the submission align exactly to `sample_submission.csv` image order to avoid any accidental mismatches, while keeping your model/training logic unchanged. Finally, I add a small safety fallback so every predicted label string is non-empty (as required by the submission format) and ensure the output is written to `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.45644) has done: 'Your current score (0.3378) is higher than the target (0.2897), so the goal is to *slightly reduce* performance toward the target band with minimal, safe changes. The smallest lever that preserves your core model/training/inference logic is prediction post-processing: instead of always emitting the single argmax class, we optionally emit multiple labels when the model is uncertain, using a fixed probability threshold. This typically lowers mean F1 on this competition (more false positives from extra labels), moving the score down toward your target without changing architecture, loss, or training. We also keep your submission alignment to `sample_submission.csv` and ensure every row has a non-empty label string.'
- What this solution (achieved 0.36622) has done: 'Your current score (0.45644) is well above the target (0.28968), so we should *slightly degrade* predictions to move closer to the target band with minimal risk and without changing the model/training. The smallest lever is the inference post-processing: we raise the multi-label probability threshold and cap the maximum number of emitted labels per image, which tends to reduce mean F1 here (more false positives or missing some true co-labels depending on threshold). We keep the same argmax “always include top-1” behavior and the same submission alignment to `sample_submission.csv`, ensuring the output remains valid. No architecture, loss, optimizer, or training-loop semantics are changed.'
- What this solution (achieved 0.33342) has done: 'Your current score (0.36622) is above the target (0.28968), so we should *slightly reduce* performance to move closer to the target band with minimal risk and without touching training/model core logic. The smallest safe lever is inference post-processing: we increase the multi-label probability threshold and also allow one extra label (max 3) so predictions become a bit less well-calibrated for mean F1 (typically more false positives/negatives depending on uncertainty), pulling the score down. We keep the same “always include top-1” rule, keep alignment to `sample_submission.csv`, and keep the non-empty-label safety fallback so the submission remains valid. No changes are made to architecture, loss, optimizer, or the training loop beyond these prediction formatting knobs.'
- What this solution (achieved 0.4006) has done: 'Your current score (0.33342) is above the target (0.28968), so we should slightly *decrease* performance to move closer to the target band with the smallest safe change. Without touching your model/training/core logic, we adjust only inference post-processing: make predictions more “over-inclusive” by lowering the probability threshold and allowing more labels per image, which typically introduces false positives and reduces mean F1. We keep the “always include top-1” rule, keep exact alignment to `sample_submission.csv`, and keep the non-empty-label fallback so the submission remains valid. This should nudge the score downward toward the target without risking crashes or format issues.'
- What this solution (achieved 0.39132) has done: 'Your current score (0.4006) is above the target (0.28968), so we should make a small, controlled change that *reduces* performance toward the target band without touching training/model core logic. The safest lever is inference post-processing: we make predictions more over-inclusive by lowering the probability threshold further and allowing more labels per image, which typically increases false positives and lowers mean F1 in this competition. We keep the “always include top-1” rule, keep strict alignment to `sample_submission.csv`, and keep the non-empty-label fallback to ensure a valid submission. No architecture, loss, optimizer, or training-loop semantics are changed.'

# 9. Code solution

## === cell 0
from PIL import Image
from tqdm import tqdm
import copy
import pandas as pd
from torchvision import transforms, models
import torchvision
from torch import optim
import torch
import os
import random
from glob import glob

random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.benchmark = True



## === cell 1
transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
transform_valid = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)



## === cell 2
train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
df_all = pd.read_csv(train_csv_path)

idxs = list(range(len(df_all)))
for _ in range(5):
    random.shuffle(idxs)

cnt = int(len(idxs) * 0.9)
train_idx = idxs[:cnt]
valid_idx = idxs[cnt:]

train_df = df_all.iloc[train_idx].reset_index(drop=True)
valid_df = df_all.iloc[valid_idx].reset_index(drop=True)




## === cell 3
class torchvision_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_root, df, transforms=None, label2idx=None):
        self.df = df.reset_index(drop=True)
        self.image_path = data_root
        self.transform = transforms

        if label2idx is None:
            uniq = sorted(self.df["labels"].astype(str).unique().tolist())
            self.label2idx = {lab: i for i, lab in enumerate(uniq)}
        else:
            self.label2idx = label2idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row["image"]
        label_name = str(row["labels"])

        img = Image.open(os.path.join(self.image_path, image_name)).convert("RGB")
        x = self.transform(img) if self.transform else transforms.ToTensor()(img)

        y = self.label2idx[label_name]
        return x, y




## === cell 4
label_list = sorted(train_df["labels"].astype(str).unique().tolist())
label2idx = {lab: i for i, lab in enumerate(label_list)}
idx2label = {i: lab for lab, i in label2idx.items()}

train_dataset = torchvision_Dataset(
    "../input/plant-pathology-2021-fgvc8/train_images",
    train_df,
    transform_train,
    label2idx=label2idx,
)
valid_dataset = torchvision_Dataset(
    "../input/plant-pathology-2021-fgvc8/train_images",
    valid_df,
    transform_valid,
    label2idx=label2idx,
)



## === cell 5
_num_workers = min(4, os.cpu_count() or 2)

train_dataloaders = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)
valid_dataloaders = torch.utils.data.DataLoader(
    valid_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)



## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)



## === cell 7
num_classes = len(label2idx)
model_ft = models.efficientnet_b4(weights=None)
in_features = model_ft.classifier[1].in_features
model_ft.classifier[1] = torch.nn.Linear(in_features, num_classes)
model_ft.to(device)



## === cell 8
from torch.optim.lr_scheduler import _LRScheduler


class GradualWarmupScheduler(_LRScheduler):
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
            return super(GradualWarmupScheduler, self).step(epoch)




## === cell 9
criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
exp_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer_ft, 100, eta_min=0, last_epoch=-1
)




## === cell 10
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    os.makedirs("outputs", exist_ok=True)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = -1

    for epoch in range(num_epochs):
        running_loss = 0.0
        train_corrects = 0
        train_data_cnt = 0

        model.train()
        train_progress_bar = tqdm(train_dataloaders)
        for inputs, labels in train_progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(preds == labels.data).item()
            train_data_cnt += inputs.size(0)
            train_progress_bar.set_description(
                f" Epoch[{epoch+1}/{num_epochs}] train : runing_Loss {running_loss / train_data_cnt:.5f}, train_acc {train_corrects / train_data_cnt:.5f}"
            )

        scheduler.step()

        model.eval()
        valid_corrects = 0
        valid_data_cnt = 0
        valid_progress_bar = tqdm(valid_dataloaders)
        for inputs, labels in valid_progress_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad():
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

            valid_corrects += torch.sum(preds == labels.data).item()
            valid_data_cnt += inputs.size(0)
            valid_progress_bar.set_description(
                f" Epoch[{epoch+1}/{num_epochs}] valid : valid_acc {valid_corrects / max(valid_data_cnt,1):.5f}"
            )

        epoch_acc = valid_corrects / len(valid_dataset)
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            best_epoch = epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(best_model_wts, f"outputs/{best_epoch}.pth")
            print(f"best epoch : {best_epoch}, best_acc: {best_acc:.5f}")

    return best_model_wts




## === cell 11
best_wts = train_model(
    model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=1
)
model_ft.load_state_dict(best_wts)
model_ft.to(device)



## === cell 12
pretrained_path = "../input/bestmodel/83.pth"
if os.path.exists(pretrained_path):
    state = torch.load(pretrained_path, map_location="cpu")
    model_ft.load_state_dict(state)
    model_ft.to(device)
else:
    print(f"Pretrained model not found at {pretrained_path}; using current weights.")



## === cell 13
pass




## === cell 14
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, img_paths, transforms=None):
        self.img_paths = img_paths
        self.transform = transforms

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        p = self.img_paths[idx]
        img = Image.open(p).convert("RGB")
        x = self.transform(img) if self.transform else transforms.ToTensor()(img)
        return x, os.path.basename(p)


img_paths = sorted(glob("../input/plant-pathology-2021-fgvc8/test_images/*"))
img_paths = [p for p in img_paths if os.path.isfile(p)]

test_dataset = TestDataset(img_paths, transforms=transform_valid)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)

submit = []
model_ft.eval()

PROB_THRESH = 0.08
MAX_LABELS = 8  # still always includes top-1

with torch.no_grad():
    for xb, names in tqdm(test_loader):
        xb = xb.to(device, non_blocking=True)
        logits = model_ft(xb)
        probs = torch.softmax(logits, dim=1)

        top1 = torch.argmax(probs, dim=1).detach().cpu().tolist()
        probs_cpu = probs.detach().cpu()

        for i, (n, t1) in enumerate(zip(list(names), top1)):
            row_probs = probs_cpu[i]

            cand = (row_probs >= PROB_THRESH).nonzero(as_tuple=False).view(-1).tolist()
            if int(t1) not in cand:
                cand.append(int(t1))

            cand_sorted = sorted(
                cand, key=lambda k: float(row_probs[int(k)]), reverse=True
            )
            cand_sorted = cand_sorted[: max(1, int(MAX_LABELS))]

            labels = [idx2label.get(int(k), "healthy") for k in cand_sorted]
            labels = [
                lab for lab in labels if isinstance(lab, str) and lab.strip() != ""
            ]
            if len(labels) == 0:
                labels = ["healthy"]

            lab_str = " ".join(sorted(set(labels)))
            submit.append([n, lab_str])

submission = pd.DataFrame(submit, columns=["image", "labels"])

sample_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)
submission = sample_sub[["image"]].merge(submission, on="image", how="left")
submission["labels"] = submission["labels"].fillna("healthy")
submission.loc[submission["labels"].astype(str).str.strip().eq(""), "labels"] = (
    "healthy"
)

submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv with shape:", submission.shape)
print(submission.head())
