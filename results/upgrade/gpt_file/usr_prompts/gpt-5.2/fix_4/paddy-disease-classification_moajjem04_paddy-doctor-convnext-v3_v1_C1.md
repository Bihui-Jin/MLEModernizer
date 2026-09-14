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

3.10

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
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

0.5933179723502304

# 6. Current score

0.96311

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.20984) has done: 'I replace the missing external train/val folder and missing pretrained `.pt` weight dependency with paths that exist in this Kaggle dataset, while keeping your core training loop, loss, and ConvNeXt-Tiny architecture intact. Concretely: build train/val splits directly from `train.csv` + `train_images` using a small custom `Dataset`, and initialize `convnext_tiny` from `torchvision` (no extra downloads) so training can run. I also fix the TorchMetrics `Accuracy` usage by specifying the multiclass setup, ensure inference runs with `model.eval()`/`torch.no_grad()`, and generate `submission.csv` with exactly the same row order as `sample_submission.csv` to avoid the “0 rows”/misalignment issue. These are execution-blocking fixes and should yield a reasonable accuracy toward your target with 1 epoch.'
- What this solution (achieved 0.50615) has done: 'Your current score is far below the target, and the biggest limiter is that you’re training ConvNeXt-Tiny from scratch for only 1 epoch and also feeding a “normalized-logit” tensor into CrossEntropyLoss (which makes optimization much harder). To move accuracy upward with minimal semantic change, I (1) switch to torchvision’s built-in ImageNet pretrained ConvNeXt-Tiny weights (available offline in Kaggle images) while keeping the same architecture and classifier head, (2) train a few more epochs (still the same loop/objective, just longer), and (3) compute the loss on the actual logits (while keeping your logit normalization for inference/metrics if you want—here we keep it only for loss removal, as the competition metric is accuracy on argmax of logits). I also make deterministic settings consistent (don’t set deterministic=True while benchmark=True) for stability, and keep the submission generation identical and aligned to `sample_submission.csv`.'
- What this solution (achieved 0.96311) has done: 'You’re currently below the target, so the smallest safe way to move accuracy upward is to (1) ensure the ConvNeXt-Tiny head is trained with a stable learning rate schedule that better fits pretrained fine-tuning (lower LR + cosine decay) without changing the model or loss, and (2) add standard ImageNet normalization-consistent resizing/cropping for val/test (center-crop after resize) to reduce train/val/test transform mismatch. I also switch inference from single-image looping to a DataLoader batch loop to reduce overhead and keep the run reliably within the 600s limit, while producing the exact same `submission.csv` schema/order. These are minimal changes that typically yield a modest accuracy lift toward your target without altering the core training loop semantics. All paths and the submission format remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
from tqdm.notebook import tqdm
from torchmetrics import Accuracy
import pandas as pd
from PIL import Image




## === cell 1
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)



## === cell 2
Prob = 0.5
train_tf = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(Prob),
        transforms.RandomVerticalFlip(Prob),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
val_tf = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)



## === cell 3
DATA_DIR = "../input/paddy-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

labels = sorted(train_df["label"].unique().tolist())
label2idx = {l: i for i, l in enumerate(labels)}
idx2label = {i: l for l, i in label2idx.items()}
num_classes = len(labels)


def _make_image_path(image_id: str, label: str) -> str:
    return os.path.join(TRAIN_IMG_DIR, label, image_id)


class PaddyDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = _make_image_path(row["image_id"], row["label"])
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        target = label2idx[row["label"]]
        return img, target


rng = np.random.RandomState(42)
train_idx = []
val_idx = []
val_frac = 0.2
for l in labels:
    idxs = train_df.index[train_df["label"] == l].to_numpy()
    rng.shuffle(idxs)
    n_val = max(1, int(len(idxs) * val_frac))
    val_idx.extend(idxs[:n_val].tolist())
    train_idx.extend(idxs[n_val:].tolist())

train_split_df = train_df.loc[train_idx].reset_index(drop=True)
val_split_df = train_df.loc[val_idx].reset_index(drop=True)

train_ds = PaddyDataset(train_split_df, transform=train_tf)
val_ds = PaddyDataset(val_split_df, transform=val_tf)

print("num_classes:", num_classes)
print("train size:", len(train_ds), "val size:", len(val_ds))



## === cell 4
class_to_idx = label2idx
class_to_idx



## === cell 5
class_dict = {v: k for k, v in class_to_idx.items()}
class_dict



## === cell 6
train_loader = DataLoader(
    train_ds, batch_size=64, shuffle=True, pin_memory=True, num_workers=2
)
val_loader = DataLoader(
    val_ds, batch_size=64, shuffle=False, pin_memory=True, num_workers=2
)



## === cell 7
weights = torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT
model = torchvision.models.convnext_tiny(weights=weights)



## === cell 8
model.classifier[-1] = torch.nn.Linear(in_features=768, out_features=num_classes)




## === cell 9
def train(data, target, model, optimizer, criterion, TRAIN):
    if TRAIN:
        optimizer.zero_grad()

    output = model(data)

    loss = criterion(output, target)

    if TRAIN:
        loss.backward()
        optimizer.step()

    return output, loss




## === cell 10
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

n_epochs = 6

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

acc = Accuracy(task="multiclass", num_classes=num_classes).to(device)

print(f"\nTraining Model on device={device}")
best_val_ = 1000
failure_count = 0

for epoch in tqdm(range(n_epochs), desc="# Epochs", position=0):
    train_loss = 0.0
    val_loss = 0.0

    model.train()
    for i, (data, target) in enumerate(
        tqdm(train_loader, desc="Training", leave=True, position=1)
    ):
        data = data.to(device, non_blocking=True)
        target = torch.as_tensor(target, device=device)

        output, loss = train(data, target, model, optimizer, criterion, TRAIN=True)

        train_loss += loss.item() * data.size(0)
        preds = torch.argmax(output, dim=1)
        acc(preds, target)

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = acc.compute().item()
    acc.reset()

    with torch.no_grad():
        model.eval()
        for i, (data, target) in enumerate(
            tqdm(val_loader, desc="Validation", leave=True, position=2)
        ):
            data = data.to(device, non_blocking=True)
            target = torch.as_tensor(target, device=device)

            output, loss = train(data, target, model, optimizer, criterion, TRAIN=False)

            val_loss += loss.item() * data.size(0)
            preds = torch.argmax(output, dim=1)
            acc(preds, target)

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = acc.compute().item()
        acc.reset()

    scheduler.step()

    if val_loss < best_val_:
        best_val_ = val_loss
        best_acc = val_acc
        failure_count = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        failure_count += 1

    if failure_count >= 10:
        break

    print(f"Epoch # {epoch+1:04d} (lr={scheduler.get_last_lr()[0]:.6g})")
    print(f"Train Loss: {train_loss: .4f},\t Val Loss: {val_loss: .4f}")
    print(f"Train Acc : {train_acc: .4f},\t Val Acc : {val_acc: .4f}")
    print(f"Best Val Loss : {best_val_: .4f},\t Best Val Acc : {best_acc: .4f}")
    print(f"Failure Count = {failure_count}")



## === cell 11
best_model = torchvision.models.convnext_tiny(weights=weights)
best_model.classifier[-1] = torch.nn.Linear(in_features=768, out_features=num_classes)
best_model.load_state_dict(torch.load("best_model.pt", map_location=device))
best_model.to(device)
best_model.eval()



## === cell 12
test_tfs = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)




## === cell 13
class TestDataset(Dataset):
    def __init__(self, image_ids, img_dir, transform=None):
        self.image_ids = list(image_ids)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, image_id)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, image_id


result_df = sample_df.copy()

test_ds = TestDataset(result_df["image_id"].tolist(), TEST_IMG_DIR, transform=test_tfs)
test_loader = DataLoader(
    test_ds, batch_size=64, shuffle=False, pin_memory=True, num_workers=2
)

print("Begin Inference\n")
pred_map = {}

with torch.no_grad():
    for xb, image_ids in tqdm(test_loader, desc="Test Inference"):
        xb = xb.to(device, non_blocking=True)
        out = best_model(xb)
        preds = torch.argmax(out, dim=1).detach().cpu().numpy().tolist()
        for img_id, p in zip(image_ids, preds):
            pred_map[img_id] = class_dict[int(p)]

result_df["label"] = result_df["image_id"].map(pred_map)
result_df.head()



## === cell 14
result_df.to_csv("submission.csv", index=False)
print("\nSubmission File Created! -> submission.csv")
print("rows:", len(result_df), "cols:", list(result_df.columns))
