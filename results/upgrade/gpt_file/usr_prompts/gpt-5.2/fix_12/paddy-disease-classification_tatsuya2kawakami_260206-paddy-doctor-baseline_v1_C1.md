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

3.14

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
seaborn==0.12.2
sklearn-pandas==2.2.0
timm==1.0.19
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

0.9158986175115208

# 6. Current score

0.56726

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.8163) has done: 'Your code already trains and predicts correctly, but it likely failed to yield a Kaggle score because the notebook “cell 0” header won’t be recognized by the required runner, and `tqdm.notebook` can also break outside notebooks; I renumber cells starting at 1 and switch tqdm to the standard version for robustness so `submission.csv` is always produced. To move accuracy upward toward the target (higher is better) with minimal logic change, I add lightweight train-time augmentation (random crop/flip) while keeping the same ResNet18, loss, optimizer, and loop semantics. I also set seeds + deterministic settings to stabilize results and fix DataLoader settings (`pin_memory`, `persistent_workers`) for reliable GPU throughput within the time limit. Finally, I add a small submission sanity check to ensure row order/count matches `sample_submission.csv` exactly before writing.'
- What this solution (achieved 0.58071) has done: 'To move your accuracy up toward the 0.9159 target without changing the core model/loop/loss, I make minimal training-signal improvements that usually give a sizable gain for this dataset: (1) switch the input preprocessing to the ResNet18 *native* `timm` config (mean/std + crop settings) so pretrained weights are used as intended, and (2) add a standard `RandomVerticalFlip` and `ColorJitter` augmentation (still lightweight and consistent with your pipeline) while keeping the same optimizer, epochs, and model. I also add a very small learning-rate scheduler (`StepLR`) that doesn’t change the training loop structure but typically improves convergence within the same 2 epochs. Finally, I keep your submission creation logic but add a strict ordering check against `sample_submission.csv` so the output aligns exactly.'
- What this solution (achieved 0.58071) has done: 'Your current score (0.58071) is far below the target (0.91590), so we should increase accuracy with minimal, metric-aligned fixes while keeping the same ResNet18 + CrossEntropy + Adam + 2-epoch loop. The biggest likely issue is that your extra augmentation is being applied *after* `timm`’s training transform, which usually converts to tensor/normalizes—so `ColorJitter`/`RandomVerticalFlip` won’t behave correctly and can quietly hurt training. I reorder the augmentations to run on PIL images *before* the `timm` transform, and I also ensure the model’s internal data config matches the one used to build transforms (small but important consistency for pretrained models). Everything else (architecture, optimizer, loss, epochs, loaders, submission format) stays the same.'
- What this solution (achieved 0.78555) has done: 'Your current score (0.58071) is far below the target (0.9159), so we should improve accuracy with the smallest changes that keep your ResNet18 + CrossEntropy + Adam + 2-epoch loop intact. The most likely cause of the low score is that `extra_aug` is being applied before `timm`’s own training transform, but `timm`’s transform already includes tensor conversion/normalization and (often) its own augmentation; composing them directly can lead to transforms being applied in an unintended order and hurting training. I switch to a safe “PIL-augment → timm’s *non-training* preprocess” pipeline so augmentations always run on PIL and normalization is applied exactly once, while keeping everything else the same. I also set `drop_last=True` for the training loader (more stable BatchNorm behavior) and ensure the model uses the same resolved data config that the transforms are built from.'
- What this solution (achieved 0.77902) has done: 'Your current score (0.78555) is well below the target (0.9159), so we should increase accuracy with minimal, metric-aligned changes while keeping your ResNet18 + CrossEntropy + Adam + 2-epoch loop intact. The biggest safe gain without changing core logic is to use the pretrained model’s *training* preprocessing from `timm` (it includes the correct RandomResizedCrop/flip policy and normalization) instead of mixing custom PIL aug with the inference transform, which can lead to suboptimal augmentation and distribution mismatch. I replace the current train transform with `timm`’s `is_training=True` transform (keeping the same model, optimizer, epochs, scheduler, loaders), and keep validation/test on `is_training=False`. I also add AMP autocast/GradScaler (no change to semantics, just stability/speed) so the model can train more effectively within the time budget.'
- What this solution (achieved 0.75711) has done: 'Your current score (0.779) is far below the target (0.9159), so we should cautiously increase accuracy while keeping your ResNet18 + CrossEntropy + Adam + 2-epoch loop intact. The biggest low-risk gain here is improving generalization without changing architecture by using label smoothing in CrossEntropy (same loss family, same semantics) and adding a tiny amount of weight decay to Adam (same optimizer, just regularization). To avoid destabilizing training, I keep your transforms/scheduler/AMP exactly as-is and only touch these two training hyperparameters. I also fix the cell numbering to start at 1 so the runner reliably executes everything end-to-end and always writes `submission.csv`.'
- What this solution (achieved 0.75404) has done: 'Your current score (0.75711) is well below the target (0.91590), so we should make the smallest changes that plausibly improve accuracy without changing your core ResNet18 + Adam + 2-epoch training loop or the timm train/val transforms. The most likely low-risk gains are (1) ensuring BatchNorm updates are stable by using an effective batch size via gradient accumulation (same optimizer/epochs, just fewer optimizer steps), and (2) slightly reducing regularization that may be hurting under a short 2-epoch schedule (lower label smoothing and weight decay). These changes typically improve convergence/generalization on this dataset while preserving your overall approach and keeping runtime within limits. I also renumber the first cell from 0→1 to ensure the runner executes end-to-end reliably and always writes `submission.csv`.'
- What this solution (achieved 0.74673) has done: 'Your current score (0.75404) is far below the target (0.91590), so we should increase accuracy with the smallest changes that keep your ResNet18 + CrossEntropy + Adam + 2-epoch loop intact. The most likely issue hurting pretrained performance is a mismatch between the model’s expected input normalization/crop settings and the transforms you build: you resolve `timm_cfg` from a temporary model, then build a new model without explicitly binding that config to it. I resolve the data config from the *actual* training model, rebuild transforms from that config, and remove the manual `model.pretrained_cfg = timm_cfg` assignment (which can be misleading if it doesn’t match the real model cfg). I also switch CUDNN back to `benchmark=True` (keeping determinism seeds) to improve throughput and slightly improve convergence in the fixed 2-epoch budget, without changing the training logic.'
- What this solution (achieved 0.74289) has done: 'Your score (0.74673) is well below the target (0.9159), so we should increase accuracy with the smallest changes that preserve your ResNet18 + CrossEntropy(label_smoothing) + Adam + 2-epoch loop. The most likely root cause is that the label↔index mapping is built from `sorted(unique_labels)`, which can silently mismatch the model’s internal class ordering expected by the dataset folders and lead to systematically wrong predictions; we instead use the canonical class list from the competition folder names / dataset (“healthy/normal” set) ordering via `train_df['label'].value_counts().index` is not safe either, so we derive it deterministically from `train_df['label'].unique()` in original appearance order and keep it fixed. Additionally, we ensure the `timm` data config and transforms are resolved from the actual instantiated model (not a temporary one), so preprocessing matches the pretrained weights exactly. These are minimal, semantics-preserving fixes that typically produce a large accuracy jump without changing architecture, optimizer, loss family, epochs, or the overall training/prediction approach.'
- What this solution (achieved 0.74596) has done: 'Your current score (0.74289) is far below the target (0.91590), so we should increase accuracy with very small, metric-aligned fixes that don’t change your core model/optimizer/loss/epochs/loop. The biggest likely issue is a potentially inconsistent class-index mapping: using `pd.unique()` can produce an order that doesn’t match the canonical label order used by the competition (and by many strong baselines), which can silently hurt accuracy even if the code runs fine. I replace the label list with the canonical 10 class names (folder names) and map labels strictly through that, while keeping everything else the same. I also ensure the `timm` preprocessing config is resolved from the *actual* instantiated model (not a temporary one) so transforms match the pretrained weights exactly.'
- What this solution (achieved 0.56726) has done: 'Your current score (0.74596) is well below the target (0.91590), so we should make small, safe changes that typically improve accuracy without altering the core ResNet18 + CrossEntropy + Adam + 2-epoch training loop. The biggest low-risk win here is to load the pretrained weights with the correct classifier reset via `timm.create_model(..., pretrained=True)` and then explicitly `reset_classifier(num_classes=10)` (this avoids occasional head-mismatch quirks across timm versions/configs while keeping the same architecture family and loss). Next, we use a slightly more standard fine-tuning learning rate for pretrained ResNet18 on this dataset (`3e-4` instead of `1e-3`) to reduce overshooting in a very short 2-epoch schedule, keeping the optimizer type and epochs unchanged. Finally, we ensure the `timm` data config/transforms are resolved from the *actual* instantiated model (the one we train) so preprocessing matches the weights exactly, which is a minimal consistency fix that often yields a noticeable accuracy lift.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

BASE_DIR = "/kaggle/input/paddy-disease-classification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")

df = pd.read_csv(TRAIN_CSV)

print(f"data size: {len(df)}")
print(df.head())
print(df.info())



## === cell 1
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.countplot(y="label", data=df, order=df["label"].value_counts().index)
plt.title("Distribution of Diseases (Labels)")
plt.xlabel("Count")
plt.ylabel("Disease Name")

plt.subplot(1, 2, 2)
sns.countplot(y="variety", data=df, order=df["variety"].value_counts().index)
plt.title("Distribution of Varieties")
plt.xlabel("Count")
plt.ylabel("Variety Name")

plt.tight_layout()
plt.show()



## === cell 2
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.histplot(df["age"], kde=True, bins=20)
plt.title("Overall Age Distribution of Paddy Crops")
plt.xlabel("Age (days)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()



## === cell 3
from PIL import Image


def get_image_path(row):
    return os.path.join(BASE_DIR, "train_images", row["label"], row["image_id"])


df["path"] = df.apply(get_image_path, axis=1)

unique_labels = df["label"].unique()
plt.figure(figsize=(15, 12))

for i, label in enumerate(unique_labels):
    sample_row = df[df["label"] == label].sample(1, random_state=0).iloc[0]
    img = Image.open(sample_row["path"])
    plt.subplot(3, 4, i + 1)
    plt.imshow(img)
    plt.title(f"{label}\n(Variety: {sample_row['variety']})")
    plt.axis("off")

plt.tight_layout()
plt.show()



## === cell 4
import timm
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)



## === cell 5
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_DIR = "/kaggle/input/paddy-disease-classification"
train_df = pd.read_csv(f"{INPUT_DIR}/train.csv")
sample_sub = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")

train_df["path"] = train_df.apply(
    lambda x: f"{INPUT_DIR}/train_images/{x['label']}/{x['image_id']}", axis=1
)
sample_sub["path"] = sample_sub["image_id"].apply(
    lambda x: f"{INPUT_DIR}/test_images/{x}"
)

CANONICAL_LABELS = [
    "bacterial_leaf_blight",
    "bacterial_leaf_streak",
    "bacterial_panicle_blight",
    "blast",
    "brown_spot",
    "dead_heart",
    "downy_mildew",
    "hispa",
    "normal",
    "tungro",
]
train_labels_set = set(train_df["label"].unique().tolist())
missing = [l for l in CANONICAL_LABELS if l not in train_labels_set]
extra = sorted(list(train_labels_set - set(CANONICAL_LABELS)))
assert len(missing) == 0, f"Missing labels in train.csv vs canonical list: {missing}"
assert len(extra) == 0, f"Unexpected labels in train.csv not in canonical list: {extra}"

labels = CANONICAL_LABELS
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for i, label in enumerate(labels)}
train_df["label_id"] = train_df["label"].map(label2id)

train_data, valid_data = train_test_split(
    train_df, test_size=0.2, stratify=train_df["label"], random_state=42
)

print(f"Train data: {len(train_data)}, Valid data: {len(valid_data)}")
print("Num classes:", len(labels))
print("Labels order:", labels)



## === cell 6
import torch
from torch.utils.data import Dataset
from PIL import Image


class PaddyDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_test:
            return image
        else:
            return image, torch.tensor(row["label_id"], dtype=torch.long)


model_name = "resnet18"

_model_for_cfg = timm.create_model(model_name, pretrained=True)
timm_cfg = timm.data.resolve_model_data_config(_model_for_cfg)

train_transform = timm.data.create_transform(**timm_cfg, is_training=True)
val_transform = timm.data.create_transform(**timm_cfg, is_training=False)

print("Dataset and transform are ready")
print("timm data config:", timm_cfg)



## === cell 7
from torch.utils.data import DataLoader

train_ds = PaddyDataset(train_data, transform=train_transform)
valid_ds = PaddyDataset(valid_data, transform=val_transform)
test_ds = PaddyDataset(sample_sub, transform=val_transform, is_test=True)

num_workers = 2
pin_memory = device.type == "cuda"

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
    drop_last=True,
)
valid_loader = DataLoader(
    valid_ds,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
)
test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=(num_workers > 0),
)

print("DataLoaders are ready.")



## === cell 8
import torch.nn as nn
import torch.optim as optim

model = timm.create_model(model_name, pretrained=True)
model.reset_classifier(num_classes=len(labels))
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=5e-5)
criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

use_amp = device.type == "cuda"
scaler = torch.amp.GradScaler(enabled=use_amp)

print("Model loaded.")



## === cell 9
from tqdm import tqdm
import torch

epochs = 2
accum_steps = 2  # effective batch size ~= 64 while keeping per-step memory like 32

print("Start Training...")
for epoch in range(epochs):
    model.train()
    train_loss = 0.0

    optimizer.zero_grad(set_to_none=True)

    for step, (images, targets) in enumerate(
        tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
    ):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, targets) / accum_steps

        scaler.scale(loss).backward()
        train_loss += loss.item() * accum_steps

        if (step + 1) % accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    if (step + 1) % accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    scheduler.step()

    print(
        f"Epoch {epoch+1} Train Loss: {train_loss / len(train_loader):.4f} | LR: {scheduler.get_last_lr()[0]:.6f}"
    )



## === cell 10
from tqdm import tqdm

model.eval()
correct_counts = 0
total_counts = 0

print("Calculating Validation Accuracy...")
with torch.no_grad():
    for images, y in tqdm(valid_loader, desc="Valid", leave=False):
        images = images.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
        predicted = outputs.argmax(dim=1)

        total_counts += y.size(0)
        correct_counts += (predicted == y).sum().item()

val_acc = correct_counts / total_counts
print(f"\nValidation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")



## === cell 11
from tqdm import tqdm

model.eval()
preds = []

print("Predicting...")
with torch.no_grad():
    for images in tqdm(test_loader, desc="Test", leave=False):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            outputs = model(images)
        predicted = outputs.argmax(dim=1)
        preds.extend(predicted.cpu().numpy().tolist())

assert len(preds) == len(
    sample_sub
), f"Pred length {len(preds)} != sample_sub length {len(sample_sub)}"

sample_sub = sample_sub.copy()
submission = sample_sub[["image_id"]].copy()
submission["label"] = [id2label[i] for i in preds]

assert submission["image_id"].tolist() == sample_sub["image_id"].tolist()

submission.to_csv("submission.csv", index=False)

print("Saved submission.csv!")
print(submission.head())



## === cell 12
import os

print(os.listdir("."))
print("submission.csv exists:", os.path.exists("submission.csv"))
print(
    "submission.csv size (bytes):",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)
