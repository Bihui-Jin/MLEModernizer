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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.12

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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8995164702326987

# 6. Current score

0.77915

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11771) has done: 'I fix the immediate runtime blocker by removing the dependency on missing external model files and replacing it with a small, local baseline model that can run in this Kaggle environment. I also fix submission validity issues by (1) disabling test-time augmentation (your current TTA pipeline returns nested lists that don’t collate correctly), (2) ensuring the test DataLoader is not shuffled, and (3) reindexing predictions to exactly match `sample_submission.csv` order/length. These changes keep the overall inference flow (dataset → loader → model → argmax → CSV) intact while guaranteeing a correctly formatted `submission.csv` is produced end-to-end.'
- What this solution (achieved 0.71973) has done: 'I fix the crash by filtering `os.listdir(test_dir)` to include only actual image files (the Kaggle folder contains a nested `test_images/` directory that your current code tries to open as an image). To move the score toward your target, I keep the same ResNet18 core approach but load it as a proper classifier by fine-tuning the final layer on `train.csv` for a short, deterministic training loop, then run inference on the test set. I also ensure the inference transform matches ResNet’s expected preprocessing by using the official `ResNet18_Weights.DEFAULT.transforms()` (this is a calibration/compatibility fix, not a model change). Finally, I keep the submission alignment logic with `sample_submission.csv` so the output is always valid and ordered correctly.'
- What this solution (achieved 0.76345) has done: 'To move your accuracy up toward the 0.8995 target without changing the core ResNet18 approach, I (1) add a small, stratified train/validation split and pick the best of a few short epochs (same training loop, just a safer checkpoint selection to avoid under/over-fitting), and (2) fine-tune not only `fc` but also `layer4` (minimal extension of the same fine-tuning approach) which typically gives a large gain on Cassava while keeping everything else intact. I also switch the loss to use class weights computed from `train.csv` (still CrossEntropyLoss) to address class imbalance, which commonly boosts accuracy on this dataset. Submission writing stays identical and still aligns exactly to `sample_submission.csv`.'
- What this solution (achieved 0.13117) has done: 'To move your 0.76345 closer to the 0.8995 target without changing the core ResNet18 fine-tuning approach, I make three minimal, high-impact correctness/quality adjustments. First, I switch the training transform from the pure pretrained “eval” preprocessing to a train-time augmentation pipeline that still preserves the same normalization/resize semantics (this commonly yields a sizeable accuracy gain on Cassava while keeping the same model/training loop). Second, I unfreeze `layer3` in addition to `layer4`+`fc` (same fine-tuning logic, just slightly more capacity), and use a small cosine LR schedule (still the same optimizer and epochs, just better LR usage). Third, I ensure determinism is consistent (seeded workers) to reduce run-to-run variance and avoid accidental regressions while still allowing cuDNN benchmarking for speed.'
- What this solution (achieved 0.79634) has done: 'I fix the runtime KeyError by not relying on `weights.meta["mean"]`/`["std"]`, which isn’t present in this torchvision version; instead I reuse the exact normalization values from `ResNet18_Weights.DEFAULT.transforms()` so preprocessing stays consistent with the pretrained backbone. I keep your model, fine-tuning scope (layer3+layer4+fc), optimizer, scheduler, and epoch count unchanged, only adjusting the augmentation pipeline to use the correct mean/std. I also make seeding for DataLoader workers fully deterministic by seeding Python’s `random` and NumPy inside `seed_worker` (score-stable, not a model change). The rest of the pipeline (split, training loop, inference, and submission alignment to `sample_submission.csv`) remains the same and produce a valid `submission.csv`.'
- What this solution (achieved 0.79709) has done: 'I make two minimal, score-relevant adjustments that typically move Cassava accuracy up without changing your core ResNet18 fine-tuning/training loop: (1) switch the validation transform to the same normalization/resize semantics as training but without augmentation (so val selection better matches train/inference preprocessing), and (2) run inference using a deterministic “center-crop eval” transform rather than the raw `weights.transforms()` (which can include a resize+center-crop, but we make it explicit and consistent with your 224 pipeline). This keeps the model architecture, unfreeze scope (layer3+layer4+fc), optimizer, scheduler, epochs, and loss intact, while reducing train/val/infer preprocessing mismatch that can hurt both checkpoint selection and test performance. Submission writing/alignment stays identical and still produces a valid `submission.csv`.'
- What this solution (achieved 0.77915) has done: 'I make two minimal, score-relevant adjustments that typically improve Cassava accuracy without changing your core ResNet18 fine-tuning/training loop. First, I add a tiny amount of label smoothing to `CrossEntropyLoss` (same loss family and semantics, just better calibration/regularization), which often yields a small but reliable gain when fine-tuning pretrained backbones. Second, I add a standard RandomErasing augmentation after normalization in the training transform (no change to model/optimizer/epochs), which usually improves generalization on leaf datasets. Everything else (data split, unfreeze scope, optimizer, scheduler, epochs, inference transform, and submission alignment) stays the same and still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

batch_size = 32
num_workers = 4
num_classes = 5
tta = False  # unchanged

from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

resnet_eval_preprocess = weights.transforms()

_mean = getattr(resnet_eval_preprocess, "mean", None)
_std = getattr(resnet_eval_preprocess, "std", None)
if _mean is None or _std is None:
    _mean = (0.485, 0.456, 0.406)
    _std = (0.229, 0.224, 0.225)

eval_transform = v2.Compose(
    [
        v2.Resize(256, interpolation=v2.InterpolationMode.BILINEAR),
        v2.CenterCrop((224, 224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_mean, std=_std),
    ]
)

normalizer = torch.nn.Softmax(dim=1)


def seed_worker(worker_id: int):
    worker_seed = 3407 + worker_id
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)




## === cell 1
class CassavaTestDataset(VisionDataset):
    """Custom dataset for Cassava test images (no labels)."""

    def __init__(self, data_dir, transform=None):
        super().__init__(root=data_dir)
        self.transform = transform

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        entries = os.listdir(data_dir)
        files = []
        for f in entries:
            p = os.path.join(data_dir, f)
            if os.path.isfile(p) and f.lower().endswith(exts):
                files.append(f)
        self.images = sorted(files)  # deterministic ordering

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.images)


class CassavaTrainDataset(VisionDataset):
    """Custom dataset for Cassava train images with labels from train.csv."""

    def __init__(self, df, data_dir, transform=None):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_id"]
        label = int(row["label"])
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.df)




## === cell 2
train_df = pd.read_csv(train_csv_path)

g_split = torch.Generator().manual_seed(3407)
perm = torch.randperm(len(train_df), generator=g_split).tolist()
train_df_shuf = train_df.iloc[perm].reset_index(drop=True)

val_frac = 0.1
val_parts = []
train_parts = []
for c in range(num_classes):
    df_c = train_df_shuf[train_df_shuf["label"] == c]
    n_c = len(df_c)
    v_c = int(round(n_c * val_frac))
    val_parts.append(df_c.iloc[:v_c])
    train_parts.append(df_c.iloc[v_c:])

val_df = (
    pd.concat(val_parts, axis=0)
    .sample(frac=1.0, random_state=3407)
    .reset_index(drop=True)
)
tr_df = (
    pd.concat(train_parts, axis=0)
    .sample(frac=1.0, random_state=3407)
    .reset_index(drop=True)
)

train_transform = v2.Compose(
    [
        v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_mean, std=_std),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.0),
    ]
)

val_transform = eval_transform

train_dataset = CassavaTrainDataset(tr_df, train_dir, transform=train_transform)
val_dataset = CassavaTrainDataset(val_df, train_dir, transform=val_transform)

g_loader = torch.Generator().manual_seed(3407)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    worker_init_fn=seed_worker,
    generator=g_loader,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    worker_init_fn=seed_worker,
)

label_counts = (
    tr_df["label"].value_counts().reindex(range(num_classes), fill_value=0).values
)
label_counts = torch.tensor(label_counts, dtype=torch.float32)
class_weights = (label_counts.sum() / torch.clamp(label_counts, min=1.0)).to(device)
class_weights = class_weights / class_weights.mean()

for p in model.parameters():
    p.requires_grad = False
for p in model.layer3.parameters():
    p.requires_grad = True
for p in model.layer4.parameters():
    p.requires_grad = True
for p in model.fc.parameters():
    p.requires_grad = True

criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

optimizer = torch.optim.AdamW(
    list(model.layer3.parameters())
    + list(model.layer4.parameters())
    + list(model.fc.parameters()),
    lr=2e-4,
    weight_decay=1e-2,
)

epochs = 4
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def evaluate_accuracy(m, loader):
    m.eval()
    correct = 0
    seen = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = m(imgs)
            pred = logits.argmax(1)
            correct += (pred == labels).sum().item()
            seen += labels.size(0)
    return correct / max(seen, 1)


best_val = -1.0
best_state = None

for epoch in range(epochs):
    running_loss = 0.0
    seen = 0
    correct = 0

    model.train()
    for imgs, labels in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        seen += labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()

    scheduler.step()

    train_acc = correct / max(seen, 1)
    val_acc = evaluate_accuracy(model, val_loader)

    if val_acc > best_val:
        best_val = val_acc
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    print(
        f"epoch {epoch+1}/{epochs} - loss {running_loss/seen:.4f} - train_acc {train_acc:.4f} - val_acc {val_acc:.4f} - best_val {best_val:.4f}"
    )

if best_state is not None:
    model.load_state_dict(best_state)
model.eval()




## === cell 3
test_dataset = CassavaTestDataset(test_dir, transform=eval_transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    worker_init_fn=seed_worker,
)

all_names = []
all_preds = []

model.eval()
with torch.no_grad():
    for inputs, filenames in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        probs = normalizer(logits)
        pred_labels = torch.argmax(probs, dim=1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

print("preds:", len(all_preds), "names:", len(all_names))




## === cell 4
sample_sub = pd.read_csv(sample_sub_path)
pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

merged = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
if merged["label"].isna().any():
    fill_label = int(pred_df["label"].mode().iloc[0]) if len(pred_df) else 0
    merged["label"] = merged["label"].fillna(fill_label).astype(int)
else:
    merged["label"] = merged["label"].astype(int)

merged.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", merged.shape)
merged.head()
