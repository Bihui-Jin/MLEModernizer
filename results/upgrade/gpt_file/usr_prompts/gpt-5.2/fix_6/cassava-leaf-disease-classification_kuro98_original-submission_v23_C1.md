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

0.8647627682079178

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime blocker by removing the missing external model dependency and replacing it with a torchvision ViT (same “vision transformer” core idea) so inference can run end-to-end in this Kaggle environment. I also fix the submission-length error by (1) turning off `shuffle` for the test loader, (2) ensuring each test image appears exactly once, and (3) aligning predictions to `sample_submission.csv` ordering. Finally, I make TTA deterministic and correctly structured (include the identity transform and use fixed seeds per batch) so the averaging logic is stable and doesn’t silently degrade or duplicate rows.'
- What this solution (achieved 0.12668) has done: 'You’re hitting a hard ViT input-size assertion because `vit_b_16` (default weights) expects 224×224 images, but the pipeline resizes to 384×384. I fix this by switching to the 384-sized ViT variant while keeping the same “torchvision ViT + single linear head + TTA averaging” core logic. I also make the TTA split/reshape robust by using the batch size rather than `len(filenames)` when reconstructing `[n_tta, B, C]`. These changes should both unblock end-to-end execution and substantially improve accuracy versus the current broken-size setup.'
- What this solution (achieved 0.70628) has done: 'Your current score is far below target, so we should improve accuracy with the smallest changes that don’t alter the overall approach (torchvision ViT + linear head + TTA averaging). The biggest issue is that the model head is randomly initialized and you never load cassava-trained weights, so predictions are near-random; the minimal fix is to add a quick fine-tuning step on `train.csv` using the same model and loss (cross-entropy). To keep the core logic intact and runtime under control, we do a single short epoch with a simple train/val split and then run the exact same inference/TTA + submission alignment as you already have. This should move accuracy substantially toward the target without changing architecture, loss, or inference semantics.'
- What this solution (achieved 0.66816) has done: 'To move your accuracy up toward the 0.8648 target without changing the ViT+linear-head + CE-loss core approach, the main fix is to actually train for a small, bounded amount of time (your current loop only runs a partial epoch). I keep the same architecture, transforms, TTA, and inference semantics, but add: (1) a proper multi-epoch fine-tune loop with a step budget so it always finishes under the time limit, (2) basic validation monitoring (no early stopping) so we don’t overshoot into instability, and (3) label smoothing in CrossEntropyLoss (same loss family) to improve generalization with minimal risk. Everything still writes `submission.csv` in the required format and preserves the sample_submission ordering.'
- What this solution (achieved 0.05531) has done: 'Your current score (0.66816) is well below the target (0.86476), so we should improve accuracy with the smallest changes that preserve your ViT+linear-head, CE-loss fine-tuning, and TTA inference. The biggest low-risk gains here are (1) using the correct pretrained normalization pipeline for the specific ViT weights you loaded (rather than hardcoded ImageNet mean/std), and (2) fine-tuning more effectively by freezing the backbone and training only the classification head for the same step budget (this typically improves stability and generalization without changing the architecture). I also switch the validation split to a stratified split (same holdout idea, less class-imbalance noise) to keep training behavior stable and reduce over/underfitting risk. Everything else (model choice, loss family, TTA averaging, submission alignment) is kept the same and still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import random
import time

import pandas as pd
import torch
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2
from torchvision import models



## === cell 1
SEED = 3407
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)

cudnn.deterministic = True
cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_root = "/kaggle/input/cassava-leaf-disease-classification/"
train_dir = os.path.join(data_root, "train_images/")
test_dir = os.path.join(data_root, "test_images/")

img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5
tta = True

weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
model = models.vit_b_16(weights=weights)
model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
model.to(device)




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava (train: returns image + label; test: returns image + filename)."""

    def __init__(self, data_dir, df=None, transform=None, ttas=None, is_train=False):
        super().__init__(root=data_dir)
        self.transform = transform
        self.ttas = ttas
        self.is_train = is_train
        self.cc = v2.CenterCrop((400, 400))

        if df is None:
            self.images = sorted(
                [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
            )
            self.labels = None
        else:
            self.images = df["image_id"].tolist()
            self.labels = df["label"].tolist() if "label" in df.columns else None

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)

        if self.ttas is not None and self.transform is not None:
            img = [self.transform(t(img)) for t in self.ttas]
        elif self.transform is not None:
            img = self.transform(img)

        if self.labels is None:
            return img, filename
        else:
            return img, int(self.labels[idx])

    def __len__(self):
        return len(self.images)




## === cell 3
w_meta = weights.meta
mean = w_meta["mean"]
std = w_meta["std"]
interp = w_meta.get("interpolation", InterpolationMode.BICUBIC)

test_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=interp),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ]
)

if tta:
    ttas = [
        v2.Identity(),
        v2.RandomHorizontalFlip(1.0),
        v2.RandomRotation(15),
    ]
else:
    ttas = None



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/3386877577.py in <cell line: 0>()
      2 # This is a minimal change that often yields a noticeable accuracy jump vs hardcoded mean/std.
      3 w_meta = weights.meta
----> 4 mean = w_meta["mean"]
      5 std = w_meta["std"]
      6 interp = w_meta.get("interpolation", InterpolationMode.BICUBIC)

KeyError: 'mean'

## === cell 4
train_csv_path = os.path.join(data_root, "train.csv")
train_df = pd.read_csv(train_csv_path)

g = torch.Generator().manual_seed(SEED)
labels = train_df["label"].astype(int).values
idx_by_class = {}
for i, y in enumerate(labels):
    idx_by_class.setdefault(int(y), []).append(i)

val_frac = 0.1
val_indices = []
trn_indices = []
for c, idxs in idx_by_class.items():
    idxs_t = torch.tensor(idxs)
    perm = idxs_t[torch.randperm(len(idxs_t), generator=g)].tolist()
    n_val_c = max(1, int(round(len(perm) * val_frac)))
    val_indices.extend(perm[:n_val_c])
    trn_indices.extend(perm[n_val_c:])

val_df = train_df.iloc[val_indices].reset_index(drop=True)
trn_df = train_df.iloc[trn_indices].reset_index(drop=True)

train_dataset = CassavaDataset(
    train_dir, df=trn_df, transform=test_transforms, ttas=None, is_train=True
)
val_dataset = CassavaDataset(
    train_dir, df=val_df, transform=test_transforms, ttas=None, is_train=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

for p in model.parameters():
    p.requires_grad = False
for p in model.heads.head.parameters():
    p.requires_grad = True

optimizer = torch.optim.AdamW(model.heads.head.parameters(), lr=5e-4, weight_decay=1e-2)

max_epochs = 2
max_train_steps = 900  # keep same runtime budget intent


def eval_acc(m, loader):
    m.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device)
            pred = torch.argmax(m(x), dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


start_time = time.time()
global_step = 0
best_val = -1.0

for epoch in range(max_epochs):
    model.train()
    for batch_idx, (x, y) in enumerate(train_loader):
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        global_step += 1
        if global_step >= max_train_steps:
            break

    val_acc = eval_acc(model, val_loader)
    best_val = max(best_val, val_acc)
    print(
        f"Epoch {epoch+1}/{max_epochs} - holdout val accuracy (sanity check): {val_acc:.4f}"
    )
    if global_step >= max_train_steps:
        break

print("Best holdout val accuracy (sanity check):", best_val)
print("Train steps:", global_step, "Elapsed(s):", round(time.time() - start_time, 1))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1325988304.py in <cell line: 0>()
     24 
     25 train_dataset = CassavaDataset(
---> 26     train_dir, df=trn_df, transform=test_transforms, ttas=None, is_train=True
     27 )
     28 val_dataset = CassavaDataset(

NameError: name 'test_transforms' is not defined

## === cell 5
test_dataset = CassavaDataset(
    test_dir, transform=test_transforms, ttas=ttas, df=None, is_train=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

normalizer = torch.nn.Softmax(dim=1)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1859997071.py in <cell line: 0>()
      1 test_dataset = CassavaDataset(
----> 2     test_dir, transform=test_transforms, ttas=ttas, df=None, is_train=False
      3 )
      4 
      5 test_loader = DataLoader(

NameError: name 'test_transforms' is not defined

## === cell 6
all_names = []
all_preds = []

model.eval()

with torch.no_grad():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        torch.manual_seed(SEED + batch_idx)
        torch.cuda.manual_seed(SEED + batch_idx)

        filenames = list(filenames)
        B = len(filenames)

        if tta:
            inputs = torch.cat(inputs, dim=0).to(
                device, non_blocking=True
            )  # [B*n_tta, C, H, W]
            preds = normalizer(model(inputs))  # [B*n_tta, num_classes]

            n_tta = len(ttas)
            preds = preds.view(n_tta, B, num_classes)  # [n_tta, B, C]
            mean_preds = preds.mean(dim=0)  # [B, C]
            pred_labels = torch.argmax(mean_preds, dim=1).tolist()
        else:
            inputs = inputs.to(device, non_blocking=True)
            preds = normalizer(model(inputs))
            pred_labels = torch.argmax(preds, dim=1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)

assert len(all_names) == len(test_dataset), (len(all_names), len(test_dataset))
assert len(all_preds) == len(test_dataset), (len(all_preds), len(test_dataset))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3903909378.py in <cell line: 0>()
      5 
      6 with torch.no_grad():
----> 7     for batch_idx, (inputs, filenames) in enumerate(test_loader):
      8         torch.manual_seed(SEED + batch_idx)
      9         torch.cuda.manual_seed(SEED + batch_idx)

NameError: name 'test_loader' is not defined

## === cell 7
sample_path = os.path.join(data_root, "sample_submission.csv")
sample_sub = pd.read_csv(sample_path)

pred_map = dict(zip(all_names, all_preds))
my_submission = sample_sub.copy()
my_submission["label"] = my_submission["image_id"].map(pred_map)

my_submission["label"] = my_submission["label"].fillna(0).astype(int)

assert my_submission.shape[0] == sample_sub.shape[0], (
    my_submission.shape,
    sample_sub.shape,
)
assert list(my_submission.columns) == ["image_id", "label"]

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)



## === cell 8
my_submission.head(10)
