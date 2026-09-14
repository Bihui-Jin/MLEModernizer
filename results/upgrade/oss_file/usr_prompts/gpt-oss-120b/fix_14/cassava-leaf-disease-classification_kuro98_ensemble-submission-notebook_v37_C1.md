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

0.8974010275007556

# 6. Current score

0.78064

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.13602) has done: 'The fix updates the Vision Transformer model initialization to use the default image size (224) by removing the unsupported `image_size` argument and adjusting the corresponding image size variables. This resolves the `ValueError` during model creation and ensures the models are defined before inference, allowing the script to run end‑to‑end and generate a valid `submission.csv`.'
- What this solution (achieved 0.787) has done: 'The update adds a short fine‑tuning stage for the Vision Transformer (model a) on the provided training CSV, using a modest number of epochs and basic augmentations. After training, the model is set back to eval mode and the original ensemble inference runs unchanged, so the core architecture and inference logic are preserved. This training step should raise the validation‑style accuracy from the initial ≈0.13 toward the target 0.897 while still producing a correct `submission.csv`.'
- What this solution (achieved 0.8068) has done: 'The changes keep the exact model architectures, training schedule, and inference logic, but speed up data loading by reading images directly as tensors, increase the batch size, use more workers, enable persistent workers, and apply mixed‑precision (AMP) with a gradient scaler for training and autocast for inference. These optimizations reduce CPU‑GPU transfer and computation time without altering any algorithmic steps or result accuracy.'
- What this solution (achieved 0.80755) has done: 'Implemented fixes to resolve data loading errors and improve model performance:
- Added PIL import and switched image loading to `Image.open`, ensuring transforms receive proper PIL images.
- Adjusted the dataset pipeline to remove unnecessary tensor permutations.
- Updated `num_epochs` from 8 to 12 for slightly better training without altering core logic.
- Minor cleanup of comments.'
- What this solution (achieved 0.78064) has done: 'Implemented mixed‑precision training, increased batch size, and enabled persistent DataLoader workers to cut overhead while keeping all model architectures, epoch counts, and augmentation logic intact. These tweaks accelerate GPU compute and data loading without altering algorithmic behavior or final predictions.'

# 9. Code solution

## === cell 0
import os
import json
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler  # <-- added for mixed‑precision

import torchvision
import torchvision.transforms as T
import torchvision.models as models

from PIL import Image  # added for proper image loading

torch.manual_seed(3407)
torch.cuda.manual_seed_all(3407)
cudnn.deterministic = False
cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)




## === cell 1
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

model_a_img_size = 224
model_b_img_size = 528
model_c_img_size = 224
batch_size = 64  # <-- increased to reduce iteration overhead
num_workers = 8  # <-- more workers for faster loading
num_classes = 5
tta = True
num_epochs = 12  # increased modestly for better accuracy
lr = 1e-4




## === cell 2
model_a = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1).to(device)
model_a_head_in = (
    model_a.heads[0].in_features
    if isinstance(model_a.heads, torch.nn.Sequential)
    else model_a.heads.in_features
)
model_a.heads = torch.nn.Linear(model_a_head_in, num_classes).to(device)
model_a = torch.compile(model_a, mode="reduce-overhead")

model_b = models.efficientnet_b0(
    weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
).to(device)
model_b.classifier[1] = torch.nn.Linear(
    model_b.classifier[1].in_features, num_classes
).to(device)
model_b = torch.compile(model_b, mode="reduce-overhead")

model_c = models.vit_l_16(weights=models.ViT_L_16_Weights.IMAGENET1K_V1).to(device)
model_c_head_in = (
    model_c.heads[0].in_features
    if isinstance(model_c.heads, torch.nn.Sequential)
    else model_c.heads.in_features
)
model_c.heads = torch.nn.Linear(model_c_head_in, num_classes).to(device)
model_c = torch.compile(model_c, mode="reduce-overhead")




## === cell 3
class CassavaDataset(Dataset):
    def __init__(self, df, img_dir, img_size, training=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.training = training
        if training:
            self.transform = T.Compose(
                [
                    T.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                    ),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        else:
            self.transform = T.Compose(
                [
                    T.Resize(int(img_size * 1.14)),
                    T.CenterCrop(img_size),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image_id"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        if self.training:
            label = int(row["label"])
            return image, label
        else:
            return image, row["image_id"]




## === cell 4
def train_one_model(model, img_size):
    scaler = GradScaler()
    df = pd.read_csv(train_csv_path)

    train_len = int(0.9 * len(df))
    val_len = len(df) - train_len
    train_subset, val_subset = torch.utils.data.random_split(
        df, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )
    train_df = df.iloc[train_subset.indices].reset_index(drop=True)
    val_df = df.iloc[val_subset.indices].reset_index(drop=True)

    train_dataset = CassavaDataset(train_df, train_dir, img_size, training=True)

    val_dataset = CassavaDataset(val_df, train_dir, img_size, training=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,  # <-- keep workers alive across epochs
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for imgs, targets in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False
        ):
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * imgs.size(0)

        scheduler.step()
        epoch_loss = running_loss / len(train_loader.dataset)

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                with autocast():
                    outputs = model(imgs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1}: TrainLoss={epoch_loss:.4f} ValAcc={val_acc:.4f}")

    model.eval()
    return model




## === cell 5
print("Training model_a (ViT‑B)…")
model_a = train_one_model(model_a, model_a_img_size)

print("\nTraining model_b (EfficientNet‑B0)…")
model_b = train_one_model(model_b, model_b_img_size)

print("\nTraining model_c (ViT‑L)…")
model_c = train_one_model(model_c, model_c_img_size)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2489219756.py in <cell line: 0>()
      1 print("Training model_a (ViT‑B)…")
----> 2 model_a = train_one_model(model_a, model_a_img_size)
      3 
      4 print("\nTraining model_b (EfficientNet‑B0)…")
      5 model_b = train_one_model(model_b, model_b_img_size)

/tmp/ipykernel_55/1523452289.py in train_one_model(model, img_size)
     70             for imgs, targets in val_loader:
     71                 imgs = imgs.to(device, non_blocking=True)
---> 72                 targets = targets.to(device, non_blocking=True)
     73                 with autocast():
     74                     outputs = model(imgs)

AttributeError: 'list' object has no attribute 'to'

## === cell 6
def get_predictions(model, img_size):
    test_df = pd.read_csv(sample_sub_path)[["image_id"]]
    test_dataset = CassavaDataset(test_df, test_dir, img_size, training=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    all_probs = []
    all_ids = []
    model.eval()
    with torch.no_grad():
        for imgs, ids in tqdm(test_loader, desc="Inference"):
            imgs = imgs.to(device, non_blocking=True)
            with autocast():
                logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu())
            all_ids.extend(ids)
    all_probs = torch.cat(all_probs, dim=0).numpy()
    return all_ids, all_probs


print("Running inference for each model...")
ids_a, probs_a = get_predictions(model_a, model_a_img_size)
ids_b, probs_b = get_predictions(model_b, model_b_img_size)
ids_c, probs_c = get_predictions(model_c, model_c_img_size)

assert ids_a == ids_b == ids_c

ensemble_probs = (probs_a + probs_b + probs_c) / 3.0
pred_labels = ensemble_probs.argmax(axis=1)

submission = pd.read_csv(sample_sub_path)
submission["label"] = pred_labels
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print(f"Submission written to {out_path}")
