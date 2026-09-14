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

3.14

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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

0.8909035962526443

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'The inference code was failing because it forced the use of CUDA even though the runtime has no GPU. I added a safe device selection that falls back to CPU, guarded the autocast usage to run only on CUDA, and made model loading robust by skipping missing checkpoint files (using a zero‑initializer fallback if no model is loaded). These fixes let the script run end‑to‑end and generate a correctly‑named `submission.csv` without altering the core model architecture or training logic.'
- What this solution (achieved 0.61099) has done: 'We keep the existing inference pipeline but add a safe fallback for when none of the checkpoint files are found. Instead of returning all‑zero predictions, we now read the training labels, compute the most frequent class, and assign that class to every test image. This simple majority‑class baseline is far better than always predicting class 0, moving the validation score much closer to the target while preserving the original model‑ensemble logic when checkpoints are available.'
- What this solution (achieved 0.61099) has done: 'The update avoids the expensive 15‑epoch quick‑training when no pretrained checkpoints are available, falling back to the fast majority‑class baseline, and reuses a single model instance across folds to eliminate repeated model construction. Minor DataLoader tweaks (persistent workers) and enabling CuDNN benchmarking further speed up data loading and GPU kernels while keeping every model architecture, loss, and inference‑logic unchanged. These changes ensure the script finishes well within 600 seconds without affecting prediction semantics.'
- What this solution (achieved 0.61099) has done: 'The quick training loop was the main bottleneck, running 20 epochs over the full dataset. Reducing the number of epochs to a smaller constant (e.g., 5) keeps the same training structure, loss, optimizer, and validation logic while dramatically cutting runtime, ensuring the script finishes well within the 600‑second limit. All other components—including model architecture, data augmentations, inference with TTA, and checkpoint handling—remain unchanged.'

# 9. Code solution

## === cell 0
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm.notebook import tqdm
import timm

torch.backends.cudnn.benchmark = True


class CFG:
    img_size = 384
    batch_size = 128
    num_workers = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
    train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
    train_csv = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
    model_paths = [
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold0.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold1.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold2.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold3.pth",
        "/kaggle/input/cassava-convnext-tiny/pytorch/default/1/best_fold4.pth",
    ]
    temp_checkpoint = "temp_fold0.pth"  # checkpoint produced by quick training




## === cell 1
test_tfms = A.Compose(
    [
        A.Resize(height=CFG.img_size, width=CFG.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

train_tfms = A.Compose(
    [
        A.RandomResizedCrop(size=(CFG.img_size, CFG.img_size), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)




## === cell 2
class TestDataset(Dataset):
    def __init__(self, folder):
        self.paths = sorted([str(p) for p in Path(folder).glob("*.jpg")])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = test_tfms(image=img)["image"]
        return img, os.path.basename(self.paths[idx])


class TrainDataset(Dataset):
    def __init__(self, df, img_folder, transforms):
        self.df = df.reset_index(drop=True)
        self.img_folder = img_folder
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_folder, row["image_id"])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transforms(image=img)["image"]
        label = int(row["label"])
        return img, label




## === cell 3
class CassavaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=True, num_classes=5
        )

    def forward(self, x):
        return self.backbone(x)




## === cell 4
def quick_train():
    print("⚡ Starting quick training because no checkpoints were found.")
    df = pd.read_csv(CFG.train_csv)
    train_ds = TrainDataset(df, CFG.train_dir, train_tfms)

    n_total = len(train_ds)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    train_subset, val_subset = random_split(train_ds, [n_train, n_val])

    train_loader = DataLoader(
        train_subset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    model = CassavaModel().to(CFG.device)

    if hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)

    epochs = 5
    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{epochs} [train]"
        ):
            imgs = imgs.to(CFG.device)
            labels = labels.to(CFG.device)

            optimizer.zero_grad()
            if CFG.device.type == "cuda":
                with autocast(device_type="cuda"):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        scheduler.step()  # update LR after each epoch

        epoch_loss = running_loss / n_train
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in tqdm(
                val_loader, desc=f"Epoch {epoch+1}/{epochs} [val]"
            ):
                imgs = imgs.to(CFG.device)
                labels = labels.to(CFG.device)
                if CFG.device.type == "cuda":
                    with autocast(device_type="cuda"):
                        outputs = model(imgs)
                else:
                    outputs = model(imgs)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total
        print(f"Epoch {epoch+1}: Train loss={epoch_loss:.4f}, Val acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), CFG.temp_checkpoint)

    print(f"Training finished. Best validation accuracy: {best_val_acc:.4f}")
    return best_val_acc




## === cell 5
@torch.no_grad()
def inference():
    dataset = TestDataset(CFG.test_dir)
    loader = DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    ensemble_preds = None
    loaded_folds = 0

    model = CassavaModel().to(CFG.device)

    if hasattr(torch, "compile"):
        model = torch.compile(model)

    model.eval()

    for fold, path in enumerate(CFG.model_paths):
        if not os.path.exists(path):
            print(f"⚠️  Checkpoint not found for fold {fold}: {path} – skipping.")
            continue

        print(f"Loading fold {fold} → {os.path.basename(path)}")
        state = torch.load(path, map_location=CFG.device)
        model.load_state_dict(state)

        fold_preds = []
        for imgs, _ in tqdm(loader, leave=False, desc=f"Fold {fold} TTA"):
            imgs = imgs.to(CFG.device)
            if CFG.device.type == "cuda":
                with autocast(device_type="cuda"):
                    p1 = torch.softmax(model(imgs), dim=1)
                    p2 = torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
            else:
                p1 = torch.softmax(model(imgs), dim=1)
                p2 = torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
            fold_preds.append(((p1 + p2) / 2).cpu().numpy())

        fold_preds = np.concatenate(fold_preds)
        ensemble_preds = (
            fold_preds if ensemble_preds is None else ensemble_preds + fold_preds
        )
        loaded_folds += 1

        del state
        torch.cuda.empty_cache()

    if loaded_folds == 0 and os.path.exists(CFG.temp_checkpoint):
        print("⚡ Loading temporary checkpoint from quick training.")
        state = torch.load(CFG.temp_checkpoint, map_location=CFG.device)
        model.load_state_dict(state)
        model.eval()
        fold_preds = []
        for imgs, _ in tqdm(loader, leave=False, desc="Temp model TTA"):
            imgs = imgs.to(CFG.device)
            if CFG.device.type == "cuda":
                with autocast(device_type="cuda"):
                    p1 = torch.softmax(model(imgs), dim=1)
                    p2 = torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
            else:
                p1 = torch.softmax(model(imgs), dim=1)
                p2 = torch.softmax(model(torch.flip(imgs, dims=[3])), dim=1)
            fold_preds.append(((p1 + p2) / 2).cpu().numpy())
        ensemble_preds = np.concatenate(fold_preds)
        loaded_folds = 1  # treat as a single‑fold ensemble

    if loaded_folds == 0:
        print(
            "⚠️  No model checkpoints loaded – falling back to majority‑class prediction."
        )
        train_df = pd.read_csv(CFG.train_csv)
        majority_label = train_df["label"].mode().iloc[0]
        final_labels = np.full(len(dataset), majority_label, dtype=int)
    else:
        final_labels = np.argmax(ensemble_preds / loaded_folds, axis=1)

    sub = pd.DataFrame(
        {
            "image_id": [os.path.basename(p) for p in dataset.paths],
            "label": final_labels,
        }
    )
    sub = sub.sort_values("image_id").reset_index(drop=True)
    sub.to_csv("submission.csv", index=False)

    print(f"\nSUBMISSION READY → {len(sub)} predictions")
    print(sub.head())
    return sub




## === cell 6
if __name__ == "__main__":
    all_missing = all(not os.path.exists(p) for p in CFG.model_paths)
    if all_missing:
        quick_train()
    else:
        print("✅ Pretrained checkpoints found – skipping quick training.")
    inference()
