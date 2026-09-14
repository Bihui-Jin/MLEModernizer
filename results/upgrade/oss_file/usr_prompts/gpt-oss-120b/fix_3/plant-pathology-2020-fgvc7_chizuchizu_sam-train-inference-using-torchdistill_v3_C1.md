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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.91089

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_image_path(image_id: str) -> str:
    """
    Return the absolute path to an image file in the Kaggle dataset.
    """
    base_dirs = [
        "/kaggle/input/plant-pathology-2020-fgvc7/images",
        "./input/plant-pathology-2020-fgvc7/images",
        "./plant-pathology-2020-fgvc7/images",
    ]
    for base in base_dirs:
        cand = os.path.join(base, f"{image_id}.jpg")
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Image {image_id} not found in expected locations.")




## === cell 1
class PlantDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool = True, transform=None):
        """
        df must contain columns: image_id, healthy, multiple_diseases, rust, scab
        If train=False the label columns are ignored.
        """
        self.df = df.reset_index(drop=True)
        self.train = train
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image_id = self.df.loc[idx, "image_id"]
        img_path = _resolve_image_path(image_id)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        if self.train:
            label = self.df.loc[
                idx, ["healthy", "multiple_diseases", "rust", "scab"]
            ].values.astype(np.float32)
            label = torch.from_numpy(label)
            return image, label
        else:
            return image, image_id  # return id for inference




## === cell 2
def get_transforms(train: bool = True):
    if train:
        return T.Compose(
            [
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, targets in tqdm(loader, desc="Training", leave=False):
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(model, loader, device):
    model.eval()
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc="Eval", leave=False):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.numpy())
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    aucs = []
    for i in range(targets.shape[1]):
        try:
            auc = roc_auc_score(targets[:, i], preds[:, i])
        except ValueError:  # single class present
            auc = 0.5
        aucs.append(auc)
    return np.mean(aucs), preds


def inference(model, loader, device):
    model.eval()
    all_preds = []
    image_ids = []
    with torch.no_grad():
        for imgs, ids in tqdm(loader, desc="Inference", leave=False):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            image_ids.extend(ids)
    preds = np.concatenate(all_preds, axis=0)
    return preds, image_ids




## === cell 3
def make_submission_file(preds: np.ndarray, image_ids: list, save_path: str):
    """
    preds shape: (num_test, 4) matching columns:
    healthy, multiple_diseases, rust, scab
    """
    sub_df = pd.DataFrame(
        {
            "image_id": image_ids,
            "healthy": preds[:, 0],
            "multiple_diseases": preds[:, 1],
            "rust": preds[:, 2],
            "scab": preds[:, 3],
        }
    )
    sub_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="submission.csv")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    base_path = "/kaggle/input/plant-pathology-2020-fgvc7"
    train_csv = os.path.join(base_path, "train.csv")
    test_csv = os.path.join(base_path, "test.csv")
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    train_split, val_split = train_test_split(
        train_df, test_size=0.2, random_state=args.seed, stratify=train_df["healthy"]
    )

    train_dataset = PlantDataset(
        train_split, train=True, transform=get_transforms(train=True)
    )
    val_dataset = PlantDataset(
        val_split, train=True, transform=get_transforms(train=False)
    )
    test_dataset = PlantDataset(
        test_df, train=False, transform=get_transforms(train=False)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4)  # 4 disease labels
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_auc, _ = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch}/{args.epochs} - Train loss: {train_loss:.4f} - Val AUC: {val_auc:.4f}"
        )
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), "best_model.pth")
            print("  > New best model saved")

    model.load_state_dict(torch.load("best_model.pth", map_location=device))

    test_preds, test_ids = inference(model, test_loader, device)
    make_submission_file(test_preds, test_ids, args.output)


if __name__ == "__main__":
    main()

## --- ERROR in cell 3, traceback:
An exception has occurred, use %tb to see the full traceback.

SystemExit: 2
