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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import copy  # needed to store the best model state
from sklearn.metrics import roc_auc_score  # for validation AUC




## === cell 1
BASE_PATH = os.path.abspath(
    os.path.join(os.getcwd(), "..", "input", "plant-pathology-2020-fgvc7")
)
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUBMISSION = os.path.join(BASE_PATH, "sample_submission.csv")
SUBMISSION_PATH = "submission.csv"
IMAGE_DIR = os.path.join(BASE_PATH, "images")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_EPOCHS = 20  # a bit longer to let the scheduler act
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_cols = ["healthy", "multiple_diseases", "rust", "scab"]  # used throughout




## === cell 2
def load_image(image_id):
    img_path = os.path.join(IMAGE_DIR, f"{image_id}.jpg")
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = np.array(img).astype(np.float32) / 255.0  # scale to [0,1]
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # C, H, W
    return img_tensor


class PlantDataset(Dataset):
    def __init__(self, df, is_train=True):
        self.df = df.reset_index(drop=True)
        self.is_train = is_train
        self.labels = ["healthy", "multiple_diseases", "rust", "scab"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        img = load_image(image_id)
        if self.is_train:
            if np.random.rand() < 0.5:  # random horizontal flip
                img = torch.flip(img, dims=[2])  # flip width dimension
            if np.random.rand() < 0.3:  # random 90‑degree rotation
                k = np.random.choice([1, 2, 3])  # number of 90° turns
                img = torch.rot90(img, k=k, dims=[1, 2])
            target = torch.tensor(row[self.labels].values.astype(np.float32))
            return img, target
        else:
            return img, image_id




## === cell 3
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(64 * (IMG_SIZE[0] // 8) * (IMG_SIZE[1] // 8), num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x




## === cell 4
train_df = pd.read_csv(TRAIN_CSV)

val_frac = 0.2
val_df = train_df.sample(frac=val_frac, random_state=42)
train_df = train_df.drop(val_df.index)

pos_counts = train_df[label_cols].sum()
neg_counts = len(train_df) - pos_counts
pos_weights = (neg_counts / (pos_counts + 1e-6)).values
pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(DEVICE)

train_dataset = PlantDataset(train_df, is_train=True)
val_dataset = PlantDataset(val_df, is_train=True)

train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

model = SimpleCNN(num_classes=4).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LR, weight_decay=1e-5
)  # slight regularization
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2, verbose=True
)

best_val_auc = -np.inf
best_state = None


def train_one_epoch(loader):
    model.train()
    epoch_loss = 0.0
    for imgs, targets in loader:
        imgs = imgs.to(DEVICE)
        targets = targets.to(DEVICE)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)
    return epoch_loss / len(loader.dataset)


def evaluate(loader):
    model.eval()
    all_logits = []
    all_targets = []
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, targets)
            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    logits = torch.cat(all_logits)
    targets = torch.cat(all_targets)
    probs = torch.sigmoid(logits)
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    return avg_loss, probs.numpy()


for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_one_epoch(train_loader)
    val_loss, val_probs = evaluate(val_loader)

    aucs = []
    for i, col in enumerate(label_cols):
        try:
            auc = roc_auc_score(val_df[col].values, val_probs[:, i])
        except ValueError:
            auc = 0.5  # fallback when only one class present
        aucs.append(auc)
    val_auc = np.mean(aucs)

    print(
        f"Epoch {epoch}: Train loss {train_loss:.4f} | Val loss {val_loss:.4f} | Val AUC {val_auc:.4f}"
    )

    scheduler.step(val_auc)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_state = copy.deepcopy(model.state_dict())

if best_state is not None:
    model.load_state_dict(best_state)




## === cell 5
test_df = pd.read_csv(TEST_CSV)
test_dataset = PlantDataset(test_df, is_train=False)
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

model.eval()
all_preds = []
all_ids = []
with torch.no_grad():
    for imgs, ids in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_preds.append(probs)
        all_ids.extend(ids)

test_predictions = np.concatenate(all_preds, axis=0)  # shape (num_test, 4)




## === cell 6
submission = pd.read_csv(SAMPLE_SUBMISSION)

id_to_pred = {img_id: pred for img_id, pred in zip(all_ids, test_predictions)}

pred_matrix = np.vstack(
    submission["image_id"]
    .map(id_to_pred)
    .apply(lambda x: x if isinstance(x, np.ndarray) else np.zeros(4))
)

submission.loc[:, label_cols] = pred_matrix
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to {SUBMISSION_PATH}")




## === cell 7
if __name__ == "__main__":
    pass
