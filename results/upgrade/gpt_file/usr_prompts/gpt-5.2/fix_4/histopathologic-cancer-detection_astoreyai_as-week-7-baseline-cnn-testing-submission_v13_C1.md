# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.13

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
numpy==1.26.4
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
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from tqdm import tqdm

import torchvision
from torchvision import transforms



## === cell 1
DATA_DIR = "/kaggle/input/histopathologic-cancer-detection"
TRAIN_DIR = f"{DATA_DIR}/train"
TEST_DIR = f"{DATA_DIR}/test"
TRAIN_CSV = f"{DATA_DIR}/train_labels.csv"

MODEL_PATH = "/kaggle/input/as-week-4-baseline-cnn-training/model_best.pth"  # Provided path (may be missing)
SUBMISSION_FILE = "submission.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TARGET_SIZE = (96, 96)
BATCH_SIZE = 64
NUM_CLASSES = 2  # Kept as in the original solution (2-logit softmax)

EPOCHS = 1
LR = 3e-4
WEIGHT_DECAY = 1e-4
VAL_FRAC = 0.1
SEED = 42

assert os.path.isdir(TEST_DIR), f"TEST_DIR not found: {TEST_DIR}"
assert os.path.isdir(TRAIN_DIR), f"TRAIN_DIR not found: {TRAIN_DIR}"
assert os.path.isfile(TRAIN_CSV), f"TRAIN_CSV not found: {TRAIN_CSV}"
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)



## === cell 2
default_weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
weights_transforms = default_weights.transforms()

train_transforms = transforms.Compose(
    [
        transforms.Resize(
            TARGET_SIZE, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights_transforms.mean, std=weights_transforms.std),
    ]
)

test_transforms = transforms.Compose(
    [
        transforms.Resize(
            TARGET_SIZE, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=weights_transforms.mean, std=weights_transforms.std),
    ]
)




## === cell 3
class HistologyDataset(Dataset):
    def __init__(self, df, img_dir, transform, with_label: bool):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.with_label = with_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = str(self.df.loc[idx, "id"])
        img_path = os.path.join(self.img_dir, f"{img_id}.tif")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        if self.with_label:
            y = int(self.df.loc[idx, "label"])
            return img, y
        else:
            return img, img_id




## === cell 4
labels_df = pd.read_csv(TRAIN_CSV)
labels_df["id"] = labels_df["id"].astype(str)
labels_df["label"] = labels_df["label"].astype(int)
print("Train labels:", labels_df.shape, "pos_rate:", labels_df["label"].mean())

pos_df = (
    labels_df[labels_df["label"] == 1]
    .sample(frac=1.0, random_state=SEED)
    .reset_index(drop=True)
)
neg_df = (
    labels_df[labels_df["label"] == 0]
    .sample(frac=1.0, random_state=SEED)
    .reset_index(drop=True)
)
pos_val_n = int(len(pos_df) * VAL_FRAC)
neg_val_n = int(len(neg_df) * VAL_FRAC)

val_df = pd.concat([pos_df.iloc[:pos_val_n], neg_df.iloc[:neg_val_n]], axis=0).sample(
    frac=1.0, random_state=SEED
)
train_df = pd.concat([pos_df.iloc[pos_val_n:], neg_df.iloc[neg_val_n:]], axis=0).sample(
    frac=1.0, random_state=SEED
)

print("Train split:", train_df.shape, "pos_rate:", train_df["label"].mean())
print("Val split:", val_df.shape, "pos_rate:", val_df["label"].mean())

train_dataset = HistologyDataset(train_df, TRAIN_DIR, train_transforms, with_label=True)
val_dataset = HistologyDataset(val_df, TRAIN_DIR, test_transforms, with_label=True)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)

sample_path = os.path.join(DATA_DIR, "sample_submission.csv")
sample_df = pd.read_csv(sample_path)
sample_df["id"] = sample_df["id"].astype(str)
test_img_ids = sample_df["id"].tolist()
print(f"Total test images (from sample_submission): {len(test_img_ids)}")

test_dataset = HistologyDataset(
    pd.DataFrame({"id": test_img_ids}), TEST_DIR, test_transforms, with_label=False
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    drop_last=False,
)




## === cell 5
class CancerClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, backbone_weights=None):
        super().__init__()
        self.model = torchvision.models.efficientnet_b0(weights=backbone_weights)
        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


ckpt_exists = isinstance(MODEL_PATH, str) and os.path.isfile(MODEL_PATH)
if ckpt_exists:
    model = CancerClassifier(backbone_weights=None).to(device)
    state = torch.load(MODEL_PATH, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            nk = k
            if nk.startswith("model."):
                nk = nk[len("model.") :]
            if nk.startswith("module."):
                nk = nk[len("module.") :]
            new_state[nk] = v
        state = new_state

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint from: {MODEL_PATH}")
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
else:
    model = CancerClassifier(backbone_weights=default_weights).to(device)
    print(f"WARNING: Checkpoint not found at '{MODEL_PATH}'.")
    print(
        "Falling back to torchvision EfficientNet-B0 ImageNet weights, then fine-tuning on train_labels.csv."
    )



## === cell 6
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)


def binary_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int32)
    y_score = y_score.astype(np.float64)

    order = np.argsort(y_score)
    y_true = y_true[order]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = np.arange(1, len(y_true) + 1, dtype=np.float64)
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def evaluate_auc(model, loader):
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            prob_pos = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
            ys.append(y.numpy())
            ps.append(prob_pos)
    y_true = np.concatenate(ys, axis=0)
    y_score = np.concatenate(ps, axis=0)
    return binary_auc_roc(y_true, y_score)


for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    n_seen = 0

    for x, y in tqdm(train_loader, desc=f"Training epoch {epoch}/{EPOCHS}"):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        running_loss += loss.item() * bs
        n_seen += bs

    train_loss = running_loss / max(1, n_seen)
    val_auc = evaluate_auc(model, val_loader)
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_auc={val_auc:.4f}")

model.eval()




## === cell 7
def generate_predictions(model, loader):
    """
    Generate predictions for the test set.
    Returns:
        ids: list[str]
        predictions: list[float] probabilities for the positive class
    """
    model.eval()
    predictions = []
    ids = []

    with torch.no_grad():
        for images, img_ids in tqdm(loader, desc="Generating predictions"):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1]  # positive class probability
            predictions.extend(probs.detach().cpu().numpy().astype(np.float32).tolist())
            ids.extend(list(img_ids))

    return ids, predictions


img_ids, predictions = generate_predictions(model, test_loader)
print("Predictions generated:", len(predictions))




## === cell 8
def prepare_submission(img_ids, predictions):
    """
    For AUC evaluation, submit probabilities (floats), not hard-thresholded labels.
    """
    submission_df = pd.DataFrame({"id": img_ids, "label": predictions})
    submission_df["id"] = submission_df["id"].astype(str)

    submission_df = sample_df[["id"]].merge(submission_df, on="id", how="left")
    if submission_df["label"].isna().any():
        submission_df["label"] = submission_df["label"].fillna(0.5)

    submission_df.to_csv(SUBMISSION_FILE, index=False)
    print(
        f"Submission file '{SUBMISSION_FILE}' created with shape {submission_df.shape}."
    )
    return submission_df


sub_df = prepare_submission(img_ids, predictions)
print(sub_df.head())



## === cell 9
assert os.path.isfile(SUBMISSION_FILE), "submission.csv was not created"
check = pd.read_csv(SUBMISSION_FILE)
assert list(check.columns) == ["id", "label"], f"Bad columns: {check.columns.tolist()}"
assert len(check) == len(sample_df), f"Bad row count: {len(check)} vs {len(sample_df)}"
assert check["label"].between(0, 1).all(), "Labels must be probabilities in [0,1]"
print("Submission looks valid.")
print(check.head())
