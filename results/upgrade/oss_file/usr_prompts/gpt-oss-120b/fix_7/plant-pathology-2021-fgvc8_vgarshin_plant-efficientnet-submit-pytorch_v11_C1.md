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

# 5. Code solution

## === cell 0
import os, gc, json, time
import cv2, pandas as pd, numpy as np
import torch, torch.nn as nn, torch.utils.data as data
from torch.utils.data.sampler import SequentialSampler
from torchvision import transforms, models as tv_models
from sklearn.metrics import f1_score  # added for threshold selection

KAGGLE = True
if not KAGGLE:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_path = (
    "../input/plant-pathology-2021-fgvc8/train.csv" if KAGGLE else "./data/train.csv"
)
train_df = pd.read_csv(train_path)

all_labels = set()
for lab_str in train_df["labels"].astype(str):
    all_labels.update(lab_str.split())
all_labels = sorted(all_labels)
LABELS_ = {lbl: idx for idx, lbl in enumerate(all_labels)}
LABELS = {str(idx): lbl for lbl, idx in LABELS_.items()}

label_counts = np.zeros(len(LABELS_), dtype=np.float32)
for labs in train_df["labels"].astype(str):
    for lab in labs.split():
        label_counts[LABELS_[lab]] += 1
label_freq = label_counts / len(train_df)  # proportion per label

eps = 1e-12
label_logits = np.log(label_freq / (1.0 - label_freq + eps)).astype(np.float32)

WORKERS = 2
print(f"Loaded {len(train_df)} training rows, {len(LABELS_)} unique labels.")
print(f"Label frequencies (first 5): {label_freq[:5]}")
print(f"Corresponding logits (first 5): {label_logits[:5]}")


def _pred_labels_from_logits(logits, thresh):
    """Return list of label strings per sample given constant logits."""
    probs = 1 / (1 + np.exp(-logits))  # sigmoid
    pred = (probs > thresh).astype(int)
    rows = []
    for row in pred:
        idxs = np.where(row)[0]
        labs = [LABELS[str(i)] for i in idxs]
        if len(labs) == 1 and labs[0] == "healthy":
            rows.append(["healthy"])
        elif len(labs) == 0:
            rows.append(["healthy"])
        else:
            rows.append(labs)
    return rows


y_true = np.zeros((len(train_df), len(LABELS_)), dtype=int)
for i, labs in enumerate(train_df["labels"].astype(str)):
    for lab in labs.split():
        y_true[i, LABELS_[lab]] = 1

candidate_thresholds = np.arange(0.5, 0.91, 0.05)  # try 0.5‑0.9
best_th = 0.4
best_f1 = -1.0
for th in candidate_thresholds:
    pred_labels = _pred_labels_from_logits(label_logits, th)
    y_pred = np.zeros_like(y_true)
    for i, labs in enumerate(pred_labels):
        for lab in labs:
            y_pred[i, LABELS_[lab]] = 1
    f1 = f1_score(y_true, y_pred, average="samples")
    print(f"Threshold {th:.2f} -> sample‑F1 {f1:.5f}")
    if f1 > best_f1:
        best_f1 = f1
        best_th = th

print(f"Selected threshold {best_th:.2f} with validation F1 {best_f1:.5f}")

TH = best_th




## === cell 1
TEST = True
VER = "v4"  # kept for compatibility; not used further
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
else:
    DATA_PATH = "./data"
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"
TRAIN_IMGS_PATH = f"{DATA_PATH}/train_images"
TTAS = [0, 1, 2]  # test‑time augmentations (indices used in flip)
FOLDS = [0]  # we will use a single model (trained or dummy)

print("Data path:", DATA_PATH)
print("Image folder:", IMGS_PATH)




## === cell 2
sample_sub_path = os.path.join(DATA_PATH, "sample_submission.csv")
df_sub = pd.read_csv(sample_sub_path, usecols=["image"])
df_sub["labels"] = "healthy"
print("Submission dataframe preview:")
print(df_sub.head())




## === cell 3
def flip(img, axis=0):
    if axis == 1:
        return img[::-1, :, :]
    elif axis == 2:
        return img[:, ::-1, :]
    elif axis == 3:
        return img[::-1, ::-1, :]
    else:
        return img


class PlantDataset(data.Dataset):
    def __init__(self, df, size, labels, transform=None, ttta=0):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.labels = labels  # None for inference
        self.transform = transform
        self.tta = ttta

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image"]
        img_path = os.path.join(IMGS_PATH, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        img = flip(img, axis=self.tta)
        img = img.transpose(2, 0, 1)  # C,H,W
        return torch.tensor(img.copy())




## === cell 4
img_size = 224
batch_size = 32
epochs = 5  # increased from 2 to allow better fine‑tuning
subset_size = len(train_df)  # use the entire training set instead of a small subset

train_subset = train_df.sample(n=subset_size, random_state=42).reset_index(drop=True)


class TrainDataset(data.Dataset):
    def __init__(self, df, size, label_map, transform=None):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["image"]
        img_path = os.path.join(TRAIN_IMGS_PATH, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)  # C,H,W
        img_tensor = torch.tensor(img, dtype=torch.float32)

        target = np.zeros(len(self.label_map), dtype=np.float32)
        for lab in row["labels"].split():
            target[self.label_map[lab]] = 1.0
        target_tensor = torch.tensor(target, dtype=torch.float32)
        return img_tensor, target_tensor


val_frac = 0.2
val_len = int(len(train_subset) * val_frac)
train_len = len(train_subset) - val_len
train_ds, val_ds = data.random_split(
    train_subset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
)
train_dataset = TrainDataset(train_ds.dataset.iloc[train_ds.indices], img_size, LABELS_)
val_dataset = TrainDataset(val_ds.dataset.iloc[val_ds.indices], img_size, LABELS_)

train_loader = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=WORKERS,
    pin_memory=True,
)

val_loader = data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=WORKERS,
    pin_memory=True,
)

model_ft = tv_models.efficientnet_b0(pretrained=True)
num_ftrs = model_ft.classifier[1].in_features
model_ft.classifier[1] = nn.Linear(num_ftrs, len(LABELS_))
model_ft = model_ft.to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_ft.parameters(), lr=1e-3)

print("Starting light fine‑tuning...")
for epoch in range(epochs):
    model_ft.train()
    running_loss = 0.0
    for imgs, targets in train_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        outputs = model_ft(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs} - Train loss: {epoch_loss:.4f}")

    model_ft.eval()
    all_val_logits = []
    all_val_targets = []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            logits = model_ft(imgs).cpu().numpy()
            all_val_logits.append(logits)
            all_val_targets.append(targets.cpu().numpy())
    val_logits = np.vstack(all_val_logits)
    val_targets = np.vstack(all_val_targets)

    best_f1_val = -1.0
    best_th_val = 0.5
    for th in candidate_thresholds:
        preds = (1 / (1 + np.exp(-val_logits)) > th).astype(int)
        f1 = f1_score(val_targets, preds, average="samples")
        if f1 > best_f1_val:
            best_f1_val = f1
            best_th_val = th
    print(f"  Validation best threshold {best_th_val:.2f} with F1 {best_f1_val:.5f}")

trained_model = model_ft
TH = best_th_val  # use validation‑derived threshold for test inference
print(f"Using threshold {TH:.2f} for final predictions.")




## === cell 5
datasets, loaders = [], []
img_size = 224  # consistent with training size
batch_size = 32

for tta in TTAS:
    ds = PlantDataset(df=df_sub, size=img_size, labels=None, transform=None, ttta=tta)
    datasets.append(ds)
    loader = data.DataLoader(
        ds,
        batch_size=batch_size,
        sampler=SequentialSampler(ds),
        num_workers=WORKERS,
        pin_memory=True,
    )
    loaders.append(loader)

print(f"Created {len(loaders)} loaders for TTAs {TTAS}")




## === cell 6
class DummyModel(nn.Module):
    def __init__(self, out_dim, freq_logits):
        super().__init__()
        self.register_buffer(
            "const_logits", torch.tensor(freq_logits, dtype=torch.float32)
        )
        self.out_dim = out_dim

    def forward(self, x):
        batch = x.shape[0]
        return self.const_logits.unsqueeze(0).repeat(batch, 1)


models = []
if "trained_model" in globals():
    model = trained_model
    model.eval()
    model.to(DEVICE)
    models.append(model)
    print("Using fine‑tuned EfficientNet model for inference.")
else:
    for n_fold in FOLDS:
        model = DummyModel(out_dim=len(LABELS_), freq_logits=label_logits)
        model.eval()
        model.to(DEVICE)
        models.append(model)
        print(f"Initialized dummy model for fold {n_fold}")
    del n_fold, model
gc.collect()




## === cell 7
def get_labels(row, idx2label, th):
    try:
        idxs = [i for i, x in enumerate(row) if x > th]
        labs = [idx2label[str(i)] for i in idxs]
        if len(labs) == 1 and labs[0] == "healthy":
            return "healthy"
        if len(labs) == 0:
            return "healthy"
        return " ".join(labs)
    except Exception as e:
        print("Error in get_labels:", e, row)
        return "healthy"


logits_per_tta = []  # will hold list of (num_images, num_labels) arrays per TTA
start_time = time.time()

with torch.no_grad():
    for tta_idx, loader in enumerate(loaders):
        batch_logits = []
        for imgs in loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            model_preds = []
            for model in models:
                preds = model(imgs).sigmoid().cpu().numpy()
                model_preds.append(preds)
            avg_preds = np.mean(model_preds, axis=0)  # shape (batch, num_labels)
            batch_logits.append(avg_preds)
        tta_logits = (
            np.vstack(batch_logits) if batch_logits else np.empty((0, len(LABELS_)))
        )
        logits_per_tta.append(tta_logits)
        print(f"TTA {tta_idx} done, shape {tta_logits.shape}")

if logits_per_tta:
    logits = np.mean(
        np.stack(logits_per_tta, axis=0), axis=0
    )  # (num_images, num_labels)
else:
    logits = np.empty((len(df_sub), len(LABELS_)))

df_sub["labels"] = [get_labels(row, LABELS, TH) for row in logits]

elapsed = time.time() - start_time
print(f"Inference completed in {int(elapsed // 60)}m {int(elapsed % 60)}s")




## === cell 8
print("Label distribution in submission:")
print(df_sub["labels"].value_counts())
print("\nSubmission preview:")
print(df_sub.head())




## === cell 9
submission_path = "submission.csv"
df_sub.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
