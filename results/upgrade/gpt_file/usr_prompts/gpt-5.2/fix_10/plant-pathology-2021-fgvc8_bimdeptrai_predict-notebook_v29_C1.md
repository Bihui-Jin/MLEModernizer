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

No external packages required in the script and installed.

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
import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

print("Python OK, seed:", SEED)



## === cell 1
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
print(train.head())



## === cell 2
from sklearn.preprocessing import MultiLabelBinarizer

label_split = train["labels"].str.split()
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
labels = pd.DataFrame(y, columns=mlb.classes_)

target_cols = list(mlb.classes_)
num_classes = len(target_cols)
print("Classes:", target_cols)
print(labels.head())



## === cell 3
for label in labels.columns:
    vc = labels[label].value_counts(normalize=True)
    print(label, vc.to_dict())



## === cell 4
h_target = 256
w_target = 256
batch_size = 32

EPOCHS_STAGE1 = 2
EPOCHS_STAGE2 = 1
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-4



## === cell 5
train_df = train.copy()
train_df["filepath"] = TRAIN_IMG_DIR + "/" + train_df["image"].astype(str)

for c in target_cols:
    train_df[c] = labels[c].values

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_size = int(0.1 * len(train_df))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train/Val:", trn_df.shape, val_df.shape)
print("Example path exists?", os.path.exists(trn_df.loc[0, "filepath"]))



## === cell 6
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "Torch:", torch.__version__, "CUDA:", torch.cuda.is_available(), "Device:", device
)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_tfms = transforms.Compose(
    [
        transforms.Resize((h_target, w_target)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(
            degrees=20, interpolation=transforms.InterpolationMode.BILINEAR, fill=0
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.10, 0.10),
            scale=(0.90, 1.10),
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)

val_tfms = transforms.Compose(
    [
        transforms.Resize((h_target, w_target)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


class PlantDataset(Dataset):
    def __init__(
        self, df, target_cols=None, transform=None, is_test=False, test_img_dir=None
    ):
        self.transform = transform
        self.is_test = is_test

        if is_test:
            self.images = df["image"].astype(str).values
            self.paths = np.array(
                [os.path.join(test_img_dir, img) for img in self.images], dtype=object
            )
            self.targets = None
        else:
            self.paths = df["filepath"].astype(str).values
            self.targets = df[target_cols].values.astype(np.float32, copy=False)
            self.images = df["image"].astype(str).values  # unused but handy

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.is_test:
            return img, str(self.images[i])
        else:
            return img, torch.from_numpy(self.targets[i])


train_ds = PlantDataset(
    trn_df, target_cols=target_cols, transform=train_tfms, is_test=False
)
val_ds = PlantDataset(
    val_df, target_cols=target_cols, transform=val_tfms, is_test=False
)
test_ds = PlantDataset(
    submissions,
    target_cols=None,
    transform=val_tfms,
    is_test=True,
    test_img_dir=TEST_IMG_DIR,
)


def _recommended_num_workers():
    cpu = os.cpu_count() or 2
    return min(8, max(2, cpu // 2))


num_workers = _recommended_num_workers()

train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)
val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)
test_loader = DataLoader(
    test_ds,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker,
    generator=g,
)




## === cell 7
class DenseNet201MultiLabel(nn.Module):
    def __init__(self, num_classes, dropout=0.25, pretrained=True):
        super().__init__()
        if pretrained:
            try:
                weights = torchvision.models.DenseNet201_Weights.IMAGENET1K_V1
            except Exception:
                weights = "IMAGENET1K_V1"
        else:
            weights = None

        self.backbone = torchvision.models.densenet201(weights=weights)
        in_feats = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(in_feats, num_classes)

    def forward(self, x):
        feats = self.backbone(x)  # already pooled to vector in torchvision densenet
        feats = self.dropout(feats)
        logits = self.head(feats)
        probs = torch.sigmoid(logits)
        return probs, logits


model = DenseNet201MultiLabel(
    num_classes=num_classes, dropout=0.25, pretrained=True
).to(device)


def set_backbone_trainable(model, trainable: bool, unfreeze_last_n_params: int = None):
    for p in model.backbone.parameters():
        p.requires_grad = bool(trainable)

    if trainable and unfreeze_last_n_params is not None:
        params = list(model.backbone.parameters())
        for p in params[:-unfreeze_last_n_params]:
            p.requires_grad = False
        for p in params[-unfreeze_last_n_params:]:
            p.requires_grad = True


set_backbone_trainable(model, trainable=False)

criterion = nn.BCEWithLogitsLoss()


def make_optimizer(lr):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.Adam(params, lr=lr)


optimizer = make_optimizer(LR_STAGE1)

print(
    "Trainable params stage1:",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)




## === cell 8
def mean_f1_score(y_true, y_pred_bin, eps=1e-9):
    tp = (y_true * y_pred_bin).sum(axis=0)
    fp = ((1 - y_true) * y_pred_bin).sum(axis=0)
    fn = (y_true * (1 - y_pred_bin)).sum(axis=0)
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    return float(np.mean(f1))


@torch.no_grad()
def validate_epoch(thresholds_vec):
    model.eval()
    all_probs = []
    all_true = []
    for xb, yb in val_loader:
        xb = xb.to(device, non_blocking=True)
        probs, _ = model(xb)
        all_probs.append(probs.detach().cpu())
        all_true.append(yb.detach().cpu())

    probs = torch.cat(all_probs, dim=0).numpy()
    y_true = torch.cat(all_true, dim=0).numpy().astype(np.int32, copy=False)
    y_pred = (probs > thresholds_vec[None, :]).astype(np.int32)
    return mean_f1_score(y_true, y_pred)


def train_one_epoch():
    model.train()
    running = 0.0
    n = 0
    for xb, yb in train_loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        probs, logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        bs = xb.size(0)
        running += float(loss.item()) * bs
        n += bs
    return running / max(1, n)


thresh = {
    "complex": 0.25,
    "frog_eye_leaf_spot": 0.25,
    "healthy": 0.25,
    "powdery_mildew": 0.25,
    "rust": 0.25,
    "scab": 0.25,
}
assert set(thresh.keys()) == set(target_cols), "Threshold keys must match class names."
thr_vec = np.array([thresh[c] for c in target_cols], dtype=np.float32)

for ep in range(EPOCHS_STAGE1):
    loss = train_one_epoch()
    val_f1 = validate_epoch(thr_vec)
    print(
        f"Stage1 epoch {ep+1}/{EPOCHS_STAGE1} - loss: {loss:.5f} - val_meanF1@0.25: {val_f1:.5f}"
    )

set_backbone_trainable(model, trainable=True, unfreeze_last_n_params=50)
optimizer = make_optimizer(LR_STAGE2)
print(
    "Trainable params stage2:",
    sum(p.numel() for p in model.parameters() if p.requires_grad),
)

for ep in range(EPOCHS_STAGE2):
    loss = train_one_epoch()
    val_f1 = validate_epoch(thr_vec)
    print(
        f"Stage2 epoch {ep+1}/{EPOCHS_STAGE2} - loss: {loss:.5f} - val_meanF1@0.25: {val_f1:.5f}"
    )




## === cell 9
@torch.no_grad()
def predict_test():
    model.eval()
    all_probs = []
    all_images = []
    for xb, img_names in test_loader:
        xb = xb.to(device, non_blocking=True)
        probs, _ = model(xb)
        all_probs.append(probs.detach().cpu().numpy())
        all_images.extend(list(img_names))
    probs = np.concatenate(all_probs, axis=0)
    return all_images, probs


test_images, preds_np = predict_test()
print("preds shape:", preds_np.shape)
print("first images:", test_images[:3])
print("first probs:", preds_np[:2])



## === cell 10
sub_df = submissions.copy()

preds_ordered = preds_np.astype(np.float32, copy=False)

label_order = list(target_cols)
lab2i = {lab: i for i, lab in enumerate(label_order)}

healthy_i = lab2i["healthy"]
row_max_i = preds_ordered.argmax(axis=1)
is_healthy_max = row_max_i == healthy_i

thr_vec = np.array([thresh[lab] for lab in label_order], dtype=np.float32)
above = preds_ordered > thr_vec[None, :]

label_arr = np.asarray(label_order, dtype=object)
out_labels = np.empty(len(sub_df), dtype=object)
out_labels[is_healthy_max] = "healthy"

nonhealthy_idx = np.where(~is_healthy_max)[0]
for i in nonhealthy_idx:
    mask = above[i]
    if not mask.any():
        out_labels[i] = label_order[int(row_max_i[i])]
        continue
    comb = label_arr[mask].tolist()
    if ("healthy" in comb) and (len(comb) > 1):
        comb = [l for l in comb if l != "healthy"]
    out_labels[i] = " ".join(comb)

sub_df["labels"] = out_labels.tolist()
sub_df[["image", "labels"]].to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with", len(sub_df), "rows")



## === cell 11
assert os.path.exists("submission.csv"), "submission.csv was not created"
check = pd.read_csv("submission.csv")
assert list(check.columns) == ["image", "labels"]
assert len(check) == len(submissions)
assert (
    check["image"].astype(str).tolist() == submissions["image"].astype(str).tolist()
), "Image order mismatch"
print(check.head())
print("submission.csv OK")
