# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import torch

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_ROOT_CANDIDATES = [
    "../input/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification",
    "../input",
    "/kaggle/input",
]
DATA_ROOT = None
for cand in DATA_ROOT_CANDIDATES:
    if os.path.exists(cand):
        if os.path.exists(os.path.join(cand, "train.csv")) and os.path.exists(
            os.path.join(cand, "train")
        ):
            DATA_ROOT = cand
            break
        if os.path.exists(
            os.path.join(cand, "aerial-cactus-identification", "train.csv")
        ):
            DATA_ROOT = os.path.join(cand, "aerial-cactus-identification")
            break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find dataset root under expected ../input paths."
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR = os.path.join(DATA_ROOT, "test")

print("Using DATA_ROOT:", DATA_ROOT)
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Train dir exists:", os.path.exists(TRAIN_DIR))
print("Test dir exists:", os.path.exists(TEST_DIR))




## === cell 1
from PIL import Image

train_labels_df = pd.read_csv(TRAIN_CSV)
print(train_labels_df.head())
print("Train rows:", len(train_labels_df))

fname0 = train_labels_df["id"].iloc[0]
img_path0 = os.path.join(TRAIN_DIR, fname0)
img0 = Image.open(img_path0).convert("RGB")
print("Opened sample image:", fname0, "size:", img0.size)




## === cell 2
print("How many pictures contain cactuses (positive rate)?")
pos_rate = train_labels_df["has_cactus"].mean()
print(pos_rate)




## === cell 3
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

vgg_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),  # outputs CHW float32 in [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class CactusDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, cache=True):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.cache = cache
        self._x_cache = [None] * len(self.df) if cache else None

    def __len__(self):
        return len(self.df)

    def _load_one(self, idx):
        fname = self.df.loc[idx, "id"]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, idx):
        if self.cache:
            x = self._x_cache[idx]
            if x is None:
                x = self._load_one(idx)
                self._x_cache[idx] = x
        else:
            x = self._load_one(idx)

        y = int(self.df.loc[idx, "has_cactus"])
        return x, torch.tensor(y, dtype=torch.long)


full_dataset = CactusDataset(
    train_labels_df, TRAIN_DIR, transform=vgg_transform, cache=True
)
print("Prefilling train cache (one-time cost)...")
for i in range(len(full_dataset)):
    _ = full_dataset[i]
print("Cache ready.")

n = len(full_dataset)
train_size = int(n * 0.9)
val_size = n - train_size
train_set, val_set = random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed)
)

num_workers = min(4, os.cpu_count() or 2)

train_loader = DataLoader(
    train_set,
    batch_size=64,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
)
val_loader = DataLoader(
    val_set,
    batch_size=128,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
)

print("Train/Val sizes:", len(train_set), len(val_set), "num_workers:", num_workers)




## === cell 4
from torchvision.models import vgg16, VGG16_Weights

weights = VGG16_Weights.IMAGENET1K_V1
model = vgg16(weights=weights, progress=True)

num_features = model.classifier[6].in_features
features = list(model.classifier.children())[:-1]
features.extend([torch.nn.Linear(num_features, 2)])
model.classifier = torch.nn.Sequential(*features)

for param in model.features.parameters():
    param.requires_grad = False

model = model.to(device)
print(model.classifier)




## === cell 5
learning_rate = 0.0001
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




## === cell 6
if torch.cuda.is_available():
    try:
        model = torch.compile(model)  # PyTorch 2.x
        print("torch.compile enabled.")
    except Exception as e:
        print("torch.compile not available, using eager mode. Reason:", repr(e))

model.train()
for epoch in range(50):
    losses = []
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        y_pred = model(batch_x)
        loss = loss_fn(y_pred, batch_y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"epoch={epoch+1:02d} loss={sum(losses)/len(losses):.5f}")




## === cell 7
from sklearn.metrics import classification_report

model.eval()
y_true = []
y_hat = []
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x = batch_x.to(device, non_blocking=True)
        logits = model(batch_x)
        preds = logits.argmax(dim=1).cpu().numpy().tolist()
        y_hat.extend(preds)
        y_true.extend(batch_y.numpy().tolist())

print(classification_report(y_true, y_hat, digits=4))




## === cell 8
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
test_ids = sample_sub["id"].tolist()


class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform=None, cache=True):
        self.ids = list(ids)
        self.img_dir = img_dir
        self.transform = transform
        self.cache = cache
        self._x_cache = [None] * len(self.ids) if cache else None

    def __len__(self):
        return len(self.ids)

    def _load_one(self, idx):
        fname = self.ids[idx]
        img_path = os.path.join(self.img_dir, fname)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return fname, img

    def __getitem__(self, idx):
        if self.cache:
            cached = self._x_cache[idx]
            if cached is None:
                cached = self._load_one(idx)
                self._x_cache[idx] = cached
            return cached
        return self._load_one(idx)


test_dataset = TestDataset(test_ids, TEST_DIR, transform=vgg_transform, cache=True)
print("Prefilling test cache (one-time cost)...")
for i in range(len(test_dataset)):
    _ = test_dataset[i]
print("Test cache ready.")

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=(num_workers > 0),
)

model.eval()
pred_proba = []
pred_ids = []

softmax = torch.nn.Softmax(dim=1)
with torch.no_grad():
    for fnames, imgs in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = softmax(logits)[:, 1].detach().cpu().numpy()  # P(has_cactus=1)
        pred_proba.extend(probs.tolist())
        pred_ids.extend(list(fnames))

pred_df = pd.DataFrame({"id": pred_ids, "has_cactus": pred_proba})
pred_df = sample_sub[["id"]].merge(pred_df, on="id", how="left")

assert pred_df["has_cactus"].isna().sum() == 0, "Missing predictions for some test ids."
assert len(pred_df) == len(sample_sub), "Submission row count mismatch."

out_path = "submission.csv"
pred_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(pred_df.head())
