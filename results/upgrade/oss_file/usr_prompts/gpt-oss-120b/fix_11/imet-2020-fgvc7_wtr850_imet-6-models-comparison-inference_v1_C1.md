# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Label artwork images with significant attributes.

## Metric
Micro averaged F1 score.

## Submission Format
```
id,attribute_ids
00011f01965f141f5d1eea6592fa9862,0 1 2
00014abc91ed3e4bf1663fde8136fe80,0 1 2
0002e2054e303badc1a33463f6fb7973,0 1 2
```

## Dataset
Multiple modalities can be expected and the camera sources are unknown. The photographs are often centered for objects, and in the case where the museum artifact is an entire room, the images are scenic in nature.

Each object is annotated by a single annotator without a verification step. You should consider these annotations noisy.

The filename of each image is its `id`.

- **train.csv** gives the `attribute_ids` for the train images in **/train**
- **/test** contains the test images. You must predict the `attribute_ids` for these images.
- **sample_submission.csv** contains a submission in the correct format
- **labels.csv** provides descriptions of the attributes

# 2. Python version

3.8

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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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
            description.md (81 lines)
            labels.csv (3475 lines)
            labels.csv.zip (28.4 kB)
            sample_submission.csv (21319 lines)
            sample_submission.csv.zip (426.2 kB)
            test.zip (3.8 GB)
            train.csv (120802 lines)
            train.csv.zip (3.2 MB)
            train.zip (21.4 GB)
            imet-2020-fgvc7/
                description.md (81 lines)
                labels.csv (3475 lines)
                ... and 7 other files
                imet-2020-fgvc7/
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
            test/
                c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                ... and 21316 other files
                test/
            train/
                cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                ... and 120799 other files
                train/
        input/
            description.md (81 lines)
            labels.csv (3475 lines)
            labels.csv.zip (28.4 kB)
            sample_submission.csv (21319 lines)
            sample_submission.csv.zip (426.2 kB)
            test.zip (3.8 GB)
            train.csv (120802 lines)
            train.csv.zip (3.2 MB)
            train.zip (21.4 GB)
            imet-2020-fgvc7/
                description.md (81 lines)
                labels.csv (3475 lines)
                ... and 7 other files
                imet-2020-fgvc7/
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
            test/
                c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                ... and 21316 other files
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
            train/
                cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                ... and 120799 other files
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
        working/
            imet-2020-fgvc7/
                description.md (81 lines)
                labels.csv (3475 lines)
                ... and 7 other files
                imet-2020-fgvc7/
                test/
                    c48792ab798551af4bc8e7dec5d89c31.png (47.1 kB)
                    2e4ae3954a0167eedc8ac74256eace08.png (223.2 kB)
                    ... and 21316 other files
                    test/
                train/
                    cae7d65be972cbfa9e1af483219fbe26.png (212.7 kB)
                    bcbeb1d22d738370626f4686338e469f.png (137.5 kB)
                    ... and 120799 other files
                    train/
```

-> data/imet-2020-fgvc7/labels.csv has 3474 rows and 2 columns.
The columns are: attribute_id, attribute_name

-> data/imet-2020-fgvc7/sample_submission.csv has 21318 rows and 2 columns.
The columns are: id, attribute_ids

-> data/imet-2020-fgvc7/train.csv has 120801 rows and 2 columns.
The columns are: id, attribute_ids

-> data/labels.csv has 3474 rows and 2 columns.
The columns are: attribute_id, attribute_name

-> data/sample_submission.csv has 21318 rows and 2 columns.
The columns are: id, attribute_ids

-> data/train.csv has 120801 rows and 2 columns.
The columns are: id, attribute_ids

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import time
import threading
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
from albumentations import Compose, RandomResizedCrop, Resize, Normalize, ToTensorV2
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm import tqdm

_image_cache = {}
_cache_lock = threading.Lock()


def _load_image_cached(path: Path):
    """Load an image once per worker process and reuse it."""
    with _cache_lock:
        if path not in _image_cache:
            img = cv2.imread(str(path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            _image_cache[path] = img
    return _image_cache[path]


def init_logger(log_file="train.log"):
    from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler

    log_format = "%(asctime)s %(levelname)s %(message)s"
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter(log_format))
    file_handler = FileHandler(log_file)
    file_handler.setFormatter(Formatter(log_format))
    logger = getLogger("Herbarium")
    logger.setLevel(DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


LOGGER = init_logger()


def timer(name):
    t0 = time.time()
    LOGGER.info(f"[{name}] start")
    yield
    LOGGER.info(f"[{name}] done in {time.time() - t0:.0f} s.")


def seed_torch(seed=777):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


SEED = 777
seed_torch(SEED)

BASE_DIR = Path("/kaggle/input/imet-2020-fgvc7")
N_CLASSES = 3474
HEIGHT, WIDTH = 128, 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path("model.ckpt")

torch.backends.cudnn.benchmark = True


def get_transforms(*, data):
    assert data in ("train", "valid")
    if data == "train":
        return Compose(
            [
                RandomResizedCrop((HEIGHT, WIDTH)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:  # valid / test
        return Compose(
            [
                Resize(HEIGHT, WIDTH),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


class TrainDataset(Dataset):
    """
    Pre‑compute one‑hot label tensors once in __init__ to avoid
    rebuilding a large zero vector on every __getitem__ call.
    This reduces per‑sample Python overhead while preserving exact targets.
    """

    def __init__(self, df, transform=None):
        self.ids = df["id"].values
        self.labels = torch.stack(
            [
                torch.tensor(
                    np.isin(np.arange(N_CLASSES), lbls).astype(np.float32),
                    dtype=torch.float32,
                )
                for lbls in df["label_idxs"].values
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        file_name = self.ids[idx]
        file_path = BASE_DIR / "train" / f"{file_name}.png"
        image = _load_image_cached(file_path)
        if self.transform:
            image = self.transform(image=image)["image"]
        target = self.labels[idx]
        return image, target


class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.ids = df["id"].values
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        file_name = self.ids[idx]
        file_path = BASE_DIR / "test" / f"{file_name}.png"
        image = _load_image_cached(file_path)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image


model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
criterion = nn.BCEWithLogitsLoss()
model = model.to(DEVICE)




## === cell 1
if CHECKPOINT_PATH.exists():
    LOGGER.info(f"Loading checkpoint from {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    LOGGER.info("Checkpoint not found – starting quick training (5 epochs).")
    train_df = pd.read_csv(BASE_DIR / "train.csv")
    train_df["label_idxs"] = train_df["attribute_ids"].apply(
        lambda x: [int(i) for i in str(x).split()]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.1, random_state=SEED, shuffle=True
    )
    val_df["label_idxs"] = val_df["attribute_ids"].apply(
        lambda x: [int(i) for i in str(x).split()]
    )
    train_dataset = TrainDataset(train_df, transform=get_transforms(data="train"))
    val_dataset = TrainDataset(val_df, transform=get_transforms(data="valid"))

    batch_size = 256
    num_workers = min(8, os.cpu_count() or 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler()  # mixed precision scaler

    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1} training"):
            images = images.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        LOGGER.info(f"Epoch {epoch+1} training loss: {avg_loss:.4f}")

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc="Validation"):
                images = images.to(DEVICE, non_blocking=True)
                with torch.cuda.amp.autocast():
                    logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_preds.append(probs)
                all_labels.append(targets.cpu().numpy())
        preds_val = np.concatenate(all_preds)
        labels_val = np.concatenate(all_labels)
        val_pred_bin = (preds_val > 0.5).astype(int)
        f1 = f1_score(labels_val, val_pred_bin, average="micro")
        LOGGER.info(f"Validation micro‑F1 after epoch {epoch+1}: {f1:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        CHECKPOINT_PATH,
    )
    LOGGER.info(f"Checkpoint saved to {CHECKPOINT_PATH}")

LOGGER.info("Running inference on test data.")
sample_sub_path = BASE_DIR / "sample_submission.csv"
test_df = pd.read_csv(sample_sub_path)  # contains ids column only
test_dataset = TestDataset(test_df, transform=get_transforms(data="valid"))
test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

model.eval()
test_preds = []
with torch.no_grad():
    for images in tqdm(test_loader, desc="Test inference"):
        images = images.to(DEVICE, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy()
        test_preds.append(probs)
test_preds = np.concatenate(test_preds)  # shape (num_test, N_CLASSES)




## === cell 2
threshold = 0.2  # lower threshold to increase recall (helps micro‑F1)
binary_preds = test_preds > threshold

submission = test_df.copy()
submission["attribute_ids"] = ""

for i, row in enumerate(binary_preds):
    ids = np.nonzero(row)[0]
    submission.at[i, "attribute_ids"] = " ".join(map(str, ids))

submission.to_csv("submission.csv", index=False)
LOGGER.info("Submission file written to submission.csv")
