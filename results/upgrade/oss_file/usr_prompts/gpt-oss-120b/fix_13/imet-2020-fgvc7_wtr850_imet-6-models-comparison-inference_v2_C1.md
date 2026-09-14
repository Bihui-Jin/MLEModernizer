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

# 5. Target score

0.302447556218187

# 6. Current score

0.33587

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0033) has done: 'I fix the import error by removing the unavailable efficientnet_pytorch dependency and instead use the ResNet‑50 model already defined in the script. I also correct the create_net function so it properly loads torchvision pretrained weights without trying to read nonexistent local weight files. The updated cells keep the original logic intact while ensuring the model is instantiated, inference runs, and a valid submission.csv is written.'
- What this solution (achieved 0.00689) has done: 'I keep the overall pipeline unchanged but improve the post‑processing step that creates the final predictions. The original code used a very low fixed threshold (0.10), which produced many false positives and a micro‑F1 of 0.0033. By raising the threshold to 0.5 and additionally limiting each image to at most the three most confident classes (while still falling back to the top‑k when no prediction passes the threshold), we reduce noise and should move the score much closer to the target 0.302447556218187.'
- What this solution (achieved 0.3788) has done: 'The fix replaces the wrong import of `contextmanager` with the correct one from `contextlib`, which resolves the initial import error and consequently makes all later definitions (like `timer`, `np`, `pd`, etc.) available. No other logic is altered, preserving the original model, training, and post‑processing while ensuring the script runs end‑to‑end and writes a valid `submission.csv`.'
- What this solution (achieved 0.33587) has done: 'I modestly raise the prediction confidence threshold and lower the fallback top‑K count. Increasing `THRESHOLD` from 0.5 to 0.7 and reducing `TOP_K` from 3 to 2 make the model output fewer labels per image, which should lower the micro‑F1 score from the current 0.3788 toward the target ≈0.30 while keeping the core training and architecture unchanged.'

# 9. Code solution

## === cell 0
import os, time, random, logging
from contextlib import contextmanager

import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.models as models
import torchvision.transforms as T

from albumentations import Compose, RandomResizedCrop, Normalize
from albumentations.pytorch import ToTensorV2


def get_data_root():
    candidates = [
        "/kaggle/input/imet-2020-fgvc7",
        "./kaggle/data/imet-2020-fgvc7",
        "./data/imet-2020-fgvc7",
        "./kaggle/data",
        "./data",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError("Could not locate the dataset root directory.")


DATA_ROOT = get_data_root()
LOGGER = None


@contextmanager
def timer(name):
    t0 = time.time()
    LOGGER.info(f"[{name}] start")
    yield
    LOGGER.info(f"[{name}] done in {time.time() - t0:.0f} s.")


def init_logger(log_file="train.log"):
    logger = logging.getLogger("Herbarium")
    logger.setLevel(logging.DEBUG)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(logging.Formatter(fmt))
    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


LOG_FILE = "train.log"
LOGGER = init_logger(LOG_FILE)


def seed_torch(seed=777):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


SEED = 777
seed_torch(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f"Using device: {device}")

HEIGHT = 128
WIDTH = 128
BATCH_SIZE = 128

THRESHOLD = 0.7  # higher confidence required
TOP_K = 2  # fewer fallback predictions

EPOCHS = 3  # a few more epochs to improve score slightly




## === cell 1
def get_transforms(*, data):
    assert data in ("train", "valid")
    if data == "train":
        return Compose(
            [
                RandomResizedCrop(size=(HEIGHT, WIDTH)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:  # valid / inference
        return Compose(
            [
                RandomResizedCrop(
                    size=(HEIGHT, WIDTH), scale=(1.0, 1.0), ratio=(1.0, 1.0)
                ),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


class TrainDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.ids = df["id"].values
        self.img_dir = img_dir
        self.transform = transform
        self.num_classes = len(pd.read_csv(os.path.join(DATA_ROOT, "labels.csv")))
        self.labels_matrix = np.zeros((len(df), self.num_classes), dtype=np.uint8)
        for idx, label_str in enumerate(df["attribute_ids"].values):
            for lbl in label_str.split():
                self.labels_matrix[idx, int(lbl)] = 1

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.ids[idx]}.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.transform(image=img)
        img = aug["image"]
        target = torch.from_numpy(self.labels_matrix[idx]).float()
        return img, target


class TestDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.ids = df["id"].values
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.ids[idx]}.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.transform(image=img)
        return aug["image"]


train_df = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
labels_df = pd.read_csv(os.path.join(DATA_ROOT, "labels.csv"))
num_classes = len(labels_df)

from sklearn.model_selection import train_test_split

train_split, valid_split = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
)

NUM_WORKERS = min(8, os.cpu_count() or 2)

train_dataset = TrainDataset(
    train_split,
    img_dir=os.path.join(DATA_ROOT, "train"),
    transform=get_transforms(data="train"),
)
valid_dataset = TrainDataset(
    valid_split,
    img_dir=os.path.join(DATA_ROOT, "train"),
    transform=get_transforms(data="valid"),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

LOGGER.info("Starting quick training")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    tk0 = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}")
    for imgs, targets in tk0:
        imgs = imgs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    LOGGER.info(f"Epoch {epoch+1} loss: {epoch_loss/len(train_loader):.4f}")

submission = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
test_dataset = TestDataset(
    submission,
    img_dir=os.path.join(DATA_ROOT, "test"),
    transform=get_transforms(data="valid"),
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)




## === cell 2
with timer("inference"):
    model.eval()
    preds = []
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader), desc="Inference")
    for i, images in tk0:
        images = images.to(device)
        with torch.no_grad():
            y_preds = model(images)
        preds.append(torch.sigmoid(y_preds).cpu().numpy())




## === cell 3
preds_array = np.concatenate(preds, axis=0)  # (n_samples, n_classes)

final_preds = np.zeros_like(preds_array, dtype=bool)
for i, row in enumerate(preds_array):
    idx_thr = np.where(row > THRESHOLD)[0]
    if len(idx_thr) == 0:
        idx_thr = np.argsort(row)[-TOP_K:]
    elif len(idx_thr) > TOP_K:
        idx_thr = idx_thr[np.argsort(row[idx_thr])[-TOP_K:]]
    final_preds[i, idx_thr] = True

for i, row in enumerate(final_preds):
    ids = np.nonzero(row)[0]
    submission.at[i, "attribute_ids"] = " ".join(map(str, ids))

submission.to_csv("submission.csv", index=False)
print("Saved submission to submission.csv")
