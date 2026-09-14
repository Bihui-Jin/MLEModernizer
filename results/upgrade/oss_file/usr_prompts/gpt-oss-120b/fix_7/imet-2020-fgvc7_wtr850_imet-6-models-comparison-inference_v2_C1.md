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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0033) has done: 'I fix the import error by removing the unavailable efficientnet_pytorch dependency and instead use the ResNet‑50 model already defined in the script. I also correct the create_net function so it properly loads torchvision pretrained weights without trying to read nonexistent local weight files. The updated cells keep the original logic intact while ensuring the model is instantiated, inference runs, and a valid submission.csv is written.'
- What this solution (achieved 0.00689) has done: 'I keep the overall pipeline unchanged but improve the post‑processing step that creates the final predictions. The original code used a very low fixed threshold (0.10), which produced many false positives and a micro‑F1 of 0.0033. By raising the threshold to 0.5 and additionally limiting each image to at most the three most confident classes (while still falling back to the top‑k when no prediction passes the threshold), we reduce noise and should move the score much closer to the target 0.302447556218187.'

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



## === cell 1

HEIGHT = 128
WIDTH = 128
BATCH_SIZE = 128
TOP_K = 3
THRESHOLD = 0.5

DATA_ROOT = "./kaggle/data/imet-2020-fgvc7"


def get_transforms(*, data):
    assert data in ("train", "valid")
    if data == "train":
        return Compose(
            [
                RandomResizedCrop(HEIGHT, WIDTH),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    else:  # valid / inference
        return Compose(
            [
                RandomResizedCrop(HEIGHT, WIDTH, scale=(1.0, 1.0), ratio=(1.0, 1.0)),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )


class TrainDataset(Dataset):
    def __init__(self, df, img_dir, transform):
        self.ids = df["id"].values
        self.targets = df["attribute_ids"].values
        self.img_dir = img_dir
        self.transform = transform
        self.num_classes = len(pd.read_csv(os.path.join(DATA_ROOT, "labels.csv")))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{self.ids[idx]}.png")
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = self.transform(image=img)
        img = aug["image"]
        target_vec = np.zeros(self.num_classes, dtype=np.float32)
        for label in self.targets[idx].split():
            target_vec[int(label)] = 1.0
        return img, torch.from_numpy(target_vec)


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
    train_df, test_size=0.1, random_state=SEED, stratify=train_df["attribute_ids"]
)

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
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
valid_loader = DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

LOGGER.info("Starting quick training")
model.train()
for epoch in range(1):
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
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/4287079147.py in <cell line: 0>()
     77 # ----------------------------------------------------------------------
     78 # Load training metadata
---> 79 train_df = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))
     80 labels_df = pd.read_csv(os.path.join(DATA_ROOT, "labels.csv"))
     81 num_classes = len(labels_df)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: './kaggle/data/imet-2020-fgvc7/train.csv'

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



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2472216904.py in <cell line: 0>()
      2 
      3 with timer("inference"):
----> 4     model.eval()
      5     preds = []
      6     tk0 = tqdm(enumerate(test_loader), total=len(test_loader), desc="Inference")

NameError: name 'model' is not defined

## === cell 3

preds_array = np.concatenate(preds, axis=0)  # shape (n_samples, N_CLASSES)

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

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/34622222.py in <cell line: 0>()
      1 # Post‑processing, thresholding, and submission file creation
      2 
----> 3 preds_array = np.concatenate(preds, axis=0)  # shape (n_samples, N_CLASSES)
      4 
      5 final_preds = np.zeros_like(preds_array, dtype=bool)

NameError: name 'preds' is not defined
