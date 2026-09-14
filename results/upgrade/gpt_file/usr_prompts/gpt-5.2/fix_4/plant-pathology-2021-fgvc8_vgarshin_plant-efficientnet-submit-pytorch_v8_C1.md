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

# 5. Target score

0.8016620498614973

# 6. Current score

0.272

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I remove the failing `pip install` cell and switch EfficientNet loading to `torchvision.models.efficientnet_b0` so imports work in this Kaggle environment. I also make the runtime robust to missing external model files by using the competition’s `sample_submission.csv` ordering, and ensure `DEVICE` falls back to CPU if CUDA isn’t available. Finally, I fix the TTA/model averaging bug that caused `np.vstack` to crash by accumulating predictions into a properly-shaped numpy array and averaging over (models × TTAs). This run end-to-end and always write a valid `submission.csv` with the correct columns and formatting.'
- What this solution (achieved 0.24507) has done: 'Your current score (0.24507) is far below the target (0.80166), and the main reason is that the notebook is very likely running with random weights because `MDLS_PATH` points to a non-existent dataset (`../input/plant-models-v100`). I make the smallest change that actually loads the provided competition model weights by auto-detecting the correct `plant-models-*` folder under `../input`, keeping the exact same architecture and inference logic. I also load checkpoints with `strict=False` as a safety net against minor key mismatches (this should improve predictions versus random init, not change core logic). Everything else (EffNet, TTA, thresholding, submission formatting) stays the same to preserve evaluation semantics.'
- What this solution (achieved 0.272) has done: 'Your current score is far below the target, so the smallest likely win is to fix the label/index alignment with the saved checkpoints: if `params.json` exists, we must use its `labels_` ordering and also force `df_sub` to follow `sample_submission.csv` ordering (already done) while ensuring we do not collapse predictions to `healthy` when other classes are present. I adjust `get_labels()` to only output `healthy` when it is the *only* predicted class (or none), which matches the competition’s multi-label semantics and avoids wiping out true positives. I also make image normalization match EfficientNet’s expected normalization (ImageNet mean/std) without changing the architecture or inference loop, which usually moves F1 up substantially versus raw [0,1] scaling. All other logic (EffNetB0, checkpoints, TTAs, thresholding, submission format) stays the same.'

# 9. Code solution

## === cell 0
import os
import gc
import json
import time
import cv2
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data.sampler import SequentialSampler

import torchvision
from torchvision import models

torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KAGGLE = True
print(
    "torch:",
    torch.__version__,
    "| torchvision:",
    torchvision.__version__,
    "| device:",
    DEVICE,
)



## === cell 1
TEST = True
VER = "v100"

if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    _input_root = "../input"
    _candidates = []
    if os.path.isdir(_input_root):
        for d in os.listdir(_input_root):
            if d.startswith("plant-models-") and os.path.isdir(
                os.path.join(_input_root, d)
            ):
                _candidates.append(d)
    _candidates = sorted(_candidates)
    if len(_candidates) > 0:
        MDLS_PATH = os.path.join(_input_root, _candidates[-1])
        print("auto-detected MDLS_PATH:", MDLS_PATH)
    else:
        MDLS_PATH = f"../input/plant-models-{VER}"
        print(
            "WARNING: could not auto-detect plant-models-*; using fallback MDLS_PATH:",
            MDLS_PATH,
        )
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TH = 0.4
TTAS = [0, 1, 2, 3]
FOLDS = [0]

IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"
start_time = time.time()

print("DATA_PATH:", DATA_PATH)
print("IMGS_PATH:", IMGS_PATH)
print("MDLS_PATH:", MDLS_PATH)




## === cell 2
def _derive_labels_from_train(train_csv_path: str):
    df = pd.read_csv(train_csv_path)
    uniq = set()
    for s in df["labels"].astype(str).values:
        for t in s.split():
            uniq.add(t)
    uniq = sorted(list(uniq))
    labels_ = {k: i for i, k in enumerate(uniq)}  # str -> int
    labels = {str(i): k for k, i in labels_.items()}  # str(int) -> str
    return labels_, labels


params_path = f"{MDLS_PATH}/params.json"
train_csv_path = f"{DATA_PATH}/train.csv"

if os.path.exists(params_path):
    with open(params_path) as file:
        params = json.load(file)
    LABELS_ = params["labels_"]
    LABELS = params["labels"]
    WORKERS = 2 if KAGGLE else int(params.get("workers", 2))
    print("loaded params from:", params_path)
else:
    LABELS_, LABELS = _derive_labels_from_train(train_csv_path)
    params = {
        "img_size": 512,
        "batch_size": 16,
        "dropout": 0.3,
        "backbone": "efficientnet_b0",
        "workers": 2,
    }
    WORKERS = 2
    print("params.json not found; using derived labels + defaults")

print("n_classes:", len(LABELS_))
print("example class mapping:", list(LABELS_.items())[:5])



## === cell 3
sub_path = f"{DATA_PATH}/sample_submission.csv"
df_sub = pd.read_csv(sub_path)
if "image" not in df_sub.columns or "labels" not in df_sub.columns:
    raise ValueError("sample_submission.csv must have columns: image, labels")

df_sub["labels"] = "healthy"
print("submission template shape:", df_sub.shape)
df_sub.head()




## === cell 4
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
    def __init__(self, df, size, labels, transform=None, tta=0, imgs_path=None):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.labels = labels
        self.transform = transform
        self.tta = int(tta)
        self.imgs_path = imgs_path

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = f"{self.imgs_path}/{img_name}"
        img = cv2.imread(img_path)

        if img is None:
            raise FileNotFoundError(f"Image not found/readable: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std

        if self.labels:
            img = img.transpose(2, 0, 1)
            label = np.zeros(len(self.labels), dtype=np.float32)
            for lbl in str(row.labels).split():
                label[self.labels[lbl]] = 1.0
            return torch.tensor(img), torch.tensor(label)
        else:
            img = flip(img, axis=self.tta)
            img = img.transpose(2, 0, 1)
            return torch.tensor(img.copy())




## === cell 5
class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super().__init__()
        backbone = params.get("backbone", "efficientnet_b0")

        if backbone != "efficientnet_b0":
            backbone = "efficientnet_b0"

        self.enet = models.efficientnet_b0(weights=None)

        nc = self.enet.classifier[-1].in_features
        self.enet.classifier = nn.Identity()

        self.myfc = nn.Sequential(
            nn.Dropout(float(params.get("dropout", 0.3))),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(float(params.get("dropout", 0.3))),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 6
models_list = []
for n_fold in FOLDS:
    model = EffNet(params, out_dim=len(LABELS_))
    path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu")
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("loaded weights:", path)
        if len(missing) > 0:
            print(
                "WARNING: missing keys when loading:",
                missing[:10],
                ("..." if len(missing) > 10 else ""),
            )
        if len(unexpected) > 0:
            print(
                "WARNING: unexpected keys when loading:",
                unexpected[:10],
                ("..." if len(unexpected) > 10 else ""),
            )
        del state_dict
    else:
        print("WARNING: weights not found, using random init:", path)

    model = model.to(DEVICE).float().eval()
    models_list.append(model)

gc.collect()



## === cell 7
datasets, loaders = [], []
for tta in TTAS:
    dataset = PlantDataset(
        df=df_sub,
        size=params["img_size"],
        labels=None,
        transform=None,
        tta=tta,
        imgs_path=IMGS_PATH,
    )
    datasets.append(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(params["batch_size"]),
        sampler=SequentialSampler(dataset),
        num_workers=int(WORKERS),
        pin_memory=(DEVICE.type == "cuda"),
    )
    loaders.append(loader)

len(df_sub), len(loaders)




## === cell 8
def get_labels(row, labels, th):
    idx = [i for i, x in enumerate(row) if x > th]
    if len(idx) == 0:
        return "healthy"
    names = [labels[str(i)] for i in idx]
    non_healthy = [n for n in names if n != "healthy"]
    if len(non_healthy) == 0:
        return "healthy"
    return " ".join(non_healthy)


n = len(df_sub)
c = len(LABELS_)
pred_sum = np.zeros((n, c), dtype=np.float32)
n_runs = 0

with torch.no_grad():
    for mi, model in enumerate(models_list):
        for ti, loader in enumerate(loaders):
            all_preds = []
            for img_data in loader:
                img_data = img_data.to(DEVICE, non_blocking=True)
                preds = model(img_data).sigmoid().detach().cpu().numpy()
                all_preds.append(preds.astype(np.float32))
            all_preds = np.concatenate(all_preds, axis=0)
            if all_preds.shape != (n, c):
                raise ValueError(
                    f"Pred shape mismatch: got {all_preds.shape}, expected {(n, c)}"
                )
            pred_sum += all_preds
            n_runs += 1
            print(f"model {mi} | tta {ti} done")

logits = pred_sum / max(n_runs, 1)
df_sub["labels"] = [get_labels(x, LABELS, TH) for x in logits]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 9
print("value counts:")
print(df_sub.labels.value_counts().head(20))
df_sub.head()



## === cell 10
out_path = "submission.csv"
df_sub[["image", "labels"]].to_csv(out_path, index=False)
print("wrote:", out_path, "| shape:", df_sub.shape)
print(df_sub.head())
