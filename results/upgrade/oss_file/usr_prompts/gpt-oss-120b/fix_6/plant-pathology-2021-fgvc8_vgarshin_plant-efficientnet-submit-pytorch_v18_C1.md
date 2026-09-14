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

0.7949030470914142

# 6. Current score

0.28656

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28656) has done: 'Implemented a safe image‑list filter so only real *.jpg files are used for the submission. This prevents directory entries from being treated as images, eliminates the potential FileNotFoundError during loading, and ensures the output CSV matches the expected number of test rows. No core modeling logic was altered.'
- What this solution (achieved 0.28656) has done: 'I adjust the data and model paths so that the pretrained EfficientNet models are correctly found and loaded when running on Kaggle. By ensuring the models are actually used (instead of the simple fallback that predicts a single common label), the pipeline generate much richer predictions and push the F1 score toward the target. The changes only affect path resolution and preserve all existing modeling logic.'

# 9. Code solution

## === cell 0
import os
import gc
import sys
import json
import time
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torch.utils.data.sampler import SequentialSampler

try:
    from efficientnet_pytorch import model as enet

    HAVE_EFFNET = True
except ImportError:
    HAVE_EFFNET = False




## === cell 1
KAGGLE = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if KAGGLE:
    possible_data_paths = [
        "/kaggle/input/plant-pathology-2021-fgvc8",
        "../input/plant-pathology-2021-fgvc8",
        "./data/plant-pathology-2021-fgvc8",
    ]
    DATA_PATH = next(
        (p for p in possible_data_paths if os.path.isdir(p)),
        "../input/plant-pathology-2021-fgvc8",
    )

    possible_model_paths = [
        "/kaggle/input/plant-models-v102",
        "../input/plant-models-v102",
        "./models_v102",
    ]
    MDLS_PATH = next(
        (p for p in possible_model_paths if os.path.isdir(p)),
        "../input/plant-models-v102",
    )
else:
    DATA_PATH = "./data"
    MDLS_PATH = "./models_v102"

TEST = True  # we are generating predictions for the test set
VER = "v102"
TTAS = [0, 1, 2]  # test‑time augmentations (not used in fallback)
FOLDS = [0]  # model folds (not used in fallback)
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()




## === cell 2
try:
    with open(f"{MDLS_PATH}/params.json") as f:
        params = json.load(f)
except Exception:
    params = {
        "labels_": {},  # will be populated from train.csv later
        "labels": {},
        "workers": 2,
        "img_size": 224,
        "batch_size": 32,
        "dropout": 0.2,
        "backbone": "efficientnet-b0",
    }

try:
    with open(f"{MDLS_PATH}/ths.json") as f:
        ths = json.load(f)
except Exception:
    ths = {}

if not params["labels_"] or not params["labels"]:
    train_df = pd.read_csv(f"{DATA_PATH}/train.csv")
    unique_labels = set()
    for lbls in train_df["labels"].astype(str):
        unique_labels.update(lbls.split())
    unique_labels = sorted(unique_labels)
    LABELS_ = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    LABELS = {idx: lbl for lbl, idx in LABELS_.items()}
    params["labels_"] = LABELS_
    params["labels"] = LABELS
else:
    LABELS_ = params["labels_"]
    LABELS = params["labels"]
    train_df = pd.read_csv(f"{DATA_PATH}/train.csv")  # needed for fallback frequency

if not ths:
    ths = {str(i): 0.5 for i in range(len(LABELS_))}

WORKERS = 2 if KAGGLE else params.get("workers", 2)
print("params loaded (or defaults used).")




## === cell 3
image_files = [
    f
    for f in os.listdir(IMGS_PATH)
    if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(IMGS_PATH, f))
]
df_sub = pd.DataFrame(image_files, columns=["image"])
df_sub["labels"] = "healthy"  # placeholder; will be overwritten if a model runs
print("test image list prepared:", df_sub.shape[0], "entries")




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
    def __init__(self, df, size, labels, transform=None, tta=0):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.labels = labels
        self.transform = transform
        self.tta = tta

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row.image
        img_path = f"{IMGS_PATH}/{img_name}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        if self.transform:
            img = self.transform(image=img)["image"]
        if self.labels:
            img = img.transpose(2, 0, 1)
            label = np.zeros(len(self.labels), dtype=np.float32)
            for lbl in row.labels.split():
                label[self.labels[lbl]] = 1.0
            return torch.tensor(img), torch.tensor(label)
        else:
            img = flip(img, axis=self.tta)
            img = img.transpose(2, 0, 1)
            return torch.tensor(img.copy())




## === cell 5
models = []
model_load_success = False
if os.path.isdir(MDLS_PATH):
    try:
        for n_fold in FOLDS:
            if params["backbone"] == "resnext":
                continue
            else:

                class EffNet(nn.Module):
                    def __init__(self, params, out_dim):
                        super().__init__()
                        if HAVE_EFFNET:
                            self.enet = enet.EfficientNet.from_name(params["backbone"])
                        else:
                            backbone_map = {
                                "efficientnet-b0": torchvision.models.efficientnet_b0,
                                "efficientnet-b1": torchvision.models.efficientnet_b1,
                                "efficientnet-b2": torchvision.models.efficientnet_b2,
                                "efficientnet-b3": torchvision.models.efficientnet_b3,
                                "efficientnet-b4": torchvision.models.efficientnet_b4,
                                "efficientnet-b5": torchvision.models.efficientnet_b5,
                                "efficientnet-b6": torchvision.models.efficientnet_b6,
                                "efficientnet-b7": torchvision.models.efficientnet_b7,
                            }
                            fn = backbone_map.get(
                                params["backbone"], torchvision.models.efficientnet_b0
                            )
                            self.enet = fn(pretrained=True)
                        nc = (
                            self.enet.classifier[1].in_features
                            if hasattr(self.enet, "classifier")
                            else self.enet._fc.in_features
                        )
                        self.enet.classifier = (
                            nn.Identity()
                            if hasattr(self.enet, "classifier")
                            else nn.Identity()
                        )
                        self.myfc = nn.Sequential(
                            nn.Dropout(params["dropout"]),
                            nn.Linear(nc, nc // 4),
                            nn.ReLU(),
                            nn.Dropout(params["dropout"]),
                            nn.Linear(nc // 4, out_dim),
                        )

                    def forward(self, x):
                        x = self.enet(x)
                        x = self.myfc(x)
                        return x

                model = EffNet(params=params, out_dim=len(LABELS_))
                path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
                state_dict = torch.load(path, map_location="cpu")
                model.load_state_dict(state_dict)
                model.to(DEVICE).eval()
                models.append(model)
                print("loaded model from", path)
        if models:
            model_load_success = True
    except Exception as e:
        print("model loading failed:", e)

if not model_load_success:
    print(
        "No pretrained models available – using fallback frequency‑based predictions."
    )
    label_counts = {}
    for lbls in train_df["labels"].astype(str):
        for lbl in lbls.split():
            if lbl != "healthy":
                label_counts[lbl] = label_counts.get(lbl, 0) + 1
    if label_counts:
        most_common = max(label_counts, key=label_counts.get)
        df_sub["labels"] = most_common
    else:
        df_sub["labels"] = "healthy"




## === cell 6
def get_labels(row, labels_dict, thresholds):
    """
    Convert model logits to a space‑delimited label string.
    """
    try:
        idxs = [i for i, x in enumerate(row) if x > thresholds.get(str(i), 0.5)]
        txt = [labels_dict[i] for i in idxs]  # use integer key
        if not txt or "healthy" in txt:
            return "healthy"
        return " ".join(txt)
    except Exception as e:
        print("error in get_labels:", e)
        return "healthy"


if model_load_success:
    datasets, loaders = [], []
    for tta in TTAS:
        ds = PlantDataset(
            df=df_sub, size=params["img_size"], labels=None, transform=None, tta=tta
        )
        datasets.append(ds)
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=params["batch_size"],
            sampler=SequentialSampler(ds),
            num_workers=WORKERS,
            pin_memory=True,
        )
        loaders.append(loader)

    all_logits = []
    with torch.no_grad():
        for model in models:
            tta_logits = []
            for loader in loaders:
                batch_logits = []
                for imgs in loader:
                    imgs = imgs.to(DEVICE)
                    preds = torch.sigmoid(model(imgs)).cpu().numpy()
                    batch_logits.append(preds)
                tta_logits.append(np.concatenate(batch_logits, axis=0))
            avg_tta = np.mean(tta_logits, axis=0)
            all_logits.append(avg_tta)

    logits = np.mean(all_logits, axis=0)
    df_sub["labels"] = [get_labels(row, LABELS, ths) for row in logits]

elapsed = time.time() - start_time
print(f"pipeline finished in {int(elapsed // 60)}m {int(elapsed % 60)}s")




## === cell 7
print("label distribution in submission:")
print(df_sub["labels"].value_counts())
df_sub.head()




## === cell 8
submission_path = "submission.csv"
df_sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
