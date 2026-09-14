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

0.818485687903971

# 6. Current score

0.28656

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.29775) has done: 'Implemented minimal fixes to get the pipeline running and generate a valid submission:

1. Added missing imports (`os`, `json`, `numpy`, `pandas`) at the start of cell 0 so parameters load without a `NameError`.
2. Ensured `PRIOR_LOGITS` is defined by executing the corrected cell 0 before later cells.
3. Filtered the test directory listing to include only image files (‑ `.jpg`, `.jpeg`, `.png`) in cell 4, preventing non‑image entries from causing a row‑count mismatch.
4. Kept all core modeling logic unchanged; only the above adjustments resolve runtime errors and guarantee the submission CSV matches the expected number of rows.'
- What this solution (achieved 0.28656) has done: 'I raise the prediction threshold from 0.1 to 0.5 and make the label‑conversion routine fall back to the highest‑probability class when no score exceeds the threshold. This modest change keeps the core model and data pipeline untouched while providing a more selective set of predictions, which should raise the mean F1‑Score toward the target.'

# 9. Code solution

## === cell 0
import os
import json
import time
import gc
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import SequentialSampler
from torchvision import models


def locate_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


KAGGLE = os.getenv("KAGGLE_KERNEL_RUN_TYPE") is not None

data_candidates = [
    "./data/plant-pathology-2021-fgvc8",
    "./input/plant-pathology-2021-fgvc8",
    "./working/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/working/plant-pathology-2021-fgvc8",
]
model_candidates = [
    "./models_v4",
    "./input/plant-models-v4",
    "./working/plant-models-v4",
    "/kaggle/input/plant-models-v4",
    "/kaggle/working/plant-models-v4",
]

DATA_PATH = locate_path(data_candidates)
if DATA_PATH is None:
    raise FileNotFoundError("Could not locate the dataset directory.")
MDLS_PATH = locate_path(model_candidates) or "./models_v4"  # fallback to default

TEST = True  # generate predictions for test set
VER = "v4"
TH = 0.5  # increased threshold to make predictions more selective
TTAS = [0, 1, 2]  # test‑time augmentations (indices)
FOLDS = [3, 4]  # folds to ensemble
IMGS_PATH = (
    os.path.join(DATA_PATH, "test_images")
    if TEST
    else os.path.join(DATA_PATH, "train_images")
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()




## === cell 1
params_path = os.path.join(MDLS_PATH, "params.json")
if os.path.isfile(params_path):
    with open(params_path) as f:
        params = json.load(f)
else:
    params = {
        "img_size": 224,
        "batch_size": 32,
        "dropout": 0.2,
        "backbone": "efficientnet_b0",
        "workers": 2,
        "labels_": None,  # will be filled below
        "labels": None,
    }

train_csv = os.path.join(DATA_PATH, "train.csv")
if not os.path.isfile(train_csv):
    raise FileNotFoundError(f"train.csv not found at expected location: {train_csv}")
train_df = pd.read_csv(train_csv)

all_labels = set()
for lbls in train_df["labels"]:
    all_labels.update(lbls.split())
all_labels = sorted(all_labels)
label_to_idx = {lbl: idx for idx, lbl in enumerate(all_labels)}
idx_to_label = {str(idx): lbl for lbl, idx in label_to_idx.items()}

params["labels_"] = label_to_idx
params["labels"] = idx_to_label

label_counts = {lbl: 0 for lbl in all_labels}
for lbls in train_df["labels"]:
    for lbl in lbls.split():
        label_counts[lbl] += 1
total_imgs = len(train_df)
PRIOR_LOGITS = np.array(
    [label_counts[lbl] / total_imgs for lbl in all_labels], dtype=np.float32
)

LABELS_ = params["labels_"]  # mapping label -> idx
LABELS = params["labels"]  # mapping idx (str) -> label
WORKERS = 2 if KAGGLE else params.get("workers", 2)
print("Parameters loaded / defaulted.")




## === cell 2
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

    def __getitem__(self, index):
        row = self.df.iloc[index]
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
        if self.labels is not None:
            img = img.transpose(2, 0, 1)
            label_vec = np.zeros(len(self.labels), dtype=np.float32)
            for lbl in row["labels"].split():
                label_vec[self.labels[lbl]] = 1.0
            return torch.tensor(img), torch.tensor(label_vec)
        else:
            img = flip(img, axis=self.tta)
            img = img.transpose(2, 0, 1)
            return torch.tensor(img.copy())




## === cell 3
try:
    from efficientnet_pytorch import model as enet

    efficientnet_cls = lambda name: enet.EfficientNet.from_name(name)
except Exception:

    def efficientnet_cls(name):
        backbone_map = {
            "efficientnet_b0": models.efficientnet_b0,
            "efficientnet_b1": models.efficientnet_b1,
            "efficientnet_b2": models.efficientnet_b2,
            "efficientnet_b3": models.efficientnet_b3,
            "efficientnet_b4": models.efficientnet_b4,
        }
        if name not in backbone_map:
            raise ValueError(f"Unsupported backbone: {name}")
        return backbone_map[name](pretrained=True)


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        self.enet = efficientnet_cls(params["backbone"])
        if hasattr(self.enet, "classifier"):
            nc = self.enet.classifier[1].in_features
            self.enet.classifier = nn.Identity()
        else:
            nc = self.enet._fc.in_features
            self.enet._fc = nn.Identity()
        self.myfc = nn.Sequential(
            nn.Dropout(params["dropout"]),
            nn.Linear(nc, nc // 4),
            nn.Dropout(params["dropout"]),
            nn.Linear(nc // 4, out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x


models = []
for n_fold in FOLDS:
    try:
        model = EffNet(params, out_dim=len(LABELS_))
        path = os.path.join(MDLS_PATH, f"model_best_{n_fold}.pth")
        state_dict = torch.load(path, map_location="cpu")
        model.load_state_dict(state_dict)
        model = model.to(DEVICE).eval()
        models.append(model)
        print(f"Loaded model from {path}")
    except Exception as e:
        print(f"Could not load model {n_fold}: {e}")
if not models:
    print("No pretrained models loaded – predictions will use class‑frequency priors.")
gc.collect()




## === cell 4
if not os.path.isdir(IMGS_PATH):
    raise FileNotFoundError(f"Test image directory not found: {IMGS_PATH}")
valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
test_files = sorted(
    [
        f
        for f in os.listdir(IMGS_PATH)
        if os.path.isfile(os.path.join(IMGS_PATH, f))
        and os.path.splitext(f)[1].lower() in valid_exts
    ]
)
df_sub = pd.DataFrame(test_files, columns=["image"])
df_sub["labels"] = "healthy"  # placeholder, will be overwritten if predictions exist
print("Submission template created with", len(df_sub), "records.")




## === cell 5
def get_labels(row, labels_map, th):
    """
    Convert a probability vector into a space‑delimited label string.
    - Prefer classes with probability > th.
    - If none exceed th, fall back to the highest‑probability class.
    - Return "healthy" only when it is the sole selected label.
    """
    try:
        idxs = [i for i, x in enumerate(row) if x > th]
        if idxs:
            names = [labels_map[str(i)] for i in idxs]
            disease_names = [n for n in names if n != "healthy"]
            return "healthy" if not disease_names else " ".join(disease_names)
        top_idx = int(np.argmax(row))
        top_name = labels_map[str(top_idx)]
        return top_name if top_name != "healthy" else "healthy"
    except Exception as e:
        print("Error in get_labels:", e, row)
        return "healthy"


if models:
    loaders = []
    for tta in TTAS:
        ds = PlantDataset(df_sub, size=params["img_size"], labels=None, tta=tta)
        loader = data.DataLoader(
            ds,
            batch_size=params["batch_size"],
            num_workers=WORKERS,
            sampler=SequentialSampler(ds),
        )
        loaders.append(loader)

    all_logits = []
    with torch.no_grad():
        for model in models:
            model_logits = []
            for loader in loaders:
                tta_logits = []
                for batch in loader:
                    batch = batch.to(DEVICE)
                    preds = torch.sigmoid(model(batch)).cpu().numpy()
                    tta_logits.append(preds)
                tta_mean = np.mean(np.vstack(tta_logits), axis=0)
                model_logits.append(tta_mean)
            model_logits = np.mean(np.stack(model_logits, axis=0), axis=0)
            all_logits.append(model_logits)
    logits = np.mean(np.stack(all_logits, axis=0), axis=0)  # (num_images, num_labels)
else:
    logits = np.tile(PRIOR_LOGITS, (len(df_sub), 1))

df_sub["labels"] = [get_labels(row, LABELS, TH) for row in logits]




## === cell 6
print("Label distribution in submission:")
print(df_sub["labels"].value_counts())
out_path = "submission.csv"
df_sub.to_csv(out_path, index=False)
print(f"Submission file written to {out_path}")
elapsed = time.time() - start_time
print(f"Total elapsed time: {int(elapsed//60)}m {int(elapsed%60)}s")
