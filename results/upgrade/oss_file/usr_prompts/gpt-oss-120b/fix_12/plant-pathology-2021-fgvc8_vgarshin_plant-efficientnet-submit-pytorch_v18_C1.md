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

0.30565

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.28656) has done: 'Implemented a safe image‑list filter so only real *.jpg files are used for the submission. This prevents directory entries from being treated as images, eliminates the potential FileNotFoundError during loading, and ensures the output CSV matches the expected number of test rows. No core modeling logic was altered.'
- What this solution (achieved 0.28656) has done: 'I adjust the data and model paths so that the pretrained EfficientNet models are correctly found and loaded when running on Kaggle. By ensuring the models are actually used (instead of the simple fallback that predicts a single common label), the pipeline generate much richer predictions and push the F1 score toward the target. The changes only affect path resolution and preserve all existing modeling logic.'
- What this solution (achieved 0.28656) has done: 'We boost the score by making sure a pretrained model is actually loaded (searching for any *.pth file if the expected naming isn’t found) and by fixing the label conversion so that predicted disease tags aren’t overwritten by the “healthy” shortcut. These minimal tweaks keep the original pipeline intact while allowing real model outputs to be used, which should raise the F1 toward the target.'
- What this solution (achieved 0.27637) has done: 'I make the model‑loading logic more robust so that a pretrained checkpoint is actually used instead of falling back to a single‑label heuristic. The changes (a) broaden the search for *.pth files to any sub‑directory, (b) load the first checkpoint found with `strict=False` to tolerate slight mismatches, and (c) if no checkpoint is found, instantiate an EfficientNet with ImageNet‑pretrained weights so that the pipeline still produces model‑based logits. These tweaks keep the original architecture and training unchanged while giving the prediction step real model outputs, which should raise the F1 score toward the target.'
- What this solution (achieved 0.32193) has done: 'Implemented missing imports, defined helper flags, and added necessary utilities so the script runs end‑to‑end without NameErrors. The core modeling and prediction logic remains unchanged; only the environment setup (torch, cv2, pandas, numpy, os, json, time, torchvision, and related classes) is added. This enables the pipeline to load data, optionally load pretrained checkpoints, generate predictions, and write a valid `submission.csv` complying with the required format.'
- What this solution (achieved 0.2292) has done: 'I lower the default prediction threshold to 0.20 and add standard ImageNet normalization to the test‑time image preprocessing (the model’s expectations), which should raise recall and improve the mean F1 without changing the model architecture or training logic. These minimal tweaks keep the core pipeline intact while moving the score closer to the target.'
- What this solution (achieved 0.30565) has done: 'We lower the per‑class thresholds to the same low value (0.20) used as the default, because the loaded `ths.json` overrides the default and makes the model too conservative, hurting recall and the mean F1. By resetting all thresholds to 0.20 we keep the core modeling unchanged while expected to raise the score toward the target.'

# 9. Code solution

## === cell 0
import os
import json
import time
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import SequentialSampler
import torchvision
import cv2
import numpy as np
import pandas as pd

HAVE_EFFNET = False

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




## === cell 1
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

DEFAULT_THRESH = 0.20

if ths:
    ths = {k: DEFAULT_THRESH for k in ths}
else:
    ths = {}




## === cell 2
image_files = [
    f
    for f in os.listdir(IMGS_PATH)
    if f.lower().endswith(".jpg") and os.path.isfile(os.path.join(IMGS_PATH, f))
]
df_sub = pd.DataFrame(image_files, columns=["image"])
df_sub["labels"] = "healthy"  # placeholder; will be overwritten if a model runs
print("test image list prepared:", df_sub.shape[0], "entries")




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
    def __init__(self, df, size, labels, transform=None, tta=0):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.labels = labels
        self.transform = transform
        self.tta = tta
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

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
        img = (img - self.mean) / self.std
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




## === cell 4
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

if not model_load_success and os.path.isdir(MDLS_PATH):
    import glob

    generic_paths = glob.glob(os.path.join(MDLS_PATH, "**", "*.pth"), recursive=True)
    for gp in generic_paths:
        try:

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
                    self.enet.classifier = nn.Identity()
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
            state_dict = torch.load(gp, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            model.to(DEVICE).eval()
            models.append(model)
            model_load_success = True
            print("loaded generic model from", gp)
            break
        except Exception as e2:
            print(f"generic model load failed for {gp}:", e2)

if not model_load_success:
    print(
        "No pretrained checkpoints found – using ImageNet‑pretrained EfficientNet as a fallback."
    )

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
            self.enet.classifier = nn.Identity()
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
    model.to(DEVICE).eval()
    models.append(model)
    model_load_success = True




## === cell 5
def get_labels(row, labels_dict, thresholds):
    """
    Convert model logits to a space‑delimited label string.
    Uses a lower default threshold (0.20) and falls back to the top‑1 label if none exceed it.
    """
    try:
        idxs = [
            i for i, x in enumerate(row) if x > thresholds.get(str(i), DEFAULT_THRESH)
        ]
        if not idxs:
            idxs = [int(np.argmax(row))]
        txt = [labels_dict[i] for i in idxs]
        return " ".join(txt) if txt else "healthy"
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




## === cell 6
print("label distribution in submission:")
print(df_sub["labels"].value_counts())
df_sub.head()




## === cell 7
submission_path = "submission.csv"
df_sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
