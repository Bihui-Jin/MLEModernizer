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

0.06777

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.06777) has done: 'I remove the failing `pip install` and instead use the already-installed `torchvision`/`torch` stack, and patch the EfficientNet import so the code runs whether `efficientnet_pytorch` exists or not. I also fix the `KAGGLE`/path variables so downstream cells don’t crash, add safe CPU fallback if CUDA isn’t available, and make the model-weight loading robust (if the external model folder isn’t present, it still produce a valid submission). Finally, I fix the TTA/model ensembling aggregation bug that caused `np.vstack` to receive a scalar, ensuring logits are stacked consistently and labels are generated correctly for the submission CSV.'
- What this solution (achieved 0.24507) has done: 'Your score is extremely low mainly because the notebook is likely not loading the intended trained weights (it silently falls back to randomly initialized models), and because the label-index-to-name mapping inside `get_labels()` can be wrong if `LABELS` is `{name: index}` (your current code tries to index it as `{index: name}`). I make two minimal, directly score-relevant fixes: (1) robustly locate `MDLS_PATH` by searching the input directory for a matching `model_best_*.pth` folder so weights actually load, and (2) fix `get_labels()` to always use an explicit `idx->name` inverse map, preserving your exact thresholding and “healthy” fallback semantics. These changes preserve the same model architectures, inference loop, TTA, and submission formatting, but should move the score sharply upward toward your target.'
- What this solution (achieved 0.06777) has done: 'The current score gap is large, so the most likely score-killer is that your inference normalization does not match what the saved weights were trained with (you currently feed raw `[0,1]` RGB without ImageNet mean/std). I make a minimal, inference-only fix: apply torchvision ImageNet normalization inside the test dataset right before converting to tensor, preserving your exact model architectures, TTA, thresholding, and ensembling logic. I also make the test-image filtering use the sample submission’s order (no change in content) to avoid any accidental ordering mismatch, while still producing a valid `submission.csv`. These changes are directly score-relevant and should move performance upward toward your target without changing the core approach.'

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
from torch.utils.data.sampler import SequentialSampler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KAGGLE = True

try:
    from efficientnet_pytorch import model as enet  # type: ignore
except Exception:
    from torchvision.models import (
        efficientnet_b0,
        efficientnet_b1,
        efficientnet_b2,
        efficientnet_b3,
        efficientnet_b4,
        efficientnet_b5,
        efficientnet_b6,
        efficientnet_b7,
    )

    class _TorchvisionEfficientNetWrapper(nn.Module):
        def __init__(self, name: str):
            super().__init__()
            name = name.lower()
            mapping = {
                "efficientnet-b0": efficientnet_b0,
                "efficientnet-b1": efficientnet_b1,
                "efficientnet-b2": efficientnet_b2,
                "efficientnet-b3": efficientnet_b3,
                "efficientnet-b4": efficientnet_b4,
                "efficientnet-b5": efficientnet_b5,
                "efficientnet-b6": efficientnet_b6,
                "efficientnet-b7": efficientnet_b7,
            }
            if name not in mapping:
                raise ValueError(
                    f"Unsupported EfficientNet backbone without efficientnet_pytorch: {name}"
                )
            self.net = mapping[name](weights=None)
            in_features = self.net.classifier[-1].in_features
            self._fc = nn.Linear(in_features, 1000)
            self.net.classifier = nn.Identity()

        def forward(self, x):
            return self.net(x)

    class enet:  # noqa: N801
        class EfficientNet:
            @staticmethod
            def from_name(name: str):
                return _TorchvisionEfficientNetWrapper(name)




## === cell 1
TEST = True
VER = "v102"

if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TTAS = [0, 1, 2]
FOLDS = [0]
IMGS_PATH = f"{DATA_PATH}/test_images" if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()



## === cell 2
default_labels = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]
default_LABELS_ = list(range(len(default_labels)))
default_LABELS = {lbl: i for i, lbl in enumerate(default_labels)}

params = {
    "labels_": default_LABELS_,
    "labels": default_LABELS,
    "workers": 2,
    "img_size": 512,
    "batch_size": 16,
    "dropout": 0.2,
    "backbone": "resnext",  # will work without extra packages
}
ths = {str(i): 0.5 for i in range(len(default_LABELS_))}


def _find_models_path(initial_path: str, folds):
    if os.path.exists(initial_path) and any(
        os.path.exists(os.path.join(initial_path, f"model_best_{f}.pth")) for f in folds
    ):
        return initial_path

    if not KAGGLE:
        return initial_path

    base = "../input"
    if not os.path.isdir(base):
        return initial_path

    best_path = None
    best_hits = -1
    for root, dirs, files in os.walk(base):
        hits = 0
        for f in folds:
            if f"model_best_{f}.pth" in files:
                hits += 1
        if hits > best_hits:
            best_hits = hits
            best_path = root
        if hits == len(folds) and hits > 0:
            best_path = root
            break

    return best_path if (best_path is not None and best_hits > 0) else initial_path


MDLS_PATH = _find_models_path(MDLS_PATH, FOLDS)

if os.path.exists(f"{MDLS_PATH}/params.json"):
    with open(f"{MDLS_PATH}/params.json") as file:
        params.update(json.load(file))

LABELS_ = params["labels_"]
LABELS = params["labels"]
WORKERS = 2 if KAGGLE else int(params.get("workers", 2))

if os.path.exists(f"{MDLS_PATH}/ths.json"):
    with open(f"{MDLS_PATH}/ths.json") as file:
        ths = json.load(file)

print("DEVICE:", DEVICE)
print("DATA_PATH:", DATA_PATH)
print("IMGS_PATH:", IMGS_PATH)
print("MDLS_PATH:", MDLS_PATH)
print("loaded params keys:", list(params.keys()))
print("num labels:", len(LABELS_))
print("thresholds loaded:", len(ths))



## === cell 3
sample_path = f"{DATA_PATH}/sample_submission.csv"
df_sub = pd.read_csv(sample_path)

if os.path.isdir(IMGS_PATH):
    existing = set(os.listdir(IMGS_PATH))
    df_sub = df_sub.loc[df_sub["image"].isin(existing)].reset_index(drop=True)

df_sub["labels"] = "healthy"
print(df_sub.head())




## === cell 4
def flip(img, axis=0):
    if axis == 1:
        return img[
            ::-1,
            :,
        ]
    elif axis == 2:
        return img[
            :,
            ::-1,
        ]
    elif axis == 3:
        return img[
            ::-1,
            ::-1,
        ]
    else:
        return img


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PlantDataset(data.Dataset):
    def __init__(self, df, size, labels, transform=None, tta=0, normalize=True):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.labels = labels
        self.transform = transform
        self.tta = tta
        self.normalize = bool(normalize)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = f"{IMGS_PATH}/{img_name}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"no img file read: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0

        if self.transform is not None:
            img = self.transform(image=img)["image"]

        if self.labels:
            img = img.transpose(2, 0, 1)
            label = np.zeros(len(self.labels)).astype(np.float32)
            for lbl in row.labels.split():
                label[self.labels[lbl]] = 1
            return torch.tensor(img), torch.tensor(label)
        else:
            img = flip(img, axis=self.tta)
            if self.normalize:
                img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
            img = img.transpose(2, 0, 1)
            return torch.tensor(img.copy())


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        self.enet = enet.EfficientNet.from_name(params["backbone"])
        nc = self.enet._fc.in_features
        self.enet._fc = nn.Identity()
        self.myfc = nn.Sequential(
            nn.Dropout(params["dropout"]),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x


class ResNext(nn.Module):
    def __init__(self, params, out_dim):
        super(ResNext, self).__init__()
        self.rsnxt = torchvision.models.resnext50_32x4d(weights=None)
        nc = self.rsnxt.fc.in_features
        self.rsnxt.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(nc, int(nc / 4)),
            nn.ReLU(),
            nn.Dropout(params["dropout"]),
            nn.Linear(int(nc / 4), out_dim),
        )
        if torch.cuda.is_available():
            self.rsnxt = nn.DataParallel(self.rsnxt)

    def forward(self, x):
        return self.rsnxt(x)




## === cell 5
models = []
weights_missing = False

for n_fold in FOLDS:
    if params["backbone"] == "resnext":
        model = ResNext(params=params, out_dim=len(LABELS_))
    else:
        model = EffNet(params=params, out_dim=len(LABELS_))

    path = "{}/model_best_{}.pth".format(MDLS_PATH, n_fold)
    if os.path.exists(path):
        state_dict = torch.load(path, map_location="cpu")
        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            new_sd = {}
            if any(k.startswith("module.") for k in state_dict.keys()):
                for k, v in state_dict.items():
                    new_sd[k.replace("module.", "")] = v
            else:
                for k, v in state_dict.items():
                    new_sd["module." + k] = v
            model.load_state_dict(new_sd, strict=False)
        print("loaded:", path)
        del state_dict
    else:
        weights_missing = True
        print("WARNING: model weights not found:", path)

    model.float()
    model.eval()
    model.to(DEVICE)
    models.append(model)

gc.collect()



## === cell 6
datasets, loaders = [], []
for tta in TTAS:
    dataset = PlantDataset(
        df=df_sub,
        size=params["img_size"],
        labels=None,
        transform=None,
        tta=tta,
        normalize=True,  # score-relevant: enable normalization for inference
    )
    datasets.append(dataset)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(params["batch_size"]),
        sampler=SequentialSampler(dataset),
        num_workers=int(WORKERS),
        pin_memory=torch.cuda.is_available(),
    )
    loaders.append(loader)



## === cell 7
if isinstance(LABELS, dict) and len(LABELS) > 0:
    if all(isinstance(v, int) for v in LABELS.values()):
        IDX2NAME = {int(v): str(k) for k, v in LABELS.items()}
    else:
        IDX2NAME = {int(k): str(v) for k, v in LABELS.items()}
else:
    IDX2NAME = {i: str(i) for i in range(len(LABELS_))}


def get_labels(row, idx2name, ths):
    idxs = [i for i, x in enumerate(row) if float(x) > float(ths.get(str(i), 0.5))]
    names = [idx2name.get(i, None) for i in idxs]
    names = [n for n in names if n is not None]
    out = "healthy" if ("healthy" in names or len(names) == 0) else " ".join(names)
    return out


all_preds = []

with torch.no_grad():
    for i, model in enumerate(models):
        tta_preds = []
        for j, loader in enumerate(loaders):
            batch_preds = []
            for img_data in loader:
                img_data = img_data.to(DEVICE, non_blocking=True)
                preds = model(img_data).sigmoid().detach().cpu().numpy()  # (bs, C)
                batch_preds.append(preds)
            tta_pred = np.vstack(batch_preds)  # (N, C)
            tta_preds.append(tta_pred)
            print(f"model {i} | loader {j} -> done, shape={tta_pred.shape}")
        tta_preds = np.mean(np.stack(tta_preds, axis=0), axis=0)  # (N, C)
        all_preds.append(tta_preds)

logits = np.mean(np.stack(all_preds, axis=0), axis=0)  # (N, C)

df_sub["labels"] = [get_labels(x, IDX2NAME, ths) for x in logits]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 8
print("value counts:")
print(df_sub.labels.value_counts().head(20))
print(df_sub.head())



## === cell 9
out_path = "submission.csv"
df_sub[["image", "labels"]].to_csv(out_path, index=False)
print("wrote:", out_path, "rows:", len(df_sub))

if weights_missing:
    print(
        "WARNING: One or more model weight files were missing; submission is valid but likely low-scoring."
    )
else:
    print("All requested fold weights found and loaded.")
