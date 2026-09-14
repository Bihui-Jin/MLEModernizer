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

0.24507

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'The notebook fails because the pretrained model package (`plant-models-v4/params.json` and fold checkpoints) is not present, causing `params` to be undefined and leaving `all_preds` empty. I make the pipeline robust by (1) locating the official `sample_submission.csv` and using its `image` order, and (2) adding a safe fallback path that writes a valid all-`healthy` submission when model files are missing (instead of crashing). This fix the “row count mismatch” by ensuring we always output exactly the sample submission rows, and it still use the existing ensemble/TTA inference unchanged when model artifacts are available. No training, architecture, or scoring semantics are altered—only guardrails and correct submission alignment are added.'
- What this solution (achieved 0.24507) has done: 'Your current 0.24507 is consistent with the all-`healthy` fallback path being triggered (missing model artifacts) or with a threshold/post-process mismatch that collapses many predictions to `healthy`. To move the score toward the 0.818 target while preserving your inference core, I (1) stop raising when checkpoints are missing and instead load whatever folds exist (so the real model runs when available), (2) fix the label post-processing so `healthy` doesn’t suppress other predicted classes, and (3) tune the decision threshold slightly downward to reduce false negatives (typical for mean-F1 in multilabel) while keeping the same averaging/ensemble/TTA logic. These are minimal changes that directly affect evaluation without changing the model architecture or training approach, and they still always write a valid `submission.csv`.'
- What this solution (achieved 0.24507) has done: 'Your current score (0.24507) is far below the 0.818 target, and it matches the behavior of your “all-healthy” fallback being used because the pretrained artifacts are missing (so no real inference happens). The smallest change that can legitimately move the score toward the target—without changing your model/inference core—is to ensure the notebook actually finds and loads the provided `plant-models-v4` dataset on Kaggle by adding the standard Kaggle input locations to `MDLS_PATH` search. I’m also adding a strict sanity check printout showing whether `params.json` and any fold checkpoints are found, so you can confirm the real model path is being used. Everything else (architecture, TTA/ensemble averaging, thresholding, and submission formatting) is kept the same.'

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

torch.backends.cudnn.benchmark = True


def _detect_kaggle():
    return os.path.exists("/kaggle") or os.path.exists("../input")


KAGGLE = _detect_kaggle()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("KAGGLE:", KAGGLE, "| DEVICE:", DEVICE)



## === cell 1
TEST = True
VER = "v4"

TH = 0.35

TTAS = [0, 1, 2]
FOLDS = [3, 4]


def _pick_existing(paths):
    for p in paths:
        if p is not None and os.path.exists(p):
            return p
    return None


DATA_PATH = _pick_existing(
    [
        "../input/plant-pathology-2021-fgvc8",
        "/kaggle/input/plant-pathology-2021-fgvc8",
        "../kaggle/data/plant-pathology-2021-fgvc8",
        "/kaggle/data/plant-pathology-2021-fgvc8",
        "../input",
        "/kaggle/input",
    ]
)

MDLS_PATH = _pick_existing(
    [
        f"../input/plant-models-{VER}",
        f"/kaggle/input/plant-models-{VER}",
        f"../kaggle/data/plant-models-{VER}",
        f"/kaggle/data/plant-models-{VER}",
        f"../input/plant-pathology-2021-fgvc8/plant-models-{VER}",
        f"/kaggle/input/plant-pathology-2021-fgvc8/plant-models-{VER}",
    ]
)

if MDLS_PATH is None:
    MDLS_PATH = f"./plant-models-{VER}"

if TEST:
    IMGS_PATH = _pick_existing(
        [
            f"{DATA_PATH}/test_images",
            "../kaggle/data/test_images",
            "/kaggle/data/test_images",
            "../input/test_images",
            "/kaggle/input/test_images",
        ]
    )
else:
    IMGS_PATH = _pick_existing(
        [
            f"{DATA_PATH}/train_images",
            "../kaggle/data/train_images",
            "/kaggle/data/train_images",
        ]
    )

assert IMGS_PATH is not None and os.path.exists(
    IMGS_PATH
), f"Could not find images folder. DATA_PATH={DATA_PATH}"
print("DATA_PATH:", DATA_PATH)
print("MDLS_PATH:", MDLS_PATH)
print("IMGS_PATH:", IMGS_PATH)

start_time = time.time()



## === cell 2
sample_sub_path = _pick_existing(
    [
        f"{DATA_PATH}/sample_submission.csv",
        "../input/plant-pathology-2021-fgvc8/sample_submission.csv",
        "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv",
        "../kaggle/data/plant-pathology-2021-fgvc8/sample_submission.csv",
        "../input/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "../kaggle/data/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)
if sample_sub_path is None:
    raise FileNotFoundError("Could not locate sample_submission.csv in known paths.")

df_sub = pd.read_csv(sample_sub_path)
if "image" not in df_sub.columns or "labels" not in df_sub.columns:
    raise ValueError(
        f"sample_submission.csv has unexpected columns: {df_sub.columns.tolist()}"
    )

df_sub["labels"] = "healthy"
print("sample_submission:", sample_sub_path, "| n_rows:", len(df_sub))
df_sub.head()



## === cell 3
params = None
LABELS_ = None
LABELS = None
params_path = os.path.join(MDLS_PATH, "params.json")

print("checking params_path:", params_path, "| exists:", os.path.exists(params_path))

if os.path.exists(params_path):
    with open(params_path) as file:
        params = json.load(file)

    LABELS_ = params["labels_"]  # list-like (index -> label string)
    LABELS = params[
        "labels"
    ]  # dict-like; expected "index as str" -> label string for inference
    WORKERS = 2 if KAGGLE else int(params.get("workers", 2))
    print("loaded params from:", params_path)
    print("loaded params keys:", sorted(list(params.keys())))
    print(
        "img_size:",
        params.get("img_size"),
        "| batch_size:",
        params.get("batch_size"),
        "| backbone:",
        params.get("backbone"),
    )
else:
    print(
        f"WARNING: Missing {params_path}. "
        "Will create a valid fallback submission with all 'healthy' labels."
    )




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


class PlantDataset(data.Dataset):
    def __init__(self, df, size, labels, transform=None, tta=0):
        self.df = df.reset_index(drop=True)
        self.size = int(size)
        self.labels = labels
        self.transform = transform
        self.tta = int(tta)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_name = row.image
        img_path = f"{IMGS_PATH}/{img_name}"
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")
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
            img = img.transpose(2, 0, 1)
            return torch.tensor(img.copy())




## === cell 5
class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        backbone = str(params.get("backbone", "efficientnet_b0"))

        tv_name = backbone.replace("efficientnet-", "efficientnet_").replace("-", "_")
        if not hasattr(torchvision.models, tv_name):
            tv_name = "efficientnet_b0"

        weights = None  # inference with loaded weights doesn't require pretrained weights here
        enet = getattr(torchvision.models, tv_name)(weights=weights)
        self.enet = enet.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        if hasattr(enet, "classifier") and isinstance(enet.classifier, nn.Sequential):
            fc = [m for m in enet.classifier if isinstance(m, nn.Linear)]
            nc = fc[0].in_features if len(fc) else 1280
        else:
            nc = 1280

        self.myfc = nn.Sequential(
            nn.Dropout(float(params["dropout"])),
            nn.Linear(nc, int(nc / 4)),
            nn.Dropout(float(params["dropout"])),
            nn.Linear(int(nc / 4), out_dim),
        )

    def extract(self, x):
        x = self.enet(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x




## === cell 6
models = []
loaded_paths = []
if params is not None:
    for n_fold in FOLDS:
        path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
        if not os.path.exists(path):
            print(f"WARNING: Missing model checkpoint (skipping): {path}")
            continue

        model = EffNet(params, out_dim=len(LABELS_))
        state_dict = torch.load(path, map_location="cpu")

        try:
            model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            new_sd = {}
            for k, v in state_dict.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                new_sd[nk] = v
            model.load_state_dict(new_sd, strict=False)

        model.float()
        model.eval()
        model.to(DEVICE)
        models.append(model)
        loaded_paths.append(path)
        print("loaded:", path)

    if len(models) == 0:
        print(
            "WARNING: params.json found but no checkpoints loaded; will use all-healthy fallback."
        )
    else:
        print("n_models_loaded:", len(models))
    del state_dict
    gc.collect()
else:
    print(
        "No params/model checkpoints found; skipping inference and using all-healthy fallback."
    )



## === cell 7
datasets, loaders = [], []
if params is not None and len(models) > 0:
    for tta in TTAS:
        dataset = PlantDataset(
            df=df_sub,
            size=params["img_size"],
            labels=None,
            transform=None,
            tta=tta,
        )
        datasets.append(dataset)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(params["batch_size"]),
            sampler=SequentialSampler(dataset),
            num_workers=WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        loaders.append(loader)

    print("n_loaders:", len(loaders), "| n_images:", len(df_sub))




## === cell 8
def get_labels(row, labels, th):
    idxs = [i for i, x in enumerate(row) if x > th]
    out = [labels[str(i)] for i in idxs]  # labels maps "index as str" -> class name
    out = [x for x in out if x != "healthy"]
    return "healthy" if len(out) == 0 else " ".join(out)


if params is not None and len(models) > 0 and len(loaders) > 0:
    all_preds = []
    with torch.no_grad():
        for i, model in enumerate(models):
            for j, loader in enumerate(loaders):
                preds_batches = []
                for img_data in loader:
                    img_data = img_data.to(DEVICE, non_blocking=True).float()
                    probs = model(img_data).sigmoid().detach().cpu().numpy()
                    preds_batches.append(probs)
                preds = np.concatenate(preds_batches, axis=0)
                all_preds.append(preds)
                print(f"model {i} | tta {j} -> preds shape {preds.shape}")

    logits = np.mean(np.stack(all_preds, axis=0), axis=0)
    df_sub["labels"] = [get_labels(x, LABELS, TH) for x in logits]

elapsed_time = time.time() - start_time
print(f"time elapsed: {elapsed_time // 60:.0f} min {elapsed_time % 60:.0f} sec")



## === cell 9
print("value counts:")
print(df_sub.labels.value_counts().head(20))
df_sub.head()



## === cell 10
sub_path = "submission.csv"
df_sub = df_sub[["image", "labels"]].copy()
df_sub.to_csv(sub_path, index=False)
print("wrote:", sub_path, "| rows:", len(df_sub), "| cols:", df_sub.columns.tolist())
print(df_sub.head())
