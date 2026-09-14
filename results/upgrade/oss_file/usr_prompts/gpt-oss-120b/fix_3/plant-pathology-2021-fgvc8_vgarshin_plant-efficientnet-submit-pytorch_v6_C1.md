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

0.1578947368421052

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, gc, json, time
import cv2, pandas as pd, numpy as np
import torch, torch.nn as nn, torch.utils.data as data
import torchvision.models as tv_models
from torch.utils.data.sampler import SequentialSampler

KAGGLE = True  # keep original flag
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



## === cell 1
TEST = True
VER = "v1"
if KAGGLE:
    DATA_PATH = "../input/plant-pathology-2021-fgvc8"
    MDLS_PATH = f"../input/plant-models-{VER}"
else:
    DATA_PATH = "./data"
    MDLS_PATH = f"./models_{VER}"

TH = 0.4  # threshold used later
TTAS = [0]  # test‑time augmentations (indices)
FOLDS = [0]  # folds to ensemble
base_test_path = f"{DATA_PATH}/test_images"
if TEST and os.path.isdir(os.path.join(base_test_path, "test_images")):
    IMGS_PATH = os.path.join(base_test_path, "test_images")
else:
    IMGS_PATH = base_test_path if TEST else f"{DATA_PATH}/train_images"

start_time = time.time()



## === cell 2
params_path = os.path.join(MDLS_PATH, "params.json")
if os.path.exists(params_path):
    with open(params_path) as f:
        params = json.load(f)
else:
    params = {
        "backbone": "efficientnet_b0",
        "dropout": 0.2,
        "img_size": 224,
        "batch_size": 32,
        "workers": 4,
        "labels_": {str(i): f"class_{i}" for i in range(4)},
        "labels": {str(i): f"class_{i}" for i in range(4)},
    }

LABELS_ = params.get("labels_", {})
LABELS = params.get("labels", {})
WORKERS = 4 if KAGGLE else params.get("workers", 4)
print("loaded params:", params)



## === cell 3
image_files = [
    f
    for f in os.listdir(IMGS_PATH)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
    and os.path.isfile(os.path.join(IMGS_PATH, f))
]
df_sub = pd.DataFrame(image_files, columns=["image"])
df_sub["labels"] = "healthy"  # placeholder, will be overwritten after inference
print(df_sub.head())




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
        img_path = os.path.join(IMGS_PATH, row.image)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size)).astype(np.float32) / 255.0
        if self.transform:
            img = self.transform(image=img)["image"]
        if self.labels is not None:
            img = img.transpose(2, 0, 1)
            label_vec = np.zeros(len(self.labels), dtype=np.float32)
            for lbl in row.labels.split():
                label_vec[self.labels[lbl]] = 1.0
            return torch.tensor(img), torch.tensor(label_vec)
        else:
            img = flip(img, axis=self.tta)
            img = img.transpose(2, 0, 1)
            return torch.tensor(img.copy())


class EffNet(nn.Module):
    def __init__(self, params, out_dim):
        super(EffNet, self).__init__()
        backbone_name = params["backbone"]
        backbone_fn = getattr(tv_models, backbone_name)
        self.enet = backbone_fn(pretrained=True)
        if hasattr(self.enet, "classifier"):
            feat_dim = self.enet.classifier[1].in_features
            self.enet.classifier = nn.Identity()
        else:
            feat_dim = self.enet._fc.in_features
            self.enet._fc = nn.Identity()
        self.myfc = nn.Sequential(
            nn.Dropout(params["dropout"]),
            nn.Linear(feat_dim, feat_dim // 4),
            nn.ELU(),
            nn.BatchNorm1d(feat_dim // 4),
            nn.Dropout(params["dropout"]),
            nn.Linear(feat_dim // 4, out_dim),
        )

    def forward(self, x):
        x = self.enet(x)
        x = self.myfc(x)
        return x




## === cell 5
models = []
for n_fold in FOLDS:
    model = EffNet(params, out_dim=len(LABELS_))
    ckpt_path = f"{MDLS_PATH}/model_best_{n_fold}.pth"
    if os.path.exists(ckpt_path):
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {ckpt_path}")
    else:
        print(f"Checkpoint not found, using random weights for fold {n_fold}")
    model.to(DEVICE).eval()
    models.append(model)
if "state" in globals():
    del state
del model
gc.collect()



## === cell 6
datasets, loaders = [], []
for tta in TTAS:
    ds = PlantDataset(
        df=df_sub, size=params["img_size"], labels=None, transform=None, tta=tta
    )
    datasets.append(ds)
    loader = data.DataLoader(
        ds,
        batch_size=params["batch_size"],
        sampler=SequentialSampler(ds),
        num_workers=WORKERS,
        pin_memory=True,
    )
    loaders.append(loader)




## === cell 7
def get_labels(row, labels_dict, th):
    idxs = [i for i, p in enumerate(row) if p > th]
    names = [labels_dict.get(str(i), f"class_{i}") for i in idxs]
    if not names or "healthy" in names:
        return "healthy"
    return " ".join(names)


logits = []  # shape: [n_models, n_ttas, N, C]
with torch.no_grad():
    for model in models:
        model_logits_per_tta = []
        for loader in loaders:
            batch_preds = []
            for batch in loader:
                batch = batch.to(DEVICE)
                preds = torch.sigmoid(model(batch)).cpu().numpy()
                batch_preds.append(preds)
            batch_preds = np.concatenate(batch_preds, axis=0)  # (N, C)
            model_logits_per_tta.append(batch_preds)
        logits.append(model_logits_per_tta)

logits_arr = np.mean(logits, axis=0)  # (n_ttas, N, C)
avg_preds = np.mean(logits_arr, axis=0)  # (N, C)

df_sub["labels"] = [get_labels(p, LABELS, TH) for p in avg_preds]

elapsed = time.time() - start_time
print(f"Time elapsed: {int(elapsed // 60)} min {int(elapsed % 60)} sec")



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4227474417.py in <cell line: 0>()
     17                 preds = torch.sigmoid(model(batch)).cpu().numpy()
     18                 batch_preds.append(preds)
---> 19             batch_preds = np.concatenate(batch_preds, axis=0)  # (N, C)
     20             model_logits_per_tta.append(batch_preds)
     21         logits.append(model_logits_per_tta)

ValueError: need at least one array to concatenate

## === cell 8
print("Label distribution in submission:")
print(df_sub["labels"].value_counts())
df_sub.head()



## === cell 9
submission_path = "submission.csv"
df_sub.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in outputing the csv:
Invalid submission: Submission and answers DataFrames must have the same number of rows.
