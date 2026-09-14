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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.10

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
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
seaborn==0.12.2
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
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.4901400601075941

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import warnings

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchvision import models
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from tqdm import tqdm

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score

warnings.filterwarnings("ignore")



## === cell 1
SEED = 8
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)




## === cell 2
def pick_existing(*paths):
    for p in paths:
        if p is not None and os.path.exists(p):
            return p
    return None


base_candidates = [
    "/kaggle/input/aptos2019-blindness-detection",
    "../input/aptos2019-blindness-detection",
    "/kaggle/data/aptos2019-blindness-detection",
    "/kaggle/data/input/aptos2019-blindness-detection",
]

train_path = pick_existing(*[os.path.join(b, "train.csv") for b in base_candidates])
test_path = pick_existing(*[os.path.join(b, "test.csv") for b in base_candidates])
sample_sub_path = pick_existing(
    *[os.path.join(b, "sample_submission.csv") for b in base_candidates]
)
train_img_dir = pick_existing(
    *[os.path.join(b, "train_images") for b in base_candidates]
)
test_img_dir = pick_existing(*[os.path.join(b, "test_images") for b in base_candidates])

if (
    train_path is None
    or test_path is None
    or sample_sub_path is None
    or train_img_dir is None
    or test_img_dir is None
):
    raise FileNotFoundError(
        f"Could not locate required competition files. "
        f"train_path={train_path}, test_path={test_path}, sample_sub_path={sample_sub_path}, "
        f"train_img_dir={train_img_dir}, test_img_dir={test_img_dir}"
    )

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print(train_df.shape, test_df.shape)
train_df.head()



## === cell 3
train_transform = None

img_tf = T.Compose(
    [
        T.Resize((512, 512)),
        T.ToTensor(),  # float32 in [0,1], CHW
    ]
)



## === cell 4
_IMAGE_TENSOR_CACHE = {}  # key: absolute img_path -> torch.FloatTensor(CHW) on CPU


class dataset(Dataset):
    def __init__(self, data_path, img_dir, dt, transform=None, data_df=None):
        self.data_path = data_path
        self.img_dir = img_dir
        self.data = (
            data_df.copy() if data_df is not None else self.__get_data(self.data_path)
        ).reset_index(drop=True)
        self.dt = dt
        self.transform = transform

    def __get_data(self, path):
        return pd.read_csv(path)

    def __len__(self):
        return self.data.shape[0]

    def _load_and_preprocess(self, img_path: str) -> torch.Tensor:
        t = _IMAGE_TENSOR_CACHE.get(img_path, None)
        if t is not None:
            return t

        image = Image.open(img_path).convert("RGB")

        image = TF.adjust_brightness(image, 1.5)
        image = TF.adjust_contrast(image, 1.2)
        image = TF.adjust_sharpness(image, 6.0)

        t = img_tf(image)  # CPU float32 tensor
        _IMAGE_TENSOR_CACHE[img_path] = t
        return t

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, "id_code"]
        img_path = os.path.join(self.img_dir, img_name + ".png")

        image = self._load_and_preprocess(img_path)

        if self.transform is not None:
            image = self.transform(image)

        if self.dt == "train":
            y = int(self.data.loc[idx, "diagnosis"])
            return image, y
        return image

    def show(self, idx):
        item = self.__getitem__(idx)
        img = item[0] if isinstance(item, tuple) else item
        img_np = (img.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype("uint8")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 6))
        plt.imshow(img_np)
        plt.axis("off")
        plt.show()




## === cell 5
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
idx_train, idx_val = next(sss.split(train_df["id_code"], train_df["diagnosis"]))
train_split_df = train_df.iloc[idx_train].reset_index(drop=True)
val_split_df = train_df.iloc[idx_val].reset_index(drop=True)

train_split_path = "train_split.csv"
val_split_path = "val_split.csv"
train_split_df.to_csv(train_split_path, index=False)
val_split_df.to_csv(val_split_path, index=False)

batch = 32  # keep identical batch size as provided
num_workers = min(4, os.cpu_count() or 2)


def seed_worker(worker_id):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)

train_data = dataset(
    train_split_path,
    train_img_dir,
    dt="train",
    transform=train_transform,
    data_df=train_split_df,
)
val_data = dataset(
    val_split_path, train_img_dir, dt="train", transform=None, data_df=val_split_df
)
test_data = dataset(test_path, test_img_dir, dt="test", transform=None, data_df=test_df)

common_loader_kwargs = dict(
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    worker_init_fn=seed_worker if num_workers > 0 else None,
)

train_load = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch,
    shuffle=True,
    generator=g,
    **{k: v for k, v in common_loader_kwargs.items() if v is not None},
)
val_load = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch,
    shuffle=False,
    **{k: v for k, v in common_loader_kwargs.items() if v is not None},
)
test_load = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch,
    shuffle=False,
    **{k: v for k, v in common_loader_kwargs.items() if v is not None},
)

len(train_data), len(val_data), len(test_data)



## === cell 6
model = models.resnet18(weights=None)
model.fc = nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 5),
)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


def evaluate_kappa(m, loader):
    m.eval()
    preds, targs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = m(xb)
            pred = torch.argmax(logits, dim=1)
            preds.append(pred.detach().cpu().numpy())
            targs.append(yb.detach().cpu().numpy())
    preds = np.concatenate(preds)
    targs = np.concatenate(targs)
    return cohen_kappa_score(targs, preds, weights="quadratic")


epochs = 3
best_kappa = -1e9
best_state = None

for ep in range(1, epochs + 1):
    model.train()
    running = 0.0
    for xb, yb in tqdm(train_load, desc=f"Train epoch {ep}/{epochs}"):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running += loss.item() * xb.size(0)

    train_loss = running / len(train_data)
    val_kappa = evaluate_kappa(model, val_load)
    print(f"epoch={ep} train_loss={train_loss:.4f} val_qwk={val_kappa:.4f}")

    if val_kappa > best_kappa:
        best_kappa = val_kappa
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

if best_state is not None:
    model.load_state_dict(best_state, strict=True)



## === cell 7
model.eval()
predict = []
with torch.no_grad():
    for x in tqdm(test_load, desc="Infer"):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).to("cpu").numpy()
        predict.extend(pred.tolist())

print("Preds:", len(predict), predict[:10])
print(pd.Series(predict).value_counts().sort_index())



## === cell 8
sub = pd.read_csv(sample_sub_path)

if len(test_df) != len(predict):
    raise ValueError(
        f"Prediction length {len(predict)} does not match test.csv length {len(test_df)}."
    )

pred_map = dict(
    zip(test_df["id_code"].values.tolist(), np.array(predict, dtype=np.int64).tolist())
)
sub["diagnosis"] = sub["id_code"].map(pred_map).astype(np.int64)

if sub["diagnosis"].isna().any():
    missing = sub.loc[sub["diagnosis"].isna(), "id_code"].head(5).tolist()
    raise ValueError(
        f"Some ids in sample_submission were not predicted (examples: {missing})."
    )

sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
