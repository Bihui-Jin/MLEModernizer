# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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


base1 = "/kaggle/input/aptos2019-blindness-detection"
base2 = "../input/aptos2019-blindness-detection"

train_path = pick_existing(
    os.path.join(base1, "train.csv"), os.path.join(base2, "train.csv")
)
test_path = pick_existing(
    os.path.join(base1, "test.csv"), os.path.join(base2, "test.csv")
)
sample_sub_path = pick_existing(
    os.path.join(base1, "sample_submission.csv"),
    os.path.join(base2, "sample_submission.csv"),
)
train_img_dir = pick_existing(
    os.path.join(base1, "train_images"), os.path.join(base2, "train_images")
)
test_img_dir = pick_existing(
    os.path.join(base1, "test_images"), os.path.join(base2, "test_images")
)

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
class dataset(Dataset):
    def __init__(self, data_path, img_dir, dt, transform=None):
        self.data_path = data_path
        self.img_dir = img_dir
        self.data = self.__get_data(self.data_path).reset_index(drop=True)
        self.dt = dt
        self.transform = transform

    def __get_data(self, path):
        return pd.read_csv(path)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        img_name = self.data.loc[idx, "id_code"]
        img_path = os.path.join(self.img_dir, img_name + ".png")

        image = Image.open(img_path).convert("RGB")

        image = TF.adjust_brightness(image, 1.5)
        image = TF.adjust_contrast(image, 1.2)
        image = TF.adjust_sharpness(image, 6.0)

        image = img_tf(image)

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

batch = 32  # reduce batch a bit to fit 512x512 on most Kaggle GPUs safely
num_workers = 2

train_data = dataset(
    train_split_path, train_img_dir, dt="train", transform=train_transform
)
val_data = dataset(val_split_path, train_img_dir, dt="train", transform=None)
test_data = dataset(test_path, test_img_dir, dt="test", transform=None)

train_load = torch.utils.data.DataLoader(
    train_data,
    batch_size=batch,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
)
val_load = torch.utils.data.DataLoader(
    val_data,
    batch_size=batch,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
)
test_load = torch.utils.data.DataLoader(
    test_data,
    batch_size=batch,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=(device == "cuda"),
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
if len(sub) != len(predict):
    raise ValueError(
        f"Prediction length {len(predict)} does not match sample_submission length {len(sub)}."
    )

sub["diagnosis"] = np.array(predict, dtype=np.int64)
sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same id_codes as answers
