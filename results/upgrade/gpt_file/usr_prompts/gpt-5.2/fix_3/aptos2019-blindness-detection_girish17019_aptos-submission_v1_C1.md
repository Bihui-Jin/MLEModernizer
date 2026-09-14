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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.437843922186533

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.0) has done: 'The immediate failure is that the notebook expects pretrained weights at `../input/model-weights/model.bin`, but that file doesn’t exist in your provided dataset, so inference never runs and `labels` is undefined. To make the pipeline run end-to-end and generate a valid `submission.csv`, I keep the same ResNet50 architecture/head but add a safe fallback: if weights are missing, run the untrained model and still write a properly formatted submission. I also fix a couple of runtime/logic issues in the dataset (channel order, tensor dtype/shape, and the invalid `show()` method signature) to ensure the model receives `(N,3,H,W)` float tensors. These fixes are execution/stability oriented; without the missing weights data, the score cannot be meaningfully improved beyond producing a valid submission.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import random
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import albumentations as A
from torchvision.models import resnet50



## === cell 1
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n Device : {device.upper()}")



## === cell 2
TEST_PATH = "../input/aptos2019-blindness-detection/test.csv"
TEST_IMG = "../input/aptos2019-blindness-detection/test_images"
SAMPLE_SUB_PATH = "../input/aptos2019-blindness-detection/sample_submission.csv"

TRAIN_PATH = "../input/aptos2019-blindness-detection/train.csv"
TRAIN_IMG = "../input/aptos2019-blindness-detection/train_images"




## === cell 3
class AptosDataset(Dataset):
    """
    Minimal extensions:
    - Support train mode (with labels) while keeping the same image loading/core behavior.
    - Apply optional normalization (score-relevant for ResNet training/inference).
    """

    def __init__(
        self,
        data_path,
        img_dir,
        name,
        transforms=None,
        resize=(512, 512),
        return_label=False,
    ):
        self.data_path = data_path
        self.img_dir = img_dir
        self.resize = resize
        self.transforms = transforms
        self.df = pd.read_csv(self.data_path)
        self.name = name
        self.return_label = return_label

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        img_id = self.df.loc[idx, "id_code"]
        img_path = os.path.join(self.img_dir, img_id + ".png")

        img = cv.imread(img_path, cv.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        if self.resize:
            img = cv.resize(img, self.resize, interpolation=cv.INTER_AREA)

        img = img.astype(np.float32) / 255.0

        if self.transforms:
            transformed = self.transforms(image=img)
            img = transformed["image"]

        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float()

        if self.return_label:
            y = int(self.df.loc[idx, "diagnosis"])
            y = torch.tensor(y, dtype=torch.long)
            return img, y

        return img

    def show(self, idx):
        if self.return_label:
            img_vector, y = self.__getitem__(idx)
            title = f'{self.df.loc[idx, "id_code"]} / y={int(y)}'
        else:
            img_vector = self.__getitem__(idx)
            title = f'{self.df.loc[idx, "id_code"]}'
        img_np = img_vector.permute(1, 2, 0).detach().cpu().numpy()
        img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
        plt.imshow(img_np)
        plt.title(title)
        plt.axis("off")
        plt.show()




## === cell 4
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

train_tfms = A.Compose(
    [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0),
    ]
)
test_tfms = A.Compose(
    [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=1.0),
    ]
)



## === cell 5
from sklearn.model_selection import train_test_split

BATCH_SIZE = 16
IMG_DIM = 512

train_df = pd.read_csv(TRAIN_PATH)
tr_idx, va_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["diagnosis"].values,
)

train_split_path = "train_split.csv"
val_split_path = "val_split.csv"
train_df.iloc[tr_idx].to_csv(train_split_path, index=False)
train_df.iloc[va_idx].to_csv(val_split_path, index=False)

train_dataset = AptosDataset(
    train_split_path,
    TRAIN_IMG,
    "train",
    transforms=train_tfms,
    resize=(IMG_DIM, IMG_DIM),
    return_label=True,
)
val_dataset = AptosDataset(
    val_split_path,
    TRAIN_IMG,
    "val",
    transforms=test_tfms,
    resize=(IMG_DIM, IMG_DIM),
    return_label=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=(device == "cuda"),
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=(device == "cuda"),
)

test_dataset = AptosDataset(
    TEST_PATH,
    TEST_IMG,
    "test",
    transforms=test_tfms,
    resize=(IMG_DIM, IMG_DIM),
    return_label=False,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=(device == "cuda"),
)



## === cell 6
model = resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(in_features=2048, out_features=5, bias=True),
    nn.Softmax(dim=1),
)
model = model.to(device)

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)




## === cell 7
def evaluate_kappa_and_loss(model, loader):
    model.eval()
    all_y = []
    all_pred = []
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            prob = model(x)  # already softmax probs
            logp = torch.log(torch.clamp(prob, min=1e-8))
            loss = criterion(logp, y)
            bs = y.size(0)
            total_loss += float(loss.item()) * bs
            n += bs
            pred = prob.argmax(dim=1)
            all_y.append(y.detach().cpu().numpy())
            all_pred.append(pred.detach().cpu().numpy())
    all_y = np.concatenate(all_y)
    all_pred = np.concatenate(all_pred)
    from sklearn.metrics import cohen_kappa_score

    kappa = cohen_kappa_score(all_y, all_pred, weights="quadratic")
    return total_loss / max(n, 1), float(kappa)


EPOCHS = 2  # minimal to move score away from 0.0 without over-optimizing
for epoch in range(1, EPOCHS + 1):
    model.train()
    running = 0.0
    n = 0
    for x, y in tqdm(train_loader, desc=f"Train epoch {epoch}/{EPOCHS}", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prob = model(x)
        logp = torch.log(torch.clamp(prob, min=1e-8))
        loss = criterion(logp, y)
        loss.backward()
        optimizer.step()
        bs = y.size(0)
        running += float(loss.item()) * bs
        n += bs

    tr_loss = running / max(n, 1)
    va_loss, va_kappa = evaluate_kappa_and_loss(model, val_loader)
    print(
        f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_qwk={va_kappa:.4f}"
    )



## === cell 8
model.eval()
labels = []
with torch.no_grad():
    for x in tqdm(test_loader, desc="Infer", leave=False):
        output = model(x.to(device, non_blocking=True))
        predictions = output.argmax(dim=1).detach().cpu().tolist()
        labels.extend(predictions)

assert len(labels) == len(
    test_dataset
), f"Prediction length mismatch: {len(labels)} vs {len(test_dataset)}"



## === cell 9
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if len(sample_sub) != len(labels):
    raise ValueError(
        f"sample_submission rows ({len(sample_sub)}) != predictions ({len(labels)})"
    )

sample_sub["diagnosis"] = labels
sample_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_sub.shape)
print(sample_sub.head())
