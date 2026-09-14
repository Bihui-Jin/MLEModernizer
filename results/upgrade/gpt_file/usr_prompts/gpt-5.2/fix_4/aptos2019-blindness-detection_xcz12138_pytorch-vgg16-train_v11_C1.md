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

3.7

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
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
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input/aptos2019-blindness-detection/"))




## === cell 1
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torchvision import transforms as tfs
from torchvision import models
from torch.utils.data import DataLoader, Dataset


class Config:
    data_dir = "../input/aptos2019-blindness-detection/"
    crop_size = 224
    train_batch_size = 64
    test_batch_size = 1
    lr = 1e-3
    momentum = 0.9
    epochs = 20
    print_every = 5
    num_classes = 5
    seed = 42


opt = Config()




## === cell 2
_train_csv = pd.read_csv(os.path.join(opt.data_dir, "train.csv"))
_test_csv = pd.read_csv(os.path.join(opt.data_dir, "test.csv"))

_train_paths_all = (
    opt.data_dir + "train_images/" + _train_csv["id_code"].astype(str) + ".png"
).to_numpy()
_train_labels_all = _train_csv["diagnosis"].to_numpy(dtype=np.int64)

_test_paths_all = (
    opt.data_dir + "test_images/" + _test_csv["id_code"].astype(str) + ".png"
).to_numpy()

_train_paths, _eval_paths, _train_labels, _eval_labels = train_test_split(
    _train_paths_all,
    _train_labels_all,
    test_size=0.2,
    random_state=opt.seed,
    stratify=_train_labels_all,
)

_TRAIN_TF = tfs.Compose(
    [
        tfs.RandomResizedCrop(opt.crop_size),
        tfs.RandomHorizontalFlip(p=0.2),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)
_EVAL_TF = tfs.Compose(
    [
        tfs.Resize(opt.crop_size),
        tfs.CenterCrop(opt.crop_size),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def transforms(img, crop_size):
    return _TRAIN_TF(img.convert("RGB"))


class APTOSSet(Dataset):
    def __init__(self, split="train"):
        self.split = split
        if split == "train":
            self.data_list = _train_paths
            self.label = _train_labels
            self._tf = _TRAIN_TF
        elif split == "eval":
            self.data_list = _eval_paths
            self.label = _eval_labels
            self._tf = _EVAL_TF
        else:
            self.data_list = _test_paths_all
            self.label = None
            self._tf = _EVAL_TF

    def __getitem__(self, idx):
        img_path = self.data_list[idx]
        with Image.open(img_path) as img:
            img = self._tf(img.convert("RGB"))

        if self.split == "test":
            return img
        else:
            return img, int(self.label[idx])

    def __len__(self):
        return len(self.data_list)


train_set = APTOSSet(split="train")
eval_set = APTOSSet(split="eval")
test_set = APTOSSet(split="test")

_num_workers = min(8, (os.cpu_count() or 2))


def _seed_worker(worker_id: int):
    worker_seed = (opt.seed + worker_id) % 2**32
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


_gen = torch.Generator()
_gen.manual_seed(opt.seed)

_loader_common = dict(
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=8 if _num_workers > 0 else None,
    worker_init_fn=_seed_worker if _num_workers > 0 else None,
    generator=_gen,
)

APT_train = DataLoader(
    train_set,
    batch_size=opt.train_batch_size,
    shuffle=True,
    **_loader_common,
)
APT_eval = DataLoader(
    eval_set,
    batch_size=opt.train_batch_size,
    shuffle=False,
    **_loader_common,
)
APT_test = DataLoader(
    test_set,
    batch_size=max(64, opt.test_batch_size),
    shuffle=False,
    **_loader_common,
)

print(len(train_set), len(eval_set), len(test_set))




## === cell 3
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(opt.seed)
np.random.seed(opt.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, opt.num_classes)
model = model.to(device)




## === cell 4
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum)


def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss_sum += loss.item() * xb.size(0)
            pred = logits.argmax(1)
            correct += (pred == yb).sum().item()
            total += xb.size(0)
    return loss_sum / max(1, total), correct / max(1, total)


for epoch in range(1, opt.epochs + 1):
    model.train()
    running_loss = 0.0
    n = 0
    for xb, yb in APT_train:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        n += xb.size(0)

    if epoch % opt.print_every == 0 or epoch == 1 or epoch == opt.epochs:
        tr_loss = running_loss / max(1, n)
        va_loss, va_acc = evaluate(model, APT_eval)
        print(
            f"epoch {epoch:02d}/{opt.epochs} train_loss={tr_loss:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )




## === cell 5
def test_predict(model):
    model.eval()
    prediction = []
    with torch.no_grad():
        for xb in APT_test:
            xb = xb.to(device, non_blocking=True)
            outputs = model(xb)
            pred = outputs.argmax(1).detach().cpu().numpy().astype(np.int64)
            prediction.extend(pred.tolist())
    return prediction


sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
preds = test_predict(model)

assert len(preds) == len(
    sub
), f"Pred length {len(preds)} != sample_submission length {len(sub)}"
sub["diagnosis"] = preds
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
