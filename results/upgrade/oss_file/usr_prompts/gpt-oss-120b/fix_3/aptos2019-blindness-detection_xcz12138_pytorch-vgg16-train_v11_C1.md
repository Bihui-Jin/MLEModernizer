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
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        input/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
            test_images/
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
            train_images/
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 2 other files
                test_images/
                    82bb8a01935f.png (4.7 MB)
                    aed4e743c230.png (2.3 MB)
                    ... and 365 other files
                train_images/
                    cb2f3c5d71a7.png (872.1 kB)
                    cd54d022e37d.png (2.9 MB)
                    ... and 3293 other files
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> input/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> working/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.7219723542467201

# 6. Current score

0.64895

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.64895) has done: 'The changes increase data‑loading speed by creating the image transform once (instead of rebuilding it for every image) and by enabling parallel image loading with multiple workers. This eliminates redundant Python work while keeping the exact same augmentations, model architecture, training loop, and evaluation logic, so the results remain unchanged.'

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd
from PIL import Image
import torch, torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as tfs
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

print(os.listdir("../input/aptos2019-blindness-detection/"))




## === cell 1
class Config:
    data_dir = "../input/aptos2019-blindness-detection/"
    crop_size = 224
    train_batch_size = 64
    test_batch_size = 1
    lr = 1e-3
    momentum = 0.9
    epochs = 3  # short fine‑tuning
    print_every = 5
    num_workers = min(4, os.cpu_count() or 1)


opt = Config()




## === cell 2
def read_file(data_dir, split="train"):
    file = os.path.join(data_dir, "test.csv" if split == "test" else "train.csv")
    dataset = pd.read_csv(file)
    if split == "test":
        data = [
            os.path.join(data_dir, f"test_images/{img_id}.png")
            for img_id in dataset.iloc[:, 0].values
        ]
        label = None
        return data, label
    else:
        data = [
            os.path.join(data_dir, f"train_images/{img_id}.png")
            for img_id in dataset.iloc[:, 0].values
        ]
        label = dataset.iloc[:, 1].values
        train_data, eval_data, train_label, eval_label = train_test_split(
            data, label, test_size=0.2, stratify=label, random_state=42
        )
        if split == "eval":
            return eval_data, eval_label
        else:
            return train_data, train_label


_GLOBAL_TRANSFORM = tfs.Compose(
    [
        tfs.RandomResizedCrop(opt.crop_size),
        tfs.RandomHorizontalFlip(p=0.2),
        tfs.ToTensor(),
        tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ]
)


def apply_transform(img):
    """Apply the globally defined transform."""
    return _GLOBAL_TRANSFORM(img)


class APTOSSet(Dataset):
    def __init__(
        self,
        split="train",
        data_dir=opt.data_dir,
    ):
        self.data_list, self.label = read_file(data_dir, split=split)
        self.split = split

    def __getitem__(self, idx):
        img_path = self.data_list[idx]
        img = Image.open(img_path).convert("RGB")
        img = apply_transform(img)
        if self.split == "test":
            return img
        else:
            return img, int(self.label[idx])

    def __len__(self):
        return len(self.data_list)


train_set = APTOSSet(split="train")
eval_set = APTOSSet(split="eval")
test_set = APTOSSet(split="test")

APT_train = DataLoader(
    train_set,
    batch_size=opt.train_batch_size,
    shuffle=True,
    num_workers=opt.num_workers,
    pin_memory=True,
)
APT_eval = DataLoader(
    eval_set,
    batch_size=opt.train_batch_size,
    shuffle=False,
    num_workers=opt.num_workers,
    pin_memory=True,
)
APT_test = DataLoader(
    test_set,
    batch_size=opt.test_batch_size,
    shuffle=False,
    num_workers=opt.num_workers,
    pin_memory=True,
)




## === cell 3
model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(in_features=4096, out_features=5)

for name, param in model.named_parameters():
    if "classifier" not in name:
        param.requires_grad = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=opt.lr,
    momentum=opt.momentum,
)




## === cell 4
model.train()
for epoch in range(1, opt.epochs + 1):
    epoch_loss = 0.0
    for batch_idx, (imgs, targets) in enumerate(APT_train):
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if (batch_idx + 1) % opt.print_every == 0:
            print(
                f"Epoch {epoch}/{opt.epochs} "
                f"Batch {batch_idx+1}/{len(APT_train)} "
                f"Loss: {epoch_loss / (batch_idx+1):.4f}"
            )
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, targets in APT_eval:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    acc = correct / total if total > 0 else 0
    print(f"Validation accuracy after epoch {epoch}: {acc:.4f}")
    model.train()




## === cell 5
def test_predict(model, loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            outputs = model(data)
            _, pred = torch.max(outputs, 1)
            predictions.append(int(pred.item()))
    return predictions


sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
sub["diagnosis"] = test_predict(model, APT_test, device)
sub.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
print(sub.head())
