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
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
timm==1.0.19
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

# 5. Target score

0.926329653636144

# 6. Current score

0.41699

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.16928) has done: 'I make the model load safely without requiring missing weight files, fall back to CPU when CUDA isn’t available, and ensure the submission CSV is written correctly. This fixes the file‑not‑found error, the CUDA initialization crash, and the empty‑submission problem while keeping the original architecture unchanged.'
- What this solution (achieved -0.04273) has done: 'I added the missing imports (pandas, random, math, ImageChops, torchvision functional transforms) and defined the `device` variable early so the script can run without NameErrors. I also renumbered the cells to start at 1 as required. No other logic was changed, preserving the original model and inference flow while ensuring a valid `submission.csv` is written.'
- What this solution (achieved 0.0626) has done: 'I renumber the cells to start at 1, add a lightweight prediction helper that uses the classifier logits (argmax) instead of the original averaging of three outputs, and modify the inference loop to call this helper. This change keeps the model architecture unchanged while providing a more direct and usually more accurate class prediction, which should raise the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.70576) has done: 'I speed up data loading by using multiple workers, pinning memory, and batching the test inference instead of loading images one‑by‑one. These changes keep exactly the same model, loss, and prediction logic, just make I/O and GPU transfer more efficient, preserving correctness.'
- What this solution (achieved 0.41699) has done: 'The fix removes the illegal `prefetch_factor` setting (it can only be used when `num_workers > 0`) so the training DataLoader can be created, which allows the model `net1` to be defined and later used for inference. No other logic changes are made, preserving the original architecture and training procedure while ensuring a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os
import warnings
import math
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import timm

from sklearn.metrics import cohen_kappa_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # keep deterministic as required




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = 0
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu().item()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5).to(device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5)).to(device)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(out[i].item()))
            l2 = int(math.ceil(out[i].item()))
            pred_prob[i][l1] = 1 - (out[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob


def combine3output(r_out, c_out, o_out):
    """Average the three heads (regression, classifier, ordinal) and round."""
    reg_pred = int(round(r_out.squeeze().cpu().item()))
    cls_pred = int(torch.argmax(F.softmax(c_out, dim=1), dim=1).cpu().item())
    ord_pred = int(torch.argmax(ordinal2class_prob(o_out), dim=1).cpu().item())
    avg_pred = (reg_pred + cls_pred + ord_pred) / 3.0
    return int(round(avg_pred))


def classifier_pred(c_out):
    """Return class index from classifier logits (softmax argmax)."""
    _, pred = torch.max(F.softmax(c_out, dim=1), dim=1)
    return int(pred.item())


def combine3output_batch(r_out, c_out, o_out):
    """Batch version of combine3output – returns a 1‑D tensor of class ids."""
    reg_pred = torch.round(r_out).long()
    cls_pred = torch.argmax(F.softmax(c_out, dim=1), dim=1)
    ord_pred = torch.argmax(ordinal2class_prob(o_out), dim=1)
    avg = (reg_pred + cls_pred + ord_pred).float() / 3.0
    return torch.round(avg).long()




## === cell 2
BASE_PATH = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
test_ids = test_df["id_code"].tolist()

img_size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

base_transform = T.Compose(
    [
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean, std),
    ]
)

aug_transform = T.Compose(
    [
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
    ]
)

val_transform = T.Compose(
    [T.Resize((img_size, img_size)), T.ToTensor(), T.Normalize(mean, std)]
)


class CachedTrainDataset(Dataset):
    """Loads all training images into RAM without augmentation, then applies lightweight flips on the fly."""

    def __init__(self, df, img_dir, base_transform, aug_transform=None):
        self.ids = df["id_code"].values
        self.labels = df["diagnosis"].values
        self.img_dir = img_dir
        self.base_transform = base_transform
        self.aug_transform = aug_transform
        self.tensors = []
        for img_id in self.ids:
            img_path = os.path.join(self.img_dir, f"{img_id}.png")
            img = Image.open(img_path).convert("RGB")
            img = self.base_transform(img)
            self.tensors.append(img)
        self.tensors = torch.stack(self.tensors)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        img = self.tensors[idx]
        if self.aug_transform:
            img = self.aug_transform(img)
        label = int(self.labels[idx])
        return img, label, self.ids[idx]


class CachedValDataset(Dataset):
    """Loads all validation images once and stores tensors in memory."""

    def __init__(self, subset, transform):
        base = subset.dataset  # original AptosDataset
        self.tensors = []
        self.labels = []
        self.ids = []
        for idx in subset.indices:
            img_id = base.ids[idx]
            label = int(base.labels[idx])
            img_path = os.path.join(base.img_dir, f"{img_id}.png")
            img = Image.open(img_path).convert("RGB")
            img = transform(img)  # deterministic validation transform
            self.tensors.append(img)
            self.labels.append(label)
            self.ids.append(img_id)
        self.tensors = torch.stack(self.tensors)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.ids = np.array(self.ids)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx], self.ids[idx]


full_dataset = CachedTrainDataset(
    train_df, TRAIN_IMG_DIR, base_transform, aug_transform
)
val_size = int(0.1 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

batch_size = 128  # unchanged batch size
num_workers = 0  # data already cached in RAM, no extra workers needed

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=None,  # disable prefetch when num_workers == 0
)

cached_val_dataset = CachedValDataset(val_dataset, val_transform)

val_loader = DataLoader(
    cached_val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)


class MultiOutputNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.backbone = timm.create_model("resnet34", pretrained=True, in_chans=3)
        feat_dim = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)
        self.classifier = nn.Linear(feat_dim, num_classes)  # c_out
        self.regressor = nn.Linear(feat_dim, 1)  # r_out
        self.ordinal = nn.Linear(feat_dim, 4)  # o_out (4 logits for ordinal)

    def forward(self, x):
        feats = self.backbone(x)
        c_out = self.classifier(feats)  # (B,5)
        r_out = self.regressor(feats).squeeze(1)  # (B,)
        o_out = self.ordinal(feats)  # (B,4)
        return c_out, r_out, o_out


net1 = MultiOutputNet().to(device)

net1 = torch.compile(net1, mode="reduce-overhead")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net1.parameters(), lr=1e-4)

epochs = 10  # unchanged training schedule
best_kappa = -1.0
best_state_path = "best_model.pth"

for epoch in range(epochs):
    net1.train()
    running_loss = 0.0
    for imgs, labels, _ in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        c_out, r_out, o_out = net1(imgs)
        loss = criterion(c_out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)

    net1.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            c_out, _, _ = net1(imgs)
            pred = torch.argmax(F.softmax(c_out, dim=1), dim=1)
            all_preds.append(pred.cpu())
            all_labels.append(labels.cpu())
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    preds_tensor = torch.cat(all_preds)
    labels_tensor = torch.cat(all_labels)

    val_kappa = cohen_kappa_score(
        preds_tensor.numpy(), labels_tensor.numpy(), weights="quadratic"
    )
    val_acc = correct / total if total > 0 else 0.0

    print(
        f"Epoch {epoch+1}/{epochs} - Train loss: {epoch_loss:.4f} - "
        f"Val acc: {val_acc:.4f} - Val QWK: {val_kappa:.4f}"
    )

    if val_kappa > best_kappa:
        best_kappa = val_kappa
        torch.save(net1.state_dict(), best_state_path)

net1.load_state_dict(torch.load(best_state_path, map_location=device))
net1.eval()




## === cell 3
existing_ids = [
    idx for idx in test_ids if os.path.exists(os.path.join(TEST_IMG_DIR, f"{idx}.png"))
]
missing_ids = set(test_ids) - set(existing_ids)


class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform=None):
        self.ids = ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, img_id


class CachedTestDataset(Dataset):
    """Loads all test images once and stores tensors in memory."""

    def __init__(self, ids, img_dir, transform):
        self.tensors = []
        self.ids = []
        for img_id in ids:
            img_path = os.path.join(img_dir, f"{img_id}.png")
            img = Image.open(img_path).convert("RGB")
            img = transform(img)
            self.tensors.append(img)
            self.ids.append(img_id)
        self.tensors = torch.stack(self.tensors)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx], self.ids[idx]


test_dataset = CachedTestDataset(existing_ids, TEST_IMG_DIR, transform=val_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

submission = []
with torch.no_grad():
    for imgs, ids_batch in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        c_out, r_out, o_out = net1(imgs)
        batch_preds = combine3output_batch(r_out, c_out, o_out).cpu().numpy()
        for idx, pred in zip(ids_batch, batch_preds):
            submission.append([idx, int(pred)])

for idx in missing_ids:
    submission.append([idx, 0])

submission_df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}, rows: {len(submission_df)}")
