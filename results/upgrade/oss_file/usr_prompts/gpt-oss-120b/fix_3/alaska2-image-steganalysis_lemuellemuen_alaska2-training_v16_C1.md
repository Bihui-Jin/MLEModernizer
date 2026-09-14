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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.13

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
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.8876821205467409

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, re, random, time, datetime
import numpy as np, pandas as pd
from glob import glob
from tqdm import tqdm
import cv2
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler
from torchsampler import ImbalancedDatasetSampler
import timm
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import GroupKFold



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_55/1436016699.py in <cell line: 0>()
      8 from torch.utils.data import Dataset, DataLoader
      9 from torch.utils.data.sampler import SequentialSampler
---> 10 from torchsampler import ImbalancedDatasetSampler
     11 import timm
     12 import albumentations as A

ModuleNotFoundError: No module named 'torchsampler'

## === cell 1
PATH = "/kaggle/input/alaska2-image-steganalysis"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")



## === cell 2
SEED = 42


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(SEED)




## === cell 3
class DatasetRetriever(Dataset):
    def __init__(self, kinds, image_names, labels, transforms=None):
        self.kinds = kinds
        self.image_names = image_names
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, idx):
        kind, img_name, label = self.kinds[idx], self.image_names[idx], self.labels[idx]
        img = cv2.imread(f"{PATH}/{kind}/{img_name}", cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.transforms:
            img = self.transforms(image=img)["image"]
        target = torch.zeros(4, dtype=torch.float32)
        target[label] = 1.0
        return img, target

    def __len__(self):
        return len(self.image_names)

    def get_labels(self):
        return list(self.labels)




## === cell 4
def get_train_transforms():
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Resize(512, 512),
            ToTensorV2(),
        ]
    )


def get_valid_transforms():
    return A.Compose([A.Resize(512, 512), ToTensorV2()])




## === cell 5
def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.0
    return vec




## === cell 6
def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]
    fpr, tpr, _ = metrics.roc_curve(y_true, y_valid, pos_label=1)
    areas = np.diff(tpr_thresholds)
    normalization = np.dot(areas, weights)
    competition_metric = 0.0
    for idx, weight in enumerate(weights):
        y_min, y_max = tpr_thresholds[idx], tpr_thresholds[idx + 1]
        mask = (tpr > y_min) & (tpr <= y_max)
        if not np.any(mask):
            continue
        x_pad = np.linspace(fpr[mask][-1], 1, 100)
        x = np.concatenate([fpr[mask], x_pad])
        y = np.concatenate([tpr[mask], np.full_like(x_pad, y_max)])
        y = y - y_min
        competition_metric += weight * metrics.auc(x, y)
    return competition_metric / normalization


class RocAucMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.y_true = np.array([0, 1])
        self.y_pred = np.array([0.5, 0.5])
        self.score = 0

    def update(self, y_true, y_pred):
        y_true = y_true.cpu().numpy().argmax(axis=1).clip(0, 1)
        y_pred = 1 - F.softmax(y_pred, dim=1).cpu().numpy()[:, 0]
        self.y_true = np.hstack((self.y_true, y_true))
        self.y_pred = np.hstack((self.y_pred, y_pred))
        self.score = alaska_weighted_auc(self.y_true, self.y_pred)

    @property
    def avg(self):
        return self.score




## === cell 7
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            logprobs = F.log_softmax(x, dim=-1)
            nll = -(logprobs * target).sum(dim=-1)
            smooth = -logprobs.mean(dim=-1)
            loss = self.confidence * nll + self.smoothing * smooth
            return loss.mean()
        else:
            return F.cross_entropy(x, target)




## === cell 8
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




## === cell 9
class EffNet(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(6, 12, 3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(12, 36, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(12)
        self.bn3 = nn.BatchNorm2d(36)

        self.net = timm.create_model("efficientnet_b0", pretrained=True)
        self.net.conv_stem.weight = nn.Parameter(
            self.net.conv_stem.weight.repeat(1, 12, 1, 1)
        )

        self.dropout = nn.Dropout(0.5)
        self.net.blocks[5] = nn.Identity()
        self.net.blocks[6] = nn.Sequential(
            nn.Conv2d(
                self.net.blocks[4][2].conv_pwl.out_channels,
                self.net.conv_head.in_channels,
                1,
            ),
            nn.BatchNorm2d(self.net.conv_head.in_channels),
            nn.ReLU6(),
        )
        self.fc = nn.Linear(self.net.classifier.in_features, out_dim)
        self.net.classifier = nn.Identity()

    def extract(self, x):
        x = F.relu6(self.bn1(self.conv1(x)))
        x = F.relu6(self.bn2(self.conv2(x)))
        x = F.relu6(self.bn3(self.conv3(x)))
        x = self.net(x)
        return x

    def forward(self, x):
        x = self.extract(x)
        x = self.fc(self.dropout(x))
        return x




## === cell 10
class Config:
    batch_size = 16
    n_epochs = 12  # more epochs for better AUC
    num_workers = 2
    lr = 0.001
    verbose = True
    verbose_step = 1
    step_scheduler = True
    validation_scheduler = True
    SchedulerClass = torch.optim.lr_scheduler.ReduceLROnPlateau
    scheduler_params = dict(
        mode="min",
        factor=0.5,
        patience=1,
        verbose=False,
        threshold=1e-4,
        threshold_mode="abs",
        cooldown=0,
        min_lr=1e-8,
        eps=1e-8,
    )




## === cell 11
CLASSES = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]
N_SPLITS = 5

df = []
for label, kind in enumerate(CLASSES):
    for path in glob(os.path.join(PATH, kind, "*.jpg")):
        df.append({"kind": kind, "image_name": os.path.basename(path), "label": label})
df = pd.DataFrame(df)
df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
df["fold"] = -1
gkf = GroupKFold(n_splits=N_SPLITS)
for fold, (tr_idx, val_idx) in enumerate(
    gkf.split(df.index, df["label"], groups=df["image_name"])
):
    df.loc[val_idx, "fold"] = fold

fold_number = 0
train_df = df[df["fold"] != fold_number]
val_df = df[df["fold"] == fold_number]

train_dataset = DatasetRetriever(
    kinds=train_df["kind"].values,
    image_names=train_df["image_name"].values,
    labels=train_df["label"].values,
    transforms=get_train_transforms(),
)
val_dataset = DatasetRetriever(
    kinds=val_df["kind"].values,
    image_names=val_df["image_name"].values,
    labels=val_df["label"].values,
    transforms=get_valid_transforms(),
)

train_loader = DataLoader(
    train_dataset,
    sampler=ImbalancedDatasetSampler(train_dataset, labels=train_dataset.get_labels()),
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
    pin_memory=False,
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),
    batch_size=Config.batch_size,
    num_workers=Config.num_workers,
    pin_memory=False,
    drop_last=False,
)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2427392720.py in <cell line: 0>()
      9 df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
     10 df["fold"] = -1
---> 11 gkf = GroupKFold(n_splits=N_SPLITS)
     12 for fold, (tr_idx, val_idx) in enumerate(
     13     gkf.split(df.index, df["label"], groups=df["image_name"])

NameError: name 'GroupKFold' is not defined

## === cell 12
class Fitter:
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.epoch = 0
        self.best_summary_loss = float("inf")
        self.log_path = "./log.txt"
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.lr)
        self.scheduler = config.SchedulerClass(
            self.optimizer, **config.scheduler_params
        )
        self.criterion = LabelSmoothing().to(self.device)

    def log(self, msg):
        if self.config.verbose:
            print(msg)
        with open(self.log_path, "a+") as f:
            f.write(msg + "\n")

    def fit(self, train_loader, val_loader):
        for e in range(self.epoch, self.config.n_epochs):
            if self.config.verbose:
                lr = self.optimizer.param_groups[0]["lr"]
                self.log(f"\nEpoch {e} - LR {lr}")
            t0 = time.time()
            train_loss, train_score = self._run_epoch(train_loader, train=True)
            self.log(
                f"[RESULT]: Train. Epoch: {e}, loss: {train_loss.avg:.5f}, score: {train_score.avg:.5f}, time: {time.time()-t0:.2f}"
            )
            self.save(f"last-checkpoint.bin")
            t0 = time.time()
            val_loss, val_score = self._run_epoch(val_loader, train=False)
            self.log(
                f"[RESULT]: Val. Epoch: {e}, loss: {val_loss.avg:.5f}, score: {val_score.avg:.5f}, time: {time.time()-t0:.2f}"
            )
            if val_loss.avg < self.best_summary_loss:
                self.best_summary_loss = val_loss.avg
                self.save(f"best-checkpoint-{e:03d}epoch.bin")
            if self.config.validation_scheduler:
                self.scheduler.step(metrics=val_loss.avg)
            self.epoch += 1

    def _run_epoch(self, loader, train):
        self.model.train() if train else self.model.eval()
        loss_meter = AverageMeter()
        score_meter = RocAucMeter()
        for step, (imgs, targets) in enumerate(loader):
            if train:
                self.optimizer.zero_grad()
            imgs = imgs.to(self.device).float()
            targets = targets.to(self.device).float()
            outputs = self.model(imgs)
            loss = self.criterion(outputs, targets)
            if train:
                loss.backward()
                self.optimizer.step()
                if self.config.step_scheduler:
                    self.scheduler.step()
            loss_meter.update(loss.item(), imgs.size(0))
            score_meter.update(targets, outputs)
        return loss_meter, score_meter

    def save(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_summary_loss": self.best_summary_loss,
                "epoch": self.epoch,
            },
            path,
        )

    def load(self, path):
        chk = torch.load(path, map_location=self.device)
        self.model.load_state_dict(chk["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(chk["optimizer_state_dict"])
        self.scheduler.load_state_dict(chk["scheduler_state_dict"])
        self.best_summary_loss = chk["best_summary_loss"]
        self.epoch = chk["epoch"]
        self.log(f"Loaded checkpoint {path}, resume epoch {self.epoch}")




## === cell 13
model = EffNet(out_dim=4).to(DEVICE)
fitter = Fitter(model, DEVICE, Config)
fitter.fit(train_loader, val_loader)

best_ckpt = max(
    [f for f in os.listdir(".") if f.startswith("best-checkpoint")], default=None
)
if best_ckpt:
    fitter.load(best_ckpt)
    model = fitter.model
else:
    print("Best checkpoint not found, using last trained model.")




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2563351807.py in <cell line: 0>()
----> 1 model = EffNet(out_dim=4).to(DEVICE)
      2 fitter = Fitter(model, DEVICE, Config)
      3 fitter.fit(train_loader, val_loader)
      4 
      5 # load the best checkpoint for inference

/tmp/ipykernel_55/1296016020.py in __init__(self, out_dim)
      9         self.bn3 = nn.BatchNorm2d(36)
     10 
---> 11         self.net = timm.create_model("efficientnet_b0", pretrained=True)
     12         # repeat stem weights to match 3‑channel input after our custom convs
     13         self.net.conv_stem.weight = nn.Parameter(

NameError: name 'timm' is not defined

## === cell 14
def get_test_transforms(mode):
    if mode == 0:
        return A.Compose([A.Resize(512, 512), ToTensorV2()])
    elif mode == 1:
        return A.Compose([A.HorizontalFlip(p=1), A.Resize(512, 512), ToTensorV2()])
    elif mode == 2:
        return A.Compose([A.VerticalFlip(p=1), A.Resize(512, 512), ToTensorV2()])
    else:
        return A.Compose(
            [
                A.HorizontalFlip(p=1),
                A.VerticalFlip(p=1),
                A.Resize(512, 512),
                ToTensorV2(),
            ]
        )




## === cell 15
class DatasetSubmissionRetriever(Dataset):
    def __init__(self, image_names, transforms=None):
        self.image_names = image_names
        self.transforms = transforms

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        name = self.image_names[idx]
        img = cv2.imread(f"{PATH}/Test/{name}", cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.transforms:
            img = self.transforms(image=img)["image"]
        return name, img




## === cell 16
test_names = np.array([os.path.basename(p) for p in glob(f"{PATH}/Test/*.jpg")])
all_results = []
model.eval()
for mode in range(4):
    sub_dataset = DatasetSubmissionRetriever(
        test_names, transforms=get_test_transforms(mode)
    )
    sub_loader = DataLoader(sub_dataset, batch_size=8, shuffle=False, num_workers=2)
    result = {"Id": [], "Label": []}
    for ids, imgs in sub_loader:
        with torch.no_grad():
            preds = model(imgs.to(DEVICE))
            probs = (
                1 - F.softmax(preds, dim=1).cpu().numpy()[:, 0]
            )  # probability of stego
        result["Id"].extend(ids)
        result["Label"].extend(probs)
    all_results.append(result)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2646767067.py in <cell line: 0>()
      1 test_names = np.array([os.path.basename(p) for p in glob(f"{PATH}/Test/*.jpg")])
      2 all_results = []
----> 3 model.eval()
      4 for mode in range(4):
      5     sub_dataset = DatasetSubmissionRetriever(

NameError: name 'model' is not defined

## === cell 17
w0, w1, w2, w3 = 5, 1, 1, 1
weight_sum = w0 + w1 + w2 + w3
ensemble = pd.DataFrame(
    {
        "Id": all_results[0]["Id"],
        "Label": (
            np.array(all_results[0]["Label"]) * w0
            + np.array(all_results[1]["Label"]) * w1
            + np.array(all_results[2]["Label"]) * w2
            + np.array(all_results[3]["Label"]) * w3
        )
        / weight_sum,
    }
)
ensemble.to_csv("submission.csv", index=False)
print("✅ submission.csv saved with", len(ensemble), "rows")

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/1589086060.py in <cell line: 0>()
      4 ensemble = pd.DataFrame(
      5     {
----> 6         "Id": all_results[0]["Id"],
      7         "Label": (
      8             np.array(all_results[0]["Label"]) * w0

IndexError: list index out of range
