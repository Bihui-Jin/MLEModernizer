# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os, random, gc
import numpy as np, pandas as pd, cv2
from glob import glob
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
from sklearn import metrics
import matplotlib.pyplot as plt

seed = 42
print(f"setting everything to seed {seed}")
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True




## === cell 1
gpu_available = torch.cuda.is_available()
base_sample_size = 8000  # increased from 3000 to provide more training data
sample_size = base_sample_size if gpu_available else 2000  # still reduced for CPU

data_dir = "../input/alaska2-image-steganalysis"
val_frac = 0.25

folder_names = ["Cover/", "JMiPOD/", "JUNIWARD/", "UERD/"]  # labels 0‑3
train_files, val_files = [], []
train_labels, val_labels = [], []

for label, folder in enumerate(folder_names):
    all_files = sorted(glob(f"{data_dir}/{folder}/*.jpg"))
    selected = all_files[:sample_size]  # take first N per class
    random.shuffle(selected)
    split = int(len(selected) * val_frac)
    val_files.extend(selected[:split])
    val_labels.extend([label] * len(selected[:split]))
    train_files.extend(selected[split:])
    train_labels.extend([label] * len(selected[split:]))

train_df = pd.DataFrame({"ImageFileName": train_files, "Label": train_labels})
val_df = pd.DataFrame({"ImageFileName": val_files, "Label": val_labels})

print("train", train_df.shape, "val", val_df.shape)




## === cell 2
img_size = 224  # EfficientNet‑B0 default input size
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)

test_transform = T.Compose(
    [
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)




## === cell 3
class Alaska2Dataset(Dataset):
    def __init__(self, df, augment=False):
        self.df = df.reset_index(drop=True)
        self.augment = augment
        self.has_label = "Label" in self.df.columns

        self.images_raw = [None] * len(self.df)

        if self.has_label:
            self.labels = torch.tensor(self.df["Label"].values, dtype=torch.long)

        self.mean = torch.tensor(imagenet_mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(imagenet_std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.df)

    def _load_image(self, idx):
        fn = self.df.loc[idx, "ImageFileName"]
        img = cv2.imread(fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(img).float() / 255.0  # (H, W, C) float in [0,1]
        tensor = tensor.permute(2, 0, 1)  # (C, H, W)
        self.images_raw[idx] = tensor

    def __getitem__(self, idx):
        if self.images_raw[idx] is None:
            self._load_image(idx)
        img = self.images_raw[idx]  # already float in [0,1]

        if self.augment:
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[1])
            brightness = 0.2
            contrast = 0.2
            saturation = 0.2
            hue = 0.1

            if brightness > 0:
                factor = 1.0 + (torch.rand(1).item() * 2 - 1) * brightness
                img = img * factor
            if contrast > 0:
                factor = 1.0 + (torch.rand(1).item() * 2 - 1) * contrast
                mean = img.mean(dim=(1, 2), keepdim=True)
                img = (img - mean) * factor + mean
            if saturation > 0:
                factor = 1.0 + (torch.rand(1).item() * 2 - 1) * saturation
                gray = img.mean(dim=0, keepdim=True)
                img = (img - gray) * factor + gray
            if hue > 0:
                img = TF.adjust_hue(img, (torch.rand(1).item() * 2 - 1) * hue)

        img = (img - self.mean) / self.std

        if self.has_label:
            label = int(self.labels[idx].item())
            return img, label
        else:
            return img




## === cell 4
from torchvision import models as tv_models


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = tv_models.efficientnet_b0(pretrained=True)
        self.head = nn.Linear(1280, 4)

    def forward(self, x):
        feats = self.backbone.features(x)
        pooled = F.adaptive_avg_pool2d(feats, 1).reshape(x.size(0), -1)
        return self.head(pooled)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)




## === cell 5
batch_size = 128 if gpu_available else 64

train_dataset = Alaska2Dataset(train_df, augment=True)
val_dataset = Alaska2Dataset(val_df, augment=False)

num_workers = min(4, os.cpu_count() or 1)  # modest workers to limit overhead

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=gpu_available,
    persistent_workers=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=gpu_available,
    persistent_workers=True,
)




## === cell 6
def alaska_weighted_auc(y_true, y_score):
    tpr_thr = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
    seg_areas = np.diff(tpr_thr)
    norm = np.dot(seg_areas, weights)
    metric = 0.0

    for w, lo, hi in zip(weights, tpr_thr[:-1], tpr_thr[1:]):
        mask = (tpr >= lo) & (tpr <= hi)
        if not np.any(mask):
            continue
        fpr_seg = np.concatenate([fpr[mask], [1.0]])
        tpr_seg = np.concatenate([tpr[mask], [hi]])
        tpr_shift = tpr_seg - lo
        seg_auc = metrics.auc(fpr_seg, tpr_shift)
        metric += w * seg_auc

    return metric / norm




## === cell 7
from torch.cuda.amp import GradScaler, autocast

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=6, gamma=0.5
)  # delayed LR decay

if gpu_available:
    scaler = GradScaler()
else:
    scaler = None


def train_one_epoch(loader):
    model.train()
    running_loss = 0.0
    for imgs, lbls in tqdm(loader, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)

        optimizer.zero_grad()
        if gpu_available:
            with autocast():
                outs = model(imgs)
                loss = criterion(outs, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outs = model(imgs)
            loss = criterion(outs, lbls)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def evaluate(loader):
    model.eval()
    all_labels, all_scores = [], []
    val_loss = 0.0
    with torch.no_grad():
        for imgs, lbls in tqdm(loader, leave=False):
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            if gpu_available:
                with autocast():
                    outs = model(imgs)
                    loss = criterion(outs, lbls)
            else:
                outs = model(imgs)
                loss = criterion(outs, lbls)

            val_loss += loss.item() * imgs.size(0)

            probs = F.softmax(outs, dim=1).cpu().numpy()
            bin_labels = (lbls.cpu().numpy() != 0).astype(int)
            bin_scores = 1.0 - probs[:, 0]
            all_labels.extend(bin_labels)
            all_scores.extend(bin_scores)

    val_loss /= len(loader.dataset)
    auc = alaska_weighted_auc(np.array(all_labels), np.array(all_scores))
    return val_loss, auc




## === cell 8
num_epochs = 20 if gpu_available else 10  # extended training
train_losses, val_losses, val_aucs = [], [], []

for epoch in range(1, num_epochs + 1):
    print(f"\nEpoch {epoch}/{num_epochs}")
    tr_loss = train_one_epoch(train_loader)
    val_loss, val_auc = evaluate(val_loader)
    train_losses.append(tr_loss)
    val_losses.append(val_loss)
    val_aucs.append(val_auc)
    print(
        f"Train loss: {tr_loss:.4f} | Val loss: {val_loss:.4f} | Weighted AUC: {val_auc:.4f}"
    )
    scheduler.step()  # update learning rate according to scheduler




## === cell 9
plt.figure(figsize=(12, 5))
plt.plot(train_losses, label="Train loss", color="r")
plt.plot(val_losses, label="Val loss", color="b")
plt.legend()
plt.title("Loss per epoch")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(val_aucs, label="Weighted AUC", marker="o")
plt.ylim(0, 1)
plt.title("Validation Weighted AUC")
plt.legend()
plt.show()




## === cell 10
test_files = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_df = pd.DataFrame({"ImageFileName": test_files})

test_dataset = Alaska2Dataset(test_df, augment=False)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=gpu_available,
    persistent_workers=True,
)

model.eval()
test_scores = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test inference"):
        imgs = batch.to(device, non_blocking=True)
        if gpu_available:
            with autocast():
                outs = model(imgs)
        else:
            outs = model(imgs)
        probs = F.softmax(outs, dim=1).cpu().numpy()
        scores = 1.0 - probs[:, 0]
        test_scores.extend(scores)

test_df["Id"] = test_df["ImageFileName"].apply(lambda p: os.path.basename(p))
test_df["Label"] = test_scores
submission = test_df[["Id", "Label"]]
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(submission.head())
