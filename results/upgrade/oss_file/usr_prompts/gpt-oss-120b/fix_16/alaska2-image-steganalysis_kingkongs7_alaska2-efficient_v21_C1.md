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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics

from albumentations import Compose, Resize, VerticalFlip, HorizontalFlip, ToFloat
from albumentations.pytorch import ToTensorV2

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from torch.cuda.amp import autocast, GradScaler

seed = 42
print(f"setting everything to seed {seed}")
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False



## === cell 1
data_dir = "../input/alaska2-image-steganalysis"
sample_size = 8000  # originally 5000
val_size = int(sample_size * 0.25)

train_files, train_labels = [], []
val_files, val_labels = [], []

folder_names = ["Cover/", "JMiPOD/", "JUNIWARD/", "UERD/"]  # label 0‑3
for label, folder in enumerate(folder_names):
    all_files = sorted(glob(f"{data_dir}/{folder}/*.jpg"))[:sample_size]
    np.random.shuffle(all_files)
    train_files.extend(all_files[val_size:])  # training portion
    val_files.extend(all_files[:val_size])  # validation portion
    train_labels.extend([label] * (len(all_files) - val_size))
    val_labels.extend([label] * val_size)

train_files = np.array(train_files)
train_labels = np.array(train_labels, dtype=np.int64)
val_files = np.array(val_files)
val_labels = np.array(val_labels, dtype=np.int64)

print(f"Train samples: {len(train_files)}, Validation samples: {len(val_files)}")




## === cell 2
class Alaska2Dataset(Dataset):
    def __init__(self, files, labels=None, augmentations=None):
        self.files = files
        self.labels = labels
        self.augment = augmentations

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        im = cv2.imread(fn)
        if im is None:
            raise FileNotFoundError(f"Image not found: {fn}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if self.augment:
            im = self.augment(image=im)["image"]
        else:
            im = ToTensorV2()(image=im)["image"]

        if self.labels is None:
            return im
        else:
            return im, self.labels[idx]




## === cell 3
sample_idx = np.random.choice(len(train_files), size=64, replace=False)
sample_dataset = Alaska2Dataset(
    train_files[sample_idx], train_labels[sample_idx], augmentations=None
)
sample_loader = DataLoader(sample_dataset, batch_size=64, shuffle=False, num_workers=0)

imgs, lbls = next(iter(sample_loader))
imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()
grid_w, grid_h = 16, 4
fig, axs = plt.subplots(grid_h, grid_w, figsize=(grid_w + 1, grid_h + 1))
for i in range(len(imgs)):
    ax = axs[i // grid_w, i % grid_w]
    ax.imshow(imgs[i])
    ax.set_title(str(lbls[i].item()))
    ax.axis("off")
plt.suptitle("Sample images (labels 0‑3)")
plt.show()
del imgs, lbls
gc.collect()



## === cell 4
img_size = 224
AUGMENTATIONS_TRAIN = Compose(
    [
        Resize(img_size, img_size, p=1.0),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        ToFloat(max_value=255),
        ToTensorV2(),
    ],
    p=1.0,
)

AUGMENTATIONS_TEST = Compose(
    [Resize(img_size, img_size, p=1.0), ToFloat(max_value=255), ToTensorV2()],
    p=1.0,
)




## === cell 5
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.model.classifier = nn.Identity()
        self.dense_output = nn.Linear(1280, 4)  # 1280 is the feature dim of B0

    def forward(self, x):
        feat = self.model(x)  # shape (batch, 1280)
        return self.dense_output(feat)




## === cell 6
batch_size = 128
val_batch_size = 256
num_workers = 8
persistent = True

train_dataset = Alaska2Dataset(
    train_files, train_labels, augmentations=AUGMENTATIONS_TRAIN
)
val_dataset = Alaska2Dataset(val_files, val_labels, augmentations=AUGMENTATIONS_TEST)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=persistent,
    prefetch_factor=2,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=persistent,
    prefetch_factor=2,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)

if hasattr(torch, "compile"):
    model = torch.compile(model)

ckpt_path = "../input/alaska/epoch_9_val_loss_6.67_auc_0.798.pth"
if os.path.exists(ckpt_path):
    try:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("External checkpoint loaded.")
    except Exception as e:
        print(f"Failed to load external checkpoint: {e}")
else:
    print("External checkpoint not found, using random weights.")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

num_epochs = 20
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

scaler = GradScaler()




## === cell 7
def alaska_weighted_auc(y_true, y_pred):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, _ = metrics.roc_curve(y_true, y_pred, pos_label=1)
    areas = np.diff(tpr_thresholds)
    normalization = np.dot(areas, weights)

    metric = 0.0
    for i, w in enumerate(weights):
        y_min, y_max = tpr_thresholds[i], tpr_thresholds[i + 1]
        mask = (tpr > y_min) & (tpr < y_max)
        if not np.any(mask):
            continue
        x_pad = np.linspace(fpr[mask][-1], 1, 100)
        x = np.concatenate([fpr[mask], x_pad])
        y = np.concatenate([tpr[mask], np.full_like(x_pad, y_max)])
        y = y - y_min
        sub = metrics.auc(x, y) * w
        metric += sub
    return metric / normalization




## === cell 8
criterion = nn.CrossEntropyLoss()
train_losses, val_losses = [], []

best_auc = -np.inf
best_checkpoint_path = None

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs = imgs.to(device, dtype=torch.float, non_blocking=True)
        labels = labels.to(device, dtype=torch.long, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    model.eval()
    val_running = 0.0
    all_y, all_pred = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            inputs = imgs.to(device, dtype=torch.float, non_blocking=True)
            labels = labels.to(device, dtype=torch.long, non_blocking=True)
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            val_running += loss.item()

            probs = F.softmax(outputs, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            binary_true = (labels.cpu().numpy() != 0).astype(int)
            binary_pred = np.where(preds == 0, probs[:, 0], probs[:, 1:].sum(axis=1))
            all_y.extend(binary_true)
            all_pred.extend(binary_pred)

    val_epoch_loss = val_running / len(val_loader)
    val_losses.append(val_epoch_loss)

    auc_score = alaska_weighted_auc(np.array(all_y), np.array(all_pred))
    print(
        f"Epoch {epoch+1}/{num_epochs} - Train loss: {epoch_loss:.4f}, "
        f"Val loss: {val_epoch_loss:.4f}, Weighted AUC: {auc_score:.4f}"
    )

    checkpoint_name = (
        f"epoch_{epoch+1}_val_loss_{val_epoch_loss:.3f}_auc_{auc_score:.3f}.pth"
    )
    torch.save(model.state_dict(), checkpoint_name)

    if auc_score > best_auc:
        best_auc = auc_score
        best_checkpoint_path = checkpoint_name
        print(
            f"*** New best model saved: {best_checkpoint_path} (AUC={best_auc:.4f}) ***"
        )

    scheduler.step()

if best_checkpoint_path and os.path.exists(best_checkpoint_path):
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=device))
    print(f"Loaded best model from {best_checkpoint_path} for final prediction.")
else:
    print("Best checkpoint not found; using the last epoch model.")



## === cell 9
if train_losses and val_losses:
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label="train loss", color="r")
    plt.plot(val_losses, label="val loss", color="b")
    plt.legend()
    plt.title("Loss curves")
    plt.show()




## === cell 10
class Alaska2TestDataset(Dataset):
    def __init__(self, files, augmentations=None):
        self.files = files
        self.augment = augmentations

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        im = cv2.imread(fn)
        if im is None:
            raise FileNotFoundError(f"Test image not found: {fn}")
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        if self.augment:
            im = self.augment(image=im)["image"]
        else:
            im = ToTensorV2()(image=im)["image"]
        return im


test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_dataset = Alaska2TestDataset(test_filenames, augmentations=AUGMENTATIONS_TEST)
test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)



## === cell 11
model.eval()
all_preds = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        inputs = batch.to(device, dtype=torch.float, non_blocking=True)

        h_flip = inputs.flip(dims=[3])
        v_flip = inputs.flip(dims=[2])
        combined = torch.cat([inputs, h_flip, v_flip], dim=0)  # (3*batch, C, H, W)

        combined_out = model(combined)
        out, out_h, out_v = torch.split(combined_out, inputs.shape[0], dim=0)

        ensemble = (out + out_h + out_v) / 3.0
        probs = F.softmax(ensemble, dim=1).cpu().numpy()
        all_preds.append(probs)

all_preds = np.concatenate(all_preds, axis=0)
pred_labels = all_preds.argmax(axis=1)

final_scores = np.zeros(len(all_preds))
final_scores[pred_labels == 0] = all_preds[pred_labels == 0, 0]
mask = pred_labels != 0
final_scores[mask] = all_preds[mask, 1:].sum(axis=1)

test_df = pd.DataFrame({"ImageFileName": test_filenames})
test_df["Id"] = test_df["ImageFileName"].apply(lambda x: os.path.basename(x))
test_df["Label"] = final_scores
submission_path = "submission.csv"
test_df[["Id", "Label"]].to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(test_df.head())
