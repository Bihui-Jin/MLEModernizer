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

# 5. Target score

0.8373268929321263

# 6. Current score

0.72032

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.6813) has done: 'Implemented fixes:
- Replaced missing `efficientnet_pytorch` with torchvision’s EfficientNet implementation.
- Added necessary imports (`torchvision.models`, `PIL.Image`).
- Updated the `Net` class to use `models.efficientnet_b0` and correctly extract features.
- Adjusted dataset classes to convert OpenCV images to PIL before applying transforms, ensuring compatibility with torchvision augmentations.
- Kept the rest of the pipeline unchanged, guaranteeing a valid CSV submission.'
- What this solution (achieved 0.72032) has done: 'I add ImageNet‑style normalization to the training and test augmentations, raise the training epochs to three, and simplify the post‑processing so the weighted AUC uses the summed probability of the three stego classes directly (instead of relying on the predicted label). These minimal tweaks keep the EfficientNet‑B0 backbone unchanged while giving the model better calibrated inputs and more learning time, which should raise the validation Weighted AUC toward the target score.'

# 9. Code solution

## === cell 0
import os, random, gc, glob
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from sklearn import metrics
from tqdm.notebook import tqdm
import cv2, matplotlib.pyplot as plt
from PIL import Image

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ["PYTHONHASHSEED"] = str(seed)




## === cell 1
data_dir = "/kaggle/input/alaska2-image-steganalysis"
img_size = 512
sample_size_per_class = 5000  # reduced size for quick run
val_ratio = 0.25

folder_names = ["Cover", "JMiPOD", "JUNIWARD", "UERD"]  # label 0‑3
train_fns, train_labels = [], []
val_fns, val_labels = [], []

for label, folder in enumerate(folder_names):
    all_files = sorted(glob.glob(os.path.join(data_dir, folder, "*.jpg")))
    if len(all_files) > sample_size_per_class:
        all_files = all_files[:sample_size_per_class]
    random.shuffle(all_files)
    split = int(len(all_files) * val_ratio)
    val_fns.extend(all_files[:split])
    val_labels.extend([label] * split)
    train_fns.extend(all_files[split:])
    train_labels.extend([label] * (len(all_files) - split))

train_df = pd.DataFrame({"ImageFileName": train_fns, "Label": train_labels})
val_df = pd.DataFrame({"ImageFileName": val_fns, "Label": val_labels})

print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")




## === cell 2
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

train_transform = T.Compose(
    [
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)

test_transform = T.Compose(
    [
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ]
)


class Alaska2Dataset(Dataset):
    def __init__(self, df, augmentations=None):
        self.df = df.reset_index(drop=True)
        self.augment = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn = self.df.loc[idx, "ImageFileName"]
        img = cv2.imread(fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.augment:
            img = self.augment(img)
        else:
            img = T.ToTensor()(img)
            img = T.Normalize(mean=imagenet_mean, std=imagenet_std)(img)
        return {"image": img, "label": self.df.loc[idx, "Label"]}


class Alaska2TestDataset(Dataset):
    def __init__(self, df, augmentations=None):
        self.df = df.reset_index(drop=True)
        self.augment = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fn = self.df.loc[idx, "ImageFileName"]
        img = cv2.imread(fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        if self.augment:
            img = self.augment(img)
        else:
            img = T.ToTensor()(img)
            img = T.Normalize(mean=imagenet_mean, std=imagenet_std)(img)
        return {"image": img, "filename": fn}




## === cell 3
class Net(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.backbone = models.efficientnet_b0(pretrained=True)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        feats = self.backbone.features(x)
        pooled = F.adaptive_avg_pool2d(feats, 1).view(x.size(0), -1)
        return self.fc(pooled)


device = "cuda" if torch.cuda.is_available() else "cpu"
model = Net().to(device)




## === cell 4
batch_size = 32
num_workers = 2
train_loader = DataLoader(
    Alaska2Dataset(train_df, augmentations=train_transform),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)
val_loader = DataLoader(
    Alaska2Dataset(val_df, augmentations=test_transform),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)


def alaska_weighted_auc(y_true, y_score):
    tpr_thr = [0.0, 0.4, 1.0]
    weights = [2, 1]
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score, pos_label=1)
    areas = np.diff(tpr_thr)
    norm = np.dot(areas, weights)
    metric = 0.0
    for i, w in enumerate(weights):
        y_min, y_max = tpr_thr[i], tpr_thr[i + 1]
        mask = (tpr > y_min) & (tpr <= y_max)
        if not np.any(mask):
            continue
        x_pad = np.linspace(fpr[mask][-1], 1, 100)
        x = np.concatenate([fpr[mask], x_pad])
        y = np.concatenate([tpr[mask], np.full_like(x_pad, y_max)])
        y = y - y_min
        sub = metrics.auc(x, y) * w
        metric += sub
    return metric / norm


num_epochs = 3  # increased from 1 to give the model more learning time
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch in tqdm(train_loader, desc=f"Train Epoch {epoch+1}"):
        imgs = batch["image"].to(device)
        lbls = batch["label"].to(device, dtype=torch.long)
        optimizer.zero_grad()
        outs = model(imgs)
        loss = criterion(outs, lbls)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1} train loss: {running_loss/len(train_loader):.4f}")

    model.eval()
    val_losses, all_labels, all_preds = [], [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validate"):
            imgs = batch["image"].to(device)
            lbls = batch["label"].to(device, dtype=torch.long)
            outs = model(imgs)
            loss = criterion(outs, lbls)
            val_losses.append(loss.item())
            probs = F.softmax(outs, dim=1).cpu().numpy()
            all_preds.append(probs)
            all_labels.append(lbls.cpu().numpy())
    val_loss = np.mean(val_losses)
    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    binary_labels = (labels_arr != 0).astype(int)
    custom_score = preds_arr[:, 1:].sum(axis=1)
    auc = alaska_weighted_auc(binary_labels, custom_score)
    print(f"Val loss: {val_loss:.4f}, Weighted AUC: {auc:.4f}")




## === cell 5
test_files = sorted(glob.glob(os.path.join(data_dir, "Test", "*.jpg")))
test_df = pd.DataFrame({"ImageFileName": test_files})

test_loader = DataLoader(
    Alaska2TestDataset(test_df, augmentations=test_transform),
    batch_size=32,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
)

model.eval()
test_preds = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Test"):
        imgs = batch["image"].to(device)
        outs = model(imgs)
        probs = F.softmax(outs, dim=1).cpu().numpy()
        test_preds.append(probs)

test_preds = np.concatenate(test_preds)
final_score = test_preds[:, 1:].sum(axis=1)

test_df["Id"] = test_df["ImageFileName"].apply(lambda x: os.path.basename(x))
test_df["Label"] = final_score
submission_path = "submission.csv"
test_df[["Id", "Label"]].to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
print(test_df.head())
