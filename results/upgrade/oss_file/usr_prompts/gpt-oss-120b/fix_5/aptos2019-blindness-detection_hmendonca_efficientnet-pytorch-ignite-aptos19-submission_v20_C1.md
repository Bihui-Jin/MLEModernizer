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
scipy==1.15.3
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

0.7005091888778096

# 6. Current score

0.88622

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.89652) has done: 'The update switches to GPU when available, enables CuDNN benchmarking, and caps the DataLoader workers to a reasonable number while using persistent workers to avoid repeated process start‑ups. These changes keep the exact model, training loop, and evaluation logic intact but dramatically reduce data‑loading and compute overhead, allowing the script to finish well within the 600‑second limit.'
- What this solution (achieved 0.88622) has done: 'I slightly reduce the model’s training time by using only one epoch and skip the quadratic‑kappa threshold optimisation (keeping the default thresholds). These minimal changes keep the core architecture and evaluation logic intact while lowering the validation score, moving it closer to the target 0.7005.'

# 9. Code solution

## === cell 0
import os, glob, pandas as pd, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import cohen_kappa_score

print("data files:", glob.glob("../input/aptos2019-blindness-detection/*"))




## === cell 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device:", device)

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True




## === cell 2
image_size = 380
train_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081]),
    ]
)
test_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081]),
    ]
)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, ids, targets=None, transform=None, ext=".png"):
        self.root = root
        self.ids = ids
        self.targets = torch.LongTensor(targets) if targets is not None else None
        self.transform = transform
        self.ext = ext
        if self.targets is not None:
            assert len(self.ids) == len(self.targets)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.ids[idx] + self.ext)
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.targets is not None:
            return img, self.targets[idx]
        return img, torch.tensor([])




## === cell 3
train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
test_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
train_ids = train_df["id_code"].values
train_labels = train_df["diagnosis"].values
test_ids = test_df["id_code"].values

train_dataset = ImageDataset(
    root="../input/aptos2019-blindness-detection/train_images",
    ids=train_ids,
    targets=train_labels,
    transform=train_transform,
)
test_dataset = ImageDataset(
    root="../input/aptos2019-blindness-detection/test_images",
    ids=test_ids,
    transform=test_transform,
)

val_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_subset, val_subset = random_split(
    train_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

batch_size = 16
max_workers = max(1, min(8, os.cpu_count()))

train_loader = DataLoader(
    train_subset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=max_workers,
    pin_memory=device.type == "cuda",
    persistent_workers=True,
)
val_loader = DataLoader(
    val_subset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=max_workers,
    pin_memory=device.type == "cuda",
    persistent_workers=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=max_workers,
    pin_memory=device.type == "cuda",
    persistent_workers=True,
)




## === cell 4
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)  # 5 classes
model = model.to(device).train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 1
for epoch in range(epochs):
    running_loss = 0.0
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{epochs} – Train loss: {epoch_loss:.4f}")




## === cell 5
model.eval()
val_preds = []
val_true = []
with torch.no_grad():
    for imgs, targets in val_loader:
        imgs = imgs.to(device)
        logits = model(imgs)  # shape [B,5]
        probs = torch.softmax(logits, dim=1)
        rating = torch.arange(5, device=device, dtype=probs.dtype)  # [0,1,2,3,4]
        expected = (probs * rating).sum(dim=1)  # continuous prediction
        val_preds.append(expected.cpu())
        val_true.append(targets)
val_preds = torch.cat(val_preds).numpy()
val_true = torch.cat(val_true).numpy()




## === cell 6
class KappaOptimizer(nn.Module):
    def __init__(self, coef=None):
        super().__init__()
        self.coef = coef if coef is not None else [0.5, 1.5, 2.5, 3.5]

    def _predict(self, coef, preds):
        preds = np.array(preds)
        y_hat = np.digitize(preds, bins=coef, right=False)
        return y_hat.astype(int)

    def predict(self, preds):
        return self._predict(self.coef, preds)

    @staticmethod
    def _quad_kappa(coef, preds, y):
        y_hat = np.digitize(preds, bins=coef, right=False).astype(int)
        return cohen_kappa_score(y, y_hat, weights="quadratic")

    def fit(self, preds, y):
        print("Skipping threshold optimisation; using default thresholds:", self.coef)


kappa_opt = KappaOptimizer()




## === cell 7
model.eval()
test_preds = []
with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)
        rating = torch.arange(5, device=device, dtype=probs.dtype)
        expected = (probs * rating).sum(dim=1)
        test_preds.append(expected.cpu())
test_preds = torch.cat(test_preds).numpy()

test_labels = kappa_opt.predict(test_preds)




## === cell 8
sub_path = "../input/aptos2019-blindness-detection/sample_submission.csv"
submission = pd.read_csv(sub_path)
submission["diagnosis"] = test_labels.astype(int)
submission.to_csv("submission.csv", index=False)
print("submission.csv saved, shape:", submission.shape)
