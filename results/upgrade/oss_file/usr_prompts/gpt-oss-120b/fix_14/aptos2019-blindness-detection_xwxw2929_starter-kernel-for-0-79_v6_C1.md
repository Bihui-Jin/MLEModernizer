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

No external packages required in the script and installed.

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

0.9020970980040276

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I import the torchvision model utilities and replace the undefined EfficientNet call with the proper torchvision EfficientNet implementation, adjusting its final layer for 5 classes. This resolves the NameError preventing training and downstream evaluation, letting the script run end‑to‑end and produce a valid `submission.csv`. No other logic is altered.'
- What this solution (achieved 0.0) has done: 'I increase the training length modestly and change the prediction step to use the expected class from the soft‑max probabilities (rounded to the nearest integer). This keeps the model architecture unchanged while usually improving the quadratic weighted kappa, moving the score upward toward the target. The script is renumbered into consecutive cells and now writes a proper `submission.csv`.'
- What this solution (achieved 0.0) has done: 'I replace the soft‑max‑based expected‑value rounding with a direct arg‑max class prediction (the usual approach for classification) in both validation and test steps, and extend training to 20 epochs to give the model a bit more learning without altering its architecture. These minimal changes keep the core logic intact while likely improving the quadratic weighted kappa toward the target score.'
- What this solution (achieved 0.0) has done: 'I add a fixed random seed for reproducibility, extend training to 30 epochs (still modest), and replace the arg‑max class prediction on validation and test with an expected‑value rounding that better respects the ordinal nature of the labels. This small adjustment should give the model a non‑zero quadratic weighted kappa, moving the score toward the target without altering the core architecture or training loop.'
- What this solution (achieved 0.0) has done: 'I replace the soft‑max expected‑value rounding with a direct arg‑max class selection in both validation and test prediction steps. This minimal change keeps the model architecture and training unchanged while producing more accurate class predictions, which should raise the quadratic weighted kappa from 0 toward the target score.'
- What this solution (achieved 0.0) has done: 'The patch adds a few lightweight tweaks that keep the original architecture and training loop but help the model learn a bit better and produce more suitable ordinal predictions.  
* Class‑frequency weighting is applied to the cross‑entropy loss to reduce the impact of the heavily skewed classes.  
* During validation and test we compute an expected‑value from the soft‑max probabilities and round it, which is more appropriate for the ordered severity grades than a plain arg‑max.  
* The best‑performing model on the validation split is saved and re‑loaded after training, ensuring the final predictions come from the checkpoint with the highest QWK.  

These changes are minimal, respect the existing pipeline, and aim to move the quadratic weighted kappa from 0 toward the target ≈ 0.902.'
- What this solution (achieved 0.0) has done: 'Implemented targeted speed‑up measures without altering any model architecture, training loop, or evaluation logic.

**Key changes**
- Enabled CuDNN benchmark for faster CUDA kernels.
- Increased `num_workers` and added `pin_memory` to DataLoaders for parallel image loading.
- Determined an optimal worker count based on CPU cores.
- Minor code refactor to construct DataLoaders after setting these parameters, preserving all original functionality and results.'

# 9. Code solution

## === cell 0
import os
import warnings
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
import torchvision.models as tv_models  # EfficientNet models
import pandas as pd
from PIL import Image
from sklearn.metrics import cohen_kappa_score

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
warnings.filterwarnings("ignore")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def locate_csv(*candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    for path in candidates:
        alt = os.path.join("/kaggle/input", path)
        if os.path.exists(alt):
            return alt
    for path in candidates:
        name = os.path.basename(path)
        for root, _, files in os.walk("."):
            if name in files:
                return os.path.join(root, name)
    raise FileNotFoundError(f"None of the candidate CSV files exist: {candidates}")


base_dir = os.path.join("data", "aptos2019-blindness-detection")
train_img_dir = os.path.join(base_dir, "train_images")
test_img_dir = os.path.join(base_dir, "test_images")

train_csv = locate_csv(
    os.path.join(base_dir, "train.csv"),
    os.path.join("data", "train.csv"),
    "train.csv",
)
test_csv = locate_csv(
    os.path.join(base_dir, "test.csv"),
    os.path.join("data", "test.csv"),
    "test.csv",
)

train_df = pd.read_csv(train_csv)
train_df["path"] = train_df["id_code"].apply(
    lambda x: os.path.join(train_img_dir, f"{x}.png")
)
test_df = pd.read_csv(test_csv)
test_df["path"] = test_df["id_code"].apply(
    lambda x: os.path.join(test_img_dir, f"{x}.png")
)

train_transform = T.Compose(
    [
        T.Resize(256),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
test_transform = T.Compose(
    [
        T.Resize(256),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_frac = 0.2
val_size = int(len(train_df) * val_frac)
train_part = train_df.iloc[:-val_size]
val_part = train_df.iloc[-val_size:]


class DRDataset(data.Dataset):
    def __init__(self, df, transform=None, train=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["path"]
        if os.path.exists(img_path):
            img = Image.open(img_path).convert("RGB")
        else:
            img = Image.new("RGB", (256, 256), (127, 127, 127))
        if self.transform:
            img = self.transform(img)
        if self.train:
            label = int(row["diagnosis"])
            return img, label
        else:
            return img, row["id_code"]


cpu_count = os.cpu_count() or 1
num_workers = min(4, cpu_count)

train_dataset = DRDataset(train_part, transform=train_transform, train=True)
val_dataset = DRDataset(val_part, transform=test_transform, train=True)
test_dataset = DRDataset(test_df, transform=test_transform, train=False)

train_loader = data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)
val_loader = data.DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)
test_loader = data.DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)



## === cell 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = tv_models.efficientnet_b5(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
model = model.to(device)

class_counts = train_df["diagnosis"].value_counts().sort_index()
class_weights = 1.0 / class_counts.values
class_weights = class_weights / class_weights.sum() * len(class_weights)  # normalize
criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(class_weights, dtype=torch.float).to(device)
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

best_kappa = -np.inf
best_state = None

model.train()
for epoch in range(30):  # 30 epochs as before
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(
            device, non_blocking=True
        )
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    model.eval()
    val_preds, val_targets = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            outputs = model(imgs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(labels.numpy())
    epoch_kappa = cohen_kappa_score(val_targets, val_preds, weights="quadratic")
    print(f"Epoch {epoch+1} completed – Validation QWK: {epoch_kappa:.4f}")
    if epoch_kappa > best_kappa:
        best_kappa = epoch_kappa
        best_state = model.state_dict()
    model.train()

if best_state is not None:
    model.load_state_dict(best_state)
print(f"Best Validation QWK achieved: {best_kappa:.4f}")



## === cell 2
model.eval()
val_preds = []
val_targets = []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        val_preds.extend(preds)
        val_targets.extend(labels.numpy())
val_kappa = cohen_kappa_score(val_targets, val_preds, weights="quadratic")
print(f"Final Validation QWK (using best model): {val_kappa:.4f}")



## === cell 3
model.eval()
test_ids = []
test_preds = []
with torch.no_grad():
    for imgs, ids in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        outputs = model(imgs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        test_ids.extend(ids)
        test_preds.extend(preds)

submission = pd.DataFrame({"id_code": test_ids, "diagnosis": test_preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
