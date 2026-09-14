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

3.9

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

# 5. Code solution

## === cell 0
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from PIL import Image
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trim(img):
    return img


threshold = [0.7, 1.5, 2.5, 3.5]


def regress2class(out_tensor):
    """
    Convert a regression tensor (already clamped to [0,4]) to integer classes
    using the predefined thresholds.
    """
    out = out_tensor.clone()
    classes = torch.zeros_like(out, dtype=torch.long)
    for i, t in enumerate(threshold):
        classes[out > t] = i + 1
    classes[out > threshold[-1]] = 4
    return classes


def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super().__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x


class Regressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("tf_efficientnet_b5_ns", pretrained=True)
        self.backbone.global_pool = GeM(flatten=True)
        self.regressor = nn.Linear(1000, 1)

    def forward(self, x):
        x = self.backbone(x)
        out = self.regressor(x)
        out = torch.sigmoid(out) * 4.0
        return out




## === cell 1
net = Regressor()
weight_path = "../input/weights/D5_regre_70epoch.pkl"
try:
    state_dict = torch.load(weight_path, map_location=device)
    net.load_state_dict(state_dict)
    print("Custom weights loaded.")
except FileNotFoundError:
    print(f"Weight file not found at {weight_path}. Using pretrained backbone only.")
net = net.to(device)




## === cell 2
input_size = 384
transform = transforms.Compose(
    [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)


class ImageDataset(Dataset):
    """Dataset used for inference – returns only transformed images."""

    def __init__(self, ids, base_path):
        self.ids = ids
        self.base_path = base_path

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_code = self.ids[idx]
        img_path = f"{self.base_path}/{id_code}.png"
        img = Image.open(img_path).convert("RGB")
        img = trim(img)
        return transform(img)


class TrainDataset(Dataset):
    """Dataset for a tiny training pass – returns (image, label)."""

    def __init__(self, ids, labels, base_path):
        self.ids = ids
        self.labels = labels.astype(np.float32)
        self.base_path = base_path

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_code = self.ids[idx]
        img_path = f"{self.base_path}/{id_code}.png"
        img = Image.open(img_path).convert("RGB")
        img = trim(img)
        img_tensor = transform(img)
        label = self.labels[idx]
        return img_tensor, torch.tensor(label, dtype=torch.float32)




## === cell 3
train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
train_ids = train_df["id_code"].values
train_labels = train_df["diagnosis"].values.astype(np.float32)

train_dataset = TrainDataset(
    train_ids, train_labels, "../input/aptos2019-blindness-detection/train_images"
)
train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False
)

net.train()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
criterion = nn.MSELoss()

print("Starting tiny training pass (1 epoch)...")
for epoch in range(1):
    epoch_loss = 0.0
    for imgs, lbls in train_loader:
        imgs = imgs.to(device)
        lbls = lbls.to(device).unsqueeze(1)  # shape [B,1]
        preds = net(imgs)
        loss = criterion(preds, lbls)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)
    epoch_loss /= len(train_loader.dataset)
    print(f"Epoch {epoch+1} completed. Avg loss: {epoch_loss:.4f}")

net.eval()




## === cell 4
def batch_predict(ids, base_path, batch_size=32, num_workers=0):
    """Return raw regression outputs for given ids using a safe DataLoader."""
    dataset = ImageDataset(ids, base_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    raw_outputs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = net(batch).squeeze(1)  # shape [B]
            raw_outputs.append(out.cpu().numpy())
    return np.concatenate(raw_outputs).astype(np.float32)




## === cell 5
raw_preds = batch_predict(
    train_ids, "../input/aptos2019-blindness-detection/train_images", batch_size=32
)

A = np.vstack([raw_preds, np.ones_like(raw_preds)]).T
a, b = np.linalg.lstsq(A, train_labels, rcond=None)[0]
print(f"Calibration parameters: a={a:.4f}, b={b:.4f}")




## === cell 6
test_ids_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = np.squeeze(test_ids_df.values)

raw_test_preds = batch_predict(
    test_ids, "../input/aptos2019-blindness-detection/test_images", batch_size=32
)

calibrated = raw_test_preds * a + b
calibrated = np.clip(calibrated, 0.0, 4.0)

calibrated_tensor = torch.from_numpy(calibrated)
pred_classes = regress2class(calibrated_tensor).numpy().astype(int)

submission = np.column_stack((test_ids, pred_classes))

df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df["diagnosis"] = df["diagnosis"].astype(int)
df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with", len(df), "rows.")
