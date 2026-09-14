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

0.9145905451464686

# 6. Current score

0.59679

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'The script now safely selects CPU when a GPU isn’t available, gracefully handles a missing weight file by loading the model with random weights, and ensures a non‑empty submission CSV is written. Minor adjustments keep the original model architecture and processing logic intact while fixing the runtime errors that prevented a valid submission.'
- What this solution (achieved 0.0) has done: 'The script was missing all required imports and the device definition, causing NameError failures throughout the notebook. I added the necessary `torch`, `torch.nn`, `torch.nn.functional`, `torchvision`, `timm`, `pandas`, `numpy`, `PIL`, `random`, and `math` imports, defined `device` to use CUDA when available, and reorganized the code into numbered cells while keeping the original model and prediction logic unchanged. This fixes the runtime errors and ensures a valid `submission.csv` is written.'
- What this solution (achieved -0.00029) has done: 'I switch the EfficientNet backbone to use ImageNet‑pretrained weights (instead of random initialization) and simplify the prediction step to use the model’s built‑in 5‑class classifier rather than the regression output. This small change keeps the original architecture untouched while providing much stronger features and a more appropriate class prediction, moving the validation score toward the target without altering the core training logic.'
- What this solution (achieved 0.0) has done: 'I set the script to skip the untrained model and use the most‑common diagnosis from the training set as a fallback for all test images. Since the model’s random predictions give a near‑zero QWK, always predicting the mode provide a positive score and move the result closer to the target. The change is limited to the flag that controls weight usage, preserving the rest of the pipeline unchanged.'
- What this solution (achieved -0.00727) has done: 'I replace the constant fallback prediction with a simple intensity‑based heuristic: compute the average pixel brightness of each test image and map it to a diagnosis using four thresholds. This keeps the original model architecture untouched, still uses the untrained network when weights are unavailable, and provides more varied predictions that are likely to move the Quadratic Weighted Kappa score toward the target.'
- What this solution (achieved 0.44502) has done: 'The changes pre‑compute the frozen EfficientNet backbone features once for the whole training set, then train the classifier only on these fixed features. This eliminates repeated heavy backbone forward passes across epochs, cutting runtime dramatically while preserving exactly the same model architecture and final predictions. The test inference now also uses the pre‑computed backbone features per batch, avoiding redundant computation. No training logic, loss function, or data paths are altered.'
- What this solution (achieved 0.49083) has done: 'I raise the validation‑type performance by training the frozen EfficientNet classifier longer and with a simple learning‑rate decay, plus class‑balanced loss weights. These tweaks keep the original architecture and inference flow unchanged while giving the model more capacity to learn from the pre‑computed features, which should move the Quadratic Weighted Kappa score closer to the target.'
- What this solution (achieved 0.59679) has done: 'I keep the overall architecture and training pipeline unchanged but extend training to capture more patterns and add a cheap test‑time augmentation (horizontal flip) that is averaged with the original predictions. These modest adjustments are expected to raise the validation QWK toward the target without altering the core model logic.'

# 9. Code solution

## === cell 0
import os
import math
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import torch.utils.data as data
import torch.optim as optim

import torchvision.transforms as transforms

import timm

torch.manual_seed(42)
torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    """Convert regression output to class label using thresholds."""
    prediction = torch.zeros(out.size(0), device=out.device)
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = (1 - out[:, 0]).squeeze()
    pred_prob[:, 1] = (out[:, 0] * (1 - out[:, 1])).squeeze()
    pred_prob[:, 2] = (out[:, 1] * (1 - out[:, 2])).squeeze()
    pred_prob[:, 3] = (out[:, 2] * (1 - out[:, 3])).squeeze()
    pred_prob[:, 4] = out[:, 3].squeeze()
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)
    for i in range(out.size(0)):
        if out[i] < 4.0:
            l1 = int(math.floor(out[i].item()))
            l2 = int(math.ceil(out[i].item()))
            pred_prob[i][l1] = 1 - (out[i] - l1)
            pred_prob[i][l2] = 1 - (l2 - out[i])
        else:
            pred_prob[i][4] = 1.0
    return pred_prob




## === cell 1
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.item():.4f}, eps={self.eps})"


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None):
        super(ThreeStage_Model, self).__init__()
        self.backbone = timm.models.tf_efficientnet_b4_ns(pretrained=True)
        self.backbone.global_pool = GeM(flatten=True)

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )
        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )
        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(1000, 500),
            nn.SiLU(),
            nn.Linear(500, 4),
        )
        self.final_regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x, final=False):
        with torch.no_grad():
            x = self.backbone(x)
        c_out = self.classifier(x)
        r_out = self.regressor(x)
        o_out = self.ordinal(x)
        if final:
            out = torch.cat((c_out, r_out, o_out), 1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 4.5
            return out
        else:
            r_out = torch.sigmoid(r_out) * 4.5
            o_out = torch.sigmoid(o_out)
            return c_out, r_out, o_out




## === cell 2
train_path = "../input/aptos2019-blindness-detection/train.csv"
test_ids_path = "../input/aptos2019-blindness-detection/test.csv"

train_df = pd.read_csv(train_path)
test_ids_df = pd.read_csv(test_ids_path)
test_ids = np.squeeze(test_ids_df.values)

input_size = 380
transform = transforms.Compose(
    [
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)


class AptosDataset(data.Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.ids = df["id_code"].values
        self.labels = df["diagnosis"].values
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        label = int(self.labels[idx])
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


train_img_dir = "../input/aptos2019-blindness-detection/train_images"
train_dataset = AptosDataset(train_df, train_img_dir, transform=transform)
train_loader = data.DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    prefetch_factor=2,
)

net = ThreeStage_Model()
net = net.to(device)

for param in net.backbone.parameters():
    param.requires_grad = False
net.backbone.eval()

train_features = []
train_labels = []
with torch.no_grad():
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        feats = net.backbone(imgs)  # (batch, 1000)
        train_features.append(feats.cpu())
        train_labels.append(labels)
train_features = torch.cat(train_features)
train_labels = torch.cat(train_labels)

class_counts = torch.bincount(train_labels.long())
class_weights = 1.0 / class_counts.float()
class_weights = class_weights * (class_counts.sum() / class_weights.sum())
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

feature_dataset = data.TensorDataset(train_features, train_labels)
feature_loader = data.DataLoader(
    feature_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=0,
)

optimizer = optim.Adam(net.classifier.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=10, gamma=0.5
)  # slower decay

epochs = 40  # extended training
net.train()
for epoch in range(epochs):
    running_loss = 0.0
    for feats, labels in feature_loader:
        feats = feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        c_out = net.classifier(feats)
        loss = criterion(c_out, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * feats.size(0)

    epoch_loss = running_loss / len(feature_dataset)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}")
    scheduler.step()

net.eval()
weight_loaded = True  # model trained successfully




## === cell 3
class TestAptosDataset(data.Dataset):
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


test_img_dir = "../input/aptos2019-blindness-detection/test_images"
test_dataset = TestAptosDataset(test_ids, test_img_dir, transform=transform)
test_loader = data.DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    prefetch_factor=2,
)

submission = []
intensity_thresh = [60, 90, 120, 150]  # fallback thresholds (unused if model works)

if weight_loaded:
    net.eval()
    with torch.no_grad():
        for imgs, ids_batch in test_loader:
            imgs = imgs.to(device)

            feats = net.backbone(imgs)
            c_out = net.classifier(feats)

            imgs_flipped = torch.flip(imgs, dims=[3])  # flip width dimension
            feats_flipped = net.backbone(imgs_flipped)
            c_out_flipped = net.classifier(feats_flipped)

            c_out_avg = (c_out + c_out_flipped) / 2.0
            probs = F.softmax(c_out_avg, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()

            for img_id, pred in zip(ids_batch, preds):
                submission.append([img_id, int(pred)])
else:
    for idx in test_ids:
        img_path = os.path.join(test_img_dir, f"{idx}.png")
        img_pil = Image.open(img_path).convert("RGB")
        avg_intensity = np.mean(np.array(img_pil))
        if avg_intensity < intensity_thresh[0]:
            pred = 4
        elif avg_intensity < intensity_thresh[1]:
            pred = 3
        elif avg_intensity < intensity_thresh[2]:
            pred = 2
        elif avg_intensity < intensity_thresh[3]:
            pred = 1
        else:
            pred = 0
        submission.append([idx, pred])

submission = np.array(submission)



## === cell 4
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
output_path = "submission.csv"
df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}, rows: {len(df)}")
