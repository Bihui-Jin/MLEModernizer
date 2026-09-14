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

0.92318552899077

# 6. Current score

0.74924

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.06372) has done: 'The fix adds a safe device selection (CPU fallback), removes the failing weight‑file load (using a try/except) and switches to a pretrained EfficientNet backbone so the model can produce sensible predictions. The inference loop now uses the classifier output to choose the most likely class, guaranteeing a non‑empty submission CSV. All changes are minimal and preserve the original model structure.'
- What this solution (achieved 0.6925) has done: 'The changes focus on eliminating the per‑image inference loop and speeding up data loading. We enable cuDNN benchmarking, increase the training batch size, and create a batched test DataLoader so predictions are computed for whole batches instead of one image at a time. This keeps the model architecture and training logic untouched while dramatically reducing I/O and GPU overhead, ensuring the script finishes well within the 600‑second limit.'
- What this solution (achieved 0.68785) has done: 'I keep the existing model and training unchanged, but improve the inference by combining the classifier’s softmax probabilities with the ordinal‑based class probabilities (computed by `ordinal2class_prob`). Averaging these two complementary predictions usually yields a better ordered prediction, which should raise the quadratic weighted kappa toward the target while preserving the core logic and output format.'
- What this solution (achieved -0.09538) has done: 'I preload all training and test images into memory once, removing the per‑epoch disk I/O that caused the long runtime, and I drop the extra workers (num_workers=0) since data is now in RAM. The rest of the logic—including model architecture, training loops, and inference‑combination steps—remains unchanged, so the predictions stay identical.'
- What this solution (achieved 0.82073) has done: 'The fix reduces the batch sizes used for both training and inference to prevent CUDA out‑of‑memory failures, adds a `torch.cuda.empty_cache()` call before inference to free any leftover memory, and renumbers the cells so the script runs end‑to‑end and creates the required `submission.csv`. No core model logic is changed.'
- What this solution (achieved 0.74924) has done: 'I add a lightweight combination of the classifier soft‑max probabilities with ordinal‑based class probabilities during inference. This keeps the model architecture unchanged, only adjusts post‑processing to better capture the ordinal nature of the labels, which should raise the quadratic weighted kappa and move the score closer to the target. The change is limited to the test inference loop and the creation of the submission dataframe.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
import timm

try:
    from timm.layers import GeM

    gem_available = True
except Exception:
    gem_available = False

    class GeM(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, x):
            return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

train_csv_path = "../input/aptos2019-blindness-detection/train.csv"
test_csv_path = "../input/aptos2019-blindness-detection/test.csv"
train_images_dir = "../input/aptos2019-blindness-detection/train_images"
test_images_dir = "../input/aptos2019-blindness-detection/test_images"

train_df = pd.read_csv(train_csv_path)
test_ids = pd.read_csv(test_csv_path)["id_code"].values

input_size = 384
transform = transforms.Compose(
    [
        transforms.Resize((input_size * 3 // 4, input_size)),  # keep aspect 4:3
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)


class AptosDataset(Dataset):
    def __init__(self, df, images_dir, transform=None):
        self.ids = df["id_code"].values
        self.labels = df["diagnosis"].values
        self.transform = transform
        self.tensors = []
        for img_id in self.ids:
            img_path = f"{images_dir}/{img_id}.png"
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            self.tensors.append(img)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]


class TestDataset(Dataset):
    def __init__(self, ids, images_dir, transform=None):
        self.ids = ids
        self.transform = transform
        self.tensors = []
        for img_id in self.ids:
            img_path = f"{images_dir}/{img_id}.png"
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            self.tensors.append(img)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.tensors[idx], self.ids[idx]


class ThreeStage_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=True, num_classes=0
        )
        num_features = self.backbone.num_features
        self.classifier = nn.Linear(num_features, 5)  # classification logits
        self.regressor = nn.Linear(num_features, 4)  # regression outputs for thresholds
        self.ordinal = nn.Linear(num_features, 4)  # ordinal logits (before sigmoid)

    def forward(self, x):
        feats = self.backbone(x)  # (B, C)
        c_out = self.classifier(feats)  # (B,5)
        r_out = self.regressor(feats)  # (B,4)
        o_out = self.ordinal(feats)  # (B,4)
        return c_out, r_out, o_out


net = ThreeStage_Model()
if gem_available:
    net.backbone.global_pool = GeM(flatten=True)

weights_path = "../input/weights/B4_3stage_17epoch_320finetune.pkl"
loaded_weights = False
try:
    state = torch.load(weights_path, map_location=device)
    net.load_state_dict(state)
    loaded_weights = True
    print("Loaded fine‑tuned weights.")
except Exception as e:
    print(f"Could not load weights ({e}); will train classifier head on the fly.")

net = net.to(device)

train_batch_size = 32
test_batch_size = 64

if not loaded_weights:
    net.backbone.eval()
    net.backbone.requires_grad_(False)

    train_dataset = AptosDataset(train_df, train_images_dir, transform=transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.classifier.parameters(), lr=1e-4)

    epochs = 8
    net.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            c_out, _, _ = net(imgs)
            loss = criterion(c_out, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * imgs.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"[Classifier] Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    net.backbone.requires_grad_(True)
    optimizer_full = torch.optim.Adam(net.parameters(), lr=1e-5)

    ft_epochs = 5
    for epoch in range(ft_epochs):
        epoch_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer_full.zero_grad()
            c_out, _, _ = net(imgs)
            loss = criterion(c_out, labels)
            loss.backward()
            optimizer_full.step()
            epoch_loss += loss.item() * imgs.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"[Fine‑tune] Epoch [{epoch+1}/{ft_epochs}] - Loss: {avg_loss:.4f}")

net.eval()  # evaluation mode for inference

if device.type == "cuda":
    torch.cuda.empty_cache()

test_dataset = TestDataset(test_ids, test_images_dir, transform=transform)


def test_collate_fn(batch):
    imgs, ids = zip(*batch)
    imgs_tensor = torch.stack(imgs, dim=0)
    ids_list = list(ids)
    return imgs_tensor, ids_list


test_loader = DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    collate_fn=test_collate_fn,
)

submission = []
with torch.no_grad():
    for imgs, ids_batch in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        c_out, _, o_out = net(imgs)

        cls_prob = F.softmax(c_out, dim=1)  # (B,5)

        cum_prob = torch.sigmoid(o_out)  # (B,4)
        prob0 = 1.0 - cum_prob[:, 0]
        prob1 = cum_prob[:, 0] - cum_prob[:, 1]
        prob2 = cum_prob[:, 1] - cum_prob[:, 2]
        prob3 = cum_prob[:, 2] - cum_prob[:, 3]
        prob4 = cum_prob[:, 3]
        ord_prob = torch.stack([prob0, prob1, prob2, prob3, prob4], dim=1)  # (B,5)

        combined_prob = (cls_prob + ord_prob) / 2.0

        preds = torch.argmax(combined_prob, dim=1).cpu().numpy()

        for img_id, pred in zip(ids_batch, preds):
            submission.append([img_id, int(pred)])

submission = np.array(submission)



## === cell 1
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv with", len(df), "rows.")
