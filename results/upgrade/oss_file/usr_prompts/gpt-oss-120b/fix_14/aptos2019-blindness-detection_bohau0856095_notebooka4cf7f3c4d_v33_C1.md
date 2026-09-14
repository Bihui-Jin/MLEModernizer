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

# 5. Code solution

## === cell 0
import os
import random
import math
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
from torch.cuda import amp




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0))
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu()
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




## === cell 2
def gem(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)


class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, flatten=False):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        self.flatten = flatten

    def forward(self, x):
        x = gem(x, p=self.p, eps=self.eps)
        if self.flatten:
            x = x.flatten(1)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.item():.4f}, eps={self.eps})"


class ThreeStage_Model(nn.Module):
    def __init__(self):
        super(ThreeStage_Model, self).__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b4_ns", pretrained=True, num_classes=0
        )
        self.backbone.global_pool = GeM(flatten=True)
        feat_dim = self.backbone.num_features  # 1792 for this backbone

        self.classifier = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_dim, 500),
            nn.SiLU(),
            nn.Linear(500, 5),
        )
        self.regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_dim, 500),
            nn.SiLU(),
            nn.Linear(500, 1),
        )
        self.ordinal = nn.Sequential(
            nn.SiLU(),
            nn.Linear(feat_dim, 500),
            nn.SiLU(),
            nn.Linear(500, 4),
        )
        self.final_regressor = nn.Sequential(
            nn.SiLU(),
            nn.Linear(10, 1),
        )

    def forward(self, x, final=False):
        x_feat = self.backbone(x)  # (batch, feat_dim)
        c_out = self.classifier(x_feat)

        if final:
            r_out = torch.sigmoid(self.regressor(x_feat)) * 4.5
            o_out = torch.sigmoid(self.ordinal(x_feat))
            out = torch.cat((c_out, r_out, o_out), 1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 4.5
            return out
        else:
            batch = x_feat.size(0)
            dummy_r = torch.zeros(batch, 1, device=x_feat.device)
            dummy_o = torch.zeros(batch, 4, device=x_feat.device)
            return c_out, dummy_r, dummy_o




## === cell 3
input_size = 380


def fast_preprocess(img):
    if random.random() < 0.5:
        img = TF.hflip(img)
    img = TF.resize(img, (input_size * 3 // 4, input_size))
    tensor = TF.to_tensor(img)
    tensor = TF.normalize(tensor, mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094])
    return tensor


class AptosDataset(Dataset):
    """Dataset that preloads all PIL images into RAM to avoid per‑epoch disk I/O."""

    def __init__(self, df, img_dir):
        self.ids = df["id_code"].values
        self.labels = df["diagnosis"].values.astype(np.int64)
        self.img_dir = img_dir
        self.images = [
            Image.open(os.path.join(img_dir, f"{img_id}.png")).convert("RGB")
            for img_id in self.ids
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = self.images[idx]  # already in memory
        img = fast_preprocess(img)  # augmentation & tensor conversion
        label = int(self.labels[idx])
        return img, label


train_csv = "../input/aptos2019-blindness-detection/train.csv"
train_img_dir = "../input/aptos2019-blindness-detection/train_images"

full_df = pd.read_csv(train_csv)
train_idx, val_idx = train_test_split(
    full_df.index, test_size=0.1, stratify=full_df["diagnosis"], random_state=42
)
train_df = full_df.loc[train_idx].reset_index(drop=True)
val_df = full_df.loc[val_idx].reset_index(drop=True)

train_dataset = AptosDataset(train_df, train_img_dir)
val_dataset = AptosDataset(val_df, train_img_dir)

num_workers = 0

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)

net = ThreeStage_Model().to(device)

class_counts = train_df["diagnosis"].value_counts().sort_index()
class_weights = 1.0 / class_counts.values
class_weights = class_weights / class_weights.sum() * len(class_counts)  # normalise
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)




## === cell 4
epochs = 12  # keep original epoch count
scaler = amp.GradScaler()  # fp16 loss scaling

for epoch in range(1, epochs + 1):
    net.train()
    epoch_loss = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with amp.autocast():
            c_out, _, _ = net(imgs)  # classifier logits
            loss = criterion(c_out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item() * imgs.size(0)

    epoch_loss /= len(train_loader.dataset)

    net.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device, non_blocking=True)
            with amp.autocast():
                c_out, _, _ = net(imgs)
            preds = torch.argmax(c_out, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    val_kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    print(
        f"Epoch {epoch}/{epochs} - Train loss: {epoch_loss:.4f} - Val QWK: {val_kappa:.5f}"
    )

net.eval()




## === cell 5
test_ids = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")[
    "id_code"
].values


class TestDataset(Dataset):
    """Test dataset that preloads all images into RAM (no augmentation needed)."""

    def __init__(self, ids, img_dir):
        self.ids = ids
        self.img_dir = img_dir
        self.images = [
            Image.open(os.path.join(img_dir, f"{id_code}.png")).convert("RGB")
            for id_code in self.ids
        ]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = fast_preprocess(img)  # same preprocessing as training (no random flip)
        return img, self.ids[idx]


test_img_dir = "../input/aptos2019-blindness-detection/test_images"

test_dataset = TestDataset(test_ids, test_img_dir)
test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
)

submission = []
net.eval()
with torch.no_grad():
    for imgs, ids in test_loader:
        imgs = imgs.to(device, non_blocking=True)
        with amp.autocast():
            c_out, _, _ = net(imgs)
        preds = torch.argmax(c_out, dim=1).cpu().numpy()
        for id_code, pred in zip(ids, preds):
            submission.append([id_code, int(pred)])

submission = np.array(submission)




## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
output_path = "submission.csv"
df.to_csv(output_path, index=False)
print(f"Submission saved to {output_path}, shape: {df.shape}")
