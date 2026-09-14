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

0.8948946735643287

# 6. Current score

-0.01596

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.03011) has done: 'I fixed the script by (1) handling missing weight files gracefully and falling back to pretrained EfficientNet weights, (2) making the device selection robust, (3) correcting the CLAHE processing of OpenCV channel tuples, (4) wrapping inference in `torch.no_grad()` for safety, and (5) renumbering cells to start at 1 while keeping the original logic unchanged.'
- What this solution (achieved 0.2673) has done: 'The update adds a lightweight fine‑tuning step that quickly trains the classifier head on the provided training images (just a few epochs, freezing the backbone). This gives the model learned class probabilities instead of random ones, which moves the quadratic weighted kappa much closer to the target while keeping the original architecture and inference pipeline unchanged.'
- What this solution (achieved 0.20733) has done: 'The changes keep the exact model architecture and training procedure but avoid unnecessary gradient work and extra branch computation. By freezing the backbone, setting it to eval mode, and manually running only the classifier part during training and inference, we cut the forward‑pass cost dramatically. Adding `persistent_workers=True` removes the repeated DataLoader worker startup overhead, and the inference loop now also uses only the classifier output. These tweaks preserve all predictions while keeping the total run time well under the 600‑second limit.'
- What this solution (achieved 0.27279) has done: 'The changes add simple in‑memory caching to the training and test datasets so each image is read and transformed only once, eliminating repeated I/O and CPU‑heavy Pillow/OpenCV work across epochs. The rest of the model, training loop, and inference logic stay exactly the same, preserving accuracy while dramatically reducing total runtime.'
- What this solution (achieved -0.01596) has done: 'The update pre‑computes backbone features once for the whole training set, eliminating the costly forward pass through the frozen EfficientNet on every epoch. Training now operates only on these cached feature vectors, dramatically reducing I/O and computation while keeping the exact same optimizer, loss, and number of epochs. Minor adjustments (adding `TensorDataset`, redefining the loader, and using the cached features in the training loop) preserve all original semantics and model behavior.'

# 9. Code solution

## === cell 0
try:
    import subprocess, sys, pathlib

    wheel_path = pathlib.Path("../input/weights/timm-0.3.1-py3-none-any.whl")
    if wheel_path.is_file():
        subprocess.check_call([sys.executable, "-m", "pip", "install", str(wheel_path)])
except Exception:
    pass


## === cell 1
import os
import random
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision.transforms as transforms
from PIL import Image, ImageChops
import cv2
from sklearn.metrics import cohen_kappa_score
import timm

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


## === cell 2
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = torch.zeros(out.size(0), device=out.device)
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze()
    return prediction


def ordinal2class_prob(out):
    pred_prob = torch.zeros(out.size(0), 5, device=out.device)
    pred_prob[:, 0] = 1 - out[:, 0]
    pred_prob[:, 1] = out[:, 0] * (1 - out[:, 1])
    pred_prob[:, 2] = out[:, 1] * (1 - out[:, 2])
    pred_prob[:, 3] = out[:, 2] * (1 - out[:, 3])
    pred_prob[:, 4] = out[:, 3]
    return F.softmax(pred_prob, dim=1)


def regress2class_prob(out):
    pred_prob = torch.zeros((out.size(0), 5), device=out.device)
    for i in range(out.size(0)):
        val = out[i].item()
        if val < 4.0:
            l1 = int(math.floor(val))
            l2 = int(math.ceil(val))
            pred_prob[i][l1] = 1 - (val - l1)
            pred_prob[i][l2] = 1 - (l2 - val)
        else:
            pred_prob[i][4] = 1.0
    return pred_prob


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
        x = self.backbone(x)
        c_out = self.classifier(x)
        r_out = self.regressor(x)
        o_out = self.ordinal(x)
        if final:
            out = torch.cat((c_out, r_out, o_out), dim=1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 4.5
            return out
        else:
            r_out = torch.sigmoid(r_out) * 4.5
            o_out = torch.sigmoid(o_out)
            return c_out, r_out, o_out




## === cell 3
class cropTo4_3(object):
    def __call__(self, image):
        w, h = image.size
        if (w / h) >= (4 / 3):
            new_h = h
            new_w = int(h * 4 / 3)
        else:
            new_h = int(w * 3 / 4)
            new_w = w
        left = (w - new_w) / 2
        top = (h - new_h) / 2
        right = left + new_w
        bottom = top + new_h
        return image.crop((left, top, right, bottom))


class trim(object):
    def __call__(self, image):
        bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
        diff = ImageChops.difference(image, bg)
        diff = ImageChops.add(diff, diff, 2.0, -10)
        bbox = diff.getbbox()
        if bbox:
            return image.crop(bbox)
        return image


input_size = 512
transform = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.Resize((input_size * 3 // 4, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

net = ThreeStage_Model()
weight_path = "../input/weights/B4_3stage_60epoch_AdamW_CLAHE.pkl"
if os.path.isfile(weight_path):
    try:
        net.load_state_dict(torch.load(weight_path, map_location=device))
        print("Custom weights loaded.")
    except Exception as e:
        print(f"Failed to load custom weights: {e}")
else:
    print("Custom weight file not found – using default pretrained backbone.")

net.to(device)

net = torch.compile(net)

for param in net.backbone.parameters():
    param.requires_grad = False
net.backbone.eval()

train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
train_img_dir = "../input/aptos2019-blindness-detection/train_images"


class DRDataset(Dataset):
    """
    Dataset with an in‑memory cache to avoid re‑reading and re‑transforming the same image
    across epochs. This preserves the exact same transform pipeline while dramatically
    reducing I/O and CPU overhead.
    """

    def __init__(self, df, img_dir, transform):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform
        self.cache = [None] * len(df)  # placeholder for cached tensors

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.cache[idx] is not None:
            img_tensor, label = self.cache[idx]
            return img_tensor, label

        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        label = int(row["diagnosis"])
        self.cache[idx] = (img, label)
        return img, label


train_dataset = DRDataset(train_df, train_img_dir, transform)

net.eval()
with torch.no_grad():
    feature_list = []
    label_list = []
    pre_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    for imgs, labels in pre_loader:
        imgs = imgs.to(device, non_blocking=True)
        feats = net.backbone(imgs)  # shape [B, 1000]
        feature_list.append(feats.cpu())
        label_list.append(labels)

    train_features = torch.cat(feature_list)  # [N, 1000]
    train_labels = torch.cat(label_list)  # [N]

train_feat_dataset = TensorDataset(train_features, train_labels)
train_loader = DataLoader(
    train_feat_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=0,  # no extra workers needed for already‑in‑memory tensors
    pin_memory=True,
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(list(net.classifier.parameters()), lr=5e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

net.train()
epochs = 20
for epoch in range(epochs):
    epoch_loss = 0.0
    for feats, labels in train_loader:
        feats = feats.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                c_out = net.classifier(feats)
                loss = criterion(c_out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.cuda.amp.autocast():
                c_out = net.classifier(feats)
                loss = criterion(c_out, labels)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * feats.size(0)

    scheduler.step()
    avg_loss = epoch_loss / len(train_feat_dataset)
    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

net.eval()
torch.cuda.empty_cache()


## === cell 4
test_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = test_df["id_code"].values.squeeze()
test_img_dir = "../input/aptos2019-blindness-detection/test_images"

clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(4, 4))


class TestDataset(Dataset):
    """
    Test dataset with caching similar to the training set. The CLAHE processing is
    performed once per image and the resulting tensor is stored for fast reuse.
    """

    def __init__(self, ids, img_dir, transform, clahe):
        self.ids = ids
        self.img_dir = img_dir
        self.transform = transform
        self.clahe = clahe
        self.cache = [None] * len(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        if self.cache[idx] is not None:
            img_id, img_tensor = self.cache[idx]
            return img_id, img_tensor

        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        bgr = cv2.imread(img_path)
        if bgr is None:
            bgr = np.zeros((512, 512, 3), dtype=np.uint8)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        lab_planes = list(lab_planes)
        lab_planes[0] = self.clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = self.transform(img)
        self.cache[idx] = (img_id, img)
        return img_id, img


test_dataset = TestDataset(test_ids, test_img_dir, transform, clahe)
test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

submission = []
net.eval()
with torch.no_grad():
    for batch_ids, batch_imgs in test_loader:
        batch_imgs = batch_imgs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            feats = net.backbone(batch_imgs)
            c_out = net.classifier(feats)
            c_prob = F.softmax(c_out, dim=1)
        _, preds = torch.max(c_prob, dim=1)
        for img_id, pred in zip(batch_ids, preds.cpu().numpy()):
            submission.append([img_id, int(pred)])


## === cell 5
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
