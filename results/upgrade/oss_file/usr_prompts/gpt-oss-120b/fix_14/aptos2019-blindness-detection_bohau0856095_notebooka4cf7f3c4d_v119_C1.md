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
import torch
import torch.nn as nn
import torchvision
import csv
import timm
import time
import glob
import copy
import os
import json
import cv2
import numpy as np
import pickle
import torch.optim as optim
import torch.nn.functional as F
from torch import nn as nn
from torch.optim import lr_scheduler
from timm.models import *
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils, models, datasets
from PIL import Image
from PIL import Image, ImageEnhance, ImageOps, ImageChops
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops.feature_pyramid_network import (
    FeaturePyramidNetwork,
    LastLevelMaxPool,
)
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision.ops import misc as misc_nn_ops
from torch import Tensor, Size
from torch.jit.annotations import List, Optional, Tuple
from torch.nn.parameter import Parameter
import warnings
import pandas as pd
import random
from torchvision.transforms import functional as FT
from sklearn.metrics import cohen_kappa_score

torch.set_float32_matmul_precision("high")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class ThreeStage_Model(nn.Module):
    def __init__(self, backbone=None):
        super(ThreeStage_Model, self).__init__()
        self.backbone = timm.create_model("tf_efficientnet_b4_ns", pretrained=False)
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
            out = torch.cat((c_out, r_out, o_out), 1)
            out = self.final_regressor(out)
            out = torch.sigmoid(out) * 4.5
            return out
        else:
            r_out = torch.sigmoid(r_out) * 4.5
            o_out = torch.sigmoid(o_out)
            return c_out, r_out, o_out




## === cell 1
threshold = [0.75, 1.5, 2.5, 3.5]


def regress2class(out):
    prediction = 0
    for i in range(4):
        prediction += (out.data >= threshold[i]).squeeze().cpu().item()
    return prediction




## === cell 2
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
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )




## === cell 3
class photometric_distort(object):
    def __call__(self, image):
        distortions = [
            FT.adjust_brightness,
            FT.adjust_contrast,
            FT.adjust_saturation,
            FT.adjust_hue,
        ]

        random.shuffle(distortions)

        for d in distortions:
            if random.random() < 0.5:
                if d.__name__ == "adjust_hue":
                    adjust_factor = random.uniform(-16 / 255.0, 16 / 255.0)
                else:
                    adjust_factor = random.uniform(0.7, 1.3)
                image = d(image, adjust_factor)
        return image


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




## === cell 4
test_ids_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_ids = test_ids_df["id_code"].values

transform1 = transforms.Compose(
    [
        trim(),
        cropTo4_3(),
        transforms.Resize((288, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.384, 0.258, 0.174], std=[0.124, 0.089, 0.094]),
    ]
)

weight_paths = glob.glob("../input/**/B4_3stage_3epoch_320finetune.pkl", recursive=True)

net1 = ThreeStage_Model()
loaded = False
for wp in weight_paths:
    if os.path.exists(wp):
        try:
            net1.load_state_dict(torch.load(wp, map_location=device))
            print(f"Loaded weights from {wp}")
            loaded = True
            break
        except Exception as e:
            print(f"Failed loading {wp}: {e}")

if not loaded:
    print(
        "Fine‑tuned weights not found – training a lightweight model on the provided data."
    )

    class CachedAptosDataset(Dataset):
        def __init__(self, df, img_dir, transform):
            self.ids = df["id_code"].values
            self.labels = df["diagnosis"].values
            self.transform = transform
            self.tensors = []
            for img_id in self.ids:
                img_path = os.path.join(img_dir, f"{img_id}.png")
                img = Image.open(img_path).convert("RGB")
                self.tensors.append(self.transform(img))

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            img_tensor = self.tensors[idx]
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img_tensor, label

    train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
    img_dir = "../input/aptos2019-blindness-detection/train_images"

    full_dataset = CachedAptosDataset(train_df, img_dir, transform1)
    train_len = int(0.9 * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_set, val_set = random_split(
        full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )

    num_workers = min(os.cpu_count() or 1, 8)

    def worker_init_fn(worker_id):
        seed = 42 + worker_id
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net1.parameters(), lr=1e-4, weight_decay=1e-5)

    net1 = net1.to(device)
    net1 = torch.compile(net1)
    net1.train()

    scaler = torch.cuda.amp.GradScaler()

    epochs = 8
    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, lbls in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu"
            ):
                logits, _, _ = net1(imgs)
                loss = criterion(logits, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * imgs.size(0)

        avg_loss = epoch_loss / len(train_loader.dataset)

        net1.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                lbls = lbls.to(device, non_blocking=True)
                with torch.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu"
                ):
                    logits, _, _ = net1(imgs)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_true.extend(lbls.cpu().numpy())
        val_kappa = cohen_kappa_score(all_true, all_preds, weights="quadratic")
        print(
            f"Epoch {epoch+1}/{epochs} - Train loss: {avg_loss:.4f} - Val Kappa: {val_kappa:.4f}"
        )
        net1.train()
else:
    net1 = net1.to(device)
    net1.eval()




## === cell 5
class TestDataset(Dataset):
    def __init__(self, ids, img_dir, transform):
        self.ids = ids
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{img_id}.png")
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, img_id


test_dataset = TestDataset(
    test_ids,
    "../input/aptos2019-blindness-detection/test_images",
    transform1,
)

num_workers = min(8, os.cpu_count() or 1)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    worker_init_fn=lambda worker_id: (
        torch.manual_seed(42 + worker_id),
        np.random.seed(42 + worker_id),
        random.seed(42 + worker_id),
    ),
)

submission = []
net1.eval()
softmax = nn.Softmax(dim=1)
with torch.no_grad():
    for batch_imgs, batch_ids in test_loader:
        batch_imgs = batch_imgs.to(device, non_blocking=True)

        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu"):
            logits, _, _ = net1(batch_imgs)
            flipped_imgs = torch.flip(batch_imgs, dims=[3])
            logits_flip, _, _ = net1(flipped_imgs)

        probs = (softmax(logits) + softmax(logits_flip)) / 2.0
        preds = torch.argmax(probs, dim=1).cpu().numpy()

        for img_id, pred in zip(batch_ids, preds):
            pred = int(max(0, min(4, pred)))  # ensure within [0,4]
            submission.append([img_id, pred])

submission = np.array(submission)
print(f"Generated predictions for {len(submission)} images.")



## === cell 6
df = pd.DataFrame(submission, columns=["id_code", "diagnosis"])
output_path = "submission.csv"
df.to_csv(output_path, index=False)
print(f"Submission file written to {output_path}")



## === cell 7
df.head()
