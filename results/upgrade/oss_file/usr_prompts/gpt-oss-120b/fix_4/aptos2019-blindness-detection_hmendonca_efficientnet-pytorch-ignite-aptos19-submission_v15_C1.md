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

# 5. Code solution

## === cell 0
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (
    Compose,
    Resize,
    ToTensor,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomResizedCrop,
    RandomRotation,
    InterpolationMode,
)




## === cell 1
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)


class SqueezeExcitation(nn.Module):
    def __init__(self, inplanes, se_planes):
        super(SqueezeExcitation, self).__init__()
        self.reduce_expand = nn.Sequential(
            nn.Conv2d(inplanes, se_planes, kernel_size=1, bias=True),
            Swish(),
            nn.Conv2d(se_planes, inplanes, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_se = torch.mean(x, dim=(-2, -1), keepdim=True)
        x_se = self.reduce_expand(x_se)
        return x_se * x


class MBConv(nn.Module):
    def __init__(
        self,
        inplanes,
        planes,
        kernel_size,
        stride,
        expand_rate=1.0,
        se_rate=0.25,
        drop_connect_rate=0.2,
    ):
        super(MBConv, self).__init__()
        expand_planes = int(inplanes * expand_rate)
        se_planes = max(1, int(inplanes * se_rate))

        self.expansion_conv = None
        if expand_rate > 1.0:
            self.expansion_conv = nn.Sequential(
                nn.Conv2d(inplanes, expand_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(expand_planes, momentum=0.01, eps=1e-3),
                Swish(),
            )
            inplanes = expand_planes

        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(
                inplanes,
                expand_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=expand_planes,
                bias=False,
            ),
            nn.BatchNorm2d(expand_planes, momentum=0.01, eps=1e-3),
            Swish(),
        )
        self.squeeze_excitation = SqueezeExcitation(expand_planes, se_planes)
        self.project_conv = nn.Sequential(
            nn.Conv2d(expand_planes, planes, kernel_size=1, bias=False),
            nn.BatchNorm2d(planes, momentum=0.01, eps=1e-3),
        )
        self.with_skip = stride == 1
        self.drop_connect_rate = torch.tensor(drop_connect_rate, requires_grad=False)

    def _drop_connect(self, x):
        keep_prob = 1.0 - self.drop_connect_rate
        mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) + keep_prob
        mask.floor_()
        return x * mask / keep_prob

    def forward(self, x):
        identity = x
        if self.expansion_conv is not None:
            x = self.expansion_conv(x)
        x = self.depthwise_conv(x)
        x = self.squeeze_excitation(x)
        x = self.project_conv(x)
        if self.with_skip and x.shape == identity.shape:
            if self.training and self.drop_connect_rate.item() > 0:
                x = self._drop_connect(x)
            x = x + identity
        return x


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_out")
    elif isinstance(module, nn.Linear):
        bound = 1.0 / math.sqrt(module.weight.shape[1])
        nn.init.uniform_(module.weight, -bound, bound)


class EfficientNet(nn.Module):
    def _setup_repeats(self, num_repeats):
        return int(math.ceil(self.depth_coefficient * num_repeats))

    def _setup_channels(self, num_channels):
        num_channels *= self.width_coefficient
        new_num = math.floor(num_channels / self.divisor + 0.5) * self.divisor
        new_num = max(self.divisor, new_num)
        if new_num < 0.9 * num_channels:
            new_num += self.divisor
        return int(new_num)

    def __init__(
        self,
        num_classes,
        width_coefficient=1.0,
        depth_coefficient=1.0,
        se_rate=0.25,
        dropout_rate=0.2,
        drop_connect_rate=0.2,
    ):
        super(EfficientNet, self).__init__()
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient
        self.divisor = 8

        list_channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        list_channels = [self._setup_channels(c) for c in list_channels]
        list_num_repeats = [1, 2, 2, 3, 3, 4, 1]
        list_num_repeats = [self._setup_repeats(r) for r in list_num_repeats]

        expand_rates = [1, 6, 6, 6, 6, 6, 6]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernel_sizes = [3, 3, 5, 3, 5, 5, 3]

        self.stem = nn.Sequential(
            nn.Conv2d(
                3, list_channels[0], kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(list_channels[0], momentum=0.01, eps=1e-3),
            Swish(),
        )

        blocks = []
        counter = 0
        num_blocks = sum(list_num_repeats)
        for idx in range(7):
            in_c = list_channels[idx]
            out_c = list_channels[idx + 1]
            repeats = list_num_repeats[idx]
            exp = expand_rates[idx]
            k = kernel_sizes[idx]
            s = strides[idx]
            drop_rate = drop_connect_rate * counter / num_blocks
            blocks.append(
                (
                    f"MBConv{exp}_{counter}",
                    MBConv(
                        in_c,
                        out_c,
                        kernel_size=k,
                        stride=s,
                        expand_rate=exp,
                        se_rate=se_rate,
                        drop_connect_rate=drop_rate,
                    ),
                )
            )
            counter += 1
            for _ in range(1, repeats):
                drop_rate = drop_connect_rate * counter / num_blocks
                blocks.append(
                    (
                        f"MBConv{exp}_{counter}",
                        MBConv(
                            out_c,
                            out_c,
                            kernel_size=k,
                            stride=1,
                            expand_rate=exp,
                            se_rate=se_rate,
                            drop_connect_rate=drop_rate,
                        ),
                    )
                )
                counter += 1

        self.blocks = nn.Sequential(dict(blocks))
        self.head = nn.Sequential(
            nn.Conv2d(list_channels[-2], list_channels[-1], kernel_size=1, bias=False),
            nn.BatchNorm2d(list_channels[-1], momentum=0.01, eps=1e-3),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(list_channels[-1], num_classes),
        )
        self.apply(init_weights)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x




## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 380

pretrained = models.efficientnet_b4(
    weights=models.EfficientNet_B4_Weights.IMAGENET1K_V1
)
num_ftrs = pretrained.classifier[1].in_features
pretrained.classifier[1] = nn.Linear(num_ftrs, 5)
best_model = pretrained.to(device)
best_model.eval()




## === cell 3
class ImageDataset(Dataset):
    def __init__(self, root, path_list, targets=None, transform=None, extension=".png"):
        self.root = root
        self.path_list = path_list
        self.targets = torch.LongTensor(targets) if targets is not None else None
        self.transform = transform
        self.extension = extension

    def __len__(self):
        return len(self.path_list)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.path_list[idx] + self.extension)
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.targets is not None:
            return img, self.targets[idx]
        else:
            return img, torch.tensor([], dtype=torch.long)




## === cell 4
train_transform = Compose(
    [
        RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(15, interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081]),
    ]
)

test_transform = Compose(
    [
        Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081]),
    ]
)

train_csv = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
train_paths = train_csv["id_code"].values
train_labels = train_csv["diagnosis"].values

train_idx, val_idx = train_test_split(
    np.arange(len(train_paths)), test_size=0.1, stratify=train_labels, random_state=42
)

train_dataset = ImageDataset(
    root="../input/aptos2019-blindness-detection/train_images",
    path_list=train_paths[train_idx],
    targets=train_labels[train_idx],
    transform=train_transform,
)

val_dataset = ImageDataset(
    root="../input/aptos2019-blindness-detection/train_images",
    path_list=train_paths[val_idx],
    targets=train_labels[val_idx],
    transform=test_transform,
)

train_loader = DataLoader(
    train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=False
)

val_loader = DataLoader(
    val_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False
)




## === cell 5
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(best_model.parameters(), lr=1e-4)
num_epochs = 10  # increased from 5 to give a slightly better model

best_kappa = -1.0
for epoch in range(1, num_epochs + 1):
    best_model.train()
    epoch_loss = 0.0
    for imgs, targets in train_loader:
        imgs, targets = imgs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = best_model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)
    epoch_loss /= len(train_loader.dataset)

    best_model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            logits = best_model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_true.extend(targets.numpy())
    kappa = cohen_kappa_score(all_true, all_preds, weights="quadratic")
    if kappa > best_kappa:
        best_kappa = kappa
        torch.save(best_model.state_dict(), "best_model.pth")
    print(
        f"Epoch {epoch}/{num_epochs} - Loss: {epoch_loss:.4f} - Val Kappa: {kappa:.4f}"
    )

print(f"Best validation Kappa: {best_kappa:.4f}")




## === cell 6
best_model.load_state_dict(torch.load("best_model.pth", map_location=device))
best_model.eval()


def ttta(x):
    """8‑fold test‑time augmentation without mutating the input."""
    aug_preds = []
    for flip_h in [False, True]:
        for flip_v in [False, True]:
            for trans in [False, True]:
                aug = x
                if flip_h:
                    aug = torch.flip(aug, dims=[2])
                if flip_v:
                    aug = torch.flip(aug, dims=[3])
                if trans:
                    aug = aug.transpose(2, 3)
                aug_preds.append(best_model(aug).unsqueeze(0))
    probs = F.softmax(torch.cat(aug_preds), dim=-1)
    return probs.mean(dim=0)




## === cell 7
test_csv = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_dataset = ImageDataset(
    root="../input/aptos2019-blindness-detection/test_images",
    path_list=test_csv["id_code"].values,
    targets=None,
    transform=test_transform,
)

test_loader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, num_workers=0, pin_memory=False
)




## === cell 8
preds = []
with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.to(device)
        probs = ttta(imgs)
        preds.extend(torch.argmax(probs, dim=1).cpu().tolist())

print("Number of predictions:", len(preds))




## === cell 9
submission = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
submission["diagnosis"] = np.array(preds, dtype=int)
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
submission.head()
