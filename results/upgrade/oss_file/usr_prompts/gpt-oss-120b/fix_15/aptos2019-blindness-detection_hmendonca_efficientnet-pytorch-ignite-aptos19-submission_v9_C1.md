# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.7583415713181103

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'Implemented fixes to resolve missing imports and tqdm usage errors:
- Added `import pandas as pd` to cell 1 for CSV handling.
- Switched to standard `tqdm` progress bar in cell 7 to avoid notebook‑specific import issues.

These changes enable the pipeline to load data, create the DataLoader, run inference, and generate a valid `submission.csv` without runtime errors.'
- What this solution (achieved 0.0) has done: 'I add a fixed random seed for reproducibility, replace the arg‑max with an expected‑value rounding (which often aligns better with quadratic weighted kappa), and keep the rest of the pipeline unchanged. These tweaks are minimal, preserve the original model and data flow, and are aimed at moving the QWK score from 0 toward the target 0.7583.'
- What this solution (achieved 0.0) has done: 'The script was failing because the pandas library was never imported, causing `NameError` exceptions in multiple cells. Adding `import pandas as pd` early resolves all reference errors, allowing the data loading, baseline prediction, and submission creation steps to execute correctly and produce a valid `submission.csv` file.'
- What this solution (achieved 0.0) has done: 'I add missing imports and define the checkpoint path so the script can run without NameErrors. These fixes enable the data loading, inference (or baseline) and creation of a proper `submission.csv` file, moving the pipeline from a score of 0.0 toward the target.'
- What this solution (achieved 0.0) has done: 'I fixed the typo `BICUBC` → `BICUBIC` in the image transforms, ensured the test dataset and `df_test` are created before they are used, and after training the model I set `checkpoint_loaded = True` so the inference branch runs instead of the constant‑baseline fallback. These minimal corrections allow the script to run end‑to‑end and generate a proper `submission.csv` while preserving the original model architecture and training logic.'
- What this solution (achieved 0.0) has done: 'We add a few lightweight tweaks: compute class‑frequency weights for the cross‑entropy loss, increase the brief training from 2 to 4 epochs, and average predictions from an additional vertical‑flip augmentation during inference. These changes keep the original architecture and training loop intact while modestly improving model calibration, helping the quadratic weighted kappa move toward the target score.'
- What this solution (achieved 0.0) has done: 'The changes preload all images into RAM (removing repeated disk I/O), set workers to 0 for training to avoid unnecessary process overhead, enable cuDNN benchmarking for faster GPU kernels, and configure PyTorch to use all CPU cores. These tweaks keep the model, training loop, and inference logic unchanged while dramatically cutting the runtime so the script finishes within the 600‑second limit.'
- What this solution (achieved 0.0) has done: 'I add a lightweight validation split and keep the model checkpoint that gives the best quadratic weighted kappa on the validation set. This modest change preserves the original architecture and training loop while adding a small calibration step that should move the score closer to the target without extensive rewrites.'
- What this solution (achieved 0.0) has done: 'I remove the nonexistent `MulticlassMauve` import (and the unused `MulticlassConfusionMatrix`) from cell 3 so the script runs without import errors and can produce a valid `submission.csv`. This change does not alter model logic or training, preserving the original approach while enabling the pipeline to complete and generate predictions.'

# 9. Code solution

## === cell 0
import os
import torch
import torch.nn as nn
import pandas as pd  # pandas for CSV handling

model_path = "best_model.pth"


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
            nn.Conv2d(
                inplanes, se_planes, kernel_size=1, stride=1, padding=0, bias=True
            ),
            Swish(),
            nn.Conv2d(
                se_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=True
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_se = torch.mean(x, dim=(-2, -1), keepdim=True)
        x_se = self.reduce_expand(x_se)
        return x_se * x


from torch.nn import functional as F


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
                nn.Conv2d(
                    inplanes,
                    expand_planes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
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
            nn.Conv2d(
                expand_planes, planes, kernel_size=1, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(planes, momentum=0.01, eps=1e-3),
        )

        self.with_skip = stride == 1
        self.drop_connect_rate = torch.tensor(drop_connect_rate, requires_grad=False)

    def _drop_connect(self, x):
        keep_prob = 1.0 - self.drop_connect_rate
        drop_mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) + keep_prob
        drop_mask = drop_mask.floor()
        return drop_mask * x / keep_prob

    def forward(self, x):
        z = x
        if self.expansion_conv is not None:
            x = self.expansion_conv(x)

        x = self.depthwise_conv(x)
        x = self.squeeze_excitation(x)
        x = self.project_conv(x)

        if x.shape == z.shape and self.with_skip:
            if self.training and self.drop_connect_rate is not None:
                x = self._drop_connect(x)
            x += z
        return x


from collections import OrderedDict
import math


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_out")
    elif isinstance(module, nn.Linear):
        init_range = 1.0 / math.sqrt(module.weight.shape[1])
        nn.init.uniform_(module.weight, a=-init_range, b=init_range)


class EfficientNet(nn.Module):
    def _setup_repeats(self, num_repeats):
        return int(math.ceil(self.depth_coefficient * num_repeats))

    def _setup_channels(self, num_channels):
        num_channels *= self.width_coefficient
        new_num_channels = math.floor(num_channels / self.divisor + 0.5) * self.divisor
        new_num_channels = max(self.divisor, new_num_channels)
        if new_num_channels < 0.9 * num_channels:
            new_num_channels += self.divisor
        return new_num_channels

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
            num_channels = list_channels[idx]
            next_num_channels = list_channels[idx + 1]
            num_repeats = list_num_repeats[idx]
            expand_rate = expand_rates[idx]
            kernel_size = kernel_sizes[idx]
            stride = strides[idx]
            drop_rate = drop_connect_rate * counter / num_blocks

            name = f"MBConv{expand_rate}_{counter}"
            blocks.append(
                (
                    name,
                    MBConv(
                        num_channels,
                        next_num_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        expand_rate=expand_rate,
                        se_rate=se_rate,
                        drop_connect_rate=drop_rate,
                    ),
                )
            )
            counter += 1
            for i in range(1, num_repeats):
                name = f"MBConv{expand_rate}_{counter}"
                drop_rate = drop_connect_rate * counter / num_blocks
                blocks.append(
                    (
                        name,
                        MBConv(
                            next_num_channels,
                            next_num_channels,
                            kernel_size=kernel_size,
                            stride=1,
                            expand_rate=expand_rate,
                            se_rate=se_rate,
                            drop_connect_rate=drop_rate,
                        ),
                    )
                )
                counter += 1

        self.blocks = nn.Sequential(OrderedDict(blocks))

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
        f = self.stem(x)
        f = self.blocks(f)
        y = self.head(f)
        return y




## === cell 1
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True  # faster GPU kernels
else:
    torch.set_num_threads(os.cpu_count())  # use all CPU cores

best_model = EfficientNet(num_classes=5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = best_model.to(device)

checkpoint_loaded = False

if os.path.isfile(model_path):
    try:
        state = torch.load(model_path, map_location=device)
        best_model.load_state_dict(state)
        checkpoint_loaded = True
        print("Checkpoint loaded.")
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
else:
    print("Checkpoint not found; using randomly initialized model.")




## === cell 2
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from PIL.Image import BICUBIC


class ImageDataset(torch.utils.data.Dataset):
    """
    Loads all images into RAM during initialization (optional) to avoid
    repeated disk I/O during training / inference. This change does not alter
    any model logic; it only accelerates data loading.
    """

    def __init__(
        self,
        root,
        path_list,
        targets=None,
        transform=None,
        extension=".png",
        preload=True,
    ):
        super().__init__()
        self.root = root
        self.path_list = list(path_list)
        self.extension = extension
        self.transform = transform
        self.preload = preload

        if targets is not None:
            assert len(self.path_list) == len(targets)
            self.targets = torch.LongTensor(targets)
        else:
            self.targets = None

        if self.preload:
            self.images = []
            for p in self.path_list:
                img = Image.open(os.path.join(self.root, p + self.extension)).convert(
                    "RGB"
                )
                if self.transform is not None:
                    img = self.transform(img)
                self.images.append(img)
            self.images = torch.stack(self.images)  # (N, C, H, W)

    def __getitem__(self, index):
        if self.preload:
            sample = self.images[index]
        else:
            path = self.path_list[index]
            sample = Image.open(os.path.join(self.root, path + self.extension)).convert(
                "RGB"
            )
            if self.transform is not None:
                sample = self.transform(sample)

        if self.targets is not None:
            return sample, self.targets[index]
        else:
            return sample, torch.LongTensor([])

    def __len__(self):
        return len(self.path_list)


image_size = 224

test_transform = Compose(
    [
        Resize((image_size, image_size), BICUBIC),
        ToTensor(),
        Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081]),
    ]
)

df_test = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
test_dataset = ImageDataset(
    root="../input/aptos2019-blindness-detection/test_images",
    path_list=df_test.id_code.values,
    transform=test_transform,
    preload=False,  # keep test lazy; memory not a bottleneck
)




## === cell 3
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from torchmetrics import QuadraticWeightedKappa  # needed for validation metric

batch_size = 64
num_workers = max(1, os.cpu_count() // 2)  # keep a few workers for test loader
print("num_workers:", num_workers)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)

if not checkpoint_loaded:
    tr = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
    train_transform = Compose(
        [
            Resize((image_size, image_size), BICUBIC),
            ToTensor(),
            Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081]),
        ]
    )
    train_ids, val_ids, train_labels, val_labels = train_test_split(
        tr.id_code.values,
        tr.diagnosis.values,
        test_size=0.1,
        stratify=tr.diagnosis.values,
        random_state=42,
    )
    train_dataset = ImageDataset(
        root="../input/aptos2019-blindness-detection/train_images",
        path_list=train_ids,
        targets=train_labels,
        transform=train_transform,
        preload=True,
    )
    val_dataset = ImageDataset(
        root="../input/aptos2019-blindness-detection/train_images",
        path_list=val_ids,
        targets=val_labels,
        transform=train_transform,
        preload=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 workers since data already in memory
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    class_counts = tr["diagnosis"].value_counts().sort_index()
    class_weights = 1.0 / class_counts.values.astype(float)
    class_weights = class_weights / class_weights.sum() * len(class_counts)  # normalize
    class_weights_tensor = torch.tensor(
        class_weights, dtype=torch.float32, device=device
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(best_model.parameters(), lr=1e-3)

    best_model.train()
    epochs = 12  # a modest increase for better learning
    best_kappa = -1.0
    best_state = None
    class_indices = torch.arange(5, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = best_model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {epoch_loss/len(train_loader):.4f}")

        best_model.eval()
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                logits = best_model(x_val)
                probs = F.softmax(logits, dim=-1)
                expected = torch.sum(probs * class_indices, dim=1)
                pred = torch.clamp(torch.round(expected), 0, 4).long()
                val_preds.append(pred.cpu())
                val_targets.append(y_val.cpu())
        val_preds = torch.cat(val_preds)
        val_targets = torch.cat(val_targets)

        kappa = (
            QuadraticWeightedKappa(num_classes=5)
            .to(device)(val_preds, val_targets)
            .item()
        )
        print(f"Validation QWK: {kappa:.4f}")

        if kappa > best_kappa:
            best_kappa = kappa
            best_state = best_model.state_dict()
        best_model.train()

    if best_state is not None:
        best_model.load_state_dict(best_state)
        print(f"Best validation QWK {best_kappa:.4f} restored.")
    best_model.eval()
    checkpoint_loaded = True
    torch.save(best_model.state_dict(), model_path)
else:
    best_model.eval()

all_pred = []




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/2596976834.py in <cell line: 0>()
      4 import numpy as np
      5 from sklearn.model_selection import train_test_split
----> 6 from torchmetrics import QuadraticWeightedKappa  # needed for validation metric
      7 
      8 batch_size = 64

ImportError: cannot import name 'QuadraticWeightedKappa' from 'torchmetrics' (/usr/local/lib/python3.11/dist-packages/torchmetrics/__init__.py)

## === cell 4
class_indices = torch.arange(5, dtype=torch.float32, device=device)  # classes 0‑4

if not checkpoint_loaded:
    tr = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
    class_counts = tr["diagnosis"].value_counts().sort_index()
    probs = class_counts / len(tr)
    probs = torch.tensor(probs.values, dtype=torch.float32, device=device)
    expected = torch.sum(probs * class_indices).item()
    baseline_label = int(round(expected))
    baseline_label = max(0, min(4, baseline_label))
    all_pred = [baseline_label] * len(df_test)
else:
    with torch.no_grad():
        for x, _ in tqdm(test_loader, total=int(len(test_loader))):
            x = x.to(device)

            y_pred1 = best_model(x)
            y_pred_h = best_model(torch.flip(x, dims=(-1,)))  # horizontal flip
            y_pred_v = best_model(torch.flip(x, dims=(-2,)))  # vertical flip

            probs = 0.5 * (F.softmax(y_pred1, dim=-1) + F.softmax(y_pred_h, dim=-1))
            probs = 0.5 * (probs + F.softmax(y_pred_v, dim=-1))

            expected = torch.sum(probs * class_indices, dim=1)
            pred_labels = torch.clamp(torch.round(expected), 0, 4).long()
            all_pred.extend(pred_labels.cpu().numpy().tolist())




## === cell 5
import numpy as np

print("Number of predictions:", len(all_pred))
assert len(all_pred) == len(df_test), "Prediction count does not match test set size."

sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
sub["diagnosis"] = np.array(all_pred, dtype=int)
sub.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
sub.head()




## === cell 6
_ = sub.hist()




## === cell 7
tr = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
_ = tr.hist()
