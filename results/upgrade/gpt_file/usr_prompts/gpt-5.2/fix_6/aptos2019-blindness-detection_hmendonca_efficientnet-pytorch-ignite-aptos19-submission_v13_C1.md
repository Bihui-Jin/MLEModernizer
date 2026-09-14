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

0.66019225921555

# 6. Current score

0.0

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.0) has done: 'I remove the hard dependency on an external `.pth` checkpoint (which isn’t present in your `/kaggle/input`), and instead train the same EfficientNet-B4 architecture on `train.csv` so the pipeline can run end-to-end and generate predictions. I also fix the CUDA/CPU dtype/device mismatch by ensuring the model and inputs are always on the same device, and I keep the existing TTA + softmax + argmax inference semantics. Finally, I make sure the submission is written as `submission.csv` with the required `id_code,diagnosis` columns and correct row alignment to `sample_submission.csv`.'
- What this solution (achieved 0.28868) has done: 'Your current 0.0 score is consistent with an “all-zeros” style failure mode caused by training for only 1 epoch on a heavily imbalanced 5-class problem and then using plain argmax; this typically collapses to predicting class 0 for most images and yields near-zero QWK. To move toward the 0.66 target with minimal semantic change, I keep the exact EfficientNet-B4 architecture, loss, and TTA+softmax+argmax inference, but (1) add a simple stratified train/valid split and train for a few epochs (no early stopping) so the model learns non-trivial class boundaries, and (2) use the best validation checkpoint by QWK during training. These are the smallest changes that directly improve QWK without changing the core approach, and they remain within Kaggle constraints and runtime by keeping epochs modest.'
- What this solution (achieved 0.0) has done: 'You’re currently not yielding a score because Kaggle never receives a submission; the notebook writes `submission.csv` to the local runtime, but you need to ensure the file is always created (even if training is skipped/failed) and that predictions align exactly to `test.csv`/`sample_submission.csv` ordering. To move the QWK upward toward the 0.660 target without changing your core model/training/inference semantics, I keep EfficientNet-B4 + CrossEntropy + your existing 8-way TTA+softmax+argmax, but I (1) guarantee deterministic execution (same seeds + deterministic cuDNN) to avoid unstable “bad runs”, and (2) fix the submission alignment to `test.csv` explicitly (instead of trusting `sample_submission.csv` order), which can otherwise silently hurt score. These changes are minimal, preserve your approach, and should produce a valid `submission.csv` every run.'

# 9. Code solution

## === cell 0
import os, glob, sys, shutil, subprocess, textwrap, pathlib, random, math

subprocess.run(["bash", "-lc", "ls ../input/*"], check=False)

target_patterns = [
    "../input/**/efficientNet_*.pth",
    "../input/**/efficientnet_*.pth",
    "../input/**/*efficientnet*.pth",
    "../input/**/*EfficientNet*.pth",
    "../input/**/*.pth",
]
model_path = "efficientNet_best.pth"

candidates = []
for pat in target_patterns:
    candidates.extend(glob.glob(pat, recursive=True))

preferred = [
    p for p in candidates if os.path.basename(p).lower().startswith("efficientnet")
]
preferred = preferred if preferred else candidates
preferred = sorted(set(preferred))

if preferred:
    src_path = preferred[0]
    print("Found checkpoint:", src_path)
    subprocess.run(["bash", "-lc", f"md5sum '{src_path}' || true"], check=False)
    shutil.copyfile(src_path, model_path)
    subprocess.run(["bash", "-lc", f"md5sum '{model_path}'"], check=False)
else:
    print(
        "No .pth checkpoint found under ../input. Will train a model in-notebook and save to",
        model_path,
    )



## === cell 1
import torch
import torch.nn as nn


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
        self.drop_connect_rate = float(drop_connect_rate)

    def _drop_connect(self, x):
        keep_prob = 1.0 - self.drop_connect_rate
        drop_mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) + keep_prob
        drop_mask = drop_mask.type_as(x)
        drop_mask.floor_()
        return drop_mask * x / keep_prob

    def forward(self, x):
        z = x
        if self.expansion_conv is not None:
            x = self.expansion_conv(x)

        x = self.depthwise_conv(x)
        x = self.squeeze_excitation(x)
        x = self.project_conv(x)

        if x.shape == z.shape and self.with_skip:
            if (
                self.training
                and self.drop_connect_rate is not None
                and self.drop_connect_rate > 0
            ):
                x = self._drop_connect(x)
            x += z
        return x


from collections import OrderedDict


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

            name = "MBConv{}_{}".format(expand_rate, counter)
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
                name = "MBConv{}_{}".format(expand_rate, counter)
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




## === cell 2
import numpy as np
import pandas as pd
from PIL import Image
from PIL.Image import BICUBIC
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torch.utils.data import DataLoader

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import cohen_kappa_score

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
try:
    torch.use_deterministic_algorithms(True)
except Exception as e:
    print("Warning: could not enable full deterministic algorithms:", repr(e))

image_size = 380
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

best_model = EfficientNet(
    num_classes=5, width_coefficient=1.4, depth_coefficient=1.8
)  # B4

if os.path.exists(model_path):
    state = torch.load(model_path, map_location="cpu")
    best_model.load_state_dict(state)
else:
    train_csv = "../input/aptos2019-blindness-detection/train.csv"
    train_img_root = "../input/aptos2019-blindness-detection/train_images"
    df_train = pd.read_csv(train_csv)

    class ImageDataset(torch.utils.data.Dataset):
        def __init__(
            self, root, path_list, targets=None, transform=None, extension=".png"
        ):
            super().__init__()
            self.root = root
            self.path_list = list(path_list)
            self.targets = targets
            self.transform = transform
            self.extension = extension
            if targets is not None:
                assert len(self.path_list) == len(self.targets)
                self.targets = torch.LongTensor(list(targets))

        def __getitem__(self, index):
            path = self.path_list[index]
            img_path = os.path.join(self.root, path + self.extension)
            sample = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                sample = self.transform(sample)
            if self.targets is not None:
                return sample, self.targets[index]
            else:
                return sample, torch.LongTensor([])

        def __len__(self):
            return len(self.path_list)

    train_transform = Compose(
        [
            Resize((image_size, image_size), BICUBIC),
            ToTensor(),
            Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081]),
        ]
    )

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    idx_train, idx_valid = next(
        splitter.split(df_train["id_code"].values, df_train["diagnosis"].values)
    )
    df_tr = df_train.iloc[idx_train].reset_index(drop=True)
    df_va = df_train.iloc[idx_valid].reset_index(drop=True)

    train_dataset = ImageDataset(
        root=train_img_root,
        path_list=df_tr.id_code.values,
        targets=df_tr.diagnosis.values,
        transform=train_transform,
    )
    valid_dataset = ImageDataset(
        root=train_img_root,
        path_list=df_va.id_code.values,
        targets=df_va.diagnosis.values,
        transform=train_transform,
    )

    batch_size = 8 if device.type == "cuda" else 4
    num_workers = min(4, os.cpu_count() or 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    best_model = best_model.to(device)
    optimizer = torch.optim.Adam(best_model.parameters(), lr=1e-4)

    y_tr = df_tr["diagnosis"].values.astype(int)
    counts = np.bincount(y_tr, minlength=5).astype(np.float32)
    inv = 1.0 / np.maximum(counts, 1.0)
    class_w = inv / inv.mean()  # normalize for stable loss scale
    class_w_t = torch.tensor(class_w, dtype=torch.float32, device=device)
    print("train class counts:", counts.tolist())
    print("class weights:", class_w.tolist())
    criterion = nn.CrossEntropyLoss(weight=class_w_t)

    epochs = 8
    print(f"Training for {epochs} epoch(s) to create missing checkpoint...")

    from tqdm import tqdm

    best_qwk = -1.0
    best_state = None

    def predict_tta_argmax(model, x):
        pred = []
        for flip1 in range(2):  # flip 1st dim
            for flip2 in range(2):  # flip 2nd dim
                for trans in range(2):  # transpose
                    xx = x
                    if flip1:
                        xx = xx.flip(2)  # height
                    if flip2:
                        xx = xx.flip(3)  # width
                    if trans:
                        xx = xx.transpose(-1, -2)
                    pred.append(model(xx).unsqueeze(0))
        pred = F.softmax(torch.cat(pred, dim=0), dim=-1).mean(dim=0)
        return torch.argmax(pred, dim=-1)

    for epoch in range(epochs):
        best_model.train()
        running = 0.0
        seen = 0
        pbar = tqdm(
            train_loader, desc=f"epoch {epoch+1}/{epochs}", total=len(train_loader)
        )
        for xb, yb in pbar:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = best_model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            running += loss.item() * bs
            seen += bs
            pbar.set_postfix(loss=running / max(1, seen))

        best_model.eval()
        va_preds = []
        va_true = []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device, non_blocking=True)
                pred = predict_tta_argmax(best_model, xb).detach().cpu().numpy()
                va_preds.append(pred)
                va_true.append(yb.detach().cpu().numpy())
        va_preds = np.concatenate(va_preds)
        va_true = np.concatenate(va_true)
        qwk = cohen_kappa_score(va_true, va_preds, weights="quadratic")
        print(f"valid QWK: {qwk:.5f}")

        if qwk > best_qwk:
            best_qwk = qwk
            best_state = {
                k: v.detach().cpu().clone() for k, v in best_model.state_dict().items()
            }

    if best_state is None:
        best_state = {
            k: v.detach().cpu().clone() for k, v in best_model.state_dict().items()
        }

    torch.save(best_state, model_path)
    print("Saved best-by-valid-QWK checkpoint to", model_path, "best_qwk=", best_qwk)

    best_model.load_state_dict(torch.load(model_path, map_location="cpu"))

best_model = best_model.to(device)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2378395366.py in <cell line: 0>()
    165             logits = best_model(xb)
    166             loss = criterion(logits, yb)
--> 167             loss.backward()
    168             optimizer.step()
    169 

/usr/local/lib/python3.11/dist-packages/torch/_tensor.py in backward(self, gradient, retain_graph, create_graph, inputs)
    624                 inputs=inputs,
    625             )
--> 626         torch.autograd.backward(
    627             self, gradient, retain_graph, create_graph, inputs=inputs
    628         )

/usr/local/lib/python3.11/dist-packages/torch/autograd/__init__.py in backward(tensors, grad_tensors, retain_graph, create_graph, grad_variables, inputs)
    345     # some Python versions print out the first line of a multi-line function
    346     # calls in the traceback and some print out the last line
--> 347     _engine_run_backward(
    348         tensors,
    349         grad_tensors_,

/usr/local/lib/python3.11/dist-packages/torch/autograd/graph.py in _engine_run_backward(t_outputs, *args, **kwargs)
    821         unregister_hooks = _register_logging_hooks_on_whole_graph(t_outputs)
    822     try:
--> 823         return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
    824             t_outputs, *args, **kwargs
    825         )  # Calls into the C++ engine to run the backward pass

RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

## === cell 3
from torchvision.transforms import Compose, Resize
from torchvision.transforms import ToTensor, Normalize

import os
import pandas as pd
from PIL import Image
from PIL.Image import BICUBIC


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, root, path_list, targets=None, transform=None, extension=".png"):
        super().__init__()
        self.root = root
        self.path_list = path_list
        self.targets = targets
        self.transform = transform
        self.extension = extension
        if targets is not None:
            assert len(self.path_list) == len(self.targets)
            self.targets = torch.LongTensor(targets)

    def __getitem__(self, index):
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
)



## === cell 4
from torch.utils.data import DataLoader

batch_size = 16
num_workers = min(4, os.cpu_count() or 0)  # safer for Kaggle CPU env
print("num_workers:", num_workers)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=torch.cuda.is_available(),
)



## === cell 5
from tqdm import tqdm

preds = []
best_model.eval()
with torch.no_grad():
    for x, _ in tqdm(test_loader, total=len(test_loader)):
        x = x.to(device, non_blocking=True)

        pred = []
        for flip1 in range(2):  # flip 1st dim
            for flip2 in range(2):  # flip 2nd dim
                for trans in range(2):  # transpose
                    xx = x
                    if flip1:
                        xx = xx.flip(2)  # height
                    if flip2:
                        xx = xx.flip(3)  # width
                    if trans:
                        xx = xx.transpose(-1, -2)
                    pred.append(best_model(xx).unsqueeze(0))

        pred = F.softmax(torch.cat(pred, dim=0), dim=-1).mean(dim=0)
        batch_preds = torch.argmax(pred, dim=-1).detach().cpu().tolist()
        preds.extend(batch_preds)

print("num test preds:", len(preds))



## === cell 6
import numpy as np
import pandas as pd

df_test = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
assert len(preds) == len(
    df_test
), f"Prediction length {len(preds)} != test length {len(df_test)}"

sub = pd.DataFrame(
    {"id_code": df_test["id_code"].values, "diagnosis": np.array(preds, dtype=int)}
)
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)



## === cell 7
_ = sub.hist()



## === cell 8
tr = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
_ = tr.hist()
