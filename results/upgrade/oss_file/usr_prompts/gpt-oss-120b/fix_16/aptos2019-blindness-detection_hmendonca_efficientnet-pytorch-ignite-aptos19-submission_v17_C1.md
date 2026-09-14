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
import os, glob, shutil, hashlib, pandas as pd, torch, torch.nn as nn, numpy as np, copy
from collections import OrderedDict
import math
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score  # added for QWK evaluation

torch.backends.cudnn.allow_tf32 = True  # enable TF32 on GPU for speed
torch.set_float32_matmul_precision("high")  # higher‑precision matmul for CPU/GPU speed



## === cell 1
target_pattern = "../input/efficientnet*/efficientNet_*.pth"
matches = glob.glob(target_pattern)
if matches:
    model_path = "efficientNet_best.pth"
    src = matches[0]
    shutil.copy(src, model_path)
    md5_hash = hashlib.md5()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)
    print(f"Copied pretrained weights from {src} to {model_path}")
    print("MD5:", md5_hash.hexdigest())
else:
    model_path = None
    print(
        "No pretrained EfficientNet weights found; the model will use random initialization."
    )




## === cell 2
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
            x = x + z
        return x


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
            for _ in range(1, num_repeats):
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
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(list_channels[-1], num_classes),
        )
        self.apply(init_weights)

    def forward(self, x):
        f = self.stem(x)
        f = self.blocks(f)
        y = self.head(f)
        return y




## === cell 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

best_model = EfficientNet(
    num_classes=5, width_coefficient=1.4, depth_coefficient=1.8
)  # B4 architecture
if model_path is not None and os.path.exists(model_path):
    try:
        best_model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded pretrained weights.")
    except Exception as e:
        print(f"Failed to load weights: {e}. Proceeding with random init.")
else:
    print("Pretrained weights not available; using randomly initialized model.")
best_model = best_model.to(device)

best_model = torch.compile(best_model, mode="reduce-overhead")

best_model.eval()



## === cell 4
image_size = 380
from PIL.Image import BICUBIC

train_transform = Compose(
    [
        Resize((image_size, image_size), BICUBIC),
        ToTensor(),
        Normalize(mean=[0.42, 0.22, 0.075], std=[0.27, 0.15, 0.081]),
    ]
)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, path_list, targets=None, transform=None, extension=".png"):
        super().__init__()
        self.root = root
        self.path_list = path_list
        self.targets = torch.LongTensor(targets) if targets is not None else None
        self.transform = transform
        self.extension = extension
        if self.targets is not None:
            assert len(self.path_list) == len(self.targets)

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


train_csv = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
full_dataset = ImageDataset(
    root="../input/aptos2019-blindness-detection/train_images",
    path_list=train_csv.id_code.values,
    targets=train_csv.diagnosis.values,
    transform=train_transform,
)

val_ratio = 0.2
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)

batch_size = 32
num_workers = 0

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=False,
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=False,
)

for param in best_model.parameters():
    param.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(best_model.parameters(), lr=1e-3)



## === cell 5
epochs = 8  # a few more epochs for better learning
best_val_qwk = -float("inf")
best_state_dict = copy.deepcopy(best_model.state_dict())

scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

class_range = torch.arange(5, device=device, dtype=torch.float32)

best_model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for imgs, labels in tqdm(
        train_loader, desc=f"Fine‑tune epoch {epoch+1}/{epochs} (train)"
    ):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            outputs = best_model(imgs)
            loss = criterion(outputs, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)
    avg_train_loss = epoch_loss / len(train_loader.dataset)

    best_model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for imgs, labels in tqdm(
            val_loader, desc=f"Fine‑tune epoch {epoch+1}/{epochs} (val)"
        ):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                outputs = best_model(imgs)
                loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)

            probs = F.softmax(outputs, dim=1)
            expected = (probs * class_range).sum(dim=1)
            pred = torch.round(expected).clamp(0, 4).long()
            val_preds.extend(pred.cpu().tolist())
            val_targets.extend(labels.cpu().tolist())
    avg_val_loss = val_loss / len(val_loader.dataset)

    val_qwk = cohen_kappa_score(val_targets, val_preds, weights="quadratic")
    print(
        f"Epoch {epoch+1}: train loss {avg_train_loss:.4f} – val loss {avg_val_loss:.4f} – val QWK {val_qwk:.4f}"
    )

    if val_qwk > best_val_qwk:
        best_val_qwk = val_qwk
        best_state_dict = copy.deepcopy(best_model.state_dict())
        print("  -> New best QWK model saved.")

    best_model.train()

best_model.load_state_dict(best_state_dict)
best_model.eval()



## === cell 6
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



## === cell 7
batch_size = 32
num_workers = 0
print("num_workers:", num_workers)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
    persistent_workers=False,
)




## === cell 8
def tta(x):
    """8‑fold Test‑Time Augmentation batched into a single forward pass"""
    orig = x
    aug = []
    for flip1 in range(2):
        x1 = orig.flip(1) if flip1 else orig
        for flip2 in range(2):
            x2 = x1.flip(2) if flip2 else x1
            for trans in range(2):
                x3 = x2.transpose(-1, -2) if trans else x2
                aug.append(x3)
    aug_batch = torch.cat(aug, dim=0)  # (8*B, C, H, W)
    with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
        logits = best_model(aug_batch)
    probs = F.softmax(logits, dim=1)  # (8*B, 5)
    probs = probs.view(8, x.shape[0], -1).mean(dim=0)  # average over augmentations
    return probs




## === cell 9
preds = []
with torch.no_grad():
    for x, _ in tqdm(test_loader, total=len(test_loader)):
        x = x.to(device, non_blocking=True)
        probs = tta(x)  # (B,5) probabilities
        expected = (probs * class_range).sum(dim=1)
        pred = torch.round(expected).clamp(0, 4).long()
        preds.extend(pred.cpu().tolist())
print(f"Generated {len(preds)} predictions.")



## === cell 10
sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
if len(preds) != len(sub):
    print("Prediction length mismatch – adjusting to match submission rows.")
    if len(preds) > len(sub):
        preds = preds[: len(sub)]
    else:
        preds = preds + [0] * (len(sub) - len(preds))
sub["diagnosis"] = np.array(preds, dtype=int)
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", sub.shape)
print(sub.head())



## === cell 11
sub_hist = sub["diagnosis"].hist()
sub_hist.set_xlabel("Diagnosis")
sub_hist.set_ylabel("Count")
sub_hist.set_title("Prediction Distribution")



## === cell 12
tr = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
_ = tr["diagnosis"].hist()
