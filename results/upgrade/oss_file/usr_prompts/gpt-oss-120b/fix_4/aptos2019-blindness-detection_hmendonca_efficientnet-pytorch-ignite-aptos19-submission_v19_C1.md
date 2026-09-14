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
scipy==1.15.3
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

0.7783388837413378

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
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
            nn.Conv2d(inplanes, se_planes, kernel_size=1, bias=True),
            Swish(),
            nn.Conv2d(se_planes, inplanes, kernel_size=1, bias=True),
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
        drop_mask = torch.rand(x.shape[0], 1, 1, 1, device=x.device) + keep_prob
        drop_mask.floor_()
        return drop_mask * x / keep_prob

    def forward(self, x):
        shortcut = x
        if self.expansion_conv is not None:
            x = self.expansion_conv(x)

        x = self.depthwise_conv(x)
        x = self.squeeze_excitation(x)
        x = self.project_conv(x)

        if self.with_skip and x.shape == shortcut.shape:
            if self.training and self.drop_connect_rate is not None:
                x = self._drop_connect(x)
            x = x + shortcut
        return x


from collections import OrderedDict
import math


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_out")
    elif isinstance(module, nn.Linear):
        init_range = 1.0 / math.sqrt(module.weight.shape[1])
        nn.init.uniform_(module.weight, -init_range, init_range)


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
            expand = expand_rates[idx]
            k = kernel_sizes[idx]
            s = strides[idx]
            drop = drop_connect_rate * counter / num_blocks

            blocks.append(
                (
                    f"MBConv{expand}_{counter}",
                    MBConv(
                        in_c,
                        out_c,
                        kernel_size=k,
                        stride=s,
                        expand_rate=expand,
                        se_rate=se_rate,
                        drop_connect_rate=drop,
                    ),
                )
            )
            counter += 1
            for _ in range(1, repeats):
                drop = drop_connect_rate * counter / num_blocks
                blocks.append(
                    (
                        f"MBConv{expand}_{counter}",
                        MBConv(
                            out_c,
                            out_c,
                            kernel_size=k,
                            stride=1,
                            expand_rate=expand,
                            se_rate=se_rate,
                            drop_connect_rate=drop,
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
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x




## === cell 1
image_size = 380
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
best_model = EfficientNet(
    num_classes=1, width_coefficient=1.4, depth_coefficient=1.8
)  # B4
if os.path.exists(model_path):
    try:
        best_model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded pretrained checkpoint.")
    except Exception as e:
        print("Failed to load checkpoint:", e)
else:
    print("Checkpoint not found; using random initialization.")

best_model = best_model.to(device).eval()




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2914418465.py in <cell line: 0>()
      4     num_classes=1, width_coefficient=1.4, depth_coefficient=1.8
      5 )  # B4
----> 6 if os.path.exists(model_path):
      7     try:
      8         best_model.load_state_dict(torch.load(model_path, map_location=device))

NameError: name 'os' is not defined

## === cell 2
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root, path_list, targets=None, transform=None, extension=".png"):
        self.root = root
        self.path_list = path_list
        self.targets = torch.LongTensor(targets) if targets is not None else None
        self.transform = transform
        self.extension = extension
        if self.targets is not None:
            assert len(self.path_list) == len(self.targets)

    def __getitem__(self, idx):
        path = self.path_list[idx]
        img = Image.open(os.path.join(self.root, path + self.extension)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.targets is not None:
            return img, self.targets[idx]
        else:
            return img, torch.LongTensor([])

    def __len__(self):
        return len(self.path_list)


from PIL.Image import BICUBIC

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




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2024415919.py in <cell line: 0>()
     37 )
     38 
---> 39 df_test = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")
     40 test_dataset = ImageDataset(
     41     root="../input/aptos2019-blindness-detection/test_images",

NameError: name 'pd' is not defined

## === cell 3
from torch.utils.data import DataLoader

batch_size = 16
num_workers = max(0, os.cpu_count() - 1)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    drop_last=False,
    pin_memory=True,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/823588886.py in <cell line: 0>()
      2 
      3 batch_size = 16
----> 4 num_workers = max(0, os.cpu_count() - 1)
      5 test_loader = DataLoader(
      6     test_dataset,

NameError: name 'os' is not defined

## === cell 4
def tta(x):
    """Simple 8‑fold test‑time augmentation."""
    preds = []
    for flip1 in range(2):
        xf = x.flip(2) if flip1 else x
        for flip2 in range(2):
            xf2 = xf.flip(3) if flip2 else xf
            for trans in range(2):
                xt = xf2.transpose(-1, -2) if trans else xf2
                preds.append(best_model(xt)[..., -1].unsqueeze(0))
    return torch.cat(preds).mean(dim=0)




## === cell 5
from tqdm import tqdm




## === cell 6
preds = []
best_model.eval()
with torch.no_grad():
    for x, _ in tqdm(test_loader, total=len(test_loader)):
        x = x.to(device)
        pred = tta(x)
        preds.extend(pred.cpu().squeeze().tolist())




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1150780948.py in <cell line: 0>()
      2 best_model.eval()
      3 with torch.no_grad():
----> 4     for x, _ in tqdm(test_loader, total=len(test_loader)):
      5         x = x.to(device)
      6         pred = tta(x)

NameError: name 'test_loader' is not defined

## === cell 7
import scipy as sp
from sklearn.metrics import cohen_kappa_score


class KappaOptimizer(nn.Module):
    def __init__(self, coef=[0.5, 1.5, 2.5, 3.5]):
        super().__init__()
        self.coef = coef

    def predict(self, preds):
        return self._predict(self.coef, preds)

    @staticmethod
    def _predict(coef, preds):
        if isinstance(preds, torch.Tensor):
            y_hat = preds.clone().view(-1)
        else:
            y_hat = torch.tensor(preds, dtype=torch.float32).view(-1)

        for i, p in enumerate(y_hat):
            if p < coef[0]:
                y_hat[i] = 0
            elif p < coef[1]:
                y_hat[i] = 1
            elif p < coef[2]:
                y_hat[i] = 2
            elif p < coef[3]:
                y_hat[i] = 3
            else:
                y_hat[i] = 4
        return y_hat.int()

    def quad_kappa(self, preds, y):
        return cohen_kappa_score(
            y, self._predict(self.coef, preds), weights="quadratic"
        )

    def fit(self, preds, y):
        print("Early score:", self.quad_kappa(preds, y))




## === cell 8
kappa_opt = KappaOptimizer([0.60775735, 0.84896643, 2.61878963, 3.35916085])
opt_preds = kappa_opt.predict(preds).tolist()




## === cell 9
import numpy as np

sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
assert len(opt_preds) == len(sub), f"Prediction length {len(opt_preds)} != {len(sub)}"
sub["diagnosis"] = np.array(opt_preds, dtype=np.int32)
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv with", len(sub), "rows.")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/104735953.py in <cell line: 0>()
      1 import numpy as np
      2 
----> 3 sub = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
      4 assert len(opt_preds) == len(sub), f"Prediction length {len(opt_preds)} != {len(sub)}"
      5 sub["diagnosis"] = np.array(opt_preds, dtype=np.int32)

NameError: name 'pd' is not defined
