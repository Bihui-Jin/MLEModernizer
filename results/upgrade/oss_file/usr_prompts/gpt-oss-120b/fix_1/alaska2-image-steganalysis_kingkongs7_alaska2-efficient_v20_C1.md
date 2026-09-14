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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.10

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
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
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.8278926088583237

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install -q efficientnet_pytorch
!pip install albumentations==0.5.2
from efficientnet_pytorch import EfficientNet
from albumentations.pytorch import ToTensorV2
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma, OneOf, Resize,
    ToFloat, ShiftScaleRotate, GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, GaussNoise, OpticalDistortion, RandomSizedCrop, VerticalFlip
)
import os
import torch
import pandas as pd
import numpy as np
import random
import torch.nn as nn
import matplotlib.pyplot as plt
from glob import glob
import torchvision
from torch.utils.data import Dataset
import time
from tqdm.notebook import tqdm
from sklearn import metrics
import cv2
import gc
import torch.nn.functional as F

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4199404493.py in <cell line: 0>()
      2 get_ipython().system('pip install albumentations==0.5.2')
      3 from efficientnet_pytorch import EfficientNet
----> 4 from albumentations.pytorch import ToTensorV2
      5 from albumentations import (
      6     Compose, HorizontalFlip, CLAHE, HueSaturationValue,

/usr/local/lib/python3.11/dist-packages/albumentations/__init__.py in <module>
      3 __version__ = "0.5.2"
      4 
----> 5 from .core.composition import *
      6 from .core.transforms_interface import *
      7 from .core.serialization import *

/usr/local/lib/python3.11/dist-packages/albumentations/core/composition.py in <module>
      6 import numpy as np
      7 
----> 8 from albumentations.augmentations.keypoints_utils import KeypointsProcessor
      9 from albumentations.core.serialization import SerializableMeta
     10 from albumentations.core.six import add_metaclass

/usr/local/lib/python3.11/dist-packages/albumentations/augmentations/__init__.py in <module>
      2 from .keypoints_utils import *
      3 from .bbox_utils import *
----> 4 from .functional import *
      5 from .transforms import *
      6 

/usr/local/lib/python3.11/dist-packages/albumentations/augmentations/functional.py in <module>
      7 import cv2
      8 import numpy as np
----> 9 from scipy.ndimage.filters import gaussian_filter
     10 
     11 from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox

/usr/local/lib/python3.11/dist-packages/scipy/ndimage/__init__.py in <module>
    154 # mypy: ignore-errors
    155 
--> 156 from ._support_alternative_backends import *
    157 
    158 # adjust __all__ and do not leak implementation details

/usr/local/lib/python3.11/dist-packages/scipy/ndimage/_support_alternative_backends.py in <module>
      5 
      6 import numpy as np
----> 7 from ._ndimage_api import *   # noqa: F403
      8 from . import _ndimage_api
      9 from . import _delegators

/usr/local/lib/python3.11/dist-packages/scipy/ndimage/_ndimage_api.py in <module>
      9 from ._filters import *    # noqa: F403
     10 from ._fourier import *   # noqa: F403
---> 11 from ._interpolation import *   # noqa: F403
     12 from ._measurements import *   # noqa: F403
     13 from ._morphology import *   # noqa: F403

/usr/local/lib/python3.11/dist-packages/scipy/ndimage/_interpolation.py in <module>
     35 from scipy._lib._util import normalize_axis_index
     36 
---> 37 from scipy import special
     38 from . import _ni_support
     39 from . import _nd_image

/usr/lib/python3.11/importlib/_bootstrap.py in _handle_fromlist(module, fromlist, import_, recursive)

/usr/local/lib/python3.11/dist-packages/scipy/__init__.py in __getattr__(name)
    132 def __getattr__(name):
    133     if name in submodules:
--> 134         return _importlib.import_module(f'scipy.{name}')
    135     else:
    136         try:

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/local/lib/python3.11/dist-packages/scipy/special/__init__.py in <module>
    824     chdtr, chdtrc, betainc, betaincc, stdtr)
    825 
--> 826 from . import _basic
    827 from ._basic import *
    828 

/usr/local/lib/python3.11/dist-packages/scipy/special/_basic.py in <module>
     20 from . import _specfun
     21 from ._comb import _comb_int
---> 22 from ._multiufuncs import (assoc_legendre_p_all,
     23                            legendre_p_all)
     24 from scipy._lib.deprecation import _deprecated

/usr/local/lib/python3.11/dist-packages/scipy/special/_multiufuncs.py in <module>
    140 
    141 
--> 142 sph_legendre_p = MultiUFunc(
    143     sph_legendre_p,
    144     r"""sph_legendre_p(n, m, theta, *, diff_n=0)

/usr/local/lib/python3.11/dist-packages/scipy/special/_multiufuncs.py in __init__(self, ufunc_or_ufuncs, doc, force_complex_output, **default_kwargs)
     39             for ufunc in ufuncs_iter:
     40                 if not isinstance(ufunc, np.ufunc):
---> 41                     raise ValueError("All ufuncs must have type `numpy.ufunc`."
     42                                      f" Received {ufunc_or_ufuncs}")
     43                 seen_input_types.add(frozenset(x.split("->")[0] for x in ufunc.types))

ValueError: All ufuncs must have type `numpy.ufunc`. Received (<ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>, <ufunc 'sph_legendre_p'>)

## === cell 1
seed = 42
print(f'setting everything to seed {seed}')
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1558417043.py in <cell line: 0>()
      2 seed = 42
      3 print(f'setting everything to seed {seed}')
----> 4 random.seed(seed)
      5 os.environ['PYTHONHASHSEED'] = str(seed)
      6 np.random.seed(seed)

NameError: name 'random' is not defined

## === cell 2
data_dir = '../input/alaska2-image-steganalysis'
sample_size = 75000
val_size = int(sample_size*0.25)
train_fn, val_fn = [], []
train_labels, val_labels = [], []

folder_names = ['Cover/','JMiPOD/', 'JUNIWARD/', 'UERD/'] # label 1 2 3
for label, folder in enumerate(folder_names):
    train_filenames = sorted(glob(f"{data_dir}/{folder}/*.jpg")[:sample_size])
    np.random.shuffle(train_filenames)
    train_fn.extend(train_filenames[val_size:])
    train_labels.extend(np.zeros(len(train_filenames[val_size:],))+label)
    val_fn.extend(train_filenames[:val_size])
    val_labels.extend(np.zeros(len(train_filenames[:val_size],))+label)

assert len(train_labels) == len(train_fn), "wrong labels"
assert len(val_labels) == len(val_fn), "wrong labels"

train_df = pd.DataFrame({'ImageFileName': train_fn, 'Label': train_labels}, columns=['ImageFileName', 'Label'])
train_df['Label'] = train_df['Label'].astype(int)
val_df = pd.DataFrame({'ImageFileName': val_fn, 'Label': val_labels}, columns=['ImageFileName', 'Label'])
val_df['Label'] = val_df['Label'].astype(int)

print(train_df)
train_df.Label.hist()

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/814782364.py in <cell line: 0>()
      8 folder_names = ['Cover/','JMiPOD/', 'JUNIWARD/', 'UERD/'] # label 1 2 3
      9 for label, folder in enumerate(folder_names):
---> 10     train_filenames = sorted(glob(f"{data_dir}/{folder}/*.jpg")[:sample_size])
     11     np.random.shuffle(train_filenames)
     12     train_fn.extend(train_filenames[val_size:])

NameError: name 'glob' is not defined

## === cell 3
class Alaska2Dataset(Dataset):

    def __init__(self, df, augmentations=None):

        self.data = df
        self.augment = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn, label = self.data.loc[idx]
        im = cv2.imread(fn)[:, :, ::-1]
        if self.augment:
            im = self.augment(image=im)
        return im, label


img_size = 512
AUGMENTATIONS_TRAIN = Compose([
    Resize(img_size, img_size, p=1),
    VerticalFlip(p=0.5),
    HorizontalFlip(p=0.5),
    JpegCompression(quality_lower=75, quality_upper=100,p=0.5),
    ToFloat(max_value=255),
    ToTensorV2()
], p=1)


AUGMENTATIONS_TEST = Compose([
    Resize(img_size, img_size, p=1),
    ToFloat(max_value=255),
    ToTensorV2()
], p=1)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1278310570.py in <cell line: 0>()
      1 # Pytorch 数据集构建
----> 2 class Alaska2Dataset(Dataset):
      3 
      4     def __init__(self, df, augmentations=None):
      5 

NameError: name 'Dataset' is not defined

## === cell 4
temp_df = train_df.sample(64).reset_index(drop=True)
train_dataset = Alaska2Dataset(temp_df, augmentations=AUGMENTATIONS_TEST)
batch_size = 64
num_workers = 0

temp_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          num_workers=num_workers, shuffle=False)


images, labels = next(iter(temp_loader))
images = images['image'].permute(0, 2, 3, 1)
max_images = 64
grid_width = 16
grid_height = int(max_images / grid_width)
fig, axs = plt.subplots(grid_height, grid_width,figsize=(grid_width+1, grid_height+1))

for i, (im, label) in enumerate(zip(images, labels)):
    ax = axs[int(i / grid_width), i % grid_width]
    ax.imshow(im.squeeze())
    ax.set_title(str(label.item()))
    ax.axis('off')

plt.suptitle("0: COVER, 1: JMiPOD, 2: JUNIWARD, 3:UERD")
plt.show()
del images
gc.collect()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1209027910.py in <cell line: 0>()
      1 # test 数据集 不做图像增强
----> 2 temp_df = train_df.sample(64).reset_index(drop=True)
      3 train_dataset = Alaska2Dataset(temp_df, augmentations=AUGMENTATIONS_TEST)
      4 batch_size = 64
      5 num_workers = 0

NameError: name 'train_df' is not defined

## === cell 5
train_dataset = Alaska2Dataset(temp_df, augmentations=AUGMENTATIONS_TRAIN)
batch_size = 64
num_workers = 0

temp_loader = torch.utils.data.DataLoader(train_dataset,
                                          batch_size=batch_size,
                                          num_workers=num_workers, shuffle=False)


images, labels = next(iter(temp_loader))
images = images['image'].permute(0, 2, 3, 1)
max_images = 64
grid_width = 16
grid_height = int(max_images / grid_width)
fig, axs = plt.subplots(grid_height, grid_width,
                        figsize=(grid_width+1, grid_height+1))

for i, (im, label) in enumerate(zip(images, labels)):
    ax = axs[int(i / grid_width), i % grid_width]
    ax.imshow(im.squeeze())
    ax.set_title(str(label.item()))
    ax.axis('off')

plt.suptitle("0: No Hidden Message, 1: JMiPOD, 2: JUNIWARD, 3:UERD")
plt.show()
del images, temp_df
gc.collect()

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/811813234.py in <cell line: 0>()
      1 # train 数据集 进行图像增强
----> 2 train_dataset = Alaska2Dataset(temp_df, augmentations=AUGMENTATIONS_TRAIN)
      3 batch_size = 64
      4 num_workers = 0
      5 

NameError: name 'Alaska2Dataset' is not defined

## === cell 6
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        self.dense_output = nn.Linear(1280, 4)

    def forward(self, x):
        feat = self.model.extract_features(x)
        feat = F.avg_pool2d(feat, feat.size()[2:]).reshape(-1, 1280)
        return self.dense_output(feat)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/730829261.py in <cell line: 0>()
      1 # 搭建 网络 efficient b0
----> 2 class Net(nn.Module):
      3     def __init__(self):
      4         super().__init__()
      5         self.model = EfficientNet.from_pretrained('efficientnet-b0')

NameError: name 'nn' is not defined

## === cell 7
batch_size = 8
num_workers = 8

train_dataset = Alaska2Dataset(train_df, augmentations=AUGMENTATIONS_TRAIN)
valid_dataset = Alaska2Dataset(val_df, augmentations=AUGMENTATIONS_TEST)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)

valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=batch_size*2,
                                           num_workers=num_workers,
                                           shuffle=False)

device = 'cuda'
model = Net().to(device)
model.load_state_dict(torch.load('../input/alaska/epoch_10_val_loss_6.73_auc_0.815.pth'))
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1872917340.py in <cell line: 0>()
      3 
      4 # 构建数据集
----> 5 train_dataset = Alaska2Dataset(train_df, augmentations=AUGMENTATIONS_TRAIN)
      6 valid_dataset = Alaska2Dataset(val_df, augmentations=AUGMENTATIONS_TEST)
      7 

NameError: name 'Alaska2Dataset' is not defined

## === cell 8
def alaska_weighted_auc(y_true, y_valid):
    tpr_thresholds = [0.0, 0.4, 1.0]
    weights = [2, 1]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_valid, pos_label=1)

    areas = np.array(tpr_thresholds[1:]) - np.array(tpr_thresholds[:-1])

    normalization = np.dot(areas, weights)

    competition_metric = 0
    for idx, weight in enumerate(weights):
        y_min = tpr_thresholds[idx]
        y_max = tpr_thresholds[idx + 1]
        mask = (y_min < tpr) & (tpr < y_max)

        x_padding = np.linspace(fpr[mask][-1], 1, 100)

        x = np.concatenate([fpr[mask], x_padding])
        y = np.concatenate([tpr[mask], [y_max] * len(x_padding)])
        y = y - y_min  # normalize such that curve starts at y=0
        score = metrics.auc(x, y)
        submetric = score * weight
        best_subscore = (y_max - y_min) * weight
        competition_metric += submetric

    return competition_metric / normalization

## === cell 9
criterion = torch.nn.CrossEntropyLoss()
num_epochs = 0
train_loss, val_loss = [], []

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    model.train()
    running_loss = 0
    tk0 = tqdm(train_loader, total=int(len(train_loader)))
    for im, labels in tk0:
        inputs = im["image"].to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        tk0.set_postfix(loss=(loss.item()))

    epoch_loss = running_loss / (len(train_loader)/batch_size)
    train_loss.append(epoch_loss)
    print('Training Loss: {:.8f}'.format(epoch_loss))

    tk1 = tqdm(valid_loader, total=int(len(valid_loader)))
    model.eval()
    running_loss = 0
    y, preds = [], []
    with torch.no_grad():
        for (im, labels) in tk1:
            inputs = im["image"].to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            y.extend(labels.cpu().numpy().astype(int))
            preds.extend(F.softmax(outputs, 1).cpu().numpy())
            running_loss += loss.item()
            tk1.set_postfix(loss=(loss.item()))

        epoch_loss = running_loss / (len(valid_loader)/batch_size)
        val_loss.append(epoch_loss)
        preds = np.array(preds)
        labels = preds.argmax(1)
        acc = (labels == y).mean()*100
        new_preds = np.zeros((len(preds),))
        temp = preds[labels != 0, 1:]
        new_preds[labels != 0] = temp.sum(1)
        new_preds[labels == 0] = preds[labels == 0, 0]
        y = np.array(y)
        y[y != 0] = 1
        auc_score = alaska_weighted_auc(y, new_preds)
        print(f'Val Loss: {epoch_loss:.3}, Weighted AUC:{auc_score:.3}, Acc: {acc:.3}')

    torch.save(model.state_dict(),f"epoch_{epoch+4}_val_loss_{epoch_loss:.3}_auc_{auc_score:.3}.pth")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/532572199.py in <cell line: 0>()
----> 1 criterion = torch.nn.CrossEntropyLoss()
      2 num_epochs = 0
      3 train_loss, val_loss = [], []
      4 
      5 for epoch in range(num_epochs):

NameError: name 'torch' is not defined

## === cell 10
plt.figure(figsize=(15,7))
plt.plot(train_loss, c='r')
plt.plot(val_loss, c='b')
plt.legend(['train_loss', 'val_loss'])
plt.title('Loss Plot')

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4031139402.py in <cell line: 0>()
----> 1 plt.figure(figsize=(15,7))
      2 plt.plot(train_loss, c='r')
      3 plt.plot(val_loss, c='b')
      4 plt.legend(['train_loss', 'val_loss'])
      5 plt.title('Loss Plot')

NameError: name 'plt' is not defined

## === cell 12
class Alaska2TestDataset(Dataset):

    def __init__(self, df, augmentations=None):

        self.data = df
        self.augment = augmentations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fn = self.data.loc[idx][0]
        im = cv2.imread(fn)[:, :, ::-1]

        if self.augment:
            im = self.augment(image=im)

        return im


test_filenames = sorted(glob(f"{data_dir}/Test/*.jpg"))
test_df = pd.DataFrame({'ImageFileName': list(test_filenames)}, columns=['ImageFileName'])

batch_size = 16
num_workers = 4
test_dataset = Alaska2TestDataset(test_df, augmentations=AUGMENTATIONS_TEST)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          shuffle=False,
                                          drop_last=False)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/130559582.py in <cell line: 0>()
      1 # 构建测试集数据
----> 2 class Alaska2TestDataset(Dataset):
      3 
      4     def __init__(self, df, augmentations=None):
      5 

NameError: name 'Dataset' is not defined

## === cell 13
model.eval()

preds = []
tk0 = tqdm(test_loader)
with torch.no_grad():
    for i, im in enumerate(tk0):
        inputs = im["image"].to(device)
        im = inputs.flip(2)
        outputs = model(im)
        im = inputs.flip(3)
        outputs = (0.25*outputs + 0.25*model(im))
        outputs = (outputs + 0.5*model(inputs))        
        preds.extend(F.softmax(outputs, 1).cpu().numpy())

preds = np.array(preds)
labels = preds.argmax(1)
new_preds = np.zeros((len(preds),))
temp = preds[labels != 0, 1:]
new_preds[labels != 0] = temp.sum(1)
new_preds[labels == 0] = preds[labels == 0, 0]

test_df['Id'] = test_df['ImageFileName'].apply(lambda x: x.split(os.sep)[-1])
test_df['Label'] = new_preds

test_df = test_df.drop('ImageFileName', axis=1)
test_df.to_csv('submission.csv', index=False)
print(test_df.head())

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3941175160.py in <cell line: 0>()
----> 1 model.eval()
      2 
      3 preds = []
      4 tk0 = tqdm(test_loader)
      5 with torch.no_grad():

NameError: name 'model' is not defined
