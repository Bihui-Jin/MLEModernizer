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
Create a model to automatically segment the stomach and intestines on MRI scans.

## Metric
Mean Dice coefficient and 3D Hausdorff distance. 

The Dice coefficient can be used to compare the pixel-wise agreement between a predicted segmentation and its corresponding ground truth. The formula is given by:

$$
\frac{2 \cdot |X \cap Y|}{|X| + |Y|}
$$

where $X$ is the predicted set of pixels and $Y$ is the ground truth. The Dice coefficient is defined to be 0 when both $X$ and $Y$ are empty. 

Hausdorff distance is a method for calculating the distance between segmentation objects A and B, by calculating the furthest point on object A from the nearest point on object B. For 3D Hausdorff, we construct 3D volumes by combining each 2D segmentation with slice depth as the Z coordinate and then find the Hausdorff distance between them. (Here the slice depth for all scans is set to 1). The expected / predicted pixel locations are normalized by image size to create a bounded 0-1 score.

The two metrics are combined, with a weight of 0.4 for the Dice metric and 0.6 for the Hausdorff distance.

## Submission Format
Use run-length encoding on the pixel values.  Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

Note that, at the time of encoding, the mask should be binary, meaning the masks for all objects in an image are joined into a single large mask. A value of 0 should indicate pixels that are not masked, and a value of 1 will indicate pixels that are masked.

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. The pixels are numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The file should contain a header and have the following format:

```
id,class,predicted
1,large_bowel,1 1 5 1
1,small_bowel,1 1
1,stomach,1 1
2,large_bowel,1 5 2 17
etc.
```

## Dataset
Each case is represented by multiple sets of scan slices (each set is identified by the day the scan took place). Some cases are split by time (early days are in train, later days are in test) while some cases are split by case - the entirety of the case is in train or test. The goal is to be able to generalize to both partially and wholly unseen cases.

### Files
- train.csv - IDs and masks for all training objects.
- sample_submission.csv - a sample submission file in the correct format
- train - a folder of case/day folders, each containing slice images for a particular case on a given day.

Note that the image filenames include 4 numbers (ex. 276_276_1.63_1.63.png). These four numbers are slice width / height (integers in pixels) and width/height pixel spacing (floating points in mm). The first two defines the resolution of the slide. The last two record the physical size of each pixel.

Physical pixel thickness in superior-inferior direction is 3mm.

### Columns
- `id` - unique identifier for object
- `class` - the predicted class for the object
- `segmentation` - RLE-encoded pixels for the identified object

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (126 lines)
            sample_submission.csv (20401 lines)
            sample_submission.csv.zip (57.4 kB)
            test.csv (20401 lines)
            test.csv.zip (55.2 kB)
            test.zip (432.9 MB)
            train.csv (95089 lines)
            train.csv.zip (6.7 MB)
            train.zip (2.0 GB)
            test/
                case110/
                    case110_day12/
                        scans/
                            ... (max depth reached)
                    case110_day16/
                        scans/
                            ... (max depth reached)
                case113/
                    case113_day22/
                        scans/
                            ... (max depth reached)
                ... and 27 other folders
            train/
                case101/
                    case101_day20/
                        scans/
                            ... (max depth reached)
                    case101_day22/
                        scans/
                            ... (max depth reached)
                    case101_day26/
                        scans/
                            ... (max depth reached)
                    case101_day32/
                        scans/
                            ... (max depth reached)
                case102/
                    case102_day0/
                        scans/
                            ... (max depth reached)
                ... and 75 other folders
            uw-madison-gi-tract-image-segmentation/
                description.md (126 lines)
                sample_submission.csv (20401 lines)
                ... and 7 other files
                test/
                    case110/
                        case110_day12/
                            ... (max depth reached)
                        case110_day16/
                            ... (max depth reached)
                    case113/
                        case113_day22/
                            ... (max depth reached)
                    ... and 27 other folders
                train/
                    case101/
                        case101_day20/
                            ... (max depth reached)
                        case101_day22/
                            ... (max depth reached)
                        case101_day26/
                            ... (max depth reached)
                        case101_day32/
                            ... (max depth reached)
                    case102/
                        case102_day0/
                            ... (max depth reached)
                    ... and 75 other folders
                uw-madison-gi-tract-image-segmentation/
        input/
            description.md (126 lines)
            sample_submission.csv (20401 lines)
            sample_submission.csv.zip (57.4 kB)
            test.csv (20401 lines)
            test.csv.zip (55.2 kB)
            test.zip (432.9 MB)
            train.csv (95089 lines)
            train.csv.zip (6.7 MB)
            train.zip (2.0 GB)
            test/
                case110/
                    case110_day12/
                        scans/
                            ... (max depth reached)
                    case110_day16/
                        scans/
                            ... (max depth reached)
                case113/
                    case113_day22/
                        scans/
                            ... (max depth reached)
                ... and 27 other folders
            train/
                case101/
                    case101_day20/
                        scans/
                            ... (max depth reached)
                    case101_day22/
                        scans/
                            ... (max depth reached)
                    case101_day26/
                        scans/
                            ... (max depth reached)
                    case101_day32/
                        scans/
                            ... (max depth reached)
                case102/
                    case102_day0/
                        scans/
                            ... (max depth reached)
                ... and 75 other folders
            uw-madison-gi-tract-image-segmentation/
                description.md (126 lines)
                sample_submission.csv (20401 lines)
                ... and 7 other files
                test/
                    case110/
                        case110_day12/
                            ... (max depth reached)
                        case110_day16/
                            ... (max depth reached)
                    case113/
                        case113_day22/
                            ... (max depth reached)
                    ... and 27 other folders
                train/
                    case101/
                        case101_day20/
                            ... (max depth reached)
                        case101_day22/
                            ... (max depth reached)
                        case101_day26/
                            ... (max depth reached)
                        case101_day32/
                            ... (max depth reached)
                    case102/
                        case102_day0/
                            ... (max depth reached)
                    ... and 75 other folders
                uw-madison-gi-tract-image-segmentation/
        working/
            uw-madison-gi-tract-image-segmentation/
                description.md (126 lines)
                sample_submission.csv (20401 lines)
                ... and 7 other files
                test/
                    case110/
                        case110_day12/
                            ... (max depth reached)
                        case110_day16/
                            ... (max depth reached)
                    case113/
                        case113_day22/
                            ... (max depth reached)
                    ... and 27 other folders
                train/
                    case101/
                        case101_day20/
                            ... (max depth reached)
                        case101_day22/
                            ... (max depth reached)
                        case101_day26/
                            ... (max depth reached)
                        case101_day32/
                            ... (max depth reached)
                    case102/
                        case102_day0/
                            ... (max depth reached)
                    ... and 75 other folders
                uw-madison-gi-tract-image-segmentation/
```

-> data/sample_submission.csv has 20400 rows and 3 columns.
The columns are: id, class, predicted

-> data/test.csv has 20400 rows and 2 columns.
The columns are: id, class

-> data/train.csv has 95088 rows and 3 columns.
The columns are: id, class, segmentation

-> data/uw-madison-gi-tract-image-segmentation/sample_submission.csv has 20400 rows and 3 columns.
The columns are: id, class, predicted

-> data/uw-madison-gi-tract-image-segmentation/test.csv has 20400 rows and 2 columns.
The columns are: id, class

-> data/uw-madison-gi-tract-image-segmentation/train.csv has 95088 rows and 3 columns.
The columns are: id, class, segmentation

-> input/sample_submission.csv has 20400 rows and 3 columns.
The columns are: id, class, predicted

-> (stopped after 10 files for performance)

# 5. Target score

0.6435472688166154

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
!pip uninstall -y torchtext
!mkdir -p frozen_packages/
!cp ../input/demo-flash-semantic-segmentation/frozen_packages/* frozen_packages/
!cp ../input/tract-segm-eda-3d-interactive-viewer/frozen_packages/* frozen_packages/
!pip install -q "lightning-flash[image]" "torchmetrics<0.8" --pre --no-index --find-links frozen_packages/
!pip install -q -U timm segmentation-models-pytorch --no-index --find-links frozen_packages/
!pip install -q 'kaggle-image-segmentation' --no-index --find-links frozen_packages/

! pip list | grep -e torch -e lightning
! nvidia-smi -L

## === cell 2
import os, glob
import pandas as pd
import matplotlib.pyplot as plt

DATASET_FOLDER = "/kaggle/input/uw-madison-gi-tract-image-segmentation"
df_train = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))
display(df_train.head())

df_pred = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
WITH_SUBMISSION = not df_pred.empty

## === cell 3
all_imgs = glob.glob(os.path.join(DATASET_FOLDER, "train", "case*", "case*_day*", "scans", "*.png"))
all_imgs = [p.replace(DATASET_FOLDER, "") for p in all_imgs]

print(f"images: {len(all_imgs)}")
print(f"annotated: {len(df_train['id'].unique())}")

## === cell 4
from pprint import pprint
from kaggle_imsegm.data_io import extract_tract_details

pprint(extract_tract_details(df_train['id'].iloc[0], DATASET_FOLDER))

df_train[['Case','Day','Slice', 'image', 'image_path', 'height', 'width']] = df_train['id'].apply(
    lambda x: pd.Series(extract_tract_details(x, DATASET_FOLDER))
)
display(df_train.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/218003621.py in <cell line: 0>()
      1 from pprint import pprint
----> 2 from kaggle_imsegm.data_io import extract_tract_details
      3 
      4 pprint(extract_tract_details(df_train['id'].iloc[0], DATASET_FOLDER))
      5 

ModuleNotFoundError: No module named 'kaggle_imsegm'

## === cell 6
import os.path
from typing import Callable, Tuple, Sequence

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from kaggle_imsegm.dataset import TractDataset2D

ds = TractDataset2D(df_train, DATASET_FOLDER)
print(len(ds))

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1970503854.py in <cell line: 0>()
      9 from tqdm.auto import tqdm
     10 
---> 11 from kaggle_imsegm.dataset import TractDataset2D
     12 
     13 ds = TractDataset2D(df_train, DATASET_FOLDER)

ModuleNotFoundError: No module named 'kaggle_imsegm'

## === cell 7
spl = ds[255]
img, seg = spl["input"], spl["target"]
print(img.shape)
fig, axarr = plt.subplots(ncols=4, figsize=(12, 3))
axarr[0].imshow(np.rollaxis(img.numpy(), 0, 3), cmap="gray")
print(np.argmax(seg, axis=0).shape)
for i in range(seg.shape[0]):
    axarr[i + 1].imshow(seg[i, ...])

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2738095957.py in <cell line: 0>()
----> 1 spl = ds[255]
      2 img, seg = spl["input"], spl["target"]
      3 print(img.shape)
      4 fig, axarr = plt.subplots(ncols=4, figsize=(12, 3))
      5 axarr[0].imshow(np.rollaxis(img.numpy(), 0, 3), cmap="gray")

NameError: name 'ds' is not defined

## === cell 8
from typing import Any, Callable, Dict, Tuple, Type, Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import albumentations as alb

from kaggle_imsegm.transform import FlashAlbumentationsAdapter
from kaggle_imsegm.dataset import TractData

COLOR_MEAN: float = 0.349977
COLOR_STD: float = 0.215829
DEFAULT_TRANSFORM = FlashAlbumentationsAdapter(
    [alb.Resize(224, 224), alb.Normalize(mean=COLOR_MEAN, std=COLOR_STD, max_pixel_value=255)]
)
    
dm = TractData(df_train, DATASET_FOLDER, dataloader_kwargs=dict(batch_size=12, num_workers=3))
dm.setup()
print(len(dm.train_dataloader()))
print(len(dm.val_dataloader()))

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/920283590.py in <cell line: 0>()
      4 import albumentations as alb
      5 
----> 6 from kaggle_imsegm.transform import FlashAlbumentationsAdapter
      7 from kaggle_imsegm.dataset import TractData
      8 

ModuleNotFoundError: No module named 'kaggle_imsegm'

## === cell 9
from kaggle_imsegm.visual import show_tract_datamodule_samples_2d

_= show_tract_datamodule_samples_2d(dm.val_dataloader(), nb=3)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/4163457972.py in <cell line: 0>()
----> 1 from kaggle_imsegm.visual import show_tract_datamodule_samples_2d
      2 
      3 _= show_tract_datamodule_samples_2d(dm.val_dataloader(), nb=3)

ModuleNotFoundError: No module named 'kaggle_imsegm'

## === cell 11
import torch

import flash
from flash.core.data.utils import download_data
from flash.image import SemanticSegmentation, SemanticSegmentationData

print(flash.__version__)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/903983094.py in <cell line: 0>()
      1 import torch
      2 
----> 3 import flash
      4 from flash.core.data.utils import download_data
      5 from flash.image import SemanticSegmentation, SemanticSegmentationData

ModuleNotFoundError: No module named 'flash'

## === cell 13
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union
import albumentations as alb
from flash.core.data.io.input_transform import InputTransform
from flash.image.segmentation.input_transform import prepare_target, remove_extra_dimensions

IMAGE_SIZE = (320, 320)
TRAIN_TRANSFORM = FlashAlbumentationsAdapter([
    alb.Resize(*IMAGE_SIZE),
    alb.VerticalFlip(p=0.5),
    alb.HorizontalFlip(p=0.5),
    alb.RandomRotate90(p=0.5),
    alb.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.08, rotate_limit=10, p=1.),
    alb.GaussNoise(var_limit=(0.001, 0.02), mean=0, per_channel=False, p=1.0),
    alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    alb.Normalize(mean=COLOR_MEAN, std=COLOR_STD, max_pixel_value=255),
])
VAL_TRANSFORM = FlashAlbumentationsAdapter([
    alb.Resize(*IMAGE_SIZE), alb.Normalize(mean=COLOR_MEAN, std=COLOR_STD, max_pixel_value=255)
])

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/392445773.py in <cell line: 0>()
      2 from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union
      3 import albumentations as alb
----> 4 from flash.core.data.io.input_transform import InputTransform
      5 from flash.image.segmentation.input_transform import prepare_target, remove_extra_dimensions
      6 # from kaggle_imsegm.augment import FlashAlbumentationsAdapter

ModuleNotFoundError: No module named 'flash'

## === cell 14
sample_imgs = glob.glob(os.path.join(DATASET_FOLDER, "test", "**", "*.png"), recursive=True)
if not sample_imgs:
    sample_imgs = glob.glob(os.path.join(DATASET_FOLDER, "train", "case123", "**", "*.png"), recursive=True)
print(f"images: {len(sample_imgs)}")
sample_imgs = [p.replace(DATASET_FOLDER + os.path.sep, "") for p in sample_imgs[70:75]]
tab_preds = pd.DataFrame({"image_path": sample_imgs})

## === cell 15
datamodule = TractData(
    df_train,
    dataset_dir=DATASET_FOLDER,
    df_predict=tab_preds,
    train_transform=TRAIN_TRANSFORM,
    input_transform=VAL_TRANSFORM,
    dataloader_kwargs=dict(batch_size=18, num_workers=3),
    val_split=0.01 if WITH_SUBMISSION else 0.1, 
)
datamodule.setup()
LABELS = datamodule.labels
assert len(LABELS) == 3

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2815859739.py in <cell line: 0>()
----> 1 datamodule = TractData(
      2     df_train,
      3     dataset_dir=DATASET_FOLDER,
      4     df_predict=tab_preds,
      5     train_transform=TRAIN_TRANSFORM,

NameError: name 'TractData' is not defined

## === cell 16
_= show_tract_datamodule_samples_2d(datamodule.train_dataloader(), nb=5, skip_empty=True)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/586402407.py in <cell line: 0>()
----> 1 _= show_tract_datamodule_samples_2d(datamodule.train_dataloader(), nb=5, skip_empty=True)

NameError: name 'show_tract_datamodule_samples_2d' is not defined

## === cell 18
from kaggle_imsegm.model import MixedLoss
from kaggle_imsegm.transform import SemanticSegmentationOutputTransform

model = SemanticSegmentation(
    backbone="efficientnet-b3",
    head="unetplusplus",
    pretrained=False,
    optimizer="AdamW",
    learning_rate=7e-3,
    loss_fn=MixedLoss("dice", smooth=0.01),
    lr_scheduler=("cosineannealinglr", {"T_max": 500, "eta_min": 1e-6}),
    num_classes=3,
    multi_label=True,
    output_transform=SemanticSegmentationOutputTransform(),
)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/4054392761.py in <cell line: 0>()
      1 # import segmentation_models_pytorch as smp
----> 2 from kaggle_imsegm.model import MixedLoss
      3 from kaggle_imsegm.transform import SemanticSegmentationOutputTransform
      4 
      5 model = SemanticSegmentation(

ModuleNotFoundError: No module named 'kaggle_imsegm'

## === cell 20
import pytorch_lightning as pl

GPUs = torch.cuda.device_count()

trainer = flash.Trainer(
    max_epochs=1 if WITH_SUBMISSION else 1,
    logger=pl.loggers.CSVLogger(save_dir='logs/'),
    gpus=GPUs,
    precision=16 if GPUs else 32,
    accumulate_grad_batches=24,
    gradient_clip_val=0.01,
    limit_train_batches=1.0 if WITH_SUBMISSION else 0.1,
    limit_val_batches=1.0 if WITH_SUBMISSION else 0.2,
)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3561810402.py in <cell line: 0>()
      3 GPUs = torch.cuda.device_count()
      4 
----> 5 trainer = flash.Trainer(
      6     max_epochs=1 if WITH_SUBMISSION else 1,
      7     logger=pl.loggers.CSVLogger(save_dir='logs/'),

NameError: name 'flash' is not defined

## === cell 21
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

trainer.finetune(model, datamodule=datamodule, strategy="no_freeze")

trainer.save_checkpoint("semantic_segmentation_model.pt")

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4124952518.py in <cell line: 0>()
      3 
      4 # Train the model
----> 5 trainer.finetune(model, datamodule=datamodule, strategy="no_freeze")
      6 
      7 # Save the model!

NameError: name 'trainer' is not defined

## === cell 22
import seaborn as sn

metrics = pd.read_csv(f'{trainer.logger.log_dir}/metrics.csv')
del metrics["step"]
metrics.set_index("epoch", inplace=True)
g = sn.relplot(data=metrics, kind="line")
plt.gcf().set_size_inches(12, 4)
plt.grid()

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3740503413.py in <cell line: 0>()
      1 import seaborn as sn
      2 
----> 3 metrics = pd.read_csv(f'{trainer.logger.log_dir}/metrics.csv')
      4 del metrics["step"]
      5 metrics.set_index("epoch", inplace=True)

NameError: name 'trainer' is not defined

## === cell 24
from itertools import chain

preds = trainer.predict(model, datamodule=datamodule)  #, output="preds"
preds = list(chain(*preds))

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1424929689.py in <cell line: 0>()
      1 from itertools import chain
      2 
----> 3 preds = trainer.predict(model, datamodule=datamodule)  #, output="preds"
      4 preds = list(chain(*preds))

NameError: name 'trainer' is not defined

## === cell 25
fig, axarr = plt.subplots(ncols=4, nrows=len(sample_imgs), figsize=(12, 3 * len(sample_imgs)))
for i, pred in enumerate(preds):
    print(pred.keys())
    img = pred['input']
    print(img.shape, img.min(), img.max())
    axarr[i, 0].imshow(img)
    for j, seg in enumerate(pred['preds']):
        print(seg.shape, seg.min(), seg.max())
        im = axarr[i, j + 1].imshow(seg, vmin=-10, vmax=10)
        plt.colorbar(im, ax=axarr[i, j + 1])

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3856801614.py in <cell line: 0>()
      1 fig, axarr = plt.subplots(ncols=4, nrows=len(sample_imgs), figsize=(12, 3 * len(sample_imgs)))
----> 2 for i, pred in enumerate(preds):
      3     print(pred.keys())
      4     img = pred['input']
      5     print(img.shape, img.min(), img.max())

NameError: name 'preds' is not defined

## === cell 27
model = SemanticSegmentation.load_from_checkpoint(
    "semantic_segmentation_model.pt"
)

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/178382558.py in <cell line: 0>()
----> 1 model = SemanticSegmentation.load_from_checkpoint(
      2     "semantic_segmentation_model.pt"
      3 )

NameError: name 'SemanticSegmentation' is not defined

## === cell 28
sfolder = "test" if WITH_SUBMISSION else "train"
ls_images = glob.glob(os.path.join(DATASET_FOLDER, sfolder, "**", "*.png"), recursive=True)
ls_images = [p.replace(DATASET_FOLDER + os.path.sep, "") for p in ls_images]
case_day = [os.path.dirname(p).split(os.path.sep)[-2] for p in ls_images]
df_pred = pd.DataFrame({'Case_Day': case_day, 'image_path': ls_images})

if not WITH_SUBMISSION:
    df_pred = df_pred[df_pred["Case_Day"].str.startswith("case123_day")]
display(df_pred.head())

## === cell 30
import numpy as np
from itertools import chain
from scipy.ndimage import binary_opening
from skimage.morphology import disk
from kaggle_imsegm.mask import rle_encode

preds = []
for case_day, tab_preds in tqdm(df_pred.groupby("Case_Day")):
    dm = TractData(
        df_train[df_train["id"].str.startswith("case123_day")],  # FAKE
        dataset_dir=DATASET_FOLDER,
        df_predict=tab_preds,
        train_transform=TRAIN_TRANSFORM,
        input_transform=VAL_TRANSFORM,
        dataloader_kwargs=dict(batch_size=10, num_workers=3),
    )
    results = trainer.predict(model, datamodule=dm)
    results = list(chain(*results))
    assert len(tab_preds["image_path"]) == len(results)
    for img_path, spl in zip(tab_preds["image_path"], results):
        name, _ = os.path.splitext(os.path.basename(img_path))
        id_ = f"{case_day}_" + "_".join(name.split("_")[:2])
        for i, mask in enumerate(spl["preds"]):
            mask = (mask >= 0).astype(np.uint8)
            mask = binary_opening(mask, structure=disk(4)).astype(np.uint8)
            rle = rle_encode(mask)[1] if np.sum(mask) > 1 else ""
            preds.append({"id": id_, "class": LABELS[i], "predicted": rle})

assert len(preds) == 3 * len(df_pred)
df_pred = pd.DataFrame(preds)

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/4134297919.py in <cell line: 0>()
      3 from scipy.ndimage import binary_opening
      4 from skimage.morphology import disk
----> 5 from kaggle_imsegm.mask import rle_encode
      6 
      7 preds = []

ModuleNotFoundError: No module named 'kaggle_imsegm'

## === cell 31
display(df_pred[df_pred["predicted"] != ""].head())

## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'predicted'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/739010976.py in <cell line: 0>()
----> 1 display(df_pred[df_pred["predicted"] != ""].head())

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'predicted'

## === cell 33
df_ssub = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
del df_ssub['predicted']
if WITH_SUBMISSION:
    assert len(df_ssub) == len(df_pred)
df_pred = df_ssub.merge(df_pred, on=['id','class'])

df_pred[['id', 'class', 'predicted']].to_csv("submission.csv", index=False)

!head submission.csv

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/2646457000.py in <cell line: 0>()
      2 del df_ssub['predicted']
      3 if WITH_SUBMISSION:
----> 4     assert len(df_ssub) == len(df_pred)
      5 df_pred = df_ssub.merge(df_pred, on=['id','class'])
      6 

AssertionError:
