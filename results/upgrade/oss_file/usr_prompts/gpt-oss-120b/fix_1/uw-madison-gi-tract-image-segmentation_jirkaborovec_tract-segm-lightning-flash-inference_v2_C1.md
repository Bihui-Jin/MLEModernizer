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

0.3812780953400622

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
!pip uninstall -y torchtext
!pip install -q "lightning-flash[image]" "torchmetrics<0.8" --no-index --find-links ../input/demo-flash-semantic-segmentation/frozen_packages
!pip install -q -U timm segmentation-models-pytorch --no-index --find-links ../input/demo-flash-semantic-segmentation/frozen_packages
!pip install -q 'kaggle-imsegm' --no-index --find-links ../input/tract-segm-eda-3d-interactive-viewer/frozen_packages

! pip list | grep torch
! pip list | grep lightning
! nvidia-smi -L

## === cell 3
import os, glob
import pandas as pd
import matplotlib.pyplot as plt

DATASET_FOLDER = "/kaggle/input/uw-madison-gi-tract-image-segmentation"
df_train = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))
display(df_train.head())

LABELS = sorted(df_train["class"].unique())
print(LABELS)

## === cell 5
import torch

import flash
from flash.core.data.utils import download_data
from flash.image import SemanticSegmentation, SemanticSegmentationData

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3853247543.py in <cell line: 0>()
      1 import torch
      2 
----> 3 import flash
      4 from flash.core.data.utils import download_data
      5 from flash.image import SemanticSegmentation, SemanticSegmentationData

ModuleNotFoundError: No module named 'flash'

## === cell 6
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union
import albumentations as alb
from flash.core.data.io.input_transform import InputTransform
from flash.image.segmentation.input_transform import prepare_target, remove_extra_dimensions
from kaggle_imsegm.augment import FlashAlbumentationsAdapter

@dataclass
class SemanticSegmentationInputTransform(InputTransform):

    image_size: Tuple[int, int] = (128, 128)

    def train_per_sample_transform(self) -> Callable:
        return FlashAlbumentationsAdapter([
            alb.Resize(*self.image_size),
            alb.VerticalFlip(p=0.5),
            alb.HorizontalFlip(p=0.5),
            alb.RandomRotate90(p=0.5),
            alb.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.05, rotate_limit=15, p=0.5),
            alb.GaussNoise(var_limit=(0.00, 0.03), mean=0, per_channel=False, p=1.0),
        ])

    def per_sample_transform(self) -> Callable:
        return FlashAlbumentationsAdapter([alb.Resize(*self.image_size)])

    def predict_input_per_sample_transform(self) -> Callable:
        return FlashAlbumentationsAdapter([alb.Resize(*self.image_size)])

    def target_per_batch_transform(self) -> Callable:
        return prepare_target

    def predict_per_batch_transform(self) -> Callable:
        return remove_extra_dimensions

    def serve_per_batch_transform(self) -> Callable:
        return remove_extra_dimensions

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1818660050.py in <cell line: 0>()
      2 from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, Union
      3 import albumentations as alb
----> 4 from flash.core.data.io.input_transform import InputTransform
      5 from flash.image.segmentation.input_transform import prepare_target, remove_extra_dimensions
      6 from kaggle_imsegm.augment import FlashAlbumentationsAdapter

ModuleNotFoundError: No module named 'flash'

## === cell 7
trainer = flash.Trainer(gpus=torch.cuda.device_count())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4145435559.py in <cell line: 0>()
----> 1 trainer = flash.Trainer(gpus=torch.cuda.device_count())

NameError: name 'flash' is not defined

## === cell 9
!ls -l ../input/tract-segm-eda-flash-deeplab-albumentatio/*.pt

model = SemanticSegmentation.load_from_checkpoint(
    "../input/tract-segm-eda-flash-deeplab-albumentatio/semantic_segmentation_model.pt"
)

DATASET_IMAGES = "/kaggle/temp/dataset-flash/images"

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3215425159.py in <cell line: 0>()
      1 get_ipython().system('ls -l ../input/tract-segm-eda-flash-deeplab-albumentatio/*.pt')
      2 
----> 3 model = SemanticSegmentation.load_from_checkpoint(
      4     "../input/tract-segm-eda-flash-deeplab-albumentatio/semantic_segmentation_model.pt"
      5 )

NameError: name 'SemanticSegmentation' is not defined

## === cell 10
df_test = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
sfolder = "test"
display(df_test.head())

if df_test.empty:
    df_test = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))
    df_test = df_test[df_test["id"].str.startswith("case123_day")]
    sfolder = "train"
    display(df_test.head())

os.makedirs(os.path.join(DATASET_IMAGES, sfolder), exist_ok=True)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3810447649.py in <cell line: 0>()
      9     display(df_test.head())
     10 
---> 11 os.makedirs(os.path.join(DATASET_IMAGES, sfolder), exist_ok=True)

NameError: name 'DATASET_IMAGES' is not defined

## === cell 11
from pprint import pprint
from kaggle_imsegm.data import extract_tract_details

pprint(extract_tract_details(df_test['id'].iloc[0], DATASET_FOLDER, folder=sfolder))

df_test[['Case','Day','Slice', 'image', 'image_path', 'height', 'width']] = df_test['id'].apply(
    lambda x: pd.Series(extract_tract_details(x, DATASET_FOLDER, folder=sfolder))
)
df_test["Case_Day"] = [f"case{r['Case']}_day{r['Day']}" for _, r in df_test.iterrows()]
display(df_test.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/501617400.py in <cell line: 0>()
      1 from pprint import pprint
----> 2 from kaggle_imsegm.data import extract_tract_details
      3 
      4 pprint(extract_tract_details(df_test['id'].iloc[0], DATASET_FOLDER, folder=sfolder))
      5 

ModuleNotFoundError: No module named 'kaggle_imsegm'

## === cell 13
import numpy as np
from itertools import chain
from kaggle_imsegm.mask import rle_encode
from kaggle_imsegm.data import preprocess_tract_scan

preds = []
for _, dfg in df_test.groupby("Case_Day"):
    test_imgs = preprocess_tract_scan(
        dfg,
        dir_data=os.path.join(DATASET_FOLDER, sfolder),
        dir_imgs=DATASET_IMAGES,
        dir_segm=None,
        labels=LABELS,
        sfolder=sfolder,
    )
    dm = SemanticSegmentationData.from_files(
        predict_files=test_imgs,
        predict_transform=SemanticSegmentationInputTransform,
        transform_kwargs=dict(image_size=(256, 256)),
        batch_size=10,
    )
    pred = trainer.predict(model, datamodule=dm, output="labels")
    pred = list(chain(*pred))
    for img, seg in zip(test_imgs, pred):
        rle = rle_encode(np.array(seg)) if np.sum(seg) > 1 else {}
        name, _ = os.path.splitext(os.path.basename(img))
        id_ = "_".join(name.split("_")[:4])
        preds += [{"id": id_, "class": lb, "predicted": rle.get(i + 1, "")} for i, lb in enumerate(LABELS)]

df_pred = pd.DataFrame(preds)
display(df_pred[df_pred["predicted"] != ""].head())

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2413759673.py in <cell line: 0>()
      1 import numpy as np
      2 from itertools import chain
----> 3 from kaggle_imsegm.mask import rle_encode
      4 from kaggle_imsegm.data import preprocess_tract_scan
      5 

ModuleNotFoundError: No module named 'kaggle_imsegm'

## === cell 15
df_ssub = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
del df_ssub['predicted']
df_pred = df_ssub.merge(df_pred, on=['id','class'])

df_pred[['id', 'class', 'predicted']].to_csv("submission.csv", index=False)

!head submission.csv

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2215473151.py in <cell line: 0>()
      1 df_ssub = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
      2 del df_ssub['predicted']
----> 3 df_pred = df_ssub.merge(df_pred, on=['id','class'])
      4 
      5 df_pred[['id', 'class', 'predicted']].to_csv("submission.csv", index=False)

NameError: name 'df_pred' is not defined
