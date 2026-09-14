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

0.5833560557254097

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser

## === cell 1
!ls /kaggle/input/modellibs/libs

## === cell 2
!cp -r /kaggle/input/modellibs/libs/efficientnet_pytorch-0.6.3/efficientnet_pytorch-0.6.3/ /tmp/efficientnet_pytorch-0.6.3

## === cell 3
!cd /tmp/efficientnet_pytorch-0.6.3 && python setup.py -q install

## === cell 4
!cp -r /tmp/efficientnet_pytorch-0.6.3/efficientnet_pytorch /opt/conda/lib/python3.7/site-packages/

## === cell 5
!cp -R /kaggle/input/modellibs/libs/pytorch-image-models-0.4.12/pytorch-image-models-0.4.12/ /tmp/pytorch-image-models-0.4.12

## === cell 6
!cd /tmp/pytorch-image-models-0.4.12/ && python setup.py -q install

## === cell 7
!cp -r /tmp/pytorch-image-models-0.4.12/timm /opt/conda/lib/python3.7/site-packages/

## === cell 8
!cp -R /kaggle/input/modellibs/libs/pretrained-models.pytorch/ /tmp/pretrained-models.pytorch

## === cell 9
!cd /tmp/pretrained-models.pytorch && python setup.py -q install

## === cell 10
!cp -r /tmp/pretrained-models.pytorch/pretrainedmodels /opt/conda/lib/python3.7/site-packages/

## === cell 11
!cd /kaggle/input/seg-model-pytorch && pip install -q segmentation_models_pytorch-0.2.0-py3-none-any.whl

## === cell 12

import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import os
import torch
from glob import glob
from pandarallel import pandarallel
from tqdm.notebook import tqdm
from torch.utils.data import Dataset,DataLoader
from typing import List,Tuple
import numpy as np
from torchvision import transforms as T
from torchvision.utils import make_grid
import cv2
import copy
from matplotlib.patches import Rectangle
from sklearn.model_selection import train_test_split
from collections import defaultdict
import segmentation_models_pytorch as smp
from torch.cuda import amp
import torch.nn as nn
import torch.optim as optim
import time
import gc
from torch.optim import lr_scheduler
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings("ignore")

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
tqdm.pandas()

pandarallel.initialize(progress_bar=True)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1207922193.py in <cell line: 0>()
      8 import torch
      9 from glob import glob
---> 10 from pandarallel import pandarallel
     11 from tqdm.notebook import tqdm
     12 from torch.utils.data import Dataset,DataLoader

ModuleNotFoundError: No module named 'pandarallel'

## === cell 13
class CFG:
    seed = 42
    debug = False
    model_name = "UNET"
    encoder_name = "resnet50"
    encoder_weights = "imagenet"
    train_batch_size = 32
    val_batch_size = 32
    img_size = (224,224)
    scheduler = 'CosineAnnealingLR'
    epochs = 22
    lr = 2e-3
    min_lr = 1e-6
    weight_decay = 1e-6
    T_max = int(30000/train_batch_size*epochs)+50  #max iterations for scheduler
    num_classes = 3
    val_split_percentage = 0.2
    n_accumulate  = max(1, 32//train_batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    thr = 0.45

## === cell 14
TRAIN_METADATA_FILE = "../input/uw-madison-gi-tract-image-segmentation/train.csv"
TRAIN_DIR = "../input/uw-madison-gi-tract-image-segmentation/train/"
TRAIN_METADATA_FILE
BASE_PATH  = '../input/uw-madison-gi-tract-image-segmentation'
CKPT_DIR = '../input/resnet50best'
SAMPLE_SUBMISSION_CSV_PATH = '../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv'
TRAIN_FILE_CSV_PATH = TRAIN_METADATA_FILE

## === cell 15
def get_metadata(df):

    df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].split("case")[1]))
    df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].split("day")[1]))
    df["slice"] = df["id"].apply(lambda x: int(x.split("_")[-1]))
    return df

def path2info(row):
    path = row['image_path']
    data = path.split('/')
    slice_ = int(data[-1].split('_')[1])
    case = int(data[-3].split('_')[0].replace('case',''))
    day = int(data[-3].split('_')[1].replace('day',''))
    width = int(data[-1].split('_')[2])
    height = int(data[-1].split('_')[3])
    row['height'] = height
    row['width'] = width
    row['case'] = case
    row['day'] = day
    row['slice'] = slice_
    return row

## === cell 16
def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.expand_dims(img, axis=2)
    img = img.astype('float32') # original is uint16
    mx = np.max(img)
    if mx:
        img/=mx # scale image to [0, 1]
    return img

def load_image(path):
    return Image.open(path).convert("RGB")

def load_msk(path):
    msk = np.load(path)
    msk = msk.astype('float32')
    msk/=255.0
    return msk

def show_img(img, mask=None):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    plt.imshow(img, cmap='bone')
    
    if mask is not None:
        plt.imshow(mask, alpha=0.5)
        handles = [Rectangle((0,0),1,1, color=_c) for _c in [(0.667,0.0,0.0), (0.0,0.667,0.0), (0.0,0.0,0.667)]]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.legend(handles,labels)
    plt.axis('off')

## === cell 17
sub_df = pd.read_csv(SAMPLE_SUBMISSION_CSV_PATH)
if not len(sub_df):
    debug = True
    sub_df = pd.read_csv(TRAIN_FILE_CSV_PATH)[:1000*3]
    sub_df = sub_df.drop(columns=['class','segmentation']).drop_duplicates()
else:
    debug = False
    sub_df = sub_df.drop(columns=['class','predicted']).drop_duplicates()
sub_df = get_metadata(sub_df)

## === cell 18
if debug:
    paths = glob(BASE_PATH+'/train/**/*png',recursive=True)
else:
    paths = glob(BASE_PATH+'/test/**/*png',recursive=True)
path_df = pd.DataFrame(paths, columns=['image_path'])
path_df = path_df.progress_apply(path2info, axis=1)
path_df.head()

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3710759327.py in <cell line: 0>()
      4     paths = glob(BASE_PATH+'/test/**/*png',recursive=True)
      5 path_df = pd.DataFrame(paths, columns=['image_path'])
----> 6 path_df = path_df.progress_apply(path2info, axis=1)
      7 path_df.head()

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'DataFrame' object has no attribute 'progress_apply'

## === cell 19
sub_df.head()

## === cell 20
test_df = sub_df.merge(path_df, on=['case','day','slice'], how='left')
test_df.head()

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1234385406.py in <cell line: 0>()
----> 1 test_df = sub_df.merge(path_df, on=['case','day','slice'], how='left')
      2 test_df.head()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in merge(self, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
  10830         from pandas.core.reshape.merge import merge
  10831 
> 10832         return merge(
  10833             self,
  10834             right,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in merge(left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, copy, indicator, validate)
    168         )
    169     else:
--> 170         op = _MergeOperation(
    171             left_df,
    172             right_df,

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in __init__(self, left, right, how, on, left_on, right_on, left_index, right_index, sort, suffixes, indicator, validate)
    792             left_drop,
    793             right_drop,
--> 794         ) = self._get_merge_keys()
    795 
    796         if left_drop:

/usr/local/lib/python3.11/dist-packages/pandas/core/reshape/merge.py in _get_merge_keys(self)
   1295                         rk = cast(Hashable, rk)
   1296                         if rk is not None:
-> 1297                             right_keys.append(right._get_label_or_level_values(rk))
   1298                         else:
   1299                             # work-around for merge_asof(right_index=True)

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _get_label_or_level_values(self, key, axis)
   1909             values = self.axes[axis].get_level_values(key)._values
   1910         else:
-> 1911             raise KeyError(key)
   1912 
   1913         # Check for duplicates

KeyError: 'case'

## === cell 21
class TestDataset(Dataset):
    def __init__(self,df,transforms=None,label=None):
        self.df         = df
        self.label      = label
        self.img_paths  = df['image_path'].tolist()
        self.ids        = df['id'].tolist()
        if 'msk_path' in df.columns:
            self.msk_paths  = df['mask_path'].tolist()
        else:
            self.msk_paths = None
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        img_path  = self.img_paths[index]
        id_       = self.ids[index]
        img = []
        img = load_image(img_path)
        img = np.array(img)
        h, w = img.shape[:2]
        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img  = data['image']
                msk  = data['mask']
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img  = data['image']
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img), id_, h, w

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3880136694.py in <cell line: 0>()
----> 1 class TestDataset(Dataset):
      2     def __init__(self,df,transforms=None,label=None):
      3         self.df         = df
      4         self.label      = label
      5         self.img_paths  = df['image_path'].tolist()

NameError: name 'Dataset' is not defined

## === cell 22
test_transforms = {
    "test": A.Compose([
        A.CenterCrop(*CFG.img_size),
        ], p=1.0)
}

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4237695192.py in <cell line: 0>()
      1 test_transforms = {
----> 2     "test": A.Compose([
      3         A.CenterCrop(*CFG.img_size),
      4         ], p=1.0)
      5 }

NameError: name 'A' is not defined

## === cell 23
def build_model():
    model = smp.Unet(
        encoder_name=CFG.encoder_name, 
        encoder_weights=None,
    in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=CFG.num_classes,                      # model output channels (number of classes in your dataset)
    )
    return model

def load_model(path):
    model = build_model()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

## === cell 24

def mask2rle(msk, thr=0.5):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    msk    = np.array(msk)
    pixels = msk.flatten()
    pad    = np.array([0])
    pixels = np.concatenate([pad, pixels, pad])
    runs   = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def masks2rles(msks, ids, heights, widths):
    pred_strings = []; pred_ids = []; pred_classes = [];
    for idx in range(msks.shape[0]):
        height = heights[idx].item()
        width = widths[idx].item()
        left = (width - msks[idx].shape[0])//2
        right = left
        top = (height - msks[idx].shape[1])//2
        bottom = top
        msk = cv2.copyMakeBorder(msks[idx], top, bottom, left, right, cv2.BORDER_CONSTANT, 0)
        rle = [None]*3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(msk[...,midx])
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]]*len(rle))
        pred_classes.extend(['large_bowel', 'small_bowel', 'stomach'])
    return pred_strings, pred_ids, pred_classes

## === cell 25

@torch.no_grad()
def infer(model_paths, test_loader, num_log=1, thr=CFG.thr):
    msks = []; imgs = [];
    pred_strings = []; pred_ids = []; pred_classes = [];
    for idx, (img, ids, heights, widths) in enumerate(tqdm(test_loader, total=len(test_loader), desc='Infer ')):
        img = img.to(CFG.device, dtype=torch.float) # .squeeze(0)
        size = img.size()
        msk = []
        msk = torch.zeros((size[0], 3, size[2], size[3]), device=CFG.device, dtype=torch.float32)
        for path in model_paths:
            model = load_model(path)
            model = model.to(CFG.device)
            out   = model(img) # .squeeze(0) # removing batch axis
            out   = nn.Sigmoid()(out) # removing channel axis
            msk+=out/len(model_paths)
        msk = (msk.permute((0,2,3,1))>thr).to(torch.uint8).cpu().detach().numpy() # shape: (n, h, w, c)
        result = masks2rles(msk, ids, heights, widths)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])
        if idx<num_log:
            img = img.permute((0,2,3,1)).cpu().detach().numpy()
            imgs.append(img[:10])
            msks.append(msk[:10])
        del img, msk, out, model, result
        gc.collect()
        torch.cuda.empty_cache()
    return pred_strings, pred_ids, pred_classes, imgs, msks

## === cell 26
test_dataset = TestDataset(test_df, transforms=test_transforms['test'])
test_loader  = DataLoader(test_dataset, batch_size=CFG.val_batch_size, 
                          num_workers=4, shuffle=False, pin_memory=False)
model_paths  = glob(f'{CKPT_DIR}/best_epoch*.bin')
pred_strings, pred_ids, pred_classes, imgs, msks = infer(model_paths, test_loader)

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2867135762.py in <cell line: 0>()
----> 1 test_dataset = TestDataset(test_df, transforms=test_transforms['test'])
      2 test_loader  = DataLoader(test_dataset, batch_size=CFG.val_batch_size, 
      3                           num_workers=4, shuffle=False, pin_memory=False)
      4 model_paths  = glob(f'{CKPT_DIR}/best_epoch*.bin')
      5 pred_strings, pred_ids, pred_classes, imgs, msks = infer(model_paths, test_loader)

NameError: name 'TestDataset' is not defined

## === cell 27
for img, msk in zip(imgs[0][:5], msks[0][:5]):
    plt.figure(figsize=(12, 7))
    plt.subplot(1, 3, 1); plt.imshow(img, cmap='bone');
    plt.axis('OFF'); plt.title('image')
    plt.subplot(1, 3, 2); plt.imshow(msk*255); plt.axis('OFF'); plt.title('mask')
    plt.subplot(1, 3, 3); plt.imshow(img, cmap='bone'); plt.imshow(msk*255, alpha=0.4);
    plt.axis('OFF'); plt.title('overlay')
    plt.tight_layout()
    plt.show()

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/782298255.py in <cell line: 0>()
----> 1 for img, msk in zip(imgs[0][:5], msks[0][:5]):
      2     plt.figure(figsize=(12, 7))
      3     plt.subplot(1, 3, 1); plt.imshow(img, cmap='bone');
      4     plt.axis('OFF'); plt.title('image')
      5     plt.subplot(1, 3, 2); plt.imshow(msk*255); plt.axis('OFF'); plt.title('mask')

NameError: name 'imgs' is not defined

## === cell 28
del imgs, msks
gc.collect()

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2676652455.py in <cell line: 0>()
----> 1 del imgs, msks
      2 gc.collect()

NameError: name 'imgs' is not defined

## === cell 29
pred_df = pd.DataFrame({
    "id":pred_ids,
    "class":pred_classes,
    "predicted":pred_strings
})
if not debug:
    sub_df = pd.read_csv(BASE_PATH+'/sample_submission.csv')
    del sub_df['predicted']
else:
    sub_df = pd.read_csv(BASE_PATH+'/train.csv')[:1000*3]
    del sub_df['segmentation']
    
sub_df = sub_df.merge(pred_df, on=['id','class'])
sub_df.to_csv('submission.csv',index=False)
print(sub_df.head(5))

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3963958707.py in <cell line: 0>()
      1 pred_df = pd.DataFrame({
----> 2     "id":pred_ids,
      3     "class":pred_classes,
      4     "predicted":pred_strings
      5 })

NameError: name 'pred_ids' is not defined
