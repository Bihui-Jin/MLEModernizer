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

3.13

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
seaborn==0.12.2
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

0.7516228812032423

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
import numpy as np
import pandas as pd

import os
import random
import re

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import ListedColormap

import seaborn as sns

from glob import glob

from tqdm.notebook import tqdm
tqdm.pandas()

from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

import albumentations as A

import segmentation_models_pytorch as smp


from monai.metrics.utils import get_mask_edges, get_surface_distance

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3932383276.py in <cell line: 0>()
     28 import albumentations as A
     29 
---> 30 import segmentation_models_pytorch as smp
     31 
     32 # from torchmetrics.segmentation import DiceScore, HausdorffDistance

ModuleNotFoundError: No module named 'segmentation_models_pytorch'

## === cell 4
print(f"Number of available CPUs: {os.cpu_count()}")
print(f"Number of available GPUs: {torch.cuda.device_count()}")

## === cell 6
DIR_PATH = "/kaggle/input/uw-madison-gi-tract-image-segmentation/"

pd.set_option('display.max_colwidth', 400) 

CMAP1 = ListedColormap([[0, 0, 0, 0], [1, 0, 0, 1]])  # black transparent, red opaque
CMAP2 = ListedColormap([[0, 0, 0, 0], [0, 1, 0, 1]])  # black transparent, green opaque
CMAP3 = ListedColormap([[0, 0, 0, 0], [0, 0, 1, 1]])  # black transparent, blue opaque

RANDOM_SEED = 0

IMAGE_NORMALIZE_MEAN = (0.485, 0.456, 0.406)
IMAGE_NORMALIZE_SD = (0.229, 0.224, 0.225)

IMAGE_RESIZE = [288, 288]

BATCH_SIZE_TRAIN = 32
BATCH_SIZE_VALID = BATCH_SIZE_TRAIN*2
BATCH_SIZE_TEST = BATCH_SIZE_TRAIN*2

DATA_LOADER_NUM_WORKERS = 4

NUM_CLASSES = 3
CLASS_NAMES = ['large_bowel', 'small_bowel', 'stomach']

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 5

MODEL_PARAMS_FILE_NAME = "GIT-Seg-efficientnet-b1-subset.pth"
MODEL_PARAMS_LOAD_FILE_PATH = "/kaggle/input/git-seg/pytorch/default/1/GIT-Seg-efficientnet-b1-subset.pth"

TRAIN_VALID_SPLIT = False
TEST_PREDICT = True

SAVE_TRAIN_VALID_MODEL = False
LOAD_MODEL_FOR_TEST_PREDICT = True

## === cell 7
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)

## === cell 9
data = pd.read_csv(DIR_PATH + "train.csv")
data.head()

## === cell 10
data_nonnaseg = data.loc[data.segmentation.notna(), :]
data_nonnaseg.head()

## === cell 11
data[['case', 'day', 'slice']] = data['id'].str.extract(r'case(\d+)_day(\d+)_slice_(\d+)')
data

## === cell 12

def get_path_df(train = True):
    if train:
        paths = glob(DIR_PATH + 'train/*/*/*/*')
    else:
        paths = glob(DIR_PATH + 'test/*/*/*/*')
    path_df = pd.DataFrame(paths, columns=['image_path'])
    path_df[['case', 'day', 'slice', 
             'slice_w', 'slice_h', 
             'px_w', 'px_h']] = \
            path_df.image_path.str.extract(r'.*/case(\d+)_day(\d+)/scans/slice_(\d+)_(\d+)_(\d+)_(\d+\.\d+)_(\d+\.\d+)\.png')
    
    return path_df

path_df = get_path_df()

## === cell 13
data.info()

## === cell 14
path_df.info()

## === cell 16
data = data.merge(path_df, on = ['case', 'day', 'slice'])
data

## === cell 17
data.info()

## === cell 18
data.px_w.unique(), data.px_h.unique()

## === cell 19
data.case.unique(), data.day.unique(), data.slice.unique(), data.slice_w.unique(), data.slice_h.unique()

## === cell 20
int_cols = ['case', 'day', 'slice', 'slice_w', 'slice_h']
data[int_cols] = data[int_cols].astype(np.uint32)

float_cols = ['px_w', 'px_h']
data[float_cols] = data[float_cols].astype(np.float32)

data.info()

## === cell 22
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formatted (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


def rle_encode(img):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = img.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

## === cell 24
def get_mask(id_, data):
    data_subset_id = data.loc[data['id']==id_]
    slice_dim = data_subset_id[['slice_h', 'slice_w']].iloc[0]
    shape = (slice_dim.slice_h, slice_dim.slice_w, 3)
    mask = np.zeros(shape, dtype=np.uint8)
    for i, class_ in enumerate(CLASS_NAMES):
        data_subset_class = data_subset_id[data_subset_id['class']==class_]
        rle = data_subset_class.segmentation.squeeze()
        if not pd.isna(rle):
            mask[..., i] = rle_decode(rle, shape[:2])
    return mask

## === cell 25
full_image_file_path = DIR_PATH + "train/case123/case123_day20/scans/slice_0065_266_266_1.50_1.50.png"

img = cv2.imread(full_image_file_path, cv2.IMREAD_UNCHANGED)

print(img.shape)
print(img)

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray')
plt.title('Gray')
plt.axis('off')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(img, cmap='bone')
plt.title('Bone')
plt.axis('off')
plt.colorbar()


plt.tight_layout()
plt.show()

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1915369017.py in <cell line: 0>()
      4 # default imread mode is IMREAD_COLOR which expects 8-bit 3 channel image, our input image is 16-bit grayscale which requires IMREAD_UNCHANGED
      5 
----> 6 print(img.shape)
      7 print(img)
      8 

AttributeError: 'NoneType' object has no attribute 'shape'

## === cell 26
img = cv2.imread(full_image_file_path, cv2.IMREAD_UNCHANGED).astype('float32')

img_norm = img
mx = np.max(img)
if mx > 0:
    img_norm /= mx

print(img_norm)
print(img_norm.shape)
print(max([max(r) for r in img_norm]))

img_norm = (img_norm*255).astype(np.uint8)
print(img_norm)
print(img_norm.shape)
print(max([max(r) for r in img_norm]))

clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2,2))
clahe3 = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))

res1 = clahe1.apply(img_norm)
res2 = clahe2.apply(img_norm)
res3 = clahe3.apply(img_norm)

print(res1)
print(res1.shape)
print(max([max(r) for r in res1]))

print(res2)
print(res2.shape)
print(max([max(r) for r in res2]))

plt.figure(figsize=(20, 4))
for i, (title, im) in enumerate(zip(['Original', 'Normalized', 'CLAHE clip=2 grid=8x8', 'CLAHE clip=2 grid=2x2', 'CLAHE clip=1 grid=2x2'], [img, img_norm, res1, res2, res3])):
    plt.subplot(1,5,i+1)
    plt.imshow(im, cmap='bone')
    plt.title(title)
    plt.colorbar()
    plt.axis('off')
plt.tight_layout()
plt.show()

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2209065833.py in <cell line: 0>()
----> 1 img = cv2.imread(full_image_file_path, cv2.IMREAD_UNCHANGED).astype('float32')
      2 
      3 img_norm = img
      4 mx = np.max(img)
      5 if mx > 0:

AttributeError: 'NoneType' object has no attribute 'astype'

## === cell 29
def load_image(id_, data):
    data_subset = data.loc[data.id == id_]
    img = cv2.imread(data_subset.image_path.iloc[0], cv2.IMREAD_UNCHANGED).astype('float32')  # convert from original 16-bit
    mx = np.max(img)
    if mx > 0:
        img /= mx
    return img

## === cell 30
def display_image(id_, data, pred_mask=None, apply_CLAHE=False,
                  show_orig_img=True, show_true_mask=True, show_pred_mask=False):
    
    img = load_image(id_, data)
    img = (img * 255).astype(np.uint8) # 0-255 range required for CLAHE. 
    if apply_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
        img = clahe.apply(img)
    
    mask = get_mask(id_, data)

    plt.figure(figsize=(9, 3))
    
    i = 1
    if show_orig_img:
        plt.subplot(1, 3, i)
        i += 1
        plt.imshow(img, cmap='bone')
        plt.title(f'{id_} image')
        plt.axis('off')

    if show_true_mask:
        plt.subplot(1, 3, i)
        i += 1
        print(f'just before imshow {id_} min : {np.min(img)} max : {np.max(img)}')
        print(img.shape)
        plt.imshow(img, cmap='bone')
        plt.title('Image with true mask')
        plt.imshow(mask[..., 0], cmap=CMAP1)
        plt.imshow(mask[..., 1], cmap=CMAP2)
        plt.imshow(mask[..., 2], cmap=CMAP3)
        
        handles = [
            Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP3(1.0))
        ]
        labels = ['Large Bowel', 'Small Bowel', 'Stomach']
        plt.axis('off')
        plt.legend(handles, labels, bbox_to_anchor=(1.0, -0.4), loc='lower right', borderaxespad=0.)
    
    if show_pred_mask and pred_mask is not None:
        plt.subplot(1, 3, i)
        plt.imshow(img, cmap='bone')
        plt.title('Image with predicted mask')
        plt.imshow(pred_mask[..., 0], cmap=CMAP1)
        plt.imshow(pred_mask[..., 1], cmap=CMAP2)
        plt.imshow(pred_mask[..., 2], cmap=CMAP3)
        
        handles = [
            Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
            Rectangle((0, 0), 1, 1, color=CMAP3(1.0))
        ]
        labels = ["Large Bowel", "Small Bowel", "Stomach"]
        plt.axis('off')
        plt.legend(handles, labels, bbox_to_anchor=(1.0, -0.4), loc='lower right', borderaxespad=0.)
    
    
    plt.tight_layout()
    plt.show()  

## === cell 31
display_image('case131_day0_slice_0066', data)

## === cell 32
display_image('case131_day0_slice_0066', data, apply_CLAHE=True)

## === cell 34
display_image('case123_day20_slice_0065', data, apply_CLAHE=True)

## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/1922044048.py in <cell line: 0>()
      1 # example image with only stomach segment
----> 2 display_image('case123_day20_slice_0065', data, apply_CLAHE=True)

/tmp/ipykernel_11/1949913104.py in display_image(id_, data, pred_mask, apply_CLAHE, show_orig_img, show_true_mask, show_pred_mask)
      2                   show_orig_img=True, show_true_mask=True, show_pred_mask=False):
      3 
----> 4     img = load_image(id_, data)
      5     #print(f'load_image result {id_} min : {np.min(img)} max : {np.max(img)}')
      6     img = (img * 255).astype(np.uint8) # 0-255 range required for CLAHE.

/tmp/ipykernel_11/843812603.py in load_image(id_, data)
      1 def load_image(id_, data):
      2     data_subset = data.loc[data.id == id_]
----> 3     img = cv2.imread(data_subset.image_path.iloc[0], cv2.IMREAD_UNCHANGED).astype('float32')  # convert from original 16-bit
      4     #print(f'Raw image {id_} min : {np.min(img)} max : {np.max(img)}')
      5     mx = np.max(img)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1750 
   1751             # validate the location
-> 1752             self._validate_integer(key, axis)
   1753 
   1754             return self.obj._ixs(key, axis=axis)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _validate_integer(self, key, axis)
   1683         len_axis = len(self.obj._get_axis(axis))
   1684         if key >= len_axis or key < -len_axis:
-> 1685             raise IndexError("single positional indexer is out-of-bounds")
   1686 
   1687     # -------------------------------------------------------------------

IndexError: single positional indexer is out-of-bounds

## === cell 35
display_image('case123_day20_slice_0001', data, apply_CLAHE=True)

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/4288798617.py in <cell line: 0>()
      1 # example image without any segment
----> 2 display_image('case123_day20_slice_0001', data, apply_CLAHE=True)

/tmp/ipykernel_11/1949913104.py in display_image(id_, data, pred_mask, apply_CLAHE, show_orig_img, show_true_mask, show_pred_mask)
      2                   show_orig_img=True, show_true_mask=True, show_pred_mask=False):
      3 
----> 4     img = load_image(id_, data)
      5     #print(f'load_image result {id_} min : {np.min(img)} max : {np.max(img)}')
      6     img = (img * 255).astype(np.uint8) # 0-255 range required for CLAHE.

/tmp/ipykernel_11/843812603.py in load_image(id_, data)
      1 def load_image(id_, data):
      2     data_subset = data.loc[data.id == id_]
----> 3     img = cv2.imread(data_subset.image_path.iloc[0], cv2.IMREAD_UNCHANGED).astype('float32')  # convert from original 16-bit
      4     #print(f'Raw image {id_} min : {np.min(img)} max : {np.max(img)}')
      5     mx = np.max(img)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1750 
   1751             # validate the location
-> 1752             self._validate_integer(key, axis)
   1753 
   1754             return self.obj._ixs(key, axis=axis)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _validate_integer(self, key, axis)
   1683         len_axis = len(self.obj._get_axis(axis))
   1684         if key >= len_axis or key < -len_axis:
-> 1685             raise IndexError("single positional indexer is out-of-bounds")
   1686 
   1687     # -------------------------------------------------------------------

IndexError: single positional indexer is out-of-bounds

## === cell 38
def display_multiple_slices(id_array, data, apply_CLAHE=False,
                            show_pred_mask=False, pred_mask_array=None):

    '''
    id_array : an array of ids like case123_day20_slice_0001
    data     : dataframe containing all the image metadata 
    apply_CLAHE : whether or not to apply CLAHE
    show_pred_mask : if this parameter is False, then true mask will be shown
                     if it is True, masks from pred_mask_array will be shown
    pred_mask_array : array of prediction masks
    '''

    l = len(id_array)
    rows = np.ceil(l/5).astype(int)
    max_cols = 5
    data_subset = data.loc[data.id.isin(id_array), ]

    plt.figure(figsize=(max_cols*3, rows*3))

    for i in range(l):
        
        img = cv2.imread(data_subset.image_path.iloc[3*i], cv2.IMREAD_UNCHANGED).astype('float32')  # 3*i because every 3 image paths are same 
        mx = np.max(img)
        if mx > 0:
            img /= mx

        if apply_CLAHE:
            clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
            img = (img * 255).astype(np.uint8)
            img = clahe3.apply(img)

        id_ = data_subset.id.iloc[3*i]

        if show_pred_mask and pred_mask_array is not None:
            mask = pred_mask_array[3*i]
        else:
            mask = get_mask(id_, data)

        plt.subplot(rows, max_cols, i+1)
        plt.imshow(img, cmap='bone')
        plt.title(id_)
        plt.imshow(mask[..., 0], cmap=CMAP1)
        plt.imshow(mask[..., 1], cmap=CMAP2)
        plt.imshow(mask[..., 2], cmap=CMAP3)
        plt.axis('off')

        if i == 0:
            handles = [
                Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
                Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
                Rectangle((0, 0), 1, 1, color=CMAP3(1.0))
            ]
            labels = ['Large Bowel', 'Small Bowel', 'Stomach']
        
            plt.legend(handles, labels, bbox_to_anchor=(0.0, 1.5), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.show()  

## === cell 39
display_multiple_slices(data.query("case == 123 and day == 20 and slice >= 63 and slice <= 70").id.unique(), 
                        data, apply_CLAHE=True)

## === cell 40
display_multiple_slices(data.query("case == 131 and day == 0 and slice > 55 and slice <= 70").id.unique(), 
                        data, apply_CLAHE=True)

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4158942996.py in <cell line: 0>()
      1 # using the below data to visualize change in segmentation mask across different slices - for a case where all masks are present
      2 # data.query("case == 131 and day == 0 and slice > 60 and slice <= 70")
----> 3 display_multiple_slices(data.query("case == 131 and day == 0 and slice > 55 and slice <= 70").id.unique(), 
      4                         data, apply_CLAHE=True)

/tmp/ipykernel_11/2720533657.py in display_multiple_slices(id_array, data, apply_CLAHE, show_pred_mask, pred_mask_array)
     29             clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
     30             img = (img * 255).astype(np.uint8)
---> 31             img = clahe3.apply(img)
     32 
     33         id_ = data_subset.id.iloc[3*i]

NameError: name 'clahe3' is not defined

## === cell 42
data.loc[data.segmentation.isna(), :]

## === cell 43
data.isna().sum()

## === cell 45
print(f'Num cases : {len(data.case.unique())} \
        Num unique days : {len(data.day.unique())}   \
        Num unique slices : {len(data.slice.unique())}')

## === cell 46
count_df = data[['id', 'slice_w', 'slice_h']].drop_duplicates()[['slice_w', 'slice_h']].value_counts().reset_index(name='count')
count_df['percent'] = count_df['count']*100 / sum(count_df['count'])
print(sum(count_df['count']))
count_df

## === cell 47
count_df = data[['id', 'px_w', 'px_h']].drop_duplicates()[['px_w', 'px_h']].value_counts().reset_index(name='count')
count_df['percent'] = count_df['count']*100 / sum(count_df['count'])
print(sum(count_df['count']))
count_df

## === cell 49
day_dist = data[['case', 'day']].drop_duplicates()['case'].value_counts().reset_index(name='num_days')

display(day_dist)

sns.histplot(data=day_dist, x='num_days', bins=range(1, day_dist['num_days'].max() + 1), discrete=True)
plt.xlabel('Number of Days per Case')
plt.ylabel('Number of Cases')
plt.title('Distribution of Days per Case')
plt.show()

## === cell 51
slice_dist = data[['case', 'day', 'slice']].drop_duplicates()[['case', 'day']].value_counts().reset_index(name='num_slices')
display(slice_dist)

sns.histplot(data=slice_dist, x='num_slices', bins=range(1, slice_dist['num_slices'].max() + 1), discrete=True)
plt.xlabel('Number of slices per case-days')
plt.ylabel('Number of specific case-days')
plt.title('Distribution of slices per case-day')
plt.show()

display(slice_dist.num_slices.value_counts())

## === cell 53
slice_dist.loc[slice_dist.num_slices == 80, :]

## === cell 54
case_day_slice_df = data[['case', 'day', 'slice', 'slice_w', 'slice_h']].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=['case', 'day']).query("(slice_w_x != slice_w_y) | (slice_h_x != slice_h_y)")

## === cell 56
case_day_slice_df = data[['case', 'day', 'slice', 'slice_w', 'slice_h']].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=['case']).query("(slice_w_x != slice_w_y) | (slice_h_x != slice_h_y)")

## === cell 58
case_day_slice_df = data[['case', 'day', 'slice', 'px_w', 'px_h']].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=['case', 'day']).query("(px_w_x != px_w_y) | (px_h_x != px_h_y)")

## === cell 59
case_day_slice_df = data[['case', 'day', 'slice', 'px_w', 'px_h']].drop_duplicates()
case_day_slice_df.merge(case_day_slice_df, on=['case']).query("(px_w_x != px_w_y) | (px_h_x != px_h_y)")

## === cell 62
num_missing_seg_masks = data.segmentation.isna().sum() 
print(f'Missing Seg Mask \n count = {num_missing_seg_masks}\n percentage = {num_missing_seg_masks/len(data)*100}')

## === cell 64
data['class'].value_counts()

## === cell 66
na_counts = (
    data.groupby('class')['segmentation']
    .apply(lambda s: s.isna().sum())
    .reset_index(name='count')
)
na_counts['percent'] = 100 * na_counts['count'] / data.groupby('class')['segmentation'].size().values

display(na_counts)

sns.set_style("whitegrid") 
ax = sns.barplot(data=na_counts, x='class', y='percent', palette=[CMAP1(1.0), CMAP2(1.0), CMAP3(1.0)])

for i, row in na_counts.iterrows():
    ax.text(i, row['percent'] + 1,  # position just above the bar
            f"{row['percent']:.2f}% ({row['count']})",
            ha='center', va='bottom', fontsize=10)

plt.ylabel('Percentage')
plt.xlabel('Segmentation Class')
plt.title('Missing Segmentation Masks')
plt.yticks(range(0, 105, 10))
plt.show()

## === cell 67
case_day_seg_missing = (
     data[['case', 'day', 'class', 'segmentation']]
     .groupby(['case', 'day', 'class'])['segmentation']
     .apply(lambda s: s.isna().sum())
     .reset_index(name='count').sort_values(by='count', ascending=False)
)
display(case_day_seg_missing)





sns.boxplot(data=case_day_seg_missing, x='class', y='count', palette=[CMAP1(1.0), CMAP2(1.0), CMAP3(1.0)])
sns.stripplot(data=case_day_seg_missing, x='class', y='count', color='black', size=3, jitter=True, alpha=0.4)
plt.ylabel('Missing Mask Count')
plt.xlabel('Segmentation Class')
plt.title('Distribution of Missing Masks per Class (by Case-Day)')
plt.show()

## === cell 70
display_image('case43_day26_slice_0057', data, apply_CLAHE=True)
display_image('case43_day26_slice_0058', data, apply_CLAHE=True)
display_image('case43_day26_slice_0121', data, apply_CLAHE=True)
display_image('case43_day26_slice_0122', data, apply_CLAHE=True)

## --- ERROR in cell 70, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/1616463412.py in <cell line: 0>()
      1 #visualizing the original image and image with true mask for border slices where segmentation classes just start appearing/disapperaing
----> 2 display_image('case43_day26_slice_0057', data, apply_CLAHE=True)
      3 display_image('case43_day26_slice_0058', data, apply_CLAHE=True)
      4 display_image('case43_day26_slice_0121', data, apply_CLAHE=True)
      5 display_image('case43_day26_slice_0122', data, apply_CLAHE=True)

/tmp/ipykernel_11/1949913104.py in display_image(id_, data, pred_mask, apply_CLAHE, show_orig_img, show_true_mask, show_pred_mask)
      2                   show_orig_img=True, show_true_mask=True, show_pred_mask=False):
      3 
----> 4     img = load_image(id_, data)
      5     #print(f'load_image result {id_} min : {np.min(img)} max : {np.max(img)}')
      6     img = (img * 255).astype(np.uint8) # 0-255 range required for CLAHE.

/tmp/ipykernel_11/843812603.py in load_image(id_, data)
      1 def load_image(id_, data):
      2     data_subset = data.loc[data.id == id_]
----> 3     img = cv2.imread(data_subset.image_path.iloc[0], cv2.IMREAD_UNCHANGED).astype('float32')  # convert from original 16-bit
      4     #print(f'Raw image {id_} min : {np.min(img)} max : {np.max(img)}')
      5     mx = np.max(img)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in __getitem__(self, key)
   1189             maybe_callable = com.apply_if_callable(key, self.obj)
   1190             maybe_callable = self._check_deprecated_callable_usage(key, maybe_callable)
-> 1191             return self._getitem_axis(maybe_callable, axis=axis)
   1192 
   1193     def _is_scalar_access(self, key: tuple):

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _getitem_axis(self, key, axis)
   1750 
   1751             # validate the location
-> 1752             self._validate_integer(key, axis)
   1753 
   1754             return self.obj._ixs(key, axis=axis)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexing.py in _validate_integer(self, key, axis)
   1683         len_axis = len(self.obj._get_axis(axis))
   1684         if key >= len_axis or key < -len_axis:
-> 1685             raise IndexError("single positional indexer is out-of-bounds")
   1686 
   1687     # -------------------------------------------------------------------

IndexError: single positional indexer is out-of-bounds

## === cell 73
display_image('case117_day15_slice_0009', data, apply_CLAHE=True)
display_image('case117_day15_slice_0010', data, apply_CLAHE=True)
display_image('case117_day15_slice_0065', data, apply_CLAHE=True)
display_image('case117_day15_slice_0066', data, apply_CLAHE=True)

## === cell 76
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
index_train, index_valid = next(sgkf.split(data.id, data.segmentation.isna(), data.case))

## === cell 77
len(index_train), len(index_valid)

## === cell 78
data_train = data.iloc[index_train, :]
data_valid = data.iloc[index_valid, :]

## === cell 79
data_train

## === cell 80
data_valid

## === cell 82
print(len(data_train.case.unique()), len(data_valid.case.unique()))

## === cell 83
data_train_sub = data_train.loc[data_train.case.isin(data_train.case.unique()[:11]), :]
data_valid_sub = data_valid.loc[data_valid.case.isin(data_valid.case.unique()[:2]), :]

print(len(data_train_sub), len(data_valid_sub), len(data_train_sub)/len(data_valid_sub))



## === cell 84
missing_masks_train = data_train_sub.segmentation.isna().sum() 
missing_masks_valid = data_valid_sub.segmentation.isna().sum() 
print(missing_masks_train, missing_masks_train*100/len(data_train_sub))
print(missing_masks_valid, missing_masks_valid*100/len(data_valid_sub))

## === cell 86
na_counts_train = (
    data_train_sub.groupby('class')['segmentation']
    .apply(lambda s: s.isna().sum())
    .reset_index(name='count')
)
na_counts_train['percent'] = 100 * na_counts_train['count'] / data_train_sub.groupby('class')['segmentation'].size().values

display(na_counts_train)


na_counts_valid = (
    data_valid_sub.groupby('class')['segmentation']
    .apply(lambda s: s.isna().sum())
    .reset_index(name='count')
)
na_counts_valid['percent'] = 100 * na_counts_valid['count'] / data_valid_sub.groupby('class')['segmentation'].size().values

display(na_counts_valid)

## === cell 88
data_train_sub = data_train_sub.reset_index(drop=True)

## === cell 89
data_valid_sub = data_valid_sub.reset_index(drop=True)

## === cell 92
class GITractDataset(Dataset):
    def __init__(self, df, is_train=True, transforms=None):
        self.df = df
        self.id_ = df['id'].unique()
        self.is_train = is_train
        self.transforms = transforms

    def __len__(self):
        return len(self.id_)
    
    def __getitem__(self, idx):
        id_ = self.id_[idx]
        img = load_image(id_, self.df)
        img = np.tile(img[..., None], [1, 1, 3])
        if self.is_train:
            mask = get_mask(id_, self.df)
            if self.transforms:
                augmented = self.transforms(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']
            return img, mask, id_
        else:
            if self.transforms:
                augmented = self.transforms(image=img)
                img = augmented['image']
                data_sub = self.df.loc[self.df.id == id_]
                height = data_sub.slice_h.iloc[0]
                width = data_sub.slice_w.iloc[0]
            return img, id_, height, width

## === cell 94
transform_train = A.Compose([
    A.Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1], interpolation=cv2.INTER_NEAREST,
             mask_interpolation=cv2.INTER_NEAREST,),
    A.Normalize(mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_SD, max_pixel_value=1.0),
    A.ToTensorV2(transpose_mask = True),
])

transform_valid = A.Compose([
    A.Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1], interpolation=cv2.INTER_NEAREST,
             mask_interpolation=cv2.INTER_NEAREST,),
    A.Normalize(mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_SD, max_pixel_value=1.0),
    A.ToTensorV2(transpose_mask = True),
])

## === cell 95
dataset_train = GITractDataset(data_train_sub, transforms=transform_train)
dataset_valid = GITractDataset(data_valid_sub, transforms=transform_valid)

dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE_TRAIN, shuffle=True, num_workers=DATA_LOADER_NUM_WORKERS)
dataloader_valid = DataLoader(dataset_valid, batch_size=BATCH_SIZE_VALID, shuffle=False, num_workers=DATA_LOADER_NUM_WORKERS)

## === cell 96
dataset = next(iter(dataloader_train))
img, mask, id_ = dataset
print(img.shape, mask.shape, len(id_))

## === cell 97
idx = 31
np.max(img[idx].numpy()), np.min(img[idx].numpy())

## === cell 98
type(img[idx].numpy()[0, 0, 0]), type(mask[idx].numpy()[0, 0, 0])

## === cell 99
def display_dataset(dataset, display_orig=False, num_images=None, denormalize=False, apply_CLAHE=False):
	'''
	dataset : dataset to be displayed
	display_orig : Should the original images prior to augmentation be shown alongside images after augmentation.
				   In this case 1st 5 images before and after augmentation is shown and num_images parameter value is ignored
	num_images : Number of images to be shown. Defaults to the full dataset size i.e. the batch size
	denormalize : Set to True if A.normalize has been applied as part of augmentations and you wish to denormalize it
	'''
	img_arr, mask_arr, id_arr = dataset
	if num_images is None:
		num_images = len(img_arr)
	max_cols = 5
	
	if display_orig:
		num_images = 5
		rows = 2
		plt.figure(figsize=(max_cols*3, rows*3))
		ids_shown = list()
	else:
		rows = np.ceil(num_images/max_cols).astype(int)
		plt.figure(figsize=(max_cols*3, rows*3))
	
	for idx in range(num_images):
		img, mask, id_ = img_arr[idx], mask_arr[idx], id_arr[idx]
		img = img.permute(1,2,0)    #after the permute, img is in HxWxC format
		if denormalize:
			img = img * torch.tensor(IMAGE_NORMALIZE_SD) + torch.tensor(IMAGE_NORMALIZE_MEAN)
			img = img.clamp(0, 1)
		img = img.cpu().numpy()
		img = (img * 255).astype(np.uint8)

		if apply_CLAHE:
			clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
			for ch in range(3):
				img[:,:,ch] = clahe.apply(img[:,:,ch])

		mask = mask.permute(1,2,0).cpu().numpy()

		plt.subplot(rows, max_cols, idx+1)
		print(img.shape)
		plt.imshow(img[:, :, 0], cmap='bone')  #img was tiled grayscale, to display just use any 1 channel
		plt.title(f'{idx} : {id_}')
		
		plt.imshow(mask[..., 0], cmap=CMAP1)
		plt.imshow(mask[..., 1], cmap=CMAP2)
		plt.imshow(mask[..., 2], cmap=CMAP3)
		plt.axis('off')

		if idx == 0:
			handles = [
				Rectangle((0, 0), 1, 1, color=CMAP1(1.0)),
				Rectangle((0, 0), 1, 1, color=CMAP2(1.0)),
				Rectangle((0, 0), 1, 1, color=CMAP3(1.0))
			]
			labels = ['Large Bowel', 'Small Bowel', 'Stomach']
			plt.legend(handles, labels, bbox_to_anchor=(0.0, 1.5), loc='upper left', borderaxespad=0.)

		if display_orig:
			ids_shown.append(id_)

	if display_orig:
		print(ids_shown)

		for id_ in ids_shown:
			idx += 1
			img = load_image(id_, data)
			img = (img * 255).astype(np.uint8) # 0-255 range required for CLAHE. 
			if apply_CLAHE:
				clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(2,2))
				img = clahe.apply(img)
				
			mask = get_mask(id_, data)
			
			plt.subplot(rows, max_cols, idx+1)
			print(img.shape)
			plt.imshow(img, cmap='bone')
			plt.title('Original')
			plt.imshow(mask[..., 0], cmap=CMAP1)
			plt.imshow(mask[..., 1], cmap=CMAP2)
			plt.imshow(mask[..., 2], cmap=CMAP3)
			plt.axis('off')

	plt.tight_layout()
	plt.show()

## === cell 100
display_dataset(dataset, num_images=5, denormalize=True)

## === cell 101
display_dataset(dataset, display_orig=True, denormalize=True, apply_CLAHE=True)

## === cell 103
smp_encoder_weights = None if TEST_PREDICT and LOAD_MODEL_FOR_TEST_PREDICT else 'imagenet'
model = smp.Unet(
    encoder_name = 'efficientnet-b1',        
    encoder_weights = smp_encoder_weights,     
    in_channels = 3,                  
    classes = NUM_CLASSES,                      
)
model.to(DEVICE)

## --- ERROR in cell 103, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2168626566.py in <cell line: 0>()
      1 # https://smp.readthedocs.io/en/latest/encoders_timm.html
      2 smp_encoder_weights = None if TEST_PREDICT and LOAD_MODEL_FOR_TEST_PREDICT else 'imagenet'
----> 3 model = smp.Unet(
      4     encoder_name = 'efficientnet-b1',
      5     encoder_weights = smp_encoder_weights,

NameError: name 'smp' is not defined

## === cell 105
optimizer = optim.Adam(model.parameters(), lr=1e-3)

## --- ERROR in cell 105, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1645629056.py in <cell line: 0>()
----> 1 optimizer = optim.Adam(model.parameters(), lr=1e-3)

NameError: name 'model' is not defined

## === cell 107
dice_loss = smp.losses.DiceLoss(mode='multilabel') 
BCE_loss = smp.losses.SoftBCEWithLogitsLoss()

def loss_fn(y_pred, y_true, loss_wt = 0.5):
    return dice_loss(y_pred, y_true) * loss_wt + BCE_loss(y_pred, y_true) * (1 - loss_wt)

## --- ERROR in cell 107, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1810557308.py in <cell line: 0>()
----> 1 dice_loss = smp.losses.DiceLoss(mode='multilabel')
      2 BCE_loss = smp.losses.SoftBCEWithLogitsLoss()
      3 
      4 def loss_fn(y_pred, y_true, loss_wt = 0.5):
      5     return dice_loss(y_pred, y_true) * loss_wt + BCE_loss(y_pred, y_true) * (1 - loss_wt)

NameError: name 'smp' is not defined

## === cell 110
class DiceScoreCustom:
	def __init__(self, num_classes, eps=1e-6):
		self.num_classes = num_classes
		self.eps = eps
		self.reset()

	def reset(self):
		self.dice_sum = 0.0
		self.image_count = 0

		self.organ_dice_sum = torch.zeros(self.num_classes)
		self.organ_count = torch.zeros(self.num_classes)

	def update(self, preds, targets):
		"""
		preds, targets: (B, C, H, W) binary {0,1} tensors
		Implements host comment: skip organs where both pred & target are empty : https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation/discussion/324934
		"""
		I = (targets & preds).sum((2, 3))
		U = (targets | preds).sum((2, 3))

		dice = (2 * I) / (U + I + self.eps)

		non_empty = U > 0  # [B, C]

		organ_counts = non_empty.sum(dim=1)  # [B]
		dice_per_image = dice.sum(dim=1) / organ_counts.clamp(min=1)


		self.dice_sum += dice_per_image.sum().item()
		self.image_count += dice_per_image.numel()

		self.organ_dice_sum += dice.sum(dim=0).detach().cpu()
		self.organ_count += non_empty.sum(dim=0).detach().cpu()

	def compute(self):
		"""
		Returns:
			overall dice: scalar tensor
			per_organ dice: (C,) tensor
		"""
		overall = (
			torch.tensor(self.dice_sum / self.image_count)
			if self.image_count > 0 
			else torch.tensor(0.0)
		)
		per_organ = torch.where(
			self.organ_count > 0,
			self.organ_dice_sum / self.organ_count,
			torch.tensor(0.0)
		)
		return overall, per_organ

## === cell 112
class HausdorffDistanceCustom:
	def __init__(self, num_classes):
		self.num_classes = num_classes
		self.reset()

	def reset(self):
		self.h3d_sum = 0.0
		self.image3d_count = 0

		self.organ_h3d_sum = np.zeros(self.num_classes)
		self.organ_count_sum = np.zeros(self.num_classes)

	def _compute_hausdorff_per_organ(self, preds, targets):
		'''
		preds and targets : (Depth, Height, Width) binary {0,1} tensors
		'''
		if np.all(preds == targets):
			return 0.0
	
		(edges_preds, edges_targets) = get_mask_edges(preds, targets)
		surface_distance = get_surface_distance(edges_preds, edges_targets, distance_metric="euclidean")
	
		if surface_distance.shape == (0,):
			return 0.0
		dist = surface_distance.max()
		max_dist = np.sqrt(np.sum((np.array(preds.shape) - 1) ** 2))
	
		if dist > max_dist:
			return 1.0
	
		return dist / max_dist

	def update(self, preds, targets):
		'''
		preds and targets : (Channel, Depth, Height, Width) binary {0,1} tensors
		'''

		U = (targets | preds).sum((1, 2, 3))  # [C]

		hausdorff = np.array([self._compute_hausdorff_per_organ(preds[i, ...], targets[i, ...]) for i in range(NUM_CLASSES)])  # [C]

		non_empty = U > 0  # [C]

		organ_count = non_empty.sum()

		if organ_count != 0:
			hausdorff_per_3dimage = hausdorff.sum() / organ_count

			self.h3d_sum += hausdorff_per_3dimage
			self.image3d_count += 1

		self.organ_h3d_sum += hausdorff
		self.organ_count_sum += non_empty

	def compute(self):
		"""
		Returns:
			overall hausdorff: scalar
			per_organ hausdorff: (C,)
		"""
		overall = self.h3d_sum / self.image3d_count

		per_organ = self.organ_h3d_sum / self.organ_count_sum

		return overall, per_organ

## === cell 113
dice_score_obj = DiceScoreCustom(num_classes=NUM_CLASSES)
hausdorff_obj = HausdorffDistanceCustom(num_classes=NUM_CLASSES)

## === cell 115
def one_epoch_train(epoch):
    
    model.train() #set model in training mode
    running_loss = 0.0
    
    loop = tqdm(dataloader_train, desc=f'Epoch {epoch+1}/{EPOCHS}')
    for data in loop:
        imgs, masks, ids = data
        imgs, masks = imgs.to(DEVICE, dtype=torch.float), masks.to(DEVICE, dtype=torch.float)
        optimizer.zero_grad()
        pred_masks = model(imgs)

        loss = loss_fn(pred_masks, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = running_loss / len(dataloader_train)
    return avg_loss

## === cell 116
slices80_casedays = set(
    data_valid[['case', 'day', 'slice']]
    .drop_duplicates()
    .value_counts(['case', 'day'])
    .loc[lambda s: s == 80]
    .index
)


## === cell 117
def one_epoch_valid():

    model.eval()
    with torch.no_grad():
        running_loss = 0.0
        pred_masks_dict, masks_dict = {}, {}
        for data in dataloader_valid:
            imgs, masks, ids = data
            imgs, masks = imgs.to(DEVICE, dtype=torch.float), masks.to(DEVICE, dtype=torch.float)
            pred_masks = model(imgs)
            loss = loss_fn(pred_masks, masks)
            running_loss += loss.item()

            pred_masks = (torch.sigmoid(pred_masks) > 0.5).int()
            masks = masks.int()
            dice_score_obj.update(pred_masks, masks)

            for p, m, id_ in zip(pred_masks, masks, ids):
                match = re.match(r"case(\d+)_day(\d+)_slice_(\d+)", id_)
                if match:
                    caseid, dayid, sliceid = map(int, match.groups())

                casedayid = (caseid, dayid) 

                pred_masks_dict.setdefault(casedayid, []).append((sliceid, p))
                masks_dict.setdefault(casedayid, []).append((sliceid, m))

                if (len(pred_masks_dict[casedayid]) == 144) or (casedayid in slices80_casedays and len(pred_masks_dict[casedayid]) == 80):
                    pred_masks_sorted = [p.cpu().numpy() for sid, p in sorted(pred_masks_dict[casedayid], key=lambda x: x[0])]
                    masks_sorted = [m.cpu().numpy() for sid, m in sorted(masks_dict[casedayid], key=lambda x: x[0])]

                    pred_masks_volume = np.stack(pred_masks_sorted, axis=1)
                    masks_volume = np.stack(masks_sorted, axis=1)

                    hausdorff_obj.update(pred_masks_volume, masks_volume)

                    del pred_masks_dict[casedayid], masks_dict[casedayid]

            
        avg_loss = running_loss / len(dataloader_valid)

        epoch_dice_score = dice_score_obj.compute()
        dice_score_obj.reset()
        
        epoch_hausdorff = hausdorff_obj.compute()
        hausdorff_obj.reset()    
        
    return avg_loss, epoch_dice_score, epoch_hausdorff
    

## === cell 118
if TRAIN_VALID_SPLIT:
    for epoch in range(EPOCHS):
        loss_train = one_epoch_train(epoch)
        loss_valid, dice_score, hausdorff = one_epoch_valid()
        dice_overall, dice_per_organ = dice_score
        hausdorff_overall, hausdorff_per_organ = hausdorff
        combined_metric = 0.4*dice_overall + 0.6*(1-hausdorff_overall)
        print(
            f'Epoch {epoch+1} | '
            f'Train Loss: {loss_train:.3f} | Valid Loss: {loss_valid:.3f} | '
            f'Combined metric: {combined_metric:.3f} | '
            f'Dice: {dice_overall:.3f} (LB {dice_per_organ[0]:.3f}, SB {dice_per_organ[1]:.3f}, S {dice_per_organ[2]:.3f}) | '
            f'Hausdorff: {hausdorff_overall:.3f} (LB {hausdorff_per_organ[0]:.3f}, SB {hausdorff_per_organ[1]:.3f}, S {hausdorff_per_organ[2]:.3f})'
        )

## === cell 119
if TRAIN_VALID_SPLIT and SAVE_TRAIN_VALID_MODEL:
    torch.save(model.state_dict(), MODEL_PARAMS_FILE_NAME)

## === cell 121
if TEST_PREDICT: 
    if LOAD_MODEL_FOR_TEST_PREDICT:
        model.load_state_dict(torch.load(MODEL_PARAMS_LOAD_FILE_PATH))
    model.eval()

    data_test = pd.read_csv(DIR_PATH + "sample_submission.csv")
    test_set_hidden = not bool(len(data_test))
    if test_set_hidden:
        data_test = data_valid_sub  # Use validation data for testing the code prior to submission
    else:
        data_test[['case', 'day', 'slice']] = data_test['id'].str.extract(r'case(\d+)_day(\d+)_slice_(\d+)')
        path_df = get_path_df(train = False)

        data_test = data_test.merge(path_df, on = ['case', 'day', 'slice'])
    
        int_cols = ['case', 'day', 'slice', 'slice_w', 'slice_h']
        data_test[int_cols] = data_test[int_cols].astype(np.uint32)
    
        float_cols = ['px_w', 'px_h']
        data_test[float_cols] = data_test[float_cols].astype(np.float32)

    transform_test = A.Compose([
        A.Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1], interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=IMAGE_NORMALIZE_MEAN, std=IMAGE_NORMALIZE_SD, max_pixel_value=1.0),
        A.ToTensorV2(transpose_mask = False),
    ])    
    dataset_test = GITractDataset(data_test, is_train=False, transforms=transform_test)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=DATA_LOADER_NUM_WORKERS)

## --- ERROR in cell 121, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/872555110.py in <cell line: 0>()
      1 if TEST_PREDICT:
      2     if LOAD_MODEL_FOR_TEST_PREDICT:
----> 3         model.load_state_dict(torch.load(MODEL_PARAMS_LOAD_FILE_PATH))
      4     model.eval()
      5 

NameError: name 'model' is not defined

## === cell 122
if TEST_PREDICT:
    test_ids, test_class, test_pred_RLE = [], [], []   # data to be written to submission file
    
    with torch.no_grad():
        for imgs, ids, heights, widths in dataloader_test:
            imgs = imgs.to(DEVICE, dtype=torch.float)
            pred_masks = model(imgs)
            pred_masks = (torch.sigmoid(pred_masks) > 0.5).int()
            pred_masks = pred_masks.permute(0, 2, 3, 1).cpu().numpy()   # shape after permute [B, H, W, C]

            for mask, id_, h, w in zip(pred_masks, ids, heights, widths):
                mask_orig_size = cv2.resize(mask, dsize=(w.item(), h.item()), interpolation=cv2.INTER_NEAREST)
                rles = [rle_encode(mask_orig_size[..., chid]) for chid in range(NUM_CLASSES)]
                
                test_ids.extend([id_] * NUM_CLASSES)
                test_class.extend(CLASS_NAMES)
                test_pred_RLE.extend(rles)

    submission_df = pd.DataFrame({
        'id': test_ids, 
        'class': test_class, 
        'predicted': test_pred_RLE
    })
    submission_df.to_csv('submission.csv', index=False)
    !head submission.csv
    display(submission_df.loc[submission_df.predicted != ""].head())

## --- ERROR in cell 122, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1397630704.py in <cell line: 0>()
      3 
      4     with torch.no_grad():
----> 5         for imgs, ids, heights, widths in dataloader_test:
      6             imgs = imgs.to(DEVICE, dtype=torch.float)
      7             pred_masks = model(imgs)

NameError: name 'dataloader_test' is not defined
