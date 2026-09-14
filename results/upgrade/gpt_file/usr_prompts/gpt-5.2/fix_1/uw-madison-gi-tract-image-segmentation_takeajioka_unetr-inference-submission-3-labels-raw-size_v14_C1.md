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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nibabel==5.3.2
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

0.8484889039601035

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import glob, os, gc

## === cell 1
sub_df = pd.read_csv('../input/uw-madison-gi-tract-image-segmentation/sample_submission.csv')
if len(sub_df):
    sub = True
else:
    sub = False

## === cell 2
DATASET_FOLDER = "../input/uw-madison-gi-tract-image-segmentation"
if sub == True:
    path_csv = os.path.join(DATASET_FOLDER, "sample_submission.csv")
    df_train = pd.read_csv(path_csv)
    folder = "test"
else:
    path_csv = os.path.join(DATASET_FOLDER, "train.csv")
    df_train = pd.read_csv(path_csv)[:1000]
    df_train = df_train.rename(columns={'segmentation': 'predicted'})
    folder = "train"
display(df_train.head())

## === cell 3
def extract_details(id_):
    id_fields = id_.split("_")
    case = id_fields[0].replace("case", "")
    day = id_fields[1].replace("day", "")
    slice_id = id_fields[3]
    return {
        "Case": int(case),
        "Day": int(day),
        "Slice": slice_id,
    }

## === cell 4
df_train[['Case','Day','Slice']] = df_train['id'].apply(lambda x: pd.Series(extract_details(x)))
display(df_train.head())

## === cell 5
train_overview = []
for (case, day), dfg in df_train.groupby(["Case", "Day"]):
    train_overview.append({"Day": int(day), "Case": int(case), "Slices": len(dfg)})
df_train_overview = pd.DataFrame(train_overview)
display(df_train_overview.head())

## === cell 6
from PIL import Image

def load_image_volume(img_dir, quant=0.01):
    imgs = sorted(glob.glob(os.path.join(img_dir, f"*.png")))
    imgs = [np.array(Image.open(p)).tolist() for p in imgs]
    vol = np.array(imgs)
    if quant:
        q_low, q_high = np.percentile(vol, [quant * 100, (1 - quant) * 100])
        vol = np.clip(vol, q_low, q_high)
    v_min, v_max = np.min(vol), np.max(vol)
    vol = (vol - v_min) / (v_max - v_min)
    vol = (vol * 255).astype(np.uint8)
    del imgs
    gc.collect()
    return vol

## === cell 7
import nibabel as nib

## === cell 8
df_train_overview['vol_path'] = ''
df_train_overview

## === cell 9
for i in range(len(df_train_overview)):
    CASE = df_train_overview['Case'][i]
    DAY = df_train_overview['Day'][i]
    IMAGE_FOLDER = os.path.join("../input/uw-madison-gi-tract-image-segmentation/", folder, f"case{CASE}", f"case{CASE}_day{DAY}", "scans")
    vol = load_image_volume(img_dir=IMAGE_FOLDER)
    print(vol.shape)
    nii1 = nib.Nifti1Image(vol,affine=None)
    nii_path = f'./{CASE}_{DAY}_vol.nii.gz'
    df_train_overview.loc[i,'vol_path'] = nii_path
    nib.save(nii1, nii_path)

## === cell 10
df_train_overview

## === cell 11
test_data = []
for i in range(len(df_train_overview)):
    CASE = df_train_overview['Case'][i]
    DAY = df_train_overview['Day'][i]
    path = {'image':  f'./{CASE}_{DAY}_vol.nii.gz'}
    test_data.append(path)

## === cell 12
import json

data1 = {
"description": "UWM",
"labels": {
    "0": "background",
    "1": "large_bowel",
    "2": "small_bowel",
    "3": "stomach",
},
"test": test_data
}

json_string = json.dumps(data1)
print(json_string)

## === cell 13
with open('json_data.json', 'w') as outfile:
    json.dump(json_string, outfile)

with open('json_data.json', 'w') as outfile:
    outfile.write(json_string)

## === cell 14
!pip install monai_weekly --no-index --find-links=file:///kaggle/input/uwm-model/monai_weekly-0.9.dev2218-py3-none-any.whl

## === cell 15
!pip install einops --no-index --find-links=file:///kaggle/input/uwm-model/einops-0.4.1-py3-none-any.whl

## === cell 16
import shutil
import tempfile

import matplotlib.pyplot as plt
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    CropForegroundd,
    Spacingd,
    ToTensord,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)


import torch

print_config()

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_10/2377996944.py in <cell line: 0>()
      5 from tqdm import tqdm
      6 
----> 7 from monai.losses import DiceCELoss
      8 from monai.inferers import sliding_window_inference
      9 from monai.transforms import (

ModuleNotFoundError: No module named 'monai'

## === cell 17
sz = (80, 144, 192)

## === cell 18
test_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        NormalizeIntensityd(keys=["image"]),
        ToTensord(keys=["image"]),
    ]
)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2586209082.py in <cell line: 0>()
----> 1 test_transforms = Compose(
      2     [
      3         LoadImaged(keys=["image"]),
      4         AddChanneld(keys=["image"]),
      5         NormalizeIntensityd(keys=["image"]),

NameError: name 'Compose' is not defined

## === cell 19
model = UNETR(
    in_channels=1,
    out_channels=3,
    img_size=sz,
    pos_embed="perceptron",
    dropout_rate=0.2,
).cuda()

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2387716984.py in <cell line: 0>()
----> 1 model = UNETR(
      2     in_channels=1,
      3     out_channels=3,
      4     img_size=sz,
      5     pos_embed="perceptron",

NameError: name 'UNETR' is not defined

## === cell 20
model.load_state_dict(torch.load('../input/unetr-raw-size/best_metric_model.pth'))

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1118108006.py in <cell line: 0>()
----> 1 model.load_state_dict(torch.load('../input/unetr-raw-size/best_metric_model.pth'))

NameError: name 'model' is not defined

## === cell 21
def rle_decode(mask_rle, shape):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
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

## === cell 22
def segm_rle(segm, df_vol):
    df = pd.DataFrame(index=[], columns=['id','class','predicted'])
    df_vol = df_vol.replace(np.nan, '')
    lbs = sorted(df_vol["class"].unique())
    for idx_, dfg in df_vol.groupby("Slice"):
        idx = int(idx_) - 1
        for i, lb in dfg[["class"]].iterrows():
            lb = lbs.index(lb.item())
            mask = segm[lb, idx, :, :]
            dfg.loc[i,"predicted"] = rle_encode(mask > 0.5)
        df = df.append(dfg.loc[:,['id','class','predicted']])
    del segm
    gc.collect()
    return df

## === cell 23
datasets = "./json_data.json"
datalist = load_decathlon_datalist(datasets, True, "test")
test_ds = CacheDataset(data=datalist, transform=test_transforms, cache_num=16, cache_rate=1.0, num_workers=2)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2131154473.py in <cell line: 0>()
      1 datasets = "./json_data.json"
----> 2 datalist = load_decathlon_datalist(datasets, True, "test")
      3 test_ds = CacheDataset(data=datalist, transform=test_transforms, cache_num=16, cache_rate=1.0, num_workers=2)

NameError: name 'load_decathlon_datalist' is not defined

## === cell 24
model.eval()
pred_df = pd.DataFrame(index=[], columns=['id','class','predicted'])
for i in range(len(df_train_overview)):
    x_data = test_ds[i]
    path = './' + x_data["image_meta_dict"]["filename_or_obj"]
    x_df = df_train_overview[df_train_overview['vol_path'] == path]
    CASE = x_df['Case'].item()
    DAY = x_df['Day'].item()
    inputs = torch.unsqueeze(x_data["image"], 0).cuda()
    print(inputs.shape)
    with torch.no_grad():
        output = sliding_window_inference(inputs, sz, 1, model, overlap=0.8).cpu()
    segm = np.squeeze(output.detach().numpy())
    df_ = df_train[(df_train["Case"] == CASE) & (df_train["Day"] == DAY)]
    pred_df = pred_df.append(segm_rle(segm, df_))
    del inputs, output, segm
    gc.collect()

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2979151910.py in <cell line: 0>()
----> 1 model.eval()
      2 pred_df = pd.DataFrame(index=[], columns=['id','class','predicted'])
      3 for i in range(len(df_train_overview)):
      4     x_data = test_ds[i]
      5     path = './' + x_data["image_meta_dict"]["filename_or_obj"]

NameError: name 'model' is not defined

## === cell 25
df_train["predicted"] = pred_df["predicted"]
df_train

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1640296287.py in <cell line: 0>()
----> 1 df_train["predicted"] = pred_df["predicted"]
      2 df_train

NameError: name 'pred_df' is not defined

## === cell 26
pred_df

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/158947453.py in <cell line: 0>()
----> 1 pred_df

NameError: name 'pred_df' is not defined

## === cell 27
if sub == True:
    sub_df = pd.read_csv(os.path.join(DATASET_FOLDER, "sample_submission.csv"))
else:
    sub_df = pd.read_csv(os.path.join(DATASET_FOLDER, "train.csv"))[:1000]
    sub_df = sub_df.rename(columns={'segmentation': 'predicted'})

## === cell 28
del sub_df['predicted']
sub_df = sub_df.merge(pred_df, on=["id", "class"])
sub_df.to_csv("submission.csv", index=False)

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/946271406.py in <cell line: 0>()
      1 del sub_df['predicted']
----> 2 sub_df = sub_df.merge(pred_df, on=["id", "class"])
      3 sub_df.to_csv("submission.csv", index=False)

NameError: name 'pred_df' is not defined

## === cell 29
sub_df
