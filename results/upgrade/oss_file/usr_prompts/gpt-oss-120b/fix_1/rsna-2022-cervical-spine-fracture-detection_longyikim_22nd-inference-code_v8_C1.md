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
Identify fractures in CT scans of the cervical spine (neck) at both the level of a single vertebrae and the entire patient.

## Metric
Weighted multi-label logarithmic loss. Each fracture sub-type is its own row for every exam, and you are expected to predict a probability for a fracture at each of the seven cervical vertebrae designated as C1, C2, C3, C4, C5, C6 and C7. There is also an any label, `patient_overall`, which indicates that a fracture of ANY kind described before exists in the examination. Fractures in the skull base, thoracic spine, ribs, and clavicles are ignored. The any label is weighted more highly than specific fracture level sub-types.

For each exam Id, you must submit a set of predicted probabilities (a separate row for each cervical level subtype). We then take the log loss for each predicted probability versus its true label.

The binary weighted log loss function for label j on exam i is specified as:

$$
L_{i j}=-w_j *\left[y_{i j} * \log \left(p_{i j}\right)+\left(1-y_{i j}\right) * \log \left(1-p_{i j}\right)\right]
$$

Finally, loss is averaged across all rows.

## Submission Format
There will be 8 rows per image Id. The label indicated by a particular row will look like [image Id]_[Sub-type Name], as follows. There is also a target column, `fractured`, indicating the probability of whether a fracture exists at the specified level. For each image ID in the test set, you must predict a probability for each of the different possible sub-types and the patient overall. The file should contain a header and have the following format:

```
row_id,fractured
1_C1,0
1_C2,0
1_C3,0
1_C4,0.6
1_C5,0
1_C6,0.9
1_C7,0.01
1_patient_overall,0.99
2_C1,0
etc.
```

## Dataset
**train.csv** Metadata for the train test set.

- `StudyInstanceUID` - The study ID. There is one unique study ID for each patient scan.
- `patient_overall` - One of the target columns. The patient level outcome, i.e. if any of the vertebrae are fractured.
- `C[1-7]` - The other target columns. Whether the given vertebrae is fractured. See [this diagram](https://en.wikipedia.org/wiki/Vertebral_column#/media/File:Gray_111_-_Vertebral_column-coloured.png) for the real location of each vertbrae in the spine.

**test.csv** Metadata for the test set prediction structure. Only the first few rows of the test set are available for download.

- `row_id` - The row ID. This will match the same column in the sample submission file.
- `StudyInstanceUID` - The study ID.
- `prediction_type` - Which one of the eight target columns needs a prediction in this row.

**[train/test]_images/[StudyInstanceUID]/[slice_number].dcm** The image data, organized with one folder per scan. Expect to see roughly 1,500 scans in the hidden test set.\

Each image is in [the dicom file format](https://www.dicomstandard.org/). The DICOM image files are ≤ 1 mm slice thickness, axial orientation, and bone kernel. Note that some of the DICOM files are JPEG compressed. You may require additional resources to read the pixel array of these files, such as GDCM and pylibjpeg.

**sample_submission.csv** A valid sample submission.

- `row_id` - The row ID. See the test.csv for what prediction needs to be filed in that row.
- `fractured` - The target column.

**train_bounding_boxes.csv** Bounding boxes for a subset of the training set.

**segmentations/** Pixel level annotations for a subset of the training set. This data is provided in the [nifti file format](https://nifti.nimh.nih.gov/).

A portion of the imaging datasets have been segmented automatically using a 3D UNET model, and radiologists modified and approved the segmentations. The provided segmentation labels have values of 1 to 7 for C1 to C7 (seven cervical vertebrae) and 8 to 19 for T1 to T12 (twelve thoracic vertebrae are located in the center of your upper and middle back), and 0 for everything else. As we focused on the cervical spine, all scans have C1 to C7 labels but not all thoracic labels.

Please be aware that the NIFTI files consist of segmentation in the sagittal plane, while the DICOM files are in the axial plane. Please use the NIFTI header information to determine the appropriate orientation such that the DICOM images and segmentation match. Otherwise, you run the risk of having the segmentations flipped in the Z axis and mirrored in the X axis.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        input/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        working/
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
```

-> data/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/rsna-2022-cervical-spine-fracture-detection/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/rsna-2022-cervical-spine-fracture-detection/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> data/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> (stopped after 10 files for performance)

# 5. Target score

1.1300110354563326

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
try:
    import pylibjpeg
except:
    !mkdir -p /root/.cache/torch/hub/checkpoints/
    !cp ../input/effdet-models/efficientnet_v2s_ra2_288-a6477665.pth /root/.cache/torch/hub/checkpoints/
    !pip install ../input/effdet-models/pydicom-2.3.0-py3-none-any.whl
    !pip install ../input/effdet-models/pylibjpeg-1.4.0-py3-none-any.whl
    !pip install ../input/effdet-models/pylibjpeg_libjpeg-1.3.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    !pip install ../input/effdet-models/python_gdcm-3.0.19-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    !pip install ../input/effdet-models/pycocotools-2.0.2-cp37-cp37m-linux_x86_64.whl

## === cell 1
import pandas as pd
import torch
import re
import sys

import pydicom as dicom
import pylibjpeg

effdet_path = "../input/effdet-models/effdet"
sys.path.append(effdet_path)


timm_path = "../input/effdet-models/timm-pytorch-image-models"
sys.path.append(timm_path)
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

omega_path = "../input/effdet-models/omegaconf"
sys.path.append(omega_path)
from omegaconf import OmegaConf

effunet_path = "../input/effdet-models/efficientunet-pytorch-0.0.6"
sys.path.append(effunet_path)


import glob
import math
import random
import numpy as np

from PIL import Image


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torchvision import transforms
import torchvision.models as models
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import warnings

warnings.filterwarnings('ignore')



device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/964151396.py in <cell line: 0>()
      8 
      9 import pydicom as dicom
---> 10 import pylibjpeg
     11 
     12 effdet_path = "../input/effdet-models/effdet"

ModuleNotFoundError: No module named 'pylibjpeg'

## === cell 2
IMAGES_DIR='../input/rsna-2022-cervical-spine-fracture-detection/test_images'
TRAIN_IMAGES_PATH='../input/rsna-2022-cervical-spine-fracture-detection/train_images'
TEST_IMAGES_PATH='../input/rsna-2022-cervical-spine-fracture-detection/test_images'

## === cell 3
segmentation_checkpoint="../input/effdet-models/axial_segmentation_effseg_163725-epoch-100.pth"
axial_det_checkpoint="../input/effdet-models/axial_detection_effdet_134352-epoch-52.pth"

## === cell 5





test_slices = glob.glob(f'{TEST_IMAGES_PATH}/*/*')
test_slices = [re.findall(f'{TEST_IMAGES_PATH}/(.*)/(.*).dcm', s)[0] for s in test_slices]
df_test_slices = pd.DataFrame(data=test_slices, columns=['StudyInstanceUID', 'Slice']).astype({'Slice': int}).sort_values(['StudyInstanceUID', 'Slice']).reset_index(drop=True)
df_test_slices

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4222074533.py in <cell line: 0>()
     11 # df_test_slices
     12 
---> 13 test_slices = glob.glob(f'{TEST_IMAGES_PATH}/*/*')
     14 test_slices = [re.findall(f'{TEST_IMAGES_PATH}/(.*)/(.*).dcm', s)[0] for s in test_slices]
     15 df_test_slices = pd.DataFrame(data=test_slices, columns=['StudyInstanceUID', 'Slice']).astype({'Slice': int}).sort_values(['StudyInstanceUID', 'Slice']).reset_index(drop=True)

NameError: name 'glob' is not defined

## === cell 6
def rescale_img_to_hu(dcm_ds):
    """Rescales the image to Hounsfield unit.
    """
    return dcm_ds.pixel_array * dcm_ds.RescaleSlope + dcm_ds.RescaleIntercept

def normalize_hu(data):
    data = np.clip(data, a_min=-2242, a_max=2242) / 4484 + 0.5
    return data

def load_dicom(path):
    """
    This supports loading both regular and compressed JPEG images. 
    See the first sell with `pip install` commands for the necessary dependencies
    """
    ds=dicom.dcmread(path) 

    img = normalize_hu(rescale_img_to_hu(ds))

    return img, ds.PixelSpacing[0]




## === cell 7
class DcmDataSet(torch.utils.data.Dataset):    
    def __init__(self, df, path, transforms=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transforms = transforms
        
    def __getitem__(self, i):
        path = os.path.join(self.path, self.df.iloc[i].StudyInstanceUID, f'{self.df.iloc[i].Slice}.dcm')        
        
        try:
            img, pixel_spacing = load_dicom(path)      
            if self.transforms is not None:
                img = self.transforms(img)
        except Exception as ex:
            print(ex)
            return None, None
        
        return img, pixel_spacing      
    
    def __len__(self):
        return len(self.df)
    
class DataTransform(nn.Module):
    def __init__(self, image_size=512):
        super().__init__()

        self.image_size = image_size

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.Normalize(0.5, 0.5),
        ])


    def forward(self, x):
        x = self.transform(torch.as_tensor(x, dtype=torch.float).unsqueeze(0))

        return x

tf = DataTransform()
ds = DcmDataSet(df_test_slices, IMAGES_DIR, tf)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3858620137.py in <cell line: 0>()
     24         return len(self.df)
     25 
---> 26 class DataTransform(nn.Module):
     27     def __init__(self, image_size=512):
     28         super().__init__()

NameError: name 'nn' is not defined

## === cell 8
batch_size = 16
dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=min(os.cpu_count(), batch_size))



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2303508690.py in <cell line: 0>()
      1 batch_size = 16
----> 2 dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=min(os.cpu_count(), batch_size))
      3 
      4 # x, pixel_spacings = next(iter(dl))
      5 # print(x.shape)

NameError: name 'DataLoader' is not defined

## === cell 9
from efficientunet import *
def get_axial_segmentation_model(checkpoint):
    model = get_efficientunet_b5(out_channels=2, concat_input=True, pretrained=False)
    
    state = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(state["model"])
    model.eval()
    return model.to(device)
    
seg_model = get_axial_segmentation_model(segmentation_checkpoint)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2132450042.py in <cell line: 0>()
----> 1 from efficientunet import *
      2 def get_axial_segmentation_model(checkpoint):
      3     model = get_efficientunet_b5(out_channels=2, concat_input=True, pretrained=False)
      4 
      5     state = torch.load(checkpoint, map_location=torch.device(device))

ModuleNotFoundError: No module named 'efficientunet'

## === cell 10
from effdet import create_model

def get_axial_detection_model(checkpoint, image_size=512):
    model = create_model('efficientdetv2_ds' , bench_task='predict' , num_classes=1 , image_size=(image_size, image_size), pretrained=False, max_det_per_image=1)

    state = torch.load(checkpoint, map_location=torch.device(device))
    model.load_state_dict(state["model"])
    model.model.eval()
    model = model.eval()
    return model.to(device)

det_model = get_axial_detection_model(axial_det_checkpoint)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2996081664.py in <cell line: 0>()
----> 1 from effdet import create_model
      2 
      3 def get_axial_detection_model(checkpoint, image_size=512):
      4     model = create_model('efficientdetv2_ds' , bench_task='predict' , num_classes=1 , image_size=(image_size, image_size), pretrained=False, max_det_per_image=1)
      5 

ModuleNotFoundError: No module named 'effdet'

## === cell 11
def get_axial_boundary_from_segmentation(seg, pixel_spacing, throw=100, tol=0.2, max_mm=100):
    """
    seg : H x W
    """
    image_size = seg.shape[0]
    min_size = min(image_size, max_mm / pixel_spacing)
    
    rows, columns = seg.nonzero(as_tuple=True)
    rows.sort()
    columns.sort()
    
    throw = min(len(rows) // 2, throw)
    
    if(len(rows)) == 0:
        return torch.tensor([0, 0, image_size, image_size]).to(device)
    
    xmin, xmax = columns[throw], columns[-throw]
    ymin, ymax = rows[throw], rows[-throw]
    
    w = (xmax - xmin) * (1 + tol)
    h = (ymax - ymax) * (1 + tol)
    new_size = max(w, h, min_size)
    new_size = min(image_size, new_size)
    
    xcenter, ycenter = (xmax + xmin) / 2, (ymax + ymin) / 2
    
    xmin = torch.min(torch.tensor(image_size - new_size), xcenter - new_size / 2)
    xmin = xmin.clip(min=0)
    
    ymin = torch.min(torch.tensor(image_size - new_size), ycenter - new_size / 2)
    ymin = ymin.clip(min=0)
    
    return torch.stack([xmin, ymin, xmin + new_size, ymin + new_size])

## === cell 12

def predict_seg(x, model, img_size=256):
    """
    return: N x 1 x H x W
    """
    x = TF.resize(x, (img_size, img_size))
    logits = model(x)

    classification_score, mse_score = logits.sigmoid().chunk(2, dim=1)
    classification_pred = classification_score.gt(0.5).float()
    pred = (classification_pred * mse_score)
    
    return pred

## === cell 14
def get_axial_boundary(segs, pixel_spacings, seg_img_size=256):
    boundary_list = []
    for i in range(segs.shape[0]):
        seg = segs[i, 0, :, :]
        
        boundary = get_axial_boundary_from_segmentation(seg, pixel_spacings[i], throw=int(100 / 512 * seg_img_size), tol=0.2, max_mm=100 / 512 * seg_img_size)
        boundary_list.append(boundary)
    boundary_list = torch.stack(boundary_list, axis=0) * (512. / seg_img_size)
    return boundary_list

## === cell 15

def predict_det(x, model):
    
    bboxes = model(x) # N x 1 x 6
    
    return bboxes[:, 0, :]
    
    
def crop_resize_images(imgs_tensor, boundary_list, img_size=512):
    croped_list = []
    for i in range(imgs_tensor.shape[0]):
        xmin, ymin, xmax, ymax = boundary_list[i, :]
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
        croped = TF.crop(imgs_tensor[i, :, :, :], top=ymin, left=xmin, height=ymax-ymin, width=xmax-xmin)
        croped = TF.resize(croped, (img_size, img_size))
        croped_list.append(croped)
        
    return torch.stack(croped_list, 0)

def get_original_bbox(bbox, boundary):
    scale = 512. / (boundary[:, [2]] - boundary[:, [0]])
    
    org_bbox = bbox / scale
    org_bbox[:, 0] += boundary[:, 0]
    org_bbox[:, 1] += boundary[:, 1]
    org_bbox[:, 2] += boundary[:, 0]
    org_bbox[:, 3] += boundary[:, 1]
    
    return org_bbox



## === cell 16
def get_bbox_class(seg, bbox):
    """
    label 은 0.125 의 단위로, 
    seg: H x W
    bbox: [xmin, ymin, xmax, ymax]
    """
    xmin, ymin, xmax, ymax = bbox.int()
    area = seg[ymin:ymax, xmin:xmax]
    
    result = torch.mean(area[area>0])
    result = torch.round(result / 0.125)
    
    return result

## === cell 17
def get_bbox_class_list(seg_list, seg_bboxes):
    class_list = []
    for i in range(seg_list.shape[0]):
        class_index = get_bbox_class(seg_list[i, :, :], seg_bboxes[i, :])
        class_list.append(class_index)
        
    return torch.stack(class_list)

## === cell 18
    
        
        
    
    

def get_class_score(scores, class_list, eps=1e-2):
    result = scores.new_zeros((scores.shape[0], 8)) + eps
    class_list = torch.nan_to_num(class_list).long()
    result[:, class_list] = scores
    
    return result

def check_detection_result(det_result, img_size=512., threshold=0.2):
    areas = (det_result[:, 2] - det_result[:, 0]) * (det_result[:, 3] - det_result[:, 1]) / (img_size * img_size)
    big_indices = torch.argwhere(areas > threshold)
    det_result[big_indices, 4] = 0.
    return det_result

## === cell 19
def predict():
    
    with torch.no_grad():
        predictions = []
        for (x, pixel_spacings) in tqdm(dl):
            x = torch.cat((x, x, x), dim=1).to(device)
            
            seg_result = predict_seg(x, seg_model)  # N x 1 x 256 x 256
            axial_boundary = get_axial_boundary(seg_result, pixel_spacings, seg_img_size=256)  # N x 4
            
            x = crop_resize_images(x, axial_boundary) # N x 3 x 512 x 512 croped
            det_result = predict_det(x, det_model)
            det_result = check_detection_result(det_result, threshold=0.1)
            
            bboxes, scores = get_original_bbox(det_result[:, :4], axial_boundary), det_result[:, 4]
            
            class_list = get_bbox_class_list(seg_result[:, 0, :, :], bboxes / 2)
            
            probs = get_class_score(scores, class_list)
            
            predictions.append(probs)
        
        return torch.concat(predictions).cpu().numpy()
predictions = predict()


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3758323000.py in <cell line: 0>()
     24 
     25         return torch.concat(predictions).cpu().numpy()
---> 26 predictions = predict()
     27 # predictions

/tmp/ipykernel_11/3758323000.py in predict()
      3     with torch.no_grad():
      4         predictions = []
----> 5         for (x, pixel_spacings) in tqdm(dl):
      6             # x : N x 3 x 512 x 512
      7             x = torch.cat((x, x, x), dim=1).to(device)

NameError: name 'tqdm' is not defined

## === cell 20
df_effnet_pred = pd.DataFrame(
    data=predictions, columns=['patient_overall'] + [f'C{i}' for i in range(1, 8)]
)
df_effnet_pred

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3184829576.py in <cell line: 0>()
      1 df_effnet_pred = pd.DataFrame(
----> 2     data=predictions, columns=['patient_overall'] + [f'C{i}' for i in range(1, 8)]
      3 )
      4 df_effnet_pred

NameError: name 'predictions' is not defined

## === cell 21
df_test_pred = pd.concat([df_test_slices, df_effnet_pred], axis=1).sort_values(['StudyInstanceUID', 'Slice'])
df_test_pred

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2119010916.py in <cell line: 0>()
----> 1 df_test_pred = pd.concat([df_test_slices, df_effnet_pred], axis=1).sort_values(['StudyInstanceUID', 'Slice'])
      2 df_test_pred

NameError: name 'df_test_slices' is not defined

## === cell 22
df_patient_pred = df_test_pred.groupby('StudyInstanceUID').apply(lambda df: df.max())
df_patient_pred

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/53170704.py in <cell line: 0>()
----> 1 df_patient_pred = df_test_pred.groupby('StudyInstanceUID').apply(lambda df: df.max())
      2 df_patient_pred

NameError: name 'df_test_pred' is not defined

## === cell 23
df_patient_pred["patient_overall"] = df_patient_pred[[f'C{i}' for i in range(1, 8)]].max(axis=1)
df_patient_pred = df_patient_pred[['patient_overall'] + [f'C{i}' for i in range(1, 8)]]
df_patient_pred

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2289433139.py in <cell line: 0>()
----> 1 df_patient_pred["patient_overall"] = df_patient_pred[[f'C{i}' for i in range(1, 8)]].max(axis=1)
      2 df_patient_pred = df_patient_pred[['patient_overall'] + [f'C{i}' for i in range(1, 8)]]
      3 df_patient_pred

NameError: name 'df_patient_pred' is not defined

## === cell 24
def cal_loss(prob, label):
    
    pos_weight = np.array([14, 2, 2, 2, 2, 2, 2, 2])
    neg_weight = np.array([7, 1, 1, 1, 1, 1, 1, 1])
    
    score = pos_weight * label * np.log(prob) + neg_weight * (1 - label) * np.log(1 - prob)
    
    weight_total = pos_weight * label + neg_weight * (1 - label)
    
    return -score.sum(axis=1) / weight_total.sum(axis=1)

## === cell 28
df_test = pd.read_csv(f'../input/rsna-2022-cervical-spine-fracture-detection/test.csv')

if df_test.iloc[0].row_id == '1.2.826.0.1.3680043.10197_C1':
    df_test = pd.DataFrame({
        "row_id": ['1.2.826.0.1.3680043.22327_C1', '1.2.826.0.1.3680043.25399_C1', '1.2.826.0.1.3680043.5876_patient_overall'],
        "StudyInstanceUID": ['1.2.826.0.1.3680043.22327', '1.2.826.0.1.3680043.25399', '1.2.826.0.1.3680043.5876'],
        "prediction_type": ["C1", "C1", "patient_overall"]}
    )

df_test

## === cell 29
df_sub = df_test.copy()
df_sub = df_sub.set_index('StudyInstanceUID').join(df_patient_pred)
df_sub['fractured'] = df_sub.apply(lambda r: r[r.prediction_type], axis=1)
df_sub

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3445447858.py in <cell line: 0>()
      1 df_sub = df_test.copy()
----> 2 df_sub = df_sub.set_index('StudyInstanceUID').join(df_patient_pred)
      3 df_sub['fractured'] = df_sub.apply(lambda r: r[r.prediction_type], axis=1)
      4 df_sub

NameError: name 'df_patient_pred' is not defined

## === cell 30
df_sub[['row_id', 'fractured']].to_csv('submission.csv', index=False)

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/323076695.py in <cell line: 0>()
----> 1 df_sub[['row_id', 'fractured']].to_csv('submission.csv', index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['fractured'] not in index"
