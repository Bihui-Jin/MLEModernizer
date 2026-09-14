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

0.7176385764508356

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
try:
    import pylibjpeg
except:
    !mkdir -p /root/.cache/torch/hub/checkpoints/

    !pip install /kaggle/input/rsna-2022-whl/{pydicom-2.3.0-py3-none-any.whl,pylibjpeg-1.4.0-py3-none-any.whl,python_gdcm-3.0.15-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl}
    !pip install /kaggle/input/rsna-2022-whl/{torch-1.12.1-cp37-cp37m-manylinux1_x86_64.whl,torchvision-0.13.1-cp37-cp37m-manylinux1_x86_64.whl}

## === cell 2
import numpy as np
import pandas as pd
import pydicom

import cv2
import os
from tqdm import tqdm
import glob
import pickle
from albumentations import *
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models.convnext import convnext_tiny, convnext_small, convnext_base
from torchvision.models.efficientnet import efficientnet_v2_l

from torch.utils.data import Dataset, DataLoader
import torch
import sys
import time


## === cell 4
config = {'seq_len': 150,
          'feature_size': 1024,
          'lstm_size': 128,
          'target_size': 368,
          'crop_size': 320,
          'num_classes':7,
          'batch_size_image_level': 32,
          'batch_size_patient_level': 8
         }

## === cell 5
def load_df_test():
    df_test = pd.read_csv(f'../input/rsna-2022-cervical-spine-fracture-detection/test.csv')

    if df_test.iloc[0].row_id == '1.2.826.0.1.3680043.10197_C1':
        df_test = pd.DataFrame({
            "row_id": ['1.2.826.0.1.3680043.22327_C1', '1.2.826.0.1.3680043.25399_C1', '1.2.826.0.1.3680043.5876_C1'],
            "StudyInstanceUID": ['1.2.826.0.1.3680043.22327', '1.2.826.0.1.3680043.25399', '1.2.826.0.1.3680043.5876'],
            "prediction_type": ["C1", "C1", "patient_overall"]}
        )
    return df_test

## === cell 6
test_df = load_df_test()
TEST_PATH = '../input/rsna-2022-cervical-spine-fracture-detection/test_images'
study_id_list = list(test_df.StudyInstanceUID.unique()) #uids
study_id_list

## === cell 7
selected_image_dict = {}
for uid in study_id_list:
    dicom_files = glob.glob(os.path.join(f'{TEST_PATH}/{uid}', '*.dcm'))
    middle_slice = int(len(dicom_files)/2)
    num_left_images = num_right_images = int(0.15*len(dicom_files)) # select 15% to the left, 15% to the right
    selected_image_dict[uid] = list(np.arange(middle_slice-num_left_images, middle_slice, 1)) +\
                               list(np.arange(middle_slice+1, middle_slice+num_right_images+1,1))

## === cell 10
def window(data, WL=400, WW=1800):
    data.PhotometricInterpretation = 'YBR_FULL'
    slope     = data.RescaleSlope
    intercept = data.RescaleIntercept
    img = data.pixel_array
    img = (img*slope +intercept)
    upper, lower = WL+WW//2, WL-WW//2
    X = np.clip(img.copy(), lower, upper)
    X = X - np.min(X)
    X = X / np.max(X)
    X = (X*255.0).astype('uint8')
    return X

def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1)) # since numpy array has [H,W,C] -> we want [C,H,W] for torch tensor
    return torch.from_numpy(img.astype(dtype, copy=False))

## === cell 12
mean=np.array([0.456, 0.456, 0.456])
std=np.array([0.224, 0.224, 0.224])

class CSFImageDataset(Dataset):
    def __init__(self, uid, image_list, target_size, crop_size):
        self.uid = uid
        self.image_list  = image_list
        self.target_size = target_size
        self.crop_size   = crop_size
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        
        PATH = os.path.join(TEST_PATH, self.uid)
        data_list  = [pydicom.dcmread(f'{PATH}/{str(self.image_list[index]-1)}.dcm'), \
                      pydicom.dcmread(f'{PATH}/{str(self.image_list[index])}.dcm'), \
                      pydicom.dcmread(f'{PATH}/{str(self.image_list[index]+1)}.dcm')]
                
        imgs  = [window(data) for data in data_list]
        
        stacked_img = np.stack(imgs, axis=-1)
        stacked_img = cv2.resize(stacked_img, (self.target_size, self.target_size))
        
        X = img2tensor((stacked_img/255.0 - mean)/std)
        
        return X

## === cell 13
class CSFInstanceDataset(Dataset):
    def __init__(self, feature_array_dict, study_id_list, seq_len):
        self.feature_array_dict = feature_array_dict
        self.study_id_list = study_id_list
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.study_id_list)
    
    def __getitem__(self, index):
        uid = self.study_id_list[index]
        feature_array = self.feature_array_dict[uid]
        if len(feature_array) > self.seq_len: 
            x = cv2.resize(feature_array, (feature_array.shape[1], self.seq_len), interpolation = cv2.INTER_LINEAR)
        else:
            x = np.pad(feature_array, pad_width=[(0,self.seq_len-feature_array.shape[0]), (0,0)], constant_values=0) 
            
        X = torch.tensor(x, dtype=torch.float32) #(seq_len,1024)
        return X, uid

## === cell 15
class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self):
        super(ConvNextCNN_B_Feature, self).__init__()
        m = convnext_base() #extract the output layer
        in_features = m.classifier[-1].in_features
        self.features = m.features
        self.avgpool = nn.AdaptiveAvgPool2d(1) 
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=in_features, out_features=7)
        
    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = self.drop(out)
        feature = out.view(x.size(0), -1) #layer before the fc
        
        out = self.fc(feature)
        
        return feature, out

## === cell 16
class CSFNet(nn.Module):
    def __init__(self, input_len, lstm_size):
        super().__init__()
        self.lstm1 = nn.GRU(input_len, lstm_size, bidirectional=True, batch_first=True)
        self.last_linear = nn.Linear(lstm_size*2, 1) #(*4)
        
    def forward(self, x):
        h_lstm1, _ = self.lstm1(x)

        max_pool, _ = torch.max(h_lstm1, 1)

        logits = self.last_linear(max_pool)
        return logits

## === cell 17
lv1_model = ConvNextCNN_B_Feature()
lv1_model.load_state_dict(torch.load('../input/cnn-lstm-oct-24/run_1/run_1/model_1.pth'))
lv1_model = lv1_model.cuda()
lv1_model.eval()

lv2_model = CSFNet(input_len=config['feature_size'], lstm_size=config['lstm_size'])
lv2_model.load_state_dict(torch.load('../input/cnn-lstm-oct-21/run_5b/run_5b/model_lstm_3.pth'))
lv2_model = lv2_model.cuda()
lv2_model.eval()

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1277514498.py in <cell line: 0>()
      2 # Image level
      3 lv1_model = ConvNextCNN_B_Feature()
----> 4 lv1_model.load_state_dict(torch.load('../input/cnn-lstm-oct-24/run_1/run_1/model_1.pth'))
      5 lv1_model = lv1_model.cuda()
      6 lv1_model.eval()

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '../input/cnn-lstm-oct-24/run_1/run_1/model_1.pth'

## === cell 20
submission_dict = {'row_id':[], 'fractured':[]}
feature_array_dict = {}
for uid in study_id_list:
    image_list = selected_image_dict[uid]
    dataset = CSFImageDataset(uid=uid, image_list=image_list,\
                              target_size=config['target_size'], \
                              crop_size=config['crop_size'])
    generator = DataLoader(dataset, batch_size=config['batch_size_image_level'],\
                           shuffle=False, pin_memory=True, drop_last=False)
    
    preds_uid = []
    feature_array = np.zeros((len(dataset), config['feature_size']), dtype=np.float32)
    for i, images in tqdm(enumerate(generator), total=len(generator)):
        with torch.no_grad():
            start = i*config['batch_size_image_level']
            end = start+config['batch_size_image_level']
            
            if i == len(generator) - 1: # last batch
                end = len(generator.dataset)
            
            images = images.cuda()
            
            features, preds = lv1_model(images)
                
            feature_array[start:end] = np.squeeze(features.cpu().data.numpy())
            preds = preds.sigmoid().detach()
            preds_uid.append(preds)
            
    feature_array_dict[uid] = feature_array
    
    mean_preds = torch.mean(torch.cat(preds_uid), dim=0).cpu().data.numpy()
    
    
    submission_dict['row_id'].append(f'{uid}_C1')
    submission_dict['fractured'].append(mean_preds[0])
        
    submission_dict['row_id'].append(f'{uid}_C2')
    submission_dict['fractured'].append(mean_preds[1])
        
    submission_dict['row_id'].append(f'{uid}_C3')
    submission_dict['fractured'].append(mean_preds[2])
        
    submission_dict['row_id'].append(f'{uid}_C4')
    submission_dict['fractured'].append(mean_preds[3])
        
    submission_dict['row_id'].append(f'{uid}_C5')
    submission_dict['fractured'].append(mean_preds[4])
        
    submission_dict['row_id'].append(f'{uid}_C6')
    submission_dict['fractured'].append(mean_preds[5])
        
    submission_dict['row_id'].append(f'{uid}_C7')
    submission_dict['fractured'].append(mean_preds[6])
   

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3717889065.py in <cell line: 0>()
     12     feature_array = np.zeros((len(dataset), config['feature_size']), dtype=np.float32)
     13     #image level prediction
---> 14     for i, images in tqdm(enumerate(generator), total=len(generator)):
     15         with torch.no_grad():
     16             start = i*config['batch_size_image_level']

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_11/1261604447.py in __getitem__(self, index)
     20         # print(data_list)
     21 
---> 22         imgs  = [window(data) for data in data_list]
     23 
     24         stacked_img = np.stack(imgs, axis=-1)

/tmp/ipykernel_11/1261604447.py in <listcomp>(.0)
     20         # print(data_list)
     21 
---> 22         imgs  = [window(data) for data in data_list]
     23 
     24         stacked_img = np.stack(imgs, axis=-1)

/tmp/ipykernel_11/1789851503.py in window(data, WL, WW)
      4     slope     = data.RescaleSlope
      5     intercept = data.RescaleIntercept
----> 6     img = data.pixel_array
      7     img = (img*slope +intercept)
      8     upper, lower = WL+WW//2, WL-WW//2

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in pixel_array(self)
   2191             that iterates through the image frames.
   2192         """
-> 2193         self.convert_pixel_data()
   2194         return cast("numpy.ndarray", self._pixel_array)
   2195 

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in convert_pixel_data(self, handler_name)
   1724             # Use 'pydicom.pixels' backend
   1725             opts["decoding_plugin"] = name
-> 1726             self._pixel_array = pixel_array(self, **opts)
   1727             self._pixel_id = get_image_pixel_ids(self)
   1728         else:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py in pixel_array(src, ds_out, specific_tags, index, raw, decoding_plugin, **kwargs)
   1428 
   1429         opts = as_pixel_options(ds, **kwargs)
-> 1430         return decoder.as_array(
   1431             ds,
   1432             index=index,

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in as_array(self, src, index, validate, raw, decoding_plugin, **kwargs)
    980             cast(
    981                 dict[str, "DecodeFunction"],
--> 982                 self._validate_plugins(decoding_plugin),
    983             ),
    984         )

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py in _validate_plugins(self, plugin)
    255         missing = "\n".join([f"\t{s}" for s in self.missing_dependencies])
    256         if self._decoder:
--> 257             raise RuntimeError(
    258                 f"Unable to decompress '{self.UID.name}' pixel data because all "
    259                 f"plugins are missing dependencies:\n{missing}"

RuntimeError: Unable to decompress 'JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])' pixel data because all plugins are missing dependencies:
	gdcm - requires gdcm>=3.0.10
	pylibjpeg - requires pylibjpeg>=2.0 and pylibjpeg-libjpeg>=2.1

## === cell 22
dataset = CSFInstanceDataset(feature_array_dict=feature_array_dict,
                             study_id_list=study_id_list,
                             seq_len=config['seq_len']
                            )
generator = DataLoader(dataset=dataset,
                       batch_size=config['batch_size_patient_level'],
                       shuffle=False,
                       pin_memory=True)

for features, list_uid in tqdm(generator, total=len(generator)):
    with torch.no_grad():
        features = features.cuda()
        preds = lv2_model(features)
        preds = np.squeeze(preds.sigmoid().cpu().data.numpy())
    
    for j in range(len(features)):# n features extracted from n patients from stage 1
        submission_dict['row_id'].append(f'{list_uid[j]}_patient_overall')
        submission_dict['fractured'].append(preds[j])

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/36040992.py in <cell line: 0>()
      9                        pin_memory=True)
     10 
---> 11 for features, list_uid in tqdm(generator, total=len(generator)):
     12     with torch.no_grad():
     13         features = features.cuda()

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_11/741485845.py in __getitem__(self, index)
     10     def __getitem__(self, index):
     11         uid = self.study_id_list[index]
---> 12         feature_array = self.feature_array_dict[uid]
     13         # if a study has more slices than seq_len
     14         # resize features

KeyError: '1.2.826.0.1.3680043.6200'

## === cell 24
sub_df = pd.DataFrame.from_dict(submission_dict)
sub_df = sub_df.sort_values(by='row_id', axis=0, ascending=True).reset_index(drop=True)
sub_df

## === cell 25
sub_df.to_csv('submission.csv', index=False)
