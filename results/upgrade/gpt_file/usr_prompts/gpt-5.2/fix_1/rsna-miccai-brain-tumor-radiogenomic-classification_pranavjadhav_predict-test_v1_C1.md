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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

0.5538840727519972

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tqdm import tqdm
import os
import glob
import numpy as np
import random
import math
import time
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import matplotlib.pyplot as plt

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class DataLoader(keras.utils.Sequence):
    def __init__(self,base_dir='/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test',\
                mods=['FLAIR']):
        self.batch_size = 1 
        self.base_dir = base_dir
        self.pat_ids = sorted(glob.glob(os.path.join(base_dir, '*')))
        self.modalities = mods
        print('PAT IDS:',len(self.pat_ids),' | Modalities:',self.modalities)

    def load_dicom_image(self,path):
        dicom = pydicom.read_file(path)
        data = apply_voi_lut(dicom.pixel_array, dicom)
        if dicom.PhotometricInterpretation == "MONOCHROME1":
            data = np.amax(data) - data
        data = data - np.min(data)
        data = data / np.max(data)
        data = (data * 255).astype(np.uint8)[:,:,np.newaxis]
        data = np.repeat(data,3,2)
        return data
        
    def __getitem__(self,index):
        batch_patids = self.pat_ids[index:index+self.batch_size]
        all_images = {}
        for K in self.modalities:
            all_images[K] = {'images':[],'ids':[]}
        for patid in batch_patids:
            for MOD in all_images.keys():
                dicom_pngs = []
                dicom_files = glob.glob(os.path.join(patid, MOD, '*.dcm'))
                dset = tqdm(enumerate(dicom_files), total=len(dicom_files),position=0, leave=True)
                dset.set_description(f'{patid}|{MOD}')
                for i,dicom_path in dset:
                    try:
                        png = self.load_dicom_image(dicom_path)
                        dicom_pngs.append(png)
                    except:
                        continue
                if len(dicom_pngs)!=0:
                    all_images[MOD]['images'].append(dicom_pngs)
                    all_images[MOD]['ids'].append(patid.replace('\\','/').split('/')[-1])
                    
                dicom_pngs = None
                
        return all_images
   
    def __len__(self):
        return int(len(self.pat_ids)/self.batch_size)

## === cell 2
class StageOne:
    def __init__(self,modelpath='',height=256,width=256,score_min_thresh=0.5):
        self.model = load_model(modelpath)
        self.score_min_thresh = score_min_thresh
        self.height = height
        self.width = width
        self.offset_perc = 0.1
    
    def infer(self,image_batch,filter_batchsize=16):
        filtered_images = {}
        for K in image_batch.keys():
            filtered_images[K] = {'images':[],'ids':[]}
            for imagesbatch, patid in zip(image_batch[K]['images'],image_batch[K]['ids']):
                div,mod = divmod(len(imagesbatch),filter_batchsize)
                if mod!=0:
                    div+=1
                dset = tqdm(range(0,len(imagesbatch),filter_batchsize),total=div,position=0, leave=True)
                dset.set_description(f'{patid}|Filtering')
                filtered_batch_images = []
                for i in dset:
                    org_batchimgs = imagesbatch[i:i+filter_batchsize]
                    batchimgs = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in org_batchimgs]
                    batchimgs = np.array([cv2.resize(img,(self.width,self.height))/255. for img in batchimgs])
                    out = self.model.predict(batchimgs)
                    maxindexes = np.argmax(out,axis=1)
                    for j in range(len(maxindexes)):
                        if maxindexes[j] == 1 and out[j][maxindexes[j]] >= self.score_min_thresh:
                            filtered_batch_images.append(org_batchimgs[j])
                            
                if len(filtered_batch_images)==0:
                    offset = math.ceil(len(imagesbatch)*self.offset_perc)
                    filtered_batch_images = imagesbatch[offset:-offset]
                    
                    
                filtered_images[K]['images'].append(filtered_batch_images)
                filtered_images[K]['ids'].append(patid)
                filtered_batch_images = None
                dset = None
                
            return filtered_images
              

## === cell 3
class StageTwo:
    def __init__(self,modelpath='',height=224,width=224):
        self.model = load_model(modelpath)
        self.height = height
        self.width = width
    
    def preprocess(self,image):
        image = cv2.resize(image,(self.width,self.height))
        return image
    
    def infer(self,imagebatch):
        out = self.model.predict(imagebatch)
        return out

## === cell 4
mods = ['FLAIR']
generator = DataLoader(base_dir='/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test')
stage_one = StageOne(modelpath='/kaggle/input/models/FINAL_MODELALL_acc0.9825_ep26.h5')
stage_two = StageTwo(modelpath='/kaggle/input/models/3d_image_classification.hdf5')
dset = tqdm(enumerate(generator),total=len(generator),position=0, leave=True)
dset.set_description('Loading_test')
all_results = {}
for i,sample in dset:
    filtered_images = stage_one.infer(sample)
    for K in filtered_images.keys():
        for batchimg,patid in zip(filtered_images[K]['images'],filtered_images[K]['ids']):
            batchimg = [stage_two.preprocess(img) for img in batchimg]
            batchimg = np.transpose(batchimg, (1,2,0,3))[None,:,:,:,:]

            out = stage_two.infer(batchimg)
            maxindexes = int(np.argmax(out,axis=1)[0])

            patidkey = str(patid).zfill(5)
            if patidkey not in all_results.keys():
                all_results[patidkey] = [maxindexes]
            else:
                all_results[patidkey].append(maxindexes)

allpatids = []
allpreds = []
for K in all_results.keys():
    one_count = all_results[K].count(1)
    zero_count = all_results[K].count(0)
    allpatids.append(K)
    if one_count > zero_count:
        allpreds.append(1)
    else:
        allpreds.append(0)
    
        
    
outcsv = 'submission.csv'
df = pd.DataFrame({'BraTS21ID':allpatids,'MGMT_value':allpreds})
df.to_csv(outcsv,index=False)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4248825711.py in <cell line: 0>()
      1 mods = ['FLAIR']
      2 generator = DataLoader(base_dir='/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test')
----> 3 stage_one = StageOne(modelpath='/kaggle/input/models/FINAL_MODELALL_acc0.9825_ep26.h5')
      4 stage_two = StageTwo(modelpath='/kaggle/input/models/3d_image_classification.hdf5')
      5 dset = tqdm(enumerate(generator),total=len(generator),position=0, leave=True)

/tmp/ipykernel_11/913215934.py in __init__(self, modelpath, height, width, score_min_thresh)
      1 class StageOne:
      2     def __init__(self,modelpath='',height=256,width=256,score_min_thresh=0.5):
----> 3         self.model = load_model(modelpath)
      4         self.score_min_thresh = score_min_thresh
      5         self.height = height

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/kaggle/input/models/FINAL_MODELALL_acc0.9825_ep26.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 5
df

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1396537375.py in <cell line: 0>()
----> 1 df

NameError: name 'df' is not defined
