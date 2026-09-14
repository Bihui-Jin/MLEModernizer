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
Detect breast cancer in mammograms.

## Metric
[Probabilistic F1 score](https://aclanthology.org/2020.eval4nlp-1.9.pdf) (pF1). This extension of the traditional F score accepts probabilities instead of binary classifications. 

With pX as the probabilistic version of X:

$$
pF_1 = 2 \frac{pPrecision \cdot pRecall}{pPrecision + pRecall}
$$

where:

$$
pPrecision = \frac{pTP}{pTP + pFP}
$$

$$
pRecall = \frac{pTP}{TP + FN}
$$

## Submission Format
For each `prediction_id`, you should predict the likelihood of cancer in the corresponding `cancer` column. The submission file should have the following format:

```
prediction_id,cancer
0-L,0
0-R,0.5
0-R,0.5
1-L,1
...
# Dataset

**[train/test]_images/[patient_id]/[image_id].dcm** The mammograms, in dicom format. You can expect roughly 8,000 patients in the hidden test set. There are usually but not always 4 images per patient. Note that many of the images use the jpeg 2000 format which may you may need special libraries to load.

**sample_submission.csv** A valid sample submission.

**[train/test].csv** Metadata for each patient and image. Only the first few rows of the test set are available for download.

- `site_id` - ID code for the source hospital.
- `patient_id` - ID code for the patient.
- `image_id` - ID code for the image.
- `laterality` - Whether the image is of the left or right breast.
- `view` - The orientation of the image. The default for a screening exam is to capture two views per breast.
- `age` - The patient's age in years.
- `implant` - Whether or not the patient had breast implants. Site 1 only provides breast implant information at the patient level, not at the breast level.
- `density` - A rating for how dense the breast tissue is, with A being the least dense and D being the most dense. Extremely dense tissue can make diagnosis more difficult. Only provided for train.
- `machine_id` - An ID code for the imaging device.
- `cancer` - Whether or not the breast was positive for malignant cancer. The target value. Only provided for train.
- `biopsy` - Whether or not a follow-up biopsy was performed on the breast. Only provided for train.
- `invasive` - If the breast is positive for cancer, whether or not the cancer proved to be invasive. Only provided for train.
- `BIRADS` - 0 if the breast required follow-up, 1 if the breast was rated as negative for cancer, and 2 if the breast was rated as normal. Only provided for train.
- `prediction_id` - The ID for the matching submission row. Multiple images will share the same prediction ID. Test only.
- `difficult_negative_case` - True if the case was unusually difficult. Only provided for train.

# 2. Python version

3.11

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        input/
            description.md (190 lines)
            sample_submission.csv (2385 lines)
            sample_submission.csv.zip (6.5 kB)
            test.csv (5475 lines)
            test.csv.zip (60.6 kB)
            test.zip (160 Bytes)
            test_images.zip (29.1 GB)
            train.csv (49233 lines)
            train.csv.zip (513.7 kB)
            train.zip (162 Bytes)
            train_images.zip (260.7 GB)
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
            test_images/
                10116/
                    1470873094.dcm (8.5 MB)
                    472095321.dcm (5.6 MB)
                    ... and 2 other files
                10130/
                    1013166704.dcm (9.7 MB)
                    1165309236.dcm (8.7 MB)
                    ... and 5 other files
                ... and 1191 other folders
            train_images/
                10006/
                    1459541791.dcm (4.4 MB)
                    1864590858.dcm (4.0 MB)
                    ... and 2 other files
                10011/
                    1031443799.dcm (2.1 MB)
                    220375232.dcm (1.7 MB)
                    ... and 2 other files
                ... and 10720 other folders
        working/
            rsna-breast-cancer-detection/
                description.md (190 lines)
                sample_submission.csv (2385 lines)
                ... and 9 other files
                rsna-breast-cancer-detection/
                test_images/
                    10116/
                        1470873094.dcm (8.5 MB)
                        472095321.dcm (5.6 MB)
                        ... and 2 other files
                    10130/
                        1013166704.dcm (9.7 MB)
                        1165309236.dcm (8.7 MB)
                        ... and 5 other files
                    ... and 1191 other folders
                train_images/
                    10006/
                        1459541791.dcm (4.4 MB)
                        1864590858.dcm (4.0 MB)
                        ... and 2 other files
                    10011/
                        1031443799.dcm (2.1 MB)
                        220375232.dcm (1.7 MB)
                        ... and 2 other files
                    ... and 10720 other folders
```

-> data/rsna-breast-cancer-detection/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/rsna-breast-cancer-detection/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/rsna-breast-cancer-detection/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> data/sample_submission.csv has 2384 rows and 2 columns.
The columns are: prediction_id, cancer

-> data/test.csv has 5474 rows and 9 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, implant, machine_id, prediction_id

-> data/train.csv has 49232 rows and 14 columns.
The columns are: site_id, patient_id, image_id, laterality, view, age, cancer, biopsy, invasive, BIRADS, implant, density, machine_id, difficult_negative_case

-> (stopped after 10 files for performance)

# 5. Target score

0.4517647058823529

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
try: 
    import dicomsdl
    
except:
    !pip install /kaggle/input/rsna-2022-whl/pylibjpeg-1.4.0-py3-none-any.whl
    !pip install /kaggle/input/rsna-2022-whl/python_gdcm-3.0.15-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
    !pip install /kaggle/input/rsna-breast-mammography-00/dicomsdl-0.109.1-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl
    !cp /kaggle/input/easy-load-the-image-with-nvjpeg2000/nvjpeg2k.so ./
    
print('install ok')

## === cell 1
import sys
sys.path.append('/kaggle/input/rsna-breast-mammography-00')

from dicom_reader import *
from preprocess import *


import pandas as pd
import numpy as np
import cv2
from timeit import default_timer as timer
from tqdm.notebook import tqdm
from joblib import Parallel, delayed
from glob import glob
from sklearn import metrics
import gc
 

import matplotlib
import matplotlib.pyplot as plt

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp
print( 'torch.cuda.device_count() = %d'%torch.cuda.device_count())
print( 'torch.cuda.get_device_properties() = %s' % str(torch.cuda.get_device_properties(0))[21:])

import timm
print('timm',timm.__version__)

import nvjpeg2k
print('import ok!')

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3542457327.py in <cell line: 0>()
      2 sys.path.append('/kaggle/input/rsna-breast-mammography-00')
      3 
----> 4 from dicom_reader import *
      5 from preprocess import *
      6 

ModuleNotFoundError: No module named 'dicom_reader'

## === cell 2
mode = [ 
    'submit',  #submit #local
    
]

convert_height = 1536
image_height = 1536
image_width  = 960


if 'local' in mode:
    csv_file = '/kaggle/input/rsna-breast-mammography-00/valid_df.fold0.ver02.csv'
    dcm_dir  = '/kaggle/input/rsna-breast-cancer-detection/train_images'

if 'submit' in mode:
    csv_file = '/kaggle/input/rsna-breast-cancer-detection/test.csv'
    dcm_dir  = '/kaggle/input/rsna-breast-cancer-detection/test_images'


test_df = pd.read_csv(csv_file)
machine_id_to_transfer = make_transfer_syntax_uid(test_df, dcm_dir)
test_df.loc[:, 'i'] = np.arange(len(test_df))
test_df.loc[:, 'TransferSyntaxUID'] = test_df.machine_id.map(machine_id_to_transfer)
if 'local' in mode:
    test_df.loc[:, 'prediction_id'] = test_df.patient_id.astype(str) + '_' + test_df.laterality
    if 'subset' in mode: 
        test_id = [
            65, 127, 152, 272, 282, 308, 477, 505, 2989, 3542, 7780, 9014, 11094, 11937,
            30, 36, 90, 111, 122, 158, 204, 289, 299, 399, 425, 454, 826, 1703, 1759, 2346, 3021, 4340, 4824, 5059,
            5769, 6654, 6658, 7053, 7493, 14292
        ]
        test_df = test_df[test_df.patient_id.isin(test_id)].reset_index(drop=True)

print('test_df', test_df.shape)
print(test_df)
print('')

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3181112142.py in <cell line: 0>()
     21 
     22 
---> 23 test_df = pd.read_csv(csv_file)
     24 machine_id_to_transfer = make_transfer_syntax_uid(test_df, dcm_dir)
     25 test_df.loc[:, 'i'] = np.arange(len(test_df))

NameError: name 'pd' is not defined

## === cell 3
def make_debug_submission():
    submit_df = pd.DataFrame({
        'prediction_id': test_df.prediction_id,
        'cancer': 0,
    })

    submit_df = submit_df.groupby('prediction_id').mean()  
    submit_df.to_csv('submission.csv', index=True)
    print('submit_df', submit_df)
    print('')
    

## === cell 4
png_dir = '/kaggle/tmp/~png' 

def run_dicom_to_png():
    patient_id = test_df['patient_id'].unique()
    for i in patient_id:
        os.makedirs(f'{png_dir}/{i}', exist_ok=True)

    j2k_df = test_df[test_df.TransferSyntaxUID == '1.2.840.10008.1.2.4.90'].reset_index(drop=True)
    non_j2k_df = test_df[test_df.TransferSyntaxUID != '1.2.840.10008.1.2.4.90'].reset_index(drop=True)

    print('process_j2k()')
    print('j2k_df:', len(j2k_df))
    start_timer = timer()
    process_j2k(j2k_df, dcm_dir, png_dir, convert_height)
    print(time_to_str(timer() - start_timer, 'sec'))

    print('process_non_j2k()')
    print('non_j2k_df:', len(non_j2k_df))
    start_timer = timer()
    process_non_j2k(non_j2k_df, dcm_dir, png_dir, convert_height, n_jobs=2)  
    print(time_to_str(timer() - start_timer, 'sec'))


if not 'skip-dicom-to-png' in mode:
    run_dicom_to_png()
else:
    png_dir = f'/home/titanx/hengck/share1/kaggle/2022/rsna-breast-mammography/data/my-norm/my-lut-8bit/{convert_height}-aspect'


if 'local' in mode: #check 
    m = cv2.imread(f'{png_dir}/21/3021/1557228175.png',cv2.IMREAD_GRAYSCALE)
    print(m.shape)
    print(m)
    plt.imshow(m,cmap='bone')
    plt.show() 

print('glob', len(glob(f'{png_dir}/**/*.png', recursive=True)))
print('gc.collect', gc.collect())
print('')

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/272596067.py in <cell line: 0>()
     25 
     26 if not 'skip-dicom-to-png' in mode:
---> 27     run_dicom_to_png()
     28 else:
     29     png_dir = f'/home/titanx/hengck/share1/kaggle/2022/rsna-breast-mammography/data/my-norm/my-lut-8bit/{convert_height}-aspect'

/tmp/ipykernel_11/272596067.py in run_dicom_to_png()
      3 
      4 def run_dicom_to_png():
----> 5     patient_id = test_df['patient_id'].unique()
      6     for i in patient_id:
      7         os.makedirs(f'{png_dir}/{i}', exist_ok=True)

NameError: name 'test_df' is not defined

## === cell 5
def run_add_breast_boc(test_df):
    test_df = test_df.drop([col for col in ['pad_breast_box','max_pad_breast_shape'] if col in test_df.columns], axis=1)

    class PreprocessDataset(Dataset):
        def __init__(self, df):
            self.df = df
            self.length = len(df)
            self.image_size = 224

        def __len__(self):
            return self.length

        def __getitem__(self, index):
            d = self.df.iloc[index]

            m = cv2.imread(f'{png_dir}/{d.machine_id}/{d.patient_id}/{d.image_id}.png', cv2.IMREAD_GRAYSCALE)
            h, w = m.shape

            s = self.image_size / h
            h, w = int(s * h), int(s * w)
            m = cv2.resize(m, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            y = (self.image_size - h) // 2
            x = (self.image_size - w) // 2
            rect = (x, y, x + w, y + h)

            image = np.zeros((self.image_size, self.image_size), np.uint8)
            image[y:y + h, x:x + w] = m

            r = {}
            r['index'] = index
            r['d'] = d
            r['rect'] = rect
            r['image'] = torch.from_numpy(image)
            return r

    def proprocess_collate(batch):
        d = {}
        key = batch[0].keys()
        for k in key:
            v = [b[k] for b in batch]
            d[k] = v
        d['image'] = torch.stack(d['image'], 0).unsqueeze(1)
        return d

    dataset = PreprocessDataset(test_df)
    loader = DataLoader(
        dataset,
        sampler = SequentialSampler(dataset),
        batch_size  = 32,
        drop_last   = False,
        num_workers = 2,
        pin_memory  = False,
        collate_fn = proprocess_collate,
    )

    checkpoint = \
        f'/kaggle/input/rsna-breast-mammography-00/resnet34d-mask-kaggle-005-00000387.model.pth'  # 00000774
    net = PreprocessNet()
    f = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    net.load_state_dict(f['state_dict'],strict=True)
    net.cuda()
    net.eval()

    box = []
    laterality = []

    start_timer = timer()
    for t, batch in enumerate(loader):
        batch_size = len(batch['index'])
        for k in ['image']: batch[k] = batch[k].cuda()
        batch['image'] = batch['image'].float() / 255

        with torch.no_grad():
            with amp.autocast(enabled = True):
                output = net(batch)

        output = post_process(batch, output)
        box.extend(output['box'])
        laterality.extend(output['laterality'])

        if t==0:
            overlay =[]
            for b in range(4):
                
                o = draw_preprocess_overlay(
                    output['image'][b],
                    output['mask'][b],
                    output['box'][b],
                    output['laterality'][b],
                )
                overlay.append(o[...,::-1])

            plt.imshow(np.hstack(overlay))
            plt.show()
 

        print(f'\r add_breast_box(): {t}/{len(loader)} { time_to_str(timer() - start_timer, "sec")}', end='',
              flush=True)
    test_df.loc[:,'pad_breast_box']=box
    test_df.loc[:,'old_laterality']=test_df.laterality.values
    test_df.loc[:,'laterality']=laterality

    def aggr_max_pad_breast_shape(df):
        box = []
        for t, d in df.iterrows():
            b = d.pad_breast_box
            box.append(b)
        box = np.array(box)

        x0, y0, x1, y1 = box.T
        w = x1 - x0
        h = y1 - y0
        max_pad_breast_box_h = h.max()
        max_pad_breast_box_w = w.max()
        return [max_pad_breast_box_h, max_pad_breast_box_w]

    gb = test_df.groupby('patient_id').apply(aggr_max_pad_breast_shape).to_frame('max_pad_breast_shape')
    gb = gb.reset_index(drop=False)
    test_df = test_df.merge(gb, on=('patient_id'))
    return test_df

if not 'skip-add-breast-box' in mode:
    test_df = run_add_breast_boc(test_df)
    
print(test_df.iloc[0], '\n')
print('gc.collect', gc.collect())
print('')



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3523479262.py in <cell line: 0>()
    124 
    125 if not 'skip-add-breast-box' in mode:
--> 126     test_df = run_add_breast_boc(test_df)
    127 
    128 print(test_df.iloc[0], '\n')

NameError: name 'test_df' is not defined

## === cell 6
def to_list(x):
    if isinstance(x, list): return x
    if isinstance(x, str): return eval(x)

class RsnaDataset(Dataset):
    def __init__(self, df):
        self.length = len(df)
        self.df = df

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        d = self.df.iloc[index]
        m = cv2.imread(f'{png_dir}/{d.machine_id}/{d.patient_id}/{d.image_id}.png',cv2.IMREAD_GRAYSCALE)
        h, w = m.shape
        
        
        image = np.zeros((image_height, image_width), np.uint8) 
        try: #for degenerate case
            xmin, ymin, xmax, ymax = (np.array(to_list(d.pad_breast_box)) * h).astype(int)
            crop = m[ymin:ymax, xmin:xmax]

            mh, mw = (np.array(to_list(d.max_pad_breast_shape)) * h).astype(int)
            scale = min(image_height/mh,  image_width/mw)
            dsize = (min(image_width, int(scale * crop.shape[1])), min(image_height, int(scale * crop.shape[0])))
            if dsize != (crop.shape[1], crop.shape[0]):
                crop = cv2.resize(crop, dsize=dsize, interpolation=cv2.INTER_LINEAR)
            ch,cw = crop.shape  
            x = (image_width  - cw) // 2
            y = (image_height - ch) // 2
            image[y:y + ch, x:x + cw] = crop

        except:
            crop  = m
            scale = min(image_height / h, image_width / w)
            dsize = (min(image_width, int(scale * crop.shape[1])), min(image_height, int(scale * crop.shape[0])))
            if dsize != (crop.shape[1], crop.shape[0]):
                crop = cv2.resize(crop, dsize=dsize, interpolation=cv2.INTER_LINEAR)
            ch, cw = crop.shape
            x = (image_width  - cw) // 2
            y = (image_height - ch) // 2
            image[y:y + ch, x:x + cw] = crop
            

        r = {}
        r['index'] = index
        r['d'] = d
        r['image'] = torch.from_numpy(image)
        return r

def null_collate(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        d[k] = [b[k] for b in batch]

    d['image'] = torch.stack(d['image']).unsqueeze(1)
    return d


from timm.models.efficientnet import *
from nextvit import *

class NextVitBNet(nn.Module):
    def __init__(self,):
        super(NextVitBNet, self).__init__()
        self.register_buffer('mean', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
        self.encoder = nextvit_base(pretrained=False)
        self.cancer = nn.Linear(1024,1)

    def forward(self, batch):
        x = batch['image']
        batch_size,C,H,W = x.shape
        x = (x - self.mean) / self.std

        e = self.encoder.forward_features(x)
        x = F.adaptive_avg_pool2d(e,1)
        x = torch.flatten(x,1,3)
        cancer = self.cancer(x).reshape(-1)
        cancer = torch.sigmoid(cancer)
        return cancer
    
class EffB4Net(nn.Module):
    def __init__(self,):
        super(EffB4Net, self).__init__()
        self.register_buffer('mean', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.5, 0.5, 0.5]).reshape(1, 3, 1, 1))
        self.encoder = efficientnet_b4(pretrained=False, drop_rate=0, drop_path_rate=0)
        self.cancer = nn.Linear(1792,1)

    def forward(self, batch):
        x = batch['image']
        batch_size,C,H,W = x.shape
        x = (x - self.mean) / self.std

        e = self.encoder.forward_features(x)
        x = F.adaptive_avg_pool2d(e,1)
        x = torch.flatten(x,1,3)
        cancer = self.cancer(x).reshape(-1)
        cancer = torch.sigmoid(cancer)
        return cancer

print('define model ok')

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3409718135.py in <cell line: 0>()
      3     if isinstance(x, str): return eval(x)
      4 
----> 5 class RsnaDataset(Dataset):
      6     def __init__(self, df):
      7         self.length = len(df)

NameError: name 'Dataset' is not defined

## === cell 7
def run_submit():
    threshold = 0.30612
    model = [
        [NextVitBNet,   '/kaggle/input/rsna-breast-mammography-weight-10/nextvit-b-1536-gpu-aug0-01-swa.model.pth'],
       
    ]
    num_net = len(model)

    net = []
    for i in range(num_net):
        Net, checkpoint = model[i]
        n = Net()
        f = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        n.load_state_dict(f['state_dict'], strict=False)  # True
        n.cuda()
        n.eval()
        net.append(n)

    test_dataset = RsnaDataset(test_df)
    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=4,
        drop_last=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=null_collate,
    )

    if 1:
        result = {
            'probability': [[] for i in range(num_net)],
        }
        test_num = 0

        start_timer = timer()
        for t, batch in enumerate(test_loader):
            batch_size = len(batch['index'])
            image0 = batch['image'].cuda().float() / 255

            batch1 = {'image': image0}
            batch2 = {'image': torch.flip(image0, dims=[3, ])}  # TTA

            p = 0
            count = 0
            if 1:
                with torch.no_grad():
                    with amp.autocast(enabled=True):
                        for i in range(num_net):
                            p += net[i](batch1)
                            count += 1

                            p += net[i](batch2)
                            count += 1

                p = p / count
                result['probability'].append(p.float().data.cpu().numpy())
                
                
            test_num += batch_size
            print('\r %8d / %d  %s' % (test_num, len(test_dataset), time_to_str(timer() - start_timer, 'sec')), end='',
                  flush=True)
            torch.cuda.empty_cache()
        print('')
        probability = np.concatenate(result['probability'])
        probability = np.nan_to_num(probability, nan=0, posinf=1, neginf=0)
        np.save('probability.npy', probability)

    probability = np.load('probability.npy')
    print('probability', probability.shape)
    print('')

    submit_df = pd.DataFrame({
        'prediction_id': test_df.prediction_id,
        'cancer': probability,
    })

    submit_df = submit_df.groupby('prediction_id').mean()
    submit_df = submit_df.sort_index()
    predict = submit_df.cancer.values

    submit_df.loc[:, 'cancer'] = (submit_df.cancer.values > threshold).astype(np.float32)
    predict_threshold = submit_df.cancer.values

    submit_df.to_csv('submission.csv', index=True)
    print('submit_df', submit_df)
    print('')

    if 'local' in mode:

        def get_f1score(probability, truth, threshold):

            if threshold is None:
                predict = [probability]
            else:
                predict = [
                    (probability > t).astype(np.float32) for t in threshold
                ]

            f1score = []
            for p in predict:
                tp = ((p >= 0.5) & (truth >= 0.5)).sum()
                fp = ((p >= 0.5) & (truth < 0.5)).sum()
                fn = ((p < 0.5) & (truth >= 0.5)).sum()

                recall = tp / (tp + fn + 1e-3)
                precision = tp / (tp + fp + 1e-3)
                f1 = 2 * recall * precision / (recall + precision + 1e-3)
                f1score.append(f1)
            f1score = np.array(f1score)
            return f1score

        truth_df = test_df[['prediction_id', 'cancer']].groupby('prediction_id').mean()
        truth_df = truth_df.sort_index()
        truth = truth_df.cancer.values.astype(int)
        print('truth_df', truth_df)
        print('')

        auc = metrics.roc_auc_score(truth, predict)
        print('auc', auc)

        thresh  = np.linspace(0, 1, 50)
        f1score = get_f1score(predict, truth, thresh)
        print('f1score.max()', f1score.max())
        print('@threshold', thresh[f1score.argmax()])
        print('')

        f1score = get_f1score(predict_threshold, truth, threshold = None)
        print('f1score', f1score)
        print('@threshold', threshold)
        print('')






print(mode)
run_submit()
print(f'***************ok!')


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/8662788.py in <cell line: 0>()
    144 
    145 print(mode)
--> 146 run_submit()
    147 print(f'***************ok!')

/tmp/ipykernel_11/8662788.py in run_submit()
      3     model = [
      4         #[EffB4Net,   '/kaggle/input/rsna-breast-mammography-weight-10/effb4-1536-baseline-gpu-aug0-01-swa.model.pth'],
----> 5         [NextVitBNet,   '/kaggle/input/rsna-breast-mammography-weight-10/nextvit-b-1536-gpu-aug0-01-swa.model.pth'],
      6 
      7     ]

NameError: name 'NextVitBNet' is not defined

## === cell 8
'''
['local', 'subset', 'skip-dicom-to-png', 'skip-add-breast-box']

auc 0.8800940438871474
f1score.max() 0.7267400144534867
threshold.max() 0.02040816326530612

f1score @threshold 0.6245315388907265

***************ok!

['local']

auc 0.8838103418305265
f1score.max() 0.4837114738826472
f1score @threshold 0.4837114738826472


***************ok!

'''
