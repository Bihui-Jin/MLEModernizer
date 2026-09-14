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

0.2425531914893617

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
try:
    import pylibjpeg
except:
   !pip install /kaggle/input/rsna-2022-whl/{pylibjpeg-1.4.0-py3-none-any.whl,python_gdcm-3.0.15-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl}

print('install pylibjpeg, python_gdcm ok!')

import sys, os
sys.path.append('/kaggle/input/rsna-breast-mammography-00')


import pandas as pd
import numpy as np
import pydicom
import cv2

from timeit import default_timer as timer

from tqdm.notebook import tqdm
from joblib import Parallel, delayed
from glob import glob
from sklearn import metrics

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

import timm
print('timm',timm.__version__)
from timm.models.resnet import *
from timm.models.efficientnet import *


def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)

    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)

    else:
        raise NotImplementedError


print('import ok!')

## === cell 1
image_size = 512

mode = 'submit-dicom'   #'local-dicom' #local-image  submit-dicom


if 'local' in mode:
    csv_file = '/kaggle/input/rsna-breast-mammography-00/valid_df.fold0.csv'
    dcm_dir  = '/kaggle/input/rsna-breast-cancer-detection/train_images'

if 'submit' in mode:
    csv_file = '/kaggle/input/rsna-breast-cancer-detection/test.csv'
    dcm_dir  = '/kaggle/input/rsna-breast-cancer-detection/test_images'
    
if 'dicom' in mode:
    image_dir = '/kaggle/tmp/~png'
    os.makedirs(image_dir, exist_ok=True)



test_df = pd.read_csv(csv_file)
test_df.loc[:, 'i'] = np.arange(len(test_df))
if 'local' in mode:
    test_df.loc[:, 'prediction_id'] = test_df.patient_id.astype(str) + '_' + test_df.laterality
    if 'dicom' in mode:
        test_id = [
            826, 1703, 1759, 2346, 2989, 3021, 3542, 4340, 4824, 5059,
            5769, 6654, 6658, 7053, 7493, 7780, 9014, 11094, 11937, 14292,
            30, 36, 65, 90, 111, 122, 127, 152, 158, 204,
            272, 282, 289, 299, 308, 399, 425, 454, 477, 505,
        ]
        test_df = test_df[test_df.patient_id.isin(test_id)].reset_index(drop=True)

print('test_df', test_df.shape)
print(test_df)



def dicom_to_png(dcm_file, image_size=image_size, image_dir=''):
    patient_id = dcm_file.split('/')[-2]
    image_id   = dcm_file.split('/')[-1][:-4]

    dicom = pydicom.dcmread(dcm_file)
    img = dicom.pixel_array
    img = (img - img.min()) / (img.max() - img.min())
    if dicom.PhotometricInterpretation == 'MONOCHROME1':
        img = 1 - img

    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = (img * 255).astype(np.uint8)
    cv2.imwrite(image_dir +'/'+ f'{patient_id}_{image_id}.png', img)


if 'dicom' in mode:
    dcm_file = dcm_dir + '/' + test_df.patient_id.astype(str) + '/'  + test_df.image_id.astype(str) + '.dcm'
    Parallel(n_jobs=2)(
        delayed(dicom_to_png)(f, image_size=image_size, image_dir=image_dir)
        for f in tqdm(dcm_file)
    )



def read_data(df):
    image = []
    for t, d in df.iterrows():
        m = cv2.imread(f'{image_dir}/{d.patient_id}_{d.image_id}.png', cv2.IMREAD_GRAYSCALE)
        image.append(m)

    image = np.stack(image)
    return image


class RsnaDataset(Dataset):
    def __init__(self, df):

        patient_id =  sorted(df.patient_id.unique())
        self.patient_id = patient_id
        self.length = len(patient_id)
        self.df = df

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        patient_id = self.patient_id[index]
        df = self.df[self.df.patient_id == patient_id].reset_index(drop=True)
        image = read_data(df)
        image = image.astype(np.float32)/255

        r = {}
        r['index'] = index
        r['patient_id'] = patient_id
        r['df'] = df
        r['num'] = len(df)
        r['image'] = torch.from_numpy(image).float()
        return r

tensor_key = ['image']
def null_collate(batch):
    d = {}
    key = batch[0].keys()
    for k in key:
        d[k] = [b[k] for b in batch]

    d['image'] = torch.cat(d['image'],0).unsqueeze(1)
    return d




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 490, in _process_worker
    r = call_item()
        ^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/externals/loky/process_executor.py", line 291, in __call__
    return self.fn(*self.args, **self.kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in __call__
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/joblib/parallel.py", line 607, in <listcomp>
    return [func(*args, **kwargs) for func, args, kwargs in self.items]
            ^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/1268529357.py", line 44, in dicom_to_png
  File "/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py", line 2193, in pixel_array
    self.convert_pixel_data()
  File "/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py", line 1726, in convert_pixel_data
    self._pixel_array = pixel_array(self, **opts)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py", line 1430, in pixel_array
    return decoder.as_array(
           ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py", line 982, in as_array
    self._validate_plugins(decoding_plugin),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py", line 257, in _validate_plugins
    raise RuntimeError(
RuntimeError: Unable to decompress 'JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])' pixel data because all plugins are missing dependencies:
	gdcm - requires gdcm>=3.0.10
	pylibjpeg - requires pylibjpeg>=2.0 and pylibjpeg-libjpeg>=2.1
"""

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/1268529357.py in <cell line: 0>()
     56     #dcm_file = glob(f'{dcm_dir}/*/*.dcm')
     57     #print(dcm_file)
---> 58     Parallel(n_jobs=2)(
     59         delayed(dicom_to_png)(f, image_size=image_size, image_dir=image_dir)
     60         for f in tqdm(dcm_file)

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in __call__(self, iterable)
   2070         next(output)
   2071 
-> 2072         return output if self.return_generator else list(output)
   2073 
   2074     def __repr__(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _get_outputs(self, iterator, pre_dispatch)
   1680 
   1681             with self._backend.retrieval_context():
-> 1682                 yield from self._retrieve()
   1683 
   1684         except GeneratorExit:

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _retrieve(self)
   1782             # worker traceback.
   1783             if self._aborting:
-> 1784                 self._raise_error_fast()
   1785                 break
   1786 

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _raise_error_fast(self)
   1857         # called directly or if the generator is gc'ed.
   1858         if error_job is not None:
-> 1859             error_job.get_result(self.timeout)
   1860 
   1861     def _warn_exit_early(self):

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in get_result(self, timeout)
    756             # callback thread, and is stored internally. It's just waiting to
    757             # be returned.
--> 758             return self._return_or_raise()
    759 
    760         # For other backends, the main thread needs to run the retrieval step.

/usr/local/lib/python3.11/dist-packages/joblib/parallel.py in _return_or_raise(self)
    771         try:
    772             if self.status == TASK_ERROR:
--> 773                 raise self._result
    774             return self._result
    775         finally:

RuntimeError: Unable to decompress 'JPEG Lossless, Non-Hierarchical, First-Order Prediction (Process 14 [Selection Value 1])' pixel data because all plugins are missing dependencies:
	gdcm - requires gdcm>=3.0.10
	pylibjpeg - requires pylibjpeg>=2.0 and pylibjpeg-libjpeg>=2.1

## === cell 2

class RGB(nn.Module):
    IMAGE_RGB_MEAN = [0.5, 0.5, 0.5] #[0.485, 0.456, 0.406]
    IMAGE_RGB_STD  = [0.5, 0.5, 0.5] #[0.229, 0.224, 0.225]

    def __init__(self, ):
        super(RGB, self).__init__()
        self.register_buffer('mean', torch.zeros(1, 3, 1, 1))
        self.register_buffer('std', torch.ones(1, 3, 1, 1))
        self.mean.data = torch.FloatTensor(self.IMAGE_RGB_MEAN).view(self.mean.shape)
        self.std.data = torch.FloatTensor(self.IMAGE_RGB_STD).view(self.std.shape)

    def forward(self, x):
        x = (x - self.mean) / self.std
        return x

class ResNet(nn.Module):
    def __init__(self,):
        super(ResNet, self).__init__()
        self.rgb = RGB()
        self.encoder = seresnext50_32x4d(pretrained=False)
        self.cancer  = nn.Linear(2048,1)

    def forward(self, batch):
        x = batch['image']
        x = x.expand(-1,3,-1,-1)
        x = self.rgb(x)

        e = self.encoder
        x = e.forward_features(x)
        x = F.adaptive_avg_pool2d(x,1)
        x = torch.flatten(x,1,3)
        cancer = self.cancer(x)

        cancer = cancer.reshape(-1)
        cancer = torch.sigmoid(cancer)
        cancer = torch.nan_to_num(cancer)
        return cancer


class EffNet(nn.Module):
    def __init__(self,):
        super(EffNet, self).__init__()
        self.rgb = RGB()
        self.encoder = efficientnet_b4(pretrained=False)
        self.cancer  = nn.Linear(1792,1)

    def forward(self, batch):
        x = batch['image']
        x = x.expand(-1,3,-1,-1)
        x = self.rgb(x)

        e = self.encoder
        x = e.forward_features(x)
        x = F.adaptive_avg_pool2d(x,1)
        x = torch.flatten(x,1,3)
        cancer = self.cancer(x)

        cancer = cancer.reshape(-1)
        cancer = torch.sigmoid(cancer)
        cancer = torch.nan_to_num(cancer)
        return cancer




## === cell 3

def run_submit():
    model = [
        [ResNet,'/kaggle/input/rsna-breast-mammography-weight-01/seresnext50_32x4d-512-fold0-00009366.model.pth'],
        [ResNet,'/kaggle/input/rsna-breast-mammography-weight-01/seresnext50_32x4d-512-fold1-00016056.model.pth'],
        [EffNet,'/kaggle/input/rsna-breast-mammography-weight-01/efficientnet_b4-512-fold0-00012042.model.pth'],
        [EffNet,'/kaggle/input/rsna-breast-mammography-weight-01/efficientnet_b4-512-fold1-00012042.model.pth'],
    ]
    num_net = len(model)

    net = []
    for i in range(num_net):
        Net, checkpoint = model[i]
        n = Net()
        f = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        n.load_state_dict(f['state_dict'], strict=True)  # True
        n.cuda()
        n.eval()
        net.append(n)

    test_dataset = RsnaDataset(test_df)
    test_loader = DataLoader(
        test_dataset,
        sampler = SequentialSampler(test_dataset),
        batch_size  = 8,
        drop_last   = False,
        num_workers = 2,
        pin_memory  = False,
        collate_fn = null_collate,
    )

    if 1:
        result = {
            'i':[],
            'probability':[],
        }
        test_num = 0

        start_timer = timer()
        for t, batch in enumerate(test_loader):
            batch_size = len(batch['index'])
            batch['image'] = batch['image'].cuda()

            p = 0
            count = 0
            with torch.no_grad():
                with amp.autocast(enabled=True):
                    for i in range(num_net):
                        p += net[i](batch)  # net(input)#
                        count += 1

                        if 1:
                            batch['image'] = torch.flip(batch['image'], dims=[3, ])
                            p += net[i](batch)
                            count += 1

            p = p / count
            result['probability'].append(p.float().data.cpu().numpy())
            result['i'].append(pd.concat(batch['df'])['i'].values)
            test_num += batch_size
            print('\r %8d / %d  %s' % (test_num, len(test_dataset), time_to_str(timer() - start_timer, 'sec')), end='', flush=True)
        print('')

        probability = np.concatenate(result['probability'])
        i = np.concatenate(result['i'])
        argsort = np.argsort(i)
        i = i[argsort]
        probability = probability[argsort]
        np.save('probability.npy',probability)

    probability = np.load('probability.npy')
    print('probability', probability.shape)
    print('')

    submit_df = pd.DataFrame({'prediction_id':test_df.prediction_id})
    submit_df.loc[:, 'cancer'] = probability
    submit_df = submit_df.groupby('prediction_id').max() #.mean() #
    submit_df = submit_df.sort_index()
    submit_df.loc[:, 'cancer'] = (submit_df.cancer.values>0.23).astype(np.float32)
    
    submit_df.to_csv('submission.csv',index=True)
    print('submit_df',submit_df)
    print(submit_df.values.mean())
    print('')

    if 'local' in mode: 

        def pfbeta(labels, predictions, beta=1):
            y_true_count = 0
            ctp = 0
            cfp = 0

            for idx in range(len(labels)):
                prediction = min(max(predictions[idx], 0), 1)
                if (labels[idx]):
                    y_true_count += 1
                    ctp += prediction
                else:
                    cfp += prediction

            beta_squared = beta * beta
            c_precision = ctp / (ctp + cfp)
            c_recall = ctp / y_true_count
            if (c_precision > 0 and c_recall > 0):
                result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
                return result
            else:
                return 0

        truth_df = test_df[['prediction_id', 'cancer']].groupby('prediction_id').mean()
        truth_df = truth_df.sort_index()
        print('truth_df', truth_df)
        print(truth_df.values.mean())
        print('')

        truth = truth_df.cancer.values
        predict = submit_df.cancer.values

        lb_score = pfbeta(truth, predict)
        print('lb_score', lb_score)

        auc = metrics.roc_auc_score(truth, predict)
        print('auc', auc)

run_submit()

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/184028075.py in <cell line: 0>()
    130         print('auc', auc)
    131 
--> 132 run_submit()

/tmp/ipykernel_11/184028075.py in run_submit()
     12         Net, checkpoint = model[i]
     13         n = Net()
---> 14         f = torch.load(checkpoint, map_location=lambda storage, loc: storage)
     15         n.load_state_dict(f['state_dict'], strict=True)  # True
     16         n.cuda()

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/rsna-breast-mammography-weight-01/seresnext50_32x4d-512-fold0-00009366.model.pth'
