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
Detect and classify harmful brain activity in electroencephalography (EEG) data: seizure (SZ), generalized periodic discharges (GPD), lateralized periodic discharges (LPD), lateralized rhythmic delta activity (LRDA), generalized rhythmic delta activity (GRDA), or "other".

## Metric
Kullback Liebler divergence between the predicted probability and the observed target.

## Submission Format
For each `eeg_id` in the test set, you must predict a probability for each of the `vote` columns. The file should contain a header and have the following format:

```
eeg_id,seizure_vote,lpd_vote,gpd_vote,lrda_vote,grda_vote,other_vote\
0,0.166,0.166,0.167,0.167,0.167,0.167\
1,0.166,0.166,0.167,0.167,0.167,0.167\
etc.
```

Your total predicted probabilities for each row must sum to one or your submission will fail.

## Dataset
**train.csv** Metadata for the train set. The expert annotators reviewed 50 second long EEG samples plus matched spectrograms covering 10 a minute window centered at the same time and labeled the central 10 seconds. Many of these samples overlapped and have been consolidated. `train.csv` provides the metadata that allows you to extract the original subsets that the raters annotated.

- `eeg_id` - A unique identifier for the entire EEG recording.
- `eeg_sub_id` - An ID for the specific 50 second long subsample this row's labels apply to.
- `eeg_label_offset_seconds` - The time between the beginning of the consolidated EEG and this subsample.
- `spectrogram_id` - A unique identifier for the entire EEG recording.
- `spectrogram_sub_id` - An ID for the specific 10 minute subsample this row's labels apply to.
- `spectogram_label_offset_seconds` - The time between the beginning of the consolidated spectrogram and this subsample.
- `label_id` - An ID for this set of labels.
- `patient_id` - An ID for the patient who donated the data.
- `expert_consensus` - The consensus annotator label. Provided for convenience only.
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The count of annotator votes for a given brain activity class. The full names of the activity classes are as follows: `lpd`: lateralized periodic discharges, `gpd`: generalized periodic discharges, `lrd`: lateralized rhythmic delta activity, and `grda`: generalized rhythmic delta activity . A detailed explanations of these patterns is [available here.](https://www.acns.org/UserFiles/file/ACNSStandardizedCriticalCareEEGTerminology_rev2021.pdf)

**test.csv** Metadata for the test set. As there are no overlapping samples in the test set, many columns in the train metadata don't apply.

- `eeg_id`
- `spectrogram_id`
- `patient_id`

**sample_submission.csv**

- `eeg_id`
- `[seizure/lpd/gpd/lrda/grda/other]_vote` - The target columns. Your predictions must be probabilities. Note that the test samples had between 3 and 20 annotators.

**train_eegs/** EEG data from one or more overlapping samples. Use the metadata in train.csv to select specific annotated subsets. The column names are [the names of the individual electrode locations for EEG leads](https://en.wikipedia.org/wiki/10%E2%80%9320_system_%28EEG%29), with one exception. The EKG column is for an electrocardiogram lead that records data from the heart. All of the EEG data (for both train and test) was collected at a frequency of 200 samples per second.

**test_eegs/** Exactly 50 seconds of EEG data.

train_spectrograms/ Spectrograms assembled EEG data. Use the metadata in train.csv to select specific annotated subsets. The column names indicate the frequency in hertz and the recording regions of the EEG electrodes. The latter are abbreviated as LL = left lateral; RL = right lateral; LP = left parasagittal; RP = right parasagittal.

**test_spectrograms/** Spectrograms assembled using exactly 10 minutes of EEG data.

**example_figures/** Larger copies of the example case images used on the overview tab.

# 2. Python version

3.12

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
librosa==0.11.0
mne==1.10.2
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
sklearn-pandas==2.2.0
timm==1.0.19
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
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        input/
            description.md (166 lines)
            example_figures.zip (14.8 MB)
            sample_submission.csv (9851 lines)
            sample_submission.csv.zip (19.0 kB)
            test.csv (9851 lines)
            test.csv.zip (22.8 kB)
            test_eegs.zip (1.5 GB)
            test_spectrograms.zip (346.5 MB)
            train.csv (96951 lines)
            train.csv.zip (1.6 MB)
            train_eegs.zip (14.4 GB)
            train_spectrograms.zip (3.2 GB)
            example_figures/
                Sample01.pdf (914.3 kB)
                Sample02.pdf (703.4 kB)
                ... and 18 other files
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
            test_eegs/
                1001717358.parquet (3.1 MB)
                1003353736.parquet (972.5 kB)
                ... and 1691 other files
            test_spectrograms/
                1002209002.parquet (713.8 kB)
                1005228554.parquet (648.5 kB)
                ... and 1112 other files
            train_eegs/
                1000913311.parquet (980.2 kB)
                1001369401.parquet (1.2 MB)
                ... and 15394 other files
            train_spectrograms/
                1000086677.parquet (564.7 kB)
                1000189855.parquet (672.7 kB)
                ... and 10022 other files
        working/
            hms-harmful-brain-activity-classification/
                description.md (166 lines)
                example_figures.zip (14.8 MB)
                ... and 10 other files
                example_figures/
                    Sample01.pdf (914.3 kB)
                    Sample02.pdf (703.4 kB)
                    ... and 18 other files
                hms-harmful-brain-activity-classification/
                test_eegs/
                    1001717358.parquet (3.1 MB)
                    1003353736.parquet (972.5 kB)
                    ... and 1691 other files
                test_spectrograms/
                    1002209002.parquet (713.8 kB)
                    1005228554.parquet (648.5 kB)
                    ... and 1112 other files
                train_eegs/
                    1000913311.parquet (980.2 kB)
                    1001369401.parquet (1.2 MB)
                    ... and 15394 other files
                train_spectrograms/
                    1000086677.parquet (564.7 kB)
                    1000189855.parquet (672.7 kB)
                    ... and 10022 other files
```

-> data/hms-harmful-brain-activity-classification/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/hms-harmful-brain-activity-classification/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/hms-harmful-brain-activity-classification/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/sample_submission.csv has 9850 rows and 7 columns.
The columns are: eeg_id, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> data/test.csv has 9850 rows and 3 columns.
The columns are: spectrogram_id, eeg_id, patient_id

-> data/train.csv has 96950 rows and 15 columns.
The columns are: eeg_id, eeg_sub_id, eeg_label_offset_seconds, spectrogram_id, spectrogram_sub_id, spectrogram_label_offset_seconds, label_id, patient_id, expert_consensus, seizure_vote, lpd_vote, gpd_vote, lrda_vote, grda_vote, other_vote

-> (stopped after 10 files for performance)

# 5. Target score

0.4077395548344591

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import random
import cv2
import json
import numpy as np
import copy
import pandas as pd
import torch
import gc

import albumentations as A
import os
import librosa
import pickle
import timm
from tqdm import tqdm
import mne

import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader

## === cell 1
CFG={
    'batch_size':32,
    'num_worker':4,
    'data':'/kaggle/input/hms-harmful-brain-activity-classification/test.csv',
    'weights_spec':'/kaggle/input/hms-baseline',
    'weights_eeg':'/kaggle/input/hms-eeg',
    'weights_mix':'/kaggle/input/hms-mix',
    
    'flip':True
}

## === cell 2
CFG['weights_spec']=[os.path.join(CFG['weights_spec'],x) for x in sorted(os.listdir(CFG['weights_spec']))]
CFG['weights_eeg']=[os.path.join(CFG['weights_eeg'],x) for x in sorted(os.listdir(CFG['weights_eeg']))]
CFG['weights_mix']=[os.path.join(CFG['weights_mix'],x) for x in sorted(os.listdir(CFG['weights_mix']))]

CFG

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1693739336.py in <cell line: 0>()
----> 1 CFG['weights_spec']=[os.path.join(CFG['weights_spec'],x) for x in sorted(os.listdir(CFG['weights_spec']))]
      2 CFG['weights_eeg']=[os.path.join(CFG['weights_eeg'],x) for x in sorted(os.listdir(CFG['weights_eeg']))]
      3 CFG['weights_mix']=[os.path.join(CFG['weights_mix'],x) for x in sorted(os.listdir(CFG['weights_mix']))]
      4 
      5 CFG

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/hms-baseline'

## === cell 3
import os
import pickle

import librosa
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch import nn
from tqdm import tqdm
data_dir='/kaggle/input/hms-harmful-brain-activity-classification/test_eegs'
NAMES = ['LL', 'LP', 'RP', 'RR']

FEATS = [['Fp1', 'F7', 'T3', 'T5', 'O1'],
         ['Fp1', 'F3', 'C3', 'P3', 'O1'],
         ['Fp2', 'F8', 'T4', 'T6', 'O2'],
         ['Fp2', 'F4', 'C4', 'P4', 'O2']]



class Transform(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.wave_transform = torchaudio.transforms.MelSpectrogram(sample_rate=200,
                                                          hop_length=10000// 256,
                                                          n_fft=1024,
                                                          n_mels=128,
                                                          f_min=0, f_max=20,
                                                          win_length=128)



    def forward(self, x):
        bs = x.size(0)


        image = self.wave_transform(x)
        return image


transform_func=Transform().to('cuda')
def spectrogram_from_eeg( parquet_path, display=False):
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle:middle + 10_000]

    img = np.zeros((128, 256, 4), dtype='float32')

    signals = []
    for k in range(4):
        COLS = FEATS[k]

        for kk in range(4):

            x = eeg[COLS[kk]].values - eeg[COLS[kk + 1]].values

            m = np.nanmean(x)
            if np.isnan(x).mean() < 1:
                x = np.nan_to_num(x, nan=m)
            else:
                x[:] = 0


            x_tensor=torch.from_numpy(x).to('cuda')
            mel_spec=transform_func(x_tensor)
            mel_spec=mel_spec.cpu().numpy()

            width = (mel_spec.shape[1] // 32) * 32
            mel_spec_db = librosa.power_to_db(mel_spec,
                                              ref=np.max).astype(np.float32)[:, :width]

            
            img[:, :, k] += mel_spec_db
            
        img[:, :, k] /= 4.0

    return img


all_fs=os.listdir(data_dir)
all_specs={}



for item in tqdm(all_fs):

    eeg_id=item.rsplit('.',1)[0]

    eeg_path = '/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/%s.parquet' % (eeg_id)



    eeg_spec=spectrogram_from_eeg(eeg_path)

    all_specs[eeg_id]=eeg_spec




with open('eeg_specs_dict.pkl', 'wb') as file:
    pickle.dump(all_specs, file)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/3344643172.py in <cell line: 0>()
     44 
     45 
---> 46 transform_func=Transform().to('cuda')
     47 def spectrogram_from_eeg( parquet_path, display=False):
     48     # LOAD MIDDLE 50 SECONDS OF EEG SERIES

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in to(self, *args, **kwargs)
   1341                     raise
   1342 
-> 1343         return self._apply(convert)
   1344 
   1345     def register_full_backward_pre_hook(

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _apply(self, fn, recurse)
    901         if recurse:
    902             for module in self.children():
--> 903                 module._apply(fn)
    904 
    905         def compute_should_use_set_data(tensor, tensor_applied):

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _apply(self, fn, recurse)
    901         if recurse:
    902             for module in self.children():
--> 903                 module._apply(fn)
    904 
    905         def compute_should_use_set_data(tensor, tensor_applied):

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _apply(self, fn, recurse)
    989         for key, buf in self._buffers.items():
    990             if buf is not None:
--> 991                 self._buffers[key] = fn(buf)
    992 
    993         return self

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in convert(t)
   1327                         memory_format=convert_to_format,
   1328                     )
-> 1329                 return t.to(
   1330                     device,
   1331                     dtype if t.is_floating_point() or t.is_complex() else None,

/usr/local/lib/python3.11/dist-packages/torch/cuda/__init__.py in _lazy_init()
    317         if "CUDA_MODULE_LOADING" not in os.environ:
    318             os.environ["CUDA_MODULE_LOADING"] = "LAZY"
--> 319         torch._C._cuda_init()
    320         # Some of the queued calls may reentrantly call _lazy_init();
    321         # we need to just return without initializing in that case.

RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and installed a driver from http://www.nvidia.com/Download/index.aspx

## === cell 4
class AlaskaDataIter():
    def __init__(self, df,
                 training_flag=False,shuffle=False,
                 use_spec=False,
                 use_eeg=False,
                 use_mix=False,
                 ll=0,rr=20):
        
        
        self.ll=0
        self.rr=rr
        self.training_flag = training_flag
        self.shuffle = shuffle

        self.raw_data_set_size = None     ##decided by self.parse_file


        self.df=df
        
        self.train_trans=A.Compose([

                                    A.HorizontalFlip(p=0.5),

                              ])

        TARS = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
        self.TARS2 = {x: y for y, x in TARS.items()}


        self.eeg_nms=['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz',
       'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG']

        self.LL = ['Fp1', 'F7', 'T3', 'T5', 'O1']

        self.RR = ['Fp2', 'F8', 'T4', 'T6', 'O2']

        self.LP = ['Fp1', 'F3', 'C3', 'P3', 'O1']

        self.RP = ['Fp2', 'F4', 'C4', 'P4', 'O2']

        self.leads_dict = {value: index for index, value in enumerate(self.eeg_nms)}

        self.use_eeg = use_eeg
        self.use_spec = use_spec
        self.use_mix = use_mix
        
        if self.use_mix:
            with open('eeg_specs_dict.pkl', mode='rb') as f:
                self.eeg_specs = pickle.load(f)
        
        
    def __getitem__(self, item):

        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):

        return len(self.df)

    def brain_lead(self, waves):
        waves = copy.deepcopy(waves)

        brain_leads = [self.LL,self.LP,self.RP,self.RP]


        leads = []

        for combine in brain_leads:
            for i in range(len(combine)-1):

                tmp_lead = waves[self.leads_dict[combine[i]]] - waves[self.leads_dict[combine[i + 1]]]
                leads.append(tmp_lead)



        data = np.concatenate([leads], axis=0)


        return data
    
    def get_eeg(self, dp, is_training):
        eeg_path = '/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/%s.parquet' % (dp['eeg_id'])
        eeg = pd.read_parquet(eeg_path)

        
        offset = 0
        eeg = eeg.iloc[int(offset * 200):int(offset * 200) + 10000]

        waves = eeg.values

        waves = np.transpose(waves, axes=[1, 0])

        for i in range(waves.shape[0]):
            m = np.nanmean(waves[i])
            if np.isnan(waves[i]).mean() < 1:
                waves[i] = np.nan_to_num(waves[i], nan=m)
            else:
                waves[i] = 0

        waves = np.array(waves, dtype=np.float64)
        waves = mne.filter.filter_data(waves, 200, self.ll, self.rr, verbose=False)

        waves = self.brain_lead(waves)

        return waves
    def get_spec(self, dp, is_training):
        spec_path = '/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/%s.parquet' % (dp['spectrogram_id'])
        spec = pd.read_parquet(spec_path)

        spec=spec.values[:,1:]

        images=[]
        r = 0

        for region in range(4):
            img = spec[r:r + 300,
                  region * 100:(region + 1) * 100].T
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            img = np.nan_to_num(img, nan=0.0)

            images.append(img)

        images=np.stack(images,-1)
        data = np.transpose(images, [2,0,1])


        return data
    
    def get_mix(self, dp, is_training):
        X = np.zeros((128, 256, 8), dtype='float32')
        kg_spec = self.get_spec(dp, is_training=False)

        eeg_spec = self.eeg_specs[str(dp['eeg_id'])]
        
        kg_spec=np.transpose(kg_spec,axes=[1,2,0])
        
        X[ 14:-14, :, :4] = kg_spec[:, 22:-22]
        X[:, :, 4:] = eeg_spec

        X = np.transpose(X, [2, 0, 1])
        return X
    def single_map_func(self, dp, is_training):
        """Data augmentation function."""

        
        if self.use_eeg:
            data=self.get_eeg(dp,is_training)
        elif self.use_spec:
            data=self.get_spec(dp,is_training)
        elif self.use_mix:
            data=self.get_mix(dp,is_training)

        
        return data.astype(np.float32)

## === cell 5



class NetSpec(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        

        self.model = timm.create_model('efficientnet_b5',
                                       pretrained=False,
                                       in_chans=3)

        self.fc = nn.Linear(2048, 6, bias=True)

        self.dropout=nn.Dropout(0.5)

        self.avg=nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)

        x1 = [x[:,  i:i + 1,:, :] for i in range(4)]
        x1 = torch.cat(x1,dim=2)
        x= torch.cat([x1,x1,x1],dim=1)
        

        if CFG['flip']:
            x_flip=torch.flip(x, [3])
            x=torch.cat([x,x_flip],dim=0)

            

        x = self.model.forward_features(x)
        x = self.avg(x)

        if CFG['flip']:
            x = x.view(2*bs, -1)
        else:
            x = x.view(bs, -1)
        x = self.dropout(x)

        x = self.fc(x)
        
        x =torch.softmax(x,-1)
        
        if CFG['flip']:
            ans= (x[:bs,...]+x[bs:,...])/2.
        else:
            ans=x
        return ans




## === cell 6

class Transform(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.wave_transform = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=25, power=1)

        self.am2db= torchaudio.transforms.AmplitudeToDB(stype='magnitude',top_db=80)

        self.resizer=nn.UpsamplingBilinear2d(size=[160,320])

    def forward(self, x):
        bs = x.size(0)


        image = self.wave_transform(x)
        image = self.am2db(image)
        n, c, h, w = image.size()
        image=image[:,:,:int(20/100*h+30),:]

        image = torch.reshape(image,shape=[n,4,-1,w])




        return image




class NetEeg(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.preprocess = Transform()

        self.model = timm.create_model('efficientnet_b5',
                                       pretrained=False,
                                       in_chans=4)

        self.fc = nn.Linear(2048, 6, bias=True)

        self.dropout=nn.Dropout(0.5)

        self.avg=nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        bs = x.size(0)

        x= self.preprocess(x)
        
        if CFG['flip']:
            x_flip=torch.flip(x, [3])
            x=torch.cat([x,x_flip],dim=0)
            
            
        x = self.model.forward_features(x)
        x = self.avg(x)
        if CFG['flip']:
            x = x.view(2*bs, -1)
        else:
            x = x.view(bs, -1)
        x = self.dropout(x)

        x = self.fc(x)
        x =torch.softmax(x,-1)
        
        if CFG['flip']:
            ans= (x[:bs,...]+x[bs:,...])/2.
        else:
            ans=x
            
        return ans


## === cell 7

class NetMix(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.preprocess = Transform()

        self.model = timm.create_model('hrnet_w18',
                                       pretrained=False,
                                       in_chans=3)

        self.fc = nn.Linear(2048, 6, bias=True)

        self.dropout=nn.Dropout(0.5)

        self.avg=nn.AdaptiveAvgPool2d(1)


    def forward(self, x):
        bs = x.size(0)

        x1 = [x[:, i:i + 1, :, :] for i in range(4)]
        x1 = torch.cat(x1, dim=2)

        x2=[x[:, i+4:i + 5, :, :] for i in range(4)]
        x2 = torch.cat(x2, dim=2)

        x= torch.cat([x1,x2],dim=3)

        x = torch.cat([x, x, x], dim=1)
        
        if CFG['flip']:
            x_flip=torch.flip(x, [3])
            x=torch.cat([x,x_flip],dim=0)
            
        x = self.model.forward_features(x)
        x = self.avg(x)
        
        if CFG['flip']:
            x = x.view(2*bs, -1)
        else:
            x = x.view(bs, -1)
            
        
        x = self.dropout(x)
        x = self.fc(x)

        x =torch.softmax(x,-1)
        
        if CFG['flip']:
            ans= (x[:bs,...]+x[bs:,...])/2.
        else:
            ans=x

        return ans



## === cell 9
def inference_function(test_loader, model, device):
    model.eval()
    
    prediction_dict = {}
    preds = []
    with tqdm(test_loader, unit="test_batch", desc='Inference') as tqdm_test_loader:
        for step, X in enumerate(tqdm_test_loader):
            X = X.to(device)
            
            
            batch_size = X.size(0)
            with torch.no_grad():
                y_preds = model(X)

            preds.append(y_preds.to('cpu').numpy()) 
                
    prediction_dict["predictions"] = np.concatenate(preds) 
    return prediction_dict

## === cell 10
test_df=pd.read_csv(CFG['data'])

test_df.head(5)


## === cell 11
predictions = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('infer with weights_spec')
for model_weight in CFG['weights_spec']:
    test_dataset = AlaskaDataIter(test_df, training_flag=False, shuffle=False,use_spec=True)
    test_loader = DataLoader(test_dataset,
                     CFG['batch_size'],
                     num_workers=CFG['num_worker'],
                     shuffle=False)

    model = NetSpec()
    state_dict = torch.load(model_weight, map_location=device)

    model.load_state_dict(state_dict,strict=False)
    model.to(device)
    prediction_dict = inference_function(test_loader, model, device)
    predictions.append(prediction_dict["predictions"])
    torch.cuda.empty_cache()
    gc.collect()
    
print('infer with weights_eeg')
for model_weight in CFG['weights_eeg']:
    test_dataset = AlaskaDataIter(test_df, training_flag=False, shuffle=False,use_eeg=True)
    test_loader = DataLoader(test_dataset,
                     CFG['batch_size'],
                     num_workers=CFG['num_worker'],
                     shuffle=False)

    model = NetEeg()
    state_dict = torch.load(model_weight, map_location=device)

    model.load_state_dict(state_dict,strict=False)
    model.to(device)
    prediction_dict = inference_function(test_loader, model, device)
    predictions.append(prediction_dict["predictions"])
    torch.cuda.empty_cache()
    gc.collect()
    
print('infer with weights_mix')
for model_weight in CFG['weights_mix']:
    test_dataset = AlaskaDataIter(test_df, training_flag=False, shuffle=False,use_mix=True)
    test_loader = DataLoader(test_dataset,
                     CFG['batch_size'],
                     num_workers=CFG['num_worker'],
                     shuffle=False)

    model = NetMix()
    state_dict = torch.load(model_weight, map_location=device)

    model.load_state_dict(state_dict,strict=False)
    model.to(device)
    prediction_dict = inference_function(test_loader, model, device)
    predictions.append(prediction_dict["predictions"])
    torch.cuda.empty_cache()
    gc.collect()



    
predictions = np.array(predictions)
predictions = np.mean(predictions, axis=0)

predictions.shape

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/1784822780.py in <cell line: 0>()
     10 
     11     model = NetSpec()
---> 12     state_dict = torch.load(model_weight, map_location=device)
     13 
     14     model.load_state_dict(state_dict,strict=False)

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

IsADirectoryError: [Errno 21] Is a directory: '/'

## === cell 12
TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
sub = pd.DataFrame({'eeg_id': test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv('submission.csv',index=False)
print(f'Submissionn shape: {sub.shape}')
sub.head()

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3276747127.py in <cell line: 0>()
      1 TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
      2 sub = pd.DataFrame({'eeg_id': test_df.eeg_id.values})
----> 3 sub[TARGETS] = predictions
      4 sub.to_csv('submission.csv',index=False)
      5 print(f'Submissionn shape: {sub.shape}')

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4297             self._setitem_frame(key, value)
   4298         elif isinstance(key, (Series, np.ndarray, list, Index)):
-> 4299             self._setitem_array(key, value)
   4300         elif isinstance(value, DataFrame):
   4301             self._set_item_frame_value(key, value)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _setitem_array(self, key, value)
   4356 
   4357             else:
-> 4358                 self._iset_not_inplace(key, value)
   4359 
   4360     def _iset_not_inplace(self, key, value):

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _iset_not_inplace(self, key, value)
   4375         if self.columns.is_unique:
   4376             if np.shape(value)[-1] != len(key):
-> 4377                 raise ValueError("Columns must be same length as key")
   4378 
   4379             for i, col in enumerate(key):

ValueError: Columns must be same length as key
