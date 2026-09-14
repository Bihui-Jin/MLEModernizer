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
scipy==1.15.3
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

0.2828732630152467

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
import copy
import torch
import gc
import os
import librosa
import pickle
import timm
import mne

import numpy as np

import albumentations as A
import pandas as pd
from tqdm import tqdm



import torch
import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.signal import butter, lfilter

## === cell 1
CFG={
    'batch_size':32,
    'num_worker':4,
    'data':'/kaggle/input/hms-harmful-brain-activity-classification/test.csv',
    'weights_eeg_raw':'/kaggle/input/hms-2nd-place-solution/pytorch/hms-eeg_raw/1',
}

## === cell 2
CFG['weights_eeg_raw']=[os.path.join(CFG['weights_eeg_raw'],x) for x in sorted(os.listdir(CFG['weights_eeg_raw']))]
CFG

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/888824173.py in <cell line: 0>()
----> 1 CFG['weights_eeg_raw']=[os.path.join(CFG['weights_eeg_raw'],x) for x in sorted(os.listdir(CFG['weights_eeg_raw']))]
      2 CFG

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/hms-2nd-place-solution/pytorch/hms-eeg_raw/1'

## === cell 3
class AlaskaDataIter():
    def __init__(self, df,
                 training_flag=False,shuffle=False,
                 use_spec=False,
                 use_eeg=False,
                 use_mix=False,
                 ll=0,rr=20,
                 flip=False,
                 use_mne_filter=True):
        
        self.flip_eeg=flip
        self.ll=ll
        self.rr=rr
        
        print(self.ll,self.rr, 'with mne filter:', use_mne_filter)
        
        
        self.training_flag = training_flag
        self.shuffle = shuffle

        self.raw_data_set_size = None     ##decided by self.parse_file


        self.df=df
        

        TARS = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other': 5}
        self.TARS2 = {x: y for y, x in TARS.items()}


        self.eeg_nms=['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1', 'Fz', 'Cz', 'Pz',
       'Fp2', 'F4', 'C4', 'P4', 'F8', 'T4', 'T6', 'O2', 'EKG']

        self.LL = ['Fp1', 'F7', 'T3', 'T5', 'O1']

        self.RL = ['Fp2', 'F8', 'T4', 'T6', 'O2']

        self.LP = ['Fp1', 'F3', 'C3', 'P3', 'O1']

        self.RP = ['Fp2', 'F4', 'C4', 'P4', 'O2']

        self.mid = ['Fz', 'Cz', 'Pz']
        self.leads_dict = {value: index for index, value in enumerate(self.eeg_nms)}

        self.use_eeg = use_eeg
        self.use_spec = use_spec
        self.use_mix = use_mix
        self.use_mne_filter=use_mne_filter
        
        
        
    def __getitem__(self, item):

        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):

        return len(self.df)

    def brain_lead(self, waves):
        waves = copy.deepcopy(waves)
        brain_leads = [self.LL, self.RL, self.LP, self.RP]

        leads = []

        for chain in brain_leads:
            for i in range(len(chain) - 1):
                tmp_lead = waves[self.leads_dict[chain[i]]] - waves[self.leads_dict[chain[i + 1]]]
                leads.append(tmp_lead)

        data = np.concatenate([leads], axis=0)
        
        return data
    def mirror_spec(self, data):

        indx = [1, 0, 3, 2]
        return data[..., indx]

    def mirror_eeg(self, data):

        indx1 = [0, 1, 2, 3, 4, 5, 6, 7]
        indx2 = [11, 12, 13, 14, 15, 16, 17, 18]

        data[indx1, ...], data[indx2, ...] = data[indx2, ...], data[indx1, ...]

        return data
    def butter_bandpass(self,lowcut, highcut, fs, order=5):
        return butter(order, [lowcut, highcut], fs=fs, btype="band")

    def butter_bandpass_filter(self,data, lowcut, highcut, fs, order=5):
        b, a = self.butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    def get_eeg(self, dp, is_training,flip=False):
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
        
        if flip:
            waves=self.mirror_eeg(waves)
        waves = self.brain_lead(waves)
        waves = np.array(waves, dtype=np.float64)

        waves = np.clip(waves, -1024, 1024)
        if self.use_mne_filter:
            waves = mne.filter.filter_data(waves, 200, self.ll, self.rr, verbose=False)
        else:
            waves = self.butter_bandpass_filter(waves,0.5,20,200,2)


        return waves
          
    
    def single_map_func(self, dp, is_training):
        
        data=self.get_eeg(dp,is_training,self.flip_eeg)
        return data.astype(np.float32)

## === cell 4
class Net1d(nn.Module):
    def __init__(self,):
        super(Net1d, self).__init__()
        self.model=timm.create_model('efficientnet_b5', pretrained=False, in_chans=3)
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.fc=nn.Linear(2048,out_features=6,bias=True)
        self.dropout=nn.Dropout(p=0.5)

    def extract_features(self, x):
        feature1=self.model.forward_features(x)
        return feature1
    def forward(self, x):
        
        bs = x.size(0)
        reshaped_tensor = x.view(bs,16,1000, 10)
        reshaped_and_permuted_tensor = reshaped_tensor.permute(0,1,3,2)
        reshaped_and_permuted_tensor= reshaped_and_permuted_tensor.reshape(bs,16*10,1000)
        x=torch.unsqueeze(reshaped_and_permuted_tensor,dim=1)
        x=torch.cat([x,x,x],dim=1)
        bs=x.size(0)

        x = self.extract_features(x)

        x = self.pool(x)
        x = x.view(bs, -1)
        
        x =self.dropout(x)
        x = self.fc(x)
        x =torch.softmax(x,-1)
        
        return x

## === cell 5
test_df=pd.read_csv(CFG['data'])

test_df.head(5)

## === cell 6
def inference_function(test_loader, model, device,double_input=False):
    model.eval()
    
    prediction_dict = {}
    preds = []
    with tqdm(test_loader, unit="test_batch", desc='Inference') as tqdm_test_loader:
        for step, X in enumerate(tqdm_test_loader):
            
            if double_input:
                wave,spec=X
                
                wave = wave.to(device)
                spec = spec.to(device)
                
                
                batch_size = wave.size(0)
                with torch.no_grad():
                    y_preds = model(wave,spec)
                    
            else:
                X = X.to(device)

                batch_size = X.size(0)
                with torch.no_grad():
                    y_preds = model(X)

            preds.append(y_preds.to('cpu').numpy()) 
                
    prediction_dict["predictions"] = np.concatenate(preds) 
    return prediction_dict

## === cell 7
def run_weight_eeg_raw():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print('infer with run_weight_eeg_raw')
    predictions=[]
    for model_weight in CFG['weights_eeg_raw']:
        test_dataset = AlaskaDataIter(test_df, training_flag=False, shuffle=False,use_eeg=True,ll=0.5,rr=20)
        test_loader = DataLoader(test_dataset,
                         CFG['batch_size'],
                         num_workers=CFG['num_worker'],
                         shuffle=False)

        model = Net1d()
        state_dict = torch.load(model_weight, map_location=device)

        model.load_state_dict(state_dict,strict=True)
        model.to(device)
        prediction_dict = inference_function(test_loader, model, device)
        predictions.append(prediction_dict["predictions"])


        test_dataset = AlaskaDataIter(test_df, training_flag=False, shuffle=False,use_eeg=True,flip=True,ll=0.5,rr=20)
        test_loader = DataLoader(test_dataset,
                         CFG['batch_size'],
                         num_workers=CFG['num_worker'],
                         shuffle=False)

        model = Net1d()
        state_dict = torch.load(model_weight, map_location=device)

        model.load_state_dict(state_dict,strict=True)
        model.to(device)
        prediction_dict = inference_function(test_loader, model, device)
        predictions.append(prediction_dict["predictions"])

        torch.cuda.empty_cache()
        gc.collect()
    predictions = np.array(predictions)
    predictions = np.mean(predictions, axis=0)
    
    return predictions

## === cell 9
predictions = []

ans=run_weight_eeg_raw()
predictions.append(ans) 


predictions=np.array(predictions)
predictions=np.mean(predictions,axis=0)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/1677361690.py in <cell line: 0>()
      2 
      3 # #***0.23 best version29, add 1000 val data
----> 4 ans=run_weight_eeg_raw()
      5 predictions.append(ans)
      6 

/tmp/ipykernel_11/2507761288.py in run_weight_eeg_raw()
     12 
     13         model = Net1d()
---> 14         state_dict = torch.load(model_weight, map_location=device)
     15 
     16         model.load_state_dict(state_dict,strict=True)

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

## === cell 10
TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
sub = pd.DataFrame({'eeg_id': test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv('submission.csv',index=False)
print(f'Submissionn shape: {sub.shape}')
sub.head()

## --- ERROR in cell 10, traceback:
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
