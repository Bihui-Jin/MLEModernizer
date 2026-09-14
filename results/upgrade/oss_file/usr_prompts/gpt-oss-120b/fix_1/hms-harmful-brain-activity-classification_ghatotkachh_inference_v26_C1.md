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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
PyWavelets==1.8.0
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

0.5859539902913767

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os


## === cell 1
import albumentations as A
import gc
import librosa
import matplotlib.pyplot as plt
import math
import multiprocessing
import numpy as np
import os
import pandas as pd
import pywt
import random
import time
import timm
import torch
import torch.nn as nn


from albumentations.pytorch import ToTensorV2
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')

## === cell 2
class config:
    model1 = "resnet50d"
    model2 = "vit_base_patch16_224"
    model3 = "resnet34d"
    epoch = 10
    lr = 1e-3
    batchsize = 32
    splits = 5
    momentum = 0.9
    MAX_GRAD_NORM = 1e7
    WEIGHT_DECAY = 0.01
    device = "cpu"
    FOLDS=5
    AMP= True
class paths:
    preloadedeeg = "/kaggle/input/brain-eeg-spectrograms/eeg_specs.npy"
    train_eeg_dir = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs"
    train_spec_dir = "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms"
    train_csv = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    test_csv = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    test_eeg = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs"
    test_spec = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms"
    out = "/kaggle/working/"

## === cell 3
USE_WAVELET = None

NAMES = ['LL','LP','RP','RR']

FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]
def maddest(d, axis: int = None):
    """
    Denoise function.
    """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x: np.ndarray, wavelet: str = 'haar', level: int = 1): 
    coeff = pywt.wavedec(x, wavelet, mode="per") # multilevel 1D Discrete Wavelet Transform of data.
    sigma = (1/0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    output = pywt.waverec(coeff, wavelet, mode='per')
    return output

def spectrogram_from_eeg(parquet_path, display=False):
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg)-10_000)//2
    eeg = eeg.iloc[middle:middle+10_000]
    
    img = np.zeros((128,256,4),dtype='float32')
    
    if display:
        plt.figure(figsize=(10,7))
    signals = []
    for k in range(4):
        COLS = FEATS[k]
        
        for kk in range(4):
        
            x = eeg[COLS[kk]].values - eeg[COLS[kk+1]].values

            m = np.nanmean(x)
            if np.isnan(x).mean()<1: x = np.nan_to_num(x,nan=m)
            else: x[:] = 0

            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)
            signals.append(x)

            mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//256, 
                  n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)

            width = (mel_spec.shape[1]//32)*32
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[:,:width]

            mel_spec_db = (mel_spec_db+40)/40 
            img[:,:,k] += mel_spec_db
                
        img[:,:,k] /= 4.0
        
        if display:
            plt.subplot(2,2,k+1)
            plt.imshow(img[:,:,k],aspect='auto',origin='lower')
            plt.title(f'EEG {eeg_id} - Spectrogram {NAMES[k]}')
            
    if display: 
        plt.show()
        plt.figure(figsize=(10,5))
        offset = 0
        for k in range(4):
            if k>0: offset -= signals[3-k].min()
            plt.plot(range(10_000),signals[k]+offset,label=NAMES[3-k])
            offset += signals[3-k].max()
        plt.legend()
        plt.title(f'EEG {eeg_id} Signals')
        plt.show()
        print(); print('#'*25); print()
        
    return img

## === cell 4
df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")

## === cell 5
all_eegs = {}

for i in os.listdir(paths.test_eeg):
    sp = spectrogram_from_eeg(os.path.join(paths.test_eeg,i))
    name = int(i.split('.')[0])
    all_eegs[name] = np.array(sp)

## === cell 6
all_eegs

## === cell 7
test_df = pd.read_csv(paths.test_csv)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()

## === cell 8
import os 
all_spectrograms = {}

for i in os.listdir(paths.test_spec):
    sp = pd.read_parquet(os.path.join(paths.test_spec,i))
    name = int(i.split('.')[0])
    all_spectrograms[name] = np.array(sp)

## === cell 9
all_spectrograms

## === cell 11
class CustomDataset():
    def __init__(self,traindf:pd.DataFrame,config,mode:str="train",specs:dict[int,np.ndarray]=all_spectrograms,eegs:dict[int,np.ndarray]=all_eegs):
        self.traindf = traindf;
        self.specs = specs;
        self.eeg = eegs;
        self.mode = mode;
    def __len__(self):
        return len(self.traindf)
    def __getitem__(self,idx):
        X = np.zeros((128, 256, 8), dtype='float32')
        y = np.zeros(6, dtype='float32')
        img = np.ones((128,256), dtype='float32')
        row = self.traindf.iloc[idx]
        if self.mode=='test': 
            r = 0
        else: 
            r = int((row['min'] + row['max']) // 4)
            
        print(r)
        for region in range(4):
            img = self.specs[row.spectrogram_id][r:r+300, region*100:(region+1)*100].T
            
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)

            ep = 1e-6
            mu = np.nanmean(img.flatten())
            std = np.nanstd(img.flatten())
            img = (img-mu)/(std+ep)
            img = np.nan_to_num(img, nan=0.0)
            X[14:-14, :, region] = img[:, 22:-22] / 2.0
            img = self.eeg[row.eeg_id]
            X[:, :, 4:] = img
        
        X = torch.tensor(X)
        spectograms = [X[:, :, i:i+1] for i in range(4)]
        spectograms = torch.cat(spectograms, dim=0)
        
        eegs = [X[:, :, i:i+1] for i in range(4,8)]
        eegs = torch.cat(eegs, dim=0)
        
        x = torch.cat([spectograms, eegs], dim=1)
        x = torch.cat([x,x,x], dim=2)
        x = x.permute(2, 0, 1)
        if self.mode != 'test':
            y = row[targets].values.astype(np.float32)
            
            
        return {"data":x,
                "target":y}
        

## === cell 12
customdataset = CustomDataset(test_df,config,mode="test")

## === cell 13
customdataset[0]

## === cell 14
test_df.iloc[0].spectrogram_id

## === cell 15
from torch.utils.data import DataLoader
test_loader = DataLoader(
    customdataset,
    batch_size=config.batchsize,
)
X= customdataset[0]["data"]
y = customdataset[0]["target"]
y



## === cell 16
class Custommodel(nn.Module):
    def __init__(self,config,numclass:int=6,):
        super(Custommodel,self).__init__()
        self.model = timm.create_model(
            config.model1,
            pretrained = False,
            drop_rate = 0.1,
            drop_path_rate = 0.2,
        )
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.customlayer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, numclass)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.customlayer(x)
        return x

## === cell 17
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    prediction_dict = {}
    preds = []
    for i in test_loader:
        x = i["data"].to(device)
        y = i["target"].to(device)
        batchsize = len(y)
        with torch.no_grad():
            ypred = model(x)
        ypred = softmax(ypred)
        preds.append(ypred.to('cpu').numpy()) 
                
    prediction_dict["predictions"] = np.concatenate(preds) 
    return prediction_dict


## === cell 18
predictions = []
for i in os.listdir("/kaggle/input/resnet5010ep2"):
    dd = torch.load(os.path.join("/kaggle/input/resnet5010ep2",i))
    model = Custommodel(config)
    model.load_state_dict(dd['model'])
    testdataset = CustomDataset(test_df,config,mode = "test")
    testloader = DataLoader(testdataset,batch_size = config.batchsize)
    model.to(device)
    prediction_dict = inference_function(testloader, model, device)
    predictions.append(prediction_dict["predictions"])
predictions = np.array(predictions)
predictions = np.mean(predictions,axis=0)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_10/1230965582.py in <cell line: 0>()
      1 predictions = []
----> 2 for i in os.listdir("/kaggle/input/resnet5010ep2"):
      3     dd = torch.load(os.path.join("/kaggle/input/resnet5010ep2",i))
      4     model = Custommodel(config)
      5     model.load_state_dict(dd['model'])

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/resnet5010ep2'

## === cell 19
class Custommodelkk(nn.Module):
    def __init__(self,config,numclass:int=6,):
        super(Custommodelkk,self).__init__()
        self.model = timm.create_model(
            config.model3,
            pretrained = False,
            drop_rate = 0.1,
            drop_path_rate = 0.2,
        )
        self.features = nn.Sequential(*list(self.model.children())[:-2])
        self.customlayer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.model.num_features, numclass)
        )
    def forward(self,x):
        x = self.features(x)
        x = self.customlayer(x)
        return x

## === cell 20
predictions3 = []
for i in os.listdir("/kaggle/input/resnet34d2"):
    dd = torch.load(os.path.join("/kaggle/input/resnet34d2",i))
    model = Custommodelkk(config)
    model.load_state_dict(dd['model'])
    testdataset = CustomDataset(test_df,config,mode = "test")
    testloader = DataLoader(testdataset,batch_size = config.batchsize)
    model.to(device)
    prediction_dict = inference_function(testloader, model, device)
    predictions3.append(prediction_dict["predictions"])
predictions3 = np.array(predictions)
predictions3 = np.mean(predictions,axis=0)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_10/384081509.py in <cell line: 0>()
      1 predictions3 = []
----> 2 for i in os.listdir("/kaggle/input/resnet34d2"):
      3     dd = torch.load(os.path.join("/kaggle/input/resnet34d2",i))
      4     model = Custommodelkk(config)
      5     model.load_state_dict(dd['model'])

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/resnet34d2'

## === cell 21
predictions

## === cell 22
from torchvision.transforms import transforms
tras = transforms.Compose([
    transforms.Resize((224,224))
])
class Custommodel2(nn.Module):
    def __init__(self,config,transform,numclass:int=6):
        super(Custommodel2,self).__init__()
        self.model = timm.create_model(
            config.model2,
            pretrained = False,
        )
        self.model.head = nn.Linear(self.model.head.in_features, numclass)
        self.transform = transform
    def forward(self,x):
        x = self.transform(x)
        x = self.model(x)
        return x


## === cell 23
predictions2 = []
for i in os.listdir("/kaggle/input/visiontransformer"):
    dd = torch.load(os.path.join("/kaggle/input/visiontransformer",i))
    model = Custommodel2(config,tras)
    model.load_state_dict(dd['model'])
    testdataset = CustomDataset(test_df,config,mode = "test")
    testloader = DataLoader(testdataset,batch_size = config.batchsize)
    model.to(device)
    prediction_dict = inference_function(testloader, model, device)
    predictions2.append(prediction_dict["predictions"])
predictions2 = np.array(predictions2)
predictions2 = np.mean(predictions2,axis=0)


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_10/604382599.py in <cell line: 0>()
      1 predictions2 = []
----> 2 for i in os.listdir("/kaggle/input/visiontransformer"):
      3     dd = torch.load(os.path.join("/kaggle/input/visiontransformer",i))
      4     model = Custommodel2(config,tras)
      5     model.load_state_dict(dd['model'])

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/visiontransformer'

## === cell 24
r34 = 1
r50 = 1
vit = 1
finalpred = (predictions*r34+predictions2*vit+predictions3*r50)
finalpred = finalpred / np.sum(finalpred,axis=1,keepdims=True)

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
AxisError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3518174610.py in <cell line: 0>()
      3 vit = 1
      4 finalpred = (predictions*r34+predictions2*vit+predictions3*r50)
----> 5 finalpred = finalpred / np.sum(finalpred,axis=1,keepdims=True)

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in sum(a, axis, dtype, out, keepdims, initial, where)
   2311         return res
   2312 
-> 2313     return _wrapreduction(a, np.add, 'sum', axis, dtype, out, keepdims=keepdims,
   2314                           initial=initial, where=where)
   2315 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in _wrapreduction(obj, ufunc, method, axis, dtype, out, **kwargs)
     86                 return reduction(axis=axis, out=out, **passkwargs)
     87 
---> 88     return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
     89 
     90 

AxisError: axis 1 is out of bounds for array of dimension 1

## === cell 26
finalpred

## === cell 27
TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
sub = pd.DataFrame({'eeg_id': test_df.eeg_id.values})
sub[TARGETS] = finalpred
sub.to_csv('submission.csv',index=False)
print(f'Submissionn shape: {sub.shape}')
sub.head()

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_10/2649871787.py in <cell line: 0>()
      1 TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
      2 sub = pd.DataFrame({'eeg_id': test_df.eeg_id.values})
----> 3 sub[TARGETS] = finalpred
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

## === cell 28
print(np.sum(finalpred))
