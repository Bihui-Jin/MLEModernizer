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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
umap-learn==0.5.9.post2

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

1.0865990405783232

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import time

from IPython.display import clear_output
from tqdm import tqdm
from tqdm.contrib import tzip

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import umap

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
import matplotlib.pyplot as plt
import torch.nn.functional as F


import albumentations as albu
from albumentations.pytorch import ToTensorV2

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/train.csv")
TARGETS = train_df.columns[-6:]

## === cell 2
test = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/test.csv')
print('Test shape',test.shape)
test.head()

## === cell 3
PATH2 = '/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/'
files2 = os.listdir(PATH2)
print(f'There are {len(files2)} test spectrogram parquets')
    
spectrograms2 = {}
for i, f in enumerate(files2):
    if i % 100 == 0:
        print(i, ', ',end='')
    tmp = pd.read_parquet(f'{PATH2}{f}')
    name = int(f.split('.')[0])
    spectrograms2[name] = tmp.iloc[:, 1:].values
    
test = test.rename({'spectrogram_id': 'spec_id'}, axis=1)

## === cell 4
import pywt, librosa

USE_WAVELET = None 

NAMES = ['LL','LP','RP','RR']

FEATS = [['Fp1','F7','T3','T5','O1'],
         ['Fp1','F3','C3','P3','O1'],
         ['Fp2','F8','T4','T6','O2'],
         ['Fp2','F4','C4','P4','O2']]

directory_path = 'EEG_Spectrograms/'
if not os.path.exists(directory_path):
    os.makedirs(directory_path)
    
def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)

def denoise(x, wavelet='haar', level=1):    
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * maddest(coeff[-level])

    uthresh = sigma * np.sqrt(2*np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])

    ret=pywt.waverec(coeff, wavelet, mode='per')
    
    return ret


def spectrogram_from_eeg(parquet_path, display=False):
    
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg)-10_000)//2
    eeg = eeg.iloc[middle:middle+10_000]
    
    img = np.zeros((128,256,4),dtype='float32')
    
    if display: plt.figure(figsize=(10,7))
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

## === cell 5
PATH2 = '/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/'
DISPLAY = 1
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print('Converting Test EEG to Spectrograms...'); print()
for i, eeg_id in enumerate(EEG_IDS2):
        
    img = spectrogram_from_eeg(f'{PATH2}{eeg_id}.parquet', i < DISPLAY)
    all_eegs2[eeg_id] = img


## === cell 6
TARS = {'Seizure':0, 'LPD':1, 'GPD':2, 'LRDA':3, 'GRDA':4, 'Other':5}
TARS2 = {x: y for y, x in TARS.items()}


class EEGDataset(Dataset):
    
    def __init__(self, data, augment=False, mode='train', eeg_specs=all_eegs2): 
        self.data = data
        self.augment = augment
        self.mode = mode
        self.eeg_specs = eeg_specs
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.__getitems__([index])
    
    def __getitems__(self, indices):
        X, y = self._generate_data(indices)
        if self.augment:
            X = self.__augment(X) 
        if self.mode == 'train':
            return list(zip(X, y))
        else:
            return X
    
        
        
                
            
    def _generate_data(self, indexes):
        X = np.zeros((len(indexes), 4, 128, 256), dtype='float32')  # Adjusted the shape here
        y = np.zeros((len(indexes), 6), dtype='float32')
        
        for j, i in enumerate(indexes):
            row = self.data.iloc[i]
        
            img = self.eeg_specs[row.eeg_id]
            img = np.transpose(img, (2, 0, 1))  # Adjusted the order here to match (channels, height, width)
            X[j] = img
                
            if self.mode != 'test':
                y[j,] = row[TARGETS]  # Make sure TARGETS is correctly defined elsewhere
            
        return X, y
    
    def _random_transform(self, img):
        composition = albu.Compose([
            albu.HorizontalFlip(p=0.5),
        ])
        return composition(image=img)['image']
            
    def __augment(self, img_batch):
        for i in range(img_batch.shape[0]):
            img_batch[i,] = self._random_transform(img_batch[i,])
        return img_batch

## === cell 7
class CustomUpsample(nn.Module):
    def __init__(self, scale_factor=(2, 2)):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, modify=False, bn=True, scale_factor=(1, 1)):
        super().__init__()
        self.modify = modify
        if modify=='downsample':
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2, stride=2, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
            self.conv2 = nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels*2, kernel_size=kernel_size, padding=kernel_size//2,bias=False)
            if bn:
                self.bn1 = nn.BatchNorm2d(in_channels*2)
                self.bn2 = nn.BatchNorm2d(in_channels*2)
            else:
                self.bn1 = nn.Identity()
                self.bn2 = nn.Identity()
                
        elif modify=='upsample':
            self.conv1 = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, stride=2, kernel_size=kernel_size, output_padding=scale_factor, padding=kernel_size//2, bias=False)
            self.conv2 = nn.Conv2d(in_channels=in_channels//2, out_channels=in_channels//2, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
            self.bn1 = nn.BatchNorm2d(in_channels//2)
            self.bn2 = nn.BatchNorm2d(in_channels//2)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=kernel_size//2)
            self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=kernel_size//2)
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(in_channels)
        self.act = nn.ReLU()
        
        if modify=='downsample':
            self.proj = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*2, stride=2, kernel_size=kernel_size, padding=kernel_size//2)
        if modify=='upsample':
            self.proj = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, stride=2, kernel_size=kernel_size, output_padding=scale_factor, padding=kernel_size//2)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.modify:
            x = self.proj(x)
        out = x + out
        out = self.act(out)
        return out

    
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 16, 7, 1, 7//2)
        self.rnb1 = ResNetBlock(16, 3, modify='downsample')
        self.rnb2 = ResNetBlock(32, 3, modify='downsample')
        self.rnb3 = ResNetBlock(64, 3, modify='downsample')
        self.rnb4 = ResNetBlock(128, 3, modify='downsample')
        self.rnb5 = ResNetBlock(256, 3, modify='downsample')
        self.rnb6 = ResNetBlock(512, 3, modify='downsample')
        self.rnb7 = ResNetBlock(1024, 3, modify='downsample')
        self.rnb8 = ResNetBlock(2048, 3, modify='downsample')
        
    def forward(self, x):
        x = self.conv(x)
        x = self.rnb1(x)
        x = self.rnb2(x)
        x = self.rnb3(x)
        x = self.rnb4(x)
        x = self.rnb5(x)
        x = self.rnb6(x)
        x = self.rnb7(x)
        x = self.rnb8(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnb1 = ResNetBlock(4096, 3, modify='upsample', scale_factor=(0, 1))
        self.rnb2 = ResNetBlock(2048, 3, modify='upsample')
        self.rnb3 = ResNetBlock(1024, 3, modify='upsample')
        self.rnb4 = ResNetBlock(512, 3, modify='upsample')
        self.rnb5 = ResNetBlock(256, 3, modify='upsample')
        self.rnb6 = ResNetBlock(128, 3, modify='upsample')
        self.rnb7 = ResNetBlock(64, 3, modify='upsample')
        self.rnb8 = ResNetBlock(32, 3, modify='upsample')
        self.conv = nn.Conv2d(16, 4, 3, 1, 3//2)

    def forward(self, x):
        x = self.rnb1(x)
        x = self.rnb2(x)
        x = self.rnb3(x)
        x = self.rnb4(x)
        x = self.rnb5(x)
        x = self.rnb6(x)
        x = self.rnb7(x)
        x = self.rnb8(x)
        x = self.conv(x)
        return x

## === cell 8
class SimpleAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            Encoder(),
            Decoder()
        )
        
    def forward(self, x):
        return self.net(x)

## === cell 9
class SimpleNet2(nn.Module):
    def __init__(self):
        super(SimpleNet2, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)  # First hidden layer
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)   # Second hidden layer
        self.fc3 = nn.Linear(512, 128)    # Third hidden layer
        self.output = nn.Linear(128, 6)   # Output layer
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)
        return x
    

class SimpleNet1(nn.Module):
    def __init__(self):
        super(SimpleNet1, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)  # First hidden layer
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 128)   # Second hidden layer
        self.output = nn.Linear(128, 6)   # Output layer
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x
    
class ShitNet(nn.Module):
    def __init__(self):
        super(ShitNet, self).__init__()
        self.fc1 = nn.Linear(4096, 1024)  # First hidden layer
        self.fc2 = nn.Linear(1024, 512)   # Second hidden layer
        self.fc3 = nn.Linear(512, 128)    # Third hidden layer
        self.output = nn.Linear(128, 6)   # Output layer
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output(x)
        return x
    

class TransformerModel(nn.Module):
    def __init__(self, input_dim=4096, d_model=512, nhead=8, num_encoder_layers=3, dim_feedforward=2048):
        super(TransformerModel, self).__init__()
        
        self.encoder = nn.Linear(input_dim, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_encoder_layers)
        
        self.output_layer = nn.Linear(d_model, 6)
        
    def forward(self, src):
        src = self.encoder(src)
        
        src = src.unsqueeze(1)  # [batch_size, 1, d_model]
        
        output = self.transformer_encoder(src)
        
        output = output.squeeze(1)  # [batch_size, d_model]
        
        output = self.output_layer(output)
        return output

## === cell 10
preds = []
test_ds = EEGDataset(test, mode='test', eeg_specs=all_eegs2)
test_loader = DataLoader(test_ds, shuffle=False, batch_size=64, num_workers=3)

for i in range(5):
    print('#'*25)
    print(f'### Testing Fold {i+1}')

    
    model_ae = SimpleAE()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_ae= nn.DataParallel(model_ae)
    model_ae.load_state_dict(torch.load('/kaggle/input/best_model.pth/pytorch/hms-autoencoder/1/best_model.pth'))
    model_ae.to(device).eval()
    encoder = nn.DataParallel(model_ae.module.net[0])
    encoder.eval()
    
    model = TransformerModel()
    model.load_state_dict(torch.load('/kaggle/input/transformerhms/pytorch/1/1/model_transformer1.pth'))
    model.to(device).eval()
    
    
    fold_preds = []

    with torch.inference_mode():
        for test_batch in test_loader:
            test_batch = test_batch.to(device)
            emb = encoder(test_batch)
            embs = emb.squeeze(2, 3)
            embs = embs.to(device)
            pred = model(embs)
            probabilities = F.softmax(pred, dim=1)
            
            fold_preds.append(probabilities.cpu().numpy())
        fold_preds = np.concatenate(fold_preds)
        
    preds.append(fold_preds)

pred = np.mean(preds,axis=0)
print()
print('Test preds shape',pred.shape)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1081849920.py in <cell line: 0>()
     14     device = 'cuda' if torch.cuda.is_available() else 'cpu'
     15     model_ae= nn.DataParallel(model_ae)
---> 16     model_ae.load_state_dict(torch.load('/kaggle/input/best_model.pth/pytorch/hms-autoencoder/1/best_model.pth'))
     17     model_ae.to(device).eval()
     18     encoder = nn.DataParallel(model_ae.module.net[0])

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/best_model.pth/pytorch/hms-autoencoder/1/best_model.pth'

## === cell 11
sub = pd.DataFrame({'eeg_id': test.eeg_id.values})
sub[TARGETS] = pred
sub.to_csv('submission.csv',index=False)
print('Submissionn shape',sub.shape)
sub.head()

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2204679609.py in <cell line: 0>()
      1 sub = pd.DataFrame({'eeg_id': test.eeg_id.values})
----> 2 sub[TARGETS] = pred
      3 sub.to_csv('submission.csv',index=False)
      4 print('Submissionn shape',sub.shape)
      5 sub.head()

NameError: name 'pred' is not defined

## === cell 12
sub.iloc[:,-6:].sum(axis=1)
