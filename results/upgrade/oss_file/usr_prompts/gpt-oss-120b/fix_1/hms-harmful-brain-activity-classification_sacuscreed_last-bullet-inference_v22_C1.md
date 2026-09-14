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

cupy-cuda12x==13.6.0
fastai==2.8.5
geopandas==0.14.4
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
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

0.4280755309867892

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.utils.data import Dataset, DataLoader
from fastai.vision.all import *
from skimage.transform import resize
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import gc
import shutil

device = "cuda" if torch.cuda.is_available() else "cpu"

## === cell 1
PATH = {
    'test':'/kaggle/input/hms-harmful-brain-activity-classification/test_',
    'train':'/kaggle/input/hms-harmful-brain-activity-classification/train_'
}
BS = 512
models = {
    'eeg':[
        "/kaggle/input/eegs-cv-kaggle/EEGS_CNN_0",
        "/kaggle/input/eegs-cv-kaggle/EEGS_CNN_1",
        "/kaggle/input/eegs-cv-kaggle/EEGS_CNN_2",
        "/kaggle/input/eegs-cv-kaggle/EEGS_CNN_3",
        "/kaggle/input/eegs-cv-kaggle/EEGS_CNN_4"
    ],
    'spec':[
        "/kaggle/input/specs-cv-kaggle/SPECS_CNN_0",
        "/kaggle/input/specs-cv-kaggle/SPECS_CNN_1",
        "/kaggle/input/specs-cv-kaggle/SPECS_CNN_2",
        "/kaggle/input/specs-cv-kaggle/SPECS_CNN_3",
        "/kaggle/input/specs-cv-kaggle/SPECS_CNN_4"
    ],
    'spec_fixed':[
        "/kaggle/input/specs-cv-kaggle-fixed/SPECS_CNN_0",
        "/kaggle/input/specs-cv-kaggle-fixed/SPECS_CNN_1",
        "/kaggle/input/specs-cv-kaggle-fixed/SPECS_CNN_2",
        "/kaggle/input/specs-cv-kaggle-fixed/SPECS_CNN_3",
        "/kaggle/input/specs-cv-kaggle-fixed/SPECS_CNN_4"
    ],
    'custom':[
        "/kaggle/input/custom-cv-kaggle/custom_SPECS_CNN_0",
        "/kaggle/input/custom-cv-kaggle/custom_SPECS_CNN_1",
        "/kaggle/input/custom-cv-kaggle/custom_SPECS_CNN_2",
        "/kaggle/input/custom-cv-kaggle/custom_SPECS_CNN_3",
        "/kaggle/input/custom-cv-kaggle/custom_SPECS_CNN_4"
    ],
    'custom_fixed':[
        "/kaggle/input/custom-cv-kaggle-fixed/custom_SPECS_CNN_0",
        "/kaggle/input/custom-cv-kaggle-fixed/custom_SPECS_CNN_1",
        "/kaggle/input/custom-cv-kaggle-fixed/custom_SPECS_CNN_2",
        "/kaggle/input/custom-cv-kaggle-fixed/custom_SPECS_CNN_3",
        "/kaggle/input/custom-cv-kaggle-fixed/custom_SPECS_CNN_4"
    ],
    'HMS':[
        "/kaggle/input/hms-xtra/HMSmodel_0",
        "/kaggle/input/hms-xtra/HMSmodel_1",
        "/kaggle/input/hms-xtra/HMSmodel_2",
        "/kaggle/input/hms-xtra/HMSmodel_3",
        "/kaggle/input/hms-xtra/HMSmodel_4"
    ]
}
DEBUG = False

## === cell 2
TH = 1# Percentile of pseudolabels to take
pseudo_votes = 10# The votes assigned to pseudo_labels
batch_size = 128
min_votes = 0
LR = 1e-3
EPOCHS = 2
start_p  = .5
end_p = 1
start_min_v = 1
end_min_v = 1
label_aug = False
N_FOLDS = 5
FOLDS = [0, 1, 2, 3, 4]
HMS_DROP = .9
CV = 'eeg_id'# Different eegs even of the same patient can be quite different.
GB = 'votation'# The string TAG of sample normalized votates, 'expert_consensus' is another reasonable choice

## === cell 3
def normalize(spec,epsilon=1e-6,NATURAL=False):
    if NATURAL:
        spec = np.clip(spec,np.exp(-4),np.exp(8))
        spec = np.log(spec)
    else:
        spec = np.clip(spec,np.exp(-4),np.exp(6))
        spec = np.log10(spec)

    mask = ~np.isnan(spec)
    mean = np.mean(spec[mask])
    std = np.std(spec[mask])
    spec[mask] = spec[mask] - mean
    if std > 0: spec[mask] /= (std + epsilon)
    
    return spec

## === cell 4
from scipy.signal import butter, lfilter
def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data

## === cell 5
class HMSmodel(nn.Module):

    def __init__(self,models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.FC = nn.Linear(1280*len(models),6).to(device)

        for model in self.models:
            for p in model.parameters():
                p.requires_grad = False
        
        weights = []
        bias = 0
        for model in self.models:
            model.features[8][0].weight.requires_grad = True
            model.classifier[0] = nn.Dropout(HMS_DROP)
            weights.append(model.classifier[1].weight)
            bias += model.classifier[1].bias
            model.classifier[1] = nn.Identity()
            
        self.FC.weight = nn.Parameter(torch.cat(weights,1).to(device))
        self.FC.bias = nn.Parameter(bias.to(device))

    def forward(self,X):
        eeg,spec,custom_spec = X
        X = torch.cat([self.models[i](X[i]) for i in range(len(self.models))],1)
        OUT = self.FC(X)
        return OUT

## === cell 7
submission = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv')
submission.head()

## === cell 8
votes = [c for c in submission.columns if '_vote' in c]
votes

## === cell 9
test = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/test.csv')
test.head()

## === cell 10
if DEBUG:
    test = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')[:1000]
    PATH['test'] = PATH['train']
    test.head()

## === cell 11
i = np.random.randint(len(test))
row = test[i:i+1]
row

## === cell 12
spec = pd.read_parquet(PATH['test']+'spectrograms/'+str(row.spectrogram_id.values[0])+'.parquet')
LL = [c for c in spec.columns if 'LL' in c]
RL = [c for c in spec.columns if 'RL' in c]
LP = [c for c in spec.columns if 'LP' in c]
RP = [c for c in spec.columns if 'RP' in c]

## === cell 14


import cupy as cp
import numpy as np
import pandas as pd
import tqdm
import os
from cupyx.scipy.ndimage import gaussian_filter
from cupyx.scipy.signal import filtfilt, iirnotch
from cupyx.scipy.signal import spectrogram as cupyx_spectrogram
from scipy.signal import filtfilt as scipy_filtfilt, butter as scipy_butter
from skimage.transform import rescale, resize, downscale_local_mean

def create_spectrogram_with_cupy(eeg_data, eeg_id,
                                 low_cut_freq=0.7, high_cut_freq=20, order_band=5,
                                 nperseg=1500, noverlap=1483, nfft=2750,
                                 sigma_gaussian=0.7,
                                 mean_montage_names=4):
    electrode_pair_name_locations = {'LL': ['Fp1', 'F7', 'T3', 'T5', 'O1'],
                                     'RL': ['Fp2', 'F8', 'T4', 'T6', 'O2'],
                                     'LP': ['Fp1', 'F3', 'C3', 'P3', 'O1'],
                                     'RP': ['Fp2', 'F4', 'C4', 'P4', 'O2']}

    nyquist_freq = 0.5 * 200
    low_cut_freq_normalized = low_cut_freq / nyquist_freq
    high_cut_freq_normalized = high_cut_freq / nyquist_freq

    notch_coefficients = iirnotch(w0=60, Q=30, fs=200)
    sci_bandpass_coefficients = scipy_butter(order_band, [low_cut_freq_normalized, high_cut_freq_normalized],
                                             btype='band')
    
    spec_size = len(eeg_data)

    fs = 200

    processed_eeg = {}

    for i, (electrode_pair_name, electrode_locs) in enumerate(electrode_pair_name_locations.items()):
        processed_eeg[electrode_pair_name] = np.zeros(spec_size)

        for j in range(4):
            signal = cp.array(eeg_data[electrode_locs[j]].values - eeg_data[electrode_locs[j + 1]].values)

            mean_signal = cp.nanmean(signal)
            signal = cp.nan_to_num(signal, nan=mean_signal) if cp.isnan(signal).mean() < 1 else cp.zeros_like(signal)

            signal_filtered = filtfilt(*notch_coefficients, signal)
            signal_filtered = scipy_filtfilt(*sci_bandpass_coefficients, signal_filtered.get())  # HOTFIX

            frequencies, times, Sxx = cupyx_spectrogram(signal_filtered, fs, nperseg=nperseg, noverlap=noverlap,
                                                        nfft=nfft)
            valid_freq = (frequencies >= 0.59) & (frequencies <= 20)
            Sxx_filtered = Sxx[valid_freq, :]

            spectrogram_slice = cp.clip(Sxx_filtered, cp.exp(-4), cp.exp(6))
            spectrogram_slice = cp.log10(spectrogram_slice)

            normalization_epsilon = 1e-6
            mean = spectrogram_slice.mean(axis=(0, 1), keepdims=True)
            std = spectrogram_slice.std(axis=(0, 1), keepdims=True)
            spectrogram_slice = (spectrogram_slice - mean) / (std + normalization_epsilon)

            try:
                spectrogram[:, :, i] += spectrogram_slice
            except:
                h,w = spectrogram_slice.shape
                spectrogram = cp.zeros((h, w, 4), dtype='float32')
                spectrogram[:, :, i] = spectrogram_slice
            
            processed_eeg[f'{electrode_locs[j]}_{electrode_locs[j + 1]}'] = signal.get()
            processed_eeg[electrode_pair_name] += signal.get()

        if mean_montage_names > 0:
            spectrogram[:, :, i] /= mean_montage_names

    spec_numpy = gaussian_filter(spectrogram, sigma=sigma_gaussian).get() if sigma_gaussian > 0 else spectrogram.get()

    ekg_signal_filtered = filtfilt(*notch_coefficients, cp.array(eeg_data["EKG"].values))
    processed_eeg['EKG'] = scipy_filtfilt(*sci_bandpass_coefficients, ekg_signal_filtered.get())  # HOTFIX
    return spec_numpy, processed_eeg

## === cell 15
os.mkdir('/kaggle/working/custom')
for eeg_id in tqdm.tqdm(test.eeg_id.unique()):
    eeg = pd.read_parquet(PATH['test']+'eegs/'+str(eeg_id)+'.parquet')
    mask = np.isnan(eeg.values).max(1)
    spec,_ = create_spectrogram_with_cupy(eeg, eeg_id,
                                 low_cut_freq=0.7, high_cut_freq=20, order_band=5,
                                 nperseg= 500, noverlap= 200,
                                 nfft=1024,
                                 sigma_gaussian=0.7,
                                 mean_montage_names=4)
    _,w,_ = spec.shape
    mask = resize(mask,(w,1))
    spec[:,mask[:,0],:] = 0
    np.save('/kaggle/working/custom/'+str(eeg_id), spec)
    del eeg,mask,spec

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
CUDARuntimeError                          Traceback (most recent call last)
/tmp/ipykernel_11/4176021537.py in <cell line: 0>()
      3     eeg = pd.read_parquet(PATH['test']+'eegs/'+str(eeg_id)+'.parquet')
      4     mask = np.isnan(eeg.values).max(1)
----> 5     spec,_ = create_spectrogram_with_cupy(eeg, eeg_id,
      6                                  low_cut_freq=0.7, high_cut_freq=20, order_band=5,
      7                                  nperseg= 500, noverlap= 200,

/tmp/ipykernel_11/792041157.py in create_spectrogram_with_cupy(eeg_data, eeg_id, low_cut_freq, high_cut_freq, order_band, nperseg, noverlap, nfft, sigma_gaussian, mean_montage_names)
     34     # Bandpass and notch filter
     35     # bandpass_coefficients = butter(order_band, [low_cut_freq_normalized, high_cut_freq_normalized], btype='band')
---> 36     notch_coefficients = iirnotch(w0=60, Q=30, fs=200)
     37     sci_bandpass_coefficients = scipy_butter(order_band, [low_cut_freq_normalized, high_cut_freq_normalized],
     38                                              btype='band')

/usr/local/lib/python3.11/dist-packages/cupyx/scipy/signal/_iir_filter_design.py in iirnotch(w0, Q, fs)
    830     """
    831 
--> 832     return _design_notch_peak_filter(w0, Q, "notch", fs)
    833 
    834 

/usr/local/lib/python3.11/dist-packages/cupyx/scipy/signal/_iir_filter_design.py in _design_notch_peak_filter(w0, Q, ftype, fs)
    946     a = [1.0, -2.0 * gain * math.cos(w0), 2.0 * gain - 1.0]
    947 
--> 948     a = cupy.asarray(a)
    949     b = cupy.asarray(b)
    950 

/usr/local/lib/python3.11/dist-packages/cupy/_creation/from_data.py in asarray(a, dtype, order, blocking)
     86 
     87     """
---> 88     return _core.array(a, dtype, False, order, blocking=blocking)
     89 
     90 

cupy/_core/core.pyx in cupy._core.core.array()

cupy/_core/core.pyx in cupy._core.core.array()

cupy/_core/core.pyx in cupy._core.core._array_default()

cupy/_core/core.pyx in cupy._core.core._try_skip_h2d_copy()

cupy/cuda/device.pyx in cupy.cuda.device.get_device_id()

cupy_backends/cuda/api/runtime.pyx in cupy_backends.cuda.api.runtime.getDevice()

cupy_backends/cuda/api/runtime.pyx in cupy_backends.cuda.api.runtime.check_status()

CUDARuntimeError: cudaErrorInsufficientDriver: CUDA driver version is insufficient for CUDA runtime version

## === cell 17
class HMS_DS(torch.utils.data.Dataset):
    '''
    '''  
    def __init__(self, df):
        self.START = (10000 - 2048)//2
        self.data = df
        self.eeg = np.zeros((8,2048),dtype=np.float32)
        self.spec = np.zeros((1,256,256))
   
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        eeg = pd.read_parquet(PATH['test']+'eegs/'+str(row.eeg_id)+'.parquet')
        eeg = eeg.iloc[self.START:self.START+2048]
    
        mask = ~np.isnan(eeg['Fp1'])
        self.eeg[:,:] = 0
        
        self.eeg[0][mask] = eeg['Fp1'][mask] - eeg['T3'][mask]
        self.eeg[1][mask] = eeg['T3'][mask] - eeg['O1'][mask]

        self.eeg[2][mask] = eeg['Fp1'][mask] - eeg['C3'][mask]
        self.eeg[3][mask] = eeg['C3'][mask] - eeg['O1'][mask]

        self.eeg[4][mask] = eeg['Fp2'][mask] - eeg['C4'][mask]
        self.eeg[5][mask] = eeg['C4'][mask] - eeg['O2'][mask]

        self.eeg[6][mask] = eeg['Fp2'][mask] - eeg['T4'][mask]
        self.eeg[7][mask] = eeg['T4'][mask] - eeg['O2'][mask]

        self.eeg = np.clip(self.eeg,-1024, 1024)/32.0

        self.eeg = butter_lowpass_filter(self.eeg)

        eeg = torch.from_numpy(self.eeg).float().to(device)
        spec = pd.read_parquet(PATH['test']+'spectrograms/'+str(row.spectrogram_id)+'.parquet')
        
        spec = spec[LL + RL + LP + RP]
        
        t = 22
        spec = spec[LL + RL + LP + RP][t:t+256]
        
        self.spec[0,:,:64] = normalize(resize(spec[LP].values,(256,64)))
        self.spec[0,:,64:128] = normalize(resize(spec[LL].values,(256,64)))
        self.spec[0,:,128:-64] = normalize(resize(spec[RP].values,(256,64)))
        self.spec[0,:,-64:] = normalize(resize(spec[RL].values,(256,64)))
        
        mask = np.isnan(self.spec)
        self.spec[mask] = 0

        spec = torch.from_numpy(self.spec).float().to(device)
        custom_spec = np.load('/kaggle/working/custom/'+str(row.eeg_id)+'.npy')
        custom_spec = np.concatenate((custom_spec[:,:,3],
                                      custom_spec[:,:,2],
                                      custom_spec[:,:,1],
                                      custom_spec[:,:,0]))
        custom_spec = resize(custom_spec,(256,300))
        
        t = 22
        self.spec[0] = custom_spec[:,t:t+256]
                    
        custom_spec = torch.from_numpy(self.spec).float().to(device)

        return int(row.eeg_id),row.spectrogram_id,eeg,spec,custom_spec

## === cell 19
if len(test) > 1:
    submission = pd.DataFrame()
    
    ds = HMS_DS(test)
    dl = DataLoader(
        ds,
        batch_size=BS
    )
    
    eeg_ensemble = []
    for model in models['eeg']:
        eeg_ensemble.append(torch.load(model).eval())

    spec_ensemble = []
    for model in models['spec']:
        spec_ensemble.append(torch.load(model).eval())
        
    custom_ensemble = []
    for model in models['custom']:
        custom_ensemble.append(torch.load(model).eval())
        
    HMS_ensemble = []
    for model in models['HMS']:
        HMS_ensemble.append(
            torch.load(model).eval()
        )
        
    with torch.no_grad():
        PREDS = torch.zeros((BS,20,6),device=device)
        for eeg_id,spectrogram_id,eegs,specs,custom_specs in dl:
            N = len(eegs)
            
            i = 0
            for model in eeg_ensemble:
                PREDS[:N,i] = torch.softmax(model(eegs),-1)
                i += 1

            for model in spec_ensemble:
                PREDS[:N,i] = torch.softmax(model(specs),-1)
                i += 1

            for model in custom_ensemble:
                PREDS[:N,i] = torch.softmax(model(custom_specs),-1)
                i += 1
               
            for model in HMS_ensemble:
                PREDS[:N,i] = torch.softmax(model([eegs,specs,custom_specs]),-1)
                i += 1
                
            mu = torch.softmax(PREDS[:N].mean(1),-1).cpu()
            sigma = PREDS[:N].std(1).cpu()
            sigma = (mu*sigma).sum(-1)/(mu.sum(-1))
            
            df = {'eeg_id':eeg_id,'spectrogram_id':spectrogram_id}
            for i in range(6):
                df[votes[i]] = mu[:,i].cpu()
                
            df['sigma'] = sigma
            
            submission = pd.concat([submission,pd.DataFrame(df)],ignore_index=True)

print(submission.head())

## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/703414019.py in <cell line: 0>()
     10     eeg_ensemble = []
     11     for model in models['eeg']:
---> 12         eeg_ensemble.append(torch.load(model).eval())
     13 
     14     spec_ensemble = []

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

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/eegs-cv-kaggle/EEGS_CNN_0'

## === cell 20
TH = np.percentile(submission['sigma'],TH)
pseudo_labels = submission[submission['sigma'] < TH].copy()
pseudo_labels[votes] *= pseudo_votes#10
pseudo_labels.head()

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1313639883.py in <cell line: 0>()
----> 1 TH = np.percentile(submission['sigma'],TH)
      2 # The predictions with sigma under threshold will be taken as confident pseudolabels
      3 # with a total weigth of pseudo_votes votes
      4 pseudo_labels = submission[submission['sigma'] < TH].copy()
      5 pseudo_labels[votes] *= pseudo_votes#10

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    415                 raise KeyError(key) from err
    416         if isinstance(key, Hashable):
--> 417             raise KeyError(key)
    418         self._check_indexing_error(key)
    419         raise KeyError(key)

KeyError: 'sigma'

## === cell 22
df = pd.read_csv('/kaggle/input/hms-harmful-brain-activity-classification/train.csv')
df['origin'] = 'train'
df.head()

## === cell 23
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(2024)

## === cell 24
m = .55# Define the majority criteria
df.expert_consensus = 'T'
S = "df.loc[(df.seizure_vote > m*(df.seizure_vote + df.lpd_vote))*(df.seizure_vote > m*(df.seizure_vote + df.gpd_vote))*(df.seizure_vote > m*(df.seizure_vote + df.lrda_vote))*(df.seizure_vote > m*(df.seizure_vote + df.grda_vote))*(df.seizure_vote > m*(df.seizure_vote + df.other_vote)),'expert_consensus'] = 'seizure'"

exec(S)
for a in ['lpd','gpd','lrda','grda','other']:
    SWAP = S.replace(a,'SWAP')
    SWAP = SWAP.replace('seizure',a)
    exec(SWAP.replace('SWAP','seizure'))

consensus = ['seizure', 'lpd', 'gpd', 'lrda', 'grda', 'other', 'T']
for c in consensus:
    c_df = df[['eeg_id']+votes].loc[df.expert_consensus == c].groupby('eeg_id').mean()
    values,counts = np.unique(c_df[votes].sum(1).values,return_counts=True)
    plt.title(c + ': ' + str(len(c_df)))
    plt.bar(values,counts)
    plt.show()
    print(c_df[votes].head())

## === cell 25
pseudo_labels['expert_consensus'] = 'T'
S = S.replace('df','pseudo_labels')

exec(S)
for a in ['lpd','gpd','lrda','grda','other']:
    SWAP = S.replace(a,'SWAP')
    SWAP = SWAP.replace('seizure',a)
    exec(SWAP.replace('SWAP','seizure'))

for c in consensus:
    c_df = pseudo_labels[['eeg_id']+votes].loc[pseudo_labels['expert_consensus'] == c].groupby('eeg_id').mean()
    print(c,':',c_df)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2008825290.py in <cell line: 0>()
----> 1 pseudo_labels['expert_consensus'] = 'T'
      2 S = S.replace('df','pseudo_labels')
      3 
      4 exec(S)
      5 for a in ['lpd','gpd','lrda','grda','other']:

NameError: name 'pseudo_labels' is not defined

## === cell 26
pseudo_labels['origin'] = 'test'
pseudo_labels[['spectrogram_label_offset_seconds','eeg_label_offset_seconds']] = 0
pseudo_labels.head()

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3388872818.py in <cell line: 0>()
----> 1 pseudo_labels['origin'] = 'test'
      2 pseudo_labels[['spectrogram_label_offset_seconds','eeg_label_offset_seconds']] = 0
      3 pseudo_labels.head()

NameError: name 'pseudo_labels' is not defined

## === cell 27
class HMS_TRAIN_DS(torch.utils.data.Dataset):
    '''
    '''  
    def __init__(self, df, W = 1024, VALID = False):
        self.W = W
        self.VALID = VALID
        self.START = (10000 - 2048)//2
        self.data = np.array(df[['spectrogram_id',
                                 'eeg_id','expert_consensus',
                                 'spectrogram_label_offset_seconds',
                                 'eeg_label_offset_seconds','origin']+votes].groupby(['eeg_id','expert_consensus']), dtype=object)
        if VALID:
            self.eeg = np.zeros((8,2048),dtype=np.float32)
        else:
            self.eeg = np.zeros((8,1024),dtype=np.float32)
        self.spec = np.zeros((1,256,256))
   
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        df = self.data[idx][1]
        if self.VALID:
            row = df.iloc[len(df)//2]
        else:
            row = df.iloc[np.random.randint(len(df))]
        eeg = eegs[row.origin][row.eeg_id]
        START = 200*int(row.eeg_label_offset_seconds) + self.START 
        if not self.VALID:
            if self.W < 2048:
                START += np.random.randint(2048 - self.W)
                eeg = eeg.iloc[START:START+self.W]         
        else:
            eeg = eeg.iloc[START:START+2048]
    
        mask = ~np.isnan(eeg['Fp1'])
        self.eeg[:,:] = 0
        
        self.eeg[0][mask] = eeg['Fp1'][mask] - eeg['T3'][mask]
        self.eeg[1][mask] = eeg['T3'][mask] - eeg['O1'][mask]

        self.eeg[2][mask] = eeg['Fp1'][mask] - eeg['C3'][mask]
        self.eeg[3][mask] = eeg['C3'][mask] - eeg['O1'][mask]

        self.eeg[4][mask] = eeg['Fp2'][mask] - eeg['C4'][mask]
        self.eeg[5][mask] = eeg['C4'][mask] - eeg['O2'][mask]

        self.eeg[6][mask] = eeg['Fp2'][mask] - eeg['T4'][mask]
        self.eeg[7][mask] = eeg['T4'][mask] - eeg['O2'][mask]

        self.eeg = np.clip(self.eeg,-1024, 1024)/32.0

        self.eeg = butter_lowpass_filter(self.eeg)

        eeg = torch.from_numpy(self.eeg).float().to(device)
        spec = pd.read_parquet(PATH[row['origin']]+'spectrograms/'+str(row.spectrogram_id)+'.parquet')

        spec_offset = int( row.spectrogram_label_offset_seconds )
        spec = spec[LL + RL + LP + RP].loc[(spec.time>=spec_offset)
                     &(spec.time<spec_offset+600)]
        
        t = 22
        if not self.VALID:
            t += int(np.clip(np.random.normal(0,11),-22,22))
        spec = spec[LL + RL + LP + RP][t:t+256]
        
        self.spec[0,:,:64] = normalize(resize(spec[LP].values,(256,64)))
        self.spec[0,:,64:128] = normalize(resize(spec[LL].values,(256,64)))
        self.spec[0,:,128:-64] = normalize(resize(spec[RP].values,(256,64)))
        self.spec[0,:,-64:] = normalize(resize(spec[RL].values,(256,64)))
        
        mask = np.isnan(self.spec)
        if not self.VALID:
            self.spec *= np.random.normal(1,.0001)
            self.spec += np.random.normal(0,.0001)
        self.spec[mask] = 0
        if not self.VALID:
            if np.random.rand(1) > mask.sum()/(2*256*256):
                w = int(np.clip(np.random.normal(256/5,256/20),0,512/5))
                t = (256 - w)/2
                t = int(np.clip(np.random.normal(0,t/2),-2*t,2*t))
            
                if t < 0:
                    self.spec[:,t-w:t] = 0
                else:
                    self.spec[:,t:t+w] = 0

        spec = torch.from_numpy(self.spec).float().to(device)
        if row['origin'] == 'train':
            custom_spec = np.load('/kaggle/input/custom-specs/custom/'+str(row.eeg_id)+'.npy')
        else:
            custom_spec = np.load('/kaggle/working/custom/'+str(row.eeg_id)+'.npy')
        spec_offset = int( 2*row.eeg_label_offset_seconds/3 )
        custom_spec = custom_spec[:,spec_offset:spec_offset+32,:]
        custom_spec = np.concatenate((custom_spec[:,:,3],
                                      custom_spec[:,:,2],
                                      custom_spec[:,:,1],
                                      custom_spec[:,:,0]))
        custom_spec = resize(custom_spec,(256,300))
        
        t = 22
        if not self.VALID:
            t += int(np.clip(np.random.normal(0,11),-22,22))
        
        self.spec[0] = custom_spec[:,t:t+256]

        mask = np.isnan(self.spec)
        if not self.VALID:
            self.spec *= np.random.normal(1,.0001)
            self.spec += np.random.normal(0,.0001)
        self.spec[mask] = 0
        if not self.VALID:
            if np.random.rand(1) > mask.sum()/(2*256*256):
                w = int(np.clip(np.random.normal(256/5,256/20),0,512/5))
                t = (256 - w)/2
                t = int(np.clip(np.random.normal(0,t/2),-2*t,2*t))
            
                if t < 0:
                    self.spec[:,:,t-w:t] = 0
                else:
                    self.spec[:,:,t:t+w] = 0
                    
        custom_spec = torch.from_numpy(self.spec).float().to(device)
        labels = row[votes].values
        if label_aug and not self.VALID:
            v = [0]*int(labels[0])+[1]*int(labels[1])+[2]*int(labels[2])+[3]*int(labels[3])+[4]*int(labels[4])+[5]*int(labels[5])
            v = np.array(random.sample(v, max([1,len(v) - int(abs(np.random.normal(0,len(v)/8)))])))
            labels[0] = np.sum(v == 0)
            labels[1] = np.sum(v == 1)
            labels[2] = np.sum(v == 2)
            labels[3] = np.sum(v == 3)
            labels[4] = np.sum(v == 4)
            labels[5] = np.sum(v == 5)
        
        labels = torch.from_numpy(labels.astype(np.float32)).to(device)

        return [eeg,spec,custom_spec],labels

## === cell 28
class SemisupervisedKLDiv(nn.KLDivLoss):
    def __init__(
            self,
            min_v=1# The minium votes to be accepted
        ):
        super().__init__(reduce=False)
        self.min_v = torch.tensor(start_min_v,dtype=torch.float).to(device)
        self.p = torch.tensor(start_p,dtype=torch.float).to(device)
        self.step = 0

    def __steps__(self,steps):
        self.steps = steps

    def set_min_v(self):
        self.step += 1
        if self.step <= self.steps: self.min_v = nt(start_min_v,end_min_v,self.step,self.steps)

    def set_p(self):
        self.step += 1
        if self.step <= self.steps: self.p = nt(start_p,end_p,self.step,self.steps)

    def forward(
            self,
            y,# Raw predictions
            t # Raw targets
        ):
        v = t.sum(-1,keepdim=True)
        mask = v[:,0] < self.min_v
        t[mask] += (self.min_v - v[mask])*(self.p*torch.softmax(y[mask].detach(),-1) + (1 - self.p)*torch.softmax(t[mask],-1))
        v[mask] = self.min_v
        t /= v
        y = nn.functional.log_softmax(y,  dim=1)
        loss = super().forward(y, t).sum(-1,keepdim=True)
        loss = loss*v
        loss = loss.sum()/v.sum()
        return loss

## === cell 29
def nt(nmin,nmax,tcur,tmax):
    return nmax - .5*(nmax-nmin)*(1+np.cos(tcur*np.pi/tmax))

## === cell 30
def cb(self):
    learn.loss_func.set_min_v()
    learn.loss_func.set_p()
min_v_cb = Callback(before_step=cb)

## === cell 31
seed_everything(2024)
splits = {}
for c in consensus:
    ID = df.loc[df.expert_consensus == c][CV].unique()
    splits[c] = []
    for t,v in KFold(N_FOLDS).split(ID):
        splits[c].append([ID[t],ID[v]])

## === cell 32
seed_everything(2024)
pseudo_labels_splits = {}
for c in consensus:
    ID = pseudo_labels.loc[pseudo_labels['expert_consensus'] == c][CV].unique()
    if len(ID) > N_FOLDS:
        pseudo_labels_splits[c] = []
        for t,v in KFold(N_FOLDS).split(ID):
            pseudo_labels_splits[c].append([ID[t],ID[v]])

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2826089542.py in <cell line: 0>()
      2 pseudo_labels_splits = {}
      3 for c in consensus:
----> 4     ID = pseudo_labels.loc[pseudo_labels['expert_consensus'] == c][CV].unique()
      5     if len(ID) > N_FOLDS:
      6         pseudo_labels_splits[c] = []

NameError: name 'pseudo_labels' is not defined

## === cell 33
if len(test) > 1:
    eegs = {'train':{}, 'test':{}}
    for eeg_id in tqdm.tqdm(df[df[votes].sum(1) >= 10].eeg_id.unique()):
        eegs['train'][eeg_id] = pd.read_parquet(PATH['train']+'eegs/'+str(eeg_id)+'.parquet')
    
    for eeg_id in tqdm.tqdm(pseudo_labels.eeg_id.unique()):
        eegs['test'][eeg_id] = pd.read_parquet(PATH['test']+'eegs/'+str(eeg_id)+'.parquet')
    
    for f in FOLDS:
        seed_everything(2024)
        print('FOLD: ',f)
        t = pd.DataFrame()
        v = pd.DataFrame()
        for c in splits:
            c_df = df.loc[df.expert_consensus == c]
            t = pd.concat([t,c_df[c_df[CV].isin(splits[c][f][0])]])
            v = pd.concat([v,c_df[c_df[CV].isin(splits[c][f][1])]])
        t = t[t[votes].sum(1) >= 10]
        v = v[v[votes].sum(1) >= 10]
        
        for c in pseudo_labels_splits:
            c_df = pseudo_labels.loc[pseudo_labels['expert_consensus'] == c]
            t = pd.concat([t,c_df[c_df[CV].isin(pseudo_labels_splits[c][f][0])]])
            v = pd.concat([v,c_df[c_df[CV].isin(pseudo_labels_splits[c][f][1])]])

        tds = HMS_TRAIN_DS(t)
        vds = HMS_TRAIN_DS(v,VALID=True)

        tdl = DataLoader(
            tds,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        vdl = DataLoader(
            vds,
            batch_size=2*batch_size
        )
        dls = DataLoaders(tdl,vdl)

        model = torch.load(models['HMS'][f])

        learn = Learner(
            dls,
            model,
            lr=LR,
            loss_func=SemisupervisedKLDiv(),
            cbs=[
                GradientClip(3.0),
                SaveModelCallback(),
                ShowGraphCallback(),
                min_v_cb
            ]
        )
        learn.loss_func.__steps__((learn.dls.dataset.__len__()//batch_size)*EPOCHS)
        learn.fit_one_cycle(EPOCHS)

        torch.save(model,'HMSmodel_'+str(f))
        del tdl,vdl,tds,vds,dls,learn,model
        gc.collect()

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/20367859.py in <cell line: 0>()
      4         eegs['train'][eeg_id] = pd.read_parquet(PATH['train']+'eegs/'+str(eeg_id)+'.parquet')
      5 
----> 6     for eeg_id in tqdm.tqdm(pseudo_labels.eeg_id.unique()):
      7         eegs['test'][eeg_id] = pd.read_parquet(PATH['test']+'eegs/'+str(eeg_id)+'.parquet')
      8 

NameError: name 'pseudo_labels' is not defined

## === cell 35
if len(test) > 1:
    del submission,PREDS
    submission = pd.DataFrame()
    
    ds = HMS_DS(test)
    dl = DataLoader(
        ds,
        batch_size=BS
    )
    
    Last_Bullet = [
        '/kaggle/working/HMSmodel_0',
        '/kaggle/working/HMSmodel_1',
        '/kaggle/working/HMSmodel_2',
        '/kaggle/working/HMSmodel_3',
        '/kaggle/working/HMSmodel_4'
    ]
    HMS_ensemble = []
    for model in Last_Bullet:
        HMS_ensemble.append(
            torch.load(model).eval()
        )
        
    with torch.no_grad():
        PREDS = torch.zeros((BS,6),device=device)
        for eeg_id,spectrogram_id,eegs,specs,custom_specs in dl:
            N = len(eegs)
            PREDS[:N,:] = 0 
            for model in HMS_ensemble:
                PREDS[:N] += torch.softmax(model([eegs,specs,custom_specs]),-1)
                
            PREDS[:N] /= PREDS[:N].sum(-1,keepdim=True)
            
            df = {'eeg_id':eeg_id,'spectrogram_id':spectrogram_id}
            for i in range(6):
                df[votes[i]] = PREDS[:N,i].cpu()
            
            submission = pd.concat([submission,pd.DataFrame(df)],ignore_index=True)
            
    submission = submission[['eeg_id']+votes]

print(submission.head())

submission.to_csv('submission.csv', index=False)
shutil.rmtree('/kaggle/working/custom')

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3785549349.py in <cell line: 0>()
      1 if len(test) > 1:
----> 2     del submission,PREDS
      3     submission = pd.DataFrame()
      4 
      5     ds = HMS_DS(test)

NameError: name 'PREDS' is not defined

## === cell 36
submission[votes].sum(1).unique()

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3718542473.py in <cell line: 0>()
----> 1 submission[votes].sum(1).unique()

NameError: name 'submission' is not defined
