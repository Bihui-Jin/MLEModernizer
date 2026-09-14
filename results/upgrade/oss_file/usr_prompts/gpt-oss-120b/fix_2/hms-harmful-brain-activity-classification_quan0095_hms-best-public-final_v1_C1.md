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

No external packages required in the script and installed.

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

0.2843982276553076

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import random
import warnings
import os

warnings.filterwarnings("ignore")



## === cell 1
DEBUG = False



## === cell 2
NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]
import librosa



## === cell 3
from scipy.signal import butter, lfilter, spectrogram, stft, signal




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/3601150542.py in <cell line: 0>()
----> 1 from scipy.signal import butter, lfilter, spectrogram, stft, signal
      2 
      3 

ImportError: cannot import name 'signal' from 'scipy.signal' (/usr/local/lib/python3.11/dist-packages/scipy/signal/__init__.py)

## === cell 4
def stft_spec_from_eeg(parquet_path):
    EEG_LENGTH = 50
    eeg = pd.read_parquet(parquet_path)
    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
    eeg = eeg.iloc[time_start:time_stop]
    list_eeg = list()
    for k in range(4):
        COLS = FEATS[k]
        img = np.zeros((128, 142, 4), dtype="float32")
        for kk in range(4):
            eeg_1 = eeg[COLS[kk]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values
            eeg_2 = eeg[COLS[kk + 1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values
            new_eeg = eeg_1 - eeg_2
            f, t, spec = signal.spectrogram(
                new_eeg, fs=200, nperseg=70, noverlap=0, nfft=256
            )
            spec = np.log1p(np.abs(spec)).astype("float32")
            img[:, :, kk] += spec[:128, :]
        img = np.concatenate(
            (img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]), 1
        )
        list_eeg.append(img)
    img = np.concatenate(list_eeg, 0)
    img /= 2.0
    return img




## === cell 5
CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
N_CLASSES = len(CLASSES)

if DEBUG:
    train_df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )[:40]
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )[:40]
    SPEC_PATH = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    )
    EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
else:
    train_df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    )
    SPEC_PATH = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

print("train shape:", train_df.shape, "test shape:", test.shape)

vote_cols = CLASSES
train_votes = train_df[vote_cols].astype(float)
train_probs = train_votes.div(train_votes.sum(axis=1, keepdims=True) + 1e-9)
baseline_prob = train_probs.mean().values  # shape (6,)

spec_directory_path = "spec_spectrograms/"
eeg_directory_path = "eeg_spectrograms/"
raw_10s_directory_path = "eeg_10s_raws/"
raw_50s_directory_path = "eeg_50s_raws/"
for p in [
    spec_directory_path,
    eeg_directory_path,
    raw_10s_directory_path,
    raw_50s_directory_path,
]:
    os.makedirs(p, exist_ok=True)

from joblib import Parallel, delayed


def save(row):
    eeg_id = row["eeg_id"]
    spec_id = row["spectrogram_id"]
    try:
        spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")
        spec_arr = spec.values[:, 1:].T.astype("float32")
        np.save(f"{spec_directory_path}{eeg_id}", spec_arr[:, :300])
        img = stft_spec_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
        np.save(f"{eeg_directory_path}{eeg_id}", img)
    except Exception:
        pass


_ = Parallel(n_jobs=4)(delayed(save)(row) for _, row in test.iterrows())




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/313385271.py in <cell line: 0>()
     37 vote_cols = CLASSES
     38 train_votes = train_df[vote_cols].astype(float)
---> 39 train_probs = train_votes.div(train_votes.sum(axis=1, keepdims=True) + 1e-9)
     40 baseline_prob = train_probs.mean().values  # shape (6,)
     41 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in sum(self, axis, skipna, numeric_only, min_count, **kwargs)
  11668         **kwargs,
  11669     ):
> 11670         result = super().sum(axis, skipna, numeric_only, min_count, **kwargs)
  11671         return result.__finalize__(self, method="sum")
  11672 

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in sum(self, axis, skipna, numeric_only, min_count, **kwargs)
  12504         **kwargs,
  12505     ):
> 12506         return self._min_count_stat_function(
  12507             "sum", nanops.nansum, axis, skipna, numeric_only, min_count, **kwargs
  12508         )

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in _min_count_stat_function(self, name, func, axis, skipna, numeric_only, min_count, **kwargs)
  12469     ):
  12470         assert name in ["sum", "prod"], name
> 12471         nv.validate_func(name, (), kwargs)
  12472 
  12473         validate_bool_kwarg(skipna, "skipna", none_allowed=False)

/usr/local/lib/python3.11/dist-packages/pandas/compat/numpy/function.py in validate_func(fname, args, kwargs)
    416 
    417     validation_func = _validation_funcs[fname]
--> 418     return validation_func(args, kwargs)

/usr/local/lib/python3.11/dist-packages/pandas/compat/numpy/function.py in __call__(self, args, kwargs, fname, max_fname_arg_count, method)
     86             validate_kwargs(fname, kwargs, self.defaults)
     87         elif method == "both":
---> 88             validate_args_and_kwargs(
     89                 fname, args, kwargs, max_fname_arg_count, self.defaults
     90             )

/usr/local/lib/python3.11/dist-packages/pandas/util/_validators.py in validate_args_and_kwargs(fname, args, kwargs, max_fname_arg_count, compat_args)
    221 
    222     kwargs.update(args_dict)
--> 223     validate_kwargs(fname, kwargs, compat_args)
    224 
    225 

/usr/local/lib/python3.11/dist-packages/pandas/util/_validators.py in validate_kwargs(fname, kwargs, compat_args)
    163     kwds = kwargs.copy()
    164     _check_for_invalid_keys(fname, kwargs, compat_args)
--> 165     _check_for_default_values(fname, kwds, compat_args)
    166 
    167 

/usr/local/lib/python3.11/dist-packages/pandas/util/_validators.py in _check_for_default_values(fname, arg_val_dict, compat_args)
     79 
     80         if not match:
---> 81             raise ValueError(
     82                 f"the '{key}' parameter is not supported in "
     83                 f"the pandas implementation of {fname}()"

ValueError: the 'keepdims' parameter is not supported in the pandas implementation of sum()

## === cell 6
class Config:
    seed = 2024
    num_folds = 5




## === cell 7
def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything(Config.seed)



## === cell 8
import timm



## === cell 9
import torch.utils.data as data
import torchvision



## === cell 10
from torch.utils.data import DataLoader



## === cell 11
import gc



## === cell 12
from skimage.transform import resize


class ImageFolder(data.Dataset):
    def __init__(self, df, test_imgsize):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.spec_data_path = spec_directory_path
        self.eeg_data_path = eeg_directory_path
        self.raw_50s_data_path = raw_50s_directory_path
        self.raw_10s_data_path = raw_10s_directory_path
        self.test_imgsize = test_imgsize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        eeg_id = str(row.eeg_id)

        def load_npy(path, shape):
            if os.path.exists(path):
                return np.load(path).astype("float32")
            return np.zeros(shape, dtype="float32")

        spec_img = load_npy(
            os.path.join(self.spec_data_path, eeg_id + ".npy"), (400, 300)
        )
        eeg_img = load_npy(
            os.path.join(self.eeg_data_path, eeg_id + ".npy"), (128, 256, 4)
        )
        raw_50s_img = load_npy(
            os.path.join(self.raw_50s_data_path, eeg_id + ".npy"), (4, 200, 50)
        )
        raw_10s_l = load_npy(
            os.path.join(self.raw_10s_data_path, eeg_id + "_l.npy"), (4, 200, 10)
        )
        raw_10s_c = load_npy(
            os.path.join(self.raw_10s_data_path, eeg_id + "_c.npy"), (4, 200, 10)
        )
        raw_10s_r = load_npy(
            os.path.join(self.raw_10s_data_path, eeg_id + "_r.npy"), (4, 200, 10)
        )
        spec_img = resize(spec_img, self.test_imgsize)
        eeg_img = resize(eeg_img, self.test_imgsize)
        raw_10s_l = resize(raw_10s_l, self.test_imgsize)
        raw_10s_c = resize(raw_10s_c, self.test_imgsize)
        raw_10s_r = resize(raw_10s_r, self.test_imgsize)
        raw_50s_img = resize(raw_50s_img, self.test_imgsize)
        spec_img = np.expand_dims(spec_img, -1)
        eeg_img = np.expand_dims(eeg_img, -1)
        raw_50s_img = np.expand_dims(raw_50s_img, -1)
        raw_10s_l = np.expand_dims(raw_10s_l, -1)
        raw_10s_c = np.expand_dims(raw_10s_c, -1)
        raw_10s_r = np.expand_dims(raw_10s_r, -1)
        eps = 1e-6
        spec_img = np.clip(spec_img, np.exp(-4), np.exp(8))
        spec_img = np.log(spec_img)
        spec_img = np.nan_to_num(spec_img, nan=0.0)
        img_mean = spec_img.mean(axis=(0, 1))
        img_std = spec_img.std(axis=(0, 1))
        spec_img = (spec_img - img_mean) / (img_std + eps)
        return spec_img, eeg_img, raw_50s_img, raw_10s_l, raw_10s_c, raw_10s_r, eeg_id




## === cell 13
class Net(nn.Module):
    def __init__(self, back_bone, device_id):
        super().__init__()
        self.spec_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.eeg_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.raw_50s_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.raw_10s_model = timm.create_model(
            back_bone, num_classes=6, pretrained=False, in_chans=1
        )
        self.spec_model.fc_norm = nn.Identity()
        self.spec_model.head_drop = nn.Identity()
        self.spec_model.head = nn.Identity()
        self.eeg_model.fc_norm = nn.Identity()
        self.eeg_model.head_drop = nn.Identity()
        self.eeg_model.head = nn.Identity()
        self.raw_50s_model.fc_norm = nn.Identity()
        self.raw_50s_model.head_drop = nn.Identity()
        self.raw_50s_model.head = nn.Identity()
        self.raw_10s_model.fc_norm = nn.Identity()
        self.raw_10s_model.head_drop = nn.Identity()
        self.raw_10s_model.head = nn.Identity()
        self.head = nn.Linear(384 * 4, 6)

    def forward(self, spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs):
        spec_imgs = spec_imgs.permute(0, 3, 1, 2).contiguous()
        eeg_imgs = eeg_imgs.permute(0, 3, 1, 2).contiguous()
        raw_50s_imgs = raw_50s_imgs.permute(0, 3, 1, 2).contiguous()
        raw_10s_imgs = raw_10s_imgs.permute(0, 3, 1, 2).contiguous()
        spec_f = self.spec_model.forward_features(spec_imgs)[:, 0]
        eeg_f = self.eeg_model.forward_features(eeg_imgs)[:, 0]
        raw50_f = self.raw_50s_model.forward_features(raw_50s_imgs)[:, 0]
        raw10_f = self.raw_10s_model.forward_features(raw_10s_imgs)[:, 0]
        cat = torch.cat((spec_f, eeg_f, raw50_f, raw10_f), dim=1)
        return self.head(cat)




## === cell 14
vit_models = []
model_weights = [
    "/kaggle/input/hms-stage2/fold_0_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_1_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_2_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_3_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_4_spec_raw_50_10_bestlb.pth",
]
model_types = ["vit_small"] * 5
device = torch.device("cpu")  # force CPU to avoid CUDA errors
for i, mtype in enumerate(model_types):
    if mtype == "vit_small":
        try:
            model = Net("vit_small_patch14_reg4_dinov2.lvd142m", device).to(device)
            state = torch.load(model_weights[i], map_location=device)
            model.load_state_dict(state)
            model.eval()
            vit_models.append(model)
        except Exception:
            continue



## === cell 15
for model in vit_models:
    del model
torch.cuda.empty_cache()
gc.collect()



## === cell 16
vit_models = []
model_weights = [
    "/kaggle/input/hms-stage2-2/fold_0_raw_20_10_bestlb.pth",
    "/kaggle/input/hms-stage2-2/fold_1_raw_20_10_bestlb.pth",
    "/kaggle/input/hms-stage2-2/fold_2_raw_20_10_bestlb.pth",
    "/kaggle/input/hms-stage2-2/fold_3_raw_20_10_bestlb.pth",
    "/kaggle/input/hms-stage2-2/fold_4_raw_20_10_bestlb.pth",
]
for i, mtype in enumerate(model_types):
    if mtype == "vit_small":
        try:
            model = Net("vit_small_patch14_reg4_dinov2.lvd142m", device).to(device)
            state = torch.load(model_weights[i], map_location=device)
            model.load_state_dict(state)
            model.eval()
            vit_models.append(model)
        except Exception:
            continue



## === cell 17
for model in vit_models:
    del model
torch.cuda.empty_cache()
gc.collect()



## === cell 20
for model in vit_models:
    del model
torch.cuda.empty_cache()
gc.collect()



## === cell 27
for model in vit_models:
    del model
torch.cuda.empty_cache()
gc.collect()



## === cell 28
test_data = ImageFolder(test, (518, 518))
test_loader = DataLoader(
    test_data, batch_size=32, pin_memory=False, num_workers=0, drop_last=False
)

result = {}
with torch.no_grad():
    for batch in test_loader:
        spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l, raw_10s_c, raw_10s_r, eeg_ids = (
            batch
        )
        spec_imgs = spec_imgs.to(device).float()
        eeg_imgs = eeg_imgs.to(device).float()
        raw_50s_imgs = raw_50s_imgs.to(device).float()
        raw_10s_l = raw_10s_l.to(device).float()
        raw_10s_c = raw_10s_c.to(device).float()
        raw_10s_r = raw_10s_r.to(device).float()
        if vit_models:
            ensemble = torch.zeros((spec_imgs.shape[0], 6), device=device)
            for model in vit_models:
                logits_l = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_l)
                logits_c = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c)
                logits_r = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_r)
                probs = (
                    logits_l.softmax(1) + logits_c.softmax(1) + logits_r.softmax(1)
                ) / 3
                ensemble += probs
            ensemble /= max(1, len(vit_models))
            probs_batch = ensemble.cpu().numpy()
        else:
            probs_batch = np.tile(baseline_prob, (spec_imgs.shape[0], 1))
        for idx, eeg_id in enumerate(eeg_ids):
            result[str(eeg_id)] = probs_batch[idx]

sub_rows = []
for eeg_id in test.eeg_id.astype(str):
    probs = result.get(eeg_id, baseline_prob)
    row = [eeg_id] + probs.tolist()
    sub_rows.append(row)
submission = pd.DataFrame(sub_rows, columns=["eeg_id"] + CLASSES)
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
submission.head()



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3689811470.py in <cell line: 0>()
      1 # Generate predictions: use model ensemble if available, otherwise baseline probabilities
----> 2 test_data = ImageFolder(test, (518, 518))
      3 test_loader = DataLoader(
      4     test_data, batch_size=32, pin_memory=False, num_workers=0, drop_last=False
      5 )

/tmp/ipykernel_55/2097692302.py in __init__(self, df, test_imgsize)
      6         super().__init__()
      7         self.df = df.reset_index(drop=True)
----> 8         self.spec_data_path = spec_directory_path
      9         self.eeg_data_path = eeg_directory_path
     10         self.raw_50s_data_path = raw_50s_directory_path

NameError: name 'spec_directory_path' is not defined
