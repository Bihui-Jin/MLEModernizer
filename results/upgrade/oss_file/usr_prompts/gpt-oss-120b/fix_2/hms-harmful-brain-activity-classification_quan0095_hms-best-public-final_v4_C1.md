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

0.284743286928261

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
DEBUG = False



## === cell 1
NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]
import librosa



## === cell 2
from scipy.signal import butter, lfilter




## === cell 3
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
            del eeg_1
            del eeg_2
            fs = 200
            nperseg = 70
            noverlap = 0
            f, t, spec = signal.spectrogram(
                new_eeg, fs, nperseg=nperseg, noverlap=noverlap, nfft=256
            )

            spec = np.abs(spec)
            spec = np.log1p(spec).astype("float32")

            img[:, :, kk] += spec[:128, :]
        img = np.concatenate(
            (img[:, :, 0], img[:, :, 1], img[:, :, 2], img[:, :, 3]), 1
        )
        list_eeg.append(img)
    img = np.concatenate(list_eeg, 0)
    img /= 2.0
    return img




## === cell 4
NAMES = ["LL", "LP", "RP", "RR"]

SFREQ = 200

filter_range = [0.5, 40]

RAW_FEATS = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

from scipy import signal

b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")


def raw10seeg_from_eeg(parquet_path, eeg_id):
    EEG_LENGTH = 10
    raw_eeg = pd.read_parquet(parquet_path)

    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = list()
    for region in RAW_FEATS.keys():
        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values

            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values

            new_eeg = eeg_1 - eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024)
            eeg[chan_i, :] = new_eeg

        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            [eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]], 1
        )
        list_eeg.append(eeg)

    eeg_c = np.concatenate(list_eeg, 1)
    eeg_c /= 104
    return eeg_c


def raw50seeg_from_eeg(parquet_path):
    EEG_LENGTH = 50
    raw_eeg = pd.read_parquet(parquet_path)
    time_temp = 0
    time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200)
    time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)

    eeg_default = raw_eeg.loc[time_start : (time_stop - 1), :].reset_index(drop=True)
    list_eeg = list()
    for region in RAW_FEATS.keys():

        eeg = np.zeros((len(RAW_FEATS[region]), eeg_default.shape[0]), dtype=np.float32)
        for chan_i, chan in enumerate(RAW_FEATS[region]):
            eeg_1 = eeg_default.loc[:, chan.split("-")[0]]
            mean_value = eeg_1.mean()
            eeg_1.fillna(value=mean_value, inplace=True)
            eeg_1 = eeg_1.values

            eeg_2 = eeg_default.loc[:, chan.split("-")[1]]
            mean_value = eeg_2.mean()
            eeg_2.fillna(value=mean_value, inplace=True)
            eeg_2 = eeg_2.values

            new_eeg = eeg_1 - eeg_2
            del eeg_1
            del eeg_2
            new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
            new_eeg = np.clip(new_eeg, -1024, 1024).astype("float32")
            eeg[chan_i, :] = new_eeg

        eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
        eeg = np.concatenate(
            (eeg[0, :, :], eeg[1, :, :], eeg[2, :, :], eeg[3, :, :]), 1
        )
        list_eeg.append(eeg)

    eeg = np.concatenate(list_eeg, 1)
    eeg /= 104

    return eeg




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2454804814.py in <cell line: 0>()
     14 from scipy import signal
     15 
---> 16 b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
     17 
     18 

NameError: name 'np' is not defined

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

if DEBUG == True:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )[:40]
    SPEC_PATH = (
        "/kaggle/input/hms-harmful-brain-activity-classification/train_spectrograms/"
    )
    EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
else:
    test = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    )
    SPEC_PATH = (
        "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
    )
    EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

print(test.shape)

spec_directory_path = "spec_spectrograms/"
if not os.path.exists(spec_directory_path):
    os.makedirs(spec_directory_path)

eeg_directory_path = "eeg_spectrograms/"
if not os.path.exists(eeg_directory_path):
    os.makedirs(eeg_directory_path)

raw_10s_directory_path = "eeg_10s_raws/"
if not os.path.exists(raw_10s_directory_path):
    os.makedirs(raw_10s_directory_path)

raw_50s_directory_path = "eeg_50s_raws/"
if not os.path.exists(raw_50s_directory_path):
    os.makedirs(raw_50s_directory_path)

from joblib import Parallel, delayed

EEG_IDS = test.eeg_id.unique()


def save(row):
    eeg_id = row["eeg_id"]
    spec_id = row["spectrogram_id"]
    spec = pd.read_parquet(f"{SPEC_PATH}{spec_id}.parquet")

    spec_arr = spec.values[:, 1:].T.astype("float32")  # (Hz, Time) = (400, 300)

    split_spec_arr = spec_arr[:, 0:300]
    np.save(f"{spec_directory_path}{eeg_id}", split_spec_arr)

    img_c = raw10seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet", eeg_id)
    np.save(f"{raw_10s_directory_path}{eeg_id}_c", img_c)

    img = raw50seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{raw_50s_directory_path}{eeg_id}", img)

    img = stft_spec_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{eeg_directory_path}{eeg_id}", img)


_ = Parallel(n_jobs=4)(delayed(save)(row) for index, row in test.iterrows())




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3764667734.py in <cell line: 0>()
     18     EEG_PATH = "/kaggle/input/hms-harmful-brain-activity-classification/train_eegs/"
     19 else:
---> 20     test = pd.read_csv(
     21         "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
     22     )

NameError: name 'pd' is not defined

## === cell 6
class Config:
    seed = 2024
    num_folds = 5




## === cell 7
def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything(Config.seed)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3459842842.py in <cell line: 0>()
      7 
      8 
----> 9 seed_everything(Config.seed)
     10 

/tmp/ipykernel_55/3459842842.py in seed_everything(seed)
      1 def seed_everything(seed):
----> 2     torch.backends.cudnn.deterministic = True
      3     torch.backends.cudnn.benchmark = True
      4     torch.manual_seed(seed)
      5     np.random.seed(seed)

NameError: name 'torch' is not defined

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
        df["eeg_id"] = df["eeg_id"]
        self.spec_data_path = spec_directory_path
        self.eeg_data_path = eeg_directory_path
        self.raw_50s_data_path = raw_50s_directory_path
        self.raw_10s_data_path = raw_10s_directory_path
        self.df = df.reset_index(drop=True)
        self.test_imgsize = test_imgsize

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.loc[index]
        eeg_id = str(row.eeg_id)
        spec_image_path = os.path.join(self.spec_data_path, eeg_id + ".npy")
        eeg_image_path = os.path.join(self.eeg_data_path, eeg_id + ".npy")
        raw_50s_image_path = os.path.join(self.raw_50s_data_path, eeg_id + ".npy")
        raw_10s_c_image_path = os.path.join(self.raw_10s_data_path, eeg_id + "_c.npy")

        spec_img = np.load(spec_image_path).astype("float32")
        raw_50s_img = np.load(raw_50s_image_path).astype("float32")
        raw_10s_c_img = np.load(raw_10s_c_image_path).astype("float32")
        eeg_img = np.load(eeg_image_path)

        eeg_img = resize(eeg_img, self.test_imgsize)
        spec_img = resize(spec_img, self.test_imgsize)
        raw_10s_c_img = resize(raw_10s_c_img, self.test_imgsize)
        raw_50s_img = resize(raw_50s_img, self.test_imgsize)

        eeg_img = np.expand_dims(eeg_img, -1)
        spec_img = np.expand_dims(spec_img, -1)
        raw_50s_img = np.expand_dims(raw_50s_img, -1)
        raw_10s_c_img = np.expand_dims(raw_10s_c_img, -1)

        eps = 1e-6
        spec_img = np.clip(spec_img, np.exp(-4), np.exp(8))
        spec_img = np.log(spec_img)
        spec_img = np.nan_to_num(spec_img, nan=0.0)

        img_mean = spec_img.mean(axis=(0, 1))
        img_std = spec_img.std(axis=(0, 1))
        spec_img = (spec_img - img_mean) / (img_std + eps)

        return (
            spec_img,
            eeg_img,
            raw_50s_img,
            raw_10s_c_img,
            raw_10s_c_img,
            raw_10s_c_img,
            eeg_id,
        )




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

        self.device_id = device_id

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
        self.head1 = nn.Linear(384, 6)
        self.head2 = nn.Linear(384, 6)
        self.head3 = nn.Linear(384, 6)
        self.head4 = nn.Linear(384, 6)

    def forward(self, spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_imgs):
        spec_imgs = spec_imgs.transpose(1, 2).transpose(1, 3).contiguous()
        eeg_imgs = eeg_imgs.transpose(1, 2).transpose(1, 3).contiguous()
        raw_50s_imgs = raw_50s_imgs.transpose(1, 2).transpose(1, 3).contiguous()
        raw_10s_imgs = raw_10s_imgs.transpose(1, 2).transpose(1, 3).contiguous()

        spec_feature = self.spec_model.forward_features(spec_imgs)[:, 0]
        eeg_feature = self.eeg_model.forward_features(eeg_imgs)[:, 0]
        raw_50s_feature = self.raw_50s_model.forward_features(raw_50s_imgs)[:, 0]
        raw_10s_feature = self.raw_10s_model.forward_features(raw_10s_imgs)[:, 0]

        feature = torch.cat(
            (spec_feature, eeg_feature, raw_50s_feature, raw_10s_feature), 1
        )
        logits = self.head(feature)
        logits_1 = self.head1(spec_feature)
        logits_2 = self.head2(eeg_feature)
        logits_3 = self.head3(raw_50s_feature)
        logits_4 = self.head4(raw_10s_feature)

        return logits, logits_1, logits_2, logits_3, logits_4




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/286530543.py in <cell line: 0>()
----> 1 class Net(nn.Module):
      2     def __init__(self, back_bone, device_id):
      3         super().__init__()
      4         self.spec_model = timm.create_model(
      5             back_bone, num_classes=6, pretrained=False, in_chans=1

NameError: name 'nn' is not defined

## === cell 14
vit_models = []
model_weights = [
    "/kaggle/input/hms-stage2/fold_0_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_1_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_2_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_3_spec_raw_50_10_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_4_spec_raw_50_10_bestlb.pth",
]
model_types = ["vit_small", "vit_small", "vit_small", "vit_small", "vit_small"]
device = torch.device("cpu")  # use CPU instead of CUDA
for i in range(len(model_types)):
    if model_types[i] == "vit_small":
        model = Net("vit_small_patch14_reg4_dinov2.lvd142m", device).to(device)
        model.load_state_dict(torch.load(model_weights[i], map_location="cpu"))
        model.eval()
        vit_models.append(model)

test_data = ImageFolder(test, (518, 518))
test_loader = DataLoader(
    test_data, batch_size=32, pin_memory=False, num_workers=4, drop_last=False
)
result_6 = {}
with torch.no_grad():
    for batch_idx, (
        spec_imgs,
        eeg_imgs,
        raw_50s_imgs,
        raw_10s_l_imgs,
        raw_10s_c_imgs,
        raw_10s_r_imgs,
        eeg_ids,
    ) in enumerate(test_loader):
        spec_imgs = spec_imgs.to(device).float()
        eeg_imgs = eeg_imgs.to(device).float()
        raw_50s_imgs = raw_50s_imgs.to(device).float()
        raw_10s_c_imgs = raw_10s_c_imgs.to(device).float()
        ensemble_probs = torch.zeros((spec_imgs.shape[0], 6)).to(device)
        for model in vit_models:
            logits_c, _, _, _, _ = model(
                spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs
            )
            probs_c = logits_c.softmax(dim=1)
            ensemble_probs += probs_c
        ensemble_probs /= len(vit_models)
        ensemble_probs = ensemble_probs.detach().cpu().numpy()
        for j in range(len(eeg_ids)):
            eeg_id = eeg_ids[j]
            if eeg_id not in result_6.keys():
                result_6[eeg_id] = np.zeros(6)
            result_6[eeg_id] += ensemble_probs[j]



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3357731805.py in <cell line: 0>()
      8 ]
      9 model_types = ["vit_small", "vit_small", "vit_small", "vit_small", "vit_small"]
---> 10 device = torch.device("cpu")  # use CPU instead of CUDA
     11 for i in range(len(model_types)):
     12     if model_types[i] == "vit_small":

NameError: name 'torch' is not defined

## === cell 15
for model in vit_models:
    del model
torch.cuda.empty_cache()
gc.collect()



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/462183320.py in <cell line: 0>()
      1 for model in vit_models:
      2     del model
----> 3 torch.cuda.empty_cache()
      4 gc.collect()
      5 

NameError: name 'torch' is not defined

## === cell 16
vit_models = []
model_weights = [
    "/kaggle/input/hms-stage2/fold_0_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_1_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_2_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_3_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_4_raw_50_10_bestlb_twostage.pth",
]
model_types = ["vit_small", "vit_small", "vit_small", "vit_small", "vit_small"]
device = torch.device("cpu")
for i in range(len(model_types)):
    if model_types[i] == "vit_small":
        model = Net("vit_small_patch14_reg4_dinov2.lvd142m", device).to(device)
        model.load_state_dict(torch.load(model_weights[i], map_location="cpu"))
        model.eval()
        vit_models.append(model)

test_data = ImageFolder(test, (518, 518))
test_loader = DataLoader(
    test_data, batch_size=32, pin_memory=False, num_workers=4, drop_last=False
)
result_7 = {}
with torch.no_grad():
    for batch_idx, (
        spec_imgs,
        eeg_imgs,
        raw_50s_imgs,
        raw_10s_l_imgs,
        raw_10s_c_imgs,
        raw_10s_r_imgs,
        eeg_ids,
    ) in enumerate(test_loader):
        spec_imgs = spec_imgs.to(device).float()
        eeg_imgs = eeg_imgs.to(device).float()
        raw_50s_imgs = raw_50s_imgs.to(device).float()
        raw_10s_c_imgs = raw_10s_c_imgs.to(device).float()
        ensemble_probs = torch.zeros((spec_imgs.shape[0], 6)).to(device)
        for model in vit_models:
            logits_c, _, _, _, _ = model(
                spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs
            )
            probs_c = logits_c.softmax(dim=1)
            ensemble_probs += probs_c
        ensemble_probs /= len(vit_models)
        ensemble_probs = ensemble_probs.detach().cpu().numpy()
        for j in range(len(eeg_ids)):
            eeg_id = eeg_ids[j]
            if eeg_id not in result_7.keys():
                result_7[eeg_id] = np.zeros(6)
            result_7[eeg_id] += ensemble_probs[j]



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4181615248.py in <cell line: 0>()
      8 ]
      9 model_types = ["vit_small", "vit_small", "vit_small", "vit_small", "vit_small"]
---> 10 device = torch.device("cpu")
     11 for i in range(len(model_types)):
     12     if model_types[i] == "vit_small":

NameError: name 'torch' is not defined

## === cell 17
for model in vit_models:
    del model
torch.cuda.empty_cache()
gc.collect()



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/462183320.py in <cell line: 0>()
      1 for model in vit_models:
      2     del model
----> 3 torch.cuda.empty_cache()
      4 gc.collect()
      5 

NameError: name 'torch' is not defined

## === cell 18
vit_models = []
model_weights = [
    "/kaggle/input/hms-bestlb-vitbase/fold_0_raw_50_10_bestlb_vitbase.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_1_raw_50_10_bestlb_vitbase.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_2_raw_50_10_bestlb_vitbase.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_3_raw_50_10_bestlb_vitbase.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_4_raw_50_10_bestlb_vitbase.pth",
]
model_types = ["vit_base", "vit_base", "vit_base", "vit_base", "vit_base"]
device = torch.device("cpu")
for i in range(len(model_types)):
    if model_types[i] == "vit_base":
        model = Net("vit_base_patch14_reg4_dinov2.lvd142m", device).to(device)
        model.load_state_dict(torch.load(model_weights[i], map_location="cpu"))
        model.eval()
        vit_models.append(model)

test_data = ImageFolder(test, (518, 518))
test_loader = DataLoader(
    test_data, batch_size=32, pin_memory=False, num_workers=4, drop_last=False
)
result_5 = {}
with torch.no_grad():
    for batch_idx, (
        spec_imgs,
        eeg_imgs,
        raw_50s_imgs,
        raw_10s_l_imgs,
        raw_10s_c_imgs,
        raw_10s_r_imgs,
        eeg_ids,
    ) in enumerate(test_loader):
        spec_imgs = spec_imgs.to(device).float()
        eeg_imgs = eeg_imgs.to(device).float()
        raw_50s_imgs = raw_50s_imgs.to(device).float()
        raw_10s_c_imgs = raw_10s_c_imgs.to(device).float()
        ensemble_probs = torch.zeros((spec_imgs.shape[0], 6)).to(device)
        for model in vit_models:
            logits_c, _, _, _, _ = model(
                spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs
            )
            probs_c = logits_c.softmax(dim=1)
            ensemble_probs += probs_c
        ensemble_probs /= len(vit_models)
        ensemble_probs = ensemble_probs.detach().cpu().numpy()
        for j in range(len(eeg_ids)):
            eeg_id = eeg_ids[j]
            if eeg_id not in result_5.keys():
                result_5[eeg_id] = np.zeros(6)
            result_5[eeg_id] += ensemble_probs[j]



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1348077763.py in <cell line: 0>()
      8 ]
      9 model_types = ["vit_base", "vit_base", "vit_base", "vit_base", "vit_base"]
---> 10 device = torch.device("cpu")
     11 for i in range(len(model_types)):
     12     if model_types[i] == "vit_base":

NameError: name 'torch' is not defined

## === cell 19
for model in vit_models:
    del model
torch.cuda.empty_cache()
gc.collect()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/462183320.py in <cell line: 0>()
      1 for model in vit_models:
      2     del model
----> 3 torch.cuda.empty_cache()
      4 gc.collect()
      5 

NameError: name 'torch' is not defined

## === cell 20
vit_models = []
model_weights = [
    "/kaggle/input/hms-stage2-2/fold_0_raw_20_10_bestlb.pth",
    "/kaggle/input/hms-stage2-2/fold_1_raw_20_10_bestlb.pth",
    "/kaggle/input/hms-stage2-2/fold_2_raw_20_10_bestlb.pth",
    "/kaggle/input/hms-stage2-2/fold_3_raw_20_10_bestlb.pth",
    "/kaggle/input/hms-stage2-2/fold_4_raw_20_10_bestlb.pth",
]
model_types = ["vit_small", "vit_small", "vit_small", "vit_small", "vit_small"]
device = torch.device("cpu")
for i in range(len(model_types)):
    if model_types[i] == "vit_small":
        model = Net("vit_small_patch14_reg4_dinov2.lvd142m", device).to(device)
        model.load_state_dict(torch.load(model_weights[i], map_location="cpu"))
        model.eval()
        vit_models.append(model)

test_data = ImageFolder(test, (518, 518))
test_loader = DataLoader(
    test_data, batch_size=32, pin_memory=False, num_workers=4, drop_last=False
)
result_3 = {}
with torch.no_grad():
    for batch_idx, (
        spec_imgs,
        eeg_imgs,
        raw_50s_imgs,
        raw_10s_l_imgs,
        raw_10s_c_imgs,
        raw_10s_r_imgs,
        eeg_ids,
    ) in enumerate(test_loader):
        spec_imgs = spec_imgs.to(device).float()
        eeg_imgs = eeg_imgs.to(device).float()
        raw_50s_imgs = raw_50s_imgs.to(device).float()
        raw_10s_c_imgs = raw_10s_c_imgs.to(device).float()
        ensemble_probs = torch.zeros((spec_imgs.shape[0], 6)).to(device)
        for model in vit_models:
            logits_c = model(spec_imgs, eeg_imgs, raw_50s_imgs, raw_10s_c_imgs)
            probs_c = logits_c.softmax(dim=1)
            ensemble_probs += probs_c
        ensemble_probs /= len(vit_models)
        ensemble_probs = ensemble_probs.detach().cpu().numpy()
        for j in range(len(eeg_ids)):
            eeg_id = eeg_ids[j]
            if eeg_id not in result_3.keys():
                result_3[eeg_id] = np.zeros(6)
            result_3[eeg_id] += ensemble_probs[j]



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2390346018.py in <cell line: 0>()
      8 ]
      9 model_types = ["vit_small", "vit_small", "vit_small", "vit_small", "vit_small"]
---> 10 device = torch.device("cpu")
     11 for i in range(len(model_types)):
     12     if model_types[i] == "vit_small":

NameError: name 'torch' is not defined

## === cell 21
for model in vit_models:
    del model
torch.cuda.empty_cache()
gc.collect()




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4005638849.py in <cell line: 0>()
      1 for model in vit_models:
      2     del model
----> 3 torch.cuda.empty_cache()
      4 gc.collect()
      5 

NameError: name 'torch' is not defined

## === cell 22
def safe_average(dicts, key):
    """Return the average over the provided dicts for a given key,
    handling missing entries gracefully."""
    vals = [d.get(key, np.zeros(6)) for d in dicts]
    return sum(vals) / len(vals)




## === cell 23
sub_rows = []
dicts = [result_6, result_7, result_3, result_5]
for eeg_id in test.eeg_id.unique():
    avg_pred = safe_average(dicts, eeg_id)
    avg_pred = np.clip(avg_pred, 0, None)
    if avg_pred.sum() == 0:
        avg_pred = np.full_like(avg_pred, 1 / 6)
    else:
        avg_pred = avg_pred / avg_pred.sum()
    row = [eeg_id] + avg_pred.tolist()
    sub_rows.append(row)

df = pd.DataFrame(sub_rows, columns=["eeg_id"] + CLASSES)
df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
df.head()

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/317351984.py in <cell line: 0>()
      1 sub_rows = []
----> 2 dicts = [result_6, result_7, result_3, result_5]
      3 for eeg_id in test.eeg_id.unique():
      4     avg_pred = safe_average(dicts, eeg_id)
      5     # Ensure the probabilities sum to 1 (numerical safety)

NameError: name 'result_6' is not defined
