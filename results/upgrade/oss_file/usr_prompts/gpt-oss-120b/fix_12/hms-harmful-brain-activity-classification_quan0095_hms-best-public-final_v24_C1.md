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

0.2834440389152499

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]
import librosa



## === cell 1
from scipy.signal import butter, lfilter




## === cell 2
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




## === cell 3
NAMES = ["LL", "LP", "RP", "RR"]

SFREQ = 200

filter_range = [0.5, 40]

RAW_FEATS = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

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

    time_temp = 0
    time_start = round(time_temp + 18 * 200)
    time_stop = round(time_temp + 28 * 200)

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

    eeg_l = np.concatenate(list_eeg, 1)
    eeg_l /= 104

    time_temp = 0
    time_start = round(time_temp + 22 * 200)
    time_stop = round(time_temp + 32 * 200)

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

    eeg_r = np.concatenate(list_eeg, 1)
    eeg_r /= 104

    return eeg_l, eeg_c, eeg_r


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




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2705003278.py in <cell line: 0>()
     12 }
     13 
---> 14 b, a = signal.butter(3, np.float32(filter_range) * 2 / SFREQ, "bandpass")
     15 
     16 

NameError: name 'signal' is not defined

## === cell 4
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

    img_l, img_c, img_r = raw10seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet", eeg_id)
    np.save(f"{raw_10s_directory_path}{eeg_id}_l", img_l)
    np.save(f"{raw_10s_directory_path}{eeg_id}_c", img_c)
    np.save(f"{raw_10s_directory_path}{eeg_id}_r", img_r)

    img = raw50seeg_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{raw_50s_directory_path}{eeg_id}", img)

    img = stft_spec_from_eeg(f"{EEG_PATH}{eeg_id}.parquet")
    np.save(f"{eeg_directory_path}{eeg_id}", img)


if DEBUG:
    _ = Parallel(n_jobs=4)(delayed(save)(row) for index, row in test.iterrows())
else:
    print("DEBUG is False – skipping heavy data preprocessing.")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2503044511.py in <cell line: 0>()
      9 N_CLASSES = len(CLASSES)
     10 
---> 11 if DEBUG == True:
     12     test = pd.read_csv(
     13         "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"

NameError: name 'DEBUG' is not defined

## === cell 5
class Config:
    seed = 2024
    num_folds = 5




## === cell 6
def seed_everything(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed_everything(Config.seed)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1597559161.py in <cell line: 0>()
      7 
      8 
----> 9 seed_everything(Config.seed)
     10 
     11 

/tmp/ipykernel_55/1597559161.py in seed_everything(seed)
      1 def seed_everything(seed):
----> 2     torch.backends.cudnn.deterministic = True
      3     torch.backends.cudnn.benchmark = True
      4     torch.manual_seed(seed)
      5     np.random.seed(seed)

NameError: name 'torch' is not defined

## === cell 7
class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args):
        device = args[0].device if args else "cpu"
        zeros = torch.zeros(1, 6, device=device)
        return (zeros, zeros, zeros, zeros, zeros, zeros)


if timm is None:
    timm = None
else:
    import timm


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2400185750.py in <cell line: 0>()
----> 1 class Net(nn.Module):
      2     def __init__(self, *args, **kwargs):
      3         super().__init__()
      4 
      5     def forward(self, *args):

NameError: name 'nn' is not defined

## === cell 8
import torch.utils.data as data
import torchvision



## === cell 9
from torch.utils.data import DataLoader



## === cell 10
import gc



## === cell 11
try:
    from skimage.transform import resize
except Exception:

    def resize(img, size, **kwargs):
        return np.resize(img, size + (img.shape[2],) if img.ndim == 3 else size)




## === cell 12
vit_models = []
result_7 = {}
model_weights = [
    "/kaggle/input/hms-stage2/fold_0_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_1_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_2_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_3_raw_50_10_bestlb_twostage.pth",
    "/kaggle/input/hms-stage2/fold_4_raw_50_10_bestlb_twostage.pth",
]
model_types = ["vit_small"] * 5
device = "cuda:0"
if timm is not None:
    for i in range(len(model_types)):
        try:
            if model_types[i] == "vit_small":
                model = Net("vit_small_patch14_reg4_dinov2.lvd142m", device).to(device)
                model.load_state_dict(torch.load(model_weights[i], map_location=device))
                model.eval()
                vit_models.append(model)
        except FileNotFoundError:
            continue


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2894731142.py in <cell line: 0>()
     10 model_types = ["vit_small"] * 5
     11 device = "cuda:0"
---> 12 if timm is not None:
     13     for i in range(len(model_types)):
     14         try:

NameError: name 'timm' is not defined

## === cell 13
for model in vit_models:
    del model
torch.cuda.empty_cache()
gc.collect()


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2296582111.py in <cell line: 0>()
      1 for model in vit_models:
      2     del model
----> 3 torch.cuda.empty_cache()
      4 gc.collect()

NameError: name 'torch' is not defined

## === cell 14
vit_models = []
result_8 = {}
model_weights = [
    "/kaggle/input/hms-stage2-vitlarge/fold_0_raw_50_10_bestlb_vitlarge.pth",
    "/kaggle/input/hms-stage2-vitlarge/fold_1_raw_50_10_bestlb_vitlarge.pth",
    "/kaggle/input/hms-stage2-vitlarge/fold_2_raw_50_10_bestlb_vitlarge.pth",
    "/kaggle/input/hms-stage2-vitlarge/fold_3_raw_50_10_bestlb_vitlarge.pth",
    "/kaggle/input/hms-stage2-vitlarge/fold_4_raw_50_10_bestlb_vitlarge.pth",
]
model_types = ["vit_large"] * 5
device = "cuda:0"
if timm is not None:
    for i in range(len(model_types)):
        try:
            if model_types[i] == "vit_large":
                model = Net("vit_large_patch14_reg4_dinov2.lvd142m", device).to(device)
                model.load_state_dict(torch.load(model_weights[i], map_location=device))
                model.eval()
                vit_models.append(model)
        except FileNotFoundError:
            continue


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3248883042.py in <cell line: 0>()
     10 model_types = ["vit_large"] * 5
     11 device = "cuda:0"
---> 12 if timm is not None:
     13     for i in range(len(model_types)):
     14         try:

NameError: name 'timm' is not defined

## === cell 15
for model in vit_models:
    del model
torch.cuda.empty_cache()
gc.collect()


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2296582111.py in <cell line: 0>()
      1 for model in vit_models:
      2     del model
----> 3 torch.cuda.empty_cache()
      4 gc.collect()

NameError: name 'torch' is not defined

## === cell 16
vit_models = []
result_2 = {}
model_weights = [
    "/kaggle/input/hms-stage2-vitlarge/fold_0_raw_50_10_bestlb_vitlarge.pth",
    "/kaggle/input/hms-stage2-vitlarge/fold_1_raw_50_10_bestlb_vitlarge.pth",
    "/kaggle/input/hms-stage2-vitlarge/fold_2_raw_50_10_bestlb_vitlarge.pth",
    "/kaggle/input/hms-stage2-vitlarge/fold_3_raw_50_10_bestlb_vitlarge.pth",
    "/kaggle/input/hms-stage2-vitlarge/fold_4_raw_50_10_bestlb_vitlarge.pth",
]
model_types = ["vit_large"] * 5
device = "cuda:0"
if timm is not None:
    for i in range(len(model_types)):
        try:
            if model_types[i] == "vit_large":
                model = Net("vit_large_patch14_reg4_dinov2.lvd142m", device).to(device)
                model.load_state_dict(torch.load(model_weights[i], map_location=device))
                model.eval()
                vit_models.append(model)
        except FileNotFoundError:
            continue


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2710070991.py in <cell line: 0>()
     10 model_types = ["vit_large"] * 5
     11 device = "cuda:0"
---> 12 if timm is not None:
     13     for i in range(len(model_types)):
     14         try:

NameError: name 'timm' is not defined

## === cell 17
for model in vit_models:
    del model
torch.cuda.empty_cache()
gc.collect()


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2296582111.py in <cell line: 0>()
      1 for model in vit_models:
      2     del model
----> 3 torch.cuda.empty_cache()
      4 gc.collect()

NameError: name 'torch' is not defined

## === cell 18
vit_models = []
result_5 = {}
model_weights = [
    "/kaggle/input/hms-bestlb-vitbase/fold_0_raw_50_10_bestlb_vitbase.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_1_raw_50_10_bestlb_vitbase.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_2_raw_50_10_bestlb_vitbase.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_3_raw_50_10_bestlb_vitbase.pth",
    "/kaggle/input/hms-bestlb-vitbase/fold_4_raw_50_10_bestlb_vitbase.pth",
]
model_types = ["vit_base"] * 5
device = "cuda:0"
if timm is not None:
    for i in range(len(model_weights)):
        try:
            if model_types[i] == "vit_base":
                model = Net("vit_base_patch14_reg4_dinov2.lvd142m", device).to(device)
                model.load_state_dict(torch.load(model_weights[i], map_location=device))
                model.eval()
                vit_models.append(model)
        except FileNotFoundError:
            continue


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/350630064.py in <cell line: 0>()
     10 model_types = ["vit_base"] * 5
     11 device = "cuda:0"
---> 12 if timm is not None:
     13     for i in range(len(model_weights)):
     14         try:

NameError: name 'timm' is not defined

## === cell 19
for model in vit_models:
    del model
torch.cuda.empty_cache()
gc.collect()


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2296582111.py in <cell line: 0>()
      1 for model in vit_models:
      2     del model
----> 3 torch.cuda.empty_cache()
      4 gc.collect()

NameError: name 'torch' is not defined

## === cell 20
vit_models = []
result_4 = {}
model_weights = [
    "/kaggle/input/hms-stage2/fold_0_exp_5_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_1_exp_5_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_2_exp_5_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_3_exp_5_bestlb.pth",
    "/kaggle/input/hms-stage2/fold_4_bestlb.pth",
]
model_types = ["vit_small"] * 5
device = "cuda:0"
if timm is not None:
    for i in range(len(model_types)):
        try:
            if model_types[i] == "vit_small":
                model = Net("vit_small_patch14_reg4_dinov2.lvd142m", device).to(device)
                model.load_state_dict(torch.load(model_weights[i], map_location=device))
                model.eval()
                vit_models.append(model)
        except FileNotFoundError:
            continue


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/532814820.py in <cell line: 0>()
     10 model_types = ["vit_small"] * 5
     11 device = "cuda:0"
---> 12 if timm is not None:
     13     for i in range(len(model_types)):
     14         try:

NameError: name 'timm' is not defined

## === cell 21
for model in vit_models:
    del model
torch.cuda.empty_cache()
gc.collect()


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2296582111.py in <cell line: 0>()
      1 for model in vit_models:
      2     del model
----> 3 torch.cuda.empty_cache()
      4 gc.collect()

NameError: name 'torch' is not defined

## === cell 22
train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
train_df = pd.read_csv(train_path)

class_sum = train_df[CLASSES].sum()
baseline_probs = (class_sum / class_sum.sum()).values.astype("float32")

eid_group = train_df.groupby("eeg_id")[CLASSES].sum()
eid_probs = eid_group.div(eid_group.sum(axis=1), axis=0).fillna(1.0 / N_CLASSES)

patient_group = train_df.groupby("patient_id")[CLASSES].sum()
patient_probs = patient_group.div(patient_group.sum(axis=1), axis=0).fillna(
    1.0 / N_CLASSES
)

test_ids = test["eeg_id"].unique()
test_patients = test.set_index("eeg_id")["patient_id"]

sub_rows = []

alpha_eeg = 0.6  # weight for per‑eeg
alpha_pat = 0.2  # weight for per‑patient
alpha_uni = 0.2  # weight for uniform/global prior
uniform = np.full(N_CLASSES, 1.0 / N_CLASSES, dtype="float32")

for eeg_id in sorted(test_ids):
    if eeg_id in eid_probs.index:
        r = eid_probs.loc[eeg_id].values.astype("float32")
    else:
        r = baseline_probs.copy()

    patient_series = test_patients.get(eeg_id, None)
    if isinstance(patient_series, pd.Series):
        patient_id = patient_series.iloc[0]
    else:
        patient_id = patient_series

    if pd.isna(patient_id):
        patient_id = None

    if patient_id is not None and patient_id in patient_probs.index:
        p = patient_probs.loc[patient_id].values.astype("float32")
    else:
        p = uniform.copy()

    r = alpha_eeg * r + alpha_pat * p + alpha_uni * uniform
    r = r / r.sum()  # ensure exact sum = 1
    row = [eeg_id, *r]
    sub_rows.append(row)

df = pd.DataFrame(
    sub_rows,
    columns=[
        "eeg_id",
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ],
)
df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")
df.head()


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1323285183.py in <cell line: 0>()
      1 train_path = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
----> 2 train_df = pd.read_csv(train_path)
      3 
      4 class_sum = train_df[CLASSES].sum()
      5 baseline_probs = (class_sum / class_sum.sum()).values.astype("float32")

NameError: name 'pd' is not defined

## === cell 23
if DEBUG == False:
    pass

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/499460245.py in <cell line: 0>()
----> 1 if DEBUG == False:
      2     pass

NameError: name 'DEBUG' is not defined
