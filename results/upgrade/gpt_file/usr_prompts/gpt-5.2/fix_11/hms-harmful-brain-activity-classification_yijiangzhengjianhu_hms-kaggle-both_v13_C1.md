# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.4134873963078464

# 6. Current score

1.41937

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the pipeline so it always produces a valid `submission.csv` even when the external pretrained model dataset isn’t attached (the current crash is due to missing `/kaggle/input/model101-mixnet-xl-896`). To keep core logic intact, I keep the same DataGenerator and inference loop, but add a safe fallback that outputs a properly normalized uniform probability distribution when no `.pt` files are found. I also make the model search more robust by checking multiple likely directories under `/kaggle/input` before falling back. Finally, I ensure the submission columns and row-wise probability sums match Kaggle’s requirements exactly.'
- What this solution (achieved 1.39779) has done: 'Your current score is much worse than the target (lower is better), and it’s consistent with the fallback path producing uniform probabilities because no compatible `.pt` model is actually being loaded. To move the score toward the target with minimal core-logic changes, I (1) make model discovery load only real HMS model checkpoints (avoid random `.pt` files in `/kaggle/input`), (2) add a safe/legitimate fallback that uses the empirical class prior from `train.csv` instead of uniform (this typically beats uniform on KL), and (3) fix a likely data mismatch in spectrogram extraction by ensuring the test dataframe’s `spec_id` matches the keys loaded into `spectrograms2`. These changes keep your generator/inference loop intact and still always produce a valid, normalized `submission.csv`.'
- What this solution (achieved 1.39779) has done: 'Your current score is far above the target (lower is better), and the most likely reason is that you are still not actually loading any compatible pretrained `.pt` models, so you’re effectively submitting a weak prior for every test row. To move the score closer to the target with minimal semantic changes, I (1) make checkpoint discovery much more precise by searching only inside the HMS competition dataset folder and filtering out non-checkpoint `.pt` files, and (2) add a safe “state_dict only” loader that reconstructs the exact same MixNet-XL-896 architecture only when a `state_dict` is present (so we can use legitimate attached checkpoints without changing the overall approach). If no compatible checkpoints exist, the code still falls back to the train prior and produces a valid, normalized `submission.csv`. These changes keep your generator, feature construction, and inference loop intact; they only unblock using real weights to improve KL.'
- What this solution (achieved 1.39779) has done: 'Your current score is far worse than the target (lower is better), and the code is almost certainly not using any meaningful trained weights (most runs fall back to a constant prior). To move the score closer to the target with minimal semantic changes, I keep your generator and MixNet inference exactly as-is, but (1) fix a key mismatch bug by ensuring spectrogram parquet keys match `test.spec_id` (you currently store `int(filename)` but `spec_id` should be `spectrogram_id`), (2) make the generator robust to missing `spec_id` in `spectrograms2` by falling back to the EEG-derived spectrogram for that sample (so “both” still works), and (3) slightly tighten checkpoint discovery to prefer likely HMS checkpoints and avoid wasting time on irrelevant `.pt` files. These changes are directly aimed at producing non-degenerate inputs to the model and enabling real checkpoint usage when present, which should reduce KL toward your target without changing the core modeling approach. The script still always writes a valid, normalized `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.39779, lower is better) suggests the model path is still falling back to a constant prior, so the biggest minimal gain is to actually find and load valid checkpoints if they exist in your environment. I keep your generator, preprocessing, MixNet-XL build, and inference loop intact, but broaden checkpoint discovery to search common Kaggle locations while filtering to likely HMS checkpoints to avoid random `.pt` files. I also fix a subtle spectrogram slicing issue: in test there is no offset, but the Kaggle spectrogram parquet is 10 minutes long, so we should take the centered 300-frame window (as the competition label is centered) instead of always starting at 0; this is a small semantic alignment that usually reduces KL without changing core logic. Finally, I keep the train-prior fallback, but compute it as a vote-weighted prior (by total annotator votes) which is a more faithful marginal for KL and typically improves over a plain row-mean prior.'
- What this solution (achieved 1.41937) has done: 'Your score is far above the target (lower is better), and the most likely cause is that you’re still not actually using any trained checkpoints, so the submission collapses to a constant prior. To move the KL down toward ~0.41 without changing your model/generator core logic, I (1) fix a bug where `generate_specs()` doesn’t apply `val_transform` in test mode (so your inputs differ from what the MixNet expects), and (2) correct the “kaggle spectrogram” slice to use frequency bands (columns) rather than time (rows), which currently makes the Kaggle spectrogram path effectively malformed. These are minimal, semantics-preserving fixes that should let the same inference loop produce meaningful, calibrated probabilities when checkpoints are found, while keeping the existing prior fallback and guaranteed valid `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'Your current KL (1.419) is far worse than the target (0.413; lower is better), and the code is likely still producing weak/near-constant predictions because the inference loop is running the model on a single sample at a time while leaving MixNet’s internal BatchNorm layers in eval mode without any chance to stabilize on the true test distribution. To move the score down toward the target with minimal core-logic change, I add a proper batched inference DataLoader (no change to architecture or preprocessing) so predictions are produced in the same way the model expects (BatchNorm/activation behavior is identical, just batched for throughput/stability). I also ensure the input tensor is contiguous and float32, and I keep your existing “train-prior fallback” unchanged for safety if no valid checkpoints are found. Finally, I keep the exact submission schema and strict row-wise normalization.'
- What this solution (achieved 1.41937) has done: 'Your current score is far above the target (lower is better), and the biggest likely issue is that your checkpoint search is too broad, so you either load no real HMS weights (falling back to the prior) or accidentally average in incompatible/random checkpoints, both of which can worsen KL. I make a minimal change to restrict checkpoint discovery to the attached competition input only (and a small set of obvious model folders), and I prefer “best/fold/ckpt” style filenames so we load fewer-but-more-relevant checkpoints. I also fix a subtle but important preprocessing mismatch: MixNet expects normalized inputs, but your `val_transform` currently only resizes—adding ImageNet normalization is a minimal semantic alignment that typically improves calibrated probabilities for pretrained MixNet backbones. Everything else (generator, feature construction, architecture, inference semantics, and the guaranteed valid normalized submission) stays the same.'
- What this solution (achieved 1.41937) has done: 'Your current KL (1.419, lower is better) is far from the target (0.413), and with this pipeline that usually means you’re still not loading a real trained checkpoint and/or your preprocessing is slightly mismatched. I make two minimal, directly score-relevant fixes: (1) correct the Albumentations transform order so `Normalize` is applied before `ToTensorV2` (your current order can silently normalize the wrong dtype/scale), and (2) strengthen checkpoint discovery to prefer true model checkpoints inside the attached `/kaggle/input/model101-mixnet-xl-896` folder (and ignore random `.pt` files elsewhere) so we more reliably run real inference instead of falling back to a constant prior. Core model architecture, generator logic, and inference semantics remain unchanged; the fallback prior path is kept for safety and still produces a valid normalized `submission.csv`. These changes should reduce KL toward the target without altering the overall approach.'

# 9. Code solution

## === cell 0
import glob
import os

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import torch
import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

FEATS2 = ["Fp1", "T3", "C3", "O1", "Fp2", "C4", "T4", "O2"]

DATA_TYPE = "both"

TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

test = pd.read_csv("/kaggle/input/hms-harmful-brain-activity-classification/test.csv")
print("Test shape", test.shape)

test = test.rename({"spectrogram_id": "spec_id"}, axis=1)
test["spec_id"] = test["spec_id"].astype(np.int64)

if "offset" not in test.columns:
    test["offset"] = 0

PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
files2 = os.listdir(PATH2)
print(f"There are {len(files2)} test spectrogram parquets")

spectrograms2 = {}
for i, f in enumerate(files2):
    if i % 200 == 0:
        print(i, ", ", end="")
    tmp = pd.read_parquet(f"{PATH2}{f}")
    spec_id = int(f.split(".")[0])
    spectrograms2[spec_id] = tmp.iloc[:, 1:].values
print()

test_spec_ids = test["spec_id"].unique()
have = sum(int(sid in spectrograms2) for sid in test_spec_ids)
print(f"Spectrogram key coverage in dict: {have}/{len(test_spec_ids)} unique spec_id")

PATH_EEG_TEST = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
DISPLAY = 0
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print("Converting Test EEG to Spectrograms...")
print()



## === cell 1
import librosa
import pywt


def eeg_from_parquet(parquet_path):
    eeg = pd.read_parquet(parquet_path, columns=FEATS2)
    rows = len(eeg)
    offset = (rows - 10_000) // 2
    eeg = eeg.iloc[offset : offset + 10_000]
    data = np.zeros((10_000, len(FEATS2)))
    for j, col in enumerate(FEATS2):
        x = eeg[col].values.astype("float32")
        m = np.nanmean(x)
        if np.isnan(x).mean() < 1:
            x = np.nan_to_num(x, nan=m)
        else:
            x[:] = 0
        data[:, j] = x
    return data


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet="haar", level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1 / 0.6745) * maddest(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:])
    ret = pywt.waverec(coeff, wavelet, mode="per")
    return ret


USE_WAVELET = None

NAMES = ["LL", "LP", "RP", "RR"]

FEATS = [
    ["Fp1", "F7", "T3", "T5", "O1"],
    ["Fp1", "F3", "C3", "P3", "O1"],
    ["Fp2", "F8", "T4", "T6", "O2"],
    ["Fp2", "F4", "C4", "P4", "O2"],
]


def spectrogram_from_eeg(parquet_path, display=False):
    eeg = pd.read_parquet(parquet_path)
    middle = (len(eeg) - 10_000) // 2
    eeg = eeg.iloc[middle : middle + 10_000]

    img = np.zeros((100, 300, 4), dtype="float32")

    if display:
        plt.figure(figsize=(10, 7))

    signals = []
    for k in range(4):
        COLS = FEATS[k]
        for kk in range(4):
            x1 = eeg[COLS[kk]].values
            x2 = eeg[COLS[kk + 1]].values

            m = np.nanmean(x1)
            if np.isnan(x1).mean() < 1:
                x1 = np.nan_to_num(x1, nan=m)
            else:
                x1[:] = 0

            m = np.nanmean(x2)
            if np.isnan(x2).mean() < 1:
                x2 = np.nan_to_num(x2, nan=m)
            else:
                x2[:] = 0

            x = x1 - x2

            if USE_WAVELET:
                x = denoise(x, wavelet=USE_WAVELET)

            signals.append(x)

            mel_spec = librosa.feature.melspectrogram(
                y=x,
                sr=200,
                hop_length=len(x) // 300,
                n_fft=1024,
                n_mels=100,
                fmin=0,
                fmax=20,
                win_length=128,
            )

            width = (mel_spec.shape[1] // 30) * 30
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max).astype(np.float32)[
                :, :width
            ]
            img[:, :, k] += mel_spec_db

        img[:, :, k] /= 4.0

        if display:
            plt.subplot(2, 2, k + 1)
            plt.imshow(img[:, :, k], aspect="auto", origin="lower")

    if display:
        plt.show()
        plt.figure(figsize=(10, 5))
        offset = 0
        for k in range(4):
            if k > 0:
                offset -= signals[3 - k].min()
            plt.plot(range(10_000), signals[k] + offset, label=NAMES[3 - k])
            offset += signals[3 - k].max()
        plt.legend()
        plt.show()

    return img


for i, eeg_id in enumerate(EEG_IDS2):
    img = spectrogram_from_eeg(f"{PATH_EEG_TEST}{eeg_id}.parquet", i < DISPLAY)
    all_eegs2[int(eeg_id)] = img




## === cell 2
class DataGenerator:
    "Generates data for Keras"

    def __init__(
        self,
        data,
        specs=None,
        eeg_specs=None,
        raw_eegs=None,
        augment=False,
        mode="train",
        data_type=DATA_TYPE,
        trans=None,
    ):
        self.data = data
        self.augment = augment
        self.mode = mode
        self.data_type = data_type
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.raw_eegs = raw_eegs
        self.on_epoch_end()
        self.trans = trans

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        X, y = self.data_generation(index)
        if self.augment:
            X = self.augmentation(X)
        return X, y

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def on_epoch_end(self):
        if self.mode == "train":
            self.data = self.data.sample(frac=1).reset_index(drop=True)

    def data_generation(self, index):
        if self.data_type == "both":
            X, y = self.generate_all_specs(index)
        elif self.data_type == "eeg" or self.data_type == "kaggle":
            X, y = self.generate_specs(index)
        elif self.data_type == "raw":
            X, y = self.generate_raw(index)
        return X, y

    def generate_all_specs(self, index):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[index]

        if self.mode == "test":
            offset = None
        else:
            offset = int(row.offset / 2)

        eeg = self.eeg_specs[int(row.eeg_id)]
        if int(row.spec_id) in self.specs:
            spec = self.specs[int(row.spec_id)]
            if offset is None:
                offset0 = max(0, (spec.shape[0] - 300) // 2)
            else:
                offset0 = offset

            imgs = [
                spec[offset0 : offset0 + 300, k * 100 : (k + 1) * 100].T
                for k in [0, 2, 1, 3]
            ]
            img = np.stack(imgs, axis=-1)
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)
            img = np.nan_to_num(img, nan=0.0)
        else:
            img = np.nan_to_num(eeg.astype(np.float32), nan=0.0)

        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)

        X[0 + 56 : 100 + 56, :256, 0] = img[:, 22:-22, 0]
        X[100 + 56 : 200 + 56, :256, 0] = img[:, 22:-22, 2]
        X[0 + 56 : 100 + 56, :256, 1] = img[:, 22:-22, 1]
        X[100 + 56 : 200 + 56, :256, 1] = img[:, 22:-22, 3]
        X[0 + 56 : 100 + 56, :256, 2] = img[:, 22:-22, 2]
        X[100 + 56 : 200 + 56, :256, 2] = img[:, 22:-22, 1]

        X[0 + 56 : 100 + 56, 256:, 0] = img[:, 22:-22, 0]
        X[100 + 56 : 200 + 56, 256:, 0] = img[:, 22:-22, 2]
        X[0 + 56 : 100 + 56, 256:, 1] = img[:, 22:-22, 1]
        X[100 + 56 : 200 + 56, 256:, 1] = img[:, 22:-22, 3]

        img = eeg
        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)

        X[200 + 56 : 300 + 56, :256, 0] = img[:, 22:-22, 0]
        X[300 + 56 : 400 + 56, :256, 0] = img[:, 22:-22, 2]
        X[200 + 56 : 300 + 56, :256, 1] = img[:, 22:-22, 1]
        X[300 + 56 : 400 + 56, :256, 1] = img[:, 22:-22, 3]
        X[200 + 56 : 300 + 56, :256, 2] = img[:, 22:-22, 2]
        X[300 + 56 : 400 + 56, :256, 2] = img[:, 22:-22, 1]

        X[200 + 56 : 300 + 56, 256:, 0] = img[:, 22:-22, 0]
        X[300 + 56 : 400 + 56, 256:, 0] = img[:, 22:-22, 2]
        X[200 + 56 : 300 + 56, 256:, 1] = img[:, 22:-22, 1]
        X[300 + 56 : 400 + 56, 256:, 1] = img[:, 22:-22, 3]

        if self.trans is not None:
            X = self.trans(image=X)["image"]
        else:
            X = torch.from_numpy(X).permute(2, 0, 1).float()

        if self.mode != "test":
            y[:] = row[TARGETS]

        return X, y

    def generate_specs(self, index):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[index]
        if self.mode == "test":
            offset = None
        else:
            offset = int(row.offset / 2)

        if self.data_type == "eeg":
            img = self.eeg_specs[int(row.eeg_id)]
        elif self.data_type == "kaggle":
            if int(row.spec_id) in self.specs:
                spec = self.specs[int(row.spec_id)]
                if offset is None:
                    offset0 = max(0, (spec.shape[0] - 300) // 2)
                else:
                    offset0 = offset

                imgs = [
                    spec[offset0 : offset0 + 300, k * 100 : (k + 1) * 100].T
                    for k in [0, 2, 1, 3]
                ]
                img = np.stack(imgs, axis=-1)
                img = np.clip(img, np.exp(-4), np.exp(8))
                img = np.log(img)
                img = np.nan_to_num(img, nan=0.0)
            else:
                img = self.eeg_specs[int(row.eeg_id)].astype(np.float32)

        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)

        X[0 + 56 : 100 + 56, :256, 0] = img[:, 22:-22, 0]
        X[100 + 56 : 200 + 56, :256, 0] = img[:, 22:-22, 2]
        X[0 + 56 : 100 + 56, :256, 1] = img[:, 22:-22, 1]
        X[100 + 56 : 200 + 56, :256, 1] = img[:, 22:-22, 3]
        X[0 + 56 : 100 + 56, :256, 2] = img[:, 22:-22, 2]
        X[100 + 56 : 200 + 56, :256, 2] = img[:, 22:-22, 1]

        X[0 + 56 : 100 + 56, 256:, 0] = img[:, 22:-22, 0]
        X[100 + 56 : 200 + 56, 256:, 0] = img[:, 22:-22, 1]
        X[0 + 56 : 100 + 56, 256:, 1] = img[:, 22:-22, 2]
        X[100 + 56 : 200 + 56, 256:, 1] = img[:, 22:-22, 3]

        X[200 + 56 : 300 + 56, :256, 0] = img[:, 22:-22, 0]
        X[300 + 56 : 400 + 56, :256, 0] = img[:, 22:-22, 1]
        X[200 + 56 : 300 + 56, :256, 1] = img[:, 22:-22, 2]
        X[300 + 56 : 400 + 56, :256, 1] = img[:, 22:-22, 3]
        X[200 + 56 : 300 + 56, :256, 2] = img[:, 22:-22, 3]
        X[300 + 56 : 400 + 56, :256, 2] = img[:, 22:-22, 2]

        X[200 + 56 : 300 + 56, 256:, 0] = img[:, 22:-22, 0]
        X[300 + 56 : 400 + 56, 256:, 0] = img[:, 22:-22, 2]
        X[200 + 56 : 300 + 56, 256:, 1] = img[:, 22:-22, 1]
        X[300 + 56 : 400 + 56, 256:, 1] = img[:, 22:-22, 3]

        if self.trans is not None:
            X = self.trans(image=X)["image"]
        else:
            X = torch.from_numpy(X).permute(2, 0, 1).float()

        if self.mode != "test":
            y[:] = row[TARGETS]

        return X, y




## === cell 3
val_transform = A.Compose(
    [
        A.Resize(p=1.0, height=896, width=896),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            p=1.0,
        ),
        ToTensorV2(p=1.0),
    ]
)



## === cell 4
from typing import Optional


def _build_mixnet_xl(num_classes: int = 6):
    import torchvision

    if not hasattr(torchvision.models, "mixnet_xl"):
        raise RuntimeError(
            "torchvision.models.mixnet_xl is not available in this environment."
        )
    model = torchvision.models.mixnet_xl(weights=None)
    if hasattr(model, "classifier") and isinstance(
        model.classifier, torch.nn.Sequential
    ):
        last = model.classifier[-1]
        if isinstance(last, torch.nn.Linear):
            in_features = last.in_features
            model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return model


def _try_load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    obj = torch.load(model_path, map_location=device)

    if isinstance(obj, torch.nn.Module):
        return obj

    if isinstance(obj, dict):
        for k in ["model", "net", "module"]:
            if k in obj and isinstance(obj[k], torch.nn.Module):
                return obj[k]

        sd = None
        for k in ["state_dict", "model_state_dict", "net_state_dict"]:
            if k in obj and isinstance(obj[k], dict):
                sd = obj[k]
                break
        if (
            sd is None
            and len(obj) > 0
            and all(isinstance(v, torch.Tensor) for v in obj.values())
        ):
            sd = obj

        if sd is not None:
            new_sd = {}
            for k, v in sd.items():
                nk = k[7:] if k.startswith("module.") else k
                new_sd[nk] = v
            sd = new_sd

            model = _build_mixnet_xl(num_classes=6)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if len(missing) > 200:
                raise RuntimeError(
                    f"Checkpoint {model_path} seems incompatible with mixnet_xl (too many missing keys)."
                )
            return model

    raise RuntimeError(f"Unsupported model format at {model_path}: {type(obj)}")


def get_train_prior(train_csv_path):
    train = pd.read_csv(train_csv_path, usecols=TARGETS)
    votes = train[TARGETS].astype(np.float64).values
    totals = votes.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1.0
    probs = votes / totals
    weights = totals[:, 0]
    prior = (probs * weights[:, None]).sum(axis=0) / np.clip(weights.sum(), 1.0, None)
    prior = np.clip(prior, 1e-12, 1.0)
    prior = prior / prior.sum()
    return prior


class _TorchDataset(torch.utils.data.Dataset):
    def __init__(self, gen: DataGenerator):
        self.gen = gen

    def __len__(self):
        return len(self.gen)

    def __getitem__(self, idx):
        x, _ = self.gen[idx]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = x.contiguous().float()
        return x


def run_inference_loop_batched(model, test_gen_both, device, batch_size: int = 16):
    model.to(device)
    model.eval()

    ds = _TorchDataset(test_gen_both)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    pred_list = []
    with torch.no_grad():
        for xb in tqdm.tqdm(dl, total=len(dl)):
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)
            pred_list.append(probs.detach().cpu().numpy())

    if len(pred_list) == 0:
        raise RuntimeError(
            "No predictions were generated. Check test loader length/data."
        )
    return np.concatenate(pred_list, axis=0)


preds = []

test_gen_both = DataGenerator(
    test,
    mode="test",
    data_type="both",
    specs=spectrograms2,
    eeg_specs=all_eegs2,
    trans=val_transform,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preferred_root = "/kaggle/input/model101-mixnet-xl-896"
candidate_roots = []
if os.path.isdir(preferred_root):
    candidate_roots.append(preferred_root)
candidate_roots.extend(
    [
        "/kaggle/input/hms-harmful-brain-activity-classification",
    ]
)

model_paths = []
for bd in candidate_roots:
    model_paths.extend(glob.glob(f"{bd}/**/*.pt", recursive=True))
    model_paths.extend(glob.glob(f"{bd}/**/*.pth", recursive=True))

model_paths = sorted(set(model_paths))

preferred = []
secondary = []
for p in model_paths:
    pl = p.lower()
    bn = os.path.basename(p).lower()
    if not (bn.endswith(".pt") or bn.endswith(".pth")):
        continue
    if any(x in pl for x in ["optimizer", "sched", "scheduler", "ema", "tokenizer"]):
        continue

    is_preferred = any(x in bn for x in ["best", "fold", "ckpt", "checkpoint", "epoch"])
    looks_related = any(x in pl for x in ["hms", "harmful", "mixnet", "model101"])

    if is_preferred and looks_related:
        preferred.append(p)
    elif looks_related:
        secondary.append(p)

model_paths = preferred[:6] + secondary[:2]

print("Found candidate checkpoint files (filtered):", len(model_paths))
for p in model_paths:
    print("  ", p)

if len(model_paths) > 0:
    for model_path in model_paths:
        try:
            print("Loading:", model_path)
            model = _try_load_model(model_path, device)
            pred = run_inference_loop_batched(
                model, test_gen_both, device, batch_size=16
            )
            preds.append(pred)
        except Exception as e:
            print(f"Skipping {model_path} due to load/inference error: {e}")

    if len(preds) > 0:
        test_pred = np.mean(np.stack(preds, axis=0), axis=0)
    else:
        prior = get_train_prior(
            "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
        )
        n = len(test)
        test_pred = np.tile(prior[None, :], (n, 1)).astype(np.float64)
else:
    prior = get_train_prior(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
    n = len(test)
    test_pred = np.tile(prior[None, :], (n, 1)).astype(np.float64)

eps = 1e-12
test_pred = np.clip(test_pred, eps, 1.0)
test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

CLASSES = TARGETS
test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df = pd.concat(
    [test[["eeg_id"]].reset_index(drop=True), test_pred_df], axis=1
)

row_sum = test_pred_df[CLASSES].sum(axis=1).values
if not np.all(np.isfinite(row_sum)):
    raise RuntimeError("Non-finite values in predictions.")
test_pred_df[CLASSES] = test_pred_df[CLASSES].values / row_sum[:, None]

test_pred_df.to_csv("submission.csv", index=False)
print(test_pred_df.head())
print("Saved submission.csv with shape:", test_pred_df.shape)
print(
    "Row-sum check (min/max):",
    test_pred_df[CLASSES].sum(axis=1).min(),
    test_pred_df[CLASSES].sum(axis=1).max(),
)
