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

0.4377007069407573

# 6. Current score

0.8815

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'The run currently fails because it expects external model checkpoints that aren’t present in your environment, so no submission file is produced. To make the pipeline run end-to-end reliably, I keep your data loading and feature construction intact, but add a safe fallback that generates a valid probability distribution when checkpoints are missing (using train label priors). I also fix notebook-only `display()` usage so it runs as a plain Python script, and make the file path selection robust to either `/kaggle/input/...` or `/kaggle/data/...` layouts. Finally, I ensure the submission columns, row count, and per-row probability sums are valid and write `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'I fix the fallback path so it no longer tries to index `all_eegs2`/`spectrograms2` (which only contain *test* IDs) using *train* IDs, which caused the `KeyError`. The minimal fix is to compute the fallback prediction using only information available without loading train EEG/spectrogram parquet files: use the normalized per-class vote prior from `train.csv` (safe, fast, and always available). I also harden the checkpoint loading so it doesn’t crash if `torch.load` returns a state_dict instead of a full model object, and ensure the written `submission.csv` has correct columns and per-row probabilities summing to 1.'
- What this solution (achieved 1.19354) has done: 'Your current run is already using a train-vote prior fallback (because checkpoints are missing), which is too “global” and explains the weak KL score. To move toward the target with minimal semantic change, I keep your pipeline exactly the same but improve the fallback to be patient-aware by using per-patient class priors from `train.csv` when available (and global prior otherwise). This is still a legitimate, leakage-free baseline for test (test has `patient_id`) and typically reduces KL substantially versus a single global prior. I also keep strict row-wise normalization (already present) to ensure Kaggle accepts the submission.'
- What this solution (achieved 0.79325) has done: 'Your current score (1.19354, lower-is-better) is far from the target (0.4377), so we should improve the fallback predictions without changing the model pipeline. I keep your existing “patient-aware prior” fallback, but make it stronger by (1) aggregating train labels at the same `eeg_id` granularity as the submission (so overlaps don’t overweight certain recordings), and (2) using a backoff chain `patient_id -> eeg_id -> global` and smoothing the patient prior toward the global prior to reduce overconfidence (which typically hurts KL). These are minimal, inference-only changes that preserve evaluation semantics and still produce a valid probability distribution summing to 1. Everything else (feature extraction, transforms, inference loop, submission writing) stays the same.'
- What this solution (achieved 0.77444) has done: 'Your current score (0.79325, lower-is-better) is still far from the target (0.4377), so we should modestly improve the *fallback* predictions while keeping your whole pipeline/model logic unchanged. The biggest safe gain is to make the prior less noisy by aggregating labels at the correct `eeg_id` level using summed votes (not mean), and to use a more stable patient prior computed from those eeg-level distributions. Then we add a small “confidence” mixing based on how much patient history exists (more history → rely more on patient prior; less history → rely more on global), which typically reduces KL without changing any modeling/training. Everything else (EEG→spectrogram creation, generator, inference, submission formatting/normalization) stays identical.'
- What this solution (achieved 0.7863) has done: 'We keep your pipeline and fallback approach intact (since checkpoints are missing) but make the fallback prior stronger in a minimal, inference-only way to reduce KL toward the target. Specifically, we compute patient priors as **vote-sum weighted** (not a plain mean of eeg-level priors) and we also add a **recording-local prior** using each patient’s **most recent prior from train** to better match patient-specific tendencies. Finally, we slightly adjust the shrinkage/mixing so rare-patient priors are safely smoothed to global while frequent patients benefit more, and we keep strict per-row normalization to ensure a valid submission.'
- What this solution (achieved 0.82528) has done: 'Your current score (0.7863, lower-is-better) is still far from the target (0.4377), so the smallest change that should legitimately move KL down is to make the fallback probabilities less “generic” by conditioning on information we already have at test-time. I keep your entire pipeline unchanged (same feature building, generator, inference loop), and only strengthen the no-checkpoint fallback by adding a patient-aware **mix with the population base rate and a light “class temperature” smoothing** to reduce overconfident priors (overconfidence is heavily penalized by KL). Concretely, I (1) compute robust patient priors from train at the correct `eeg_id` granularity, (2) mix patient prior with global prior using a count-based weight, and (3) apply a tiny Dirichlet-like smoothing + optional mild temperature (both preserve probabilities and usually reduce KL). Submission formatting and row-wise normalization remain strict to avoid Kaggle submission failure.'
- What this solution (achieved 0.8446) has done: 'We keep your entire feature extraction and inference pipeline unchanged, and only adjust the no-checkpoint fallback (which you’re currently using) to reduce KL toward the target. Specifically, we replace the current patient prior construction with a more stable Dirichlet-multinomial style estimate: compute patient vote-count posteriors from train at `eeg_id` granularity, then back off `patient -> global` using a count-based weight tied to the *amount of voting evidence* (total votes), not just number of EEGs. Finally, we remove the extra temperature smoothing (which can hurt KL when priors are already conservative) and keep only a small symmetric Dirichlet smoothing to avoid overconfident zeros while preserving valid per-row normalization.'
- What this solution (achieved 0.8815) has done: 'Your current score (0.8446, lower-is-better) is still far above the target (0.4377), and because checkpoints are missing you’re effectively submitting a prior-based fallback; the smallest legitimate improvement is to make that fallback better calibrated per test sample while keeping the rest of your pipeline unchanged. I keep your patient/global Dirichlet fallback core, but add a minimal “patient → (if seen) train-eeg_id prior → global” backoff (this uses only test-time columns and train metadata, no EEG/spectrogram loading) and make the patient weight depend on both total votes and number of distinct train EEGs for that patient (more stable than votes alone). I also make the smoothing strictly Dirichlet-like by adding the pseudo-counts directly (instead of mixing with uniform at the end), which usually reduces KL by avoiding overconfident small probabilities while preserving semantics and row-normalization. Everything else (feature extraction, DataGenerator, inference loop, submission writing/format) remains the same.'

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

try:
    from IPython.display import display  # type: ignore
except Exception:

    def display(x):
        print(x)


def resolve_comp_path(rel_path: str) -> str:
    candidates = [
        os.path.join(
            "/kaggle/input/hms-harmful-brain-activity-classification", rel_path
        ),
        os.path.join(
            "/kaggle/data/hms-harmful-brain-activity-classification", rel_path
        ),
        os.path.join("/kaggle/input", rel_path),
        os.path.join("/kaggle/data", rel_path),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return candidates[0]


COMP_ROOT = os.path.dirname(resolve_comp_path("test.csv"))

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

test = pd.read_csv(resolve_comp_path("test.csv"))
print("Test shape", test.shape)
display(test.head())

PATH2 = resolve_comp_path("test_spectrograms/") + (
    "" if resolve_comp_path("test_spectrograms/").endswith("/") else "/"
)
files2 = os.listdir(PATH2)
print(f"There are {len(files2)} test spectrogram parquets")

spectrograms2 = {}
for i, f in enumerate(files2):
    if i % 100 == 0:
        print(i, ", ", end="")
    tmp = pd.read_parquet(f"{PATH2}{f}")
    name = int(f.split(".")[0])
    spectrograms2[name] = tmp.iloc[:, 1:].values
print()

test = test.rename({"spectrogram_id": "spec_id"}, axis=1)
if "offset" not in test.columns:
    test["offset"] = 0

PATH2 = resolve_comp_path("test_eegs/") + (
    "" if resolve_comp_path("test_eegs/").endswith("/") else "/"
)
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
    img = spectrogram_from_eeg(f"{PATH2}{eeg_id}.parquet", i < DISPLAY)
    all_eegs2[eeg_id] = img




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
            offset = 0
        else:
            offset = int(row.offset / 2)

        eeg = self.eeg_specs[row.eeg_id]
        spec = self.specs[row.spec_id]

        imgs = [
            spec[offset : offset + 300, k * 100 : (k + 1) * 100].T for k in [0, 2, 1, 3]
        ]
        img = np.stack(imgs, axis=-1)
        img = np.clip(img, np.exp(-4), np.exp(8))
        img = np.log(img)
        img = np.nan_to_num(img, nan=0.0)

        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)

        X[0 + 56 : 100 + 56, :256, 0] = img[:, 22:-22, 0]  # LL_k
        X[100 + 56 : 200 + 56, :256, 0] = img[:, 22:-22, 2]  # RL_k
        X[0 + 56 : 100 + 56, :256, 1] = img[:, 22:-22, 1]  # LP_k
        X[100 + 56 : 200 + 56, :256, 1] = img[:, 22:-22, 3]  # RP_k
        X[0 + 56 : 100 + 56, :256, 2] = img[:, 22:-22, 2]  # RL_k
        X[100 + 56 : 200 + 56, :256, 2] = img[:, 22:-22, 1]  # LP_k

        X[0 + 56 : 100 + 56, 256:, 0] = img[:, 22:-22, 0]  # LL_k
        X[100 + 56 : 200 + 56, 256:, 0] = img[:, 22:-22, 2]  # RL_k
        X[0 + 56 : 100 + 56, 256:, 1] = img[:, 22:-22, 1]  # LP_k
        X[100 + 56 : 200 + 56, 256:, 1] = img[:, 22:-22, 3]  # RP_K

        img = eeg
        mn = img.flatten().min()
        mx = img.flatten().max()
        ep = 1e-5
        img = 255 * (img - mn) / (mx - mn + ep)
        X[200 + 56 : 300 + 56, :256, 0] = img[:, 22:-22, 0]  # LL_e
        X[300 + 56 : 400 + 56, :256, 0] = img[:, 22:-22, 2]  # RL_e
        X[200 + 56 : 300 + 56, :256, 1] = img[:, 22:-22, 1]  # LP_e
        X[300 + 56 : 400 + 56, :256, 1] = img[:, 22:-22, 3]  # RP_e
        X[200 + 56 : 300 + 56, :256, 2] = img[:, 22:-22, 2]  # RL_e
        X[300 + 56 : 400 + 56, :256, 2] = img[:, 22:-22, 1]  # LP_e

        X[200 + 56 : 300 + 56, 256:, 0] = img[:, 22:-22, 0]  # LL_e
        X[300 + 56 : 400 + 56, 256:, 0] = img[:, 22:-22, 2]  # RL_e
        X[200 + 56 : 300 + 56, 256:, 1] = img[:, 22:-22, 1]  # LP_e
        X[300 + 56 : 400 + 56, 256:, 1] = img[:, 22:-22, 3]  # RP_e

        if self.trans is not None:
            X = self.trans(image=X)["image"]

        if self.mode != "test":
            y[:] = row[TARGETS]

        return X, y

    def generate_specs(self, index):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[index]
        if self.mode == "test":
            offset = 0
        else:
            offset = int(row.offset / 2)

        if self.data_type == "eeg":
            img = self.eeg_specs[row.eeg_id]
        elif self.data_type == "kaggle":
            spec = self.specs[row.spec_id]
            imgs = [
                spec[offset : offset + 300, k * 100 : (k + 1) * 100].T
                for k in [0, 2, 1, 3]
            ]
            img = np.stack(imgs, axis=-1)
            img = np.clip(img, np.exp(-4), np.exp(8))
            img = np.log(img)
            img = np.nan_to_num(img, nan=0.0)

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

        if self.mode != "test":
            y[:] = row[TARGETS]

        return X, y




## === cell 3
val_transform = A.Compose([ToTensorV2(p=1.0)])




## === cell 4
def run_inference_loop(model, test_gen_both, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch_data, _ in tqdm.tqdm(test_gen_both(), total=len(test_gen_both)):
            batch_data = batch_data.unsqueeze(0).to(device)
            logits = model(batch_data)
            probs = logits.softmax(dim=1).detach().cpu().numpy()
            probs = np.asarray(probs)
            if probs.ndim == 1:
                probs = probs[None, :]
            if probs.shape[0] == 1 and probs.ndim == 2:
                pred_list.append(probs)
            else:
                pred_list.append(probs.reshape(-1, probs.shape[-1]))
    pred_arr = np.concatenate(pred_list, axis=0)
    return pred_arr


def _row_normalize_probs(p: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    s = p.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    p = p / s
    return p.astype(np.float32)


def fallback_prior_predict(test_df: pd.DataFrame, targets) -> np.ndarray:
    usecols = ["eeg_id", "patient_id"] + list(targets)
    train = pd.read_csv(resolve_comp_path("train.csv"), usecols=usecols)

    eeg_votes = train.groupby("eeg_id")[targets].sum().astype("float64")
    eeg_total = eeg_votes.sum(axis=1).astype("float64")

    eeg_pid = (
        train[["eeg_id", "patient_id"]]
        .drop_duplicates("eeg_id")
        .set_index("eeg_id")["patient_id"]
        .reindex(eeg_votes.index)
    )

    global_counts = eeg_votes.sum(axis=0).values.astype(np.float64)
    if not np.isfinite(global_counts).all() or global_counts.sum() <= 0:
        global_prior = np.ones(len(targets), dtype=np.float64) / len(targets)
        global_counts = np.ones(len(targets), dtype=np.float64)
    else:
        global_prior = global_counts / global_counts.sum()
    global_prior = np.clip(global_prior, 1e-8, 1.0)
    global_prior = global_prior / global_prior.sum()

    eeg_post = eeg_votes.div(eeg_votes.sum(axis=1), axis=0).astype("float64")
    eeg_post = eeg_post.replace([np.inf, -np.inf], np.nan).fillna(1.0 / len(targets))

    eeg_votes2 = eeg_votes.copy()
    eeg_votes2["patient_id"] = eeg_pid.values
    eeg_votes2["total_votes"] = eeg_total.values
    eeg_votes2 = eeg_votes2.dropna(subset=["patient_id"])

    patient_counts = eeg_votes2.groupby("patient_id")[targets].sum().astype("float64")
    patient_total_votes = (
        eeg_votes2.groupby("patient_id")["total_votes"].sum().astype("float64")
    )
    patient_num_eegs = eeg_votes2.groupby("patient_id").size().astype("float64")

    alpha0 = 3.0  # slightly stronger than before to reduce overconfidence (helps KL)
    alpha = alpha0 * global_prior  # vector, sums to alpha0

    patient_post = patient_counts.add(alpha, axis=1)
    patient_post = patient_post.div(patient_post.sum(axis=1), axis=0).astype("float64")

    tau_votes = 180.0
    tau_eegs = 6.0
    max_w = 0.97

    out = np.zeros((len(test_df), len(targets)), dtype=np.float64)
    for i, (pid, eeg_id) in enumerate(
        zip(test_df["patient_id"].values, test_df["eeg_id"].values)
    ):
        if pid in patient_post.index:
            p_pat = patient_post.loc[pid].values.astype(np.float64)
            tv = float(patient_total_votes.get(pid, 0.0))
            ne = float(patient_num_eegs.get(pid, 0.0))
            wv = tv / (tv + tau_votes) if tv > 0 else 0.0
            wn = ne / (ne + tau_eegs) if ne > 0 else 0.0
            w_pat = float(np.clip(0.5 * (wv + wn), 0.0, max_w))
            prior = w_pat * p_pat + (1.0 - w_pat) * global_prior
        else:
            prior = global_prior

        if eeg_id in eeg_post.index:
            p_eeg = eeg_post.loc[eeg_id].values.astype(np.float64)
            w_eeg = 0.08
            prior = (1.0 - w_eeg) * prior + w_eeg * p_eeg

        prior = np.clip(prior, 1e-8, 1.0)
        prior = prior / prior.sum()
        out[i, :] = prior

    out = _row_normalize_probs(out, eps=1e-8).astype(np.float32)
    return out


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
print("Using device:", device)

model_paths = sorted(
    glob.glob("/kaggle/input/model101-mixnet-xl-label0-02" + "/*/*.pt")
)
if len(model_paths) == 0:
    print(
        "WARNING: No model checkpoints found; using strengthened patient/(eeg_id)/global prior fallback."
    )
    test_pred = fallback_prior_predict(test, TARGETS)
else:
    for model_path in model_paths:
        print("Loading:", model_path)
        obj = torch.load(model_path, map_location=device)
        if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
            print("WARNING: Checkpoint looks like a state_dict; skipping:", model_path)
            continue
        model = obj
        pred = run_inference_loop(model, test_gen_both, device)
        preds.append(pred)
    if len(preds) == 0:
        print(
            "WARNING: No usable models loaded; using strengthened patient/(eeg_id)/global prior fallback."
        )
        test_pred = fallback_prior_predict(test, TARGETS)
    else:
        test_pred = np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)

CLASSES = TARGETS[:]  # ensure exact column order

if test_pred.shape[0] != len(test) or test_pred.shape[1] != len(CLASSES):
    raise ValueError(
        f"Prediction shape {test_pred.shape} does not match expected ({len(test)}, {len(CLASSES)})"
    )

test_pred = _row_normalize_probs(test_pred, eps=1e-8)

test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df = pd.concat(
    [test[["eeg_id"]].reset_index(drop=True), test_pred_df], axis=1
)

out_path = "submission.csv"
test_pred_df.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", test_pred_df.shape)
display(test_pred_df.head())
print(
    "Row sums (min/max):",
    test_pred_df[CLASSES].sum(axis=1).min(),
    test_pred_df[CLASSES].sum(axis=1).max(),
)
