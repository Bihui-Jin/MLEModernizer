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

0.5260450191020286

# 6. Current score

1.40434

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'You’re failing because no model checkpoints are being found, so the `preds` list is empty and `np.stack(preds, ...)` crashes. I (1) make the model search path robust by trying a few common Kaggle input folder patterns and printing what’s available, (2) add a safe fallback that outputs a valid uniform-probability submission if no models are present (so you always get a `.csv`), and (3) fix a test-time metadata issue: the generator expects an `offset` column but test.csv doesn’t have one, so we add `offset=0` for test. These changes are execution- and submission-validity focused and should be score-neutral if models are found; if not, at least you produce a valid submission rather than erroring.'
- What this solution (achieved 1.405) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.5260), so we should make a small change that legitimately improves KL without changing the model or feature pipeline. The biggest likely issue is a distribution mismatch: your test predictions are raw model softmaxes, but the competition target is a vote-distribution; calibrating predictions toward the empirically observed label distribution (from train votes) usually reduces KL on HMS. I add a minimal “prior-mix” calibration step that blends each prediction with the global mean vote distribution from `train.csv`, plus a tiny floor/renorm to keep the submission valid. This keeps the same inference core logic and should move the score toward the target by reducing overconfident miscalibration.'
- What this solution (achieved 1.40289) has done: 'Your current score (1.405, lower-is-better) is far from the target (0.526), so we should make a small, legitimate improvement that mainly reduces KL by improving probability calibration without changing the model or feature pipeline. The biggest low-risk lever is the “prior-mix” step: we make it slightly more adaptive by mixing more strongly only when the model is overconfident (peaky), which tends to reduce KL on this competition while preserving your core inference logic. We also compute the prior from the consolidated label distribution per `eeg_id` (instead of per overlapping row) to better match the test distribution granularity. Finally, we keep the strict probability validity safeguards (clip + renorm) and still write a correct `submission.csv`.'
- What this solution (achieved 1.40653) has done: 'I keep your inference pipeline and model loading intact, but fix two calibration issues that can materially reduce KL with minimal code changes. First, I compute the global prior using the correct unit (“label_id”, which represents consolidated labels) instead of aggregating by `eeg_id`, which can overweight recordings with many overlapping rows and miscalibrate the prior. Second, I replace the current pmax-based adaptive mixing with a more stable entropy-based mixing (still small alpha range), which better targets overconfident predictions (a common KL failure mode) without changing model outputs themselves. I keep the strict clip+renorm safeguards and the same submission schema.'
- What this solution (achieved 1.40554) has done: 'Your current score (1.40653, lower-is-better) is still far from the target (0.526), so we should only make small, low-risk changes that reduce KL mainly via better probability calibration while keeping your model/inference pipeline intact. The biggest likely issue is that the adaptive prior-mix is too weak/unstable and the chosen prior (simple mean over label_id distributions) may not match the true vote distribution; I (1) compute a more appropriate global prior by summing votes over `label_id` (then normalizing) to match the population vote mass, and (2) replace entropy-based mixing with a small, stable temperature scaling + fixed small prior blend (both are post-processing only, no architecture/training changes). I keep strict clipping+renorm so every row sums to 1 and submission stays valid. Everything else (data reading, spectrogram creation, model loading, inference loop, submission schema/path) remains the same.'
- What this solution (achieved 1.40432) has done: 'Your current score (1.40554, lower-is-better) is far above the target (0.5260), so we should make the smallest legitimate post-processing change that tends to reduce KL without changing your model or feature pipeline. The safest lever here is stronger, but still simple, probability calibration: increase softening (temperature) and prior-blending a bit, and additionally apply a very small per-row “uniform mix” to avoid extreme probabilities that are heavily penalized by KL. These are strictly inference-time transformations (no training/architecture changes) and keep rows summing to 1. Everything else (data loading, spectrogram construction, model loading, inference loop, submission format/path) is preserved.'
- What this solution (achieved 1.40413) has done: 'Your current KL (1.40432, lower-is-better) is still far from the target (0.5260), so we should make a small post-processing change that more strongly reduces overconfident predictions (a common source of high KL) without touching your model, spectrogram generation, or inference loop. I keep your existing temperature + prior-mix + uniform-mix structure, but make the prior blend slightly stronger and the temperature slightly softer, and make the uniform mix a touch larger to further avoid near-zero probabilities that get heavily penalized by KL. I also ensure all probability safety steps remain (clip + renorm) and preserve the exact submission schema. Everything else (data paths, generator logic, model loading, inference) stays unchanged.'
- What this solution (achieved 1.40442) has done: 'Your current KL (1.40413, lower-is-better) is still far above the target (0.5260), so we should only make a minimal, low-risk post-processing change that reduces overconfident/peaky probabilities, which are heavily penalized by KL. I keep your model loading, spectrogram construction, and inference loop identical, and only adjust the calibration step to be slightly more conservative. Specifically, I (1) soften a bit more via temperature scaling, (2) increase the global-prior blend modestly, and (3) raise the uniform-mix slightly, while keeping strict clip+renorm so every row sums to 1. This is the smallest change that is plausibly directionally beneficial for KL without altering core logic.'
- What this solution (achieved 1.40434) has done: 'Your current KL (1.40442, lower-is-better) is still far above the target (0.5260), so we should make a small, low-risk improvement that preserves your model/inference pipeline and only adjusts post-processing calibration. The most likely issue is that the current calibration (T=2.60, beta=0.50, gamma=0.09) is oversmoothing toward the global prior/uniform, which can wash out any useful signal in the model predictions and keep KL high. I keep the same exact prediction generation, ensembling, and safety clipping, but make the calibration less aggressive (slightly lower temperature and weaker prior/uniform mixing) so the model’s probabilities remain informative while still avoiding near-zeros that hurt KL. This is a minimal parameter change and should move KL downward toward the target without altering core logic.'
- What this solution (achieved 1.40434) has done: 'Your KL is much worse than target, and repeated prior/temperature tweaks aren’t moving it, which strongly suggests a core inference mismatch rather than calibration. I keep your model/inference core intact, but fix a likely major issue: you are generating an EEG-derived mel spectrogram and then treating it like the Kaggle spectrogram parquet (log-power) by clipping with `exp(-4)..exp(8)` and taking `log`, which severely distorts the EEG image block. I make the EEG spectrogram generation output a **positive power** mel spectrogram (not dB) so the later `log` transform is semantically consistent, which should legitimately reduce KL without changing architecture/training. I also make the preprocessing numerically safe (power floor before log) and keep the same probability calibration and submission formatting.'
- What this solution (achieved 1.40434) has done: 'Your KL is far worse than the target (lower is better), and repeated calibration tweaks haven’t helped, which strongly suggests the core issue is a **feature mismatch**: the Kaggle spectrogram parquets are log-power like, but your EEG-derived mel output is being log-transformed after being clipped to `exp(-4)..exp(8)`, which can badly distort dynamic range. I keep your model loading, generator structure, inference loop, and submission schema identical, and make only a minimal preprocessing alignment: convert both Kaggle and EEG spectrogram blocks into a consistent **log1p(power)** domain with robust percentile-based scaling (instead of per-image min/max), which typically reduces KL by avoiding sample-wise contrast explosions. I also ensure the EEG mel uses the same effective frequency band (0–20 Hz as you already do) and stays strictly positive, and keep your existing temperature/prior/uniform calibration unchanged.'

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

if "offset" not in test.columns:
    test["offset"] = 0

PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
files2 = os.listdir(PATH2)
print(f"There are {len(files2)} test spectrogram parquets")

spectrograms2 = {}
for i, f in enumerate(files2):
    if i % 100 == 0:
        print(i, ", ", end="")
    tmp = pd.read_parquet(f"{PATH2}{f}")
    name = int(f.split(".")[0])
    spectrograms2[name] = tmp.iloc[:, 1:].values

test = test.rename({"spectrogram_id": "spec_id"}, axis=1)

PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
DISPLAY = 0
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print("\nConverting Test EEG to Spectrograms...\n")




## === cell 1
def eeg_from_parquet(parquet_path):
    eeg = pd.read_parquet(parquet_path, columns=FEATS2)
    rows = len(eeg)
    offset = (rows - 10_000) // 2
    eeg = eeg.iloc[offset : offset + 10_000]
    data = np.zeros((10_000, len(FEATS2)), dtype=np.float32)
    for j, col in enumerate(FEATS2):
        x = eeg[col].values.astype("float32")
        m = np.nanmean(x)
        if np.isnan(x).mean() < 1:
            x = np.nan_to_num(x, nan=m)
        else:
            x[:] = 0
        data[:, j] = x
    return data


import librosa
import pywt


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
                y=x.astype(np.float32),
                sr=200,
                hop_length=len(x) // 300,
                n_fft=1024,
                n_mels=100,
                fmin=0,
                fmax=20,
                win_length=128,
                power=2.0,
            ).astype(np.float32)

            width = (mel_spec.shape[1] // 30) * 30
            mel_spec = mel_spec[:, :width]

            mel_spec = np.maximum(mel_spec, 1e-12)
            img[:, :, k] += mel_spec

        img[:, :, k] /= 4.0

        if display:
            plt.subplot(2, 2, k + 1)
            plt.imshow(np.log1p(img[:, :, k]), aspect="auto", origin="lower")

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
    if i % 200 == 0:
        print(i, "/", len(EEG_IDS2))
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
        self.trans = trans
        self.on_epoch_end()

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

    @staticmethod
    def _to_log_and_scale(img_100_300_4: np.ndarray) -> np.ndarray:
        img = np.maximum(img_100_300_4, 1e-12).astype(np.float32)
        img = np.log1p(img)  # stable log transform for power-like spectrograms
        out = np.empty_like(img, dtype=np.float32)
        for c in range(img.shape[-1]):
            ch = img[..., c]
            lo = np.percentile(ch, 1.0)
            hi = np.percentile(ch, 99.0)
            if not np.isfinite(lo):
                lo = float(np.nanmin(ch))
            if not np.isfinite(hi):
                hi = float(np.nanmax(ch))
            if hi - lo < 1e-6:
                out[..., c] = 0.0
            else:
                ch = np.clip(ch, lo, hi)
                out[..., c] = 255.0 * (ch - lo) / (hi - lo)
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out

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
        img = np.stack(imgs, axis=-1).astype(np.float32)

        img = self._to_log_and_scale(img)

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

        img = eeg.astype(np.float32)
        img = self._to_log_and_scale(img)

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
            img = self.eeg_specs[row.eeg_id].astype(np.float32)
        elif self.data_type == "kaggle":
            spec = self.specs[row.spec_id]
            imgs = [
                spec[offset : offset + 300, k * 100 : (k + 1) * 100].T
                for k in [0, 2, 1, 3]
            ]
            img = np.stack(imgs, axis=-1).astype(np.float32)

        img = self._to_log_and_scale(img)

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
val_transform = A.Compose(
    [
        ToTensorV2(p=1.0),
    ]
)




## === cell 4
def safe_torch_load(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location)
    except TypeError:
        return torch.load(path, map_location=map_location, weights_only=False)


def run_inference_loop(model, test_gen_both, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for X, _ in tqdm.tqdm(test_gen_both(), total=len(test_gen_both)):
            if not torch.is_tensor(X):
                X = torch.from_numpy(X)
            X = X.permute(2, 0, 1).unsqueeze(0).contiguous().to(device)
            logits = model(X)
            probs = logits.softmax(dim=1).detach().cpu().numpy()
            pred_list.append(probs)
    pred_arr = np.concatenate(pred_list, axis=0)
    return pred_arr


CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

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

candidate_globs = [
    "/kaggle/input/model101-mixnet-xl-two-lr/model_101/*/*.pt",
    "/kaggle/input/model101-mixnet-xl-two-lr/model_101/**/*.pt",
    "/kaggle/input/model101-mixnet-xl-two-lr/**/*.pt",
    "/kaggle/input/**/model_101/*/*.pt",
    "/kaggle/input/**/model_101/**/*.pt",
    "/kaggle/input/**/*.pt",
]
model_paths = []
for pat in candidate_globs:
    model_paths = sorted(glob.glob(pat, recursive=True))
    if len(model_paths) > 0:
        break

print("Found models:", len(model_paths))
if len(model_paths) == 0:
    print("No .pt model files found under /kaggle/input. Available input datasets:")
    try:
        print(sorted(os.listdir("/kaggle/input"))[:50])
    except Exception as e:
        print("Could not list /kaggle/input:", repr(e))

for model_path in model_paths:
    print("Loading:", model_path)
    model = safe_torch_load(model_path, map_location=device)
    pred = run_inference_loop(model, test_gen_both, device)
    preds.append(pred)

if len(preds) == 0:
    test_pred = np.full((len(test), len(CLASSES)), 1.0 / len(CLASSES), dtype=np.float32)
else:
    test_pred = np.mean(np.stack(preds, axis=0), axis=0)

test_pred = np.nan_to_num(
    test_pred,
    nan=1.0 / len(CLASSES),
    posinf=1.0 / len(CLASSES),
    neginf=1.0 / len(CLASSES),
)
test_pred = np.clip(test_pred, 1e-9, 1.0)
test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

train = pd.read_csv(
    "/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
    usecols=["label_id"] + CLASSES,
)
train_votes_by_label = train.groupby("label_id", as_index=False)[CLASSES].sum()
global_votes = train_votes_by_label[CLASSES].sum(axis=0).to_numpy(dtype=np.float64)
global_prior = global_votes / np.clip(global_votes.sum(), 1e-12, None)

p = test_pred.astype(np.float64)

T = 2.05
p_cal = np.power(np.clip(p, 1e-12, 1.0), 1.0 / T)
p_cal = p_cal / p_cal.sum(axis=1, keepdims=True)

beta = 0.30
p_mix = (1.0 - beta) * p_cal + beta * global_prior[None, :]

u = (1.0 / len(CLASSES)) * np.ones((1, len(CLASSES)), dtype=np.float64)
gamma = 0.04
test_pred = (1.0 - gamma) * p_mix + gamma * u

eps = 1e-6
test_pred = np.clip(test_pred, eps, 1.0)
test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)
test_pred = test_pred.astype(np.float32)

test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df.insert(0, "eeg_id", test["eeg_id"].values)

test_pred_df.to_csv("submission.csv", index=False)
print(test_pred_df.head())
print("Saved submission.csv with shape:", test_pred_df.shape)
print(
    "Row sums (min/mean/max):",
    test_pred_df[CLASSES].sum(axis=1).min(),
    test_pred_df[CLASSES].sum(axis=1).mean(),
    test_pred_df[CLASSES].sum(axis=1).max(),
)
print("Global prior used:", dict(zip(CLASSES, global_prior.tolist())))
print("Calibration params: T=", T, " beta=", beta, " gamma=", gamma)
