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

0.417958599207278

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'Your current script likely doesn’t yield a score because it doesn’t run end-to-end on Kaggle: it uses `## === cell 0` (not allowed by your runner) and `tqdm.tqdm(test_gen_both)` iterates over a generator that is extremely slow here (9850 EEGs) and can time out before writing `submission.csv`. I keep your model/inference logic identical, but switch inference to iterate by index (no per-epoch shuffle needed in test), add a hard guarantee that `submission.csv` is always written (even if models are missing/unloadable), and add a fast-path to avoid the heavy EEG->mel conversion if it’s not needed (i.e., if no model files are found). These are minimal execution/stability fixes that make a valid submission reliably, enabling you to actually obtain a Kaggle score and then iterate toward the target.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.41796), so we should improve predictions rather than degrade them. The biggest issue is that your per-eeg preprocessing is inconsistent: Kaggle spectrograms are used directly, but the EEG-derived spectrogram isn’t clipped/log-scaled the same way, and the “both” path normalizes each tile independently (per-sample min/max), which tends to hurt KL calibration. With minimal changes and without altering the model/loop/architecture, we (1) apply the same clip+log transform to the EEG-derived spectrogram before it enters the generator, and (2) change the final post-processing from hard clipping to a tiny “label-smoothing-like” blend with uniform to reduce overconfident probabilities (usually improves KL). We also rename your first cell from `cell 0` to `cell 1` so it runs in stricter cell runners while keeping the rest intact.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.41796), so we should improve calibration/consistency without changing the model or inference semantics. The biggest likely issue is that your “both” pipeline applies per-sample min/max scaling separately to Kaggle spectrogram and EEG-derived spectrogram tiles, which hurts KL by making probabilities overconfident/inconsistent; we switch that normalization to a fixed, global scaling after the same clip+log (still the same inputs, just a more stable mapping to 0–255). We also increase the uniform-mix smoothing slightly (still valid probabilities) because KL heavily penalizes near-zero probabilities; this typically improves public score when models are miscalibrated. Finally, we rename `cell 0` to `cell 1` and shift numbering so the notebook runs in stricter runners unchanged otherwise.'
- What this solution (achieved 1.40995) has done: 'Your score (1.40995, lower-is-better) is far worse than the target (0.41796), so we should improve calibration/consistency while keeping your exact model inference loop and architecture unchanged. The largest likely issue is a mismatch in preprocessing: Kaggle spectrogram tiles are clip+log transformed, but the EEG-derived spectrogram is currently only scaled, not clip+log transformed in the generator, causing a distribution shift that hurts KL. I apply the same clip+log+nan handling to the EEG spectrogram inside the generator (minimal, preserves semantics), then keep your existing fixed scaling. I also slightly increase the uniform-mix smoothing (alpha) because KL strongly penalizes near-zero probabilities; this is a small post-processing change aimed at reducing the KL gap, not maximizing accuracy. Finally, I renumber `cell 0` to `cell 1` so the notebook runs cleanly in stricter runners.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.41796), so we should improve without changing the model or training approach. The biggest likely score drag here is that `test.csv` contains multiple rows per `eeg_id` (same EEG repeated for different `spectrogram_id`), but your inference runs per-row and always uses offset=0, effectively duplicating/overweighting some EEGs and misaligning what Kaggle evaluates (one row per `eeg_id`). I make the minimal fix to aggregate predictions to one per `eeg_id` (mean over rows), then merge back to `sample_submission.csv` so the submission matches the required 9850 unique EEGs. I keep your preprocessing/model inference unchanged, only adjusting the final assembly step to prevent duplicates and improve KL calibration toward the target.'
- What this solution (achieved 1.40995) has done: 'Your score is much worse than the target (lower-is-better), so we should improve calibration/consistency with minimal risk while keeping your model and inference loop intact. The largest likely KL penalty here is overconfident probabilities (near-zeros) and a small preprocessing mismatch: you already clip+log the EEG-derived spectrogram, but the Kaggle spectrogram path uses hardcoded `-4..8` while the rest of the pipeline uses `CLIP_LO/HI`; we make that consistent to reduce distribution shift. Then we slightly increase the uniform-mix smoothing (still valid probabilities) because KL heavily penalizes assigning extremely small probabilities to classes that have non-zero votes. Finally, we keep your `groupby(eeg_id).mean()` aggregation and submission alignment unchanged to preserve semantics and correctness.'
- What this solution (achieved 1.40995) has done: 'We keep your model/inference core intact and focus on two minimal, high-impact KL fixes: (1) adjust the post-processing smoothing (`alpha`) to reduce extreme probabilities that KL heavily penalizes, and (2) apply smoothing after the `groupby(eeg_id).mean()` aggregation (so the final per-eeg distribution is also protected from near-zeros). We also slightly lower the clipping floor from `1e-12` to `1e-6` (still valid probabilities) to further reduce KL blow-ups when the true class has non-zero votes. No changes to architecture, weights, or how inputs are constructed—just safer calibration and ordering of the existing steps to move the score down toward the target.'
- What this solution (achieved 1.40995) has done: 'We keep your model and inference loop intact and focus on reducing KL by making the final probabilities less overconfident, since near-zero probabilities are heavily penalized by KL and your current score is far above the target (lower is better). Concretely, we (1) apply the uniform-mix smoothing after the per-`eeg_id` aggregation (so the final evaluated distribution is protected), (2) increase the clip floor a bit to avoid tiny probabilities, and (3) slightly increase `alpha` to further reduce extreme predictions. We also renumber the first cell from `cell 0` to `cell 1` so the script runs cleanly in stricter runners, without changing any modeling logic.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.41796), so we should reduce KL by making the final probabilities less overconfident and more consistent with the label distribution, without changing your model or inference loop. The smallest, safest lever for KL here is post-processing: increase the uniform-mix smoothing a bit and raise the probability floor so you avoid near-zero probabilities that KL penalizes heavily. I keep your preprocessing/model ensemble unchanged, but adjust `alpha` and `clip_floor` in a controlled way and apply the same normalization steps as you already do. I also renumber the first cell from `cell 0` to `cell 1` so the script runs cleanly in stricter runners (no change to semantics).'
- What this solution (achieved 1.40995) has done: 'Your score is much worse than the target (lower-is-better), so the safest way to move toward 0.418 without changing model architecture or inference is to reduce KL blow-ups caused by overconfident near-zero probabilities. I keep your model ensemble and generator logic intact, but (1) apply the uniform-mix smoothing earlier (per-row) as well as after `groupby(eeg_id)` so averaging doesn’t reintroduce near-zeros, and (2) slightly raise the probability floor used for clipping to better match KL’s sensitivity to tiny probabilities. I also renumber the first cell from `cell 0` to `cell 1` so the script runs cleanly in stricter runners, with no semantic changes otherwise. The output remains a valid `submission.csv` with the required columns and row sums equal to 1.'

# 9. Code solution

## === cell 0
import glob
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import tqdm

try:
    import pywt  # noqa: F401
except Exception:
    pywt = None

np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)




## === cell 1
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
print(test.head())

PATH_SPEC = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
PATH_EEG = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"

test = test.rename({"spectrogram_id": "spec_id"}, axis=1)
if "offset" not in test.columns:
    test["offset"] = 0




## === cell 2
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


import librosa


def maddest(d, axis=None):
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def denoise(x, wavelet="haar", level=1):
    if pywt is None:
        return x
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

    img = np.clip(img, np.exp(-4), np.exp(8))
    img = np.log(img)
    img = np.nan_to_num(img, nan=0.0)

    return img




## === cell 3
CLIP_LO = -4.0
CLIP_HI = 8.0
SCALE_EP = 1e-5


def scale_0_255_fixed(img):
    img = img.astype(np.float32, copy=False)
    img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
    img = np.clip(img, CLIP_LO, CLIP_HI)
    img = 255.0 * (img - CLIP_LO) / (CLIP_HI - CLIP_LO + SCALE_EP)
    return img.astype(np.float32, copy=False)


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
    ):
        self.data = data
        self.augment = augment
        self.mode = mode
        self.data_type = data_type
        self.specs = specs
        self.eeg_specs = eeg_specs
        self.raw_eegs = raw_eegs
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
        ]  # to match kaggle with eeg
        img = np.stack(imgs, axis=-1)

        img = np.clip(img, np.exp(CLIP_LO), np.exp(CLIP_HI))
        img = np.log(img)
        img = np.nan_to_num(img, nan=0.0)

        img = scale_0_255_fixed(img)

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
        img = np.clip(img, np.exp(CLIP_LO), np.exp(CLIP_HI))
        img = np.log(img)
        img = np.nan_to_num(img, nan=0.0)

        img = scale_0_255_fixed(img)

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
            img = np.clip(img, np.exp(CLIP_LO), np.exp(CLIP_HI))
            img = np.log(img)
            img = np.nan_to_num(img, nan=0.0)
        elif self.data_type == "kaggle":
            spec = self.specs[row.spec_id]
            imgs = [
                spec[offset : offset + 300, k * 100 : (k + 1) * 100].T
                for k in [0, 2, 1, 3]
            ]  # to match kaggle with eeg
            img = np.stack(imgs, axis=-1)

            img = np.clip(img, np.exp(CLIP_LO), np.exp(CLIP_HI))
            img = np.log(img)
            img = np.nan_to_num(img, nan=0.0)

        img = scale_0_255_fixed(img)

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




## === cell 4
def run_inference_loop(model, test_gen_both, device):
    model.to(device)
    model.eval()
    preds = np.zeros((len(test_gen_both), 6), dtype=np.float32)
    with torch.no_grad():
        for i in tqdm.tqdm(range(len(test_gen_both)), total=len(test_gen_both)):
            batch_data, _y = test_gen_both[i]
            batch_data_t = (
                torch.from_numpy(batch_data).permute(2, 0, 1).unsqueeze(0).to(device)
            )
            out = model(batch_data_t)
            prob = out.softmax(dim=1).detach().cpu().numpy().astype(np.float32)  # (1,6)
            preds[i] = prob[0]
    return preds


def safe_load_model(model_path, device):
    obj = torch.load(model_path, map_location=device)
    if isinstance(obj, torch.nn.Module):
        return obj
    if isinstance(obj, dict) and any(
        k.endswith("weight") or k.endswith("bias") for k in obj.keys()
    ):
        return None
    return None


CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model_glob = r"/kaggle/input/model-90-both/model_90/*/*.pt"
model_paths = sorted(glob.glob(model_glob))
print("Found model files:", len(model_paths))

if len(model_paths) == 0:
    sample_sub = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )
    sub = sample_sub.copy()
    sub[CLASSES] = 1.0 / 6.0
    sub.to_csv("submission.csv", index=False)
    print("No models found; wrote uniform submission.csv with shape:", sub.shape)
else:
    print(f"Loading {len(os.listdir(PATH_SPEC))} test spectrogram parquets...")
    files2 = os.listdir(PATH_SPEC)
    spectrograms2 = {}
    for i, f in enumerate(files2):
        if i % 200 == 0:
            print(i, ", ", end="")
        tmp = pd.read_parquet(f"{PATH_SPEC}{f}")
        name = int(f.split(".")[0])
        spectrograms2[name] = tmp.iloc[:, 1:].values
    print()

    DISPLAY = 0
    EEG_IDS2 = test.eeg_id.unique()
    all_eegs2 = {}
    print("Converting Test EEG to Spectrograms...")
    for i, eeg_id in enumerate(EEG_IDS2):
        if i % 200 == 0:
            print(i, ", ", end="")
        img = spectrogram_from_eeg(f"{PATH_EEG}{eeg_id}.parquet", i < DISPLAY)
        all_eegs2[eeg_id] = img
    print()

    test_gen_both = DataGenerator(
        test, mode="test", data_type="both", specs=spectrograms2, eeg_specs=all_eegs2
    )

    preds = []
    for model_path in model_paths:
        print("Loading:", model_path)
        try:
            model = safe_load_model(model_path, device)
            if model is None:
                raise TypeError(
                    "Unsupported checkpoint format (not a full torch.nn.Module)."
                )

            pred = run_inference_loop(model, test_gen_both, device)
            if pred.shape != (len(test), 6):
                raise ValueError(
                    f"Bad prediction shape {pred.shape}, expected {(len(test), 6)}"
                )
            preds.append(pred)
        except Exception as e:
            print("WARNING: model failed, skipping:", model_path, "error:", repr(e))

    if len(preds) == 0:
        print("WARNING: No valid model predictions; using uniform probabilities.")
        test_pred = np.full((len(test), 6), 1.0 / 6.0, dtype=np.float32)
    else:
        test_pred = np.mean(np.stack(preds, axis=0), axis=0).astype(np.float32)

    test_pred = np.nan_to_num(
        test_pred, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0
    ).astype(np.float32)

    alpha = 0.75
    clip_floor = 5e-3

    test_pred = np.clip(test_pred, clip_floor, 1.0)
    test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)
    test_pred = (1.0 - alpha) * test_pred + alpha * (1.0 / 6.0)
    test_pred = np.clip(test_pred, clip_floor, 1.0)
    test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

    test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
    test_pred_df.insert(0, "eeg_id", test["eeg_id"].values)
    test_pred_df = test_pred_df.groupby("eeg_id", as_index=False)[CLASSES].mean()

    vals = test_pred_df[CLASSES].to_numpy(dtype=np.float32)
    vals = np.nan_to_num(vals, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0)
    vals = np.clip(vals, clip_floor, 1.0)
    vals = vals / vals.sum(axis=1, keepdims=True)
    vals = (1.0 - alpha) * vals + alpha * (1.0 / 6.0)
    vals = np.clip(vals, clip_floor, 1.0)
    vals = vals / vals.sum(axis=1, keepdims=True)
    test_pred_df[CLASSES] = vals

    sample_sub = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv"
    )

    test_pred_df = sample_sub[["eeg_id"]].merge(test_pred_df, on="eeg_id", how="left")

    missing = test_pred_df[CLASSES].isna().any(axis=1)
    if missing.any():
        test_pred_df.loc[missing, CLASSES] = 1.0 / 6.0

    vals = test_pred_df[CLASSES].to_numpy(dtype=np.float32)
    vals = np.nan_to_num(vals, nan=1.0 / 6.0, posinf=1.0 / 6.0, neginf=1.0 / 6.0)
    vals = np.clip(vals, clip_floor, 1.0)
    vals = vals / vals.sum(axis=1, keepdims=True)
    test_pred_df[CLASSES] = vals

    row_sums = test_pred_df[CLASSES].sum(axis=1).to_numpy()
    if not np.all(np.isfinite(row_sums)) or np.max(np.abs(row_sums - 1.0)) > 1e-3:
        raise ValueError(
            "Invalid submission probabilities: non-finite or rows not summing to 1."
        )

    if test_pred_df.shape[0] != sample_sub.shape[0]:
        raise ValueError(
            f"Submission row count mismatch: got {test_pred_df.shape[0]}, expected {sample_sub.shape[0]}"
        )

    test_pred_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", test_pred_df.shape)
    print(test_pred_df.head())
