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

0.4152680441202221

# 6. Current score

1.39766

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'The only runtime blocker is that the notebook expects an external dataset of pretrained `.pt` checkpoints that is not present in your environment, so inference cannot start. I keep the spectrogram/EEG feature extraction and the `DataGenerator` unchanged, but add a safe fallback that produces a valid submission when no checkpoints are found. To nudge the KL score toward a reasonable baseline (and avoid submission failure), the fallback use the empirical class prior from `train.csv` (vote proportions averaged across rows) and output that distribution for every test `eeg_id`, with proper clipping and row-wise normalization.'
- What this solution (achieved 1.39779) has done: 'Your current score is much worse than the target (lower-is-better), and the main reason is that you’re not actually using the intended pretrained checkpoints; the class-prior fallback is producing a weak baseline. I keep your feature extraction + `DataGenerator` + inference semantics unchanged, but make the checkpoint discovery robust by searching common Kaggle input locations (not just one hardcoded dataset path) and loading models safely even if they were saved as `state_dict`s. This is the smallest change that can move the score substantially toward the target by enabling real model inference. If no checkpoints are found anywhere, it still fall back to the same class-prior submission and remain valid.'
- What this solution (achieved 1.39779) has done: 'Your current score (1.39779, lower-is-better) is far worse than the target (0.4153), and the main cause is that you’re effectively submitting a weak class-prior because no usable checkpoints are being loaded. I keep your feature extraction, `DataGenerator`, and inference loop intact, but make checkpoint loading actually work for the most common `.pt` formats by rebuilding the exact model architecture in-code and loading `state_dict`s into it. This is a minimal, directly score-relevant change: it enables real model inference when only weights are available, while preserving the existing fallback to class-prior if no compatible checkpoints exist. I also ensure the `test` dataframe has the `spec_id` column expected by the generator (currently only `spec_id` is created, but `DataGenerator` reads `row.spec_id`, so we must guarantee it exists and is correct).'
- What this solution (achieved 1.39779) has done: 'Your current score is far worse than the target (lower-is-better), and the most likely cause is that you’re still not actually using a compatible set of checkpoints—so you either fall back to class-prior or load mismatched weights into the wrong architecture. I keep your feature extraction + `DataGenerator` + inference loop semantics intact, but make checkpoint loading robust for common EfficientNet-B0 export variants by (1) detecting whether the head is stored as `classifier.1.*` or `classifier.*`, and (2) allowing a safe non-strict load that still preserves correct inference when only the head naming differs. I also fix one runtime issue: your `DataGenerator` expects an `offset` column for non-test mode but it’s absent; we explicitly set `test["offset"]=0` to keep behavior consistent and avoid accidental KeyErrors if mode toggles. Finally, I ensure only likely competition checkpoints are used (filtering out irrelevant `.pt` files) to avoid averaging garbage predictions that can hurt KL.'
- What this solution (achieved 1.39779) has done: 'Your current score (1.39779, lower-is-better) is far from the target (0.4153), and the biggest controllable issue in your script is that inference is effectively running with an untrained/random EfficientNet head whenever checkpoints are missing or incompatible—yielding near-uniform probabilities and a poor KL. I keep your feature extraction, `DataGenerator`, and inference loop intact, but (1) tighten checkpoint selection to only those that actually match EfficientNet-B0+6-class head (reduces averaging “garbage” models), and (2) add a KL-safe blend of model predictions with the empirical class prior (a tiny calibration step that usually improves KL when models are miscalibrated), used only when real checkpoints are loaded. If no valid checkpoints load, it still produce a valid submission using the class prior (as before). These are minimal, score-relevant changes that should move the score down toward the target without changing your core modeling logic.'
- What this solution (achieved 1.39779) has done: 'Your current score is far worse than the target (lower-is-better), and the main likely cause is that you’re still not using any meaningful trained model predictions (no checkpoints found/loaded, or averaging incompatible checkpoints), so KL stays very high. I keep your entire feature extraction, `DataGenerator`, EfficientNet-B0 architecture, and inference loop intact, but make two minimal, score-relevant fixes: (1) ensure spectrogram arrays are oriented the way your generator expects (so inference isn’t effectively on scrambled inputs), and (2) add a small, KL-safe per-row “temperature” sharpening after ensembling (plus a tiny prior blend) to improve probability calibration. These changes preserve semantics (still EfficientNet softmax outputs, still normalized probabilities) while typically moving KL down substantially when the pipeline is miscalibrated or inputs are mis-shaped. The script still falls back to the empirical class prior if no compatible checkpoints are found and always writes a valid `submission.csv`.'
- What this solution (achieved 1.39779) has done: 'Your current score is far worse than the target (lower-is-better), so the smallest reliable way to move KL down is to make the inference more metric-aligned without changing your model or feature pipeline. I keep your EfficientNet-B0 + generator + averaging logic intact, but add one safe post-processing step that is especially effective for KL: per-row Dirichlet-style smoothing (mix each prediction with a tiny uniform mass) to avoid overconfident near-zeros that get heavily penalized. I also cap the number of checkpoint files we try (keeping the first N after filtering) to avoid averaging many weak/irrelevant models that can worsen KL and blow runtime, while still using real checkpoints when present. Finally, the submission writing and row-wise normalization remain unchanged and guaranteed valid.'
- What this solution (achieved 1.39766) has done: 'Your current score is far worse than the target, so the smallest score-relevant change is to reduce the KL penalty from overconfident near-zero probabilities while keeping your model/generator/inference intact. I keep your checkpoint loading and ensembling logic unchanged, but add a single, metric-aligned post-processing step: mix each prediction with a small uniform distribution (stronger than your current smoothing) and use a slightly softer temperature (closer to 1.0) to improve calibration. This does not change the architecture, features, or inference loop—only the final probabilities—and it remains guaranteed to sum to 1 per row. If no checkpoints load, the same smoothing is also applied to the class-prior fallback to avoid extreme probabilities.'
- What this solution (achieved 1.39766) has done: 'Your current score is far worse than the target (lower-is-better), so the smallest score-relevant change is to fix a likely silent input mismatch: your test spectrogram parquet arrays are loaded with the wrong orientation for how `DataGenerator` slices them, which can make inference effectively run on scrambled inputs and yield near-random probabilities. I keep your model, generator, and inference loop unchanged, but transpose the loaded test spectrogram arrays to match the expected `(time, freq*4)` layout used by `spec[offset:offset+300, k*100:(k+1)*100].T`. I also add a defensive fallback in `DataGenerator` to use `row.spectrogram_id` if `spec_id` is missing (doesn’t change behavior in your current run), ensuring robustness without altering semantics. These changes should move KL down substantially toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 1.39766) has done: 'Your score is far worse than the target (lower-is-better), and in this script the main likely reason is that inference is not actually using the Kaggle-provided spectrograms in the intended orientation, causing the model to see effectively “scrambled” inputs and output near-random probabilities. I make one minimal, directly score-relevant change: always transpose the loaded test spectrogram arrays to the `(time, 400)` layout expected by `DataGenerator`’s `spec[offset:offset+300, k*100:(k+1)*100].T` slicing. I keep your model, generator, checkpoint loading, and post-processing intact, and still guarantee probabilities are clipped and normalized to sum to 1 and that `submission.csv` is produced. This should move KL down materially toward the target without changing the core approach.'
- What this solution (achieved 1.39766) has done: 'Your current KL (1.39766, lower-is-better) is far above the target (0.4153), and the biggest likely issue remaining (given no external checkpoints) is that the test-time Kaggle spectrogram parquet orientation check is too narrow, so many inputs may still be “scrambled” relative to what `DataGenerator` expects. I make one minimal, score-relevant fix: enforce a consistent `(time, 400)` layout for every loaded Kaggle spectrogram by detecting the 400-frequency dimension in either axis and transposing as needed. This preserves your model/generator/inference core logic and only corrects the input feeding, which should move KL down toward the target if checkpoints are actually being used (and is harmless/neutral if you’re still falling back to the class prior). Everything else (checkpoint logic, smoothing, submission formatting) is kept the same.'
- What this solution (achieved 1.39766) has done: 'Your score is far above the target (lower-is-better), and the biggest likely reason in this code path is that you’re still effectively submitting a weak class-prior because no compatible checkpoints are found/loaded. I keep your feature extraction, `DataGenerator`, EfficientNet-B0 model, and inference loop intact, but make checkpoint discovery/load compatibility broader and more robust so real inference actually runs when checkpoints exist (including common Lightning and “ema/teacher” key formats). To avoid averaging in “garbage” checkpoints that can worsen KL, I (minimally) rank candidate checkpoints by loadability and only ensemble those that pass a strict shape/key sanity check. Finally, I keep your current KL-safe probability post-processing, but reduce the prior-blend slightly when real models load (so we don’t drown model signal), while leaving the fallback behavior unchanged.'

# 9. Code solution

## === cell 0
import glob

import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt
import torch
import tqdm

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
test.head()

PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/"
files2 = os.listdir(PATH2)
print(f"There are {len(files2)} test spectrogram parquets")

spectrograms2 = {}

for i, f in enumerate(files2):
    if i % 100 == 0:
        print(i, ", ", end="")
    tmp = pd.read_parquet(f"{PATH2}{f}")
    name = int(f.split(".")[0])

    arr = tmp.iloc[:, 1:].astype("float32").values

    if arr.ndim != 2:
        raise ValueError(f"Unexpected spectrogram array ndim={arr.ndim} for file {f}")

    if arr.shape[1] == 400:
        pass  # already (time, 400)
    elif arr.shape[0] == 400:
        arr = arr.T  # (time, 400)
    else:
        pass

    spectrograms2[name] = arr

test = test.rename({"spectrogram_id": "spec_id"}, axis=1)
if "offset" not in test.columns:
    test["offset"] = 0

PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
DISPLAY = 0
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print("Converting Test EEG to Spectrograms...")
print()




## === cell 1
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
import pywt, librosa


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

        spec_id = getattr(row, "spec_id", None)
        if spec_id is None and hasattr(row, "spectrogram_id"):
            spec_id = row.spectrogram_id

        eeg = self.eeg_specs[row.eeg_id]
        spec = self.specs[int(spec_id)]

        imgs = [
            spec[offset : offset + 300, k * 100 : (k + 1) * 100].T for k in [0, 2, 1, 3]
        ]  # to match kaggle with eeg
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
            spec_id = getattr(row, "spec_id", None)
            if spec_id is None and hasattr(row, "spectrogram_id"):
                spec_id = row.spectrogram_id
            spec = self.specs[int(spec_id)]
            imgs = [
                spec[offset : offset + 300, k * 100 : (k + 1) * 100].T
                for k in [0, 2, 1, 3]
            ]  # to match kaggle with eeg
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
def run_inference_loop(model, test_gen_both, device):
    model.to(device)
    model.eval()
    pred_list = []
    with torch.no_grad():
        for batch_data, y in tqdm.tqdm(test_gen_both):
            batch_data = torch.from_numpy(batch_data)
            batch_data = batch_data.permute(2, 0, 1)
            batch_data = batch_data.unsqueeze(0)
            batch_data = batch_data.to(device)
            y = model(batch_data)
            pred_list.append(y.softmax(dim=1).detach().cpu().numpy())

    pred_arr = np.concatenate(pred_list, axis=0)
    del pred_list
    return pred_arr


preds = []

test_gen_both = DataGenerator(
    test, mode="test", data_type="both", specs=spectrograms2, eeg_specs=all_eegs2
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

candidate_globs = [
    "/kaggle/input/model100-two-stage-0-8kl-0-2mutilabel/model_100/*/*.pt",
    "/kaggle/input/**/model_100/*/*.pt",
    "/kaggle/input/**/*.pt",
]
model_paths = []
for g in candidate_globs:
    model_paths.extend(glob.glob(g, recursive=True))
model_paths = sorted(set(model_paths))


def _looks_like_checkpoint(p: str) -> bool:
    lp = p.lower()
    good = ["hms", "harmful", "model_100", "efficientnet", "effnet", "eeg", "spec"]
    if any(g in lp for g in good):
        return True
    return False


model_paths = [p for p in model_paths if _looks_like_checkpoint(p)]
model_paths = sorted(set(model_paths))

MAX_CKPTS = 12
if len(model_paths) > MAX_CKPTS:
    model_paths = model_paths[:MAX_CKPTS]


def _build_compatible_model():
    import torchvision

    m = torchvision.models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier[1] = torch.nn.Linear(in_features, 6)
    return m


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in [
            "state_dict",
            "model_state_dict",
            "net",
            "model",
            "ema",
            "ema_state_dict",
            "teacher",
            "student",
        ]:
            if key in obj:
                v = obj[key]
                if isinstance(v, dict):
                    if "state_dict" in v and isinstance(v["state_dict"], dict):
                        return v["state_dict"]
                    return v
        if all(isinstance(k, str) for k in obj.keys()) and any(
            k.endswith("weight") or k.endswith("bias") for k in obj.keys()
        ):
            return obj
    return None


def _clean_state_dict_keys(sd):
    cleaned = {}
    for k, v in sd.items():
        nk = k
        for pref in ("module.", "model.", "net."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        for pref in ("backbone.", "encoder.", "module.model.", "module.backbone."):
            if nk.startswith(pref):
                nk = nk[len(pref) :]
        cleaned[nk] = v
    return cleaned


def _remap_head_keys(sd):
    if (
        "classifier.weight" in sd
        and "classifier.bias" in sd
        and "classifier.1.weight" not in sd
    ):
        sd = dict(sd)
        sd["classifier.1.weight"] = sd.pop("classifier.weight")
        sd["classifier.1.bias"] = sd.pop("classifier.bias")
    return sd


def _is_plausible_efficientnet_b0_sd(sd: dict) -> bool:
    if not isinstance(sd, dict) or len(sd) == 0:
        return False
    keys = list(sd.keys())
    required_any = [
        "features.0.0.weight",
        "features.1.0.block.0.0.weight",
        "features.2",
    ]
    if not any(any(r in k for r in required_any) for k in keys):
        return False
    head_w = sd.get("classifier.1.weight", sd.get("classifier.weight", None))
    if head_w is None or not hasattr(head_w, "shape"):
        return False
    if tuple(head_w.shape)[0] != 6:
        return False
    return True


def _try_load_torch_model(path, device):
    try:
        obj = torch.load(path, map_location=device)
    except Exception:
        return None

    if isinstance(obj, torch.nn.Module):
        try:
            out = obj.classifier[1].out_features
            if out != 6:
                return None
        except Exception:
            pass
        return obj

    sd = _extract_state_dict(obj)
    if sd is None:
        return None

    sd = _clean_state_dict_keys(sd)
    sd = _remap_head_keys(sd)

    if not _is_plausible_efficientnet_b0_sd(sd):
        return None

    model = _build_compatible_model().to(device)
    try:
        model.load_state_dict(sd, strict=True)
        return model
    except Exception:
        try:
            missing, unexpected = model.load_state_dict(sd, strict=False)
            if len(missing) > 10 or len(unexpected) > 10:
                return None
            return model
        except Exception:
            return None


def _class_prior_fallback(test_df):
    train = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
    vote_counts = train[CLASSES].astype("float32").values
    row_sums = vote_counts.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums <= 0, 1.0, row_sums)
    vote_props = vote_counts / row_sums
    prior = vote_props.mean(axis=0).astype("float32")
    prior = np.clip(prior, 1e-12, 1.0)
    prior = prior / prior.sum()
    return np.tile(prior[None, :], (len(test_df), 1)).astype("float32")


def _blend_with_prior(pred, prior, alpha: float):
    blended = (1.0 - alpha) * pred + alpha * prior[None, :]
    blended = np.clip(blended, 1e-12, 1.0)
    blended = blended / blended.sum(axis=1, keepdims=True)
    return blended


def _apply_temperature(pred, t: float):
    pred = np.clip(pred, 1e-12, 1.0)
    logp = np.log(pred)
    logp = logp / float(t)
    logp = logp - logp.max(axis=1, keepdims=True)
    p = np.exp(logp)
    p = p / p.sum(axis=1, keepdims=True)
    return p.astype("float32")


def _dirichlet_smooth(pred, eps: float = 0.01):
    k = pred.shape[1]
    sm = (1.0 - eps) * pred + eps * (1.0 / k)
    sm = np.clip(sm, 1e-12, 1.0)
    sm = sm / sm.sum(axis=1, keepdims=True)
    return sm.astype("float32")


prior_pred = _class_prior_fallback(test)
prior = prior_pred[0].astype("float32")

loadable_paths = []
for p in model_paths:
    m = _try_load_torch_model(p, device)
    if m is not None:
        loadable_paths.append(p)
    del m
    if len(loadable_paths) >= MAX_CKPTS:
        break

if len(loadable_paths) == 0:
    print(
        "WARNING: No loadable/compatible model checkpoints found in /kaggle/input. "
        "Falling back to class-prior predictions computed from train.csv vote proportions."
    )
    test_pred = prior_pred
    test_pred = _dirichlet_smooth(test_pred, eps=0.01)
else:
    print(
        f"Found {len(loadable_paths)} loadable EfficientNet-B0 checkpoints. Running inference and ensembling..."
    )
    for model_path in loadable_paths:
        model = _try_load_torch_model(model_path, device)
        if model is None:
            continue
        print("Using checkpoint:", model_path)
        pred = run_inference_loop(model, test_gen_both, device)
        preds.append(pred)

    test_pred = np.mean(np.stack(preds, axis=0), axis=0)

    test_pred = _blend_with_prior(test_pred, prior, alpha=0.08)
    test_pred = _apply_temperature(test_pred, t=0.95)

    test_pred = _dirichlet_smooth(test_pred, eps=0.01)

test_pred = np.clip(test_pred, 1e-12, 1.0)
test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df = pd.concat(
    [test[["eeg_id"]].reset_index(drop=True), test_pred_df.reset_index(drop=True)],
    axis=1,
)
test_pred_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_pred_df.shape)
test_pred_df.head()
