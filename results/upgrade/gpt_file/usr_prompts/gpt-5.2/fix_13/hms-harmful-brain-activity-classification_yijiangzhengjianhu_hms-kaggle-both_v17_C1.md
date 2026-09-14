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

0.5851202940182169

# 6. Current score

2.34706

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I fix the runtime failure by making the model checkpoint discovery robust to the actual Kaggle input directory structure (and fall back safely if no external model dataset is attached). I also fix a critical bug in `denoise()` (it returned `None`), which can silently break spectrogram generation if wavelet denoising is enabled. To ensure a valid submission is always produced, I add a deterministic “prior” fallback prediction (derived from normalized train vote counts) that writes a correctly formatted `submission.csv` even when no `.pt` files are available. These changes keep the existing data pipeline and inference semantics intact when models are present, and only alter behavior to avoid crashes and produce a valid CSV otherwise.'
- What this solution (achieved 1.41937) has done: 'Your score gap is large (1.41937 vs target 0.5851; lower is better), and the biggest issue is that you are likely submitting the fallback “train-vote prior” because no `.pt` checkpoints are being found/loaded; that baseline is typically weak for KL on this competition. I make the model checkpoint discovery/load robust to common HMS checkpoint formats (full model object, `state_dict` with metadata, and TorchScript), while keeping the same inference loop, softmax, and ensembling logic. I also ensure the test generator uses the expected `spec_id` field (it’s currently renamed, but the generator still expects `spec_id` consistently) and that inference runs in reasonable time by enabling pinned-memory transfers only when CUDA is available (no semantic change). The goal is simply to actually use the provided pretrained models if present, which should move the score down toward your target without changing the model architecture or training scheme.'
- What this solution (achieved 1.41937) has done: 'Your current score (1.419) is far worse than the target (0.585, lower-is-better), and the most likely cause is that you’re still effectively near a weak “prior-like” baseline because inference is not actually using the intended pretrained models correctly and/or the input normalization expected by those models isn’t being applied. I keep your feature pipeline and inference loop intact, but make two minimal score-relevant fixes: (1) apply the same ImageNet-style normalization commonly required by these pretrained vision backbones (this usually yields a large KL improvement without changing architecture), and (2) make checkpoint loading robust to “dict with model under common keys” and ensure Tensor output shape is handled consistently. The submission writing stays identical, including strict probability normalization and correct columns.'
- What this solution (achieved 1.41937) has done: 'Your score gap is large (1.419 → target 0.585, lower-is-better), and the most likely reason is that you’re still submitting the weak train-vote prior because no usable `.pt` checkpoints are actually being found/loaded. I make the checkpoint discovery more precise to the HMS dataset’s actual `/kaggle/input/hms-harmful-brain-activity-classification/...` structure and add support for common “state_dict-only” checkpoints by instantiating the same backbone via `torchvision` (keeping inference the same: resize+normalize+softmax, no training changes). I also make the generator robust to missing `offset` in test mode (even if a code path accidentally accesses it) so it doesn’t silently break. These are minimal, score-relevant changes: they aim to ensure you’re using the intended pretrained weights rather than the fallback baseline, which should reduce KL toward the target.'
- What this solution (achieved 1.22399) has done: 'Your gap to the target is large (1.41937 vs 0.58512; lower is better), and the most likely score limiter is that inference is effectively using weak/unintended checkpoints (or none), plus a critical mismatch: your generator outputs 896×896 inputs while the fallback torchvision backbones you construct expect 3×224×224, so those models would be unusable or produce poor results. I make two minimal, score-relevant fixes without changing the core pipeline: (1) ensure the input size matches the constructed torchvision backbones by changing only the validation resize to 224×224, and (2) if no usable checkpoints are found, use a stronger-but-still-legit fallback: patient-grouped mean label distribution from train (instead of global prior), mapped to test by patient_id. This keeps your overall logic (spectrogram construction → image tensor → model → softmax → average) intact, but should substantially reduce KL toward your target.'
- What this solution (achieved 1.22399) has done: 'Your current score (1.22399) is still far from the target (0.58512, lower-is-better), and the most likely reason is that you are not actually using any meaningful pretrained checkpoints (so you’re effectively submitting a weak prior). To move the score down with minimal semantic change, I keep your exact spectrogram construction and inference loop, but (1) restrict checkpoint discovery to “likely-real” HMS model files and ignore tiny/random `.pt` artifacts, and (2) correctly handle checkpoints saved from `DataParallel`/Lightning by stripping a leading `module.` prefix and loading into the same fallback torchvision backbone (so weights actually apply). If no usable checkpoints exist, I keep your patient-mean fallback unchanged. This should reduce KL toward the target without changing the model architecture or training approach.'
- What this solution (achieved 1.22399) has done: 'Your score is much worse than the target (1.22399 vs 0.58512; lower is better), so we should improve performance with minimal semantic changes. The biggest likely issue is that your test `DataGenerator` in `data_type="both"` expects a column named `offset`, but `test.csv` does not have it; depending on how your notebook ran, this can silently break or push you into the weak fallback prior. I (1) create an explicit `offset=0` column for test rows and (2) make `generate_all_specs()`/`generate_specs()` use the correct train column name (`spectrogram_label_offset_seconds`) when present, while keeping test behavior identical. This keeps your architecture/inference loop intact, but fixes a correctness bug that can materially affect the resulting predictions and thus reduce KL toward your target.'
- What this solution (achieved 2.34706) has done: 'Your current score (1.22399, lower-is-better) is still far from the target (0.5851), so we should improve performance with minimal semantic change. The most likely limiter is that the model inference path is either not being used (or is effectively random due to mismatched checkpoint/model pairing), and the fallback “patient mean” prior is too weak. I keep your exact spectrogram construction and inference loop, but (1) tighten checkpoint discovery to only load checkpoints from attached *model* datasets (avoid accidentally loading irrelevant `.pt` files), and (2) add a stronger-yet-legit fallback that uses the train distribution conditional on `expert_consensus` *within each patient* (and backs off to patient-mean, then global-mean). This should reduce KL meaningfully toward your target while preserving your core pipeline and still always producing a valid `submission.csv`.'
- What this solution (achieved 2.34706) has done: 'Your score (2.347, lower-is-better) is far worse than the target (0.585), so we should make a small, high-impact correctness fix rather than tune anything. The biggest issue in your current pipeline is a scale mismatch: Kaggle spectrogram parquets are already log-like values, but your generator treats them as linear power and applies `clip(exp(-4), exp(8))` then `log`, which badly distorts the input and can make any pretrained checkpoints (if found) effectively useless. I keep your exact model inference/ensembling logic and image assembly layout, but change only the Kaggle-spec preprocessing to a robust “use as-is, sanitize, then min-max to 0..255” path. This should materially reduce KL while preserving core semantics and still producing a valid `submission.csv`.'
- What this solution (achieved 2.34706) has done: 'Your score is far above (worse than) the target, so the smallest high-impact improvement is to ensure the inference models (when present) see inputs in the exact layout they were trained on. Right now `generate_all_specs()` applies `self.trans(...)` but `generate_specs()` does not, so if you ever run in `data_type="kaggle"` or `"eeg"` you’d feed unnormalized, wrong-sized tensors and get near-random predictions; I apply the same transform there (no architecture/training change). I also fix a critical iteration bug in `run_inference_loop`: `total=len(test_gen_both)` is currently invalid (len() on an object), which can break the loop or progress bar; I set it to `len(test_gen_both)` (the generator length) while keeping the loop semantics identical. Finally, I add a tiny numerical safeguard for offsets so we never slice outside the Kaggle spectrogram array (which can silently create degenerate images and worsen KL).'
- What this solution (achieved 2.34706) has done: 'Your current score (2.347, lower-is-better) is far worse than the target (0.585), so we should focus on a small, correctness-first fix that can materially improve KL without changing the model or training logic. The biggest silent bug is that `DataGenerator.__call__()` reshuffles `self.data` at the end of the test iteration (because `on_epoch_end()` shuffles whenever `mode=="train"` is false-checked incorrectly via `self.mode` but still gets called unconditionally); this can misalign predictions vs `eeg_id` ordering when you later concatenate with `test[["eeg_id"]]`, producing a valid CSV with wrong rows (catastrophic for KL). I (1) prevent any shuffling/on_epoch_end calls during test iteration and (2) build the submission using the exact `eeg_id` sequence emitted by the generator (tracked alongside predictions) to guarantee alignment. These are minimal changes that preserve your core feature pipeline, model inference, softmax, and ensembling semantics, but should substantially reduce the score toward your target by fixing label/id mismatch.'

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

if "offset" not in test.columns:
    test["offset"] = 0

PATH2 = "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
DISPLAY = 0
EEG_IDS2 = test.eeg_id.unique()
all_eegs2 = {}

print("\nConverting Test EEG to Spectrograms...\n")



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
    coeff[1:] = [pywt.threshold(i, value=uthresh, mode="hard") for i in coeff[1:]]
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
        if self.mode == "train":
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

    def _get_spec_offset(self, row):
        if self.mode == "test":
            return 0
        if "offset" in row.index:
            return int(row.offset / 2)
        if "spectrogram_label_offset_seconds" in row.index:
            return int(row.spectrogram_label_offset_seconds / 2)
        return 0

    def _kaggle_spec_to_image(self, spec, offset):
        h = spec.shape[0]
        offset = int(np.clip(offset, 0, max(0, h - 300)))

        imgs = [
            spec[offset : offset + 300, k * 100 : (k + 1) * 100].T for k in [0, 2, 1, 3]
        ]
        img = np.stack(imgs, axis=-1).astype(np.float32)

        img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)

        lo = np.percentile(img, 1)
        hi = np.percentile(img, 99)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = img.min(), img.max()
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return np.zeros_like(img, dtype=np.float32)
        img = np.clip(img, lo, hi)
        return img

    def generate_all_specs(self, index):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[index]
        offset = self._get_spec_offset(row)

        eeg = self.eeg_specs[row.eeg_id]
        spec = self.specs[row.spec_id]

        img = self._kaggle_spec_to_image(spec, offset)

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

        X = self.trans(image=X)["image"]
        if self.mode != "test":
            y[:] = row[TARGETS]

        return X, y

    def generate_specs(self, index):
        X = np.zeros((512, 512, 3), dtype="float32")
        y = np.zeros((6,), dtype="float32")

        row = self.data.iloc[index]
        offset = self._get_spec_offset(row)

        if self.data_type == "eeg":
            img = self.eeg_specs[row.eeg_id]
            img = img.astype(np.float32)
            img = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0)
        elif self.data_type == "kaggle":
            spec = self.specs[row.spec_id]
            img = self._kaggle_spec_to_image(spec, offset)

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

        if self.mode != "test":
            y[:] = row[TARGETS]

        return X, y




## === cell 3
val_transform = A.Compose(
    [
        A.Resize(p=1.0, height=224, width=224),
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
import torchvision


def run_inference_loop(model, test_gen_both, device):
    model.to(device)
    model.eval()
    pred_list = []
    id_list = []
    with torch.no_grad():
        for i, (X, _) in enumerate(
            tqdm.tqdm(test_gen_both(), total=len(test_gen_both))
        ):
            id_list.append(int(test_gen_both.data.iloc[i].eeg_id))
            X = X.unsqueeze(0).to(device, non_blocking=(device.type == "cuda"))
            logits = model(X)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            probs = torch.softmax(logits, dim=1)
            pred_list.append(probs.detach().cpu().numpy())

    pred_arr = np.concatenate(pred_list, axis=0)
    del pred_list
    return pred_arr, np.asarray(id_list, dtype=np.int64)


def _strip_module_prefix_in_state_dict(sd):
    if not isinstance(sd, dict) or len(sd) == 0:
        return sd
    keys = list(sd.keys())
    if all(isinstance(k, str) and k.startswith("module.") for k in keys):
        return {k[len("module.") :]: v for k, v in sd.items()}
    return sd


def _build_fallback_arch_for_statedict(sd, num_classes=6):
    sd = _strip_module_prefix_in_state_dict(sd)

    candidates = [
        ("efficientnet_b0", torchvision.models.efficientnet_b0),
        ("efficientnet_b2", torchvision.models.efficientnet_b2),
        ("resnet18", torchvision.models.resnet18),
        ("resnet34", torchvision.models.resnet34),
    ]

    for name, ctor in candidates:
        try:
            m = ctor(weights=None)
            if hasattr(m, "classifier") and isinstance(
                m.classifier, torch.nn.Sequential
            ):
                if isinstance(m.classifier[-1], torch.nn.Linear):
                    in_f = m.classifier[-1].in_features
                    m.classifier[-1] = torch.nn.Linear(in_f, num_classes)
            elif hasattr(m, "fc") and isinstance(m.fc, torch.nn.Linear):
                in_f = m.fc.in_features
                m.fc = torch.nn.Linear(in_f, num_classes)

            missing, unexpected = m.load_state_dict(sd, strict=False)

            if len(unexpected) == 0 and len(missing) < 20:
                print(
                    f"  Loaded state_dict into torchvision.{name} (missing={len(missing)})"
                )
                return m
        except Exception:
            pass

    raise ValueError("Could not match state_dict to a supported torchvision backbone.")


def _as_model_or_torchscript(ckpt_obj, model_path, device):
    if isinstance(ckpt_obj, torch.nn.Module):
        return ckpt_obj

    if isinstance(ckpt_obj, dict):
        for k in ["model", "net", "module", "student", "ema"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], torch.nn.Module):
                return ckpt_obj[k]

        for k in ["state_dict", "model_state_dict", "net_state_dict"]:
            if k in ckpt_obj and isinstance(ckpt_obj[k], dict):
                return _build_fallback_arch_for_statedict(ckpt_obj[k], num_classes=6)

        if len(ckpt_obj) > 0 and all(
            isinstance(v, torch.Tensor) for v in ckpt_obj.values()
        ):
            return _build_fallback_arch_for_statedict(ckpt_obj, num_classes=6)

    ts = torch.jit.load(model_path, map_location=device)
    return ts


def find_model_paths():
    patterns = [
        "/kaggle/input/**/hms*/*.pt",
        "/kaggle/input/**/hms*/*.pth",
        "/kaggle/input/**/model*/*.pt",
        "/kaggle/input/**/model*/*.pth",
        "/kaggle/input/**/checkpoint*/*.pt",
        "/kaggle/input/**/checkpoint*/*.pth",
        "/kaggle/input/**/fold*/*.pt",
        "/kaggle/input/**/fold*/*.pth",
        "/kaggle/input/**/best*/*.pt",
        "/kaggle/input/**/best*/*.pth",
    ]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(pat, recursive=True))

    paths = [p for p in paths if (p.endswith(".pt") or p.endswith(".pth"))]
    paths = sorted(set(paths))

    good = []
    for p in paths:
        if "hms-harmful-brain-activity-classification" in p:
            continue
        try:
            sz = os.path.getsize(p)
            if 1_000_000 <= sz <= 2_000_000_000:
                good.append(p)
        except OSError:
            continue
    return good


def patient_expert_consensus_fallback_predictions(test_df, train_csv_path, classes):
    train = pd.read_csv(
        train_csv_path, usecols=["patient_id", "expert_consensus"] + classes
    )

    y = train[classes].to_numpy(dtype=np.float64)
    y = np.clip(y, 0.0, None)
    row_sum = y.sum(axis=1, keepdims=True)
    row_sum = np.clip(row_sum, 1e-12, None)
    y = y / row_sum

    train_probs = pd.DataFrame(y, columns=classes)
    train_probs["patient_id"] = train["patient_id"].values
    train_probs["expert_consensus"] = train["expert_consensus"].astype(str).values

    group_mean = train_probs.groupby(["patient_id", "expert_consensus"])[classes].mean()
    patient_mean = train_probs.groupby("patient_id")[classes].mean()

    global_mean = train_probs[classes].mean(axis=0).to_numpy(dtype=np.float64)
    global_mean = np.clip(global_mean, 1e-9, None)
    global_mean = global_mean / global_mean.sum()

    patient_mode_cons = (
        train[["patient_id", "expert_consensus"]]
        .assign(expert_consensus=train["expert_consensus"].astype(str))
        .groupby("patient_id")["expert_consensus"]
        .agg(lambda s: s.value_counts().index[0])
    )

    out = np.zeros((len(test_df), len(classes)), dtype=np.float32)
    for i, pid in enumerate(test_df["patient_id"].values):
        if pid in patient_mode_cons.index:
            cons = patient_mode_cons.loc[pid]
            key = (pid, cons)
            if key in group_mean.index:
                p = group_mean.loc[key].to_numpy(dtype=np.float64)
                p = np.clip(p, 1e-9, None)
                p = p / p.sum()
                out[i] = p.astype(np.float32)
                continue
        if pid in patient_mean.index:
            p = patient_mean.loc[pid].to_numpy(dtype=np.float64)
            p = np.clip(p, 1e-9, None)
            p = p / p.sum()
            out[i] = p.astype(np.float32)
        else:
            out[i] = global_mean.astype(np.float32)

    return out


CLASSES = TARGETS[:]  # same order

test_gen_both = DataGenerator(
    test,
    mode="test",
    data_type="both",
    specs=spectrograms2,
    eeg_specs=all_eegs2,
    trans=val_transform,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_paths = find_model_paths()
print("Found likely external .pt/.pth checkpoint files:", len(model_paths))
for p in model_paths[:30]:
    print("  ", p)

preds = []
id_ref = None

if len(model_paths) > 0:
    for model_path in model_paths:
        print("Loading:", model_path)
        try:
            ckpt = torch.load(model_path, map_location=device)
            model = _as_model_or_torchscript(ckpt, model_path, device)
            pred, ids = run_inference_loop(model, test_gen_both, device)
            if id_ref is None:
                id_ref = ids
            else:
                if not np.array_equal(id_ref, ids):
                    raise RuntimeError(
                        "Mismatch in eeg_id order between inference runs."
                    )
            preds.append(pred)
        except Exception as e:
            print("  Skipping checkpoint (unusable format):", model_path)
            print("   ", repr(e))

    if len(preds) > 0:
        test_pred = np.mean(np.stack(preds, axis=0), axis=0)
        eeg_id_out = id_ref
    else:
        print(
            "All discovered checkpoints were unusable; falling back to patient+consensus prior."
        )
        test_pred = patient_expert_consensus_fallback_predictions(
            test,
            "/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
            CLASSES,
        )
        eeg_id_out = test["eeg_id"].to_numpy(dtype=np.int64)
else:
    print(
        "No external model checkpoints found; falling back to patient+consensus prior."
    )
    test_pred = patient_expert_consensus_fallback_predictions(
        test,
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
        CLASSES,
    )
    eeg_id_out = test["eeg_id"].to_numpy(dtype=np.int64)

test_pred = np.nan_to_num(test_pred, nan=0.0, posinf=0.0, neginf=0.0)
test_pred = np.clip(test_pred, 1e-9, 1.0)
test_pred = test_pred / test_pred.sum(axis=1, keepdims=True)

test_pred_df = pd.DataFrame(test_pred, columns=CLASSES)
test_pred_df.insert(0, "eeg_id", eeg_id_out)

test_pred_df.to_csv("submission.csv", index=False)
print(test_pred_df.head())
print("Wrote submission.csv with shape:", test_pred_df.shape)
print(
    "Row-sum check (min/max):",
    test_pred_df[CLASSES].sum(axis=1).min(),
    test_pred_df[CLASSES].sum(axis=1).max(),
)
