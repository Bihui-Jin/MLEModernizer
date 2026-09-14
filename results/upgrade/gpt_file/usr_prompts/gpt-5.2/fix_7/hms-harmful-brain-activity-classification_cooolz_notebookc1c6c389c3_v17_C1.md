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
mne==1.10.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
sklearn-pandas==2.2.0
timm==1.0.19
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

0.4102354133265912

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the missing/incorrect model weight paths by auto-detecting valid weight files under `/kaggle/input` and filtering out directories so `torch.load` never receives a folder path. I also correct the dataset’s lead grouping bug (`brain_leads` mistakenly repeats `RP` and omits `RR`) and remove references to undefined variables in the EEG offset logic so inference can run. Finally, I guarantee the submission has the right shape and that each row sums to 1 by applying a small numerical renormalization after ensembling, then write `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far worse than the target (0.4102), so we should improve it with minimal, metric-aligned fixes without changing the overall model/inference design. The biggest issue is that you’re applying `softmax` inside the models even though the loaded weights were almost certainly trained with logits + `CrossEntropyLoss`; double-softmax (or training on logits) hurts calibration and KL. I remove the internal softmax, then apply a single stable softmax at inference time (still producing valid per-row probabilities that sum to 1). I also load checkpoints a bit more robustly (handling common `module.` prefixes) to ensure weights actually apply correctly rather than silently missing keys, which can drastically degrade predictions.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far worse than the target (0.4102), so we should improve calibration/validity with minimal, metric-aligned fixes rather than changing the modeling approach. The biggest likely score-killer here is that you ensemble logits from potentially different checkpoints without any temperature calibration; a small, global temperature chosen from the training-label prior typically improves KL noticeably while keeping the exact same models and inference loop. I add a single, deterministic “prior-matching temperature” computed from `train.csv` vote-distribution, then apply it to the ensembled logits before softmax (one softmax only). I also make the spectrogram region slicing bug-free (your `r` never increments, so all 4 channels are identical), which is a minimal correctness fix to the existing feature extraction and should materially reduce KL.'
- What this solution (achieved 1.40995) has done: 'I make two minimal, correctness-and-metric-aligned changes that should materially reduce KL without altering your core model/inference design. First, I fix the spectrogram slicing to match the HMS baseline convention (use the correct 300-row band per region without drifting `r`), because the current `r += 300` logic can misalign channels and hurt predictions. Second, I make the prior-temperature step deterministic and safer by selecting the temperature using only a small, fixed subset of test logits (so it’s stable and fast) while keeping your prior-matching approach and single softmax; this should improve calibration toward the target score. Everything else (models, ensembling, softmax-once, submission formatting) stays the same and the script still writes `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.4102), so we should improve it with minimal, metric-aligned fixes while keeping the same models, weights, and inference loop. The biggest likely score-killer is that the spectrogram slicing is wrong: the current code crops both time and frequency by region, producing nonstandard/mostly-empty tiles instead of the usual “4 regions × 300 freq rows, full time” inputs the baseline expects; fixing this is a correctness bug in feature extraction (not a modeling change) and should materially reduce KL. I also ensure the spectrogram input always has exactly 4 channels (pad/repeat) to match `NetSpec`’s forward logic, preventing silent mis-shaping that can ruin predictions. Everything else (ensembling logits, optional prior temperature, single softmax, submission format) stays the same.'
- What this solution (achieved 1.40995) has done: 'I make two minimal, correctness-and-metric-aligned fixes that can substantially lower KL while keeping your models, weights, and inference loop intact. First, I reorder the spectrogram tensor to match EfficientNet’s expected (N, C, H, W) layout (your current code feeds time as height and freq as width, which is very likely wrong and can destroy performance). Second, I ensure the spectrogram “image” height is stable across files by cropping/padding the time axis to a fixed length, avoiding variable-sized inputs that can shift features unpredictably and harm calibration. Everything else (weight discovery/loading, models, ensembling logits, optional prior-temperature, single softmax, submission formatting) stays the same.'

# 9. Code solution

## === cell 0
import random
import cv2
import json
import numpy as np
import copy
import pandas as pd
import torch
import gc

import albumentations as A
import os
import librosa
import pickle
import timm
from tqdm import tqdm
import mne

import torchaudio
import torch.nn as nn
from torch.utils.data import DataLoader



## === cell 1
CFG = {
    "batch_size": 32,
    "num_worker": 4,
    "data": "/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
    "weights_spec": "/kaggle/input/hms-baseline",
    "weights_eeg": "/kaggle/input/hms-eeg",
    "flip": True,
    "train_csv": "/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
    "use_prior_temperature": True,
    "temperature_grid": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5],
    "temp_fit_max_rows": 2048,
    "seed": 2024,
    "spec_time_bins": 512,  # crop/pad time dimension deterministically for consistent CNN receptive field
}




## === cell 2
def _list_weight_files(root_dir: str):
    exts = (".pt", ".pth", ".bin")
    files = []
    if root_dir and os.path.exists(root_dir):
        for dirpath, _, filenames in os.walk(root_dir):
            for fn in filenames:
                if fn.lower().endswith(exts):
                    files.append(os.path.join(dirpath, fn))
    return sorted(files)


def _discover_weights_fallback():
    base = "/kaggle/input"
    exts = (".pt", ".pth", ".bin")
    spec_files, eeg_files = [], []
    for dirpath, _, filenames in os.walk(base):
        for fn in filenames:
            if not fn.lower().endswith(exts):
                continue
            p = os.path.join(dirpath, fn)
            lfn = fn.lower()
            ldp = dirpath.lower()
            if ("eeg" in lfn) or ("eeg" in ldp):
                eeg_files.append(p)
            elif ("spec" in lfn) or ("spect" in lfn) or ("baseline" in ldp):
                spec_files.append(p)
            else:
                spec_files.append(p)
    return sorted(set(spec_files)), sorted(set(eeg_files))


def _extract_state_dict(state):
    if isinstance(state, dict):
        for k in ["state_dict", "model", "model_state_dict", "net", "weights"]:
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    if not isinstance(state, dict):
        raise ValueError("Loaded checkpoint is not a state_dict-like dict.")
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    if any(k.startswith("model.") for k in state.keys()):
        state = {k.replace("model.", "", 1): v for k, v in state.items()}
    return state


def _stable_softmax_np(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _compute_train_prior(train_csv_path: str):
    targets = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    df = pd.read_csv(train_csv_path, usecols=targets)
    votes = df[targets].to_numpy(dtype=np.float64)
    votes = np.clip(votes, 0.0, None)
    row_sum = votes.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    probs = votes / row_sum
    prior = probs.mean(axis=0)
    prior = np.clip(prior, 1e-12, 1.0)
    prior = prior / prior.sum()
    return prior


def _pick_temperature_to_match_prior(logits: np.ndarray, prior: np.ndarray, grid):
    best_T = 1.0
    best_loss = float("inf")
    for T in grid:
        p = _stable_softmax_np(logits / float(T), axis=1)
        m = p.mean(axis=0)
        m = np.clip(m, 1e-12, 1.0)
        m = m / m.sum()
        loss = float(np.sum(prior * (np.log(prior) - np.log(m))))
        if loss < best_loss:
            best_loss = loss
            best_T = float(T)
    return best_T, best_loss


spec_weights = _list_weight_files(CFG["weights_spec"])
eeg_weights = _list_weight_files(CFG["weights_eeg"])

if len(spec_weights) == 0 and len(eeg_weights) == 0:
    spec_weights, eeg_weights = _discover_weights_fallback()

CFG["weights_spec"] = [p for p in spec_weights if os.path.isfile(p)]
CFG["weights_eeg"] = [p for p in eeg_weights if os.path.isfile(p)]

print(f"Found spec weight files: {len(CFG['weights_spec'])}")
print(f"Found eeg  weight files: {len(CFG['weights_eeg'])}")
print("Example spec weights:", CFG["weights_spec"][:3])
print("Example eeg  weights:", CFG["weights_eeg"][:3])




## === cell 3
class AlaskaDataIter:
    def __init__(
        self, df, training_flag=False, shuffle=False, use_eeg=False, ll=0, rr=20
    ):

        self.ll = ll
        self.rr = rr
        self.training_flag = training_flag
        self.shuffle = shuffle
        self.raw_data_set_size = None
        self.df = df

        self.train_trans = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
            ]
        )

        TARS = {"Seizure": 0, "LPD": 1, "GPD": 2, "LRDA": 3, "GRDA": 4, "Other": 5}
        self.TARS2 = {x: y for y, x in TARS.items()}

        self.eeg_nms = [
            "Fp1",
            "F3",
            "C3",
            "P3",
            "F7",
            "T3",
            "T5",
            "O1",
            "Fz",
            "Cz",
            "Pz",
            "Fp2",
            "F4",
            "C4",
            "P4",
            "F8",
            "T4",
            "T6",
            "O2",
            "EKG",
        ]

        self.LL = ["Fp1", "F7", "T3", "T5", "O1"]
        self.RR = ["Fp2", "F8", "T4", "T6", "O2"]
        self.LP = ["Fp1", "F3", "C3", "P3", "O1"]
        self.RP = ["Fp2", "F4", "C4", "P4", "O2"]

        self.leads_dict = {value: index for index, value in enumerate(self.eeg_nms)}
        self.use_eeg = use_eeg

    def __getitem__(self, item):
        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):
        return len(self.df)

    def brain_lead(self, waves):
        waves = copy.deepcopy(waves)

        brain_leads = [self.LL, self.LP, self.RP, self.RR]

        leads = []
        for combine in brain_leads:
            for i in range(len(combine) - 1):
                tmp_lead = (
                    waves[self.leads_dict[combine[i]]]
                    - waves[self.leads_dict[combine[i + 1]]]
                )
                leads.append(tmp_lead)

        data = np.concatenate([leads], axis=0)
        return data

    def _crop_or_pad_time(self, x_2d: np.ndarray, target_t: int):
        t = x_2d.shape[0]
        if t == target_t:
            return x_2d
        if t > target_t:
            return x_2d[:target_t, :]
        pad = target_t - t
        return np.pad(x_2d, ((0, pad), (0, 0)), mode="edge")

    def single_map_func(self, dp, is_training):
        if self.use_eeg:
            eeg_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/{dp['eeg_id']}.parquet"
            eeg = pd.read_parquet(eeg_path)

            offset = 0
            eeg = eeg.iloc[offset * 200 : (offset + 50) * 200]

            waves = eeg.values
            waves = np.transpose(waves, axes=[1, 0])

            for i in range(waves.shape[0]):
                m = np.nanmean(waves[i])
                if np.isnan(waves[i]).mean() < 1:
                    waves[i] = np.nan_to_num(waves[i], nan=m)
                else:
                    waves[i] = 0

            waves = np.array(waves, dtype=np.float64)
            waves = mne.filter.filter_data(waves, 200, self.ll, self.rr, verbose=False)
            waves = self.brain_lead(waves)
            data = waves
        else:
            spec_path = f"/kaggle/input/hms-harmful-brain-activity-classification/test_spectrograms/{dp['spectrogram_id']}.parquet"
            spec = pd.read_parquet(spec_path)
            spec = spec.values[:, 1:]

            images = []
            for region in range(4):
                band = spec[region * 300 : (region + 1) * 300, :].T  # (time, freq=300)
                band = np.clip(band, np.exp(-4), np.exp(8))
                band = np.log(band)
                band = np.nan_to_num(band, nan=0.0)

                band = self._crop_or_pad_time(band, int(CFG["spec_time_bins"]))
                images.append(band)

            if len(images) < 4:
                while len(images) < 4:
                    images.append(images[-1].copy())
            elif len(images) > 4:
                images = images[:4]

            images = np.stack(images, -1)  # (time, freq, ch=4)

            data = np.transpose(
                images, [2, 1, 0]
            )  # (ch=4, freq=300, time=spec_time_bins)

        return data.astype(np.float32)




## === cell 4
class NetSpec(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=3)
        self.fc = nn.Linear(2048, 6, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)
        x1 = [x[:, i : i + 1, :, :] for i in range(4)]
        x1 = torch.cat(x1, dim=2)
        x = torch.cat([x1, x1, x1], dim=1)

        if CFG["flip"]:
            x_flip = torch.flip(x, [3])
            x = torch.cat([x, x_flip], dim=0)

        x = self.model.forward_features(x)
        x = self.avg(x)

        if CFG["flip"]:
            x = x.view(2 * bs, -1)
        else:
            x = x.view(bs, -1)

        x = self.dropout(x)
        x = self.fc(x)

        if CFG["flip"]:
            ans = (x[:bs, ...] + x[bs:, ...]) / 2.0
        else:
            ans = x
        return ans




## === cell 5
class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.wave_transform = torchaudio.transforms.Spectrogram(
            n_fft=512, hop_length=25, power=1
        )
        self.am2db = torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80)
        self.resizer = nn.UpsamplingBilinear2d(size=[160, 320])

    def forward(self, x):
        image = self.wave_transform(x)
        image = self.am2db(image)
        n, c, h, w = image.size()
        image = image[:, :, : int(20 / 100 * h + 30), :]
        image = torch.reshape(image, shape=[n, 4, -1, w])
        return image


class NetEeg(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.preprocess = Transform()
        self.model = timm.create_model("efficientnet_b5", pretrained=False, in_chans=4)
        self.fc = nn.Linear(2048, 6, bias=True)
        self.dropout = nn.Dropout(0.5)
        self.avg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        bs = x.size(0)
        x = self.preprocess(x)

        if CFG["flip"]:
            x_flip = torch.flip(x, [3])
            x = torch.cat([x, x_flip], dim=0)

        x = self.model.forward_features(x)
        x = self.avg(x)

        if CFG["flip"]:
            x = x.view(2 * bs, -1)
        else:
            x = x.view(bs, -1)

        x = self.dropout(x)
        x = self.fc(x)

        if CFG["flip"]:
            ans = (x[:bs, ...] + x[bs:, ...]) / 2.0
        else:
            ans = x
        return ans




## === cell 6
def inference_function(test_loader, model, device):
    model.eval()
    preds = []
    with tqdm(test_loader, unit="test_batch", desc="Inference") as tqdm_test_loader:
        for _, X in enumerate(tqdm_test_loader):
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            preds.append(y_preds.detach().to("cpu").numpy())
    return {"predictions": np.concatenate(preds, axis=0)}




## === cell 7
test_df = pd.read_csv(CFG["data"])
print(test_df.head(3))
print("Test rows:", len(test_df))



## === cell 8
predictions_list = []
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if len(CFG["weights_spec"]) > 0:
    print("infer with weights_spec")
    test_dataset = AlaskaDataIter(
        test_df, training_flag=False, shuffle=False, use_eeg=False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_worker"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    for model_weight in CFG["weights_spec"]:
        model = NetSpec()
        state = torch.load(model_weight, map_location=device)
        state = _extract_state_dict(state)
        model.load_state_dict(state, strict=False)
        model.to(device)
        prediction_dict = inference_function(test_loader, model, device)
        predictions_list.append(prediction_dict["predictions"])  # logits
        del model, state, prediction_dict
        torch.cuda.empty_cache()
        gc.collect()
else:
    print("No spec weights found; skipping spec inference.")

if len(CFG["weights_eeg"]) > 0:
    print("infer with weights_eeg")
    test_dataset = AlaskaDataIter(
        test_df, training_flag=False, shuffle=False, use_eeg=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG["batch_size"],
        num_workers=CFG["num_worker"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    for model_weight in CFG["weights_eeg"]:
        model = NetEeg()
        state = torch.load(model_weight, map_location=device)
        state = _extract_state_dict(state)
        model.load_state_dict(state, strict=False)
        model.to(device)
        prediction_dict = inference_function(test_loader, model, device)
        predictions_list.append(prediction_dict["predictions"])  # logits
        del model, state, prediction_dict
        torch.cuda.empty_cache()
        gc.collect()
else:
    print("No eeg weights found; skipping eeg inference.")

if len(predictions_list) == 0:
    print(
        "WARNING: No model weights found. Creating uniform predictions for a valid submission."
    )
    predictions = np.full((len(test_df), 6), 1.0 / 6.0, dtype=np.float64)
else:
    logits = np.mean(np.stack(predictions_list, axis=0), axis=0).astype(np.float64)

    if CFG.get("use_prior_temperature", True) and os.path.exists(CFG["train_csv"]):
        rng = np.random.default_rng(CFG["seed"])
        n = logits.shape[0]
        m = min(int(CFG["temp_fit_max_rows"]), n)
        idx = rng.choice(n, size=m, replace=False) if m < n else np.arange(n)

        train_prior = _compute_train_prior(CFG["train_csv"])
        T, loss = _pick_temperature_to_match_prior(
            logits[idx], train_prior, CFG["temperature_grid"]
        )
        print(
            f"Using prior-matching temperature T={T:.3f} (grid KL(prior||mean_pred)={loss:.6f}, fit_rows={m})"
        )
        logits = logits / T

    predictions = _stable_softmax_np(logits, axis=1)

if predictions.shape != (len(test_df), 6):
    raise ValueError(
        f"Predictions shape mismatch: got {predictions.shape}, expected {(len(test_df), 6)}"
    )

predictions = np.clip(predictions, 1e-12, 1.0)
predictions = predictions / predictions.sum(axis=1, keepdims=True)



## === cell 9
TARGETS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]
sub = pd.DataFrame({"eeg_id": test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv("submission.csv", index=False)
print(f"Submission shape: {sub.shape}")
print(sub.head())
print(
    "Row sums min/max:", sub[TARGETS].sum(axis=1).min(), sub[TARGETS].sum(axis=1).max()
)
