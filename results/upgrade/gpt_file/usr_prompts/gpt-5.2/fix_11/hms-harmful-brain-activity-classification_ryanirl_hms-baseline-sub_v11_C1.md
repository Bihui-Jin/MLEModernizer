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

cudf-polars-cu12==25.6.0
geopandas==0.14.4
librosa==0.11.0
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
polars==1.25.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scipy==1.15.3
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

0.3642930579177063

# 6. Current score

1.17594

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I remove the hard dependency on the missing `/kaggle/input/hms-models/` package by falling back to a safe baseline prediction when those external model files/modules are unavailable, so the notebook runs end-to-end and always writes `submission.csv`. I fix the `df_test[i]` row access bug (Polars returns a Series, not a 1-row DataFrame) and make the EEG preprocessing robust to the actual test EEG length by center-cropping/padding to the expected 10,000 samples so the existing `reshape(..., 2500)` logic works. I also enforce valid probabilities (non-negative, sum-to-1) and guarantee the submission columns match `sample_submission.csv`. These changes are runtime/correctness focused; they should yield a valid submission and a reasonable score rather than failing.'
- What this solution (achieved 1.39637) has done: 'Your current score is far worse than the target (lower is better), and the biggest issue is that you’re effectively submitting a uniform baseline because the external models aren’t available. To move the score toward the target with minimal logic change, I keep your inference pipeline intact but add a lightweight “prior” fallback learned from `train.csv` vote distributions (smoothed and normalized), so predictions reflect realistic class frequencies rather than uniform. I also cache EEG feature extraction per `eeg_id` to avoid repeated parquet reads and heavy preprocessing, keeping runtime safely within limits while preserving your model inference behavior when models exist. Finally, I keep the strict probability safety normalization to avoid invalid submissions and KL blow-ups.'
- What this solution (achieved 0.88764) has done: 'Your current score (1.39637, lower-is-better) is far from the target (0.36429), and the main driver is that when external models are missing you fall back to a single global prior, which is too uninformative for KL divergence. To move toward the target with minimal semantic change, I keep the exact same model paths and inference flow, but make the fallback smarter by conditioning the prior on `patient_id` (patient-specific class distribution from train) and blending it with the global prior (and uniform smoothing) for robustness. This is still a “prior-only” baseline (no new model, no new features) but it is materially better calibrated to the test distribution and typically reduces KL substantially. I also keep strict probability normalization and keep runtime low by precomputing the patient priors once.'
- What this solution (achieved 1.07336) has done: 'Your current score (0.88764, lower-is-better) is still far from the target (0.36429), and the main limitation is that when external models are missing you’re only using a patient/global prior, which can’t adapt to the specific EEG sample. To move closer with minimal semantic change, I keep your exact inference flow, but make the fallback prior slightly more informative by also conditioning on `spectrogram_id` from `train.csv` when available in test, then blending spectrogram-prior + patient-prior + global prior with small smoothing. This remains a “prior-only” fallback (no new model, no new features, no new training loop), but it better matches annotation distributions for recordings and typically reduces KL. I also keep strict probability safety and ensure the submission columns/order exactly match `sample_submission.csv`.'
- What this solution (achieved 1.1887) has done: 'Your current score (1.07336, lower-is-better) is still far from the target (0.36429), and the main weakness is that the no-external-model fallback is still a static prior that can’t adapt per EEG sample. With minimal changes and preserving your core inference flow, I keep the existing priors but make the fallback use lightweight EEG-derived evidence: compute a cheap per-sample feature vector (bandpower ratios + amplitude stats) and map it to class probabilities via nearest-centroid prototypes learned from `train_eegs/` for a small, fixed number of examples per class. This doesn’t change any model architecture/loss/training loop (it only improves the fallback path when models are missing) and stays within runtime by sampling a small prototype set and caching file reads. I also keep strict probability normalization (eps-clipping + sum-to-1) to avoid KL blow-ups and invalid submissions.'
- What this solution (achieved 1.16396) has done: 'Your current score (1.1887, lower-is-better) is far above the target (0.3643), and the main lever available without changing your core model/inference logic is to make the fallback (no external models) better calibrated and less noisy. I (1) fix a subtle centroid/logit scaling issue that can create overly peaky probabilities (bad for KL) by using a stabler temperature based on centroid dispersion, (2) blend the EEG-centroid evidence with the metadata priors in a more conservative, KL-friendly way (slightly less weight on the EEG “guess”), and (3) add light Dirichlet-style smoothing to the blended distribution before final normalization to avoid near-zeros that blow up KL. These are minimal changes confined to the fallback path and probability post-processing; your existing model path remains untouched.'
- What this solution (achieved 1.17696) has done: 'Your current score (1.16396, lower-is-better) is still far above the target (0.36429), and the external-model path is unused, so the only lever is the fallback. I keep your exact EEG-centroid + (spectrogram/patient/global) priors design, but make it less KL-risky by (1) computing an `eeg_id`-conditioned prior from `train.csv` and blending it into the metadata prior (this is still “prior-only”, no new model), and (2) slightly reducing the centroid influence while adding a bit more smoothing to avoid near-zero probabilities that inflate KL. These are minimal, local changes confined to the fallback path and probability calibration; submission format and all existing preprocessing/model code remain intact. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 1.20564) has done: 'Your current score (1.17696, lower-is-better) is still far from the target (0.36429), so we should improve the fallback path (since external models are absent) with minimal, low-risk changes. The most direct lever for KL is to reduce overconfident near-zero probabilities and improve per-sample calibration without changing your core pipeline: we (1) use log-space blending (geometric mean) of priors + EEG evidence instead of linear mixing, which is typically more KL-friendly, (2) strengthen and make consistent epsilon-flooring/smoothing for both fallback and model predictions to avoid KL blow-ups, and (3) fix a small but important indexing bug in centroid prototype building (`df_train["eeg_id"][ridx]` in Polars) that can silently produce wrong eeg_ids/centroids. These are localized changes: no new model, no new training loop, no changes to your existing external-model inference path, and the script still writes a valid `submission.csv`. Runtime remains within limits because we keep the same capped centroid-building and caching.'
- What this solution (achieved 1.16153) has done: 'Your current score (1.20564, lower-is-better) is still far above the target (0.36429), and because external models are unavailable the only effective lever is the fallback distribution quality and KL-safety. I keep your entire pipeline and fallback structure intact, but make two minimal, directly score-relevant calibration fixes: (1) switch the metadata prior and EEG evidence blending to use a slightly higher epsilon floor (reduces near-zero probs that explode KL), and (2) reduce the centroid “evidence” sharpness by increasing the centroid temperature a bit (less overconfident, typically better for KL). Finally, I slightly re-balance the fallback weights toward metadata priors (more stable than noisy EEG centroid guesses) while preserving the same components and logic, and keep the submission normalization unchanged.'
- What this solution (achieved 1.17594) has done: 'Your current score (1.16153, lower-is-better) is still far from the target (0.36429), and since external models are absent the only viable lever is improving the fallback probabilities without changing the overall pipeline. I keep your exact fallback components (spec/patient/eeg_id/global priors + EEG-centroid evidence + log-space blending) but make two minimal, KL-relevant calibration tweaks: (1) use an `eeg_id` prior keyed by `label_id` consolidation (train rows for the same eeg_id can represent different overlapping segments) by averaging per `label_id` first to reduce noise, and (2) slightly increase the EEG evidence weight while simultaneously increasing smoothing/epsilon a bit to avoid near-zeros (which dominate KL). These changes preserve your architecture/inference logic, don’t add new training loops/models, and should move KL down toward the target while keeping the submission valid.'

# 9. Code solution

## === cell 0
from tqdm.auto import tqdm
import pandas as pd
import polars as pl
import numpy as np
import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from typing import Optional, Tuple, List, Dict, Any



## === cell 1
DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
SPEC_DIR = os.path.join(DATA_DIR, "test_spectrograms/")
EEG_DIR = os.path.join(DATA_DIR, "test_eegs/")

TRAIN_EEG_DIR = os.path.join(DATA_DIR, "train_eegs/")

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

df_test.head()



## === cell 2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()




## === cell 3
def load_model(path: str, model: nn.Module) -> nn.Module:
    ckpt = torch.load(path, map_location=torch.device("cpu"))
    state = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    model.load_state_dict(state)
    return model




## === cell 4
import sys
import glob

HMS_MODELS_DIR = "/kaggle/input/hms-models/"
HAVE_EXTERNAL_MODELS = os.path.isdir(HMS_MODELS_DIR)

models_0, models_1 = [], []

if HAVE_EXTERNAL_MODELS:
    sys.path.append(HMS_MODELS_DIR)
    try:
        from eeg_cnn_rnn_w1 import EegModel as EegModel0
        from eeg_cnn_rnn import EegModel as EegModel1

        for fold_path in glob.glob("/kaggle/input/hms-models/new_fold/send_kaggle/*"):
            model = EegModel0()
            model = load_model(fold_path, model).to(device)
            models_0.append(model)

        for fold_path in glob.glob("/kaggle/input/hms-models/baseline_eeg_diff/*"):
            model_path = os.path.join(fold_path, "model_best_val_g10.pt")
            if os.path.isfile(model_path):
                model = EegModel1()
                model = load_model(model_path, model).to(device)
                models_1.append(model)
    except Exception as e:
        print(
            "External models present but failed to import/load; falling back to baseline. Error:",
            repr(e),
        )
        models_0, models_1 = [], []
else:
    print(
        f"External models dir not found at {HMS_MODELS_DIR}; using baseline submission."
    )

print("Loaded models:", len(models_0) + len(models_1))



## === cell 5
import polars as pl
import librosa
import numpy as np

from scipy.ndimage import convolve

KERNEL = np.array([-1, -1, -1, 0, 1, 1, 1])


def compute_spec(eeg: np.ndarray) -> np.ndarray:
    spectrogram = librosa.stft(
        eeg,
        n_fft=1024,
        hop_length=39,
        win_length=256,
        window="hann",
        center=True,
        pad_mode="constant",
        out=None,
        dtype=None,
    )
    spectrogram = librosa.power_to_db(np.abs(spectrogram) ** 2, ref=np.max).astype(
        np.float32
    )
    spectrogram = (spectrogram + 80) / 80
    spectrogram = spectrogram**2
    return spectrogram[:256][::2, ::2]


def spec(eeg: np.ndarray) -> np.ndarray:
    s = compute_spec(eeg)
    eeg2 = convolve(eeg, KERNEL)
    s = s + compute_spec(eeg2)
    return s / 2


def compute_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    F3 = df_eeg["F3"].to_numpy()
    F4 = df_eeg["F4"].to_numpy()
    F7 = df_eeg["F7"].to_numpy()
    F8 = df_eeg["F8"].to_numpy()
    C3 = df_eeg["C3"].to_numpy()
    C4 = df_eeg["C4"].to_numpy()
    P3 = df_eeg["P3"].to_numpy()
    P4 = df_eeg["P4"].to_numpy()
    T3 = df_eeg["T3"].to_numpy()
    T4 = df_eeg["T4"].to_numpy()
    T5 = df_eeg["T5"].to_numpy()
    T6 = df_eeg["T6"].to_numpy()
    O1 = df_eeg["O1"].to_numpy()
    O2 = df_eeg["O2"].to_numpy()

    ll = np.stack([spec(Fp1 - F7), spec(F7 - T3), spec(T3 - T5), spec(T5 - O1)])
    lp = np.stack([spec(Fp1 - F3), spec(F3 - C3), spec(C3 - P3), spec(P3 - O1)])
    rp = np.stack([spec(Fp2 - F4), spec(F4 - C4), spec(C4 - P4), spec(P4 - O2)])
    rl = np.stack([spec(Fp2 - F8), spec(F8 - T4), spec(T4 - T6), spec(T6 - O2)])

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_chain(df_eeg)
    return chain




## === cell 6
import numpy as np
import math

from typing import Union, Tuple, List


def bin_array(
    array,
    bin_size,
    axis=-1,
    pad_dir="symmetric",
    mode="edge",
    return_padding=False,
    **padding_kwargs,
) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    if axis == -1:
        axis = array.ndim - 1

    curr_len = array.shape[axis]
    n_bins = math.ceil(curr_len / bin_size)
    new_len = n_bins * bin_size

    new_shape = list(array.shape)
    new_shape[axis] = n_bins
    new_shape.insert(axis + 1, bin_size)

    padding = [(0, 0)] * array.ndim
    if curr_len != new_len:
        if pad_dir == "left":
            pad_l = new_len - curr_len
            pad_r = 0
        elif pad_dir == "right":
            pad_l = 0
            pad_r = new_len - curr_len
        else:
            pad_l = (new_len - curr_len) // 2
            pad_r = (new_len - curr_len) - pad_l

        padding[axis] = (pad_l, pad_r)
        array = np.pad(array, padding, mode=mode, **padding_kwargs)

    array = array.reshape(new_shape)

    if return_padding:
        return array, padding

    return array




## === cell 7
from scipy.signal import butter, filtfilt


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    b, a = butter(N=order, Wn=cutoff_freq / (0.5 * fs), btype=btype, analog=False)
    return filtfilt(b, a, eeg_data)


def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg, cutoff_freq=np.array([0.25, 50]), btype="bandpass")
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)
    return eeg


def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    F3 = df_eeg["F3"].to_numpy()
    F4 = df_eeg["F4"].to_numpy()
    F7 = df_eeg["F7"].to_numpy()
    F8 = df_eeg["F8"].to_numpy()
    C3 = df_eeg["C3"].to_numpy()
    C4 = df_eeg["C4"].to_numpy()
    P3 = df_eeg["P3"].to_numpy()
    P4 = df_eeg["P4"].to_numpy()
    T3 = df_eeg["T3"].to_numpy()
    T4 = df_eeg["T4"].to_numpy()
    T5 = df_eeg["T5"].to_numpy()
    T6 = df_eeg["T6"].to_numpy()
    O1 = df_eeg["O1"].to_numpy()
    O2 = df_eeg["O2"].to_numpy()

    ll = np.stack(
        [
            compute_eeg(Fp1 - F7),
            compute_eeg(F7 - T3),
            compute_eeg(T3 - T5),
            compute_eeg(T5 - O1),
        ]
    )
    lp = np.stack(
        [
            compute_eeg(Fp1 - F3),
            compute_eeg(F3 - C3),
            compute_eeg(C3 - P3),
            compute_eeg(P3 - O1),
        ]
    )
    rp = np.stack(
        [
            compute_eeg(Fp2 - F4),
            compute_eeg(F4 - C4),
            compute_eeg(C4 - P4),
            compute_eeg(P4 - O2),
        ]
    )
    rl = np.stack(
        [
            compute_eeg(Fp2 - F8),
            compute_eeg(F8 - T4),
            compute_eeg(T4 - T6),
            compute_eeg(T6 - O2),
        ]
    )

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_eeg_chain(df_eeg)
    return chain




## === cell 8
np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})

EXPECTED_T = 2500


def _fix_length_last_axis(x: np.ndarray, target: int) -> np.ndarray:
    t = x.shape[-1]
    if t == target:
        return x
    if t > target:
        start = (t - target) // 2
        return x[..., start : start + target]
    pad = target - t
    left = pad // 2
    right = pad - left
    return np.pad(x, [(0, 0)] * (x.ndim - 1) + [(left, right)], mode="edge")


def proc_0(x):
    x = x.copy()
    x = _fix_length_last_axis(x, EXPECTED_T)
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (np.mean(x.std(axis=-1)) + 1e-5)
    x = x.reshape(4, 4, EXPECTED_T)
    return x


def proc_1(x):
    x = x.copy()
    x = _fix_length_last_axis(x, EXPECTED_T)
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (x.std(axis=-1, keepdims=True) + 1e-5)
    x = x.reshape(16, 1, EXPECTED_T)
    return x


SAFE_EPS = 7e-5


def _safe_probs(p: np.ndarray, eps: float = SAFE_EPS) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p[~np.isfinite(p)] = 0.0
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= 0:
        p = np.ones(6, dtype=np.float64) / 6.0
    else:
        p = p / s
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return p.astype(np.float32)


LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

_train_votes = df_train.select(LABELS).to_numpy()
_train_votes = np.nan_to_num(_train_votes, nan=0.0, posinf=0.0, neginf=0.0).astype(
    np.float64
)
_train_row_sums = _train_votes.sum(axis=1, keepdims=True)
_train_row_sums[_train_row_sums <= 0] = 1.0
_train_probs = _train_votes / _train_row_sums

alpha = 0.5  # uniform smoothing strength
GLOBAL_PRIOR = _safe_probs(
    (_train_probs.mean(axis=0) + alpha * (1.0 / 6.0)) / (1.0 + alpha),
    eps=SAFE_EPS,
)

train_patient_ids = df_train["patient_id"].to_numpy()
uniq_patients, invp = np.unique(train_patient_ids, return_inverse=True)
patient_sum = np.zeros((len(uniq_patients), 6), dtype=np.float64)
patient_cnt = np.zeros((len(uniq_patients), 1), dtype=np.float64)
np.add.at(patient_sum, invp, _train_probs)
np.add.at(patient_cnt, invp, 1.0)
patient_mean = patient_sum / np.clip(patient_cnt, 1.0, None)

PATIENT_PRIOR: Dict[int, np.ndarray] = {}
for pid, vec in zip(uniq_patients.tolist(), patient_mean):
    vec = (vec + alpha * (1.0 / 6.0)) / (1.0 + alpha)
    PATIENT_PRIOR[int(pid)] = _safe_probs(vec, eps=SAFE_EPS)

train_spec_ids = df_train["spectrogram_id"].to_numpy()
uniq_specs, invs = np.unique(train_spec_ids, return_inverse=True)
spec_sum = np.zeros((len(uniq_specs), 6), dtype=np.float64)
spec_cnt = np.zeros((len(uniq_specs), 1), dtype=np.float64)
np.add.at(spec_sum, invs, _train_probs)
np.add.at(spec_cnt, invs, 1.0)
spec_mean = spec_sum / np.clip(spec_cnt, 1.0, None)

SPEC_PRIOR: Dict[int, np.ndarray] = {}
for sid, vec in zip(uniq_specs.tolist(), spec_mean):
    vec = (vec + alpha * (1.0 / 6.0)) / (1.0 + alpha)
    SPEC_PRIOR[int(sid)] = _safe_probs(vec, eps=SAFE_EPS)

df_train_votes = df_train.select(["label_id", "eeg_id"] + LABELS)
df_label = df_train_votes.group_by("label_id").agg(
    [
        pl.col("eeg_id").first().alias("eeg_id"),
        *[pl.col(c).mean().alias(c) for c in LABELS],
    ]
)
_label_votes = df_label.select(LABELS).to_numpy()
_label_votes = np.nan_to_num(_label_votes, nan=0.0, posinf=0.0, neginf=0.0).astype(
    np.float64
)
_label_sums = _label_votes.sum(axis=1, keepdims=True)
_label_sums[_label_sums <= 0] = 1.0
_label_probs = _label_votes / _label_sums
_label_eeg_ids = df_label["eeg_id"].to_numpy().astype(np.int64)

uniq_eegs, inve = np.unique(_label_eeg_ids, return_inverse=True)
eeg_sum = np.zeros((len(uniq_eegs), 6), dtype=np.float64)
eeg_cnt = np.zeros((len(uniq_eegs), 1), dtype=np.float64)
np.add.at(eeg_sum, inve, _label_probs)
np.add.at(eeg_cnt, inve, 1.0)
eeg_mean = eeg_sum / np.clip(eeg_cnt, 1.0, None)

EEGID_PRIOR: Dict[int, np.ndarray] = {}
for eid, vec in zip(uniq_eegs.tolist(), eeg_mean):
    vec = (vec + alpha * (1.0 / 6.0)) / (1.0 + alpha)
    EEGID_PRIOR[int(eid)] = _safe_probs(vec, eps=SAFE_EPS)

W_SPEC = 0.48
W_PATIENT = 0.33
W_EEGID = 0.10
W_GLOBAL = 0.09

_EEG_CACHE: Dict[int, np.ndarray] = {}

_EEG_FEAT_CACHE: Dict[int, np.ndarray] = {}
_PROTOTYPES_READY = False
_CLASS_CENTROIDS = np.zeros((6, 10), dtype=np.float64)
_CENTROID_COUNTS = np.zeros(6, dtype=np.float64)
_CENTROID_TEMP = 1.0


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def _eeg_features_from_chain(x_chain: np.ndarray, fs_eff: float = 50.0) -> np.ndarray:
    x = np.asarray(x_chain, dtype=np.float64)
    x[~np.isfinite(x)] = 0.0
    x = _fix_length_last_axis(x, EXPECTED_T)

    x = x - x.mean(axis=-1, keepdims=True)

    rfft = np.fft.rfft(x, axis=-1)
    psd = (np.abs(rfft) ** 2) + 1e-12
    freqs = np.fft.rfftfreq(x.shape[-1], d=1.0 / fs_eff)

    def band_power(f_lo, f_hi):
        m = (freqs >= f_lo) & (freqs < f_hi)
        if not np.any(m):
            return np.zeros((x.shape[0],), dtype=np.float64)
        return psd[:, m].mean(axis=-1)

    p_delta = band_power(0.5, 4.0)
    p_theta = band_power(4.0, 8.0)
    p_alpha = band_power(8.0, 13.0)
    p_beta = band_power(13.0, 30.0)

    p_total = p_delta + p_theta + p_alpha + p_beta + 1e-12
    r_delta = p_delta / p_total
    r_theta = p_theta / p_total
    r_alpha = p_alpha / p_total
    r_beta = p_beta / p_total

    rms = np.sqrt(np.mean(x**2, axis=-1) + 1e-12)
    abs_mean = np.mean(np.abs(x), axis=-1)
    abs_p95 = np.percentile(np.abs(x), 95, axis=-1)

    def agg(v):
        return np.array([v.mean(), v.std()], dtype=np.float64)

    feat = np.concatenate(
        [
            agg(r_delta),
            agg(r_theta),
            agg(r_alpha),
            agg(r_beta),
            agg(rms),
            agg(abs_mean),
            agg(abs_p95),
        ],
        axis=0,
    )  # 14 dims

    idx = np.array([0, 1, 2, 3, 4, 5, 8, 9, 12, 13], dtype=int)
    feat10 = feat[idx]
    feat10 = (feat10 - feat10.mean()) / (feat10.std() + 1e-6)
    return feat10.astype(np.float64)


def _prepare_centroids_if_needed(max_per_class: int = 25, seed: int = 123) -> None:
    global _PROTOTYPES_READY, _CLASS_CENTROIDS, _CENTROID_COUNTS, _CENTROID_TEMP
    if _PROTOTYPES_READY:
        return

    if not os.path.isdir(TRAIN_EEG_DIR):
        _PROTOTYPES_READY = True
        return

    rng = np.random.default_rng(seed)

    y_idx = np.argmax(_train_probs, axis=1).astype(int)

    chosen_rows = []
    for c in range(6):
        idxs = np.where(y_idx == c)[0]
        if len(idxs) == 0:
            continue
        take = min(max_per_class, len(idxs))
        sel = rng.choice(idxs, size=take, replace=False)
        chosen_rows.append(sel)
    if len(chosen_rows) == 0:
        _PROTOTYPES_READY = True
        return

    chosen_rows = np.concatenate(chosen_rows)

    eeg_ids_train_np = df_train["eeg_id"].to_numpy()

    for ridx in tqdm(
        chosen_rows.tolist(), desc="Building fallback EEG centroids", leave=False
    ):
        eeg_id = int(eeg_ids_train_np[ridx])
        w = _train_probs[ridx].astype(np.float64)
        if np.sum(w) <= 0:
            continue

        if eeg_id in _EEG_CACHE:
            x_chain = _EEG_CACHE[eeg_id]
        else:
            fpath = os.path.join(TRAIN_EEG_DIR, f"{eeg_id}.parquet")
            if not os.path.isfile(fpath):
                continue
            try:
                x_chain = compute_eeg_from_file(fpath)
            except Exception:
                continue
            x_chain = np.asarray(x_chain)
            x_chain[~np.isfinite(x_chain)] = 0.0
            _EEG_CACHE[eeg_id] = x_chain

        try:
            feat = _eeg_features_from_chain(x_chain)
        except Exception:
            continue

        for c in range(6):
            wc = float(w[c])
            if wc <= 0:
                continue
            _CLASS_CENTROIDS[c] += wc * feat
            _CENTROID_COUNTS[c] += wc

    for c in range(6):
        if _CENTROID_COUNTS[c] > 0:
            _CLASS_CENTROIDS[c] /= _CENTROID_COUNTS[c]

    valid = _CENTROID_COUNTS > 0
    if np.sum(valid) >= 2:
        C = _CLASS_CENTROIDS[valid]
        d = np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(axis=-1) + 1e-12)
        d = d[np.triu_indices(d.shape[0], k=1)]
        spread = float(np.median(d)) if d.size else 1.0
        _CENTROID_TEMP = max(0.75, spread)
    else:
        _CENTROID_TEMP = 1.0

    _PROTOTYPES_READY = True


def _eeg_centroid_probs(eeg_id: int, filepath: str) -> np.ndarray:
    _prepare_centroids_if_needed()

    if np.all(_CENTROID_COUNTS <= 0):
        return np.ones(6, dtype=np.float64) / 6.0

    if eeg_id in _EEG_CACHE:
        x_chain = _EEG_CACHE[eeg_id]
    else:
        x_chain = compute_eeg_from_file(filepath)
        x_chain = np.asarray(x_chain)
        x_chain[~np.isfinite(x_chain)] = 0.0
        _EEG_CACHE[eeg_id] = x_chain

    if eeg_id in _EEG_FEAT_CACHE:
        feat = _EEG_FEAT_CACHE[eeg_id]
    else:
        feat = _eeg_features_from_chain(x_chain)
        _EEG_FEAT_CACHE[eeg_id] = feat

    d2 = np.sum((_CLASS_CENTROIDS - feat[None, :]) ** 2, axis=1)
    logits = -d2 / (float(_CENTROID_TEMP) ** 2 + 1e-6)
    p = _softmax(logits)
    return p.astype(np.float64)


def _log_blend(
    probs: List[np.ndarray], weights: List[float], eps: float = SAFE_EPS
) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        w = np.ones_like(w) / len(w)
    else:
        w = w / w.sum()

    logs = 0.0
    for p, wi in zip(probs, w.tolist()):
        p = np.asarray(p, dtype=np.float64)
        p = np.clip(p, eps, 1.0)
        p = p / p.sum()
        logs = logs + wi * np.log(p)
    out = np.exp(logs)
    return _safe_probs(out, eps=eps)




## === cell 9
@torch.no_grad()
def gen_ensemble_pred(models_0, models_1, df_row: pl.DataFrame) -> np.ndarray:
    eeg_id = int(df_row["eeg_id"].item())
    pid = int(df_row["patient_id"].item())
    sid = int(df_row["spectrogram_id"].item())
    filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")

    if (len(models_0) + len(models_1)) == 0:
        p_spec = SPEC_PRIOR.get(sid, GLOBAL_PRIOR).astype(np.float64)
        p_patient = PATIENT_PRIOR.get(pid, GLOBAL_PRIOR).astype(np.float64)
        p_eegid = EEGID_PRIOR.get(eeg_id, GLOBAL_PRIOR).astype(np.float64)
        p_global = GLOBAL_PRIOR.astype(np.float64)

        try:
            if os.path.isfile(filepath):
                p_eeg = _eeg_centroid_probs(eeg_id, filepath)
            else:
                p_eeg = np.ones(6, dtype=np.float64) / 6.0
        except Exception:
            p_eeg = np.ones(6, dtype=np.float64) / 6.0

        p_meta = _log_blend(
            probs=[p_spec, p_patient, p_eegid, p_global],
            weights=[W_SPEC, W_PATIENT, W_EEGID, W_GLOBAL],
            eps=SAFE_EPS,
        )

        W_EEG = 0.18
        blended = _log_blend(
            probs=[p_eeg, p_meta, GLOBAL_PRIOR.astype(np.float64)],
            weights=[W_EEG, 1.0 - W_EEG, 0.10],
            eps=SAFE_EPS,
        )
        return blended

    if eeg_id in _EEG_CACHE:
        x = _EEG_CACHE[eeg_id]
    else:
        x = compute_eeg_from_file(filepath)
        x = np.asarray(x)
        x[np.isnan(x) | np.isinf(x)] = 0
        _EEG_CACHE[eeg_id] = x

    x0 = proc_0(x)
    x0 = torch.tensor(x0, dtype=torch.float32, device=device).unsqueeze(0)

    x1 = proc_1(x)
    x1 = torch.tensor(x1, dtype=torch.float32, device=device).unsqueeze(0)

    preds = []

    for model in models_0:
        model.eval()
        pred = model(x0).exp()
        preds.append(pred.detach().cpu().numpy().reshape(-1))

    for model in models_1:
        model.eval()
        pred = model(x1).exp()
        preds.append(pred.detach().cpu().numpy().reshape(-1))

    preds = np.mean(np.stack(preds, axis=0), axis=0)
    return _safe_probs(preds, eps=SAFE_EPS)




## === cell 10
preds_final = []
for i in tqdm(range(len(df_test))):
    df_row = df_test.slice(i, 1)
    pred = gen_ensemble_pred(models_0, models_1, df_row)
    preds_final.append(pred)

preds_final = np.asarray(preds_final, dtype=np.float32)
print("preds_final shape:", preds_final.shape)



## === cell 11
df_sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].to_list()})
df_sub[LABELS] = preds_final

vals = df_sub[LABELS].to_numpy(dtype=np.float64)
vals = np.clip(vals, SAFE_EPS, None)
vals = vals / vals.sum(axis=1, keepdims=True)
df_sub[LABELS] = vals.astype(np.float32)

sub_cols = sample_submission.columns
df_sub = df_sub[[c for c in sub_cols]]

df_sub.to_csv("submission.csv", index=False)
df_sub.head()
