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

0.3545480479570443

# 6. Current score

0.86158

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I make the notebook runnable by removing the hard dependency on the missing `/kaggle/input/hms-models/` modules and weights, and provide a safe fallback inference path that still outputs valid probabilities. I fix the shape bug in `compute_eeg_chain` (EKG was incorrectly taken from `O2` and channel stacking was wrong) and make `MAD` accept the `keepdims` argument used elsewhere. Finally, I ensure we always generate a correctly-formatted `submission.csv` with rows summing to 1; if no external models are available, we use the empirical class prior from `train.csv` as calibrated probabilities (score-improving vs uniform and fully metric-consistent).'
- What this solution (achieved 0.77826) has done: 'Your current score is much worse than the target (lower-is-better), and the main reason is that you’re effectively submitting the global class prior when external weights are missing, which is not competitive for this KL metric. I keep your existing feature extraction exactly as-is, but replace the fallback with a minimal, legitimate model-free predictor: a patient-conditioned prior computed from `train.csv`, which is often substantially closer to the true label distribution than the global prior. For patients not seen in training, we smoothly back off to the global prior (Dirichlet/Laplace-style smoothing) to avoid overconfident spikes that can hurt KL. This keeps the same inference semantics (probabilities that sum to 1) and only changes the fallback path, which is directly relevant to improving score toward your target.'
- What this solution (achieved 0.77826) has done: 'Your current submission falls back to a smoothed patient-conditioned prior, which is still too coarse for this KL metric; we can legitimately use more metadata signal without changing your core EEG model/inference path. I keep the external-model path identical, and only strengthen the fallback by switching from patient-only to a patient+eeg_id hierarchical prior computed from train vote counts (with the same Dirichlet/Laplace-style smoothing to stay calibrated). For test rows whose `eeg_id` exists in train, this gives a much closer target distribution; otherwise we back off to patient prior, then global prior. This is a minimal change directly aimed at reducing KL (lower is better) while preserving submission semantics (valid probabilities summing to 1).'
- What this solution (achieved 0.77826) has done: 'Your current score (0.77826, lower-is-better) is still far above the target (0.35455), and since external EEG models are missing, the only lever we can use (without changing your core EEG pipeline) is improving the fallback probability estimator. I keep the entire EEG feature code and the external-model inference path unchanged, but strengthen the fallback by using a hierarchical prior that also conditions on `spectrogram_id` (in addition to `eeg_id` and `patient_id`), because `spectrogram_id` often carries strong shared-label signal in this dataset. I also compute these priors on *unique label groups* (`label_id`) to reduce duplicate/overlap bias in vote accumulation, which should make the fallback probabilities better calibrated for KL. Finally, I keep the same smoothing/backoff behavior and ensure probabilities remain strictly valid (positive and sum to 1) for submission.'
- What this solution (achieved 0.86429) has done: 'Your current score (0.77826, lower-is-better) is still far from the target (0.35455), and since the external models are missing the only safe lever is improving the fallback probabilities without touching your EEG pipeline. I keep your entire feature code and external-model inference path identical, but make the fallback a *mixture-of-priors* (eeg_id / spectrogram_id / patient_id / global) instead of a hard backoff, which tends to reduce KL by avoiding overconfident conditional priors when they’re noisy. I compute all priors on unique `label_id` as you already do, and add a tiny “temperature” smoothing (probability sharpening/flattening) tuned conservatively to reduce extreme probabilities that can hurt KL. Finally, I keep strict probability validity (positive and sum-to-1) and still write `submission.csv`.'
- What this solution (achieved 1.38156) has done: 'Your current score (0.86429, lower-is-better) is far above the target (0.35455), and since external model weights are missing, the only safe lever is improving the fallback probabilities while keeping your EEG pipeline untouched. The main issue with the current fallback is that it mixes fixed weights regardless of how reliable each conditional prior is, which can overfit noisy IDs and worsen KL. I keep the same hierarchical priors (eeg/spec/patient/global) and same smoothing, but change the mixture weights to be data-adaptive based on the amount of vote mass available for each key (more votes ⇒ more trust), and I remove the extra temperature step (which currently makes probabilities sharper and can increase KL when wrong). These are minimal, directly metric-relevant changes and still guarantee valid probabilities summing to 1 and a proper `submission.csv`.'
- What this solution (achieved 0.83731) has done: 'Your current score is far above the target (lower-is-better), and since external model weights are unavailable, the only safe way to improve is to make the fallback probabilities better match the expected per-sample vote distributions. I keep your entire EEG/spectrogram feature code and the external-model inference path unchanged, but I adjust the fallback mixer so its weights are on a consistent scale: right now the global prior weight is not comparable (it’s a raw constant while others are in [0,1]), which can dominate and hurt KL. I replace the global weight with the same “reliability” form as the others (tot/(tot+gamma)), and add a tiny uniform “floor” epsilon before normalization to reduce KL blow-ups on rare classes while preserving valid probabilities summing to 1. These are minimal, directly metric-relevant changes and should move the score down toward your target band.'
- What this solution (achieved 0.84846) has done: 'Your current score is much worse than the target (lower-is-better), and since external weights aren’t available, the only safe lever is improving the fallback probabilities while keeping your EEG pipeline untouched. I keep the same hierarchical priors you already compute (eeg/spec/patient/global on unique `label_id`), but fix the mixer to be more metric-consistent for KL by (1) using reliability weights that increase with evidence and (2) adding a small Dirichlet-style “pseudo-count” directly to the *mixed* distribution (not a uniform blend) to avoid near-zero probabilities that can explode KL. These are minimal changes localized to `mixed_fallback`, preserve valid probabilities summing to 1, and should move the score downward toward your target band.'
- What this solution (achieved 0.82894) has done: 'Your current score (0.84846, lower-is-better) is far above the target (0.35455), and with no external weights available the only legitimate lever is improving the fallback probabilities without touching your EEG model path. I keep your hierarchical prior idea and smoothing, but fix a key calibration issue: your mixer currently uses evidence-weighting while your per-key priors also already include strong smoothing (ALPHA_*), which effectively double-smooths and can wash out informative eeg/spec signals. The minimal change is to compute mixture weights using the same ALPHA_* “equivalent sample size” so the mixer trusts a key in proportion to its actual label evidence, and to slightly reduce the added Dirichlet mass (which can over-flatten and hurt KL). This keeps the rest of your pipeline identical and still guarantees strictly positive probabilities that sum to 1.'
- What this solution (achieved 0.84181) has done: 'Your current score is far above the target (lower is better), and because external model weights aren’t available the only legitimate lever is the fallback probability estimator. I keep your entire EEG pipeline untouched and only adjust the fallback mixer to be more KL-consistent by making the global prior weight scale compatible with the evidence-based weights (so it doesn’t dominate or vanish unpredictably). I also replace the fixed global-weight term with a proper reliability weight computed from global “equivalent sample size”, and reduce the extra Dirichlet mass slightly to avoid over-flattening informative conditional priors. These are minimal, localized changes that preserve valid probabilities (strictly positive, sum to 1) and should move the score downward toward the target.'
- What this solution (achieved 0.84181) has done: 'Your current score (0.84181, lower-is-better) is still far above the target (0.35455), and since external weights are unavailable the only legitimate lever is improving the fallback probability estimator while keeping the EEG model path untouched. I make one minimal but high-impact calibration change: compute the fallback priors on a **patient-grouped** basis (patient_id + eeg_id / spectrogram_id) instead of globally mixing independent priors, which better matches how labels cluster by patient and reduces KL. To keep it stable and not overconfident, I keep your same Dirichlet-style smoothing and reliability weighting, but apply it to these patient-conditional priors (and back off to global when patient is unseen). The rest of your pipeline (feature code, external-model inference, submission formatting) stays identical.'
- What this solution (achieved 0.86158) has done: 'Your current score (0.84181, lower-is-better) is still far above the target (0.35455), and since external weights are missing, the only safe lever is improving the fallback probabilities without changing the EEG model path. I make a minimal, metric-consistent adjustment to the fallback: add an explicit **unconditional spectrogram prior** and **unconditional EEG prior** into the mixture alongside the existing patient-conditioned priors, because test `eeg_id`/`spectrogram_id` can be unseen for a given patient but still informative globally. I keep your smoothing and reliability-weighting idea, but compute weights for each prior from its own evidence (vote mass) and then allocate the remaining weight to the global prior so weights are always comparable and stable. This stays fully legitimate (train-only aggregation), preserves probability semantics (strictly positive, sums to 1), and keeps the external-model inference unchanged.'

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

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

df_test.head()



## === cell 2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




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
import glob
import sys

HMS_MODELS_DIR = "/kaggle/input/hms-models/"
HAS_EXTERNAL_MODELS = os.path.isdir(HMS_MODELS_DIR)

EegModel4 = None
models_4: List[nn.Module] = []

if HAS_EXTERNAL_MODELS:
    try:
        sys.path.append(HMS_MODELS_DIR)
        from eeg_cnn_rnn_att import EegModel as EegModel4  # type: ignore

        for model_path in glob.glob(
            "/kaggle/input/hms-models/kaggle_send/kaggle_send/*"
        ):
            model = EegModel4()
            print("Loading:", model_path)
            model = load_model(model_path, model)
            model = model.to(device)
            model.eval()
            models_4.append(model)
    except Exception as e:
        print(
            "Warning: Could not import/load external models; using fallback predictions."
        )
        print("Reason:", repr(e))
        models_4 = []
else:
    print("Warning: /kaggle/input/hms-models/ not found; using fallback predictions.")

print("Ensemble size:", len(models_4))




## === cell 5
def MAD(signal, axis=-1, keepdims=True):
    """Compute the robust standard deviation (MAD) of a signal."""
    median = np.median(signal, axis=axis, keepdims=keepdims)
    absolute_deviations = np.abs(signal - median)
    median_absolute_deviation = np.median(
        absolute_deviations, axis=axis, keepdims=keepdims
    )
    scale_factor = 1.4826  # constant for normal distribution
    robust_std = median_absolute_deviation * scale_factor
    return robust_std


from scipy.signal import welch
from scipy.stats import linregress
from scipy.signal import butter, filtfilt


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    b, a = butter(N=order, Wn=cutoff_freq / (0.5 * fs), btype=btype, analog=False)
    return filtfilt(b, a, eeg_data)




## === cell 6
import scipy

STRIDE = 100
WINDOW_SIZE = 500
NOVERLAP = WINDOW_SIZE - STRIDE
PARAMS = dict(
    fs=200,
    window=("tukey", 0.25),
    nperseg=WINDOW_SIZE,
    noverlap=NOVERLAP,
    nfft=1024,
    detrend="constant",
    return_onesided=True,
    scaling="density",
    axis=-1,
    mode="psd",
)


def compute_spec(eeg: np.ndarray) -> np.ndarray:
    freqs, _, Sxx = scipy.signal.spectrogram(eeg, **PARAMS)
    valid_freq = (freqs >= 0.5) & (freqs <= 20)
    return Sxx[valid_freq, :]


def compute_spec_eeg(a, b) -> np.ndarray:
    eeg = a - b
    eeg = butter_filter(
        eeg, cutoff_freq=np.array([0.25, 40.0]), order=5, btype="bandpass"
    )
    return eeg


def compute_spec_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    Fz = df_eeg["Fz"].to_numpy()
    Cz = df_eeg["Cz"].to_numpy()
    Pz = df_eeg["Pz"].to_numpy()
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
            (
                compute_spec_eeg(Fp1, F7),
                compute_spec_eeg(F7, T3),
                compute_spec_eeg(T3, T5),
                compute_spec_eeg(T5, O1),
            )
        ]
    )
    lp = np.stack(
        [
            (
                compute_spec_eeg(Fp1, F3),
                compute_spec_eeg(F3, C3),
                compute_spec_eeg(C3, P3),
                compute_spec_eeg(P3, O1),
            )
        ]
    )
    rp = np.stack(
        [
            (
                compute_spec_eeg(Fp2, F4),
                compute_spec_eeg(F4, C4),
                compute_spec_eeg(C4, P4),
                compute_spec_eeg(P4, O2),
            )
        ]
    )
    rl = np.stack(
        [
            (
                compute_spec_eeg(Fp2, F8),
                compute_spec_eeg(F8, T4),
                compute_spec_eeg(T4, T6),
                compute_spec_eeg(T6, O2),
            )
        ]
    )

    chain = np.stack([ll, lp, rp, rl])[:, 0]

    mads = MAD(chain, axis=-1, keepdims=True)
    mads = np.median(mads.reshape(-1))
    chain = chain / (mads + 1e-5)

    outer = []
    for i in range(4):
        inner = []
        for j in range(4):
            inner.append(compute_spec(chain[i, j]))
        outer.append(inner)

    chain = np.array(outer)
    chain = np.log(chain.clip(np.exp(-4), np.exp(8)))
    chain = chain.mean(axis=1, keepdims=True)

    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_spec_chain(df_eeg)
    return chain




## === cell 7
import math
from typing import Union


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




## === cell 8
def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg, cutoff_freq=np.array([0.25, 50]), btype="bandpass")
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)
    return eeg


def compute_eeg_chain(
    df_eeg: pl.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Fp1 = df_eeg["Fp1"].to_numpy()
    Fp2 = df_eeg["Fp2"].to_numpy()
    Fz = df_eeg["Fz"].to_numpy()
    Cz = df_eeg["Cz"].to_numpy()
    Pz = df_eeg["Pz"].to_numpy()
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

    if "EKG" in df_eeg.columns:
        ekg = df_eeg["EKG"].to_numpy()
    else:
        ekg = np.zeros_like(O2)

    ekg = butter_filter(ekg, cutoff_freq=np.array([0.50, 20.0]), btype="bandpass")
    ekg = bin_array(ekg, bin_size=4, mode="reflect").mean(axis=-1)
    ekg = ekg.reshape(1, -1)

    ll = np.stack(
        [
            compute_eeg(Fp1 - F7),
            compute_eeg(F7 - T3),
            compute_eeg(T3 - T5),
            compute_eeg(T5 - O1),
        ],
        axis=0,
    )
    lp = np.stack(
        [
            compute_eeg(Fp1 - F3),
            compute_eeg(F3 - C3),
            compute_eeg(C3 - P3),
            compute_eeg(P3 - O1),
        ],
        axis=0,
    )
    rp = np.stack(
        [
            compute_eeg(Fp2 - F4),
            compute_eeg(F4 - C4),
            compute_eeg(C4 - P4),
            compute_eeg(P4 - O2),
        ],
        axis=0,
    )
    rl = np.stack(
        [
            compute_eeg(Fp2 - F8),
            compute_eeg(F8 - T4),
            compute_eeg(T4 - T6),
            compute_eeg(T6 - O2),
        ],
        axis=0,
    )

    mid = np.stack([compute_eeg(Fz - Cz), compute_eeg(Cz - Pz)], axis=0)  # (2, T)

    chain = np.stack([ll, lp, rp, rl], axis=0)  # (4, 4, T)

    return chain, mid, ekg


def compute_eeg_from_file(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    return compute_eeg_chain(df_eeg)




## === cell 9
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 10
np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})


def proc_0(x):
    x = x.copy()
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (np.mean(x.std(axis=-1)) + 1e-5)
    x = x.reshape(4, 4, 2_500)
    return x


def proc_1(x):
    x = x.copy()
    x = x - x.mean(axis=-1, keepdims=True)
    x = x / (x.std(axis=-1, keepdims=True) + 1e-5)
    x = x.reshape(16, 1, 2_500)
    return x


def proc_2(spec):
    spec = spec - spec.mean(axis=(-1, -2), keepdims=True)
    spec = spec / (spec.std(axis=(-1, -2), keepdims=True) + 1e-5)
    return spec


def proc_3(eeg):
    eeg = eeg.copy()
    eeg[np.isnan(eeg) | np.isinf(eeg)] = 0
    eeg = eeg - eeg.mean(axis=-1, keepdims=True)
    mad_std = MAD(eeg, axis=-1, keepdims=True).reshape(-1)
    mad_std = np.median(mad_std) + 1e-5
    eeg = eeg / mad_std
    eeg = eeg.clip(-10, 10)
    eeg = eeg.reshape(4, 4, 2_500)
    return eeg


def proc_4(eeg, mid, ekg):
    eeg = eeg.copy()
    mid = mid.copy()
    ekg = ekg.copy()

    eeg[np.isnan(eeg) | np.isinf(eeg)] = 0
    ekg[np.isnan(ekg) | np.isinf(ekg)] = 0
    mid[np.isnan(mid) | np.isinf(mid)] = 0

    eeg = eeg - eeg.mean(axis=-1, keepdims=True)
    mid = mid - mid.mean(axis=-1, keepdims=True)

    mad_std = MAD(eeg, axis=-1, keepdims=True).reshape(-1)
    mad_std = np.median(mad_std) + 1e-5

    eeg = eeg / mad_std
    eeg = eeg.clip(-10, 10)

    mid = mid / mad_std
    mid = mid.clip(-10, 10)

    ekg = ekg / (MAD(ekg, axis=-1, keepdims=True).reshape(-1) + 1e-5)

    eeg = eeg.reshape(16, -1)
    eeg = np.concatenate([eeg, mid, ekg], axis=0)
    eeg = eeg.reshape(19, 2_500)

    return eeg


def flip_h(eeg):
    eeg = eeg.copy()
    eeg = eeg[::-1]
    return eeg.copy()


def flip_v(eeg):
    eeg = eeg.copy()
    eeg = eeg[:, ::-1]
    return eeg.copy()




## === cell 11
train_unique = df_train.unique(subset=["label_id"], keep="first")

train_votes = train_unique.select(LABELS).to_numpy()
train_votes = np.clip(train_votes, 0, None)
train_probs = train_votes / (train_votes.sum(axis=1, keepdims=True) + 1e-12)
PRIOR = train_probs.mean(axis=0)
PRIOR = PRIOR / PRIOR.sum()

train_patient_ids = train_unique["patient_id"].to_numpy()
train_eeg_ids = train_unique["eeg_id"].to_numpy()
train_spec_ids = train_unique["spectrogram_id"].to_numpy()
train_votes_mat = train_votes.astype(np.float64)

patient_vote_sums: Dict[Any, np.ndarray] = {}
patient_total_sums: Dict[Any, float] = {}

eeg_vote_sums: Dict[Any, np.ndarray] = {}
eeg_total_sums: Dict[Any, float] = {}

spec_vote_sums: Dict[Any, np.ndarray] = {}
spec_total_sums: Dict[Any, float] = {}

pe_vote_sums: Dict[Tuple[Any, Any], np.ndarray] = {}
pe_total_sums: Dict[Tuple[Any, Any], float] = {}
ps_vote_sums: Dict[Tuple[Any, Any], np.ndarray] = {}
ps_total_sums: Dict[Tuple[Any, Any], float] = {}


def _get_key(x):
    return int(x) if isinstance(x, (int, np.integer)) or str(x).isdigit() else x


for pid, eid, sid, v in zip(
    train_patient_ids, train_eeg_ids, train_spec_ids, train_votes_mat
):
    pkey = _get_key(pid)
    ekey = _get_key(eid)
    skey = _get_key(sid)

    if pkey not in patient_vote_sums:
        patient_vote_sums[pkey] = v.copy()
        patient_total_sums[pkey] = float(v.sum())
    else:
        patient_vote_sums[pkey] += v
        patient_total_sums[pkey] += float(v.sum())

    if ekey not in eeg_vote_sums:
        eeg_vote_sums[ekey] = v.copy()
        eeg_total_sums[ekey] = float(v.sum())
    else:
        eeg_vote_sums[ekey] += v
        eeg_total_sums[ekey] += float(v.sum())

    if skey not in spec_vote_sums:
        spec_vote_sums[skey] = v.copy()
        spec_total_sums[skey] = float(v.sum())
    else:
        spec_vote_sums[skey] += v
        spec_total_sums[skey] += float(v.sum())

    pek = (pkey, ekey)
    psk = (pkey, skey)

    if pek not in pe_vote_sums:
        pe_vote_sums[pek] = v.copy()
        pe_total_sums[pek] = float(v.sum())
    else:
        pe_vote_sums[pek] += v
        pe_total_sums[pek] += float(v.sum())

    if psk not in ps_vote_sums:
        ps_vote_sums[psk] = v.copy()
        ps_total_sums[psk] = float(v.sum())
    else:
        ps_vote_sums[psk] += v
        ps_total_sums[psk] += float(v.sum())

ALPHA_PATIENT = 50.0
ALPHA_SPEC = 150.0
ALPHA_EEG = 200.0

FALLBACK_TEMPERATURE = 1.0

GAMMA_EEG = ALPHA_EEG
GAMMA_SPEC = ALPHA_SPEC
GAMMA_PAT = ALPHA_PATIENT
GAMMA_GLOBAL = 120.0

FALLBACK_DIRICHLET_MASS = 0.03  # pseudo-votes total mass added as (mass * PRIOR)


def _normalize_prob(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 1e-12, None)
    return p / p.sum()


def _apply_temperature(p: np.ndarray, t: float) -> np.ndarray:
    p = _normalize_prob(p)
    if t is None or abs(t - 1.0) < 1e-12:
        return p
    logp = np.log(np.clip(p, 1e-12, 1.0))
    logp = logp / float(t)
    p2 = np.exp(logp)
    return _normalize_prob(p2)


def patient_prior(patient_id) -> Optional[np.ndarray]:
    pkey = _get_key(patient_id)
    if pkey not in patient_vote_sums:
        return None
    s = patient_vote_sums[pkey]
    tot = patient_total_sums[pkey]
    p = (s + ALPHA_PATIENT * PRIOR) / (tot + ALPHA_PATIENT)
    return _normalize_prob(p)


def spec_prior(spec_id) -> Optional[np.ndarray]:
    skey = _get_key(spec_id)
    if skey not in spec_vote_sums:
        return None
    s = spec_vote_sums[skey]
    tot = spec_total_sums[skey]
    p = (s + ALPHA_SPEC * PRIOR) / (tot + ALPHA_SPEC)
    return _normalize_prob(p)


def eeg_prior(eeg_id) -> Optional[np.ndarray]:
    ekey = _get_key(eeg_id)
    if ekey not in eeg_vote_sums:
        return None
    s = eeg_vote_sums[ekey]
    tot = eeg_total_sums[ekey]
    p = (s + ALPHA_EEG * PRIOR) / (tot + ALPHA_EEG)
    return _normalize_prob(p)


def patient_eeg_prior(patient_id, eeg_id) -> Optional[np.ndarray]:
    pkey = _get_key(patient_id)
    ekey = _get_key(eeg_id)
    k = (pkey, ekey)
    if k not in pe_vote_sums:
        return None
    s = pe_vote_sums[k]
    tot = pe_total_sums[k]
    p = (s + ALPHA_EEG * PRIOR) / (tot + ALPHA_EEG)
    return _normalize_prob(p)


def patient_spec_prior(patient_id, spec_id) -> Optional[np.ndarray]:
    pkey = _get_key(patient_id)
    skey = _get_key(spec_id)
    k = (pkey, skey)
    if k not in ps_vote_sums:
        return None
    s = ps_vote_sums[k]
    tot = ps_total_sums[k]
    p = (s + ALPHA_SPEC * PRIOR) / (tot + ALPHA_SPEC)
    return _normalize_prob(p)


def mixed_fallback(df_row: pl.DataFrame) -> np.ndarray:
    eeg_id = df_row["eeg_id"].item()
    spec_id = df_row["spectrogram_id"].item()
    patient_id = df_row["patient_id"].item()

    pkey = _get_key(patient_id)
    ekey = _get_key(eeg_id)
    skey = _get_key(spec_id)

    pe = patient_eeg_prior(patient_id, eeg_id)
    ps = patient_spec_prior(patient_id, spec_id)
    pp = patient_prior(patient_id)
    ee = eeg_prior(eeg_id)
    ss = spec_prior(spec_id)

    tot_pe = float(pe_total_sums.get((pkey, ekey), 0.0))
    tot_ps = float(ps_total_sums.get((pkey, skey), 0.0))
    tot_p = float(patient_total_sums.get(pkey, 0.0))
    tot_e = float(eeg_total_sums.get(ekey, 0.0))
    tot_s = float(spec_total_sums.get(skey, 0.0))

    parts: List[np.ndarray] = []
    raw_w: List[float] = []

    def _relw(tot: float, gamma: float) -> float:
        return float(tot) / (float(tot) + float(gamma)) if tot > 0 else 0.0

    if pe is not None:
        parts.append(pe)
        raw_w.append(_relw(tot_pe, GAMMA_EEG))
    if ps is not None:
        parts.append(ps)
        raw_w.append(_relw(tot_ps, GAMMA_SPEC))
    if pp is not None:
        parts.append(pp)
        raw_w.append(_relw(tot_p, GAMMA_PAT))
    if ee is not None:
        parts.append(ee)
        raw_w.append(_relw(tot_e, GAMMA_EEG))
    if ss is not None:
        parts.append(ss)
        raw_w.append(_relw(tot_s, GAMMA_SPEC))

    parts.append(PRIOR.copy())
    sum_raw = float(np.sum(raw_w)) if len(raw_w) else 0.0
    raw_w.append(
        max(0.0, 1.0 - sum_raw)
        + (
            GAMMA_GLOBAL
            / (GAMMA_GLOBAL + tot_pe + tot_ps + tot_p + tot_e + tot_s + 1e-12)
        )
    )

    w = np.asarray(raw_w, dtype=np.float64)
    w = np.clip(w, 1e-12, None)
    w = w / w.sum()

    p = np.zeros_like(PRIOR, dtype=np.float64)
    for wi, pi in zip(w, parts):
        p += wi * pi

    p = _apply_temperature(p, FALLBACK_TEMPERATURE)

    if FALLBACK_DIRICHLET_MASS is not None and FALLBACK_DIRICHLET_MASS > 0:
        p = p + float(FALLBACK_DIRICHLET_MASS) * PRIOR

    return _normalize_prob(p)


@torch.no_grad()
def gen_ensemble_pred(df_row: pl.DataFrame) -> np.ndarray:
    eeg_id = df_row["eeg_id"].item()

    if len(models_4) == 0:
        return mixed_fallback(df_row)

    filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")

    eeg, mid, ekg = compute_eeg_from_file(filepath)
    eeg_4 = proc_4(eeg, mid, ekg)
    eeg_4 = torch.tensor(eeg_4, dtype=torch.float32, device=device).unsqueeze(0)

    preds = []
    for model in models_4:
        model.eval()
        out = model(eeg_4)
        p = out.exp().detach().cpu().numpy().reshape(-1)
        preds.append(p)

    preds = np.mean(preds, axis=0)
    preds = np.clip(preds, 1e-8, None)
    preds = preds / preds.sum()
    return preds




## === cell 12
from tqdm.auto import tqdm

preds_final = []
for i in tqdm(range(len(df_test))):
    pred = gen_ensemble_pred(df_test[i])
    preds_final.append(pred)

preds_final = np.asarray(preds_final, dtype=np.float64)

preds_final = np.clip(preds_final, 1e-8, None)
preds_final = preds_final / preds_final.sum(axis=1, keepdims=True)

print("preds_final shape:", preds_final.shape)
print(
    "row sums (min/max):", preds_final.sum(axis=1).min(), preds_final.sum(axis=1).max()
)



## === cell 13
df_sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].to_list()})
df_sub[LABELS] = preds_final
df_sub.to_csv("submission.csv", index=False)

assert df_sub.shape[0] == len(df_test)
assert list(df_sub.columns) == ["eeg_id"] + LABELS
row_sums = df_sub[LABELS].sum(axis=1).to_numpy()
assert np.allclose(row_sums, 1.0, atol=1e-6)

df_sub.head()
