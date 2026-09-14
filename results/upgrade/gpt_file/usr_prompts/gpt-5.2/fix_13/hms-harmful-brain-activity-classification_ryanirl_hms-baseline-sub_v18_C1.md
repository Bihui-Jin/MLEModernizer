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

0.3467497614795021

# 6. Current score

1.09952

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I first make the notebook robust to the missing `/kaggle/input/hms-models/` dependency by falling back to a simple, valid baseline prediction (the normalized mean label distribution from train) when the model package/files are not present, so a submission is always produced. Next, I fix the Polars row access bug (`df_test[i]` returns a Series) by iterating rows properly, and I remove per-sample debug printing that would otherwise explode runtime. Finally, I fix the EEG reshape crash by padding/cropping each computed EEG channel to the expected length (2500) before reshaping, keeping the same preprocessing semantics while ensuring inference runs end-to-end and outputs probabilities that sum to 1.'
- What this solution (achieved 1.40333) has done: 'Your current score (1.39779, lower-is-better) is far worse than the target (0.34675), so we should improve performance with minimal risk while keeping your overall approach intact. The biggest likely issue is that you are falling back to a constant prior because the external `/kaggle/input/hms-models/` dependency is not available, which produces weak predictions. I remove the dependency on external model files by switching the fallback to a stronger, still-simple baseline: train a tiny multinomial logistic regression on cheap per-EEG features computed from the provided `train_eegs/` and then predict for `test_eegs/`. This preserves evaluation semantics (probability outputs summing to 1) and keeps runtime within limits by sampling a capped number of training EEGs and using cached feature extraction.'
- What this solution (achieved 1.38475) has done: 'Your current score (1.40333, lower-is-better) is far from the target (0.34675), so we should improve predictive quality with minimal changes while keeping your existing feature-extraction + multinomial logistic regression fallback intact. The biggest issue is that the model is trained on hard argmax labels, while the metric is KL divergence against soft vote distributions; we switch to fitting on the soft targets using sample weights (one-vs-rest logistic regression per class) and then renormalize to probabilities. To further reduce KL without changing the core approach, we calibrate only the final probability smoothing toward the prior (`alpha`) using a small patient-grouped validation split from the same extracted features (no extra training loops; just choosing `alpha`). This keeps runtime reasonable (feature cache + capped training EEGs) and should move the score substantially toward the target.'
- What this solution (achieved 1.35873) has done: 'We keep your existing “EEG-chain → robust normalization → simple features → 6 one-vs-rest logistic regressions → prior blend” core logic unchanged, but fix the main source of poor KL: the models are trained on hard 0/1 labels with sample weights, which throws away the soft vote targets the metric evaluates. We switch the fallback training to a proper soft-label fit by solving a small ridge-regularized multinomial regression in closed form on the same extracted features (no new deep model, no training loop changes), then use softmax to get probabilities. We keep your existing prior blending and calibrate both the blend alpha and an optional temperature on the validation split to reduce KL without changing semantics. This should move the score substantially toward the target while staying fast enough (<600s) by reusing the same capped feature extraction and caching.'
- What this solution (achieved 1.36698) has done: 'Your current score (1.35873, lower-is-better) is still far from the target (0.34675), so we should improve predictive quality while keeping your existing “compute_eeg_chain → proc_3 → simple features → closed-form ridge → softmax → prior blend” core logic intact. The biggest low-risk gain is to train on more representative data: instead of randomly sampling EEGs, aggregate labels per `eeg_id` and then choose up to `max_train_eegs` in a stratified way based on the dominant class, which reduces sampling variance and should lower KL. Next, we standardize features using training-set mean/std and apply the same transform at inference; this is a calibration/conditioning change that usually improves ridge+softmax stability without changing the modeling approach. Finally, we add tiny Dirichlet-style smoothing to the training soft labels (and include it as a candidate during alpha/temp calibration) to reduce extreme targets that can inflate KL, while still preserving semantics.'
- What this solution (achieved 1.35965) has done: 'Your current score (1.36698, lower-is-better) is still far from the target (0.34675), so the safest path is to improve calibration/robustness without changing the core “EEG chain → proc_3 → simple stats features → closed-form ridge → softmax → prior blend” approach. I keep the same model/feature logic but reduce avoidable train/test mismatch by extracting train features at the same 50s window used in test (center-crop/pad to 10,000 samples before montage+downsample), and I add a small grid search over the ridge L2 strength during the existing patient-grouped validation calibration. Finally, I make the feature extraction more robust to missing EEG columns by filling absent leads with zeros (rare but can otherwise silently break feature consistency), which should reduce pathological predictions and improve KL.'
- What this solution (achieved 1.35947) has done: 'Your current score (1.35965, lower-is-better) is still far from the target (0.34675), so we should improve toward the target with minimal, low-risk calibration changes while keeping your exact core pipeline (same EEG chain, same features, same closed-form ridge + softmax + prior blend). The biggest KL issue that remains is that we tune hyperparameters on a single small split; we make that calibration more stable by using multiple patient-grouped folds and choosing the average-best (y_eps, l2, temp, alpha) without changing the model form. We also compute and report out-of-fold KL during calibration to avoid picking a lucky split, which should move the leaderboard score down reliably. Finally, we keep submission formatting identical and continue guaranteeing per-row probabilities sum to 1.'
- What this solution (achieved 1.34806) has done: 'Your current score (1.35947, lower-is-better) is still far above the target (0.34675), so we should improve predictive quality without changing your core pipeline (same EEG chain, same proc_3, same feature set, same closed-form ridge + softmax + prior blend). The main likely issue is underfitting: the ridge model is purely linear on simple stats, so adding a small, safe nonlinearity via feature expansion (squares + a few interactions) can materially reduce KL while keeping the same training approach (still one closed-form solve). I also fix the calibration loop bug where the chosen `TEMP_SOFTMAX`/`ALPHA_BLEND` aren’t averaged across folds (it currently only selects `y_eps` and `l2` by fold-KL, then tunes `t/a` on one split), by selecting all hyperparameters by multi-fold average KL in one pass (no extra training loops beyond the existing grid). Finally, I keep submission formatting identical and continue enforcing per-row probability normalization.'
- What this solution (achieved 1.2886) has done: 'Your current score (1.34806, lower-is-better) is still far from the target (0.34675), so we should improve generalization with very small, low-risk changes while keeping your exact “EEG chain → proc_3 → simple stats features → feature expansion → closed-form ridge → softmax → prior blend” pipeline unchanged. The biggest likely issue is that the linear ridge-to-softmax model can’t capture basic nonlinearities even with squares/interactions, so we add a tiny, safe amount of frequency information by computing a few bandpower ratios from the same already-processed chains and appending them as extra features (no new model, still one closed-form solve). To better match KL on soft targets, we also add a very small logit-centering step (subtract per-sample mean logit before softmax) to stabilize probability spread without changing the training objective. Finally, we keep your multi-fold calibration, submission formatting, and per-row normalization exactly as required.'
- What this solution (achieved 1.1267) has done: 'Your current score (1.2886, lower-is-better) is still far from the target (0.34675), so we should improve predictive quality without changing your core pipeline (same EEG chain, same proc_3, same feature set/expansion, same closed-form ridge + softmax + prior blend). The biggest likely remaining issue is that the ridge is trained on raw probability targets even though KL scoring is log-sensitive; we can keep the same closed-form solver but fit in logit-space by regressing to `log(Y)` (with smoothing) and then softmax at inference, which better matches KL while preserving the same model form. To keep this minimal-risk, we add this as a calibration option (target transform type) inside the existing multi-fold grid search, then refit once on all training features using the chosen setting. Finally, we keep submission formatting identical and continue enforcing per-row probability normalization.'
- What this solution (achieved 1.14542) has done: 'We keep your exact fallback pipeline (EEG chain → proc_3 → feature stats/expansion → closed-form ridge → softmax → prior blend) but fix one key mismatch: you calibrate (y_eps/l2/temp/alpha/transform) with standardized features, yet you refit the final ridge on all data using the *whole-dataset* standardization computed before calibration (leaks validation distribution into training in each fold). We recompute standardization *within each fold* during calibration to make the chosen hyperparameters generalize better and reduce KL on the leaderboard, then refit once on all data as before. Additionally, we apply the `logprob` transform consistently at inference (right now it only affects training targets, but logits aren’t adjusted back; we keep the same model form and just treat logits as log-prob predictions by re-softmaxing them, which is already what you do, but we ensure temperature is applied after centering in a stable way). These are minimal, metric-aligned calibration fixes and should move the score down toward the target without changing architecture or adding training loops beyond the existing grid.'
- What this solution (achieved 1.09952) has done: 'Your current score (1.14542, lower-is-better) is still far above the target (0.34675), so the most “minimal but effective” improvement is to keep your exact fallback pipeline but reduce a major train/test mismatch: you train the fallback on `train_eegs/` (raw EEG) but the competition labels were generated using both EEG + matched spectrogram context; we can add a very small set of spectrogram-derived features (from `train_spectrograms/` and `test_spectrograms/`) and concatenate them to your existing EEG features, keeping the same closed-form ridge + softmax + prior blend. This does not change your core modeling approach (still one linear closed-form solve and the same calibration loop), but should materially reduce KL by giving the model information closer to what annotators saw. We also keep your patient-grouped multi-fold calibration and only extend feature extraction + caching to stay within the 600s budget. Submission formatting and per-row probability normalization remain unchanged.'

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
TRAIN_SPEC_DIR = os.path.join(DATA_DIR, "train_spectrograms/")

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
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    return model




## === cell 4
import sys

HAVE_HMS_MODELS = False
try:
    sys.path.append("/kaggle/input/hms-models/")
    from eeg_cnn_rnn_w1 import EegModel as EegModel0  # noqa: F401
    from eeg_cnn_rnn import EegModel as EegModel1  # noqa: F401
    from spc_cnn_rnn import SpectrogramModel  # noqa: F401
    from eeg_cnn_rnn_final import EegModel as EegModel3

    HAVE_HMS_MODELS = True
except Exception as e:
    print(
        "WARNING: Could not import external model package from /kaggle/input/hms-models/."
    )
    print(
        "Will fall back to an internal lightweight EEG+spectrogram-feature ridge baseline. Import error:",
        repr(e),
    )
    HAVE_HMS_MODELS = False



## === cell 5
import glob

models_3 = []
if HAVE_HMS_MODELS:
    for fold_path in glob.glob("/kaggle/input/hms-models/newest_10f/kaggle/*"):
        model_path = os.path.join(fold_path, "model_best_val_g10.pt")
        if not os.path.exists(model_path):
            continue
        model = EegModel3()
        print(model_path)
        model = load_model(model_path, model)
        model = model.to(device)
        model.eval()
        models_3.append(model)

print("Loaded models_3:", len(models_3))




## === cell 6
def MAD(signal, axis=-1, keepdims=True):
    """Compute the robust standard deviation (MAD) of a signal."""
    median = np.median(signal, axis=axis, keepdims=True)
    absolute_deviations = np.abs(signal - median)
    median_absolute_deviation = np.median(
        absolute_deviations, axis=axis, keepdims=keepdims
    )
    scale_factor = 1.4826  # constant for normal distribution
    robust_std = median_absolute_deviation * scale_factor
    robust_std = np.asarray(robust_std)
    return robust_std


from scipy.signal import butter, filtfilt


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    b, a = butter(N=order, Wn=cutoff_freq / (0.5 * fs), btype=btype, analog=False)
    return filtfilt(b, a, eeg_data)




## === cell 7
import scipy
import scipy.signal

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




## === cell 8
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




## === cell 9
EXPECTED_LEN = 2500

RAW_TARGET_LEN = 50 * 200  # 50 seconds at 200 Hz = 10000 samples


def _fix_len_1d(x: np.ndarray, target_len: int = EXPECTED_LEN) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        x = x.reshape(-1)
    n = x.shape[0]
    if n == target_len:
        return x
    if n > target_len:
        return x[:target_len]
    pad = target_len - n
    return np.pad(x, (0, pad), mode="edge")


def _center_crop_or_pad_1d(
    x: np.ndarray, target_len: int = RAW_TARGET_LEN
) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        x = x.reshape(-1)
    n = x.shape[0]
    if n == target_len:
        return x
    if n > target_len:
        start = (n - target_len) // 2
        return x[start : start + target_len]
    pad = target_len - n
    left = pad // 2
    right = pad - left
    return np.pad(x, (left, right), mode="edge")


def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg, cutoff_freq=np.array([0.25, 50]), btype="bandpass")
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)
    eeg = _fix_len_1d(eeg, EXPECTED_LEN)
    return eeg


EEG_LEADS = [
    "Fp1",
    "Fp2",
    "F3",
    "F4",
    "F7",
    "F8",
    "C3",
    "C4",
    "P3",
    "P4",
    "T3",
    "T4",
    "T5",
    "T6",
    "O1",
    "O2",
]


def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    cols = set(df_eeg.columns)
    data = {}
    for c in EEG_LEADS:
        if c in cols:
            v = df_eeg[c].to_numpy()
        else:
            v = np.zeros(df_eeg.height, dtype=np.float32)
        data[c] = _center_crop_or_pad_1d(v, RAW_TARGET_LEN)

    Fp1 = data["Fp1"]
    Fp2 = data["Fp2"]
    F3 = data["F3"]
    F4 = data["F4"]
    F7 = data["F7"]
    F8 = data["F8"]
    C3 = data["C3"]
    C4 = data["C4"]
    P3 = data["P3"]
    P4 = data["P4"]
    T3 = data["T3"]
    T4 = data["T4"]
    T5 = data["T5"]
    T6 = data["T6"]
    O1 = data["O1"]
    O2 = data["O2"]

    ll = np.stack(
        [
            (
                compute_eeg(Fp1 - F7),
                compute_eeg(F7 - T3),
                compute_eeg(T3 - T5),
                compute_eeg(T5 - O1),
            )
        ]
    )
    lp = np.stack(
        [
            (
                compute_eeg(Fp1 - F3),
                compute_eeg(F3 - C3),
                compute_eeg(C3 - P3),
                compute_eeg(P3 - O1),
            )
        ]
    )
    rp = np.stack(
        [
            (
                compute_eeg(Fp2 - F4),
                compute_eeg(F4 - C4),
                compute_eeg(C4 - P4),
                compute_eeg(P4 - O2),
            )
        ]
    )
    rl = np.stack(
        [
            (
                compute_eeg(Fp2 - F8),
                compute_eeg(F8 - T4),
                compute_eeg(T4 - T6),
                compute_eeg(T6 - O2),
            )
        ]
    )

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain


def compute_eeg_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_eeg_chain(df_eeg)
    return chain




## === cell 10
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]



## === cell 11
np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})


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


def flip_h(eeg):
    eeg = eeg.copy()
    eeg = eeg[::-1]
    return eeg.copy()


def flip_v(eeg):
    eeg = eeg.copy()
    eeg = eeg[:, ::-1]
    return eeg.copy()


train_votes = df_train.select(LABELS).to_pandas().values.astype(np.float64)
train_votes_sum = train_votes.sum(axis=1, keepdims=True)
train_probs = train_votes / np.clip(train_votes_sum, 1e-12, None)
PRIOR = train_probs.mean(axis=0)
PRIOR = PRIOR / PRIOR.sum()



## === cell 12
FEATURE_W: Optional[np.ndarray] = None  # (n_features+1, 6)
FEATURES_READY = False
ALPHA_BLEND: float = 0.10  # will be calibrated
TEMP_SOFTMAX: float = 1.0  # will be calibrated
Y_SMOOTH_EPS: float = 0.0  # will be calibrated
L2_CHOSEN: float = 3.0

TARGET_TRANSFORM: str = "prob"  # "prob" or "logprob"

FEATURE_MEAN: Optional[np.ndarray] = None
FEATURE_STD: Optional[np.ndarray] = None


def _safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, 0, None))


def _bandpower_ratios_from_chain(
    x_4x4xT: np.ndarray, fs_down: float = 50.0
) -> np.ndarray:
    x = x_4x4xT.astype(np.float64, copy=False).reshape(16, -1)
    x = x - x.mean(axis=1, keepdims=True)
    n = x.shape[1]
    Xf = np.fft.rfft(x, axis=1)
    psd = (np.abs(Xf) ** 2) / max(n, 1)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs_down)

    def bp(f1, f2):
        m = (freqs >= f1) & (freqs < f2)
        if not np.any(m):
            return np.zeros((x.shape[0],), dtype=np.float64)
        return psd[:, m].sum(axis=1)

    p_delta = bp(0.5, 4.0)
    p_theta = bp(4.0, 8.0)
    p_alpha = bp(8.0, 13.0)
    p_beta = bp(13.0, 25.0)
    p_total = p_delta + p_theta + p_alpha + p_beta + 1e-12

    feats = np.array(
        [
            (p_delta / p_total).mean(),
            (p_theta / p_total).mean(),
            (p_alpha / p_total).mean(),
            (p_beta / p_total).mean(),
            (p_delta / (p_alpha + 1e-12)).mean(),
            (p_delta / (p_beta + 1e-12)).mean(),
        ],
        dtype=np.float32,
    )
    feats[np.isnan(feats) | np.isinf(feats)] = 0.0
    return feats


def extract_features_from_chain(chain_4x4xT: np.ndarray) -> np.ndarray:
    x = chain_4x4xT.astype(np.float32, copy=False).reshape(16, -1)
    mean = x.mean(axis=1)
    std = x.std(axis=1)
    q05 = np.quantile(x, 0.05, axis=1)
    q50 = np.quantile(x, 0.50, axis=1)
    q95 = np.quantile(x, 0.95, axis=1)
    rms = np.sqrt((x * x).mean(axis=1) + 1e-12)
    g_mean = np.array([mean.mean()], dtype=np.float32)
    g_std = np.array([std.mean()], dtype=np.float32)
    g_rms = np.array([rms.mean()], dtype=np.float32)

    bp = _bandpower_ratios_from_chain(chain_4x4xT)

    feats = np.concatenate(
        [mean, std, q05, q50, q95, _safe_log1p(rms), g_mean, g_std, g_rms, bp]
    ).astype(np.float32)
    feats[np.isnan(feats) | np.isinf(feats)] = 0.0
    return feats


def extract_features_from_spectrogram_file(filepath: str) -> np.ndarray:
    df_sp = pl.read_parquet(filepath).fill_null(0)
    cols = [c for c in df_sp.columns if c != "time"]
    if len(cols) == 0 or df_sp.height == 0:
        return np.zeros((12,), dtype=np.float32)

    X = df_sp.select(cols).to_numpy().astype(np.float64, copy=False)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    X = np.log1p(np.clip(X, 0.0, None))

    g_mean = X.mean()
    g_std = X.std()
    g_q05 = np.quantile(X, 0.05)
    g_q50 = np.quantile(X, 0.50)
    g_q95 = np.quantile(X, 0.95)

    mt = X.mean(axis=1)  # (time,)
    mt_mean = mt.mean()
    mt_std = mt.std()
    mt_q05 = np.quantile(mt, 0.05)
    mt_q50 = np.quantile(mt, 0.50)
    mt_q95 = np.quantile(mt, 0.95)

    mf = X.mean(axis=0)  # (freq_bins*regions,)
    mf_mean = mf.mean()
    mf_std = mf.std()

    feats = np.array(
        [
            g_mean,
            g_std,
            g_q05,
            g_q50,
            g_q95,
            mt_mean,
            mt_std,
            mt_q05,
            mt_q50,
            mt_q95,
            mf_mean,
            mf_std,
        ],
        dtype=np.float32,
    )
    feats[np.isnan(feats) | np.isinf(feats)] = 0.0
    return feats


def _expand_features(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X2 = X * X
    if X.shape[1] >= 3:
        a = X[:, -3]
        b = X[:, -2]
        c = X[:, -1]
        inter = np.stack([a * b, a * c, b * c], axis=1)
    else:
        inter = np.zeros((X.shape[0], 3), dtype=np.float64)
    return np.concatenate([X, X2, inter], axis=1)


def _kl_divergence(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    eps = 1e-12
    y_true = np.clip(y_true, eps, 1.0)
    y_true = y_true / y_true.sum(axis=1, keepdims=True)
    y_pred = np.clip(y_pred, eps, 1.0)
    y_pred = y_pred / y_pred.sum(axis=1, keepdims=True)
    return float(np.mean(np.sum(y_true * np.log(y_true / y_pred), axis=1)))


def _softmax(z: np.ndarray, axis: int = 1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.clip(ez.sum(axis=axis, keepdims=True), 1e-12, None)


def _fit_ridge_multinomial_closed_form(
    X: np.ndarray, Y: np.ndarray, l2: float = 3.0
) -> np.ndarray:
    """
    Closed-form ridge regression to targets (either probs or log-probs), then softmax at inference.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n, d = X.shape
    Xb = np.concatenate([X, np.ones((n, 1), dtype=np.float64)], axis=1)  # bias term
    A = Xb.T @ Xb
    A.flat[:: A.shape[0] + 1] += float(l2)  # add l2 to diagonal
    W = np.linalg.solve(A, Xb.T @ Y)  # (d+1, 6)
    return W.astype(np.float64)


def _standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd < 1e-6, 1.0, sd)
    return mu.astype(np.float64), sd.astype(np.float64)


def _standardize_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (X - mu) / sd


def _prepare_feature_model(max_train_eegs: int = 3000, random_state: int = 0) -> None:
    global FEATURE_W, FEATURES_READY, ALPHA_BLEND, TEMP_SOFTMAX, Y_SMOOTH_EPS, FEATURE_MEAN, FEATURE_STD, L2_CHOSEN, TARGET_TRANSFORM

    if FEATURES_READY:
        return

    df_lab = (
        df_train.select(["eeg_id", "patient_id", "spectrogram_id"] + LABELS)
        .group_by(["eeg_id", "patient_id", "spectrogram_id"])
        .sum()
        .to_pandas()
    )
    y_votes = df_lab[LABELS].values.astype(np.float64)
    y_probs = y_votes / np.clip(y_votes.sum(axis=1, keepdims=True), 1e-12, None)
    eeg_ids = df_lab["eeg_id"].values.astype(np.int64)
    patient_ids = df_lab["patient_id"].values.astype(np.int64)
    spec_ids = df_lab["spectrogram_id"].values.astype(np.int64)

    rng = np.random.default_rng(random_state)
    n_total = len(eeg_ids)
    if n_total > max_train_eegs:
        dom = np.argmax(y_probs, axis=1)
        chosen = []
        per_class = max(1, max_train_eegs // 6)
        for c in range(6):
            idx_c = np.where(dom == c)[0]
            if idx_c.size == 0:
                continue
            take = min(per_class, idx_c.size)
            chosen.append(rng.choice(idx_c, size=take, replace=False))
        chosen = (
            np.unique(np.concatenate(chosen))
            if len(chosen)
            else np.array([], dtype=int)
        )
        if chosen.size < max_train_eegs:
            remaining = np.setdiff1d(np.arange(n_total), chosen, assume_unique=False)
            need = max_train_eegs - chosen.size
            if remaining.size > 0:
                extra = rng.choice(
                    remaining, size=min(need, remaining.size), replace=False
                )
                chosen = np.concatenate([chosen, extra])
        eeg_ids = eeg_ids[chosen]
        patient_ids = patient_ids[chosen]
        spec_ids = spec_ids[chosen]
        y_probs = y_probs[chosen]

    X_list = []
    Y_list = []
    P_list = []
    feat_cache_eeg: Dict[int, np.ndarray] = {}
    feat_cache_spc: Dict[int, np.ndarray] = {}

    for eeg_id, sp_id, pid, y in tqdm(
        list(zip(eeg_ids, spec_ids, patient_ids, y_probs)),
        desc="Extract train EEG+SPC features",
        total=len(eeg_ids),
    ):
        path_eeg = os.path.join(TRAIN_EEG_DIR, f"{int(eeg_id)}.parquet")
        if not os.path.exists(path_eeg):
            continue

        if int(eeg_id) in feat_cache_eeg:
            feats_eeg = feat_cache_eeg[int(eeg_id)]
        else:
            chain = compute_eeg_from_file(path_eeg)
            chain = proc_3(chain)
            feats_eeg = extract_features_from_chain(chain)
            feat_cache_eeg[int(eeg_id)] = feats_eeg

        path_sp = os.path.join(TRAIN_SPEC_DIR, f"{int(sp_id)}.parquet")
        if int(sp_id) in feat_cache_spc:
            feats_sp = feat_cache_spc[int(sp_id)]
        else:
            if os.path.exists(path_sp):
                feats_sp = extract_features_from_spectrogram_file(path_sp)
            else:
                feats_sp = np.zeros((12,), dtype=np.float32)
            feat_cache_spc[int(sp_id)] = feats_sp

        feats = np.concatenate([feats_eeg, feats_sp], axis=0).astype(
            np.float32, copy=False
        )

        X_list.append(feats)
        Y_list.append(y)
        P_list.append(int(pid))

    X0 = np.asarray(X_list, dtype=np.float32)
    Y = np.asarray(Y_list, dtype=np.float64)
    P = np.asarray(P_list, dtype=np.int64)

    X0 = X0.astype(np.float64)
    X = _expand_features(X0)

    uniq_p = np.unique(P)
    rng = np.random.default_rng(random_state)
    rng.shuffle(uniq_p)

    n_folds = 3
    folds = np.array_split(uniq_p, n_folds)

    eps_grid = np.array([0.0, 0.002, 0.005, 0.01, 0.02], dtype=np.float64)
    l2_grid = np.array([0.3, 1.0, 3.0, 10.0, 30.0], dtype=np.float64)
    temps = np.array([0.8, 1.0, 1.2, 1.5, 2.0], dtype=np.float64)
    alphas = np.array([0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30], dtype=np.float64)

    transforms = ("prob", "logprob")

    best = (1e18, "prob", 0.0, 3.0, 1.0, 0.10)

    for transform in transforms:
        for y_eps in eps_grid:
            Y_smooth = (Y + y_eps) / (1.0 + 6.0 * y_eps)

            if transform == "logprob":
                Y_train_t_full = np.log(np.clip(Y_smooth, 1e-12, None))
            else:
                Y_train_t_full = Y_smooth

            for l2 in l2_grid:
                fold_logits = []
                fold_Y_va = []
                for k in range(n_folds):
                    val_p = set(folds[k].tolist())
                    is_val = np.array([pid in val_p for pid in P], dtype=bool)
                    X_tr, Y_tr_t = X[~is_val], Y_train_t_full[~is_val]
                    X_va, Y_va = X[is_val], Y[is_val]
                    if X_va.shape[0] < 20 or X_tr.shape[0] < 50:
                        continue

                    mu_k, sd_k = _standardize_fit(X_tr.astype(np.float64))
                    X_tr_s = _standardize_apply(X_tr.astype(np.float64), mu_k, sd_k)
                    X_va_s = _standardize_apply(X_va.astype(np.float64), mu_k, sd_k)

                    Wk = _fit_ridge_multinomial_closed_form(
                        X_tr_s, Y_tr_t, l2=float(l2)
                    )
                    Xb_va = np.concatenate(
                        [X_va_s, np.ones((X_va_s.shape[0], 1), dtype=np.float64)],
                        axis=1,
                    )
                    logits_va = Xb_va @ Wk
                    fold_logits.append(logits_va)
                    fold_Y_va.append(Y_va)

                if len(fold_logits) == 0:
                    continue

                for t in temps:
                    for a in alphas:
                        kls = []
                        for logits_va, Y_va in zip(fold_logits, fold_Y_va):
                            z = logits_va
                            z = z - z.mean(axis=1, keepdims=True)
                            p0 = _softmax(z / float(t), axis=1)
                            pred = (1.0 - float(a)) * p0 + float(a) * PRIOR.reshape(
                                1, -1
                            )
                            pred = np.clip(pred, 1e-12, None)
                            pred = pred / pred.sum(axis=1, keepdims=True)
                            kls.append(_kl_divergence(Y_va, pred))
                        avg_kl = float(np.mean(kls))
                        if avg_kl < best[0]:
                            best = (
                                avg_kl,
                                transform,
                                float(y_eps),
                                float(l2),
                                float(t),
                                float(a),
                            )

    chosen_avg_kl, chosen_transform, chosen_y_eps, chosen_l2, chosen_t, chosen_a = best

    Y_SMOOTH_EPS = float(chosen_y_eps)
    TEMP_SOFTMAX = float(chosen_t)
    ALPHA_BLEND = float(chosen_a)
    L2_CHOSEN = float(chosen_l2)
    TARGET_TRANSFORM = str(chosen_transform)

    FEATURE_MEAN, FEATURE_STD = _standardize_fit(X.astype(np.float64))
    X_all_s = _standardize_apply(X.astype(np.float64), FEATURE_MEAN, FEATURE_STD)

    Y_smooth = (Y + Y_SMOOTH_EPS) / (1.0 + 6.0 * Y_SMOOTH_EPS)
    if TARGET_TRANSFORM == "logprob":
        Y_fit = np.log(np.clip(Y_smooth, 1e-12, None))
    else:
        Y_fit = Y_smooth

    FEATURE_W = _fit_ridge_multinomial_closed_form(X_all_s, Y_fit, l2=L2_CHOSEN)

    print(
        f"Calibrated (multi-fold) avgKL~={chosen_avg_kl:.5f} with TARGET_TRANSFORM={TARGET_TRANSFORM}, "
        f"Y_SMOOTH_EPS={Y_SMOOTH_EPS:.3f}, L2={L2_CHOSEN:.1f}, TEMP_SOFTMAX={TEMP_SOFTMAX:.2f}, ALPHA_BLEND={ALPHA_BLEND:.2f}"
    )

    FEATURES_READY = True


@torch.no_grad()
def gen_ensemble_pred(df_row: dict) -> np.ndarray:
    eeg_id = int(df_row["eeg_id"])
    filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")

    if HAVE_HMS_MODELS and (len(models_3) > 0):
        eeg = compute_eeg_from_file(filepath)

        eeg_3 = proc_3(eeg)
        eeg_30 = flip_h(eeg_3)
        eeg_31 = flip_v(eeg_3)
        eeg_32 = flip_v(flip_h(eeg_3))

        eeg_3 = torch.tensor(eeg_3, device=device, dtype=torch.float32).unsqueeze(0)
        eeg_30 = torch.tensor(eeg_30, device=device, dtype=torch.float32).unsqueeze(0)
        eeg_31 = torch.tensor(eeg_31, device=device, dtype=torch.float32).unsqueeze(0)
        eeg_32 = torch.tensor(eeg_32, device=device, dtype=torch.float32).unsqueeze(0)

        preds = []
        for model in models_3:
            model.eval()
            preds.append(model(eeg_3).exp().cpu().numpy().reshape(-1))
            preds.append(model(eeg_30).exp().cpu().numpy().reshape(-1))
            preds.append(model(eeg_31).exp().cpu().numpy().reshape(-1))
            preds.append(model(eeg_32).exp().cpu().numpy().reshape(-1))

        preds = np.mean(preds, axis=0)
        preds = np.clip(preds, 1e-12, None)
        preds = preds / preds.sum()
        return preds

    if not FEATURES_READY:
        _prepare_feature_model(max_train_eegs=3000, random_state=0)

    if FEATURE_W is None or (not os.path.exists(filepath)):
        return PRIOR.copy()

    chain = compute_eeg_from_file(filepath)
    chain = proc_3(chain)
    feats_eeg = extract_features_from_chain(chain).astype(np.float64, copy=False)

    sp_id = int(df_row["spectrogram_id"])
    path_sp = os.path.join(SPEC_DIR, f"{sp_id}.parquet")
    if os.path.exists(path_sp):
        feats_sp = extract_features_from_spectrogram_file(path_sp).astype(
            np.float64, copy=False
        )
    else:
        feats_sp = np.zeros((12,), dtype=np.float64)

    feats0 = (
        np.concatenate([feats_eeg, feats_sp], axis=0)
        .reshape(1, -1)
        .astype(np.float64, copy=False)
    )

    feats = _expand_features(feats0)

    if (FEATURE_MEAN is not None) and (FEATURE_STD is not None):
        feats = _standardize_apply(
            feats, FEATURE_MEAN.reshape(1, -1), FEATURE_STD.reshape(1, -1)
        )

    Xb = np.concatenate([feats, np.ones((1, 1), dtype=np.float64)], axis=1)
    logits = Xb @ FEATURE_W  # (1,6)

    logits = logits - logits.mean(axis=1, keepdims=True)
    p_raw = _softmax(logits / float(TEMP_SOFTMAX), axis=1).reshape(-1)

    a = float(ALPHA_BLEND)
    p = (1.0 - a) * p_raw + a * PRIOR
    p = np.clip(p, 1e-12, None)
    p = p / p.sum()
    return p




## === cell 13
preds_final = []
for row in tqdm(
    df_test.iter_rows(named=True), total=df_test.height, desc="Predict test"
):
    pred = gen_ensemble_pred(row)
    preds_final.append(pred)

preds_final = np.asarray(preds_final, dtype=np.float64)
print("preds_final shape:", preds_final.shape)



## === cell 14
df_sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].to_list()})
df_sub[LABELS] = preds_final

row_sums = df_sub[LABELS].sum(axis=1).to_numpy()
if not np.all(np.isfinite(row_sums)) or np.any(row_sums <= 0):
    raise ValueError("Non-finite or non-positive row sums in predictions.")
df_sub[LABELS] = df_sub[LABELS].div(row_sums, axis=0)

df_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", df_sub.shape)
df_sub.head()
