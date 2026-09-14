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

0.87468

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'The fix adds robust handling for missing model files and replaces the failing deep‑learning inference with a simple prior‑based prediction: it computes the average class probabilities from the training votes and uses those as the prediction for every test record. This removes the import error, the reshaping mismatch, and guarantees a valid CSV submission while keeping the overall pipeline structure intact. If any models are successfully loaded, they still be used; otherwise the fallback ensures the script runs end‑to‑end and produces a properly formatted submission.'
- What this solution (achieved 1.39779) has done: 'I fix the model‑loading path so that any available pretrained checkpoints are actually found and used. The fallback to class‑prior predictions only be used when no models are discovered, keeping the rest of the pipeline unchanged. This should lower the KL‑divergence score toward the target without altering the core logic.'
- What this solution (achieved 1.64506) has done: 'The fixes add missing imports (`os`, `polars`, `pandas`, `torch.nn`), ensure all libraries are available before they are used, and keep the existing fallback‑to‑class‑prior logic unchanged. This resolves the NameErrors, allows the script to run end‑to‑end, and writes a valid `submission.csv` with properly normalized probability rows.'
- What this solution (achieved 1.64506) has done: 'I broaden the model‑loading logic to actually discover and load any checkpoint files under the provided `hms-models` directory (instead of only files named `model_best_val_g10.pt`). The code now searches for all “.pt” files, attempts to instantiate the available `EegModel3` (or any other imported model class) and adds robust error handling so that failed loads are skipped without stopping the pipeline. This small change enables the ensemble predictions to be generated when pretrained models are present, which should markedly lower the KL‑divergence score toward the target while keeping the original fallback behaviour unchanged.'
- What this solution (achieved 1.68479) has done: 'I improve the fallback‑prediction logic by weighting the class‑prior and per‑patient priors with the actual number of annotator votes for each record, which yields a more representative probability estimate and should lower the KL‑divergence toward the target. The rest of the pipeline stays unchanged.'
- What this solution (achieved 0.85517) has done: 'I slightly adjust the fallback prediction logic so that when no trained models are loaded the code blends the per‑patient prior with the overall class‑prior instead of using either one alone. This modest smoothing (using a weighted average with α≈0.6) keeps the core pipeline unchanged while producing more calibrated probabilities, which should lower the KL‑divergence score toward the target.'
- What this solution (achieved 1.0413) has done: 'I slightly reduce the blending weight for the per‑patient prior (α) from 0.6 to 0.3 so the fallback prediction leans more toward the globally‑derived class prior, which empirically brings the KL‑divergence closer to the target while keeping the original logic intact.'
- What this solution (achieved 0.81597) has done: 'I increase the blending weight α for the per‑patient prior in the fallback prediction (from 0.3 to 0.7). This gives the per‑patient distribution more influence, which should lower the KL‑divergence and move the score closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.78004) has done: 'I increase the blending weight α for the per‑patient prior from 0.7 to 0.85 so the fallback prediction relies more on the per‑patient distribution, which from earlier experiments should lower the KL‑divergence and move the score closer to the target. No other logic is altered.'
- What this solution (achieved 0.78698) has done: 'I increase the blending weight α for the per‑patient prior from 0.85 to 0.95 so the fallback prediction relies more on the patient‑specific distribution, which previous experiments showed moves the KL‑divergence lower. This is a single‑line change in the `gen_ensemble_pred` function, keeping all other logic untouched.'
- What this solution (achieved 0.77801) has done: 'I add a lightweight smoothing step to the fallback prediction: after blending the per‑patient prior with the global class prior, I raise the probabilities to a power β (β < 1) and renormalize. This makes the distribution less confident, which empirically lowers KL‑divergence when the model cannot be loaded, moving the score closer to the target while keeping all core logic unchanged.'
- What this solution (achieved 0.87468) has done: 'I adjust the fallback prediction to add a modest uniform mixing after the existing per‑patient / global prior blending and smoothing. By mixing in a small portion of a uniform distribution (γ = 0.1) and using a slightly stronger smoothing exponent (β = 0.5), the predictions become less confident and should lower the KL‑divergence, moving the score closer to the target while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import polars as pl
import torch
import torch.nn as nn

DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
SPEC_DIR = os.path.join(DATA_DIR, "test_spectrograms/")
EEG_DIR = os.path.join(DATA_DIR, "test_eegs/")

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

df_test.head()




## === cell 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print()




## === cell 2
def load_model(path: str, model: nn.Module) -> nn.Module:
    model.load_state_dict(
        torch.load(path, map_location=torch.device("cpu"))["model_state_dict"]
    )
    return model




## === cell 3
import sys

sys.path.append("/kaggle/input/hms-models/")

try:
    from eeg_cnn_rnn_w1 import EegModel as EegModel0
    from eeg_cnn_rnn import EegModel as EegModel1
    from spc_cnn_rnn import SpectrogramModel
    from eeg_cnn_rnn_final import EegModel as EegModel3
except ModuleNotFoundError:
    EegModel0 = None
    EegModel1 = None
    SpectrogramModel = None
    EegModel3 = None
    print("Model source files not found – proceeding with fallback predictions.")




## === cell 4
import glob

models_3 = []
if EegModel3 is not None:
    base_model_dir = "/kaggle/input/hms-models/"
    pattern = os.path.join(base_model_dir, "**", "*.pt")
    for model_path in glob.glob(pattern, recursive=True):
        if not os.path.isfile(model_path):
            continue
        try:
            model = EegModel3()
            model = load_model(model_path, model)
            model = model.to(device)
            model.eval()
            models_3.append(model)
            print(f"Loaded model checkpoint: {model_path}")
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")

size = len(models_3)
print(f"Number of loaded EegModel3 ensembles: {size}")

if size == 0:
    label_cols = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]
    votes_np = df_train.select(label_cols).to_pandas().values.astype(float)
    row_sums = votes_np.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    probs = votes_np / row_sums
    row_weights = row_sums.squeeze()

    class_prior = np.average(probs, axis=0, weights=row_weights)  # shape (6,)

    patient_ids = df_train["patient_id"].to_pandas().values
    patient_df = pd.DataFrame(probs, columns=label_cols)
    patient_df["patient_id"] = patient_ids
    patient_df["weight"] = row_weights

    patient_prior = {}
    for pid, grp in patient_df.groupby("patient_id"):
        w = grp["weight"].values
        prob_vals = grp[label_cols].values
        weighted = np.average(prob_vals, axis=0, weights=w)
        patient_prior[pid] = weighted.astype(float)

    print("Using weighted class‑prior fallback predictions with per‑patient priors.")
else:
    class_prior = None
    patient_prior = {}




## === cell 5
import scipy
from scipy.signal import welch, butter, filtfilt

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




## === cell 6
def MAD(signal, axis=-1):
    """Compute the robust standard deviation (MAD) of a signal."""
    median = np.median(signal, axis=axis, keepdims=True)
    absolute_deviations = np.abs(signal - median)
    median_absolute_deviation = np.median(absolute_deviations, axis=axis, keepdims=True)
    scale_factor = 1.4826  # constant for normal distribution
    robust_std = median_absolute_deviation * scale_factor
    return robust_std




## === cell 7
def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    b, a = butter(N=order, Wn=cutoff_freq / (0.5 * fs), btype=btype, analog=False)
    return filtfilt(b, a, eeg_data)




## === cell 8
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




## === cell 9
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

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
    mad_std = MAD(eeg, axis=-1).reshape(-1)
    mad_std = np.median(mad_std) + 1e-5
    eeg = eeg / mad_std
    eeg = eeg.clip(-10, 10)
    signal_len = eeg.shape[-1] // 16
    try:
        eeg = eeg.reshape(4, 4, signal_len)
    except ValueError:
        eeg = eeg.reshape(4, 4, -1)
    return eeg


def flip_h(eeg):
    return eeg[::-1]


def flip_v(eeg):
    return eeg[:, ::-1]


@torch.no_grad()
def gen_ensemble_pred(df_row: pl.DataFrame) -> np.ndarray:
    """
    Returns a prediction vector for a single row.
    If model ensemble is unavailable, returns a blended per‑patient / global prior
    with added smoothing and a small uniform mix to improve KL divergence.
    """
    if class_prior is not None:
        eeg_id = df_row["eeg_id"].item()
        patient_series = df_test.filter(pl.col("eeg_id") == eeg_id)["patient_id"]
        patient_id = None if patient_series.is_empty() else patient_series.to_numpy()[0]

        pred = class_prior.copy()

        if patient_id is not None and patient_id in patient_prior:
            alpha = 0.95  # weight for patient‑specific prior
            pred = alpha * patient_prior[patient_id] + (1 - alpha) * class_prior

        beta = 0.5
        pred = np.power(pred, beta)
        pred = pred / pred.sum()

        gamma = 0.1
        uniform = np.full_like(pred, 1.0 / len(pred))
        pred = (1 - gamma) * pred + gamma * uniform
        pred = pred / pred.sum()  # final renormalisation

        return pred

    eeg_id = df_row["eeg_id"].item()
    filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")
    eeg = compute_eeg_from_file(filepath)

    eeg_3 = proc_3(eeg)
    eeg_30 = flip_h(eeg_3)
    eeg_31 = flip_v(eeg_3)
    eeg_32 = flip_v(flip_h(eeg_3))

    eeg_3 = torch.Tensor(eeg_3).to(device).unsqueeze(0)
    eeg_30 = torch.Tensor(eeg_30).to(device).unsqueeze(0)
    eeg_31 = torch.Tensor(eeg_31).to(device).unsqueeze(0)
    eeg_32 = torch.Tensor(eeg_32).to(device).unsqueeze(0)

    preds = []
    for model in models_3:
        preds.append(model(eeg_3).exp().cpu().detach().numpy().reshape(-1))
        preds.append(model(eeg_30).exp().cpu().detach().numpy().reshape(-1))
        preds.append(model(eeg_31).exp().cpu().detach().numpy().reshape(-1))
        preds.append(model(eeg_32).exp().cpu().detach().numpy().reshape(-1))

    if not preds:
        return np.full(len(LABELS), 1.0 / len(LABELS))

    preds = np.mean(preds, axis=0)
    preds = preds / preds.sum()
    return preds




## === cell 10
from tqdm.auto import tqdm

preds_final = []
for eeg_id in df_test["eeg_id"].to_list():
    row_df = pl.DataFrame({"eeg_id": [eeg_id]})
    pred = gen_ensemble_pred(row_df)
    preds_final.append(pred)




## === cell 11
df_sub = pd.DataFrame({"eeg_id": df_test["eeg_id"].to_list()})
df_sub[LABELS] = np.vstack(preds_final)
submission_path = "submission.csv"
df_sub.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
df_sub.head()
