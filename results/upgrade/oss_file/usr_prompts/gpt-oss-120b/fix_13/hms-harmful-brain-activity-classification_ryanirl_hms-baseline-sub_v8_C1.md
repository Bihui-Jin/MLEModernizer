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

0.4134363371417311

# 6. Current score

1.19111

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I add the missing imports, correctly define the dummy model and related variables, compute a baseline probability from the training vote distribution (which should give a better KL score than a uniform guess), and ensure the script writes a proper `submission.csv` with the required columns.'
- What this solution (achieved 1.68479) has done: 'I add simple per‑recording and per‑patient probability look‑ups derived from the training vote counts and use them as predictions (falling back to the global baseline when no match is found). This keeps the original dummy model untouched while giving the model much more relevant priors, which should lower the KL score toward the target.'
- What this solution (achieved 1.05318) has done: 'The update adds a small smoothing term and blends per‑eeg and per‑patient priors instead of picking one exclusively, which reduces KL divergence by providing more calibrated probabilities while keeping the original dummy model untouched. The changes are confined to the prediction function and keep the rest of the pipeline identical.'
- What this solution (achieved 0.85517) has done: 'I adjust the prediction blending to include a contribution from the global baseline distribution, which should regularize overly confident per‑eeg or per‑patient priors and typically lowers KL divergence. The new weights keep the original logic but add a baseline term (e.g., 0.3 when both eeg & patient are known, 0.4 when only patient is known). This small change is expected to move the score toward the lower target while preserving the core pipeline.'
- What this solution (achieved 0.81597) has done: 'I slightly increase the influence of the specific per‑eeg and per‑patient priors in the blending step, reducing the contribution of the global baseline. This should make predictions more tailored to each record and is expected to lower the KL‑divergence toward the target while keeping the rest of the pipeline unchanged.'
- What this solution (achieved 0.96642) has done: 'The update adjusts the blending in `gen_ensemble_pred` to rely more on the global baseline distribution (which is a stable reference) and less on the noisy per‑eeg / per‑patient priors. The new weights (eeg 0.3 + patient 0.2 + baseline 0.5, or patient 0.4 + baseline 0.6) soften over‑confident predictions, and a slightly larger epsilon improves numerical stability. This modest change is expected to lower the KL‑divergence toward the target while leaving the core pipeline untouched.'
- What this solution (achieved 0.83438) has done: 'I adjust the blending weights in the ensemble prediction to rely more on the per‑eeg and per‑patient priors (which are more informative) and less on the global baseline. By increasing `_EEG_WEIGHT` and `_PATIENT_WEIGHT_EEG` while decreasing `_BASELINE_WEIGHT_EEG` (and similarly for the patient‑only case), the predictions become better calibrated to the training distribution, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.90514) has done: 'I lower the per‑eeg weight and raise the baseline contribution in the blending step, because relying heavily on a single‑record prior can over‑fit and increase KL‑divergence. By giving the global baseline a larger share (both when an EEG‑specific prior exists and when only a patient‑specific prior is available) the predictions become better calibrated toward the target distribution, which should move the score nearer to the lower target value.'
- What this solution (achieved 1.13457) has done: 'We raise the contribution of the global baseline distribution and lower the influence of the noisy per‑eeg and per‑patient priors, which should make the predictions more conservative and move the KL‑divergence nearer the lower target score. Only the blending weight constants are adjusted; the rest of the pipeline stays unchanged.'
- What this solution (achieved 1.04144) has done: 'I slightly increase the influence of the per‑eeg and per‑patient priors while decreasing the global baseline weight in the blending step. This modest adjustment keeps the core logic unchanged but makes the predictions more tailored to each recording, which should lower the KL‑divergence toward the target value.'
- What this solution (achieved 1.19111) has done: 'I reduce the influence of the per‑eeg and per‑patient priors and increase the contribution of the global baseline distribution, which should make the predictions more conservative and bring the KL‑divergence closer to the lower target score. This is done by adjusting the blending weights and slightly enlarging the epsilon for numerical stability.'

# 9. Code solution

## === cell 0
import os
import polars as pl
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd

DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
SPEC_DIR = os.path.join(DATA_DIR, "test_spectrograms/")
EEG_DIR = os.path.join(DATA_DIR, "test_eegs/")

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))




## === cell 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 2
def load_model(path: str, model: nn.Module) -> nn.Module:
    model.load_state_dict(
        torch.load(path, map_location=torch.device("cpu"))["model_state_dict"]
    )
    return model




## === cell 3
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def knn_predict(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        probs = torch.full((batch, 6), 1.0 / 6.0, device=x.device)
        return probs


CurrModel = DummyModel




## === cell 4
models = [CurrModel().to(device)]




## === cell 5
import librosa
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
    )
    spectrogram = librosa.power_to_db(np.abs(spectrogram) ** 2, ref=np.max).astype(
        np.float32
    )
    spectrogram = (spectrogram + 80) / 80
    spectrogram = spectrogram**2
    return spectrogram[:256][::2, ::2]


def spec(eeg: np.ndarray) -> np.ndarray:
    spec_val = compute_spec(eeg)
    eeg = convolve(eeg, KERNEL)
    spec_val = spec_val + compute_spec(eeg)
    return spec_val / 2


def compute_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    channels = [
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
    arrays = [df_eeg[ch].to_numpy() for ch in channels]
    (Fp1, Fp2, F3, F4, F7, F8, C3, C4, P3, P4, T3, T4, T5, T6, O1, O2) = arrays

    ll = np.stack([(spec(Fp1 - F7), spec(F7 - T3), spec(T3 - T5), spec(T5 - O1))])
    lp = np.stack([(spec(Fp1 - F3), spec(F3 - C3), spec(C3 - P3), spec(P3 - O1))])
    rp = np.stack([(spec(Fp2 - F4), spec(F4 - C4), spec(C4 - P4), spec(P4 - O2))])
    rl = np.stack([(spec(Fp2 - F8), spec(F8 - T4), spec(T4 - T6), spec(T6 - O2))])

    chain = np.stack([ll, lp, rp, rl])[:, 0]
    return chain


def preprocess_chain(chain: np.ndarray) -> np.ndarray:
    chain = chain - chain.mean()
    chain = chain / (chain.std() + 1e-6)
    return chain


def compute_spec_from_file(filepath: str) -> np.ndarray:
    df_eeg = pl.read_parquet(filepath).fill_null(0)
    chain = compute_chain(df_eeg)
    chain = preprocess_chain(chain)
    return chain




## === cell 6
from scipy.signal import butter, filtfilt
import math


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4, btype="lowpass"):
    b, a = butter(N=order, Wn=cutoff_freq / (0.5 * fs), btype=btype, analog=False)
    return filtfilt(b, a, eeg_data)


def bin_array(
    array,
    bin_size,
    axis=-1,
    pad_dir="symmetric",
    mode="edge",
    return_padding=False,
    **padding_kwargs,
) -> np.ndarray:
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


def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg, cutoff_freq=np.array([0.25, 50]), btype="bandpass")
    eeg = bin_array(eeg, bin_size=4, mode="reflect").mean(axis=-1)
    return eeg


def compute_eeg_chain(df_eeg: pl.DataFrame) -> np.ndarray:
    channels = [
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
    arrays = [df_eeg[ch].to_numpy() for ch in channels]
    (Fp1, Fp2, Fp3, Fp4, F7, F8, C3, C4, P3, P4, T3, T4, T5, T6, O1, O2) = (
        arrays  # noqa: F841
    )

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
    return compute_eeg_chain(df_eeg)




## === cell 7
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]




## === cell 8
train_votes_sum = df_train.select(LABELS).sum()
total_votes = sum(train_votes_sum[0, col] for col in LABELS)
BASE_PROB = np.array(
    [train_votes_sum[0, col] / total_votes for col in LABELS], dtype=np.float32
)

df_train_pd = df_train.to_pandas()

eeg_group = df_train_pd.groupby("eeg_id")[LABELS].sum()
eeg_total = eeg_group.sum(axis=1)
eeg_prob_dict = {
    eid: (eeg_group.loc[eid] / eeg_total.loc[eid]).values.astype(np.float32)
    for eid in eeg_group.index
}

patient_group = df_train_pd.groupby("patient_id")[LABELS].sum()
patient_total = patient_group.sum(axis=1)
patient_prob_dict = {
    pid: (patient_group.loc[pid] / patient_total.loc[pid]).values.astype(np.float32)
    for pid in patient_group.index
}




## === cell 9
np.set_printoptions(formatter={"all": lambda x: f"{x:0.3f}"})

_EPS = 1e-3  # slightly larger epsilon for better numerical stability

_EEG_WEIGHT = 0.05  # weight for per‑eeg prior when both eeg and patient are known
_PATIENT_WEIGHT_EEG = 0.05  # weight for per‑patient prior when both are known
_BASELINE_WEIGHT_EEG = 0.90  # weight for global baseline in the same case (sums to 1)

_PATIENT_WEIGHT_ONLY = 0.15  # weight for per‑patient prior when eeg is unknown
_BASELINE_WEIGHT_ONLY = (
    0.85  # weight for baseline when only patient is known (sums to 1)
)


@torch.no_grad()
def gen_ensemble_pred(models, row_df: pl.DataFrame) -> np.ndarray:
    """
    Produce a prediction for a single test row using a more conservative blending:
    the global baseline now dominates, which typically lowers KL divergence
    toward the target while keeping the original logic untouched.
    """
    eeg_id = row_df["eeg_id"].item()
    patient_id = row_df["patient_id"].item()

    if eeg_id in eeg_prob_dict:
        prob = _EEG_WEIGHT * eeg_prob_dict[eeg_id]
        if patient_id in patient_prob_dict:
            prob += _PATIENT_WEIGHT_EEG * patient_prob_dict[patient_id]
        prob += _BASELINE_WEIGHT_EEG * BASE_PROB
    elif patient_id in patient_prob_dict:
        prob = (
            _PATIENT_WEIGHT_ONLY * patient_prob_dict[patient_id]
            + _BASELINE_WEIGHT_ONLY * BASE_PROB
        )
    else:
        prob = BASE_PROB.copy()

    prob = prob + _EPS
    prob = prob / prob.sum()
    return prob




## === cell 10
preds_final = []
eeg_ids = df_test["eeg_id"].to_list()
patient_ids = df_test["patient_id"].to_list()
for eeg_id, patient_id in tqdm(
    zip(eeg_ids, patient_ids), total=len(eeg_ids), desc="Predicting"
):
    row_df = pl.DataFrame({"eeg_id": [eeg_id], "patient_id": [patient_id]})
    pred = gen_ensemble_pred(models, row_df)
    preds_final.append(pred)




## === cell 11
df_sub = pd.DataFrame({"eeg_id": eeg_ids})
preds_array = np.stack(preds_final)  # shape (n_rows, 6)
df_sub[LABELS] = preds_array
df_sub.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
