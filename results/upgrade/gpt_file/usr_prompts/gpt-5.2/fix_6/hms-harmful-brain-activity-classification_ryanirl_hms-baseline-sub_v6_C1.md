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

0.5138472402117003

# 6. Current score

1.18422

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40985) has done: 'I remove the hard dependency on the missing `/kaggle/input/hms-models/` modules by providing a small fallback model that preserves the same “EEG → logits → softmax probabilities” inference semantics, so the notebook runs end-to-end in this environment. I also fix the current runtime errors: undefined `preprocess_chain`, incorrect Polars row access, and the Pandas assignment shape mismatch when writing predictions to the 6 vote columns. To keep the submission valid for the KL metric, predictions be forced to be proper probabilities (softmax + row-wise normalization + small epsilon). Finally, I remove per-sample debug printing (which would time out on 9850 test rows) to ensure the script finishes and writes `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'Your current score is far worse than the target (lower-is-better), so the biggest legitimate gain with minimal semantic change is to stop using the untrained fallback model and instead use the label distribution from `train.csv` as a calibrated prior; this matches the KL metric much better than near-random predictions. I keep your per-row probability normalization and submission schema unchanged, but add a safe “prior mode” that triggers automatically when no external weights are found. To avoid overly sharp probabilities (which are punished by KL when wrong), I apply a tiny Dirichlet-style smoothing to the prior before normalizing. This should move the score substantially toward the target while keeping runtime well under the limit.'
- What this solution (achieved 0.76744) has done: 'Your current score (1.41937, lower-is-better) is far from the target (0.5138), so we should make a small, metric-aligned improvement without changing your model/feature core. The biggest issue is that your “prior-only” fallback uses a global class prior, but the label distribution is strongly patient-dependent; switching to a patient-specific prior (computed from train by `patient_id`) is still a legitimate, minimal change that typically reduces KL a lot. We keep the same inference path when external weights exist, and only improve the no-weights fallback by using a smoothed per-patient prior with a safe fallback to the global prior for unseen patients. We also preserve your probability normalization and output schema exactly, ensuring a valid submission.'
- What this solution (achieved 1.19844) has done: 'Your current score (0.76744, lower-is-better) is still worse than the target (0.51385), so we should make a small, metric-aligned improvement that doesn’t change your model/inference core. The simplest gain is to make the “prior-only” fallback less biased by repeated overlapping train rows by computing priors at the `eeg_id` level (average label distribution per EEG, then aggregate), and then per-patient using those per-EEG distributions; this usually improves calibration for KL. We keep the same smoothing, normalization, and the exact same behavior when external weights exist; only the no-weights fallback prior computation changes. We also keep the submission schema identical and still enforce row-wise probability sums to 1.'
- What this solution (achieved 1.18422) has done: 'Your current score (1.19844, lower-is-better) is still far from the target (0.51385), so we should improve the prior-only fallback in a metric-aligned way without changing your model/feature pipeline. The simplest high-impact fix is to compute patient-specific priors at the *label_id* (unique label set) level rather than the *eeg_id* level, because your current eeg-level averaging still over-weights duplicated/overlapping labels within the same label set. This keeps the same “use external weights if present, else use a smoothed prior” semantics, but makes the fallback prior closer to the true distribution and typically reduces KL. We keep the same smoothing + row-normalization to guarantee valid probabilities and a valid submission.'

# 9. Code solution

## === cell 0
from tqdm.auto import tqdm
import pandas as pd
import polars as pl
import numpy as np
import os
import glob
import sys

import torch
import torch.nn as nn



## === cell 1
DATA_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
SPEC_DIR = os.path.join(DATA_DIR, "test_spectrograms/")
EEG_DIR = os.path.join(DATA_DIR, "test_eegs/")

df_train = pl.read_csv(os.path.join(DATA_DIR, "train.csv"))
df_test = pl.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_submission = pl.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

df_test.head()



## === cell 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## === cell 3
def load_model(path: str, model: nn.Module) -> nn.Module:
    ckpt = torch.load(path, map_location=torch.device("cpu"))
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type at {path}: {type(ckpt)}")
    model.load_state_dict(state, strict=True)
    return model




## === cell 4
try:
    sys.path.append("/kaggle/input/hms-models/")
    from eeg_cnn_rnn_softmax import EegModel as EegModelSoftmax  # type: ignore

    CurrModel = EegModelSoftmax
    HAVE_EXTERNAL_MODEL = True
except Exception:
    HAVE_EXTERNAL_MODEL = False

    class CurrModel(nn.Module):
        def __init__(self, n_classes: int = 6):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(4, 32, kernel_size=9, padding=4),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=9, padding=4),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, n_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            logits = self.net(x)
            return torch.softmax(logits, dim=1)

    print(
        "Warning: external model modules not found; using fallback model for inference."
    )



## === cell 5
ensembles = []

fold_dirs = [
    "/kaggle/input/hms-models/softmax_simple_eeg_cnn_rnn_stage_2/softmax_simple_eeg_cnn_rnn_stage_2/*"
]

models = []
found_any_weight = False
for fold_dir in fold_dirs:
    for fold_path in glob.glob(fold_dir):
        model_path = os.path.join(fold_path, "model_final.pt")
        if os.path.exists(model_path):
            model = CurrModel()
            model = load_model(model_path, model)
            model = model.to(device)
            models.append(model)
            found_any_weight = True

if not models:
    models = [CurrModel().to(device)]

print(f"Loaded {len(models)} model(s). External weights found: {found_any_weight}")



## === cell 6
import librosa
from scipy.ndimage import convolve

KERNEL = np.array([-1, -1, -1, 0, 1, 1, 1])


def preprocess_chain(chain: np.ndarray) -> np.ndarray:
    return chain.astype(np.float32)


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
    chain = preprocess_chain(chain)
    return chain




## === cell 7
from scipy.signal import butter, filtfilt


def butter_filter(eeg_data, fs=200, cutoff_freq=22, order=4):
    b, a = butter(N=order, Wn=cutoff_freq / (0.5 * fs), btype="low", analog=False)
    return filtfilt(b, a, eeg_data)


def compute_eeg(eeg: np.ndarray) -> np.ndarray:
    eeg = butter_filter(eeg)
    eeg = eeg[::2]  # 100 Hz
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
    return chain.astype(np.float32)




## === cell 8
LABELS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]


def compute_train_prior_label_level(
    train_df: pl.DataFrame, labels: list[str], alpha: float = 1.0
) -> np.ndarray:
    eps = 1e-12
    row_sum = pl.sum_horizontal([pl.col(c) for c in labels]).cast(pl.Float64) + eps
    per_row = train_df.select(
        ["label_id"] + [(pl.col(c).cast(pl.Float64) / row_sum).alias(c) for c in labels]
    )

    per_label = per_row.group_by("label_id").agg(
        [pl.col(c).mean().alias(c) for c in labels]
    )

    global_mean = (
        per_label.select([pl.col(c).mean().alias(c) for c in labels])
        .to_numpy()
        .reshape(-1)
        .astype(np.float64)
    )

    global_mean = global_mean + float(alpha)
    global_mean = np.clip(global_mean, 1e-8, None)
    global_mean = global_mean / global_mean.sum()
    return global_mean.astype(np.float32)


def compute_patient_priors_label_level(
    train_df: pl.DataFrame, labels: list[str], alpha: float = 1.0
) -> dict[int, np.ndarray]:
    eps = 1e-12
    row_sum = pl.sum_horizontal([pl.col(c) for c in labels]).cast(pl.Float64) + eps
    per_row = train_df.select(
        ["patient_id", "label_id"]
        + [(pl.col(c).cast(pl.Float64) / row_sum).alias(c) for c in labels]
    )

    per_label = per_row.group_by(["patient_id", "label_id"]).agg(
        [pl.col(c).mean().alias(c) for c in labels]
    )

    per_patient = per_label.group_by("patient_id").agg(
        [pl.col(c).mean().alias(c) for c in labels]
    )

    patient_ids = per_patient["patient_id"].to_list()
    mat = per_patient.select(labels).to_numpy().astype(np.float64)  # (n_patients, 6)

    mat = mat + float(alpha)
    mat = np.clip(mat, 1e-8, None)
    mat = mat / mat.sum(axis=1, keepdims=True)
    return {int(pid): mat[i].astype(np.float32) for i, pid in enumerate(patient_ids)}


TRAIN_PRIOR = compute_train_prior_label_level(df_train, LABELS, alpha=1.0)
PATIENT_PRIORS = compute_patient_priors_label_level(df_train, LABELS, alpha=1.0)

USE_PRIOR_ONLY = not found_any_weight
print(
    "USE_PRIOR_ONLY:",
    USE_PRIOR_ONLY,
    "| TRAIN_PRIOR:",
    TRAIN_PRIOR,
    "| #PATIENT_PRIORS:",
    len(PATIENT_PRIORS),
)


@torch.no_grad()
def gen_ensemble_pred(models, eeg_id: int, patient_id: int) -> np.ndarray:
    if USE_PRIOR_ONLY:
        return PATIENT_PRIORS.get(int(patient_id), TRAIN_PRIOR).copy()

    filepath = os.path.join(EEG_DIR, f"{eeg_id}.parquet")
    x = compute_eeg_from_file(filepath)  # (4, T)
    x = torch.tensor(x, device=device)
    x = x - x.mean(dim=-1, keepdim=True)
    x = x / (x.std(dim=-1, keepdim=True) + 1e-5)
    x = x.unsqueeze(0)  # (1, 4, T)

    preds = []
    for model in models:
        model.eval()
        pred = model(x)  # expected (1, 6)
        pred = pred.float().cpu().numpy().reshape(-1)
        preds.append(pred)

    preds = np.mean(np.stack(preds, axis=0), axis=0)
    preds = np.clip(preds, 1e-8, None)
    preds = preds / preds.sum()
    return preds.astype(np.float32)




## === cell 9
eeg_ids = df_test["eeg_id"].to_list()
patient_ids = df_test["patient_id"].to_list()

preds_final = np.zeros((len(eeg_ids), len(LABELS)), dtype=np.float32)
for i, (eeg_id, patient_id) in enumerate(
    tqdm(list(zip(eeg_ids, patient_ids)), total=len(eeg_ids))
):
    preds_final[i] = gen_ensemble_pred(models, int(eeg_id), int(patient_id))

preds_final.shape



## === cell 10
df_sub = pd.DataFrame({"eeg_id": eeg_ids})
df_sub[LABELS] = preds_final

row_sums = df_sub[LABELS].sum(axis=1).to_numpy()
df_sub[LABELS] = (df_sub[LABELS].to_numpy() / row_sums[:, None]).astype(np.float32)

df_sub.to_csv("submission.csv", index=False)
df_sub.head()
