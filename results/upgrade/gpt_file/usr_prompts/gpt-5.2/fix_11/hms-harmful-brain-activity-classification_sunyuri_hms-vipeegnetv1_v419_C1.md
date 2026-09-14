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

3.13

# 3. Installed packages

No external packages required in the script and installed.

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

0.274080816395206

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the runtime crash by removing the unused `keras_hub` import that triggers a protobuf incompatibility (`MessageFactory.GetPrototype`) in this Kaggle environment. I also make the inference path robust on Kaggle by (1) defaulting to `sample_submission.csv` output if no model weights are found, and (2) ensuring predictions are valid probabilities (non-negative, normalized to sum to 1) to avoid submission rejection. Finally, I correct a slicing bug in the test batching loop that can create empty/shifted batches and misalign `preds_all` vs `eeg_id`, which would otherwise yield an invalid submission. These changes keep the core model/training logic intact and focus on producing a valid end-to-end `submission.csv`.'
- What this solution (achieved 1.41937) has done: 'I fix the runtime crash by removing the incompatible `torchaudio`/`torch` imports that trigger the `MessageFactory.GetPrototype` protobuf error in this Kaggle environment (they are not needed when `DATATYPE=["eeg"]`). I also ensure the notebook runs end-to-end on Kaggle by setting `NEEDTRAIN=False` only on Kaggle as originally intended, and by producing a valid `submission.csv` even when no weights exist (already present). Finally, to move the score toward the target (lower is better) with minimal semantic change, I replace the fallback “copy sample_submission” with a label-prior baseline computed from `train.csv` vote proportions (normalized), which typically scores much better than uniform/sample defaults while keeping core modeling logic unchanged.'
- What this solution (achieved 0.78004) has done: 'I fix the protobuf-related runtime crash by avoiding the TensorFlow/Keras import path that triggers `MessageFactory.GetPrototype` in this Kaggle environment and instead run the script in a safe “submission-only” mode by default on Kaggle. Then, to move the (lower-is-better) score strongly toward your target with minimal semantic change, I keep the existing fallback logic but improve it from a global prior to a patient-aware prior (computed only from `train.csv` without any label leakage into test labels), blended slightly with the global prior for stability. Finally, I ensure the submission is always valid (correct columns, float32, non-negative, row-normalized to sum to 1) and is written to `submission.csv`.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

import os
import warnings
import gc
import time

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"
    for dir_name in os.listdir("./input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"
    NEEDTRAIN = False
    for dir_name in os.listdir("/kaggle/input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print("DATATYPE:", DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

SFREQ = 200
RSFREQ = 200

EEG_LENGTH = 50
EEG_LENGTH_USED = 50
EEG_CHANNEL_USED = 16
EEG_MULTIPLY = 1

filter_range = [0.5, 45]
filter_range2 = [0.1, 35]

SEED = 2024
BATCHSIZE = 16
LEARN_RATE = 1e-3
EPOCHS = 15
SPLITS = 5

READ_EEG_FILES = False
READ_SPE_FILES = False

spectrograms = {}
eegs = {}
stfts = {}
imgs = {}

spectrograms_test = {}
eegs_test = {}
stfts_test = {}
imgs_test = {}

BRAIN = [
    "Fp1-F7",
    "F7-T3",
    "T3-T5",
    "T5-O1",  # LL
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",  # LP
    "Fz-Cz",
    "Cz-Pz",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",  # RP
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",  # RL
]

TEST_BATCHSIZE = 128

np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets:", list(TARGETS))



## === cell 1
if NEEDTRAIN:
    TARGETS_RAW = [i + "_raw" for i in TARGETS]

    if READ_EEG_FILES:
        train = df.drop_duplicates(
            [
                "eeg_id",
                "seizure_vote",
                "lpd_vote",
                "gpd_vote",
                "lrda_vote",
                "grda_vote",
                "other_vote",
            ]
        ).reset_index(drop=True)

        train["sign_id"] = train.index.values
        df["sign_id"] = df.index.values

        y_data = train[TARGETS].values
        train[TARGETS_RAW] = y_data
        y_data = y_data / (y_data.sum(axis=1, keepdims=True) + 1e-12)
        train[TARGETS] = y_data.astype(np.float32)

        train.to_csv("train.csv", index=False)
    else:
        if os.path.exists("train.csv"):
            train = pd.read_csv("train.csv")
        else:
            train = df.drop_duplicates(
                [
                    "eeg_id",
                    "seizure_vote",
                    "lpd_vote",
                    "gpd_vote",
                    "lrda_vote",
                    "grda_vote",
                    "other_vote",
                ]
            ).reset_index(drop=True)
            train["sign_id"] = train.index.values
            df["sign_id"] = df.index.values
            y_data = train[TARGETS].values
            train[TARGETS_RAW] = y_data
            y_data = y_data / (y_data.sum(axis=1, keepdims=True) + 1e-12)
            train[TARGETS] = y_data.astype(np.float32)
            train.to_csv("train.csv", index=False)



## === cell 2
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_eegs") + "/"
    if READ_EEG_FILES:
        raise RuntimeError(
            "READ_EEG_FILES=True path requires SciPy-based preprocessing; set READ_EEG_FILES=False."
        )
    else:
        if PLATFORM == "local":
            datapath = "./" + os.path.join("input", "preprocess")
        elif PLATFORM == "kaggle":
            datapath = "/kaggle/" + os.path.join("input", "preprocess")

        eegs = np.load(os.path.join(datapath, "eegs.npy"), allow_pickle=True).item()



## === cell 3
if NEEDTRAIN:
    PATH = os.path.join(LOAD_DATA_FROM, "train_spectrograms") + "/"
    files = os.listdir(PATH)
    print(f"There are {len(files)} spectrogram parquets")
    if READ_SPE_FILES:
        raise RuntimeError(
            "READ_SPE_FILES=True path is disabled here to keep runtime under limits."
        )
    else:
        if "spe" in DATATYPE:
            if PLATFORM == "local":
                spectrograms = np.load(
                    "./input/preprocess/spectrograms.npy", allow_pickle=True
                ).item()
            elif PLATFORM == "kaggle":
                spectrograms = np.load(
                    "/kaggle/input/preprocess/spectrograms.npy", allow_pickle=True
                ).item()




## === cell 4
def _normalize_probs(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, np.inf)
    s = np.sum(p, axis=1, keepdims=True)
    p = p / (s + eps)
    return p


def _train_prior_probs(train_df: pd.DataFrame, targets) -> np.ndarray:
    v = train_df[list(targets)].to_numpy(dtype=np.float64)
    v = np.clip(v, 0.0, np.inf)
    prior = v.sum(axis=0)
    prior = prior / (prior.sum() + 1e-12)
    return prior.astype(np.float32)


def _patient_prior_probs_by_eeg(
    train_df: pd.DataFrame, targets, alpha_smooth: float = 1.0
) -> pd.DataFrame:
    eeg_level = train_df.groupby(["patient_id", "eeg_id"], as_index=False)[
        list(targets)
    ].sum()
    g = eeg_level.groupby("patient_id")[list(targets)].sum()
    g = g.clip(lower=0.0).astype(np.float64)
    g = g + float(alpha_smooth)
    g = g.div(g.sum(axis=1), axis=0)
    return g.astype(np.float32)


def _patient_eeg_counts(train_df: pd.DataFrame) -> pd.Series:
    eeg_level = train_df[["patient_id", "eeg_id"]].drop_duplicates()
    return eeg_level.groupby("patient_id")["eeg_id"].nunique()


def _apply_dirichlet_floor(p: np.ndarray, floor: float = 0.002) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = p + float(floor)
    p = p / (np.sum(p, axis=1, keepdims=True) + 1e-12)
    p = np.clip(p, 0.0, np.inf)
    p = p / (np.sum(p, axis=1, keepdims=True) + 1e-12)
    return p


def _temperature_smooth(p: np.ndarray, t: float = 1.08) -> np.ndarray:
    p = np.asarray(p, dtype=np.float32)
    p = np.nan_to_num(p, nan=1e-12, posinf=1e-12, neginf=1e-12)
    p = np.clip(p, 1e-12, 1.0)
    p = np.power(p, 1.0 / float(t))
    return _normalize_probs(p)


def _expert_consensus_prior(train_df: pd.DataFrame, targets) -> np.ndarray:
    mapping = {
        "Seizure": "seizure_vote",
        "LPD": "lpd_vote",
        "GPD": "gpd_vote",
        "LRDA": "lrda_vote",
        "GRDA": "grda_vote",
        "Other": "other_vote",
    }
    prior = np.ones(len(targets), dtype=np.float64)
    if "expert_consensus" not in train_df.columns:
        prior = prior / prior.sum()
        return prior.astype(np.float32)

    vc = train_df["expert_consensus"].value_counts(dropna=True)
    for lab, cnt in vc.items():
        col = mapping.get(str(lab), None)
        if col in list(targets):
            j = list(targets).index(col)
            prior[j] += float(cnt)
        else:
            if "other_vote" in list(targets):
                j = list(targets).index("other_vote")
                prior[j] += float(cnt)
    prior = prior / (prior.sum() + 1e-12)
    return prior.astype(np.float32)


def _patient_expert_consensus_prior(train_df: pd.DataFrame, targets) -> pd.DataFrame:
    mapping = {
        "Seizure": "seizure_vote",
        "LPD": "lpd_vote",
        "GPD": "gpd_vote",
        "LRDA": "lrda_vote",
        "GRDA": "grda_vote",
        "Other": "other_vote",
    }
    idx = list(targets)
    if "expert_consensus" not in train_df.columns:
        return pd.DataFrame(columns=idx, dtype=np.float32)

    tmp = train_df[["patient_id", "expert_consensus"]].copy()
    tmp["mapped"] = (
        tmp["expert_consensus"].astype(str).map(mapping).fillna("other_vote")
    )
    counts = (
        tmp.groupby(["patient_id", "mapped"])
        .size()
        .unstack(fill_value=0)
        .astype(np.float64)
    )

    for c in idx:
        if c not in counts.columns:
            counts[c] = 0.0
    counts = counts[idx]

    counts = counts + 1.0  # Laplace smoothing
    counts = counts.div(counts.sum(axis=1), axis=0)
    return counts.astype(np.float32)


def _spectrogram_vector_from_parquet(path: str) -> np.ndarray:
    tmp = pd.read_parquet(path)
    arr = tmp.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.log1p(np.maximum(arr, 0.0))
    v = np.concatenate([arr.mean(axis=0), arr.std(axis=0)], axis=0)
    return v.astype(np.float32, copy=False)


def _build_spec_nn_priors(
    train_meta: pd.DataFrame,
    test_meta: pd.DataFrame,
    targets,
    train_spec_dir: str,
    test_spec_dir: str,
    n_train: int = 700,
    k: int = 9,
    seed: int = 2024,
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    train_spec = train_meta.groupby("spectrogram_id", as_index=False)[
        list(targets)
    ].sum()
    train_spec_ids = train_spec["spectrogram_id"].to_numpy()

    exists_mask = np.array(
        [
            os.path.exists(os.path.join(train_spec_dir, f"{int(sid)}.parquet"))
            for sid in train_spec_ids
        ],
        dtype=bool,
    )
    train_spec = train_spec.loc[exists_mask].reset_index(drop=True)

    if len(train_spec) == 0:
        return None

    n_train = int(max(1, min(n_train, len(train_spec))))
    train_idx = rng.choice(len(train_spec), size=n_train, replace=False)
    train_spec_small = train_spec.iloc[train_idx].reset_index(drop=True)

    Xtr_list = []
    for sid in train_spec_small["spectrogram_id"].to_numpy():
        v = _spectrogram_vector_from_parquet(
            os.path.join(train_spec_dir, f"{int(sid)}.parquet")
        )
        Xtr_list.append(v)
    Xtr = np.stack(Xtr_list, axis=0)
    Xtr = Xtr / (np.linalg.norm(Xtr, axis=1, keepdims=True) + 1e-12)

    Ytr = train_spec_small[list(targets)].to_numpy(dtype=np.float32)
    Ytr = _normalize_probs(Ytr)

    global_prior = _train_prior_probs(train_meta, targets)
    out = np.tile(global_prior.reshape(1, -1), (len(test_meta), 1)).astype(np.float32)

    test_spec_ids = test_meta["spectrogram_id"].to_numpy()
    for i, sid in enumerate(test_spec_ids):
        fpath = os.path.join(test_spec_dir, f"{int(sid)}.parquet")
        if not os.path.exists(fpath):
            continue
        v = _spectrogram_vector_from_parquet(fpath)
        v = v / (np.linalg.norm(v) + 1e-12)

        sims = Xtr @ v
        kk = min(int(k), sims.shape[0])
        nn = np.argpartition(-sims, kth=kk - 1)[:kk]
        w = np.maximum(sims[nn], 0.0)
        if float(w.sum()) <= 1e-12:
            continue
        w = (w / (w.sum() + 1e-12)).astype(np.float32)
        out[i] = (Ytr[nn] * w.reshape(-1, 1)).sum(axis=0).astype(np.float32)

    return _normalize_probs(out)




## === cell 5
def _need_tf_for_this_run(load_models_from: str) -> bool:
    if NEEDTRAIN:
        return True
    if not os.path.isdir(load_models_from):
        return False
    for model_i in range(100):
        wpath = os.path.join(load_models_from, f"fold{model_i}_stage2.weights.h5")
        if os.path.exists(wpath):
            return True
    return False


TF_AVAILABLE = _need_tf_for_this_run(LOAD_MODELS_FROM)
print("TF required for this run:", TF_AVAILABLE)

if TF_AVAILABLE:
    os.environ["KERAS_BACKEND"] = "tensorflow"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    import tensorflow as tf
    from tensorflow.keras.models import clone_model

    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    MIX = True
    if MIX:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)

    def _tf_butter_bandpass(low_hz: float, high_hz: float, fs: float, order: int = 3):
        return None

    def _tf_zero_phase_bandpass(
        x: np.ndarray, low_hz: float, high_hz: float, fs: float
    ):
        xt = tf.convert_to_tensor(x, dtype=tf.float32)
        T = tf.shape(xt)[1]
        X = tf.signal.rfft(xt)
        freqs = tf.linspace(0.0, fs / 2.0, tf.shape(X)[1])
        mask = tf.logical_and(freqs >= float(low_hz), freqs <= float(high_hz))
        mask = tf.cast(mask, X.dtype)
        Xf = X * mask[tf.newaxis, :]
        y = tf.signal.irfft(Xf, fft_length=[T])
        return y.numpy()




## === cell 6
if TF_AVAILABLE:

    class DataGenerator(tf.keras.utils.Sequence):
        def __init__(
            self,
            dataframe,
            batch_size=32,
            shuffle=False,
            sample_weights=False,
            mode="train",
            eegs=None,
            stfts=None,
            specs=None,
            imgs=None,
            stage=2,
        ):
            self.dataframe = dataframe
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.sample_weights = sample_weights
            self.mode = mode
            self.eegs = eegs
            self.stfts = stfts
            self.specs = specs
            self.imgs = imgs
            self.stage = stage
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.dataframe) / self.batch_size))

        def __getitem__(self, index):
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            x, y, sample_weights = self.__data_generation(indexes)
            return x, y, sample_weights

        def on_epoch_end(self):
            self.nan = 0
            self.indexes = np.arange(len(self.dataframe))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indexes):
            if "eeg" in DATATYPE:
                x_eeg = np.zeros(
                    (
                        len(indexes),
                        EEG_CHANNEL_USED * EEG_MULTIPLY,
                        round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                    ),
                    dtype="float32",
                )

            y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
            sample_weights = np.zeros((len(indexes), 1), dtype="float32")

            for j, i in enumerate(indexes):
                row = self.dataframe.iloc[i]
                if self.mode != "test":
                    sample_weight = float(np.sum(row[TARGETS_RAW].values)) / 20.0

                if self.mode == "test":
                    r_eeg = 0
                else:
                    rows = df.loc[
                        (df.eeg_id == row.eeg_id)
                        * (df.seizure_vote == row.seizure_vote_raw)
                        * (df.lpd_vote == row.lpd_vote_raw)
                        * (df.gpd_vote == row.gpd_vote_raw)
                        * (df.lrda_vote == row.lrda_vote_raw)
                        * (df.grda_vote == row.grda_vote_raw),
                        :,
                    ].reset_index(drop=True)
                    if self.mode == "train":
                        rows = rows.iloc[np.random.permutation(len(rows))].reset_index(
                            drop=True
                        )
                        row = rows.loc[0, :]
                    elif self.mode == "valid":
                        row = (
                            rows.sort_values(by="eeg_sub_id")
                            .reset_index(drop=True)
                            .iloc[len(rows) // 2]
                        )

                    r_eeg = row.eeg_label_offset_seconds
                    if self.mode == "train":
                        r_eeg = r_eeg + np.random.random() * 10 - 5
                        r_eeg = max(0, r_eeg)
                        r_eeg = min(r_eeg, self.eegs[row.eeg_id].shape[1] / RSFREQ - 50)

                if "eeg" in DATATYPE:
                    eeg = self.eegs[row.eeg_id][
                        :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                    ]
                    eeg = np.concatenate(
                        (
                            eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                            eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                        ),
                        axis=0,
                    )

                    eeg = eeg[
                        :,
                        round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                            (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                        ),
                    ]

                    if self.mode == "train":
                        if (self.stage == 2) and (np.random.rand() > 0):
                            eeg2 = eeg.copy()
                            eeg[4:8, :] = eeg2[12:16, :]
                            eeg[8:12, :] = eeg2[4:8, :]
                            eeg[12:16, :] = eeg2[8:12, :]
                        else:
                            if np.random.rand() > 0.5:
                                mask = round(np.random.rand() * eeg.shape[1])
                                eeg[
                                    :,
                                    mask : round(
                                        mask + np.random.rand() * eeg.shape[1] * 0.02
                                    ),
                                ] = 0
                            if np.random.rand() > 0.5:
                                mask = round(np.random.rand() * eeg.shape[1])
                                eeg[
                                    :,
                                    mask : round(
                                        mask + np.random.rand() * eeg.shape[1] * 0.02
                                    ),
                                ] = 0
                            if np.random.rand() > 0.5:
                                mask = round(np.random.rand() * eeg.shape[1])
                                eeg[
                                    :,
                                    mask : round(
                                        mask + np.random.rand() * eeg.shape[1] * 0.02
                                    ),
                                ] = 0

                            if np.random.rand() > 0.5:
                                eeg[np.random.permutation(eeg.shape[0])[0], :] = 0
                            if np.random.rand() > 0.5:
                                eeg[np.random.permutation(eeg.shape[0])[0], :] = 0

                            eeg[0 : round(EEG_CHANNEL_USED / 2), :] = eeg[
                                0 : round(EEG_CHANNEL_USED / 2), :
                            ][np.random.permutation(8), :]
                            eeg[-round(EEG_CHANNEL_USED / 2) :, :] = eeg[
                                -round(EEG_CHANNEL_USED / 2) :, :
                            ][np.random.permutation(8), :]

                            eeg2 = eeg.copy()
                            eeg[4:8, :] = eeg2[12:16, :]
                            eeg[8:12, :] = eeg2[4:8, :]
                            eeg[12:16, :] = eeg2[8:12, :]

                            if np.random.rand() > 0.5:
                                eeg = eeg[::-1, :]
                            if np.random.rand() > 0.5:
                                eeg = -eeg
                            if np.random.rand() > 0.5:
                                eeg = eeg[:, ::-1]
                    else:
                        eeg2 = eeg.copy()
                        eeg[4:8, :] = eeg2[12:16, :]
                        eeg[8:12, :] = eeg2[4:8, :]
                        eeg[12:16, :] = eeg2[8:12, :]

                    eeg = np.clip(eeg, a_min=-1024, a_max=1024)
                    eeg = eeg + 1024
                    eeg = eeg / 2048 * 255
                    x_eeg[j] = eeg

                if self.mode != "test":
                    yj = row[TARGETS].values.astype("float32")
                    yj = yj / (np.sum(yj) + 1e-12)
                    y[j] = yj
                    sample_weights[j] = sample_weight if self.sample_weights else 1.0

            x = {}
            if "eeg" in DATATYPE:
                x["eeg"] = x_eeg
            return x, y, sample_weights

    class IniToOne(tf.keras.initializers.Initializer):
        def __call__(self, shape, dtype=None):
            assert len(shape) == 3
            filter_length, input_channel, filter_count = shape
            kernel = np.zeros(shape, dtype=np.float32)
            for i in range(filter_count):
                kernel[i % filter_length, 0, i] = 1.0
            return tf.convert_to_tensor(kernel, dtype=dtype)

        def get_config(self):
            return {}

    class SumToOne(tf.keras.constraints.Constraint):
        def __call__(self, w):
            w = tf.abs(w)
            return w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)

        def get_config(self):
            return {}

    def build_model():
        inp = []
        y = 0

        if "eeg" in DATATYPE:
            inp_eeg = tf.keras.Input(
                shape=(
                    EEG_CHANNEL_USED * EEG_MULTIPLY,
                    round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                ),
                name="eeg",
            )
            x_eeg_raw = tf.keras.layers.Reshape(
                (inp_eeg.shape[1], inp_eeg.shape[2], 1)
            )(inp_eeg)

            strides = 10
            if PLATFORM == "local":
                eeg_embed = tf.keras.layers.Conv1D(
                    filters=strides * 3,
                    kernel_size=strides,
                    strides=strides,
                    padding="same",
                    use_bias=False,
                    activation=None,
                    kernel_initializer=IniToOne(),
                    kernel_constraint=SumToOne(),
                    input_shape=(None, 1),
                )
            else:
                eeg_embed = tf.keras.layers.Conv1D(
                    filters=strides * 3,
                    kernel_size=strides,
                    strides=strides,
                    padding="same",
                    use_bias=False,
                    activation=None,
                )

            x_eeg = tf.keras.layers.TimeDistributed(eeg_embed)(x_eeg_raw)

            x_eeg = tf.keras.layers.Concatenate(axis=-1)(
                [
                    tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                        x_eeg[:, :, :, 0 * strides : 1 * strides]
                    ),
                    tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                        x_eeg[:, :, :, 1 * strides : 2 * strides]
                    ),
                    tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1, 1))(
                        x_eeg[:, :, :, 2 * strides : 3 * strides]
                    ),
                ]
            )
            x_eeg = tf.keras.layers.Permute([4, 2, 1, 3])(x_eeg)
            x_eeg = tf.keras.layers.Reshape((x_eeg.shape[1], x_eeg.shape[2], -1))(x_eeg)
            x_eeg = tf.keras.layers.Permute((3, 2, 1))(x_eeg)

            base_model_eeg = tf.keras.applications.EfficientNetV2B3(
                include_top=False, weights=None, include_preprocessing=True
            )
            if NEEDTRAIN:
                if PLATFORM == "local":
                    base_model_eeg.load_weights(
                        f"./input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
                    )
                if PLATFORM == "kaggle":
                    base_model_eeg.load_weights(
                        f"/kaggle/input/pre-trained-weights/{base_model_eeg.name}_notop.h5"
                    )
            base_model_eeg.name = "eeg_extractor"

            x_eeg = base_model_eeg(x_eeg)
            x_eeg = x_eeg[
                :, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :
            ]
            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

            inp.append(inp_eeg)
            y = x_eeg * 1

        y = tf.keras.layers.Dense(len(TARGETS), activation="softmax", dtype="float32")(
            y
        )
        return tf.keras.Model(inputs=inp, outputs=y)




## === cell 7
if __name__ == "__main__":
    test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
    test["sign_id"] = test.index.values
    print("Test shape", test.shape)

    eeg_level_votes = df.groupby(["patient_id", "eeg_id"], as_index=False)[
        list(TARGETS)
    ].sum()

    global_prior_votes = _train_prior_probs(eeg_level_votes, TARGETS)
    global_prior_expert = _expert_consensus_prior(df, TARGETS)
    global_prior = _normalize_probs(
        (
            0.80 * global_prior_votes.reshape(1, -1)
            + 0.20 * global_prior_expert.reshape(1, -1)
        )
    )[0]

    patient_vote_priors = _patient_prior_probs_by_eeg(
        eeg_level_votes, TARGETS, alpha_smooth=1.0
    )
    patient_counts = _patient_eeg_counts(eeg_level_votes)
    patient_expert_priors = _patient_expert_consensus_prior(df, TARGETS)

    spec_dir = os.path.join(LOAD_DATA_FROM, "train_spectrograms")
    spec_test_dir = os.path.join(LOAD_DATA_FROM, "test_spectrograms")

    spec_nn_pred = None
    try:
        spec_nn_pred = _build_spec_nn_priors(
            df,
            test,
            TARGETS,
            train_spec_dir=spec_dir,
            test_spec_dir=spec_test_dir,
            n_train=700,
            k=9,
            seed=SEED,
        )
    except Exception as e:
        print("Spectrogram NN prior failed (continuing without it):", repr(e))
        spec_nn_pred = None

    def _make_patient_prior_predictions() -> np.ndarray:
        pred = np.zeros((len(test), len(TARGETS)), dtype=np.float32)
        for i, pid in enumerate(test["patient_id"].values):
            has_vote = pid in patient_vote_priors.index
            has_expert = pid in patient_expert_priors.index

            if has_vote:
                pv = patient_vote_priors.loc[pid].to_numpy(dtype=np.float32)
            if has_expert:
                pe = patient_expert_priors.loc[pid].to_numpy(dtype=np.float32)

            if has_vote and has_expert:
                p_patient = _normalize_probs(
                    (0.70 * pv.reshape(1, -1) + 0.30 * pe.reshape(1, -1))
                )[0]
            elif has_vote:
                p_patient = pv
            elif has_expert:
                p_patient = pe
            else:
                p_patient = global_prior

            n = int(patient_counts.get(pid, 0))
            alpha = float(n / (n + 2.0))
            pred[i] = alpha * p_patient + (1.0 - alpha) * global_prior

        pred = _normalize_probs(pred)
        if spec_nn_pred is not None:
            pred = _normalize_probs(0.80 * pred + 0.20 * spec_nn_pred)
        pred = _temperature_smooth(pred, t=1.06)
        pred = _apply_dirichlet_floor(pred, floor=0.0015)
        return _normalize_probs(pred)

    if not TF_AVAILABLE:
        pred = _make_patient_prior_predictions()

        sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
        pred_df = pd.DataFrame({"eeg_id": test["eeg_id"].values})
        pred_df[TARGETS] = pred.astype(np.float32)

        sub = sub[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")
        if sub[list(TARGETS)].isna().any().any():
            fill = np.tile(global_prior.reshape(1, -1), (len(sub), 1)).astype(
                np.float32
            )
            sub.loc[:, list(TARGETS)] = sub.loc[:, list(TARGETS)].to_numpy(np.float32)
            sub.loc[:, list(TARGETS)] = np.where(
                np.isnan(sub.loc[:, list(TARGETS)].to_numpy(np.float32)),
                fill,
                sub.loc[:, list(TARGETS)].to_numpy(np.float32),
            )

        sub[TARGETS] = _normalize_probs(sub[TARGETS].to_numpy(np.float32))
        sub.to_csv("submission.csv", index=False)
        print("Wrote fallback submission.csv", sub.shape)
        print(sub.head())
    else:
        preds_all = []
        models = []

        model_template = build_model()

        for model_i in range(100):
            wpath = os.path.join(LOAD_MODELS_FROM, f"fold{model_i}_stage2.weights.h5")
            if os.path.exists(wpath):
                print(f"Fold {model_i + 1}")
                model = clone_model(model_template)
                model.load_weights(wpath)
                models.append(model)

        if len(models) == 0:
            print(
                f"WARNING: No model weights found in {LOAD_MODELS_FROM}. "
                f"Writing a patient-aware + spectrogram-NN prior fallback submission."
            )
            pred = _make_patient_prior_predictions()

            sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))
            pred_df = pd.DataFrame({"eeg_id": test["eeg_id"].values})
            pred_df[TARGETS] = pred.astype(np.float32)
            sub = sub[["eeg_id"]].merge(pred_df, on="eeg_id", how="left")
            if sub[list(TARGETS)].isna().any().any():
                fill = np.tile(global_prior.reshape(1, -1), (len(sub), 1)).astype(
                    np.float32
                )
                sub.loc[:, list(TARGETS)] = sub.loc[:, list(TARGETS)].to_numpy(
                    np.float32
                )
                sub.loc[:, list(TARGETS)] = np.where(
                    np.isnan(sub.loc[:, list(TARGETS)].to_numpy(np.float32)),
                    fill,
                    sub.loc[:, list(TARGETS)].to_numpy(np.float32),
                )
            sub[TARGETS] = _normalize_probs(sub[TARGETS].to_numpy(np.float32))
            sub.to_csv("submission.csv", index=False)
            print("Submission shape", sub.shape)
            print(sub.head())
        else:
            PATH_test = os.path.join(LOAD_DATA_FROM, "test_eegs") + "/"

            batch_start = 0
            for i, eeg_id in enumerate(test.eeg_id):
                if i % 100 == 0:
                    print(i, ", ", end="")

                eeg_default = pd.read_parquet(
                    os.path.join(PATH_test, (str(eeg_id) + ".parquet"))
                )

                eeg = []
                for channel in BRAIN:
                    eeg_temp = (
                        eeg_default.loc[:, channel.split("-")[0]]
                        - eeg_default.loc[:, channel.split("-")[1]]
                    ).values
                    eeg_temp[np.isnan(eeg_temp)] = 0
                    eeg.append(np.reshape(eeg_temp, (1, -1)))
                eeg = np.concatenate(eeg, axis=0)

                eeg = np.clip(eeg, a_min=-1024, a_max=1024)
                eegshape = eeg.shape[1]
                eeg = np.concatenate((eeg[:, ::-1], eeg, eeg[:, ::-1]), axis=1)

                if filter_range is not None:
                    eeg = _tf_zero_phase_bandpass(
                        eeg,
                        low_hz=float(filter_range[0]),
                        high_hz=float(filter_range[1]),
                        fs=float(RSFREQ),
                    )

                eeg = eeg[:, eegshape : eegshape * 2]
                eeg = np.array(eeg, dtype=np.float32)

                if "eeg" in DATATYPE:
                    eegs_test[eeg_id] = eeg

                if ((i + 1) % TEST_BATCHSIZE == 0) or ((i + 1) == len(test.eeg_id)):
                    batch_end = i + 1
                    batch_df = test.iloc[batch_start:batch_end].reset_index(drop=True)

                    test_gen = DataGenerator(
                        batch_df,
                        shuffle=False,
                        sample_weights=False,
                        batch_size=TEST_BATCHSIZE,
                        mode="test",
                        specs=spectrograms_test,
                        eegs=eegs_test,
                        stfts=stfts_test,
                        imgs=imgs_test,
                        stage=2,
                    )

                    preds = []
                    for m in models:
                        pred = m.predict(test_gen, verbose=0)
                        preds.append(pred)

                    pred = np.mean(preds, axis=0)
                    pred = _normalize_probs(pred)

                    if len(preds_all) == 0:
                        preds_all = pred.copy()
                    else:
                        preds_all = np.concatenate((preds_all, pred), axis=0)

                    eegs_test = {}
                    stfts_test = {}
                    imgs_test = {}
                    gc.collect()

                    batch_start = batch_end

            preds_all = np.asarray(preds_all, dtype=np.float32)
            if preds_all.shape[0] != test.shape[0]:
                raise RuntimeError(
                    f"Prediction rows ({preds_all.shape[0]}) do not match test rows ({test.shape[0]})."
                )

            sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
            sub[TARGETS] = _normalize_probs(preds_all)
            sub.to_csv("submission.csv", index=False)
            print("Submission shape", sub.shape)
            print(sub.head())
