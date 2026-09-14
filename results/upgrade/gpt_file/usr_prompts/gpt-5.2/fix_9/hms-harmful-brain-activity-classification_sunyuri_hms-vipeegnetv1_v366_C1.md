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

0.2894050003040934

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I fix the TensorFlow/protobuf crash (`MessageFactory ... GetPrototype`) by avoiding TensorFlow entirely on Kaggle and switching the Kaggle path to a lightweight, deterministic fallback that still produces a valid submission CSV. Because your current run yields no score (no submission), the priority is to run end-to-end and write a correctly formatted `.csv` with probabilities summing to 1. The fallback uses the empirical class prior from `train.csv` (vote-normalized then averaged) and applies it to every test row; this is score-reasonable for KL and is safe/stable within the 600s limit. All original training/model code is preserved and only guarded so it won’t execute in the Kaggle environment where TF is broken.'
- What this solution (achieved 1.01346) has done: 'I remove the forced `SystemExit` that currently stops the notebook (and is why you see an “error”), while keeping the Kaggle-safe “no-TensorFlow” fallback path as the core runtime path. To move the KL score down toward your target (lower is better) with minimal logic change, I upgrade the fallback from a global class prior to a patient-conditioned prior computed from `train.csv` and used per `patient_id` in `test.csv` (with a global fallback for unseen patients). I also make the file-path selection robust to both `/kaggle/input/...` and `/kaggle/data/...` layouts and ensure the submission probabilities are clipped and row-normalized to exactly sum to 1. The output always be a valid `submission.csv` in the working directory.'
- What this solution (achieved 0.73988) has done: 'I remove the intentional `RuntimeError` that currently stops execution after writing `submission.csv`, replacing it with a clean, non-error early-exit that still prevents the broken TensorFlow section from running on Kaggle. To move your KL score downward (lower is better) toward the target with minimal semantic change, I keep the patient-conditioned prior approach but add a small, safe smoothing toward the global prior to reduce overconfident patient-specific priors (common KL failure mode on unseen/rare patients). I also make the data-root selection robust across `/kaggle/input/...` and `/kaggle/data/...` and enforce strict row-wise probability normalization/clipping so the submission always passes format checks. The rest of the original code (including the TensorFlow model/training) is preserved and only guarded so it won’t execute in the Kaggle runtime path.'
- What this solution (achieved 0.853) has done: 'I keep your Kaggle-safe fallback path (no TensorFlow) but improve its KL score by conditioning on both `patient_id` and `spectrogram_id` in a smoothed way, because test includes both identifiers and this usually captures more label distribution signal than patient-only priors. I also fix a likely label-column selection bug by explicitly using the known `*_vote` columns (instead of `df_train.columns[-6:]`, which can be wrong if column order changes), and I keep strict clipping + row-normalization so every row sums to 1 and the submission always validates. Finally, I replace the `sys.exit(0)` with a clean guard flag so the script runs end-to-end in Kaggle without raising `SystemExit` (which is currently shown as an “error”), while still preventing the TensorFlow section from executing.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri
email: syuri@tju.edu.cn
"""

import os
import warnings

warnings.filterwarnings("ignore")

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
    if os.path.exists("/kaggle/input/"):
        for dir_name in os.listdir("/kaggle/input/"):
            if dir_name[:6] == "models":
                LOAD_MODELS_FROM = dir_name
else:
    PLATFORM = "unknown"

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"
else:
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

_KAGGLE_FALLBACK_DONE = False
if os.path.exists("/kaggle") and (
    os.path.exists("/kaggle/input") or os.path.exists("/kaggle/data")
):
    import numpy as np
    import pandas as pd

    _candidates = [
        "/kaggle/input/hms-harmful-brain-activity-classification",
        "/kaggle/data/hms-harmful-brain-activity-classification",
        "/kaggle/input/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification",
        "/kaggle/data/hms-harmful-brain-activity-classification/hms-harmful-brain-activity-classification",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for _p in _candidates:
        if os.path.exists(os.path.join(_p, "train.csv")) and os.path.exists(
            os.path.join(_p, "test.csv")
        ):
            LOAD_DATA_FROM = _p
            break

    train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
    test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
    sample_path = os.path.join(LOAD_DATA_FROM, "sample_submission.csv")
    if not os.path.exists(sample_path):
        for _alt in [
            "/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv",
            "/kaggle/data/hms-harmful-brain-activity-classification/sample_submission.csv",
            "/kaggle/input/sample_submission.csv",
            "/kaggle/data/sample_submission.csv",
        ]:
            if os.path.exists(_alt):
                sample_path = _alt
                break

    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    df_sample = pd.read_csv(sample_path)

    TARGETS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

    eps = 1e-6

    votes = df_train[TARGETS].astype(np.float64).to_numpy()
    row_sum = votes.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    probs = votes / row_sum

    global_prior = probs.mean(axis=0)
    global_prior = np.clip(global_prior, eps, 1.0)
    global_prior = global_prior / global_prior.sum()

    df_probs = pd.DataFrame(probs, columns=TARGETS)
    df_probs["patient_id"] = df_train["patient_id"].values
    df_probs["spectrogram_id"] = df_train["spectrogram_id"].values
    df_probs["eeg_id"] = df_train["eeg_id"].values

    patient_prior = df_probs.groupby("patient_id", sort=False)[TARGETS].mean()
    spec_prior = df_probs.groupby("spectrogram_id", sort=False)[TARGETS].mean()
    eeg_prior = df_probs.groupby("eeg_id", sort=False)[TARGETS].mean()

    pair_prior = df_probs.groupby(["patient_id", "spectrogram_id"], sort=False)[
        TARGETS
    ].mean()

    def _smooth_by_count(prior_df: pd.DataFrame, counts_series: pd.Series, k: float):
        counts = counts_series.value_counts(dropna=False).astype(np.float64)
        w = (
            (counts / (counts + float(k)))
            .reindex(prior_df.index)
            .fillna(0.0)
            .to_numpy()[:, None]
        )
        gp = global_prior[None, :]
        arr = prior_df.to_numpy(dtype=np.float64)
        out = w * arr + (1.0 - w) * gp
        return pd.DataFrame(out, index=prior_df.index, columns=TARGETS)

    patient_prior = _smooth_by_count(patient_prior, df_train["patient_id"], k=25.0)
    spec_prior = _smooth_by_count(spec_prior, df_train["spectrogram_id"], k=60.0)
    eeg_prior = _smooth_by_count(eeg_prior, df_train["eeg_id"], k=80.0)

    pair_key = pd.Series(
        list(zip(df_train["patient_id"].values, df_train["spectrogram_id"].values))
    )
    pair_prior = _smooth_by_count(pair_prior, pair_key, k=30.0)

    tmp = df_test[["eeg_id", "patient_id", "spectrogram_id"]].copy()
    tmp["row_idx"] = np.arange(len(tmp), dtype=np.int64)

    tmp = tmp.merge(patient_prior.reset_index(), on="patient_id", how="left")
    tmp = tmp.merge(
        spec_prior.reset_index().rename(columns={c: f"{c}_spec" for c in TARGETS}),
        on="spectrogram_id",
        how="left",
    )
    tmp = tmp.merge(
        eeg_prior.reset_index().rename(columns={c: f"{c}_eeg" for c in TARGETS}),
        on="eeg_id",
        how="left",
    )

    pair_prior_reset = pair_prior.reset_index()
    pair_prior_reset["pair_key"] = (
        pair_prior_reset["patient_id"].astype(str)
        + "_"
        + pair_prior_reset["spectrogram_id"].astype(str)
    )
    pair_prior_reset = pair_prior_reset.drop(columns=["patient_id", "spectrogram_id"])
    pair_prior_reset = pair_prior_reset.rename(
        columns={c: f"{c}_pair" for c in TARGETS}
    )

    tmp["pair_key"] = (
        tmp["patient_id"].astype(str) + "_" + tmp["spectrogram_id"].astype(str)
    )
    tmp = tmp.merge(pair_prior_reset, on="pair_key", how="left")
    tmp = tmp.drop(columns=["pair_key"])

    tmp = tmp.sort_values("row_idx").reset_index(drop=True)

    pred_patient = tmp[TARGETS].to_numpy(dtype=np.float64)
    pred_spec = tmp[[f"{c}_spec" for c in TARGETS]].to_numpy(dtype=np.float64)
    pred_eeg = tmp[[f"{c}_eeg" for c in TARGETS]].to_numpy(dtype=np.float64)
    pred_pair = tmp[[f"{c}_pair" for c in TARGETS]].to_numpy(dtype=np.float64)

    def _fill_missing(a: np.ndarray) -> np.ndarray:
        m = np.isnan(a).any(axis=1)
        if m.any():
            a[m, :] = global_prior[None, :]
        return a

    pred_patient = _fill_missing(pred_patient)
    pred_spec = _fill_missing(pred_spec)
    pred_eeg = _fill_missing(pred_eeg)
    pred_pair = _fill_missing(pred_pair)

    W_PAIR = 0.25
    W_PATIENT = 0.40
    W_SPEC = 0.20
    W_EEG = 0.15
    pred = (
        W_PAIR * pred_pair
        + W_PATIENT * pred_patient
        + W_SPEC * pred_spec
        + W_EEG * pred_eeg
    )

    FINAL_SHRINK = 0.08
    pred = (1.0 - FINAL_SHRINK) * pred + FINAL_SHRINK * global_prior[None, :]

    pred = np.clip(pred, eps, 1.0)
    pred = pred / pred.sum(axis=1, keepdims=True)

    sub = pd.DataFrame({"eeg_id": tmp["eeg_id"].values})
    for k, col in enumerate(TARGETS):
        sub[col] = pred[:, k].astype(np.float32)

    sub = sub[["eeg_id"] + TARGETS]

    s = sub[TARGETS].sum(axis=1).to_numpy(dtype=np.float64)
    s[s == 0] = 1.0
    sub[TARGETS] = sub[TARGETS].div(s, axis=0)

    sub = df_sample[["eeg_id"]].merge(sub, on="eeg_id", how="left")
    if sub[TARGETS].isna().any().any():
        for c, v in zip(TARGETS, global_prior):
            sub[c] = sub[c].fillna(float(v))
        s = sub[TARGETS].sum(axis=1).to_numpy(dtype=np.float64)
        s[s == 0] = 1.0
        sub[TARGETS] = sub[TARGETS].div(s, axis=0)

    sub = sub[["eeg_id"] + TARGETS]
    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv", sub.shape)
    print(sub.head())

    _KAGGLE_FALLBACK_DONE = True

SKIP_TF = bool(_KAGGLE_FALLBACK_DONE)



## === cell 1
if not SKIP_TF:
    os.environ["KERAS_BACKEND"] = "tensorflow"

    SFREQ = 200  # EEG sampling rate
    RSFREQ = 200  # resampled EEG sampling rate

    EEG_LENGTH = 50  # the length of EEG data used for each sample
    EEG_LENGTH_USED = 50
    EEG_CHANNEL_USED = 16  # 16 18

    EEG_MULTIPLY = 1

    IMG_LENGTH = 20
    IMG_HIGH = 324
    IMG_WIDE = 324

    SPE_HIGH = 100  # the height of the spectrogram
    SPE_WIDE = 256  # the width of the spectrogram  10 * 30

    STFT_LENGTH = 45
    STFT_TIME = 0.15
    STFT_HIGH = 32  # the height of the STFT (eeg spectrogram)
    STFT_WIDE = round(
        STFT_LENGTH / STFT_TIME
    )  # the width of the STFT (eeg spectrogram)  50 * 5

    filter_range = [0.5, 45]  # eeg filtering range   [0.5, 45] None
    filter_range2 = [0.1, 35]  # eeg filtering range

    SEED = 2024  # seed

    BATCHSIZE = 16  # batch size

    LEARN_RATE = 1e-3
    EPOCHS = 15
    PATIENCE = 5

    SPLITS = 5

    READ_EEG_FILES = False  # preprocess eeg
    READ_SPE_FILES = False  # preprocess spectrogram

    spectrograms = {}  # preprocessed spectrograms for training
    eegs = {}  # preprocessed eegs for training
    stfts = {}  # preprocessed short-time fourier transform plots for training
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

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
    import io
    from PIL import Image
    import pandas as pd, numpy as np
    from sklearn.metrics import confusion_matrix

    from tensorflow.keras import optimizers
    from tensorflow.keras.models import clone_model
    from tensorflow.python.framework.ops import reset_default_graph

    import matplotlib
    import matplotlib.pyplot as plt

    from scipy import signal
    import time
    import gc

    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    import tensorflow as tf

    print(tf.version.VERSION)
    print(tf.config.list_physical_devices("GPU"))
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)  # 按需分配显存
        except RuntimeError as e:
            print(e)

    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    MIX = True
    if MIX:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
    else:
        print("Using full precision")

    if NEEDTRAIN:
        import itertools

    df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
    TARGETS = df.columns[-6:]
    print("Train shape:", df.shape)
    print("Targets", list(TARGETS))



## === cell 2
if not SKIP_TF:
    if NEEDTRAIN:
        TARGETS_RAW = list()
        for i in TARGETS:
            TARGETS_RAW.append(i + "_raw")

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
            y_data = y_data / y_data.sum(axis=1, keepdims=True)
            train[TARGETS] = y_data

            train.to_csv("train.csv", index=False)
        else:
            train = pd.read_csv("train.csv")

    if filter_range != None:
        from scipy import signal

        b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")



## === cell 3
if not SKIP_TF:
    if NEEDTRAIN:
        PATH = os.path.join(LOAD_DATA_FROM, "train_eegs") + "/"
        if READ_EEG_FILES:
            if ("stft" in DATATYPE) or ("img" in DATATYPE):
                from scipy import signal

                b2, a2 = signal.butter(
                    3, np.float32(filter_range2) * 2 / RSFREQ, "bandpass"
                )
            import time, gc
            import pandas as pd, numpy as np

            time_start_time = time.time()

            for i, eeg_id in enumerate(train.eeg_id.unique()):

                if i % 200 == 0:
                    gc.collect()
                    xx = time.time() - time_start_time
                    yy = xx / (i + 1) * len(train.eeg_id.unique())
                    print(i, f"time: {round(xx / 60, 2)} min / {round(yy / 60, 2)} min")
                eeg_default = pd.read_parquet(
                    os.path.join(PATH, (str(eeg_id) + ".parquet"))
                )

                eeg = list()
                for channel in BRAIN:
                    eeg_temp = (
                        eeg_default.loc[:, channel.split("-")[0]]
                        - eeg_default.loc[:, channel.split("-")[1]]
                    ).values
                    eeg_temp[np.isnan(eeg_temp)] = 0
                    eeg.append(np.reshape(eeg_temp, (1, -1)))
                eeg = np.concatenate(eeg, axis=0)

                if SFREQ != RSFREQ:
                    from scipy import signal

                    eeg = signal.resample_poly(eeg, RSFREQ, SFREQ, axis=1)

                if "stft" in DATATYPE:

                    from scipy import signal

                    eeg2 = signal.filtfilt(b, a, eeg, axis=1)
                    nperseg = round(RSFREQ * 0.2)
                    ff, tt, ss = signal.spectrogram(
                        eeg2,
                        axis=1,
                        fs=RSFREQ,
                        nperseg=nperseg,
                        noverlap=round(nperseg - RSFREQ * STFT_TIME),
                        nfft=320,
                    )
                    ss[np.isnan(ss)] = 0
                    ss = ss[:, (ff > 0) * (ff <= 20), :]

                if "img" in DATATYPE:
                    from scipy import signal

                    eeg2 = signal.filtfilt(b2, a2, eeg, axis=1)

                    eeg2 = np.clip(eeg2, a_min=-1024, a_max=1024)

                    train_plot = train[train.eeg_id == eeg_id].reset_index(drop=True)
                    for j in range(len(train_plot)):
                        row = train_plot.iloc[j]
                        rows = df.loc[
                            (df.eeg_id == row.eeg_id)
                            * (df.seizure_vote == row.seizure_vote_raw)
                            * (df.lpd_vote == row.lpd_vote_raw)
                            * (df.gpd_vote == row.gpd_vote_raw)
                            * (df.lrda_vote == row.lrda_vote_raw)
                            * (df.grda_vote == row.grda_vote_raw),
                            :,
                        ].reset_index(drop=True)
                        row = (
                            rows.sort_values(by="eeg_sub_id")
                            .reset_index(drop=True)
                            .iloc[len(rows) // 2]
                        )

                        eeg_plot = eeg2[
                            :,
                            round(row.eeg_label_offset_seconds * RSFREQ) : round(
                                (row.eeg_label_offset_seconds + EEG_LENGTH) * RSFREQ
                            ),
                        ]
                        eeg_plot = eeg_plot[
                            :,
                            round((EEG_LENGTH - IMG_LENGTH) / 2 * RSFREQ) : round(
                                (EEG_LENGTH + IMG_LENGTH) / 2 * RSFREQ
                            ),
                        ]

                        img_save = np.zeros(
                            (eeg_plot.shape[0], 36, IMG_WIDE), dtype=np.float32
                        )
                        for ii in range(eeg_plot.shape[0]):
                            fig = plt.figure(clear=True, figsize=(3.93, 2 / 18 * 2))
                            fig.patch.set_facecolor("black")

                            plt.plot(eeg_plot[ii, :] + 100, color="red", linewidth=0.2)

                            plt.xlim(-5, eeg_plot.shape[1] + 5)
                            plt.ylim(0, 200)
                            plt.axis("off")

                            byte_stream = io.BytesIO()
                            plt.savefig(
                                byte_stream, format="png", bbox_inches="tight", dpi=100
                            )
                            byte_stream.seek(0)
                            img = Image.open(byte_stream)
                            img = np.array(img)[:, :, :1]
                            img = img / 255
                            img = np.array(img, dtype=np.float32)
                            byte_stream.truncate()
                            plt.close("all")

                            if img.shape != (36, IMG_WIDE, 1):
                                img = np.concatenate((img, img, img), 2)
                                img = np.array(
                                    tf.image.resize(img, (36, IMG_WIDE)),
                                    dtype=np.float32,
                                )
                            img = img[:, :, 0]

                            img_save[ii, :, :] = img

                        imgs[train_plot.sign_id[j]] = img_save

                eeg = np.clip(eeg, a_min=-1024, a_max=1024)
                if filter_range != None:
                    from scipy import signal

                    eeg = signal.filtfilt(b, a, eeg, axis=1)
                eeg = np.array(eeg, dtype=np.float32)

                if "eeg" in DATATYPE:
                    eegs[eeg_id] = eeg
                if "stft" in DATATYPE:
                    stfts[eeg_id] = ss
                    stfts[-eeg_id] = tt

            if not os.path.exists("./input/preprocess"):
                os.makedirs("./input/preprocess")
            if "eeg" in DATATYPE:
                np.save("./input/preprocess/eegs.npy", eegs, allow_pickle=True)
            if "stft" in DATATYPE:
                np.save("./input/preprocess/stfts.npy", stfts, allow_pickle=True)
            if "img" in DATATYPE:
                np.save("./input/preprocess/imgs.npy", imgs, allow_pickle=True)

        else:
            if PLATFORM == "local":
                datapath = "./" + os.path.join("input", "preprocess")
            elif PLATFORM == "kaggle":
                datapath = "/kaggle/" + os.path.join("input", "preprocess")

            if "eeg" in DATATYPE:
                eegs = np.load(
                    os.path.join(datapath, "eegs.npy"), allow_pickle=True
                ).item()
            if "stft" in DATATYPE:
                stfts = np.load(
                    os.path.join(datapath, "stfts.npy"), allow_pickle=True
                ).item()
            if "img" in DATATYPE:
                imgs = np.load(
                    os.path.join(datapath, "imgs.npy"), allow_pickle=True
                ).item()



## === cell 4
if not SKIP_TF:
    if NEEDTRAIN:
        PATH = os.path.join(LOAD_DATA_FROM, "train_spectrograms") + "/"
        files = os.listdir(PATH)
        print(f"There are {len(files)} spectrogram parquets")
        import time, gc
        import pandas as pd, numpy as np

        time_start_time = time.time()
        if READ_SPE_FILES:
            for i, f in enumerate(files):
                if i % 200 == 0:
                    gc.collect()
                    xx = time.time() - time_start_time
                    yy = xx / (i + 1) * len(files)
                    print(i, f"time: {round(xx / 60, 2)} min / {round(yy / 60, 2)} min")
                tmp = pd.read_parquet(f"{PATH}{f}")
                name = int(f.split(".")[0])
                spectrograms[name] = tmp.iloc[:, 1:].values
            if not os.path.exists("./input/preprocess"):
                os.makedirs("./input/preprocess")
            np.save(
                "./input/preprocess/spectrograms.npy", spectrograms, allow_pickle=True
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



## === cell 5
if not SKIP_TF:

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
            self.on_epoch_end()

        def __len__(self):
            ct = int(np.ceil(len(self.dataframe) / self.batch_size))
            return ct

        def __getitem__(self, index):
            indexes = self.indexes[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            x, y, sample_weights = self.__data_generation(indexes)
            return x, y, sample_weights

        def on_epoch_end(self):
            self.indexes = np.arange(len(self.dataframe))
            if self.shuffle:
                np.random.shuffle(self.indexes)

        def __data_generation(self, indexes):
            if "spe" in DATATYPE:
                x_spe = np.zeros((len(indexes), 4, SPE_HIGH, SPE_WIDE), dtype="float32")
            if "eeg" in DATATYPE:
                x_eeg = np.zeros(
                    (
                        len(indexes),
                        EEG_CHANNEL_USED * EEG_MULTIPLY,
                        round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
                    ),
                    dtype="float32",
                )
            if "img" in DATATYPE:
                x_img = np.zeros((len(indexes), IMG_HIGH, IMG_WIDE, 3), dtype="float32")

            y = np.zeros((len(indexes), len(TARGETS)), dtype="float32")
            sample_weights = np.zeros((len(indexes), 1), dtype="float32")

            targets_batch = list()

            for j, i in enumerate(indexes):
                row = self.dataframe.iloc[i]
                sign_id = row.sign_id
                if self.mode != "test":
                    sample_weight = sum(row[TARGETS_RAW].values) / 20
                    targets_batch.append(row.expert_consensus)

                if self.mode == "test":
                    r_spe = 0
                    r_eeg = 0
                    r_stft = 0
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
                    r_spe = round(row.spectrogram_label_offset_seconds / 2)
                    r_eeg = row.eeg_label_offset_seconds

                if "spe" in DATATYPE:
                    spe = list()  # LL RL LP RP
                    for k in range(4):
                        spe.append(
                            np.reshape(
                                self.specs[row.spectrogram_id][
                                    r_spe : (r_spe + 300), k * 100 : (k + 1) * 100
                                ].T,
                                (1, 100, 300),
                            )
                        )
                    spe = np.concatenate(spe, axis=0)

                if "eeg" in DATATYPE:
                    eeg = self.eegs[row.eeg_id][
                        :, round(r_eeg * RSFREQ) : round((r_eeg + 50) * RSFREQ)
                    ]

                if "stft" in DATATYPE:
                    stft_t = self.stfts[-row.eeg_id]
                    r_stft = (np.where(stft_t >= (r_eeg - min(stft_t))))[0][0]
                    r_stft2 = (np.where(stft_t <= (50 + r_eeg - min(stft_t))))[0][-1]
                    stft = self.stfts[row.eeg_id][:, :, r_stft:r_stft2]
                    stft = stft[
                        :,
                        :,
                        round((50 - STFT_LENGTH) / 2 / STFT_TIME) : round(
                            (50 + STFT_LENGTH) / 2 / STFT_TIME
                        ),
                    ]
                    if stft.shape[2] < STFT_WIDE:
                        stft = np.concatenate((stft, stft[:, :, ::-1]), 2)
                        stft = stft[:, :, :STFT_WIDE]

                if "img" in DATATYPE:
                    img = self.imgs[sign_id]

                if "spe" in DATATYPE:
                    spe[np.isnan(spe)] = 0
                    exp_min, exp_max = -4, 6
                    spe = np.clip(spe, a_min=np.exp(exp_min), a_max=np.exp(exp_max))
                    spe = np.log(spe)

                    spe = spe[
                        :,
                        :,
                        round((spe.shape[2] - SPE_WIDE) / 2) : -round(
                            (spe.shape[2] - SPE_WIDE) / 2
                        ),
                    ]

                    if self.mode == "train":
                        spe2 = spe.copy()
                        if np.random.rand() > 0.5:
                            spe[0] = spe2[2]
                            spe[2] = spe2[0]
                        if np.random.rand() > 0.5:
                            spe[1] = spe2[3]
                            spe[3] = spe2[1]
                        if np.random.rand() > 0.5:
                            spe[0] = spe2[1]
                            spe[2] = spe2[3]
                            spe[1] = spe2[0]
                            spe[3] = spe[2]

                    spe = (spe - exp_min) / (exp_max - exp_min) * 255
                    spe = np.clip(spe, a_min=0, a_max=255)
                    x_spe[j] = spe

                if "eeg" in DATATYPE:
                    eeg = eeg[
                        :,
                        round((EEG_LENGTH - EEG_LENGTH_USED) * RSFREQ / 2) : round(
                            (EEG_LENGTH + EEG_LENGTH_USED) * RSFREQ / 2
                        ),
                    ]
                    eeg_save = np.zeros(
                        (x_eeg.shape[1], x_eeg.shape[2]), dtype=np.float32
                    )

                    eeg = np.concatenate(
                        (
                            eeg[0 : round(EEG_CHANNEL_USED / 2), :],
                            eeg[-round(EEG_CHANNEL_USED / 2) :, :],
                        ),
                        axis=0,
                    )

                    if self.mode == "train":
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

                        for ii in range(eeg_save.shape[0]):
                            eeg_save[ii, :] = eeg[
                                ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                            ]
                    else:
                        eeg2 = eeg.copy()
                        eeg[4:8, :] = eeg2[12:16, :]
                        eeg[8:12, :] = eeg2[4:8, :]
                        eeg[12:16, :] = eeg2[8:12, :]
                        for ii in range(eeg_save.shape[0]):
                            eeg_save[ii, :] = eeg[
                                ii // EEG_MULTIPLY, ii % EEG_MULTIPLY :: EEG_MULTIPLY
                            ]

                    eeg = np.clip(eeg_save, a_min=-255, a_max=255)
                    eeg = eeg + 255
                    eeg = eeg / 2
                    x_eeg[j] = eeg

                if "stft" in DATATYPE:
                    exp_min, exp_max = -6, 6
                    stft = np.clip(stft, a_min=np.exp(exp_min), a_max=np.exp(exp_max))
                    stft = np.log(stft)

                    stft = np.concatenate(
                        (
                            stft[0 : round(EEG_CHANNEL_USED / 2), :, :],
                            stft[-round(EEG_CHANNEL_USED / 2) :, :, :],
                        ),
                        axis=0,
                    )

                    if self.mode == "train":
                        stft[0 : round(EEG_CHANNEL_USED / 2), :] = stft[
                            0 : round(EEG_CHANNEL_USED / 2), :
                        ][np.random.permutation(8), :]
                        stft[-round(EEG_CHANNEL_USED / 2) :, :] = stft[
                            -round(EEG_CHANNEL_USED / 2) :, :
                        ][np.random.permutation(8), :]
                        stft2 = stft.copy()
                        stft[4:8, :] = stft2[12:16, :]
                        stft[8:12, :] = stft2[4:8, :]
                        stft[12:16, :] = stft2[8:12, :]

                        if np.random.rand() > 0.5:
                            stft = stft[::-1, :]
                    else:
                        stft2 = stft.copy()
                        stft[4:8, :] = stft2[12:16, :]
                        stft[8:12, :] = stft2[4:8, :]
                        stft[12:16, :] = stft2[8:12, :]

                    stft_save = np.zeros(
                        (stft.shape[0] * stft.shape[1] // 2, stft.shape[2] * 2),
                        dtype=np.float32,
                    )
                    for ii in range(stft.shape[0]):
                        stft_save[
                            ii // 2 * stft.shape[1] : (ii // 2 + 1) * stft.shape[1],
                            ii % 2 :: 2,
                        ] = stft[ii, :, :]

                    stft = (stft_save - exp_min) / (exp_max - exp_min) * 255
                    stft = np.clip(stft, a_min=0, a_max=255)

                    if self.mode == "train":
                        if np.random.rand() > 0.5:
                            mask = round(np.random.rand() * stft.shape[1])
                            stft[
                                :,
                                mask : round(
                                    mask + np.random.rand() * stft.shape[1] * 0.02
                                ),
                            ] = 0
                        if np.random.rand() > 0.5:
                            mask = round(np.random.rand() * stft.shape[1])
                            stft[
                                :,
                                mask : round(
                                    mask + np.random.rand() * stft.shape[1] * 0.02
                                ),
                            ] = 0
                        if np.random.rand() > 0.5:
                            mask = round(np.random.rand() * stft.shape[1])
                            stft[
                                :,
                                mask : round(
                                    mask + np.random.rand() * stft.shape[1] * 0.02
                                ),
                            ] = 0

                    if "stft" in DATATYPE:
                        if j == 0:
                            x_stft = np.zeros(
                                (len(indexes), stft.shape[0], stft.shape[1]),
                                dtype="float32",
                            )
                            x_stft[j] = stft
                        else:
                            x_stft[j] = stft

                if "img" in DATATYPE:
                    img_save = np.zeros((IMG_HIGH, IMG_WIDE), dtype=np.float32)

                    if self.mode == "train":
                        img[0:8, :, :] = img[0:8, :, :][np.random.permutation(8), :, :]
                        img[10:18, :, :] = img[10:18, :, :][
                            np.random.permutation(8), :, :
                        ]
                        if np.random.rand() > 0.5:
                            img = img[::-1, :, :]

                    for ii in range(img.shape[0]):
                        axis_temp = img_save.shape[1] / img.shape[0] / 2 * (2 * ii + 1)
                        start_temp = round(
                            max(axis_temp - img_save.shape[1] / img.shape[0], 0)
                        )
                        end_temp = round(
                            min(
                                img_save.shape[0],
                                axis_temp + img_save.shape[1] / img.shape[0],
                            )
                        )
                        temp_temp = round(img.shape[1] / 2 - (axis_temp - start_temp))
                        img_save[start_temp:end_temp, :] = (
                            img_save[start_temp:end_temp, :]
                            + img[
                                ii,
                                temp_temp : round(temp_temp + end_temp - start_temp),
                                :,
                            ]
                        )
                    img_save = np.clip(img_save, a_min=0, a_max=1)

                    img = np.reshape(
                        img_save, (img_save.shape[0], img_save.shape[1], 1)
                    )
                    img = np.concatenate((img, img, img), -1)

                    img = (img - np.mean(img)) / (np.std(img) + 1e-6)
                    x_img[j] = img

                if self.mode != "test":
                    y[j] = row[TARGETS].values / sum(row[TARGETS].values)
                    if self.sample_weights:
                        sample_weights[j] = sample_weight
                    else:
                        sample_weights[j] = 1

            x = {}
            if "spe" in DATATYPE:
                x["spe"] = x_spe
            if "eeg" in DATATYPE:
                x["eeg"] = x_eeg
            if "stft" in DATATYPE:
                x["stft"] = x_stft
            if "img" in DATATYPE:
                x["img"] = x_img

            return x, y, sample_weights




## === cell 6
if not SKIP_TF:

    class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
        def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
            super(CosineAnnealingLRScheduler, self).__init__()
            self.total_step = total_step

            if warmth_rate == 0:
                self.warm_step = 1
            else:
                self.warm_step = int(warmth_rate)

            self.lr_max = lr_max
            self.lr_min = lr_min

        def __call__(self, step):
            step = step + 1
            if step < self.warm_step:
                lr = self.lr_max / self.warm_step * step
            else:
                if self.total_step == 1:
                    lr = self.lr_max
                else:
                    lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (
                        1.0
                        + tf.cos(
                            (step - self.warm_step)
                            / (self.total_step - self.warm_step)
                            * np.pi
                        )
                    )
            return np.float32(lr)

    class IniToOne(tf.keras.initializers.Initializer):
        def __init__(self):
            super(IniToOne, self).__init__()

        def __call__(self, shape, dtype=None):
            assert len(shape) == 3
            filter_length, input_channel, filter_count = shape
            kernel = np.zeros(shape, dtype=np.float32)
            for i in range(filter_count):
                kernel[i % filter_length, 0, i] = 1.0
            kernel = tf.convert_to_tensor(kernel, dtype=dtype)
            return kernel

        def get_config(self):
            return {}

    class SumToOne(tf.keras.constraints.Constraint):
        def __init__(self):
            super(SumToOne, self).__init__()

        def __call__(self, w):
            w = tf.abs(w)
            w_normed = w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)
            return w_normed

        def get_config(self):
            return {}

    class IniToOneAtten(tf.keras.initializers.Initializer):
        def __init__(self):
            super(IniToOneAtten, self).__init__()

        def __call__(self, shape, dtype=None):
            assert len(shape) == 3
            filter_length, input_channel, filter_count = shape
            kernel = np.zeros(shape, dtype=np.float32)
            kernel[(filter_length - 1) // 2 : (filter_length) // 2 + 1, :, :] = 1 / (
                (filter_length) // 2 + 1 - (filter_length - 1) // 2
            )
            kernel = tf.convert_to_tensor(kernel, dtype=dtype)
            return kernel

        def get_config(self):
            return {}

    class SumToOneAtten(tf.keras.constraints.Constraint):
        def __init__(self):
            super(SumToOneAtten, self).__init__()

        def __call__(self, w):
            w = tf.abs(w)
            w_normed = w / tf.reduce_sum(w, axis=[0, 1], keepdims=True)
            return w_normed

        def get_config(self):
            return {}

    class TransformerBlock(tf.keras.layers.Layer):
        def __init__(self, embed_dim, feat_dim, num_heads, ff_dim, rate=0.1):
            super(TransformerBlock, self).__init__()
            self.att = tf.keras.layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim
            )
            self.ffn = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(ff_dim, activation="gelu"),
                    tf.keras.layers.Dense(feat_dim),
                ]
            )
            self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
            self.dropout1 = tf.keras.layers.Dropout(rate)
            self.dropout2 = tf.keras.layers.Dropout(rate)

        def call(self, inputs, training):
            attn_output, weights = self.att(
                inputs, inputs, return_attention_scores=True
            )
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(inputs + attn_output)
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.layernorm2(out1 + ffn_output), weights

    class ClassToken(tf.keras.layers.Layer):
        """Append a class token to an input layer."""

        def build(self, input_shape):
            cls_init = tf.zeros_initializer()
            self.hidden_size = input_shape[-1]
            self.cls = tf.Variable(
                name="cls",
                initial_value=cls_init(shape=(1, 1, self.hidden_size), dtype="float32"),
                trainable=True,
            )

        def call(self, inputs):
            batch_size = tf.shape(inputs)[0]
            cls_broadcasted = tf.cast(
                tf.broadcast_to(self.cls, [batch_size, 1, self.hidden_size]),
                dtype=inputs.dtype,
            )
            return tf.concat([cls_broadcasted, inputs], 1)




## === cell 7
if not SKIP_TF:

    def build_model():
        inp = list()
        if "spe" in DATATYPE:
            inp_spe = tf.keras.Input(shape=(4, SPE_HIGH, SPE_WIDE), name="spe")
            x_spe = tf.keras.layers.Reshape(
                (inp_spe.shape[1], inp_spe.shape[2], inp_spe.shape[3], 1)
            )(inp_spe)
            x_spe = tf.keras.layers.Concatenate(axis=-1)([x_spe, x_spe, x_spe])

            base_model_spe = tf.keras.applications.EfficientNetV2B0(
                include_top=False, weights=None, include_preprocessing=True
            )
            if NEEDTRAIN:
                if PLATFORM == "local":
                    base_model_spe.load_weights(
                        f"./input/pre-trained-weights/{base_model_spe.name}_notop.h5"
                    )
                if PLATFORM == "kaggle":
                    base_model_spe.load_weights(
                        f"/kaggle/input/pre-trained-weights/{base_model_spe.name}_notop.h5"
                    )
            base_model_spe._name = "spe_extractor"

            base_model_spe_pre = tf.keras.Model(
                base_model_spe.input, base_model_spe.get_layer("block3b_add").output
            )
            base_model_spe_pre._name = "spe_extractor_pre"
            x_spe1 = base_model_spe_pre(x_spe[:, 0, :, :, :])
            x_spe2 = base_model_spe_pre(x_spe[:, 1, :, :, :])
            x_spe3 = base_model_spe_pre(x_spe[:, 2, :, :, :])
            x_spe4 = base_model_spe_pre(x_spe[:, 3, :, :, :])

            x_spe = tf.keras.layers.Concatenate(axis=1)(
                [x_spe1, x_spe2, x_spe3, x_spe4]
            )
            base_model_spe_after = tf.keras.Model(
                base_model_spe_pre.output, base_model_spe.output
            )
            base_model_spe_after._name = "spe_extractor_after"
            x_spe = base_model_spe_after(x_spe)

            x_spe = x_spe[
                :, :, (x_spe.shape[2] - 1) // 2 : (x_spe.shape[2]) // 2 + 1, :
            ]
            x_spe = tf.keras.layers.GlobalAveragePooling2D()(x_spe)
            x_spe = tf.keras.layers.Dropout(0.9)(x_spe)
            inp.append(inp_spe)
            y_spe = tf.keras.layers.Dense(
                len(TARGETS), activation="softmax", dtype="float32"
            )(x_spe)

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

            base_model_eeg = tf.keras.applications.EfficientNetV2M(
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
            base_model_eeg._name = "eeg_extractor"

            x_eeg = base_model_eeg(x_eeg)

            x_eeg = x_eeg[
                :, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :
            ]
            x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
            x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)

            inp.append(inp_eeg)
            y_eeg = tf.keras.layers.Dense(
                len(TARGETS), activation="softmax", dtype="float32"
            )(x_eeg)

        if "stft" in DATATYPE:
            inp_stft = tf.keras.Input(
                shape=(STFT_HIGH * EEG_CHANNEL_USED // 2, STFT_WIDE * 2), name="stft"
            )
            x_stft = tf.keras.layers.Reshape((inp_stft.shape[1], inp_stft.shape[2], 1))(
                inp_stft
            )
            x_stft = tf.keras.layers.Concatenate(axis=-1)([x_stft, x_stft, x_stft])

            base_model_stft = tf.keras.applications.EfficientNetV2B0(
                include_top=False, weights=None, include_preprocessing=True
            )
            if NEEDTRAIN:
                if PLATFORM == "local":
                    base_model_stft.load_weights(
                        f"./input/pre-trained-weights/{base_model_stft.name}_notop.h5"
                    )
                if PLATFORM == "kaggle":
                    base_model_stft.load_weights(
                        f"/kaggle/input/pre-trained-weights/{base_model_stft.name}_notop.h5"
                    )
            base_model_stft._name = "stft_extractor"

            base_model_stft_pre = tf.keras.Model(
                base_model_stft.input, base_model_stft.get_layer("block3b_add").output
            )
            base_model_stft_pre._name = "stft_extractor_pre"
            x_stft1 = base_model_stft_pre(x_stft[:, 0 * 32 : 1 * 32, :, :])
            x_stft2 = base_model_stft_pre(x_stft[:, 1 * 32 : 2 * 32, :, :])
            x_stft3 = base_model_stft_pre(x_stft[:, 2 * 32 : 3 * 32, :, :])
            x_stft4 = base_model_stft_pre(x_stft[:, 3 * 32 : 4 * 32, :, :])
            x_stft5 = base_model_stft_pre(x_stft[:, 4 * 32 : 5 * 32, :, :])
            x_stft6 = base_model_stft_pre(x_stft[:, 5 * 32 : 6 * 32, :, :])
            x_stft7 = base_model_stft_pre(x_stft[:, 6 * 32 : 7 * 32, :, :])
            x_stft8 = base_model_stft_pre(x_stft[:, 7 * 32 : 8 * 32, :, :])
            x_stft = tf.keras.layers.Concatenate(axis=1)(
                [x_stft1, x_stft2, x_stft3, x_stft4, x_stft5, x_stft6, x_stft7, x_stft8]
            )
            base_model_stft_after = tf.keras.Model(
                base_model_stft_pre.output, base_model_stft.output
            )
            base_model_stft_after._name = "stft_extractor_after"
            x_stft = base_model_stft_after(x_stft)

            x_stft = x_stft[
                :, :, (x_stft.shape[2] - 1) // 2 : (x_stft.shape[2]) // 2 + 1, :
            ]
            x_stft = tf.keras.layers.GlobalAveragePooling2D()(x_stft)
            x_stft = tf.keras.layers.Dropout(0.2)(x_stft)

            inp.append(inp_stft)
            y_stft = tf.keras.layers.Dense(
                len(TARGETS), activation="softmax", dtype="float32"
            )(x_stft)

        if "img" in DATATYPE:
            inp_img = tf.keras.Input(shape=(IMG_HIGH, IMG_WIDE, 3), name="img")

            base_model_img = tf.keras.applications.EfficientNetB0(
                include_top=False, weights=None, input_tensor=inp_img
            )
            if NEEDTRAIN:
                if PLATFORM == "local":
                    base_model_img.load_weights(
                        f"./input/pre-trained-weights/{base_model_img.name}_notop.h5"
                    )
                if PLATFORM == "kaggle":
                    base_model_img.load_weights(
                        f"/kaggle/input/pre-trained-weights/{base_model_img.name}_notop.h5"
                    )
            base_model_img._name = "img_extractor"
            x_img = base_model_img.output
            x_img = tf.keras.layers.GlobalAveragePooling2D()(x_img)
            inp.append(inp_img)
            y_img = tf.keras.layers.Dense(
                len(TARGETS), activation="softmax", dtype="float32"
            )(x_img)

        y = y_eeg * 1
        model = tf.keras.Model(inputs=inp, outputs=y)
        return model




## === cell 8
if not SKIP_TF:

    def train_fold(
        i,
        stage,
        train_index,
        valid_index,
        df_train_stage1,
        df_valid_stage1,
        df_train_stage2,
        df_valid_stage2,
        build_model,
        BATCHSIZE,
        EPOCHS,
        LEARN_RATE,
        PATIENCE,
        TARGETS,
        TARGETS_RAW,
    ):

        print("#" * 25)
        print(f"### Fold {i+1}")

        if stage == 1:
            train_gen_stage1 = DataGenerator(
                df_train_stage1,
                shuffle=True,
                sample_weights=True,
                batch_size=BATCHSIZE,
                specs=spectrograms,
                eegs=eegs,
                stfts=stfts,
                imgs=imgs,
            )
            valid_gen_stage1 = DataGenerator(
                df_valid_stage1,
                shuffle=False,
                sample_weights=True,
                batch_size=BATCHSIZE * 2,
                mode="valid",
                specs=spectrograms,
                eegs=eegs,
                stfts=stfts,
                imgs=imgs,
            )

            model = build_model()
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE)
            loss = tf.keras.losses.KLDivergence()
            model.compile(loss=loss, optimizer=opt)

            callbacks_stage1 = [
                tf.keras.callbacks.LearningRateScheduler(
                    CosineAnnealingLRScheduler(EPOCHS, LEARN_RATE, LEARN_RATE * 0.01, 5)
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join("models", f"fold{i}_stage1.weights.h5"),
                    monitor="val_loss",
                    mode="min",
                    save_weights_only=True,
                    save_best_only=True,
                ),
            ]

            history = model.fit(
                train_gen_stage1,
                verbose=1,
                validation_data=valid_gen_stage1,
                epochs=EPOCHS,
                callbacks=callbacks_stage1,
            )

            model.load_weights(os.path.join("models", f"fold{i}_stage1.weights.h5"))

            loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            epochs = range(1, len(loss) + 1)
            plt.figure()
            plt.plot(epochs, loss, "bo", label="loss")
            plt.plot(epochs, val_loss, "b", label="val_loss")
            plt.title(
                f"loss: {round(min(loss), 4)}, val loss: {round(min(val_loss), 4)}",
                fontsize=12,
            )
            plt.legend()
            plt.savefig(os.path.join("models", f"fold{i}_stage1.svg"))
            plt.close()

            valid_stage1 = df_valid_stage1[TARGETS].values
            predict_stage1 = model.predict(valid_gen_stage1)

            del train_gen_stage1, valid_gen_stage1, history, model
            tf.keras.backend.clear_session()
            import gc

            gc.collect()

            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(
                np.argmax(valid_stage1, 1), np.argmax(predict_stage1, 1)
            )
            cm = cm / np.sum(cm, 1, keepdims=True)

            plt.figure()
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            tick_marks = np.arange(6)
            plt.xticks(
                tick_marks,
                [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]],
                fontsize=10,
            )
            plt.yticks(
                tick_marks,
                [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]],
                fontsize=10,
            )
            thresh = cm.max() / 2.0
            for ii in range(cm.shape[0]):
                for jj in range(cm.shape[1]):
                    if cm[ii, jj] > -0.1:
                        plt.text(
                            jj,
                            ii,
                            str(round(cm[ii, jj] * 1e4) * 1e-2)[:5],
                            horizontalalignment="center",
                            color="white" if cm[ii, jj] > thresh else "black",
                            fontsize=10,
                        )
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.tight_layout()
            plt.savefig(os.path.join("models", f"fold{i}_stage1_cm.svg"))
            plt.close()

        if stage == 2:
            train_gen_stage2 = DataGenerator(
                df_train_stage2,
                shuffle=True,
                sample_weights=False,
                batch_size=BATCHSIZE,
                specs=spectrograms,
                eegs=eegs,
                stfts=stfts,
                imgs=imgs,
            )
            valid_gen_stage2 = DataGenerator(
                df_valid_stage2,
                shuffle=False,
                sample_weights=False,
                batch_size=BATCHSIZE * 2,
                mode="valid",
                specs=spectrograms,
                eegs=eegs,
                stfts=stfts,
                imgs=imgs,
            )

            model = build_model()
            opt = tf.keras.optimizers.AdamW(learning_rate=LEARN_RATE * 0.1)
            loss = tf.keras.losses.KLDivergence()
            model.compile(loss=loss, optimizer=opt)
            model.load_weights(os.path.join("models", f"fold{i}_stage1.weights.h5"))

            callbacks_stage2 = [
                tf.keras.callbacks.LearningRateScheduler(
                    CosineAnnealingLRScheduler(
                        max(round(EPOCHS / 3), 1),
                        LEARN_RATE * 0.1,
                        LEARN_RATE * 0.1 * 0.1,
                        0,
                    )
                ),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=os.path.join("models", f"fold{i}_stage2.weights.h5"),
                    monitor="val_loss",
                    mode="min",
                    save_weights_only=True,
                    save_best_only=True,
                ),
            ]

            history = model.fit(
                train_gen_stage2,
                verbose=1,
                validation_data=valid_gen_stage2,
                epochs=max(round(EPOCHS / 3), 1),
                callbacks=callbacks_stage2,
            )

            model.load_weights(os.path.join("models", f"fold{i}_stage2.weights.h5"))

            plt.figure()
            loss = history.history["loss"]
            val_loss = history.history["val_loss"]
            epochs = range(1, len(loss) + 1)
            plt.plot(epochs, loss, "bo", label="loss")
            plt.plot(epochs, val_loss, "b", label="val_loss")
            plt.title(
                f"loss: {round(min(loss), 4)}, val loss: {round(min(val_loss), 4)}",
                fontsize=12,
            )
            plt.legend()
            plt.savefig(os.path.join("models", f"fold{i}_stage2.svg"))
            plt.close()

            valid_stage2 = df_valid_stage2[TARGETS].values
            predict_stage2 = model.predict(valid_gen_stage2)

            del train_gen_stage2, valid_gen_stage2, history, model
            tf.keras.backend.clear_session()
            import gc

            gc.collect()

            from sklearn.metrics import confusion_matrix

            cm = confusion_matrix(
                np.argmax(valid_stage2, 1), np.argmax(predict_stage2, 1)
            )
            cm = cm / np.sum(cm, 1, keepdims=True)

            plt.figure()
            plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            plt.title("Confusion Matrix")
            plt.colorbar()
            tick_marks = np.arange(6)
            plt.xticks(
                tick_marks,
                [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]],
                fontsize=10,
            )
            plt.yticks(
                tick_marks,
                [f"{TARGETS[i][:-5]}" for i in [0, 1, 2, 3, 4, 5]],
                fontsize=10,
            )
            thresh = cm.max() / 2.0
            for ii in range(cm.shape[0]):
                for jj in range(cm.shape[1]):
                    if cm[ii, jj] > -0.1:
                        plt.text(
                            jj,
                            ii,
                            str(round(cm[ii, jj] * 1e4) * 1e-2)[:5],
                            horizontalalignment="center",
                            color="white" if cm[ii, jj] > thresh else "black",
                            fontsize=10,
                        )
            plt.xlabel("Predicted label")
            plt.ylabel("True label")
            plt.tight_layout()
            plt.savefig(os.path.join("models", f"fold{i}_stage2_cm.svg"))
            plt.close()

        del df_train_stage1, df_valid_stage1, df_train_stage2, df_valid_stage2
        import gc

        gc.collect()




## === cell 9
if not SKIP_TF:
    if __name__ == "__main__":
        if NEEDTRAIN:
            if not os.path.exists("models"):
                os.makedirs("models")

            from sklearn.model_selection import GroupKFold
            import multiprocessing as mp

            mp.set_start_method("spawn")  # 设置进程启动方式为 spawn

            gkf = GroupKFold(n_splits=SPLITS)

            for i, (train_index, valid_index) in enumerate(
                gkf.split(train, train.expert_consensus, train.patient_id)
            ):
                print("#" * 25)
                print(f"### Fold {i+1}")

                df_train_stage1 = train.iloc[train_index].reset_index(drop=True)
                df_valid_stage1 = train.iloc[valid_index].reset_index(drop=True)

                df_train_stage2 = df_train_stage1[
                    np.sum(df_train_stage1[TARGETS_RAW].values, 1) >= 10
                ].reset_index(drop=True)
                df_valid_stage2 = df_valid_stage1[
                    np.sum(df_valid_stage1[TARGETS_RAW].values, 1) >= 10
                ].reset_index(drop=True)

                for stage in [1, 2]:
                    p = mp.Process(
                        target=train_fold,
                        args=(
                            i,
                            stage,
                            train_index,
                            valid_index,
                            df_train_stage1,
                            df_valid_stage1,
                            df_train_stage2,
                            df_valid_stage2,
                            build_model,
                            BATCHSIZE,
                            EPOCHS,
                            LEARN_RATE,
                            PATIENCE,
                            TARGETS,
                            TARGETS_RAW,
                        ),
                    )
                    p.start()
                    p.join()
        else:
            pass
