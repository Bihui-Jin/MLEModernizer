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

0.3404325331581851

# 6. Current score

1.46771

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I remove the intentional RuntimeError stop so the notebook always completes and writes `submission.csv`. Because TensorFlow import is currently failing due to a protobuf incompatibility (`MessageFactory.GetPrototype`), I make the script robust by catching that failure and deterministically falling back to a valid uniform-probability submission (rows sum to 1). I also adjust the cell numbering to start at 1 (your provided script starts at cell 0), which is required by the execution harness. These changes are minimal, unblock end-to-end execution, and guarantee a valid `.csv` submission even when TF can’t be used.'
- What this solution (achieved 1.39779) has done: 'I fix the runtime failure by avoiding the TensorFlow import entirely (the protobuf `MessageFactory.GetPrototype` crash happens during import and is not needed to write a valid submission). Then I replace the uniform fallback with a score-improving, still-minimal “prior-based” prediction: compute the average class distribution from `train.csv` vote counts (converted to probabilities) and use that same distribution for every test row—this is a legitimate, stable baseline that typically beats uniform on KL for this competition. Finally, I ensure the submission columns match exactly and rows sum to 1, and keep paths unchanged.'
- What this solution (achieved 1.39779) has done: 'Your current approach (constant class-prior for all rows) is a solid safe baseline, but it’s leaving score on the table because KL on this competition strongly benefits from predicting *per-eeg_id* label distributions rather than per-row distributions (train.csv has multiple overlapping clips per eeg_id). The smallest legitimate improvement that preserves your “no TF, no model” core logic is to compute the empirical class prior **grouped by eeg_id** (aggregate votes across all rows belonging to the same eeg_id), then use that as the prediction for any matching eeg_id in test; for any test eeg_id not seen in train, fall back to the global prior. This uses only metadata already allowed, avoids leakage (train labels only), stays deterministic, and usually moves KL much closer to your target. I also keep the strict “rows sum to 1” normalization and column ordering identical to sample_submission to avoid submission-format penalties.'
- What this solution (achieved 1.46771) has done: 'Your current baseline is already deterministic and valid, but it’s still far from the target because it can’t distinguish test eeg_ids that aren’t present in train and it ignores a strong available signal: patient-level label priors. The smallest legitimate improvement (no model, no TF, same “vote-prior” logic) is to add a second fallback: if a test `eeg_id` is unseen in train, use the aggregated prior for that `patient_id`; only if the patient is also unseen do we fall back to the global prior. This preserves evaluation semantics, keeps outputs as proper probabilities summing to 1, and typically reduces KL materially versus global-only fallback. I also keep the submission column order identical to `sample_submission.csv` and add a tiny safety smoothing/renorm to prevent any numerical issues.'

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

import numpy as np
import pandas as pd

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

os.environ["KERAS_BACKEND"] = "tensorflow"

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"  # local
    for dir_name in os.listdir("./input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name
elif os.getcwd().split(os.sep)[1] == "kaggle":
    PLATFORM = "kaggle"  # kaggle
    NEEDTRAIN = False
    for dir_name in os.listdir("/kaggle/input/"):
        if dir_name[:6] == "models":
            LOAD_MODELS_FROM = dir_name
else:
    PLATFORM = "kaggle"
    NEEDTRAIN = False

datatype = "spe"
print(datatype)

DATATYPE = ["spe", "eeg"]  # kept for compatibility with original code

if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))




## === cell 1
def make_hierarchical_prior_submission(
    load_data_from: str, train_df: pd.DataFrame, targets
):
    """
    Minimal score-improving change vs eeg_id-only prior:
    - Still no TensorFlow / no model.
    - Hierarchical fallback for unseen test eeg_id:
        1) use per-eeg_id aggregated vote distribution if eeg_id seen in train
        2) else use per-patient_id aggregated vote distribution if patient_id seen in train
        3) else fall back to global prior
    This usually reduces KL because patient_id carries strong label bias.
    """
    test = pd.read_csv(os.path.join(load_data_from, "test.csv"))
    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})

    votes = train_df.loc[:, targets].astype(np.float64).values
    row_sums = votes.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    probs = votes / row_sums
    global_prior = probs.mean(axis=0)
    global_prior = np.clip(global_prior, 1e-12, None)
    global_prior = global_prior / global_prior.sum()

    g_eeg = (
        train_df.groupby("eeg_id", sort=False)[list(targets)].sum().astype(np.float64)
    )
    g_eeg_sums = g_eeg.sum(axis=1).values
    g_eeg_sums[g_eeg_sums == 0.0] = 1.0
    eeg_probs = (g_eeg.values.T / g_eeg_sums).T
    eeg_probs = np.clip(eeg_probs, 1e-12, None)
    eeg_probs = (eeg_probs.T / eeg_probs.sum(axis=1)).T
    eeg_to_prob = {
        int(eeg_id): eeg_probs[i] for i, eeg_id in enumerate(g_eeg.index.values)
    }

    g_pat = (
        train_df.groupby("patient_id", sort=False)[list(targets)]
        .sum()
        .astype(np.float64)
    )
    g_pat_sums = g_pat.sum(axis=1).values
    g_pat_sums[g_pat_sums == 0.0] = 1.0
    pat_probs = (g_pat.values.T / g_pat_sums).T
    pat_probs = np.clip(pat_probs, 1e-12, None)
    pat_probs = (pat_probs.T / pat_probs.sum(axis=1)).T
    pat_to_prob = {int(pid): pat_probs[i] for i, pid in enumerate(g_pat.index.values)}

    p = np.empty((len(test), len(targets)), dtype=np.float64)
    test_eeg = test["eeg_id"].astype(int).values
    test_pat = test["patient_id"].astype(int).values

    for i in range(len(test)):
        eeg_id = int(test_eeg[i])
        pid = int(test_pat[i])
        if eeg_id in eeg_to_prob:
            p[i] = eeg_to_prob[eeg_id]
        elif pid in pat_to_prob:
            p[i] = pat_to_prob[pid]
        else:
            p[i] = global_prior

    eps = 1e-12
    p = np.clip(p, eps, None)
    p = (p.T / p.sum(axis=1)).T

    for j, c in enumerate(targets):
        sub[c] = p[:, j].astype(np.float64)

    sample = pd.read_csv(os.path.join(load_data_from, "sample_submission.csv"))
    sub = sub[sample.columns.tolist()]

    out_path = "submission.csv"
    sub.to_csv(out_path, index=False)
    print("Wrote:", out_path)
    print("Submission shape:", sub.shape)
    print(
        "Global prior (final fallback):",
        {t: float(v) for t, v in zip(targets, global_prior)},
    )
    print(
        "Known eeg_id priors:",
        len(eeg_to_prob),
        " / train unique eeg_id:",
        train_df["eeg_id"].nunique(),
    )
    print(
        "Known patient_id priors:",
        len(pat_to_prob),
        " / train unique patient_id:",
        train_df["patient_id"].nunique(),
    )
    print(sub.head())
    return sub


_ = make_hierarchical_prior_submission(LOAD_DATA_FROM, df, TARGETS)



## === cell 2
print("Done. submission.csv is ready.")
print(
    "Note: TensorFlow import skipped; submission created via hierarchical vote-prior baseline (eeg_id -> patient_id -> global prior)."
)
