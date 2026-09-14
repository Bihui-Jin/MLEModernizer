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

0.2840643394395511

# 6. Current score

0.87563

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The crash happens before any of your competition logic runs: TensorFlow’s import chain hits an incompatible `protobuf` API (`MessageFactory.GetPrototype`) in this Kaggle/Python 3.13 environment. The minimal safe fix is to avoid importing TensorFlow/Keras entirely at runtime and instead fall back to a simple, valid probabilistic submission built from the training label priors (this is score-worse than a real model but yields a valid `.csv` and unblocks end-to-end execution). I also fix a hard bug in your test-time batching slice (`len(preds_all)` used as a row index) and enforce strict row-wise normalization so the submission never fails the “sum to one” requirement. The core model/training code is preserved but gated behind a TensorFlow-availability check so it won’t execute (and won’t crash) in the current environment.'
- What this solution (achieved 1.65161) has done: 'I fix the TensorFlow/protobuf crash by making TensorFlow imports fully optional and non-fatal, so the notebook always runs end-to-end in the Python 3.13 Kaggle environment. Since your current score (1.41937, lower-is-better) is far worse than the target (0.284), I improve the fallback (no-TF) path in a minimal, metric-aligned way by predicting patient-conditioned label priors (patient_id → average normalized vote distribution), with a global prior fallback for unseen patients. I also fix the test-time concatenation bug in the TF inference path (`len(preds_all)` on a numpy array/scalar issue) by using a list-append + concatenate pattern, without changing model logic. Finally, I enforce strict row-wise normalization and output a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 1.65161) has done: 'The timeout is dominated by TensorFlow inference: repeatedly building a `DataGenerator` per chunk, calling `model.predict` separately for each model and chunk, and paying Python/Sequence overhead plus repeated `.iloc` access in `__getitem__`. To preserve identical core logic and outputs, the optimized version keeps the same preprocessing, model architecture, and averaging semantics, but removes the Sequence bottleneck by precomputing all test EEG tensors once and running inference on plain NumPy arrays (same inputs, same batching). It also avoids storing all per-model predictions (streaming sum/mean instead), and speeds up parquet reading and montage construction by using vectorized NumPy operations instead of per-channel pandas slicing. These changes cut constant factors heavily without changing what is computed.'
- What this solution (achieved 1.65161) has done: 'The timeout is dominated by TensorFlow inference: the script loads thousands of EEG parquet files into one giant `x_test_eeg` array, then runs `predict()` repeatedly for up to 100 cloned models—this multiplies both I/O and compute. To keep identical model logic and outputs while finishing under 600s, I switch to streaming inference in fixed-size chunks so we never materialize the full test tensor and we reuse a single preallocated buffer per chunk. I also eliminate per-row pandas overhead by using a faster parquet-to-numpy path (still via pandas) and reduce repeated Python work inside the EEG loop (precomputed indices/slices, fewer temporaries) while keeping the exact same preprocessing math and model calls. The non-TF fallback (patient prior submission) remains unchanged and still writes `submission.csv` quickly if TF is unavailable.'
- What this solution (achieved 1.65161) has done: 'I fix the immediate crash by preventing any TensorFlow/Keras import attempt in this Python 3.13 environment (the protobuf incompatibility happens even inside a try/except) so the notebook can run end-to-end. Then I keep your non-TF fallback (patient-conditioned priors) but make its output strictly match the required submission schema by using the official `sample_submission.csv` as the template and writing predictions in that exact row order. Finally, I harden probability normalization with float64 + clipping + renormalization and ensure the written `submission.csv` has per-row sums equal to 1 within numeric tolerance, preventing Kaggle “sum to 1” submission failures.'
- What this solution (achieved 0.83738) has done: 'Your current score (1.65161, lower-is-better) is far above the target (0.284), and since TensorFlow is disabled in this runtime the only way to improve score is to strengthen the non-TF fallback predictions while keeping the same “prior-based” core logic. I (1) compute a more robust Bayesian-smoothed patient prior (patient distribution shrunk toward global prior) to reduce overfitting/noise for patients with few samples, (2) add an additional backoff level using per-spectrogram priors (also smoothed) since test provides `spectrogram_id`, and (3) ensemble these priors with small fixed weights and strict row-wise normalization to keep the submission valid. This keeps the solution lightweight, deterministic, and end-to-end, and should move KLDiv down toward the target without changing the TF model path.'
- What this solution (achieved 0.85944) has done: 'Your current score (0.83738, lower-is-better) is still far above the target (0.28406), and TensorFlow is disabled, so the only legitimate way to move toward target is to strengthen the non-TF fallback while keeping the same “prior-based” approach. I keep your global/patient/spectrogram smoothed priors, but add one minimal, metric-aligned backoff: a patient×spectrogram joint prior when that pair exists in training (also Dirichlet-smoothed), then mix it with the existing patient/spec/global priors. I also choose weights in a data-adaptive way based on available counts (more trust when the group has more examples) without changing the overall semantics (still just a shrinkage prior ensemble). Finally, I keep strict clipping + renormalization and continue to write `submission.csv` using `sample_submission.csv` row order.'
- What this solution (achieved 0.85591) has done: 'Your current score (0.85944, lower-is-better) is still far above the target (0.28406), and TensorFlow is disabled here, so the only legitimate way to move toward target is to strengthen the non-TF prior-based fallback without changing its core approach. I keep the same global/patient/spectrogram/patient×spectrogram smoothed priors, but make the ensemble weights evidence-adaptive in a more metric-aligned way (softly increasing reliance on higher-evidence groups instead of fixed base weights). I also switch smoothing to be done on *vote counts* (Dirichlet on counts) rather than mean-of-row-probabilities, which is still the same “smoothed prior” logic but better matches the true target distribution and typically reduces KL. Finally, I keep strict clipping+renormalization and still write `submission.csv` in the exact `sample_submission.csv` row order.'
- What this solution (achieved 0.8507) has done: 'Your current score (0.85591, lower-is-better) is still far above the target (0.28406), and TensorFlow is disabled in this runtime, so the only legitimate way to move toward the target is to improve the non-TF “prior-based” fallback while keeping the same overall approach (smoothed empirical label priors). I make a minimal metric-aligned change: replace the linear mixture of priors with a geometric (log-space) product-of-experts blend, which typically reduces KL because it yields sharper distributions when the group priors agree, while still being pure prior-ensemble logic. I keep the same evidence-adaptive weighting idea but apply it as exponents in log-space, and keep the same strict clipping+renormalization and `sample_submission.csv` row order. This preserves end-to-end runtime and submission validity while being a small, targeted change expected to improve KL.'
- What this solution (achieved 0.94691) has done: 'Your current score (0.8507, lower-is-better) is still far from the target (0.2841), and TensorFlow is intentionally disabled, so the only viable improvements are within the non-TF prior-ensemble fallback. The smallest metric-aligned change is to make the evidence weights depend on *both* (a) how many total votes exist for the group and (b) how “confident” (low-entropy) the group prior is—this tends to reduce KL by trusting sharp, well-supported priors more and backing off to global when priors are diffuse/noisy. I keep the same priors (global/patient/spec/patient×spec) and the same geometric (log-space) blending core logic, only adjusting how the blend weights are computed. I also keep strict clipping+renormalization and still write `submission.csv` in `sample_submission.csv` row order.'
- What this solution (achieved 0.94691) has done: 'Your current score (0.94691, lower-is-better) is still far above the target (0.28406), and since TensorFlow is disabled the only lever is improving the non-TF prior-based predictor without changing its overall approach. The main minimal fix is to compute all group priors on **de-duplicated label sets** (unique `label_id`) so repeated overlapping windows from the same label don’t overweight the same votes, which should reduce KL. Then, keep the same global/patient/spec/patient×spec priors and geometric blending, but make the “evidence” use the **sum of votes** in each group (not number of rows), which better matches the metric and reduces noise. Finally, keep strict clipping+renormalization and still write `submission.csv` in exact `sample_submission.csv` order to ensure a valid submission.'
- What this solution (achieved 0.87563) has done: 'Your current score (0.94691, lower-is-better) is still far from the target (0.28406), and TensorFlow is disabled, so the only realistic path is improving the non-TF prior-based predictor while keeping its core logic intact. The minimal metric-aligned fix here is to compute group priors (patient/spec/patient×spec) on true **vote-count aggregates per unique `label_id`** (not per-row), and to use the **sum of votes as evidence** for weighting (since KL is defined vs the true vote distribution). I keep the same priors set and the same geometric (log-space) blending, but make the weights strictly evidence-based (votes) with a small global backoff so predictions stay well-calibrated. The submission is still written using `sample_submission.csv` row order and strict per-row normalization to guarantee a valid file.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri

email: syuri@tju.edu.cn
"""

NEEDTRAIN = True  # train the model
LOAD_MODELS_FROM = "modelsxxxxxxx"  # the path of trained model weights for testing

import os
import warnings

warnings.filterwarnings("ignore")

os.environ["KERAS_BACKEND"] = "tensorflow"

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
else:
    PLATFORM = "kaggle"
    NEEDTRAIN = False

DATATYPE = ["eeg"]  # *** spe, eeg, stft, img *** the data type used
print(DATATYPE)

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

IMG_LENGTH = 20
IMG_HIGH = 324
IMG_WIDE = 324

SPE_HIGH = 40
SPE_WIDE = 1000

STFT_LENGTH = 50
STFT_TIME = 0.1
STFT_HIGH = 50
STFT_WIDE = 200

filter_range = [0.5, 45]
filter_range2 = [0.1, 35]

SEED = 2024
BATCHSIZE = 16
LEARN_RATE = 1e-3
EPOCHS = 15
PATIENCE = 5
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
    "T5-O1",
    "Fp1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "Fz-Cz",
    "Cz-Pz",
    "Fp2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "Fp2-F8",
    "F8-T4",
    "T4-T6",
    "T6-O2",
]

TEST_BATCHSIZE = 128

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
os.environ["PYTHONHASHSEED"] = str(SEED)

import numpy as np
import pandas as pd

np.random.seed(SEED)

TF_AVAILABLE = False
TF_IMPORT_ERROR = "Disabled to avoid protobuf/TensorFlow crash in this runtime"
print(f"TF_AVAILABLE={TF_AVAILABLE}")
print("TensorFlow disabled; using non-TF fallback submission.")
print("TF import note:", TF_IMPORT_ERROR)



## === cell 1
df = pd.read_csv(os.path.join(LOAD_DATA_FROM, "train.csv"))
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

if "label_id" in df.columns:
    grp = (
        df.groupby("label_id", as_index=False)[
            ["patient_id", "spectrogram_id"] + list(TARGETS)
        ]
        .agg(
            {
                "patient_id": "first",
                "spectrogram_id": "first",
                **{c: "first" for c in TARGETS},  # votes are identical within label_id
            }
        )
        .copy()
    )
    df_lbl = grp
else:
    df_lbl = df.copy()

vote_sums = df_lbl[list(TARGETS)].sum(axis=0).values.astype(np.float64)
prior_global = vote_sums / np.clip(vote_sums.sum(), 1e-12, None)
prior_global = np.clip(prior_global.astype(np.float64), 1e-15, None)
prior_global = prior_global / prior_global.sum()
print("Global prior sum=", float(prior_global.sum()))
print("Using label-level rows for priors:", len(df_lbl), "of", len(df))

df_v = df_lbl[["patient_id", "spectrogram_id"] + list(TARGETS)].copy()
df_v["patient_spec"] = (
    df_v["patient_id"].astype(str) + "_" + df_v["spectrogram_id"].astype(str)
)


def make_smoothed_group_prior_from_counts(
    df_votes: pd.DataFrame,
    group_col: str,
    targets,
    global_prior: np.ndarray,
    alpha: float,
):
    """
    Dirichlet-like smoothing on vote counts:
      count_group = sum(votes) for group (over label-level rows)
      p_group_smoothed = (count_group + alpha * global_prior) / (sum(count_group) + alpha)
    Returns:
      prior_df: index=group, columns=targets with smoothed probs
      grp_total_votes: total annotator votes per group (sum over targets)
    """
    grp_counts = df_votes.groupby(group_col)[list(targets)].sum().astype(np.float64)
    grp_total_votes = grp_counts.sum(axis=1).astype(np.float64)

    smoothed = grp_counts.values + alpha * global_prior.reshape(1, -1)
    smoothed = smoothed / np.clip(smoothed.sum(axis=1, keepdims=True), 1e-12, None)

    smoothed = np.clip(smoothed, 1e-15, None)
    smoothed = smoothed / np.clip(smoothed.sum(axis=1, keepdims=True), 1e-15, None)

    prior_df = grp_counts.copy()
    prior_df.loc[:, list(targets)] = smoothed.astype(np.float32)
    return prior_df, grp_total_votes


patient_prior, patient_total_votes = make_smoothed_group_prior_from_counts(
    df_v, "patient_id", TARGETS, prior_global, alpha=60.0
)
spec_prior, spec_total_votes = make_smoothed_group_prior_from_counts(
    df_v, "spectrogram_id", TARGETS, prior_global, alpha=30.0
)
ps_prior, ps_total_votes = make_smoothed_group_prior_from_counts(
    df_v, "patient_spec", TARGETS, prior_global, alpha=80.0
)

print("Patient priors:", patient_prior.shape, "unique patients")
print("Spectrogram priors:", spec_prior.shape, "unique spectrograms")
print("Patient×Spectrogram priors:", ps_prior.shape, "unique pairs")



## === cell 2
test = pd.read_csv(os.path.join(LOAD_DATA_FROM, "test.csv"))
sample_sub = pd.read_csv(os.path.join(LOAD_DATA_FROM, "sample_submission.csv"))

assert "eeg_id" in sample_sub.columns, "sample_submission must contain eeg_id"
for c in TARGETS:
    if c not in sample_sub.columns:
        raise RuntimeError(f"Missing target column in sample_submission: {c}")

eeg_to_patient = dict(zip(test["eeg_id"].values, test["patient_id"].values))
eeg_to_spec = dict(zip(test["eeg_id"].values, test["spectrogram_id"].values))

patient_index = patient_prior.index
patient_values = patient_prior[list(TARGETS)].values.astype(np.float64)
patient_to_row = {pid: i for i, pid in enumerate(patient_index.values)}
patient_votes_map = patient_total_votes.to_dict()

spec_index = spec_prior.index
spec_values = spec_prior[list(TARGETS)].values.astype(np.float64)
spec_to_row = {sid: i for i, sid in enumerate(spec_index.values)}
spec_votes_map = spec_total_votes.to_dict()

ps_index = ps_prior.index
ps_values = ps_prior[list(TARGETS)].values.astype(np.float64)
ps_to_row = {k: i for i, k in enumerate(ps_index.values)}
ps_votes_map = ps_total_votes.to_dict()

n_test = len(sample_sub)
preds = np.empty((n_test, len(TARGETS)), dtype=np.float64)

LOG_EPS = 1e-15


def evidence_weight_votes(total_votes: float, tau: float) -> float:
    total_votes = 0.0 if total_votes is None else float(total_votes)
    if total_votes <= 0:
        return 0.0
    return total_votes / (total_votes + tau)


TAU_PS = 300.0
TAU_PAT = 600.0
TAU_SPEC = 600.0

W_GLOB_BASE = 0.10  # small global backoff to avoid overconfident group priors

for i, eeg_id in enumerate(sample_sub["eeg_id"].values):
    pid = eeg_to_patient.get(eeg_id, None)
    sid = eeg_to_spec.get(eeg_id, None)
    ps_key = None if (pid is None or sid is None) else (str(pid) + "_" + str(sid))

    p_ps = None
    p_pat = None
    p_spec = None

    v_ps = None
    v_pat = None
    v_spec = None

    if ps_key is not None and ps_key in ps_to_row:
        p_ps = ps_values[ps_to_row[ps_key]]
        v_ps = ps_votes_map.get(ps_key, 0.0)

    if pid is not None and pid in patient_to_row:
        p_pat = patient_values[patient_to_row[pid]]
        v_pat = patient_votes_map.get(pid, 0.0)

    if sid is not None and sid in spec_to_row:
        p_spec = spec_values[spec_to_row[sid]]
        v_spec = spec_votes_map.get(sid, 0.0)

    w_ps = evidence_weight_votes(v_ps, TAU_PS) if p_ps is not None else 0.0
    w_pat = evidence_weight_votes(v_pat, TAU_PAT) if p_pat is not None else 0.0
    w_spec = evidence_weight_votes(v_spec, TAU_SPEC) if p_spec is not None else 0.0
    w_glob = W_GLOB_BASE

    w_sum = w_ps + w_pat + w_spec + w_glob
    if w_sum <= 0:
        out = prior_global
    else:
        w_ps /= w_sum
        w_pat /= w_sum
        w_spec /= w_sum
        w_glob /= w_sum

        logp = w_glob * np.log(np.clip(prior_global, LOG_EPS, None))
        if p_ps is not None:
            logp += w_ps * np.log(np.clip(p_ps, LOG_EPS, None))
        if p_pat is not None:
            logp += w_pat * np.log(np.clip(p_pat, LOG_EPS, None))
        if p_spec is not None:
            logp += w_spec * np.log(np.clip(p_spec, LOG_EPS, None))

        out = np.exp(logp)
        out = np.clip(out, LOG_EPS, None)
        out = out / np.clip(out.sum(), LOG_EPS, None)

    preds[i] = out

preds = np.clip(preds, 1e-15, None)
preds = preds / np.clip(preds.sum(axis=1, keepdims=True), 1e-15, None)

sub = sample_sub.copy()
sub[list(TARGETS)] = preds.astype(np.float32)

row_sums = sub[list(TARGETS)].sum(axis=1).values
max_abs_err = float(np.max(np.abs(row_sums - 1.0)))
print("Max |row_sum-1|:", max_abs_err)
if not np.isfinite(max_abs_err) or max_abs_err > 1e-4:
    vals = sub[list(TARGETS)].values.astype(np.float64)
    vals = np.clip(vals, 1e-15, None)
    vals = vals / np.clip(vals.sum(axis=1, keepdims=True), 1e-15, None)
    sub[list(TARGETS)] = vals.astype(np.float32)
    row_sums2 = sub[list(TARGETS)].sum(axis=1).values
    print("After renorm max |row_sum-1|:", float(np.max(np.abs(row_sums2 - 1.0))))

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv")
print("Submission shape", sub.shape)
print(sub.head())



## === cell 3
if TF_AVAILABLE:
    import gc
    import time
    from scipy import signal
    from tensorflow.keras import optimizers
    from tensorflow.keras.models import clone_model

    import tensorflow as tf  # only if actually available

    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass

    MIX = True
    if MIX:
        try:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
        except Exception:
            pass

    if filter_range is not None:
        b, a = signal.butter(3, np.float32(filter_range) * 2 / RSFREQ, "bandpass")

    if NEEDTRAIN:
        pass

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
                if self.mode == "test":
                    r_eeg = 0

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
                    eeg = np.clip(eeg, a_min=-255, a_max=255)
                    eeg = eeg + 255
                    eeg = eeg / 2
                    x_eeg[j] = eeg

            x = {}
            if "eeg" in DATATYPE:
                x["eeg"] = x_eeg
            return x, y, sample_weights

    class CosineAnnealingLRScheduler(optimizers.schedules.LearningRateSchedule):
        def __init__(self, total_step, lr_max, lr_min=0, warmth_rate=0):
            super(CosineAnnealingLRScheduler, self).__init__()
            self.total_step = total_step
            self.warm_step = 1 if warmth_rate == 0 else int(warmth_rate)
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
        inp_eeg = tf.keras.Input(
            shape=(
                EEG_CHANNEL_USED * EEG_MULTIPLY,
                round(EEG_LENGTH_USED * RSFREQ / EEG_MULTIPLY),
            ),
            name="eeg",
        )
        x_eeg_raw = tf.keras.layers.Reshape((inp_eeg.shape[1], inp_eeg.shape[2], 1))(
            inp_eeg
        )

        strides = 10
        if PLATFORM == "local":
            eeg_embed = tf.keras.layers.Conv1D(
                filters=strides * 3,
                kernel_size=strides,
                strides=strides,
                padding="same",
                use_bias=True,
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
                use_bias=True,
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

        base_model_eeg = tf.keras.applications.EfficientNetV2B0(
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
        x_eeg = x_eeg[:, :, (x_eeg.shape[2] - 1) // 2 : (x_eeg.shape[2]) // 2 + 1, :]
        x_eeg = tf.keras.layers.GlobalAveragePooling2D()(x_eeg)
        x_eeg = tf.keras.layers.Dropout(0.5)(x_eeg)
        y_eeg = tf.keras.layers.Dense(
            len(TARGETS), activation="softmax", dtype="float32"
        )(x_eeg)

        inp.append(inp_eeg)
        model = tf.keras.Model(inputs=inp, outputs=y_eeg)
        return model
