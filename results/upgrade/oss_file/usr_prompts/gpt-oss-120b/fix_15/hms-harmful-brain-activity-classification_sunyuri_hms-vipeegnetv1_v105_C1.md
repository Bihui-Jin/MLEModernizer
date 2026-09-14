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

0.35366687509036

# 6. Current score

0.71539

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'The fix removes the failing local wheel installation and replaces the heavy model‑inference step with a lightweight baseline that directly creates a valid submission.  
1. The pip install line is commented out because the wheel isn’t available and caused a protobuf error.  
2. `efficientnet.tfkeras` is imported inside a try/except; if unavailable we fall back to TensorFlow’s built‑in EfficientNet (though it isn’t used in the simplified path).  
3. When `NEEDTRAIN` is False, instead of building a model and loading large weight files, we compute the average class vote distribution from the training data and assign this normalized vector to every test row, guaranteeing that each prediction sums to 1 and a proper `submission.csv` is written.'
- What this solution (achieved 1.41937) has done: 'The fix avoids the TensorFlow import and configuration when `NEEDTRAIN` is False, because the baseline only computes simple class‑frequency predictions and does not require any TensorFlow functionality. By wrapping all TensorFlow‑related code inside an `if NEEDTRAIN:` block, the protobuf‑related `AttributeError` is prevented, allowing the script to run end‑to‑end and generate a valid `submission.csv`. This change does not affect the core prediction logic and keeps the score‑relevant behavior unchanged.'
- What this solution (achieved 1.68479) has done: 'The script failed because it unconditionally imported TensorFlow internals, which raise an error when TensorFlow isn’t available (NEEDTRAIN = False). I wrapped that import in a safe try/except block.  
To move the KL‑divergence score closer to the target, I replace the single global‑mean baseline with a per‑patient mean prediction: for each test row we use the average vote distribution of its patient (fallback to the global mean). The predictions are normalized so each row sums to 1, keeping the submission format valid while improving relevance without altering the core modeling logic.'
- What this solution (achieved 1.64506) has done: 'The fix removes any unconditional TensorFlow imports (which caused the protobuf AttributeError) and moves them inside the `NEEDTRAIN` guard. It also improves the baseline by predicting class‑probability distributions: votes are first normalized to proportions per training row, then a global mean and per‑patient means of these proportions are computed and used for test predictions. This keeps the core logic unchanged while ensuring a valid submission and moving the KL‑divergence score closer to the target.'
- What this solution (achieved 1.64506) has done: 'I replace the simple global‑mean baseline with a hierarchy that first uses the average vote distribution for each eeg_id (if the test eeg_id appears in the training data), then falls back to the per‑patient average, and finally to the overall global average. This adds only a few lines, preserves the original workflow, and keeps all predictions normalized to sum to 1, which should lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.78544) has done: 'We keep the overall workflow but improve the baseline predictions by blending the per‑patient distribution with the global mean (instead of using the patient mean alone). A small weight α, based on how many training rows a patient has, lets well‑represented patients keep most of their specific pattern while sparse patients fall back toward the overall average. This adjustment keeps rows normalized and moves the KL‑divergence lower, bringing the score nearer the target.'
- What this solution (achieved 0.71539) has done: 'I keep the overall mean‑based baseline but give the patient‑specific distributions more influence when enough training rows exist. By increasing the blending weight α to counts / (counts + 5) and capping it at 1.0, well‑represented patients rely more on their own history, which should lower the KL‑divergence toward the target while preserving the existing workflow.'
- What this solution (achieved 0.71423) has done: 'I keep the overall workflow unchanged but improve the baseline predictions by (1) blending the per‑eeg‑id mean with the global mean using a confidence‑based weight (instead of fully overwriting) and (2) giving patient‑specific means a slightly larger influence by reducing the smoothing denominator. These changes keep every row normalized to 1, retain the original model‑free approach, and are expected to lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.7224) has done: 'I tighten the blending between the specific per‑eeg and per‑patient vote distributions and the global mean by using a smaller smoothing denominator (0.5 for eeg‑level and 1.0 for patient‑level). This gives more weight to the observed distributions when enough training rows exist, which should lower the KL‑divergence and move the score closer to the target while keeping the same overall workflow and ensuring the submission rows still sum to 1.'
- What this solution (achieved 0.75839) has done: 'I tighten the Bayesian smoothing used when blending the per‑eeg and per‑patient vote distributions with the global mean. By reducing the smoothing denominator (from 0.5 → 0.1 for eeg‑level and 1.0 → 0.2 for patient‑level) the predictions rely more on the observed class frequencies where enough training rows exist, which should lower the KL‑divergence toward the target while keeping the overall workflow unchanged and preserving the required submission format.'
- What this solution (achieved 0.71423) has done: 'I loosen the Bayesian smoothing that blends per‑eeg and per‑patient vote distributions with the global mean. By increasing the denominator in the α‑weight calculations (using counts / (counts + 5) for eeg‑level and counts / (counts + 2) for patient‑level) the predictions rely more on the overall global distribution, which reduces over‑fitting and should lower the KL‑divergence toward the target 0.3537. The rest of the pipeline and submission format remain unchanged.'
- What this solution (achieved 1.64506) has done: 'I keep the overall workflow unchanged but make the blending weights deterministic: when a test `eeg_id` appears in the training data we now use its exact mean distribution (α = 1), and when only the patient matches we also use the patient‑level mean directly (α = 1). This gives more specific information to each prediction while still falling back to the global mean for unseen cases, and it keeps the row‑normalisation unchanged, moving the KL‑divergence score closer to the target.'
- What this solution (achieved 0.71539) has done: 'The changes replace the simple “eeg‑or‑patient‑or‑global” fill with a Bayesian‑smoothed blend: each test row gets a weighted mixture of its specific eeg‑mean, patient‑mean, and the overall global mean, where the weights are derived from the number of training samples for that eeg or patient (count / (count+α)). This keeps predictions normalized, improves calibration, and moves the KL‑divergence closer to the target while preserving the original workflow.'
- What this solution (achieved 0.71539) has done: 'I keep the overall workflow but improve the baseline blending: instead of using only the EEG‑specific mean (or patient mean) with the global mean, I blend all three sources (EEG, patient, global) using Bayesian‑style weights derived from their sample counts. This adds only a few lines, preserves the existing pipeline, keeps rows normalised, and moves the KL‑divergence closer to the target score.'

# 9. Code solution

## === cell 0
PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
DATATYPE = ["eeg", "spe", "img"]  # 'eeg', 'spe', 'img', 'stft'
STAGETRAIN = [2, 3]
STAGETEST = 3
print(DATATYPE)
LOAD_MODELS_FROM = "models2024030401"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 30  # s
SFREQ = 100

HIGH = 128  # 128
LENGTH = 256  # 256

IMG_HIGH = 64
IMG_WIDE = 256

SEED = 2024

BATCHSIZE = 16

AMP = 150

READ_SPEC_FILES = False
READ_EEG_FILES = False
READ_IMG_FILES = False
READ_STFT_FILES = False

spectrograms = {}
eegs = {}
imgs = {}
stfts = {}

spectrograms2 = {}
eegs2 = {}
imgs2 = {}
stfts2 = {}

filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

import os
import io
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import pandas as pd, numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if NEEDTRAIN:
    try:
        from tensorflow.python.framework.ops import reset_default_graph
    except Exception:
        reset_default_graph = None

    import tensorflow as tf

    print("TensorFlow version =", tf.__version__)

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")

    VER = 1

    np.random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()

    MIX = True
    if MIX:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
        print("Mixed precision enabled")
    else:
        print("Using full precision")
else:
    reset_default_graph = None
    tf = None

from sklearn.metrics import confusion_matrix
import librosa

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
else:
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))

if not NEEDTRAIN:
    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    else:
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)

    vote_sums = df[TARGETS].sum(axis=1).replace(0, 1)
    prob_df = df[TARGETS].astype(float).div(vote_sums, axis=0)

    global_mean = prob_df.mean()
    patient_means = prob_df.groupby(df["patient_id"]).mean()
    eeg_means = prob_df.groupby(df["eeg_id"]).mean()

    patient_counts = df.groupby("patient_id").size()
    eeg_counts = df.groupby("eeg_id").size()

    ALPHA_EEG = 5.0  # larger => more shrinkage to global
    ALPHA_PAT = 5.0

    pred_df = pd.DataFrame(
        np.tile(global_mean.values, (len(test), 1)),
        columns=TARGETS,
        index=test.index,
    )

    mask_eeg = test["eeg_id"].isin(eeg_means.index)
    mask_patient_only = (~mask_eeg) & test["patient_id"].isin(patient_means.index)

    if mask_eeg.any():
        eeg_ids = test.loc[mask_eeg, "eeg_id"]
        cnts_eeg = eeg_counts.reindex(eeg_ids).fillna(0).values
        w_eeg = cnts_eeg / (cnts_eeg + ALPHA_EEG)  # weight for EEG mean
        w_eeg = w_eeg[:, None]

        pat_ids = test.loc[mask_eeg, "patient_id"]
        cnts_pat = patient_counts.reindex(pat_ids).fillna(0).values
        w_pat_raw = cnts_pat / (cnts_pat + ALPHA_PAT)  # raw patient weight
        w_pat = (1 - w_eeg.squeeze()) * w_pat_raw
        w_pat = w_pat[:, None]

        w_global = 1 - w_eeg - w_pat

        eeg_vecs = eeg_means.loc[eeg_ids].values
        pat_vecs = patient_means.reindex(pat_ids).fillna(global_mean).values

        blended = w_eeg * eeg_vecs + w_pat * pat_vecs + w_global * global_mean.values
        pred_df.loc[mask_eeg, TARGETS] = blended

    if mask_patient_only.any():
        pat_ids = test.loc[mask_patient_only, "patient_id"]
        cnts_pat = patient_counts.reindex(pat_ids).fillna(0).values
        w_pat = cnts_pat / (cnts_pat + ALPHA_PAT)
        w_pat = w_pat[:, None]

        w_global = 1 - w_pat
        pat_vecs = patient_means.loc[pat_ids].values
        blended = w_pat * pat_vecs + w_global * global_mean.values
        pred_df.loc[mask_patient_only, TARGETS] = blended

    row_sums = pred_df[TARGETS].sum(axis=1).replace(0, 1)
    pred_df[TARGETS] = pred_df[TARGETS].div(row_sums, axis=0)

    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    sub[TARGETS] = pred_df[TARGETS].values
    sub.to_csv("submission.csv", index=False)
    print("Submission written to submission.csv")
    print("Submission shape", sub.shape)
    print("Row sums (should be 1.0):", sub[TARGETS].sum(axis=1).unique()[:5])
