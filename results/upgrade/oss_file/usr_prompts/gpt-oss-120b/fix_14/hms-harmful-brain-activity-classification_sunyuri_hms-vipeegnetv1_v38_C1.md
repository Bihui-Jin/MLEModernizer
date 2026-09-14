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

0.4808613297613214

# 6. Current score

0.7838

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.48867) has done: 'The fix disables TensorFlow to avoid import‑related crashes and ensures the fallback baseline prediction path runs, producing a valid CSV submission. No core modeling logic is altered; only the TF setup is safely bypassed.'
- What this solution (achieved 1.67064) has done: 'I fix the error that occurs when filling missing patient‑specific means: instead of passing a NumPy array to `fillna`, I build a column‑wise dictionary of the global class means. This resolves the `ValueError` and lets the baseline generate a valid submission CSV. No other logic is changed.'
- What this solution (achieved 0.8262) has done: 'I keep the overall pipeline unchanged but adjust the patient‑aware baseline to be a weighted blend of patient‑specific means and the global mean, followed by a tiny smoothing step so every row sums exactly to 1. This modest change is expected to reduce overly confident wrong predictions and therefore lower the KL‑divergence toward the target score.'
- What this solution (achieved 0.7921) has done: 'I added a quick internal validation that searches for the optimal blend weight between patient‑specific means and the global mean. The script now splits the aggregated training data, evaluates several candidate weights on a hold‑out set using the KL‑divergence, picks the weight that gives the lowest loss, and then uses this weight for the final test predictions. This modest tuning is expected to lower the KL score toward the target while keeping the core baseline logic unchanged.'
- What this solution (achieved 0.78512) has done: 'I refine the weight‑search to a two‑stage grid (coarse 0.0‑1.0 step 0.1 then fine‑grained 0.01 around the best coarse value) and add a tiny uniform smoothing before the final normalization. This keeps the baseline logic unchanged but should lower the KL divergence, moving the score closer to the target.'
- What this solution (achieved 0.78486) has done: 'I add a very small grid search for the uniform smoothing factor after the patient‑weight tuning. The code still keeps the same baseline logic, only evaluates a few smoothing values on the validation split and picks the one that yields the lowest KL score, then uses that smoothing when generating the final test predictions.'
- What this solution (achieved 0.7838) has done: 'I extend the validation‑time grid‑search to a finer resolution for both the patient‑weight and the uniform smoothing factor, which can modestly improve the KL‑divergence without altering any core modeling logic. By searching a tighter range around the previously best weight (±0.02 with 0.005 steps) and a finer smoothing range (0.0‑0.01 with 0.0005 steps), we can likely lower the validation score and thus move the final Kaggle score closer to the target.'
- What this solution (achieved 0.7838) has done: 'I added a lightweight temperature‑scaling step to the baseline blending. After the existing grid‑search for the patient‑weight and smoothing, the script now searches a small range of temperature values on the validation split, picks the one that gives the lowest KL‑divergence, and applies the same scaling to the final test predictions. This keeps the original baseline logic intact while providing a modest calibration improvement expected to move the KL score closer to the target.'
- What this solution (achieved 0.7838) has done: 'I slightly broaden the calibration search ranges (temperature up to 3.0 and smoothing up to 0.02) and add a tiny clipping step after temperature scaling to keep predictions numerically stable. These minimal adjustments keep the baseline logic intact while giving the validation search a bit more flexibility, which should lower the KL‑divergence and move the score closer to the target.'
- What this solution (achieved 0.7838) has done: 'I extend the temperature‑scaling search range so the model can apply stronger flattening when it is over‑confident, which often reduces KL‑divergence for this baseline. The only change is the loop that scans `T`; the rest of the logic, blending, smoothing and validation remain untouched.'

# 9. Code solution

## === cell 0
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

TF_AVAILABLE = False
tf = None

import pandas as pd, numpy as np
import matplotlib
import matplotlib.pyplot as plt

print("TensorFlow available:", TF_AVAILABLE)

if TF_AVAILABLE:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) <= 1:
        strategy = (
            tf.distribute.OneDeviceStrategy(device="/gpu:0") if TF_AVAILABLE else None
        )
        print(f"Using {len(gpus)} GPU")
    else:
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using {len(gpus)} GPUs")

VER = 1
MIX = True
if MIX and TF_AVAILABLE:
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})
    print("Mixed precision enabled")
else:
    print("Using full precision")

PLATFORM = "kaggle"  # local kaggle
NEEDTRAIN = False
LOAD_MODELS_FROM = "models202402051"
if PLATFORM == "local":
    LOAD_MODELS_FROM = f"./input/{LOAD_MODELS_FROM}"
elif PLATFORM == "kaggle":
    LOAD_MODELS_FROM = f"/kaggle/input/{LOAD_MODELS_FROM}"

EEG_LENGTH = 20.48  # s
SFREQ = 200
HIGH = 128  # 128
LENGTH = 32  # 256
READ_SPEC_FILES = False
READ_EEG_FILES = False
filter_range = [0.5, 40]

BRAIN = {
    "LL": ["Fp1-F7", "F7-T3", "T3-T5", "T5-O1"],
    "RL": ["Fp2-F8", "F8-T4", "T4-T6", "T6-O2"],
    "LP": ["Fp1-F3", "F3-C3", "C3-P3", "P3-O1"],
    "RP": ["Fp2-F4", "F4-C4", "C4-P4", "P4-O2"],
}

if PLATFORM == "local":
    df = pd.read_csv("./input/hms-harmful-brain-activity-classification/train.csv")
elif PLATFORM == "kaggle":
    df = pd.read_csv(
        "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    )
TARGETS = df.columns[-6:]
print("Train shape:", df.shape)
print("Targets", list(TARGETS))
df.head()



## === cell 1
train = df.groupby("eeg_id")[
    ["spectrogram_id", "spectrogram_label_offset_seconds", "eeg_label_offset_seconds"]
].agg(
    {
        "spectrogram_id": "first",
        "spectrogram_label_offset_seconds": "min",
        "eeg_label_offset_seconds": "median",
    }
)
train.columns = ["spec_id", "min", "eeg_median"]

tmp = df.groupby("eeg_id")[["spectrogram_id", "spectrogram_label_offset_seconds"]].agg(
    {"spectrogram_label_offset_seconds": "max"}
)
train["max"] = tmp

tmp = df.groupby("eeg_id")[["patient_id"]].agg("first")
train["patient_id"] = tmp

tmp = df.groupby("eeg_id")[TARGETS].agg("sum")
for t in TARGETS:
    train[t] = tmp[t].values

y_data = train[TARGETS].values
y_data = y_data / y_data.sum(axis=1, keepdims=True)
train[TARGETS] = y_data

tmp = df.groupby("eeg_id")[["expert_consensus"]].agg("first")
train["target"] = tmp

train = train.reset_index()
print("Train non-overlapp eeg_id shape:", train.shape)
train.head()



## === cell 2
if not NEEDTRAIN:
    rng = np.random.RandomState(42)
    eeg_ids = train["eeg_id"].unique()
    rng.shuffle(eeg_ids)
    split_idx = int(0.8 * len(eeg_ids))
    train_ids = eeg_ids[:split_idx]
    val_ids = eeg_ids[split_idx:]

    train_fold = train[train["eeg_id"].isin(train_ids)].copy()
    val = train[train["eeg_id"].isin(val_ids)].copy()

    global_means_fold = train_fold[TARGETS].mean().to_dict()
    patient_means_fold = train_fold.groupby("patient_id")[TARGETS].mean()

    def kl_div(true_probs, pred_probs):
        eps = 1e-12
        true = np.clip(true_probs, eps, 1.0)
        pred = np.clip(pred_probs, eps, 1.0)
        return np.sum(true * np.log(true / pred), axis=1)

    best_w = 0.7  # placeholder
    best_score = np.inf
    for w in np.arange(0.0, 1.01, 0.1):
        preds_val = val[["eeg_id", "patient_id"]].merge(
            patient_means_fold, left_on="patient_id", right_index=True, how="left"
        )
        preds_val[TARGETS] = preds_val[TARGETS].fillna(
            {col: global_means_fold[col] for col in TARGETS}
        )
        for col in TARGETS:
            preds_val[col] = w * preds_val[col] + (1 - w) * global_means_fold[col]

        epsilon = 1e-6
        pred_arr = preds_val[TARGETS].values + epsilon
        pred_arr = pred_arr / pred_arr.sum(axis=1, keepdims=True)

        true_arr = val[TARGETS].values
        score = np.mean(kl_div(true_arr, pred_arr))
        if score < best_score:
            best_score = score
            best_w = w

    fine_start = max(0.0, best_w - 0.1)
    fine_end = min(1.0, best_w + 0.1)
    for w in np.arange(fine_start, fine_end + 0.001, 0.01):
        preds_val = val[["eeg_id", "patient_id"]].merge(
            patient_means_fold, left_on="patient_id", right_index=True, how="left"
        )
        preds_val[TARGETS] = preds_val[TARGETS].fillna(
            {col: global_means_fold[col] for col in TARGETS}
        )
        for col in TARGETS:
            preds_val[col] = w * preds_val[col] + (1 - w) * global_means_fold[col]

        epsilon = 1e-6
        pred_arr = preds_val[TARGETS].values + epsilon
        pred_arr = pred_arr / pred_arr.sum(axis=1, keepdims=True)

        true_arr = val[TARGETS].values
        score = np.mean(kl_div(true_arr, pred_arr))
        if score < best_score:
            best_score = score
            best_w = w

    extra_start = max(0.0, best_w - 0.02)
    extra_end = min(1.0, best_w + 0.02)
    for w in np.arange(extra_start, extra_end + 0.0001, 0.005):
        preds_val = val[["eeg_id", "patient_id"]].merge(
            patient_means_fold, left_on="patient_id", right_index=True, how="left"
        )
        preds_val[TARGETS] = preds_val[TARGETS].fillna(
            {col: global_means_fold[col] for col in TARGETS}
        )
        for col in TARGETS:
            preds_val[col] = w * preds_val[col] + (1 - w) * global_means_fold[col]

        epsilon = 1e-6
        pred_arr = preds_val[TARGETS].values + epsilon
        pred_arr = pred_arr / pred_arr.sum(axis=1, keepdims=True)

        true_arr = val[TARGETS].values
        score = np.mean(kl_div(true_arr, pred_arr))
        if score < best_score:
            best_score = score
            best_w = w

    best_smooth = 0.0
    best_score_smooth = best_score
    for smooth in np.arange(0.0, 0.0201, 0.0005):
        preds_val = val[["eeg_id", "patient_id"]].merge(
            patient_means_fold, left_on="patient_id", right_index=True, how="left"
        )
        preds_val[TARGETS] = preds_val[TARGETS].fillna(
            {col: global_means_fold[col] for col in TARGETS}
        )
        for col in TARGETS:
            preds_val[col] = (
                best_w * preds_val[col] + (1 - best_w) * global_means_fold[col]
            )

        pred_arr = preds_val[TARGETS].values + smooth
        pred_arr = pred_arr / pred_arr.sum(axis=1, keepdims=True)

        true_arr = val[TARGETS].values
        score = np.mean(kl_div(true_arr, pred_arr))
        if score < best_score_smooth:
            best_score_smooth = score
            best_smooth = smooth

    best_T = 1.0
    best_score_T = best_score_smooth
    for T in np.arange(0.5, 5.01, 0.1):
        preds_val = val[["eeg_id", "patient_id"]].merge(
            patient_means_fold, left_on="patient_id", right_index=True, how="left"
        )
        preds_val[TARGETS] = preds_val[TARGETS].fillna(
            {col: global_means_fold[col] for col in TARGETS}
        )
        for col in TARGETS:
            preds_val[col] = (
                best_w * preds_val[col] + (1 - best_w) * global_means_fold[col]
            )

        pred_arr = preds_val[TARGETS].values + best_smooth
        pred_arr = pred_arr / pred_arr.sum(axis=1, keepdims=True)

        pred_arr = np.power(pred_arr, 1.0 / T)
        pred_arr = pred_arr / pred_arr.sum(axis=1, keepdims=True)

        pred_arr = np.clip(pred_arr, 1e-12, None)

        true_arr = val[TARGETS].values
        score = np.mean(kl_div(true_arr, pred_arr))
        if score < best_score_T:
            best_score_T = score
            best_T = T

    print(
        f"Chosen patient weight: {best_w:.3f} (KL={best_score:.5f}); "
        f"smoothing: {best_smooth:.4f} (KL={best_score_smooth:.5f}); "
        f"temperature: {best_T:.2f} (KL={best_score_T:.5f})"
    )

    if PLATFORM == "local":
        test = pd.read_csv("./input/hms-harmful-brain-activity-classification/test.csv")
    elif PLATFORM == "kaggle":
        test = pd.read_csv(
            "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
        )
    print("Test shape", test.shape)
    test.head()

    if TF_AVAILABLE:
        pass
    else:
        print(
            "TensorFlow not available – using improved patient‑aware baseline with tuned weight, smoothing and temperature scaling."
        )
        global_means = train[TARGETS].mean().to_dict()
        patient_means = train.groupby("patient_id")[TARGETS].mean()

        test_preds = test[["eeg_id", "patient_id"]].merge(
            patient_means, left_on="patient_id", right_index=True, how="left"
        )
        test_preds[TARGETS] = test_preds[TARGETS].fillna(
            {col: global_means[col] for col in TARGETS}
        )

        for col in TARGETS:
            test_preds[col] = (
                best_w * test_preds[col] + (1 - best_w) * global_means[col]
            )

        pred_array = test_preds[TARGETS].values + best_smooth
        pred_array = pred_array / pred_array.sum(axis=1, keepdims=True)

        pred_array = np.power(pred_array, 1.0 / best_T)
        pred_array = pred_array / pred_array.sum(axis=1, keepdims=True)

        pred_array = np.clip(pred_array, 1e-12, None)
        pred_array = pred_array / pred_array.sum(axis=1, keepdims=True)

        pred = pred_array  # final numpy array (len(test), 6)

    sub = pd.DataFrame({"eeg_id": test.eeg_id.values})
    sub[TARGETS] = pred
    sub.to_csv("submission.csv", index=False)
    print("Submission shape", sub.shape)
    print("Row sums (should be ~1.0):")
    print(sub.iloc[:, -6:].sum(axis=1).head())
