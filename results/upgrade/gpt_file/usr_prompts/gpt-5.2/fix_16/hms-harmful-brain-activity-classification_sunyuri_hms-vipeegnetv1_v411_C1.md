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

0.2783055038141946

# 6. Current score

1.4782

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'I fix the runtime crash by removing the unused `keras_hub` import (it triggers the protobuf `MessageFactory.GetPrototype` error in this Kaggle environment) and by avoiding TensorFlow determinism calls that are not consistently supported across builds. I also make inference robust on Kaggle by ensuring `TARGETS` is loaded from `sample_submission.csv` (so it’s always defined even when `NEEDTRAIN=False`) and by providing a safe fallback when no model weights are found (uniform probabilities that still pass submission validation). Finally, I correct the test batch slicing bug that could produce empty/incorrect batches and ensure predictions are clipped/renormalized to sum to 1 for every row, producing a valid `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation before importing TensorFlow (this is the common root cause in Kaggle for that exact error). I also ensure inference always defines `TARGETS` (even when `NEEDTRAIN=False`) by loading them from `sample_submission.csv` unconditionally, so the submission columns are always correct. Finally, I add a small safeguard so test-time filtering doesn’t reference undefined filter coefficients when `NEEDTRAIN=False`, while keeping the model/training logic unchanged and still writing a valid `submission.csv` with row-wise probabilities summing to 1.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf runtime *before any protobuf/TensorFlow import* and by avoiding TF determinism calls that can trigger this failure in Kaggle’s TF build. I also make the inference path robust and scoring-improving by loading the provided fold weights correctly (most public model datasets use `fold{i}_stage2.weights.h5`, not `fold{i}_stage2...` for i up to 99) and by ensuring we actually run model inference instead of falling back to uniform predictions (which is consistent with your current high KL score). Finally, I keep submission formatting strict: correct column names from `sample_submission.csv`, probabilities clipped and renormalized per row, and always write `submission.csv`.'
- What this solution (achieved 1.40995) has done: 'I fix the protobuf/TensorFlow crash (`MessageFactory` has no `GetPrototype`) by forcing the pure-Python protobuf runtime early and (critically) disabling C++ protobuf via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and also preventing TensorFlow from importing the C++ implementation. Next, I ensure inference actually runs the model instead of accidentally falling back to uniform predictions by robustly locating weight files and compiling the cloned models before `predict()` (needed on some TF/Keras builds). Finally, I keep submission validity strict by clipping + renormalizing probabilities to sum to 1 and writing `submission.csv` with the exact sample submission columns/order.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any TensorFlow/protobuf-related import* and by avoiding the determinism call that can trigger the same issue on Kaggle builds. Then I fix the uniform-submission fallback bug: the current merge logic explodes the row count and causes a length mismatch when assigning probabilities; I instead write the submission by directly aligning to `test.csv` order and enforcing the exact sample-submission columns. These changes are execution-unblocking and score-neutral (they just ensure the model can run and that a valid `submission.csv` is always produced). Finally, I also make the inference path load test EEGs directly (minimal, only when needed) so model inference can actually run on Kaggle without requiring an unavailable `/kaggle/input/preprocess/eegs.npy`.'
- What this solution (achieved 1.40995) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *and* disabling the C++ protobuf backend before any TensorFlow/protobuf import happens. Then, because your current KL score (1.40995) indicates the script is likely falling back to uniform predictions on Kaggle (no weights found), I make the weight discovery more robust by also checking common weight locations under `/kaggle/input/` and by searching recursively for `fold*_stage*.weights.h5` files. Finally, I ensure inference has all required test inputs by loading EEG parquet files on-the-fly (since `spectrograms_test`/`stfts_test`/`imgs_test` are empty for `DATATYPE=["eeg"]`), and keep the submission strictly valid (correct columns, clipped and row-normalized probabilities).'
- What this solution (achieved 1.40995) has done: 'We fix the immediate TensorFlow/protobuf crash by ensuring the pure-Python protobuf runtime is enforced before any TensorFlow import and by importing `google.protobuf` early (this is the common root cause of the `MessageFactory.GetPrototype` failure on Kaggle). Then, to materially reduce the KL score from the current uniform-like behavior (1.40995) toward the target, we make inference actually use the available EEG inputs by loading test EEG parquets on-the-fly (since `NEEDTRAIN=False` on Kaggle and no preprocessed arrays are guaranteed). Finally, we harden the submission writing so probabilities are always finite, clipped, and row-normalized to sum to 1 with the exact sample-submission column order.'
- What this solution (achieved 1.03911) has done: 'I fix the protobuf/TensorFlow crash (`MessageFactory` missing `GetPrototype`) by avoiding the problematic protobuf/TensorFlow import path entirely and switching to a lightweight, deterministic scikit-learn baseline that fits within the Kaggle runtime and does not require TensorFlow at all. To improve the KL score from the current near-uniform behavior toward the target, the patch train a simple multinomial Logistic Regression on aggregated spectrogram features per `spectrogram_id`, using properly normalized vote targets (probabilities). Finally, I ensure the submission is strictly valid: correct columns/order from `sample_submission.csv`, per-row probabilities clipped and renormalized to sum to 1, and saved as `submission.csv`.'
- What this solution (achieved 1.01246) has done: 'Your current score is far worse than the target (lower is better), so we should make the smallest change that aligns training with the KL metric without changing the overall approach. The main issue is that you convert the soft vote targets into a hard class (`argmax`) and train standard multinomial logistic regression, which optimizes log-loss for hard labels rather than KL to soft labels. We can keep the same feature extraction and the same LogisticRegression model, but train it in a “soft-label” way by expanding each spectrogram into per-class pseudo-samples weighted by the vote probabilities via `sample_weight`. This typically improves KL substantially while remaining lightweight and within the same core logic.'
- What this solution (achieved 1.05357) has done: 'I fix the FileNotFoundError by stopping the code from trying to read test spectrogram IDs out of the *train* spectrogram folder (that’s what `X_all = build_feature_matrix(all_sids, is_train=True)` is doing) and by building features separately for train/test spectrogram folders only. I also remove the unused `X_all` creation (it’s not used later and is what crashes), so `X_train`/`X_test` are always defined and downstream cells no longer hit `NameError`. Finally, I add a tiny robustness check so if any spectrogram parquet is unexpectedly missing, we fall back to an all-zero feature vector instead of crashing, ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 1.47043) has done: 'Your current KL (1.05357, lower is better) is far from the target (0.2783), so we should make a small, metric-aligned improvement without changing the overall “spectrogram-aggregate features + multinomial LogisticRegression” approach. The biggest easy gain here is to ensure the model always sees comparable train/test spectrogram feature scales by standardizing *within the spectrogram feature extraction* (per-parquet z-scoring across time) before computing summary statistics; this reduces train/test distribution shift and typically improves KL. I also add a tiny ridge regularization in the scaler stage (numerical stability for near-constant features) and set `class_weight=None` explicitly to avoid any accidental reweighting that fights the soft-label sample weights. Submission writing stays identical, including clipping + row-normalization to sum to 1.'
- What this solution (achieved 1.48342) has done: 'Your current score is much worse than the target (lower is better), so we should make a small, metric-aligned improvement without changing the overall “spectrogram summary features → scaler → multinomial LogisticRegression with soft-label sample_weight” core approach. The biggest issue here is label/feature mismatch: you aggregate targets at the `eeg_id` level, but your features are averaged over *all* spectrograms for that EEG (including non-central/overlapping windows), which adds noise and hurts KL. I keep the same feature function and model, but change the training feature aggregation to use only the single “best” spectrogram per `eeg_id` (the one with the highest total votes / strongest consensus), while leaving test aggregation unchanged (test has one row per EEG). This is a minimal change that typically reduces KL by aligning training inputs more tightly with the labels.'
- What this solution (achieved 1.4782) has done: 'Your current KL is far worse than the target (lower is better), so we should make a minimal, metric-aligned change that reduces noise in the training signal without changing the core “spectrogram summary features → scaler → multinomial LogisticRegression with soft-label sample_weight” pipeline. The biggest issue is a mismatch between how features are chosen (single best spectrogram per `eeg_id`) and how targets are constructed (sum of votes across all overlapping segments), which injects label noise. I rebuild the training targets to match the exact same “best row per `eeg_id`” selection already used for features, keeping everything else (feature extraction, scaler, logistic regression, soft-label expansion, submission formatting) identical. This typically improves KL materially while staying within minimal-change constraints and still producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
"""
Created on Tue Oct 22 20:48:49 2024

@author: yuri
email: syuri@tju.edu.cn
"""

import os
import gc
import time
import numpy as np
import pandas as pd


SEED = 2024
np.random.seed(SEED)

if os.getcwd().split(os.sep)[1] == "home":
    PLATFORM = "local"
    LOAD_DATA_FROM = "./input/hms-harmful-brain-activity-classification"
else:
    PLATFORM = "kaggle"
    LOAD_DATA_FROM = "/kaggle/input/hms-harmful-brain-activity-classification"

sample_sub_path = os.path.join(LOAD_DATA_FROM, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)
TARGETS = list(sample_sub.columns[1:])
print("Targets:", TARGETS)

train_path = os.path.join(LOAD_DATA_FROM, "train.csv")
test_path = os.path.join(LOAD_DATA_FROM, "test.csv")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
print("Train shape:", train_df.shape, "Test shape:", test_df.shape)

SPECTRO_PATH_TRAIN = os.path.join(LOAD_DATA_FROM, "train_spectrograms")
SPECTRO_PATH_TEST = os.path.join(LOAD_DATA_FROM, "test_spectrograms")




## === cell 1
def _read_spectrogram(spectrogram_id: int, is_train: bool):
    base = SPECTRO_PATH_TRAIN if is_train else SPECTRO_PATH_TEST
    p = os.path.join(base, f"{int(spectrogram_id)}.parquet")
    return pd.read_parquet(p)


def spectro_features_from_parquet(df_parquet: pd.DataFrame) -> np.ndarray:
    arr = df_parquet.iloc[:, 1:].to_numpy(dtype=np.float32, copy=False)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    arr = np.log1p(np.clip(arr, 0.0, None))

    mu_t = arr.mean(axis=0, keepdims=True)
    sig_t = arr.std(axis=0, keepdims=True)
    arr = (arr - mu_t) / (sig_t + 1e-6)

    mean_cols = arr.mean(axis=0)
    std_cols = arr.std(axis=0)

    gmean = arr.mean()
    gstd = arr.std()

    q = np.quantile(arr.reshape(-1), [0.1, 0.5, 0.9]).astype(np.float32)

    feat = np.concatenate(
        [mean_cols, std_cols, np.array([gmean, gstd], np.float32), q], axis=0
    )
    return feat.astype(np.float32)


def build_feature_matrix(ids, is_train: bool, verbose_every=250):
    feats = []
    t0 = time.time()
    for i, sid in enumerate(ids):
        if i % verbose_every == 0:
            gc.collect()
            dt = time.time() - t0
            print(
                f"Features ({'train' if is_train else 'test'}): {i}/{len(ids)} loaded, {dt/60:.2f} min"
            )

        try:
            pq = _read_spectrogram(sid, is_train=is_train)
            feats.append(spectro_features_from_parquet(pq))
        except FileNotFoundError:
            feats.append(None)

    first = None
    for f in feats:
        if f is not None:
            first = f
            break
    if first is None:
        raise RuntimeError(
            "No spectrogram parquet files could be read; cannot build features."
        )

    feat_dim_local = first.shape[0]
    out = np.zeros((len(feats), feat_dim_local), dtype=np.float32)
    for i, f in enumerate(feats):
        if f is not None:
            out[i] = f
    return out


train_meta = train_df[["eeg_id", "spectrogram_id"] + TARGETS].copy()
test_meta = test_df[["eeg_id", "spectrogram_id"]].copy()

train_meta["_vote_sum"] = train_meta[TARGETS].sum(axis=1)
best_train_rows = (
    train_meta.sort_values(["eeg_id", "_vote_sum"], ascending=[True, False])
    .drop_duplicates("eeg_id", keep="first")[["eeg_id", "spectrogram_id"] + TARGETS]
    .copy()
)

y_votes = best_train_rows[TARGETS].to_numpy(dtype=np.float32)
y_prob = y_votes / np.clip(y_votes.sum(axis=1, keepdims=True), 1e-6, None)
best_train_rows[TARGETS] = y_prob

best_train_rows = best_train_rows.sort_values("eeg_id").reset_index(drop=True)
train_eeg_ids = best_train_rows["eeg_id"].values

test_eeg_ids = sample_sub["eeg_id"].values  # enforce submission order

train_sids_unique = pd.Index(train_meta["spectrogram_id"].unique())
test_sids_unique = pd.Index(test_meta["spectrogram_id"].unique())

print(
    "Unique train spectrograms:",
    len(train_sids_unique),
    "Unique test spectrograms:",
    len(test_sids_unique),
)

X_train_sids = build_feature_matrix(
    train_sids_unique.values, is_train=True, verbose_every=300
)
X_test_sids = build_feature_matrix(
    test_sids_unique.values, is_train=False, verbose_every=300
)

sid_feat = {}
for sid, feat in zip(train_sids_unique.values, X_train_sids):
    sid_feat[int(sid)] = feat
for sid, feat in zip(test_sids_unique.values, X_test_sids):
    sid_feat[int(sid)] = feat  # overwrite not expected; safe if any overlap

feat_dim = next(iter(sid_feat.values())).shape[0]
print("Feature dim:", feat_dim)


def build_eeg_feature_matrix(meta_df: pd.DataFrame, eeg_ids: np.ndarray) -> np.ndarray:
    groups = meta_df.groupby("eeg_id")["spectrogram_id"].apply(list).to_dict()
    X = np.zeros((len(eeg_ids), feat_dim), dtype=np.float32)
    for i, eid in enumerate(eeg_ids):
        sids = groups.get(eid, [])
        if not sids:
            continue
        mats = [sid_feat[int(s)] for s in sids if int(s) in sid_feat]
        if not mats:
            continue
        X[i] = np.mean(np.stack(mats, axis=0), axis=0)
    return X


X_train = build_eeg_feature_matrix(
    best_train_rows[["eeg_id", "spectrogram_id"]], train_eeg_ids
)

X_test = build_eeg_feature_matrix(test_meta[["eeg_id", "spectrogram_id"]], test_eeg_ids)

print("X_train:", X_train.shape, "X_test:", X_test.shape)




## === cell 2
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

y_soft = best_train_rows[TARGETS].to_numpy(dtype=np.float32)
K = len(TARGETS)
n = X_train.shape[0]

scaler = StandardScaler(with_mean=True, with_std=True)
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

X_rep = np.repeat(X_train_s, repeats=K, axis=0)
y_rep = np.tile(np.arange(K, dtype=np.int32), reps=n)
w_rep = y_soft.reshape(-1).astype(np.float64, copy=False)

nz = w_rep > 0.0
X_rep = X_rep[nz]
y_rep = y_rep[nz]
w_rep = w_rep[nz]

clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    C=1.0,
    max_iter=400,
    n_jobs=1,
    random_state=SEED,
    class_weight=None,  # explicit: rely on soft-label sample_weight only
)
clf.fit(X_rep, y_rep, sample_weight=w_rep)

probs_test = clf.predict_proba(X_test_s).astype(np.float32)

probs_full = np.full((len(test_eeg_ids), K), 1.0 / K, dtype=np.float32)
for j, c in enumerate(getattr(clf, "classes_", np.arange(probs_test.shape[1]))):
    c = int(c)
    if 0 <= c < K:
        probs_full[:, c] = probs_test[:, j]

probs_full = np.nan_to_num(probs_full, nan=1.0 / K, posinf=1.0, neginf=0.0)
probs_full = np.clip(probs_full, 1e-8, 1.0)
probs_full = probs_full / probs_full.sum(axis=1, keepdims=True)

print(
    "Pred probs shape:",
    probs_full.shape,
    "row-sum min/max:",
    probs_full.sum(1).min(),
    probs_full.sum(1).max(),
)




## === cell 3
sub = pd.DataFrame({"eeg_id": test_eeg_ids})
for k, col in enumerate(TARGETS):
    sub[col] = probs_full[:, k]

sub = sub[["eeg_id"] + TARGETS]

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote", out_path, "shape", sub.shape)
print(sub.head())
