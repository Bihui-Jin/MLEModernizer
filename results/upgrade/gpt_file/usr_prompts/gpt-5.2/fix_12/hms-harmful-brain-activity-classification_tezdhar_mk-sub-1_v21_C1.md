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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

0.3256166983473387

# 6. Current score

0.81463

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'I make the pipeline robust to missing external datasets/modules by (1) auto-detecting whether the MK codebase and checkpoints actually exist and only running them if they do, and (2) providing a fast, deterministic fallback that produces a valid submission when inference can’t run. The main bug is the fold inference `os.system` call failing (exit code 512) because the referenced `/kaggle/input/hms-mk-data/...` and/or `/kaggle/input/hms-mk-codes` aren’t available in your environment; this currently prevents any CSV from being generated. I also fix the cell numbering to start at 1 and ensure we always write `/kaggle/working/submission.csv` with the exact required columns and per-row normalization. The fallback uses the training label distribution as a prior (score-neutral baseline), guaranteeing a valid CSV so you can submit and then iterate when the missing inputs are restored.'
- What this solution (achieved 0.73577) has done: 'Your current score indicates the external MK inference isn’t being used (or isn’t aligned), so the simplest way to move toward the target is to make the fallback stronger without changing the overall approach: keep it as a “no-model” baseline but condition the class-prior by `patient_id` (using train metadata) and apply light additive smoothing so probabilities are never too sharp or too flat. This stays within your existing logic (still just priors from train votes), but usually reduces KL versus a single global prior because label distributions vary by patient. I also harden ID alignment (unique `eeg_id` in submission order) and ensure probabilities are clipped and renormalized exactly once at the end to avoid numerical drift. Paths and submission schema remain unchanged and it still write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.77275) has done: 'Your current score (0.73577, lower-is-better) is far from the target (0.3256), and since the external MK inference isn’t available, the only legitimate way to move toward the target without changing core modeling is to make the metadata-prior baseline more informative. I keep the same “patient-conditioned prior from train votes” core idea, but add a second conditioning signal using `spectrogram_id` (also available in test/train metadata) and then blend patient- and spectrogram-priors with a small weight toward the more specific one when it has enough support. I also compute priors using total vote counts aggregated per group (more statistically stable than averaging per-row normalized distributions when vote totals vary), while keeping the same smoothing and final normalization so the submission stays valid. Paths, output schema, and fallback/inference branching remain unchanged.'
- What this solution (achieved 0.81232) has done: 'Your current score (0.77275, lower-is-better) is far from the target (0.3256), and since external inference isn’t available, the only safe lever is improving the metadata-prior baseline without changing its core “priors from train vote counts” logic. I keep the same patient+spectrogram Dirichlet-smoothed priors, but make the blend weight depend on actual vote mass (total annotator votes) rather than row counts, which better matches the competition target distribution and usually reduces KL. I also add a small “pooling” step that mixes each group prior slightly back toward the global prior when group vote mass is low (hierarchical shrinkage), which stabilizes rare patients/spectrograms. Finally, I keep the same strict normalization/clipping and submission alignment to avoid format-induced score regressions.'
- What this solution (achieved 0.81232) has done: 'Your score is much worse than the target (lower-is-better), so the smallest safe move is to keep your existing “metadata priors from train vote counts” logic but make it closer to the true test distribution by conditioning more directly on `eeg_id` whenever possible. Specifically, we add an `eeg_id`-level Dirichlet-smoothed prior (aggregated from all train rows sharing that `eeg_id`) and blend it with your existing patient+spectrogram priors using vote-mass-based weights, falling back smoothly when an `eeg_id` is unseen. This preserves your core approach (no model, just smoothed priors from vote counts), but typically reduces KL substantially because many test EEGs share IDs/patients/spectrograms with train and `eeg_id` is the most specific grouping. We also keep your strict normalization/clipping and submission alignment unchanged to avoid format-related regressions.'
- What this solution (achieved 0.81232) has done: 'Your current score (0.81232, lower-is-better) is far from the target (0.3256), so we should improve the fallback priors without changing the overall “smoothed metadata priors from train vote counts” approach. The smallest high-impact fix is to avoid overfitting/sharpness from directly using (train-derived) `eeg_id` priors on test: instead compute `eeg_id` priors in a patient-grouped out-of-fold manner (LOO-within-patient), so each `eeg_id`’s prior is built from *other* eegs of the same patient and then blended as before. This preserves the same core logic (Dirichlet-smoothed count priors + hierarchical shrinkage + blending), but reduces leakage-like brittleness and typically improves KL on this competition. I also keep external inference untouched and only apply the improved baseline when external checkpoints aren’t available.'
- What this solution (achieved 0.8144) has done: 'Your current KL (0.81232, lower-is-better) is far from the target (0.3256), so we should keep the same “smoothed priors from train vote counts” fallback but make it less overconfident and more test-like. The smallest high-impact change is to stop relying on exact ID matches (which are rare for `eeg_id`/`spectrogram_id` in test) and instead add a patient-based similarity prior: for each test patient, blend priors from the most similar *train* patients (by their class-distribution), which stays within your core logic (still just vote-count priors + smoothing + blending). To avoid regressions, we keep your existing patient/spectrogram priors and simply add this kNN-patient prior as an additional blended component, with conservative weights and the same final clipping/normalization. External MK inference remains untouched; this only improves the fallback path.'
- What this solution (achieved 0.8144) has done: 'Your current score (0.8144, lower-is-better) is far from the target (0.3256), so we should improve the fallback priors while keeping the same “smoothed metadata vote-count priors + blending” core logic. The smallest high-impact fix is to correct a key mismatch: most `patient_id` and `spectrogram_id` values are not integers in this dataset, so casting to `int` silently breaks almost all lookups and forces fallback to the global prior (hurting KL). I keep all your priors, smoothing, shrinkage, and blending unchanged, but switch all ID keys/maps to use the native dtypes consistently (no `int()` casting), and make the indexing/alignments robust to dtype differences. This should materially improve score without changing evaluation semantics, and it still writes a valid `/kaggle/working/submission.csv` with per-row normalization.'
- What this solution (achieved 0.81463) has done: 'Your current score (0.8144, lower-is-better) is far from the target (0.3256), so we should improve the fallback path while preserving the same “Dirichlet-smoothed priors from train vote counts + shrinkage + blending” core logic. The most likely reason your recent “key mismatch fix” didn’t help is that `patient_id`/`spectrogram_id` are being read with inconsistent dtypes between train/test (often causing near-total misses in the maps), and the kNN prior is built for train patients but not applied correctly to unseen test patients. I make ID dtypes consistent across train/test (string keys everywhere) and extend the kNN mapping to produce priors for *unseen* test patients by comparing each test patient’s *patient-level* distribution (from any available history in train; otherwise global) against train-patient priors—this keeps the same “metadata prior” approach but makes the kNN actually useful for test. I also keep the blending structure intact and only adjust lookup robustness and the kNN coverage so more rows avoid falling back to the global prior.'
- What this solution (achieved 0.81463) has done: 'I fix the submission schema mismatch by ensuring we always output exactly the competition’s required column names/order (those in `sample_submission.csv`), regardless of what `MK_CODES_DIR` provides. I keep your existing inference/fallback logic intact, but standardize on a single `SUB_COLS` list derived from the sample submission and use it consistently for training aggregation, blending, and final CSV writing. This is primarily a correctness/stability fix so the notebook produces a valid `submission.csv` end-to-end; it should be score-neutral aside from preventing accidental column misalignment. I also add a final defensive reindex to the sample submission columns before writing.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import subprocess
import pandas as pd
import numpy as np

MK_CODES_DIR = "/kaggle/input/hms-mk-codes"
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

SAMPLE_SUB_PATH = f"{DATA_PATH}/sample_submission.csv"
TEST_CSV_PATH = f"{DATA_PATH}/test.csv"
TRAIN_CSV_PATH = f"{DATA_PATH}/train.csv"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if "eeg_id" not in sample_sub.columns:
    raise ValueError("sample_submission.csv must contain eeg_id")
SUB_COLS = ["eeg_id"] + [c for c in sample_sub.columns if c != "eeg_id"]
TARGET_COLS = SUB_COLS[1:]  # enforce exact submission target columns/order

print("TARGET_COLS (from sample_submission) =", TARGET_COLS)
print("MK_CODES_DIR exists:", os.path.isdir(MK_CODES_DIR))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB_PATH))
print("Train exists:", os.path.exists(TRAIN_CSV_PATH))
print("Test exists:", os.path.exists(TEST_CSV_PATH))




## === cell 1
def run_parquet_to_npy_conversion():
    if not os.path.isdir(MK_CODES_DIR):
        print(f"MK_CODES_DIR not found ({MK_CODES_DIR}); skipping conversion.")
        return False

    expected_any = glob.glob(os.path.join(OUT_PATH, "*.npy"))
    if len(expected_any) > 0:
        print(
            f"Found {len(expected_any)} .npy files in {OUT_PATH}, skipping conversion."
        )
        return True

    cmd = [
        sys.executable,
        "-m",
        "src.convert_parquet_to_npy",
        f"--data_dir={DATA_PATH}",
        f"--out_dir={OUT_PATH}",
    ]
    print("Running conversion:", " ".join(cmd), " (cwd=", MK_CODES_DIR, ")")
    try:
        subprocess.run(cmd, cwd=MK_CODES_DIR, check=True)
        return True
    except Exception as e:
        print("Conversion failed; will proceed without .npy files. Error:", repr(e))
        return False


_ = run_parquet_to_npy_conversion()




## === cell 2
def try_run_external_inference():
    if not os.path.isdir(MK_CODES_DIR):
        print(
            f"MK_CODES_DIR not found ({MK_CODES_DIR}); cannot run external inference."
        )
        return False

    ckpts = [
        "/kaggle/input/hms-mk-data/fold0_pseudo_resv2.ckpt",
        "/kaggle/input/hms-mk-data/fold1_pseudo_resv2.ckpt",
        "/kaggle/input/hms-mk-data/fold2_pseudo_resv2.ckpt",
        "/kaggle/input/hms-mk-data/fold3_pseudo_resv2.ckpt",
        "/kaggle/input/hms-mk-data/fold4_pseudo_resv2.ckpt",
    ]
    missing_ckpts = [p for p in ckpts if not os.path.exists(p)]
    if missing_ckpts:
        print(
            "Missing checkpoints; cannot run external inference. Missing (showing up to 3):",
            missing_ckpts[:3],
        )
        return False

    for fold, ckpt in enumerate(ckpts):
        out_csv = f"/kaggle/working/submission_fold{fold}_v2.csv"
        if os.path.exists(out_csv):
            print(f"Exists, skipping fold {fold}:", out_csv)
            continue

        cmd = [
            sys.executable,
            "-m",
            "test",
            f"paths.data_dir={DATA_PATH}",
            f"data.test_eegs_dir={OUT_PATH}",
            f"ckpt_path={ckpt}",
            "hydra=test",
            f"+model.test_output_dir={OUT_PATH}",
            "experiment=conv1d_resv2",
            "+model.net.pretrained=False",
        ]
        print("Running inference:", " ".join(cmd), " (cwd=", MK_CODES_DIR, ")")
        try:
            subprocess.run(cmd, cwd=MK_CODES_DIR, check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"Inference failed for fold={fold} with returncode={e.returncode}. Falling back."
            )
            return False

        src_csv = "/kaggle/working/submission.csv"
        if not os.path.exists(src_csv):
            print(f"Expected {src_csv} was not created for fold={fold}. Falling back.")
            return False

        os.replace(src_csv, out_csv)
        print("Wrote:", out_csv)

    return True


external_ok = try_run_external_inference()
print("external_ok =", external_ok)




## === cell 3
def _finalize_probs(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype="float64")
    mat = np.clip(mat, 1e-12, 1.0)
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat


def merge_preds(folds=(0, 1, 2, 3, 4), version="v2"):
    base = pd.read_csv(SAMPLE_SUB_PATH)
    base = base[SUB_COLS].copy()  # enforce required schema/order

    base_ids = base["eeg_id"].values
    all_fold_preds = []

    for fold in folds:
        path = f"/kaggle/working/submission_fold{fold}_{version}.csv"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing fold submission: {path}")

        df = pd.read_csv(path)
        missing = [c for c in (["eeg_id"] + TARGET_COLS) if c not in df.columns]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")

        df = df.set_index("eeg_id").loc[base_ids, TARGET_COLS].astype("float64")
        all_fold_preds.append(df.values)

    preds = np.mean(np.stack(all_fold_preds, axis=0), axis=0)
    preds = _finalize_probs(preds)

    out = base.copy()
    out[TARGET_COLS] = preds
    return out


def _dirichlet_smooth_from_counts(
    counts: pd.DataFrame, alpha: float, global_prior: np.ndarray
) -> pd.DataFrame:
    cnt = counts.sum(axis=1).astype("float64").values.reshape(-1, 1)
    raw = counts[TARGET_COLS].astype("float64").values
    smooth = raw + alpha * global_prior.reshape(1, -1)
    denom = cnt + alpha
    denom = np.where(denom == 0, 1.0, denom)
    out = smooth / denom
    out = _finalize_probs(out)
    return pd.DataFrame(out, index=counts.index, columns=TARGET_COLS)


def _shrink_group_prior_to_global(
    group_prior_df: pd.DataFrame,
    group_vote_mass: pd.Series,
    global_prior: np.ndarray,
    tau: float,
) -> pd.DataFrame:
    m = (
        group_vote_mass.astype("float64")
        .reindex(group_prior_df.index)
        .fillna(0.0)
        .values
    )
    w = (m / (m + float(tau))).reshape(-1, 1)
    out = w * group_prior_df.values + (1.0 - w) * global_prior.reshape(1, -1)
    out = _finalize_probs(out)
    return pd.DataFrame(out, index=group_prior_df.index, columns=TARGET_COLS)


def _make_patient_loo_eeg_priors(
    train_meta_votes: pd.DataFrame,
    alpha: float,
    global_prior: np.ndarray,
    tau: float,
) -> tuple[dict, dict]:
    eeg_counts = train_meta_votes.groupby("eeg_id")[TARGET_COLS].sum()
    eeg_patient = train_meta_votes.groupby("eeg_id")["patient_id"].first()
    patient_counts = train_meta_votes.groupby("patient_id")[TARGET_COLS].sum()

    eeg_prior_map = {}
    eeg_mass_map = {}

    for eeg_id, pid in eeg_patient.items():
        p_cnt = patient_counts.loc[pid].astype("float64").values
        e_cnt = eeg_counts.loc[eeg_id].astype("float64").values
        loo_cnt = p_cnt - e_cnt
        loo_cnt = np.clip(loo_cnt, 0.0, None)
        mass = float(loo_cnt.sum())

        post = loo_cnt + alpha * global_prior
        denom = mass + alpha
        if denom <= 0:
            prior = global_prior.copy()
        else:
            prior = post / denom
        prior = _finalize_probs(prior.reshape(1, -1)).reshape(-1)

        w = mass / (mass + float(tau))
        prior = _finalize_probs(
            (w * prior + (1.0 - w) * global_prior).reshape(1, -1)
        ).reshape(-1)

        eeg_prior_map[eeg_id] = prior
        eeg_mass_map[eeg_id] = mass

    return eeg_prior_map, eeg_mass_map


def _make_patient_knn_prior_map_train_only(
    patient_prior_df: pd.DataFrame,
    patient_vote_mass: pd.Series,
    global_prior: np.ndarray,
    k: int = 32,
    tau: float = 120.0,
) -> dict:
    idx = patient_prior_df.index.values
    X = patient_prior_df.values.astype("float64")
    mass = (
        patient_vote_mass.reindex(patient_prior_df.index)
        .fillna(0.0)
        .values.astype("float64")
    )

    X = _finalize_probs(X)

    knn_map = {}
    x2 = (X * X).sum(axis=1, keepdims=True)  # (n,1)
    XT = X.T  # (c,n)

    n = X.shape[0]
    k_eff = int(min(max(k, 1), max(n - 1, 1)))

    for i in range(n):
        dots = (X[i : i + 1, :] @ XT).reshape(-1, 1)  # (n,1)
        d2 = (x2 + x2[i] - 2.0 * dots).reshape(-1)  # (n,)
        d2[i] = np.inf  # exclude self

        nn = np.argpartition(d2, k_eff)[:k_eff]
        w = mass[nn].copy()
        if not np.isfinite(w).all() or w.sum() <= 0:
            nn_prior = global_prior.copy()
            support = 0.0
        else:
            nn_prior = (X[nn] * w.reshape(-1, 1)).sum(axis=0) / w.sum()
            nn_prior = _finalize_probs(nn_prior.reshape(1, -1)).reshape(-1)
            support = float(w.sum())

        mix = support / (support + float(tau))
        out = _finalize_probs(
            (mix * nn_prior + (1.0 - mix) * global_prior).reshape(1, -1)
        ).reshape(-1)
        knn_map[idx[i]] = out

    return knn_map


def _knn_predict_for_queries(
    X_train: np.ndarray,
    mass_train: np.ndarray,
    queries: np.ndarray,
    global_prior: np.ndarray,
    k: int = 32,
    tau: float = 120.0,
) -> np.ndarray:
    X_train = _finalize_probs(X_train.astype("float64"))
    queries = _finalize_probs(queries.astype("float64"))

    x2 = (X_train * X_train).sum(axis=1, keepdims=True)  # (n,1)
    XT = X_train.T  # (c,n)

    n = X_train.shape[0]
    k_eff = int(min(max(k, 1), max(n, 1)))

    out = np.zeros((queries.shape[0], X_train.shape[1]), dtype="float64")
    for i in range(queries.shape[0]):
        dots = (queries[i : i + 1, :] @ XT).reshape(-1, 1)  # (n,1)
        d2 = (x2 - 2.0 * dots).reshape(-1) + float((queries[i] * queries[i]).sum())
        nn = np.argpartition(d2, k_eff - 1)[:k_eff]
        w = mass_train[nn].copy()
        if not np.isfinite(w).all() or w.sum() <= 0:
            nn_prior = global_prior.copy()
            support = 0.0
        else:
            nn_prior = (X_train[nn] * w.reshape(-1, 1)).sum(axis=0) / w.sum()
            nn_prior = _finalize_probs(nn_prior.reshape(1, -1)).reshape(-1)
            support = float(w.sum())
        mix = support / (support + float(tau))
        out[i] = _finalize_probs(
            (mix * nn_prior + (1.0 - mix) * global_prior).reshape(1, -1)
        ).reshape(-1)
    return out


def make_prior_baseline():
    base = pd.read_csv(SAMPLE_SUB_PATH)
    base = base[SUB_COLS].copy()  # enforce required schema/order

    test = pd.read_csv(
        TEST_CSV_PATH, usecols=["eeg_id", "patient_id", "spectrogram_id"]
    )
    train = pd.read_csv(
        TRAIN_CSV_PATH, usecols=["eeg_id", "patient_id", "spectrogram_id"] + TARGET_COLS
    )

    for c in ["eeg_id", "patient_id", "spectrogram_id"]:
        test[c] = test[c].astype("string")
        train[c] = train[c].astype("string")
    base["eeg_id"] = base["eeg_id"].astype("string")

    global_counts = train[TARGET_COLS].astype("float64").sum(axis=0).values
    global_prior = np.clip(global_counts, 1e-12, None)
    global_prior = global_prior / global_prior.sum()

    alpha = 0.5  # keep same deterministic smoothing strength

    patient_counts = train.groupby("patient_id")[TARGET_COLS].sum()
    spectro_counts = train.groupby("spectrogram_id")[TARGET_COLS].sum()

    patient_prior = _dirichlet_smooth_from_counts(
        patient_counts, alpha=alpha, global_prior=global_prior
    )
    spectro_prior = _dirichlet_smooth_from_counts(
        spectro_counts, alpha=alpha, global_prior=global_prior
    )

    train_vote_mass = train[TARGET_COLS].sum(axis=1).astype("float64")
    patient_vote_mass = train_vote_mass.groupby(train["patient_id"]).sum()
    spectro_vote_mass = train_vote_mass.groupby(train["spectrogram_id"]).sum()

    patient_prior = _shrink_group_prior_to_global(
        patient_prior, patient_vote_mass, global_prior=global_prior, tau=30.0
    )
    spectro_prior = _shrink_group_prior_to_global(
        spectro_prior, spectro_vote_mass, global_prior=global_prior, tau=30.0
    )

    eeg_prior_map, eeg_mass_map = _make_patient_loo_eeg_priors(
        train_meta_votes=train[["eeg_id", "patient_id"] + TARGET_COLS],
        alpha=alpha,
        global_prior=global_prior,
        tau=30.0,
    )

    patient_knn_map_train = _make_patient_knn_prior_map_train_only(
        patient_prior_df=patient_prior,
        patient_vote_mass=patient_vote_mass,
        global_prior=global_prior,
        k=32,
        tau=120.0,
    )

    patient_map = {k: patient_prior.loc[k].values for k in patient_prior.index.values}
    spectro_map = {k: spectro_prior.loc[k].values for k in spectro_prior.index.values}
    patient_m_map = {k: float(v) for k, v in patient_vote_mass.to_dict().items()}
    spectro_m_map = {k: float(v) for k, v in spectro_vote_mass.to_dict().items()}

    test = test.drop_duplicates(subset=["eeg_id"]).set_index("eeg_id")
    base_ids = base["eeg_id"].values
    test = test.loc[base_ids]

    eid = test.index.values
    pid = test["patient_id"].values
    sid = test["spectrogram_id"].values

    train_patients = patient_prior.index.astype("string").values
    X_train_pat = patient_prior.values.astype("float64")
    mass_train_pat = (
        patient_vote_mass.reindex(patient_prior.index)
        .fillna(0.0)
        .values.astype("float64")
    )

    unique_test_pids = pd.Index(pd.Series(pid, dtype="string").unique())
    unseen_mask = ~unique_test_pids.isin(train_patients)
    unseen_pids = unique_test_pids[unseen_mask]

    unseen_knn_map = {}
    if len(unseen_pids) > 0 and X_train_pat.shape[0] > 0:
        Q = np.tile(global_prior.reshape(1, -1), (len(unseen_pids), 1))
        Q_pred = _knn_predict_for_queries(
            X_train=X_train_pat,
            mass_train=mass_train_pat,
            queries=Q,
            global_prior=global_prior,
            k=32,
            tau=120.0,
        )
        for p, v in zip(unseen_pids.astype("string").values, Q_pred):
            unseen_knn_map[p] = v

    pred = np.zeros((len(base_ids), len(TARGET_COLS)), dtype="float64")
    for i, (e, p, s) in enumerate(zip(eid, pid, sid)):
        ep = eeg_prior_map.get(e, global_prior)
        pp = patient_map.get(p, global_prior)
        sp = spectro_map.get(s, global_prior)
        pk = patient_knn_map_train.get(p, unseen_knn_map.get(p, global_prior))

        me = float(eeg_mass_map.get(e, 0.0))
        mp = float(patient_m_map.get(p, 0.0))
        ms = float(spectro_m_map.get(s, 0.0))

        w_eeg = me / (me + 25.0)
        w_eeg = float(np.clip(w_eeg, 0.0, 0.90))

        w_ps = (ms + 1.0) / (ms + mp + 2.0)
        w_ps = float(np.clip(w_ps, 0.15, 0.85))
        ps = (1.0 - w_ps) * pp + w_ps * sp

        w_knn = 60.0 / (mp + 60.0)
        w_knn = float(np.clip(w_knn, 0.10, 0.70))
        ps2 = (1.0 - w_knn) * ps + w_knn * pk

        pred[i] = w_eeg * ep + (1.0 - w_eeg) * ps2

    pred = _finalize_probs(pred)

    out = base.copy()
    out[TARGET_COLS] = pred
    return out


if external_ok:
    sol = merge_preds(folds=(0, 1, 2, 3, 4), version="v2")
else:
    sol = make_prior_baseline()

sol = sol.reindex(columns=SUB_COLS)
sol[TARGET_COLS] = _finalize_probs(sol[TARGET_COLS].values)

print(sol.head())
print("sol shape:", sol.shape)



## === cell 4
sub_path = "/kaggle/working/submission.csv"
sol.to_csv(sub_path, index=False)

chk = pd.read_csv(sub_path)
assert list(chk.columns) == SUB_COLS, "Submission columns mismatch"
row_sum = chk[TARGET_COLS].sum(axis=1).values
assert np.all(np.isfinite(row_sum)), "Non-finite probabilities"
assert np.max(np.abs(row_sum - 1.0)) < 1e-6, "Row probabilities do not sum to 1"
print("Saved valid submission:", sub_path, "shape=", chk.shape)
print(chk.head())
