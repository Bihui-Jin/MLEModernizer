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

0.323113851522087

# 6. Current score

0.82259

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.39779) has done: 'Your run fails because the fold prediction CSVs are never created: the earlier `python -m test ...` commands depend on external Kaggle inputs (`/kaggle/input/hms-mk-codes` and `/kaggle/input/hms-mk-data`) that are not present in this environment, so the subsequent merge can’t find `/kaggle/working/submission_fold*_v3.csv`. To make the notebook run end-to-end and always produce a valid `submission.csv`, I add guards that detect missing external assets and skip those steps instead of crashing. When predictions aren’t available, the script falls back to a safe baseline using the class prior from `train.csv` (normalized vote counts), ensuring rows sum to 1 and the submission format is correct. The merge function is also made robust to either produce an averaged ensemble (when files exist) or the baseline (when they don’t), which should yield a reasonable (not necessarily optimal) KL score rather than “Not yielded”.'
- What this solution (achieved 1.43453) has done: 'Your current score (1.39779; lower is better) is far worse than the target (0.3231), so we should improve the *baseline* fallback because that’s what you’re likely submitting when the external repo/checkpoints aren’t available. I keep your pipeline and merging logic intact, but replace the global-prior baseline with a patient-aware baseline: compute per-patient class priors from `train.csv` and use the matching `patient_id` in `test.csv`, with a safe fallback to the global prior for unseen patients. This is still leakage-free, preserves evaluation semantics (probabilities sum to 1), and usually yields a large KL improvement vs a single global prior. I also align `sol` rows explicitly to `test.csv` order (via `eeg_id`) to avoid any accidental misalignment.'
- What this solution (achieved 1.43453) has done: 'Your current score is much worse than the target (lower is better), and because the external model repo/checkpoints are often missing here, your effective submission is the fallback baseline; the highest-impact minimal change is to make that baseline closer to the true label distribution. I keep your pipeline structure and submission semantics unchanged, but strengthen the baseline from “patient prior only” to a leakage-free hierarchical prior: patient-specific when available, otherwise a per-(patient_id, eeg_id) prior built from all overlapping train rows for that eeg, and finally the global prior. I also vectorize the baseline construction (no Python loop) to be faster and less error-prone, while preserving the same probability normalization and alignment to `test.csv` order.'
- What this solution (achieved 0.78093) has done: 'Your score is far worse than the target (lower is better), and since the external repo/checkpoints often aren’t available here your actual submission is the fallback baseline, so the highest-impact minimal improvement is to make that baseline closer to the true per-eeg distribution. I keep your pipeline and merging logic unchanged, but strengthen `_train_priors` to aggregate votes at the consolidated `eeg_id` level (not per overlapping window) before computing patient/eeg priors, which reduces label-noise from duplicated/overlapping segments. I also add a deterministic “shrinkage” mix between the (patient,eeg) prior, patient prior, and global prior so that small-sample priors don’t become overconfident, which typically improves KL while preserving probability semantics. The submission format, paths, and model-inference branches remain the same, and the output still be a valid `submission.csv` with rows summing to 1.'
- What this solution (achieved 0.78093) has done: 'Your current KL (0.78093; lower is better) is still far from the target (0.3231), and since the external repo/checkpoints often aren’t present your score is dominated by the fallback baseline. I keep your inference/merge pipeline unchanged, but improve the baseline in a minimal, leakage-free way by using a spectrogram-level prior (because test provides `spectrogram_id`) computed from train after consolidating overlaps at `eeg_id` level. Then I combine priors hierarchically in this order: (patient_id, spectrogram_id) → spectrogram_id → patient_id → global, with the same shrinkage-to-global idea to avoid overconfident sparse groups. This should move KL noticeably toward the target without changing the model logic or submission semantics.'
- What this solution (achieved 0.78004) has done: 'Your current KL (0.78093; lower is better) is still far above the target (0.3231), and since external fold predictions are often missing, the score is dominated by the fallback baseline. I keep your overall pipeline/merge logic intact but make the baseline priors more faithful by using vote-weighted aggregation (use total votes as weights) instead of a simple mean, which is more aligned to how the competition targets are formed and typically improves KL. I also actually use the already-computed `(patient_id, eeg_id)` priors as the highest-priority signal (it was computed but unused), then fall back through `(patient_id, spectrogram_id) → spectrogram_id → patient_id → global`, with the same shrinkage and strict row normalization. These are minimal, leakage-free changes that preserve submission semantics and should move KL toward the target.'
- What this solution (achieved 0.79938) has done: 'Your current score (0.78004, lower-is-better) is still far above the target (0.32311), and given the missing external repo/checkpoints this pipeline is almost certainly scoring via the metadata-only fallback baseline. I keep your overall flow and hierarchy intact, but make the fallback priors closer to the true label construction by (1) consolidating votes at the `eeg_id` level across overlaps before building any group priors (reduces duplicate-window bias), and (2) using principled shrinkage based on each group’s total vote mass (rather than fixed alphas) so sparse groups don’t get overconfident (typically lowers KL). I also fix a small merge bug in `merge_preds` where `versions`/`weights` zip could silently ignore folds if lengths differ, keeping behavior deterministic and consistent. The submission schema/paths remain unchanged and rows are still strictly normalized to sum to 1.'
- What this solution (achieved 0.82259) has done: 'I keep your overall inference/merge pipeline intact and focus only on the fallback baseline (which is likely what’s scoring, given missing external checkpoints). The main minimal improvement is to make the hierarchical prior closer to how the competition labels behave by using label smoothing (Dirichlet-style pseudocounts) when forming group priors, which reduces overconfident sparse-group distributions and usually lowers KL. I also make the final shrinkage step use the *group’s effective sample size* (vote mass plus pseudocount mass) so the alpha mixing is calibrated consistently with the smoothed priors. Finally, I keep strict row alignment to `test.csv` and renormalization to ensure a valid submission.'
- What this solution (achieved 0.82259) has done: 'Your current KL (0.82259; lower is better) is far above the target (0.32311), and given the missing external checkpoints your score is dominated by the metadata-only fallback baseline. I keep the same overall pipeline and hierarchy, but change the fallback prior estimation to match the competition target construction more closely: build priors from raw vote counts (Dirichlet-smoothed) instead of “probabilities then re-averaged,” which reduces bias from variable annotator counts and overlaps. I also make the hierarchy use consolidated per-`eeg_id` vote totals (one row per eeg_id) before grouping, which typically lowers KL by reducing duplicated-window noise. Finally, I keep strict alignment to `test.csv` and renormalize to ensure a valid submission with rows summing to 1.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
from pathlib import Path


def _in_ipython():
    try:
        get_ipython  # noqa: F401
        return True
    except Exception:
        return False


MK_CODES_PATH = "/kaggle/input/hms-mk-codes/"
if os.path.isdir(MK_CODES_PATH):
    sys.path.append(MK_CODES_PATH)



## === cell 1
if _in_ipython():
    wheels_dir = "/kaggle/input/requirements-mk"
    if os.path.isdir(wheels_dir):
        get_ipython().run_line_magic(
            "pip",
            f"install {wheels_dir}/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall",
        )
        get_ipython().run_line_magic(
            "pip",
            f"install {wheels_dir}/omegaconf-2.3.0-py3-none-any.whl --no-index --no-deps",
        )
        get_ipython().run_line_magic(
            "pip",
            f"install {wheels_dir}/hydra_core-1.3.2-py3-none-any.whl --no-index --no-deps",
        )
        get_ipython().run_line_magic(
            "pip",
            f"install {wheels_dir}/lightning-2.2.1-py3-none-any.whl --no-deps --no-index",
        )
    else:
        print(f"[WARN] {wheels_dir} not found; skipping custom wheel installs.")



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

Path(OUT_PATH).mkdir(parents=True, exist_ok=True)



## === cell 3
if _in_ipython() and os.path.isdir(MK_CODES_PATH):
    get_ipython().system(
        f"cd {MK_CODES_PATH} && python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}"
    )
else:
    print("[WARN] hms-mk-codes not available; skipping parquet->npy conversion.")



## === cell 4
if _in_ipython():
    if os.path.exists("/kaggle/input/hms-mk-data"):
        get_ipython().system("ls /kaggle/input/hms-mk-data")
    else:
        print(
            "[WARN] /kaggle/input/hms-mk-data not found; cannot list model checkpoints."
        )




## === cell 5
def _run_fold_infer(fold: int, version: str = "v3") -> bool:
    if not _in_ipython():
        return False
    if not os.path.isdir(MK_CODES_PATH):
        return False
    ckpt = f"/kaggle/input/hms-mk-data/fold{fold}_effb3_sim_pseudo.ckpt"
    if not os.path.exists(ckpt):
        return False

    cmd = (
        f"cd {MK_CODES_PATH} && python -m test "
        f"paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} "
        f"ckpt_path={ckpt} hydra=test +model.test_output_dir={OUT_PATH} "
        f"experiment=conv1d_effb1_v3_pseudo +model.net.pretrained=False"
    )
    get_ipython().system(cmd)
    src = f"{OUT_PATH}/submission.csv"
    dst = f"{OUT_PATH}/submission_fold{fold}_{version}.csv"
    if os.path.exists(src):
        get_ipython().system(f"mv {src} {dst}")
        return True
    return False


any_fold = False
for f in [0, 1, 2, 3, 4]:
    ok = _run_fold_infer(f, version="v3")
    any_fold = any_fold or ok

if not any_fold:
    print(
        "[WARN] No fold prediction files were generated (missing repo/checkpoints). Will use baseline submission."
    )



## === cell 6
import pandas as pd
import numpy as np

try:
    from src.settings import TARGET_COLS  # type: ignore
except Exception:
    TARGET_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]


def _normalize_rows(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    mat = np.clip(mat, eps, 1.0)
    mat = mat / mat.sum(axis=1, keepdims=True)
    return mat


def _train_priors(data_path: str):
    """
    Improvement (score->target):
    - Compute priors from *Dirichlet-smoothed aggregated vote counts* (not "per-row probs then
      weighted-average"), which matches the competition's target construction more directly
      and typically lowers KL for a metadata-only fallback.
    - Consolidate overlaps at eeg_id first to reduce duplicated-window noise before building
      any group priors (same intent as before, but now in vote-count space).
    """
    train_path = os.path.join(data_path, "train.csv")
    usecols = ["eeg_id", "patient_id", "spectrogram_id"] + TARGET_COLS
    train = pd.read_csv(train_path, usecols=usecols)

    eeg_votes = (
        train.groupby(["patient_id", "spectrogram_id", "eeg_id"], sort=False)[
            TARGET_COLS
        ]
        .sum()
        .reset_index()
    )

    v_eeg = eeg_votes[TARGET_COLS].to_numpy(dtype=np.float64)
    w_eeg = v_eeg.sum(axis=1).astype(np.float64)
    w_eeg[~np.isfinite(w_eeg)] = 0.0

    votes_df = eeg_votes[["patient_id", "spectrogram_id", "eeg_id"]].copy()
    votes_df[TARGET_COLS] = v_eeg
    votes_df["w"] = w_eeg

    global_counts = votes_df[TARGET_COLS].sum(axis=0).to_numpy(dtype=np.float64)
    global_counts = np.clip(global_counts, 0.0, np.inf)
    global_alpha = 1.0  # small symmetric pseudocount per class
    global_prior = global_counts + global_alpha
    global_prior = np.clip(global_prior, 1e-12, np.inf)
    global_prior = global_prior / global_prior.sum()

    def _group_prior_and_mass_from_counts(df: pd.DataFrame, by, tau: float):
        """
        Improvement (score->target):
        - Dirichlet smoothing in count-space:
            counts_g = sum counts_i + tau * global_prior
            p_g = counts_g / sum(counts_g)
          Here tau is a pseudocount mass in "vote units".
        This avoids overconfident small groups and is better aligned to KL on soft targets.
        """
        g_counts = df.groupby(by, sort=False)[TARGET_COLS].sum()
        g_mass = df.groupby(by, sort=False)["w"].sum().rename("w_sum")

        counts = g_counts.to_numpy(dtype=np.float64)
        counts = np.clip(counts, 0.0, np.inf)

        counts = counts + (tau * global_prior.reshape(1, -1))
        den = counts.sum(axis=1, keepdims=True)
        bad = (
            (~np.isfinite(den[:, 0]))
            | (den[:, 0] <= 0)
            | (~np.isfinite(counts).all(axis=1))
        )
        if np.any(bad):
            counts[bad] = global_prior.reshape(1, -1)
            den[bad] = 1.0

        probs = counts / den
        probs = _normalize_rows(probs, eps=1e-12)

        out_df = pd.DataFrame(probs, columns=TARGET_COLS, index=g_counts.index)
        return out_df, g_mass.to_frame()

    patient_priors, patient_mass = _group_prior_and_mass_from_counts(
        votes_df, by=["patient_id"], tau=30.0
    )
    spect_priors, spect_mass = _group_prior_and_mass_from_counts(
        votes_df, by=["spectrogram_id"], tau=20.0
    )
    ps_priors, ps_mass = _group_prior_and_mass_from_counts(
        votes_df, by=["patient_id", "spectrogram_id"], tau=15.0
    )
    pe_priors, pe_mass = _group_prior_and_mass_from_counts(
        votes_df, by=["patient_id", "eeg_id"], tau=12.0
    )

    return (
        global_prior,
        patient_priors,
        patient_mass,
        spect_priors,
        spect_mass,
        ps_priors,
        ps_mass,
        pe_priors,
        pe_mass,
    )


def _baseline_hierarchical_priors(data_path: str) -> pd.DataFrame:
    """
    Same hierarchy and shrinkage behavior as before; only the prior estimation is improved
    (count-space Dirichlet smoothing), which should move KL toward the target.
    """
    test_path = os.path.join(data_path, "test.csv")
    test = pd.read_csv(
        test_path, usecols=["eeg_id", "patient_id", "spectrogram_id"]
    ).copy()

    (
        global_prior,
        patient_priors,
        patient_mass,
        spect_priors,
        spect_mass,
        ps_priors,
        ps_mass,
        pe_priors,
        pe_mass,
    ) = _train_priors(data_path)

    base = test.merge(
        pe_priors.reset_index().rename(columns={c: f"{c}__pe" for c in TARGET_COLS}),
        how="left",
        on=["patient_id", "eeg_id"],
    ).merge(
        pe_mass.reset_index().rename(columns={"w_sum": "w__pe"}),
        how="left",
        on=["patient_id", "eeg_id"],
    )

    base = base.merge(
        ps_priors.reset_index().rename(columns={c: f"{c}__ps" for c in TARGET_COLS}),
        how="left",
        on=["patient_id", "spectrogram_id"],
    ).merge(
        ps_mass.reset_index().rename(columns={"w_sum": "w__ps"}),
        how="left",
        on=["patient_id", "spectrogram_id"],
    )

    base = base.merge(
        spect_priors.reset_index().rename(columns={c: f"{c}__s" for c in TARGET_COLS}),
        how="left",
        on=["spectrogram_id"],
    ).merge(
        spect_mass.reset_index().rename(columns={"w_sum": "w__s"}),
        how="left",
        on=["spectrogram_id"],
    )

    base = base.merge(
        patient_priors.reset_index().rename(
            columns={c: f"{c}__p" for c in TARGET_COLS}
        ),
        how="left",
        on=["patient_id"],
    ).merge(
        patient_mass.reset_index().rename(columns={"w_sum": "w__p"}),
        how="left",
        on=["patient_id"],
    )

    pe_mat = base[[f"{c}__pe" for c in TARGET_COLS]].to_numpy(dtype=np.float64)
    ps_mat = base[[f"{c}__ps" for c in TARGET_COLS]].to_numpy(dtype=np.float64)
    s_mat = base[[f"{c}__s" for c in TARGET_COLS]].to_numpy(dtype=np.float64)
    p_mat = base[[f"{c}__p" for c in TARGET_COLS]].to_numpy(dtype=np.float64)

    pe_ok = np.isfinite(pe_mat).all(axis=1)
    ps_ok = np.isfinite(ps_mat).all(axis=1)
    s_ok = np.isfinite(s_mat).all(axis=1)
    p_ok = np.isfinite(p_mat).all(axis=1)

    preds = np.tile(global_prior.reshape(1, -1), (len(base), 1))
    preds[p_ok] = p_mat[p_ok]
    preds[s_ok] = s_mat[s_ok]
    preds[ps_ok] = ps_mat[ps_ok]
    preds[pe_ok] = pe_mat[pe_ok]

    w = np.zeros(len(base), dtype=np.float64)
    w[p_ok] = base.loc[p_ok, "w__p"].to_numpy(dtype=np.float64)
    w[s_ok] = base.loc[s_ok, "w__s"].to_numpy(dtype=np.float64)
    w[ps_ok] = base.loc[ps_ok, "w__ps"].to_numpy(dtype=np.float64)
    w[pe_ok] = base.loc[pe_ok, "w__pe"].to_numpy(dtype=np.float64)
    w[~np.isfinite(w)] = 0.0

    tau = np.full(len(base), 30.0, dtype=np.float64)  # patient default
    tau[s_ok] = 20.0
    tau[ps_ok] = 15.0
    tau[pe_ok] = 12.0

    alpha = (w / (w + tau)).reshape(-1, 1)
    alpha = np.clip(alpha, 0.0, 0.98)
    preds = alpha * preds + (1.0 - alpha) * global_prior.reshape(1, -1)
    preds = _normalize_rows(preds, eps=1e-12)

    sub = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    sub[TARGET_COLS] = preds
    return sub


def merge_preds(folds=(0, 1, 2), versions=("v0",), weights=(1.0,)):
    """
    Merge per-fold submissions by weighted averaging and renormalize to sum to 1.
    If expected fold files are missing, fall back to the hierarchical prior baseline.
    """
    test_path = os.path.join(DATA_PATH, "test.csv")
    test = pd.read_csv(test_path, usecols=["eeg_id"]).copy()

    versions = tuple(versions)
    weights = tuple(weights)
    if len(versions) == 1 and len(weights) > 1:
        versions = versions * len(weights)
    if len(weights) == 1 and len(versions) > 1:
        weights = weights * len(versions)

    preds_sum = np.zeros((len(test), len(TARGET_COLS)), dtype=np.float64)
    w_sum = 0.0
    used = 0

    for fold in folds:
        for version, weight in zip(versions, weights):
            path = f"{OUT_PATH}/submission_fold{fold}_{version}.csv"
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path)

            if "eeg_id" in df.columns and not df["eeg_id"].equals(test["eeg_id"]):
                df = df.set_index("eeg_id").loc[test["eeg_id"]].reset_index()

            w = float(weight)
            preds_sum += df[TARGET_COLS].to_numpy(dtype=np.float64) * w
            w_sum += w
            used += 1

    if used == 0 or w_sum <= 0:
        return _baseline_hierarchical_priors(DATA_PATH)

    preds = preds_sum / w_sum
    preds = np.clip(preds, 1e-12, 1.0)
    preds = preds / preds.sum(axis=1, keepdims=True)

    sol = pd.DataFrame({"eeg_id": test["eeg_id"].values})
    sol[TARGET_COLS] = preds
    return sol




## === cell 7
sol = merge_preds(folds=[0, 1, 2, 3, 4], versions=["v3"], weights=[1.0])



## === cell 8
out_file = "/kaggle/working/submission.csv"
sol.to_csv(out_file, index=False)

row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
print("Saved:", out_file)
print("Submission shape:", sol.shape)
print("Row-sum min/max:", float(row_sums.min()), float(row_sums.max()))
print(sol.head())
