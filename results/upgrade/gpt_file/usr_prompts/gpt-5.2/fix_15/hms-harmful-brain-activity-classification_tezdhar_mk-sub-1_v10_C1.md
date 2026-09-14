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

0.3556530763563889

# 6. Current score

0.7806

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The crashes come from assuming an external code repo (`/kaggle/input/hms-mk-codes`) and checkpoints exist; in your environment they don’t, so no fold submissions are ever created and the merge step fails. To make the notebook run end-to-end and still produce a valid `.csv`, I add a robust fallback path: if the repo (or any fold file) is missing, generate a submission directly from `sample_submission.csv` using safe, normalized probabilities. This keeps the original multi-fold merge logic intact when fold files exist, but guarantees a correct-format submission otherwise (rows sum to 1, correct columns, correct row count/order). This is score-neutral vs “no submission” and should at least yield a valid leaderboard score.'
- What this solution (achieved 1.41937) has done: 'Your current 1.40995 score suggests you’re already producing a valid submission, but the fallback path is still extremely weak because it predicts uniform probabilities for every class, which is far from the empirical label distribution and hurts KL divergence. To move toward the target 0.3556 with minimal change and without altering the core “use fold predictions if available” logic, I only improve the fallback by using the class prior estimated from `train.csv` vote totals (normalized) and use that as a constant probability vector for all test rows. This keeps evaluation semantics (probabilities summing to 1), avoids any leakage (train labels are allowed for priors), and typically yields a much better baseline than uniform when no model outputs exist. If fold submissions are present, the original merge behavior remains unchanged.'
- What this solution (achieved 1.68479) has done: 'The crash is caused by duplicate `eeg_id` values introduced during `fallback_submission()` when merging `sample_submission` with `test.csv` (which can contain repeated `eeg_id`), making `set_index(...).reindex(...)` invalid. I fix this by de-duplicating `test_meta` on `eeg_id` before the merge, and (defensively) aggregating any duplicates after the merge so the index is unique. This is a correctness/stability fix and keeps your core “use fold predictions if available, else patient-prior fallback” logic unchanged, while ensuring a valid submission CSV is always written. The rest of the pipeline (priors, smoothing, normalization, output format) is preserved.'
- What this solution (achieved 0.81005) has done: 'Your current score (1.68479, lower-is-better) is far from the target (0.35565), and the biggest gap likely comes from the fallback producing overly generic priors that don’t match each test EEG’s true distribution. With minimal changes and preserving your core “use fold predictions if available, else fallback” logic, I improve only the fallback to use per‑`eeg_id` priors (aggregated over all train rows for that same `eeg_id`) with a backoff chain `eeg_id -> patient_id -> global`, which is still legitimate (train labels only) and typically much closer than patient-only. I also apply the same small prior-smoothing you already use in `merge_preds` to the fallback output for calibration consistency. No model/inference code is changed; this only improves the fallback probabilities when no fold files exist.'
- What this solution (achieved 0.76233) has done: 'Your score is still far from the target (0.81005 vs 0.35565, lower-is-better), and since no fold model predictions are being produced in this environment, the only lever is improving the fallback probabilities while keeping the overall “use fold preds if available else fallback” logic intact. I keep your fallback’s eeg→patient→global backoff, but make it closer to the evaluation target distribution by (1) converting vote totals into **Dirichlet-smoothed probabilities** (reduces overconfident zeros and improves KL) and (2) **mixing eeg-specific priors with the patient/global priors** using a small, stable weight based on how many total votes that eeg_id has in train (more votes → trust eeg prior more). I also fix a correctness bug in `merge_preds` where NaN filling uses a 2D mask incorrectly (could mis-fill), without changing its semantics. These are minimal changes, affect only fallback/merging robustness, and still guarantee a valid submission with rows summing to 1.'
- What this solution (achieved 0.76167) has done: 'Your current gap to the target is large (0.76233 vs 0.35565, lower-is-better), and since this environment is not producing fold model predictions, the only safe lever is improving the fallback probabilities while keeping your overall “use folds if present else fallback” logic unchanged. I make the fallback more KL-friendly by (1) calibrating the eeg→patient→global backoff with a stronger but still conservative EEG-specific blend weight tied to vote strength, and (2) applying a small temperature-like shrink toward the global prior via a single mixing coefficient that’s tuned to reduce overconfident priors (which KL penalizes). I also tighten the patient/eeg prior lookup to avoid any silent misalignment and keep all probabilities strictly normalized with a small floor. These are minimal, localized changes that preserve your semantics and still always write a valid `submission.csv`.'
- What this solution (achieved 0.78499) has done: 'To move your KL score down toward the 0.3556 target without changing the core “use fold preds if available else fallback” logic, I only adjust the fallback’s probability calibration (since no fold predictions are produced here). Specifically, I make the fallback less overconfident by (1) slightly increasing Dirichlet smoothing (alpha) on the per-eeg/per-patient vote priors and (2) strengthening the final shrinkage toward the global class prior (a metric-aligned calibration that typically improves KL). I also keep the output strictly normalized with a small floor to avoid KL blow-ups from near-zeros. No model/inference code, file paths, or submission schema is changed.'
- What this solution (achieved 0.77673) has done: 'We can’t change the (missing) fold-model inference here, so the only lever to move your KL score down toward the 0.3557 target is improving the fallback probability calibration while keeping the same fallback structure (eeg→patient→global priors + smoothing + shrinkage). Your current fallback is likely still too “prior-like” and not conditioned on test metadata beyond patient_id; a minimal, metric-aligned improvement is to add a very light, data-driven conditioning on `spectrogram_id` using train vote priors as another backoff branch (spectrogram_id is available in test.csv). I also slightly retune the final shrinkage-to-global (`eps`) downward to avoid over-washing any informative (eeg/patient/spectrogram) prior signal, while keeping Dirichlet smoothing and strict normalization intact. All changes are localized to the fallback priors; submission schema, paths, and overall control flow remain unchanged.'
- What this solution (achieved 0.7806) has done: 'You’re far above the target (0.77673 vs 0.35565, lower-is-better), and because fold inference isn’t running here the only lever is improving the fallback probabilities while keeping the same “train-vote priors + backoff + smoothing + shrinkage” semantics. I make two minimal, metric-aligned calibration tweaks: (1) reduce the aggressiveness of the EEG/spectrogram blending weights so the fallback is less overconfident and less “spiky” (KL-friendly), and (2) slightly increase the final shrink-to-global mix to further dampen overconfident group priors without changing the structure. I also keep the strict normalization/flooring so every row sums to 1 and avoids near-zero KL blow-ups. No file paths, submission schema, or overall control flow is changed.'

# 9. Code solution

## === cell 0
import os
import sys
from pathlib import Path

REPO_DIR = Path("/kaggle/input/hms-mk-codes")
if REPO_DIR.exists():
    sys.path.insert(0, str(REPO_DIR))

print("Python:", sys.version)
print("Repo exists:", REPO_DIR.exists())
print("Repo in sys.path:", str(REPO_DIR) in sys.path)



## === cell 1
import subprocess


def _pip_install(wheel_path, extra_args=None):
    extra_args = extra_args or []
    if Path(wheel_path).exists():
        cmd = ["pip", "install", wheel_path] + extra_args
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd)
    else:
        print(f"Wheel not found, skipping: {wheel_path}")


_pip_install(
    "/kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl",
    ["--no-index", "--no-deps", "--force-reinstall"],
)
_pip_install(
    "/kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl",
    ["--no-index", "--no-deps"],
)
_pip_install(
    "/kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl",
    ["--no-index", "--no-deps"],
)
_pip_install(
    "/kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl",
    ["--no-deps", "--no-index"],
)



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

os.makedirs(OUT_PATH, exist_ok=True)



## === cell 3
convert_ok = False
print("Skipping parquet->npy conversion for speed. convert_ok:", convert_ok)



## === cell 4
from pathlib import Path


def _run_test_with_ckpt(ckpt_path: str) -> bool:
    ckpt = Path(ckpt_path)
    if not REPO_DIR.exists():
        print("Skipping test run: missing repo:", REPO_DIR)
        return False
    if not ckpt.exists():
        print("Skipping test run: missing checkpoint:", ckpt)
        return False

    try:
        subprocess.check_call(
            [
                "python",
                "-m",
                "test",
                f"paths.data_dir={DATA_PATH}",
                f"data.test_eegs_dir={OUT_PATH}",
                f"ckpt_path={ckpt_path}",
                "hydra=test",
                f"+model.test_output_dir={OUT_PATH}",
                "+model.net.pretrained=False",
            ],
            cwd=str(REPO_DIR),
        )
        return True
    except Exception as e:
        print("Warning: test run failed:", ckpt_path, repr(e))
        return False


FOLD_CKPTS = {
    2: "/kaggle/input/hms-mk-data/epoch_014_val_loss_0.4944.ckpt",
    1: "/kaggle/input/hms-mk-data/epoch_014_val_loss_0.4728.ckpt",
    0: "/kaggle/input/hms-mk-data/epoch_012_val_loss_0.5037.ckpt",
}

results = {}
produced_folds = []

for fold in (2, 1, 0):
    ckpt = FOLD_CKPTS[fold]
    print(f"Running fold {fold} inference...")
    ok = _run_test_with_ckpt(ckpt)
    results[fold] = ok
    print(f"fold{fold}_ok:", ok)
    if ok:
        produced_folds.append(fold)
        break  # speed: don't run more folds once we already have predictions



## === cell 5
from pathlib import Path

for fold in (2, 1, 0):
    src_sub = Path("/kaggle/working/submission.csv")
    dst_sub = Path(f"/kaggle/working/submission_fold{fold}.csv")
    if results.get(fold, False) and src_sub.exists():
        src_sub.replace(dst_sub)
        print("Wrote:", dst_sub)
    else:
        print(
            f"No submission.csv produced for fold{fold} (or fold failed); will rely on other folds/fallback."
        )



## === cell 6
import pandas as pd
import numpy as np
from functools import lru_cache

try:
    from src.settings import TARGET_COLS  # type: ignore
except Exception as e:
    print("Warning: could not import TARGET_COLS from src.settings:", repr(e))
    TARGET_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

DIRICHLET_ALPHA_GLOBAL = 1.0
DIRICHLET_ALPHA_GROUP = 3.0  # keep as-is (prior change history)


def _dirichlet_smooth_probs(votes: np.ndarray, alpha: float) -> np.ndarray:
    """
    Minimal, metric-aligned smoothing: KL divergence heavily penalizes near-zeros
    when the true class has probability mass. Dirichlet smoothing prevents
    overconfident zeros/near-zeros while preserving the original prior direction.
    """
    v = votes.astype(np.float64, copy=False)
    v = np.clip(v, 0.0, None)
    v = v + float(alpha)
    denom = v.sum(axis=-1, keepdims=True)
    denom = np.where(denom > 0, denom, 1.0)
    p = v / denom
    p = np.clip(p, 1e-15, None)
    p = p / p.sum(axis=-1, keepdims=True)
    return p


def _mix_with_prior(probs: np.ndarray, prior: np.ndarray, eps: float) -> np.ndarray:
    """
    Small convex mix toward a prior to reduce overconfidence (KL-friendly).
    """
    eps = float(eps)
    prior = prior.reshape(1, -1).astype(np.float64, copy=False)
    out = (1.0 - eps) * probs + eps * prior
    out = np.clip(out, 1e-15, None)
    out = out / out.sum(axis=1, keepdims=True)
    return out


@lru_cache(maxsize=2)
def _train_priors_from_votes(data_path: str = DATA_PATH) -> np.ndarray:
    train_path = Path(data_path) / "train.csv"
    if not train_path.exists():
        train_path = Path("/kaggle/input") / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Could not find train.csv at {train_path}")

    train = pd.read_csv(train_path, usecols=TARGET_COLS)
    vote_sums = train[TARGET_COLS].sum(axis=0).to_numpy(dtype=np.float64)

    priors = _dirichlet_smooth_probs(
        vote_sums.reshape(1, -1), alpha=DIRICHLET_ALPHA_GLOBAL
    ).reshape(-1)
    priors = np.clip(priors, 1e-15, None)
    priors = priors / priors.sum()
    return priors


@lru_cache(maxsize=2)
def _train_patient_priors(data_path: str = DATA_PATH):
    train_path = Path(data_path) / "train.csv"
    if not train_path.exists():
        train_path = Path("/kaggle/input") / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Could not find train.csv at {train_path}")

    usecols = ["patient_id"] + TARGET_COLS
    train = pd.read_csv(train_path, usecols=usecols)

    vote_sums_global = train[TARGET_COLS].sum(axis=0).to_numpy(dtype=np.float64)
    global_priors = _dirichlet_smooth_probs(
        vote_sums_global.reshape(1, -1), alpha=DIRICHLET_ALPHA_GLOBAL
    ).reshape(-1)

    grp = train.groupby("patient_id", sort=False)[TARGET_COLS].sum()
    patient_vote_sums = grp.to_numpy(dtype=np.float64)
    patient_priors = _dirichlet_smooth_probs(
        patient_vote_sums, alpha=DIRICHLET_ALPHA_GROUP
    )

    patient_ids = grp.index.to_numpy()
    patient_to_priors = {
        int(pid): patient_priors[i] for i, pid in enumerate(patient_ids)
    }
    return global_priors, patient_to_priors


@lru_cache(maxsize=2)
def _train_eeg_priors(data_path: str = DATA_PATH):
    train_path = Path(data_path) / "train.csv"
    if not train_path.exists():
        train_path = Path("/kaggle/input") / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Could not find train.csv at {train_path}")

    usecols = ["eeg_id"] + TARGET_COLS
    train = pd.read_csv(train_path, usecols=usecols)

    grp = train.groupby("eeg_id", sort=False)[TARGET_COLS].sum()
    eeg_vote_sums = grp.to_numpy(dtype=np.float64)

    eeg_priors = _dirichlet_smooth_probs(eeg_vote_sums, alpha=DIRICHLET_ALPHA_GROUP)
    eeg_strength = eeg_vote_sums.sum(axis=1).astype(np.float64)

    eeg_ids = grp.index.to_numpy()
    eeg_to_priors = {int(eid): eeg_priors[i] for i, eid in enumerate(eeg_ids)}
    eeg_to_strength = {
        int(eid): float(eeg_strength[i]) for i, eid in enumerate(eeg_ids)
    }
    return eeg_to_priors, eeg_to_strength


@lru_cache(maxsize=2)
def _train_spectrogram_priors(data_path: str = DATA_PATH):
    train_path = Path(data_path) / "train.csv"
    if not train_path.exists():
        train_path = Path("/kaggle/input") / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Could not find train.csv at {train_path}")

    usecols = ["spectrogram_id"] + TARGET_COLS
    train = pd.read_csv(train_path, usecols=usecols)

    grp = train.groupby("spectrogram_id", sort=False)[TARGET_COLS].sum()
    sp_vote_sums = grp.to_numpy(dtype=np.float64)

    sp_priors = _dirichlet_smooth_probs(sp_vote_sums, alpha=DIRICHLET_ALPHA_GROUP)
    sp_strength = sp_vote_sums.sum(axis=1).astype(np.float64)

    sp_ids = grp.index.to_numpy()
    sp_to_priors = {int(sid): sp_priors[i] for i, sid in enumerate(sp_ids)}
    sp_to_strength = {int(sid): float(sp_strength[i]) for i, sid in enumerate(sp_ids)}
    return sp_to_priors, sp_to_strength


def _get_priors_for_patients(
    patient_ids: np.ndarray, global_priors: np.ndarray, patient_to_priors: dict
) -> np.ndarray:
    out = np.empty((len(patient_ids), len(TARGET_COLS)), dtype=np.float64)
    for i, pid in enumerate(patient_ids):
        try:
            key = int(pid)
        except Exception:
            key = None
        if key is not None and key in patient_to_priors:
            out[i] = patient_to_priors[key]
        else:
            out[i] = global_priors

    out = np.clip(out, 1e-15, None)
    out = out / out.sum(axis=1, keepdims=True)
    return out


def _get_priors_for_eegs_and_spects(
    eeg_ids: np.ndarray,
    patient_ids: np.ndarray,
    spectrogram_ids: np.ndarray,
    global_priors: np.ndarray,
    patient_to_priors: dict,
    eeg_to_priors: dict,
    eeg_to_strength: dict,
    sp_to_priors: dict,
    sp_to_strength: dict,
) -> np.ndarray:
    """
    Minimal, metric-aligned calibration change:
    - Keep the same backoff/blending structure (patient base, then spectrogram, then eeg),
      but reduce blend caps so outputs are less overconfident/spiky (helps KL).
    """
    base = _get_priors_for_patients(patient_ids, global_priors, patient_to_priors)

    sp_prior_out = np.empty_like(base, dtype=np.float64)
    sp_ok = np.zeros((base.shape[0],), dtype=bool)
    sp_strength_arr = np.zeros((base.shape[0],), dtype=np.float64)

    for i, sid in enumerate(spectrogram_ids):
        try:
            key = int(sid)
        except Exception:
            key = None
        if key is not None and key in sp_to_priors:
            sp_prior_out[i] = sp_to_priors[key]
            sp_ok[i] = True
            sp_strength_arr[i] = float(sp_to_strength.get(key, 0.0))

    if sp_ok.any():
        w_sp = 0.25 * (sp_strength_arr / (sp_strength_arr + 60.0))
        w_sp = np.clip(w_sp, 0.0, 0.25).reshape(-1, 1)
        blended_sp = (1.0 - w_sp) * base + w_sp * sp_prior_out
        base[sp_ok] = blended_sp[sp_ok]

    eeg_prior_out = np.empty_like(base, dtype=np.float64)
    eeg_ok = np.zeros((base.shape[0],), dtype=bool)
    eeg_strength_arr = np.zeros((base.shape[0],), dtype=np.float64)

    for i, eid in enumerate(eeg_ids):
        try:
            key = int(eid)
        except Exception:
            key = None
        if key is not None and key in eeg_to_priors:
            eeg_prior_out[i] = eeg_to_priors[key]
            eeg_ok[i] = True
            eeg_strength_arr[i] = float(eeg_to_strength.get(key, 0.0))

    if eeg_ok.any():
        w_eeg = 0.45 * (eeg_strength_arr / (eeg_strength_arr + 60.0))
        w_eeg = np.clip(w_eeg, 0.0, 0.45).reshape(-1, 1)
        blended_eeg = (1.0 - w_eeg) * base + w_eeg * eeg_prior_out
        base[eeg_ok] = blended_eeg[eeg_ok]

    base = np.clip(base, 1e-15, None)
    base = base / base.sum(axis=1, keepdims=True)
    return base


def merge_preds(folds=(0, 1, 2), data_path: str = DATA_PATH, sample_eeg_ids=None):
    preds_frames = []

    for fold in folds:
        path = f"/kaggle/working/submission_fold{fold}.csv"
        if not Path(path).exists():
            raise FileNotFoundError(path)

        df = pd.read_csv(path)

        missing = [c for c in (["eeg_id"] + TARGET_COLS) if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")

        df = df[["eeg_id"] + TARGET_COLS].copy()
        if df["eeg_id"].duplicated().any():
            df = df.groupby("eeg_id", as_index=False)[TARGET_COLS].mean()

        preds_frames.append(df.set_index("eeg_id"))

    if sample_eeg_ids is None:
        sample_path = Path(data_path) / "sample_submission.csv"
        if not sample_path.exists():
            sample_path = Path("/kaggle/input") / "sample_submission.csv"
        sample_eeg_ids = pd.read_csv(sample_path, usecols=["eeg_id"])[
            "eeg_id"
        ].to_numpy()
    eeg_ids = sample_eeg_ids

    test_meta_path = Path(data_path) / "test.csv"
    if not test_meta_path.exists():
        test_meta_path = Path("/kaggle/input") / "test.csv"
    test_meta = pd.read_csv(
        test_meta_path, usecols=["eeg_id", "patient_id"]
    ).drop_duplicates("eeg_id")
    test_meta = test_meta.set_index("eeg_id").reindex(eeg_ids)

    global_priors, patient_to_priors = _train_patient_priors(data_path=data_path)
    fill_priors = _get_priors_for_patients(
        test_meta["patient_id"].to_numpy(), global_priors, patient_to_priors
    )

    mats = []
    for fdf in preds_frames:
        aligned = fdf.reindex(eeg_ids)
        aligned_arr = aligned.to_numpy(dtype=np.float64)

        bad_row = ~np.isfinite(aligned_arr).all(axis=1)
        if bad_row.any():
            aligned_arr[bad_row, :] = fill_priors[bad_row, :]

        mats.append(aligned_arr)

    preds = np.mean(np.stack(mats, axis=0), axis=0)

    eps = 0.02
    priors = _train_priors_from_votes(data_path=data_path)
    preds = _mix_with_prior(preds, priors, eps=eps)

    sol = pd.DataFrame(preds, columns=TARGET_COLS)
    sol.insert(0, "eeg_id", eeg_ids)
    return sol


def fallback_submission(
    data_path: str = DATA_PATH, sample_eeg_ids=None
) -> pd.DataFrame:
    sample_path = Path(data_path) / "sample_submission.csv"
    if not sample_path.exists():
        sample_path = Path("/kaggle/input") / "sample_submission.csv"
    if not sample_path.exists():
        raise FileNotFoundError(
            f"Could not find sample_submission.csv at {sample_path}"
        )

    sub = pd.read_csv(sample_path, usecols=["eeg_id"] + TARGET_COLS)

    test_meta_path = Path(data_path) / "test.csv"
    if not test_meta_path.exists():
        test_meta_path = Path("/kaggle/input") / "test.csv"
    test_meta = pd.read_csv(
        test_meta_path, usecols=["eeg_id", "patient_id", "spectrogram_id"]
    ).drop_duplicates("eeg_id")

    sub = sub.merge(test_meta, on="eeg_id", how="left")

    if sub["eeg_id"].duplicated().any():
        sub = sub.groupby(["eeg_id"], as_index=False).agg(
            {
                **{c: "mean" for c in TARGET_COLS},
                "patient_id": "first",
                "spectrogram_id": "first",
            }
        )

    global_priors, patient_to_priors = _train_patient_priors(data_path=data_path)
    eeg_to_priors, eeg_to_strength = _train_eeg_priors(data_path=data_path)
    sp_to_priors, sp_to_strength = _train_spectrogram_priors(data_path=data_path)

    probs = _get_priors_for_eegs_and_spects(
        sub["eeg_id"].to_numpy(),
        sub["patient_id"].to_numpy(),
        sub["spectrogram_id"].to_numpy(),
        global_priors,
        patient_to_priors,
        eeg_to_priors,
        eeg_to_strength,
        sp_to_priors,
        sp_to_strength,
    )

    priors = _train_priors_from_votes(data_path=data_path)
    probs = _mix_with_prior(probs, priors, eps=0.10)  # was 0.08

    sub.loc[:, TARGET_COLS] = probs
    sub = sub.drop(columns=["patient_id", "spectrogram_id"])

    if sample_eeg_ids is not None:
        sub = sub.set_index("eeg_id").reindex(sample_eeg_ids).reset_index()

    return sub[["eeg_id"] + TARGET_COLS].copy()




## === cell 7
from pathlib import Path

available_folds = [
    f for f in (0, 1, 2) if Path(f"/kaggle/working/submission_fold{f}.csv").exists()
]

sample_path = Path(DATA_PATH) / "sample_submission.csv"
if not sample_path.exists():
    sample_path = Path("/kaggle/input") / "sample_submission.csv"
sample = pd.read_csv(sample_path, usecols=["eeg_id"] + TARGET_COLS)
sample_ids = sample["eeg_id"].to_numpy()

if available_folds:
    print("Merging available folds:", available_folds)
    sol = merge_preds(
        folds=tuple(available_folds), data_path=DATA_PATH, sample_eeg_ids=sample_ids
    )
else:
    print("No fold submissions found; using fallback submission.")
    sol = fallback_submission(data_path=DATA_PATH, sample_eeg_ids=sample_ids)

arr = sol[TARGET_COLS].to_numpy(dtype=np.float64)
arr = np.clip(arr, 1e-15, None)
arr = arr / arr.sum(axis=1, keepdims=True)
sol.loc[:, TARGET_COLS] = arr

print("Final sol shape (must match sample):", sol.shape, "sample:", sample.shape)
assert (
    sol.shape[0] == sample.shape[0]
), "Submission length mismatch vs sample_submission"
assert list(sol.columns) == ["eeg_id"] + TARGET_COLS, "Submission columns mismatch"
sol.head()



## === cell 8
out_file = "/kaggle/working/submission.csv"
sol.to_csv(out_file, index=False)
print("Final submission written to:", out_file)
print("Shape:", sol.shape)
print("Columns:", list(sol.columns))

row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
print("Row sum stats:", float(row_sums.min()), float(row_sums.max()))



## === cell 9
sol
