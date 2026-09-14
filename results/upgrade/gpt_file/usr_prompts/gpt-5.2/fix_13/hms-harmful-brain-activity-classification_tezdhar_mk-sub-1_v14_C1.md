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

0.3391724675284024

# 6. Current score

1.40985

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The runtime failure is because the fold inference commands never produced `/kaggle/working/submission.csv`, so the subsequent `mv` commands didn’t create `submission_fold*.csv` files and the ensemble step can’t find them. I make the pipeline robust by (1) skipping IPython-only `pip` magics when not available, (2) checking each fold run’s return code and whether `submission.csv` was created before moving it, and (3) allowing `merge_preds` to ensemble only the folds that actually exist (or fall back to the sample submission probabilities if none exist) so a valid `.csv` is always produced. This preserves the core modeling/inference logic (still calls the same external `python -m test` with the same checkpoints), but prevents missing-file crashes and guarantees the final submission sums to 1. The output be written to `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.33917), so we should improve predictive quality without changing the underlying model/test invocation. The most likely silent issue hurting score is row misalignment when ensembling: each fold’s `submission_fold*.csv` may be in a different order than `sample_submission.csv`, and your current merge can reorder rows and introduce NaNs, corrupting the averaged probabilities. I change `merge_preds` to strictly align predictions to the sample’s `eeg_id` order via an index-based reindex (no sorting), verify no duplicate/missing ids per fold, and only then average and renormalize. This preserves the core fold inference pipeline and ensemble semantics, but fixes alignment so probabilities correspond to the correct `eeg_id`s, which should substantially reduce KL.'
- What this solution (achieved 1.40995) has done: 'Your score is far worse than the target (lower-is-better), so we should fix the most likely remaining “silent” issue that can destroy KL while still producing a valid-looking CSV: averaging raw fold probabilities instead of averaging in log-space (geometric mean), which is usually better calibrated for KL and ensembling. I keep the same fold inference pipeline and the same checkpoints, but change only the ensembling step to compute a log-mean of probabilities with a tiny epsilon for numerical safety, then renormalize to sum to 1. I also add strict clipping/renormalization per fold before ensembling to prevent any tiny negatives/zeros from breaking log-space averaging. This is a minimal change focused on improving predictive quality without altering the core model/test calls.'
- What this solution (achieved 1.40995) has done: 'The current score is far worse than the target (lower-is-better), so the most likely remaining high-impact issue is that the external `python -m test` command is not actually running different folds: it never passes a `fold=` override, so you can end up producing identical predictions five times (or the same default fold each time), making the “ensemble” ineffective. I make a minimal change to `run_fold` to pass `fold={fold}` into Hydra so each checkpoint is evaluated on its intended fold configuration, while keeping the same model/test entrypoint and checkpoints. I also add a lightweight sanity check that warns if fold outputs look identical (helps catch silent failures without changing semantics), and keep your aligned log-mean ensembling and strict normalization.'
- What this solution (achieved 1.40995) has done: 'The biggest remaining reason for a very poor KL score despite “valid-looking” CSVs is usually that predictions are not actually aligned to the competition’s `eeg_id` list at inference time (or are duplicated/misaligned when `test.csv` ordering differs), so I make the inference and ensembling explicitly keyed to `test.csv`/`sample_submission.csv` order and also aggregate multiple rows per `eeg_id` if the external script outputs duplicates. I keep your core approach (convert parquet→npy, run the same external `python -m test` per fold, ensemble folds) unchanged, but I harden the postprocessing: groupby `eeg_id` (mean in probability space) per fold, then log-mean ensemble across folds, then final renormalization. This is a minimal change targeted at fixing silent ID/order/duplication issues that can destroy KL while still producing a syntactically correct submission. The output remains `/kaggle/working/submission.csv` with the required columns and rows summing to 1.'
- What this solution (achieved 1.4093) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.33917), so we should fix a likely remaining silent calibration issue without changing the external model/test logic. The biggest minimal win for KL in this competition is ensuring outputs are proper probabilities (softmax) rather than already-logits or unnormalized scores; if any fold file contains logits, your current “clip+normalize” can produce badly miscalibrated distributions. I add a conservative “auto-detect logits vs probabilities” step per fold: if rows don’t look like probabilities (e.g., negatives or sums far from 1), apply a numerically-stable softmax; otherwise keep as-is, then normalize. I also add a tiny amount of label-prior smoothing (blend with global class prior estimated from train votes) because KL strongly penalizes near-zero probabilities; this is a minimal post-processing change that often reduces KL without altering core inference.'
- What this solution (achieved 1.40962) has done: 'Your current KL (1.4093, lower-is-better) is much worse than the target (0.33917), so we should make a minimal, low-risk improvement in post-processing that is directly aligned with the KL metric. The biggest likely remaining issue is overconfidence (near-zero probabilities) which KL penalizes heavily, so I replace the fixed prior-blend (`alpha=0.02`) with a tiny, data-driven “temperature” smoothing (power transform toward uniform) applied after ensembling. This keeps your core inference exactly the same (same external `python -m test`, same checkpoints, same log-mean ensemble, same alignment), but typically reduces KL a lot by preventing extreme probabilities without changing row order or submission semantics. I keep the prior blend but reduce it slightly so we don’t overly bias away from the model while still avoiding zeros.'
- What this solution (achieved 1.40978) has done: 'Your score is much worse than the target (lower-is-better), so we should only make small post-processing adjustments that reduce KL without changing the model/test inference core. The highest-impact minimal fix is to prevent overconfident near-zero probabilities (KL explodes) by adding a tiny per-row probability floor before any log-mean and again after ensembling; this keeps distributions valid but avoids extreme penalties. Next, we should slightly soften the current temperature smoothing (it may be too weak) and make the prior blend adaptive per-row (stronger only when the model is very peaky), which improves calibration while preserving rankings. All changes stay within the existing pipeline: same external fold inference, same alignment/groupby, same log-mean ensemble, and still writes `/kaggle/working/submission.csv`.'
- What this solution (achieved 1.40978) has done: 'Your current KL (1.40978, lower-is-better) is far from the target (0.33917), so we should make a minimal post-processing change that specifically reduces KL’s heavy penalty on near-zero probabilities without touching the model/test inference core. The safest high-impact tweak here is to slightly increase the probability floor used during normalization/log-ensembling so no class goes extremely close to zero, and to apply the temperature smoothing a bit more (still very mild) to reduce overconfidence. I keep your fold running, alignment, groupby aggregation, and log-mean ensemble exactly the same, and only adjust `eps`/temperature values and make the row-wise floor explicit right before the final write. This preserves evaluation semantics (valid probabilities summing to 1) while improving calibration toward the target KL.'
- What this solution (achieved 1.40985) has done: 'Your current KL (1.40978, lower-is-better) is far above the target (0.33917), so we should improve calibration with the smallest post-processing changes that are directly relevant to KL without touching the external model/test inference core. The two highest-leverage, low-risk adjustments are (1) reduce the probability floor from `2e-4` to a much smaller value to avoid overly “uniformizing” predictions, while still preventing zeros that explode KL, and (2) slightly reduce temperature smoothing so we don’t wash out the model’s signal. I keep the same fold running, alignment/groupby, logits auto-detect, and log-mean ensembling; only the eps/temperature/prior-blend strength are adjusted. The submission format and strict sum-to-1 guarantees are preserved.'
- What this solution (achieved 1.40985) has done: 'Your current KL (1.40985, lower-is-better) is far from the target (0.33917), so we should fix the most likely remaining high-impact, silent issue without changing your core fold inference pipeline: the external `python -m test` may be writing predictions keyed by `spectrogram_id` or in a non-`eeg_id` identifier, and our current loader then “aligns” them incorrectly, effectively randomizing rows and exploding KL. I make the fold loader robust by (1) accepting `eeg_id` *or* `spectrogram_id` and mapping `spectrogram_id -> eeg_id` via `test.csv` when needed, and (2) never skipping a fold just because some ids are missing—fill missing rows with a conservative prior (so KL doesn’t blow up) and continue. This keeps your model/test invocation, log-mean ensemble, and smoothing logic intact, but prevents catastrophic ID/key mismatches and missing-row failures that can dominate KL. The submission writing and sum-to-1 guarantees remain unchanged.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
from pathlib import Path

sys.path.append("/kaggle/input/hms-mk-codes/")




## === cell 1
def _safe_pip_install(cmd: str) -> int:
    """
    Run pip install command safely. In Kaggle, this may or may not be needed
    depending on the environment; failures shouldn't crash the whole run.
    """
    try:
        return subprocess.run(cmd, shell=True, check=False).returncode
    except Exception as e:
        print(f"[WARN] pip install command failed to execute: {e}")
        return 1


wheels = [
    "/kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl",
    "/kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl",
    "/kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl",
    "/kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl",
]
if all(os.path.exists(w) for w in wheels):
    _safe_pip_install(
        "pip install /kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl "
        "--no-index --no-deps --force-reinstall"
    )
    _safe_pip_install(
        "pip install /kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl --no-index --no-deps"
    )
    _safe_pip_install(
        "pip install /kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl --no-index --no-deps"
    )
    _safe_pip_install(
        "pip install /kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl --no-deps --no-index"
    )
else:
    print(
        "[INFO] requirements-mk wheels not found in /kaggle/input; skipping local wheel installs."
    )


## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

Path(OUT_PATH).mkdir(parents=True, exist_ok=True)


## === cell 3
cmd = (
    f"cd /kaggle/input/hms-mk-codes && "
    f'python -m src.convert_parquet_to_npy --data_dir="{DATA_PATH}" --out_dir="{OUT_PATH}"'
)
ret = subprocess.run(cmd, shell=True, check=False)
if ret.returncode != 0:
    print(
        f"[WARN] convert_parquet_to_npy returned non-zero exit code: {ret.returncode}"
    )
else:
    print("[INFO] convert_parquet_to_npy completed.")


## === cell 4
subprocess.run("ls -la /kaggle/input | head -n 200", shell=True, check=False)


## === cell 5
checkpoints = [
    "epoch_012_val_loss_0.5037.ckpt",
    "epoch_013_val_loss_0.5103.ckpt",
    "epoch_014_val_loss_0.4728.ckpt",
    "epoch_014_val_loss_0.4944.ckpt",
    "epoch_014_val_loss_0.5053.ckpt",
]




## === cell 6
def run_fold(
    fold: int, ckpt_path: str, data_path: str = DATA_PATH, out_path: str = OUT_PATH
) -> str | None:
    fold_out = f"/kaggle/working/submission_fold{fold}.csv"
    base_sub = "/kaggle/working/submission.csv"

    if os.path.exists(base_sub):
        try:
            os.remove(base_sub)
        except Exception as e:
            print(f"[WARN] Could not remove stale {base_sub}: {e}")

    cmd = (
        f"cd /kaggle/input/hms-mk-codes && "
        f"python -m test "
        f'paths.data_dir="{data_path}" '
        f'data.test_eegs_dir="{out_path}" '
        f'ckpt_path="{ckpt_path}" '
        f"hydra=test "
        f'+model.test_output_dir="{out_path}" '
        f"experiment=conv1d_pseudo "
        f"+model.net.pretrained=False "
        f"fold={fold}"
    )
    print(f"[INFO] Running fold {fold}: {cmd}")
    ret = subprocess.run(cmd, shell=True, check=False)
    if ret.returncode != 0:
        print(
            f"[WARN] Fold {fold} test command failed with exit code {ret.returncode}. Skipping this fold."
        )
        return None

    if not os.path.exists(base_sub):
        print(
            f"[WARN] Fold {fold} completed but {base_sub} not found. Skipping this fold."
        )
        return None

    try:
        os.replace(base_sub, fold_out)
        print(f"[INFO] Wrote {fold_out}")
        return fold_out
    except Exception as e:
        print(f"[WARN] Could not move {base_sub} -> {fold_out}: {e}")
        return None


fold_ckpts = {
    0: "/kaggle/input/hms-mk-data/fold0_pseudo_log.ckpt",
    1: "/kaggle/input/hms-mk-data/fold1_pseudo_log.ckpt",
    2: "/kaggle/input/hms-mk-data/fold2_pseudo_log.ckpt",
    3: "/kaggle/input/hms-mk-data/fold3_pseudo_log.ckpt",
    4: "/kaggle/input/hms-mk-data/fold4_pseudo_log.ckpt",
}
produced = {}
for f, ckpt in fold_ckpts.items():
    produced[f] = run_fold(f, ckpt)

print(
    "[INFO] Produced fold files:", {k: v for k, v in produced.items() if v is not None}
)


## === cell 7
import pandas as pd
import numpy as np

TARGET_COLS = [
    "seizure_vote",
    "lpd_vote",
    "gpd_vote",
    "lrda_vote",
    "grda_vote",
    "other_vote",
]

EPS_FLOOR = 1e-6
TEMP_T = 1.25

ALPHA_BASE = 0.003
ALPHA_MAX_EXTRA = (
    0.015  # max alpha becomes ALPHA_BASE + ALPHA_MAX_EXTRA when very peaky
)


def _softmax_stable(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64, copy=False)
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    s = np.sum(ex, axis=1, keepdims=True)
    s[s == 0] = 1.0
    return ex / s


def _normalize_probs(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float64, copy=False
    )
    arr = np.clip(arr, 0.0, None)
    rs = arr.sum(axis=1, keepdims=True)
    zero_mask = rs[:, 0] == 0
    if np.any(zero_mask):
        arr[zero_mask] = 1.0 / arr.shape[1]
        rs = arr.sum(axis=1, keepdims=True)
    arr = arr / rs
    arr = np.clip(arr, eps, 1.0)
    arr = arr / arr.sum(axis=1, keepdims=True)
    return arr


def _maybe_logits_to_probs(arr: np.ndarray) -> np.ndarray:
    """
    If fold outputs are logits (or otherwise not probabilities), convert with softmax;
    otherwise keep as probabilities. This preserves core inference and only fixes post-processing.
    """
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float64, copy=False
    )

    row_sums = arr.sum(axis=1)
    has_neg = np.any(arr < 0)
    sums_far = np.mean(np.abs(row_sums - 1.0) > 0.05)  # tolerant band
    too_large = np.any(np.abs(arr) > 5.0)

    if has_neg or sums_far > 0.5 or too_large:
        probs = _softmax_stable(arr)
        return probs

    return arr


def _compute_train_prior(
    train_csv_path="/kaggle/input/hms-harmful-brain-activity-classification/train.csv",
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Blend predictions slightly with global class prior to avoid extreme zeros (KL penalty).
    """
    try:
        tr = pd.read_csv(train_csv_path, usecols=TARGET_COLS)
        tr = tr.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        prior = tr.sum(axis=0).to_numpy(dtype=np.float64)
        if prior.sum() <= 0:
            return np.full(len(TARGET_COLS), 1.0 / len(TARGET_COLS), dtype=np.float64)
        prior = prior / prior.sum()
        prior = np.clip(prior, eps, 1.0)
        prior = prior / prior.sum()
        return prior
    except Exception as e:
        print(f"[WARN] Could not compute train prior from train.csv: {e}")
        return np.full(len(TARGET_COLS), 1.0 / len(TARGET_COLS), dtype=np.float64)


TRAIN_PRIOR = _compute_train_prior()


def _build_id_maps(
    test_csv_path: str,
) -> tuple[pd.Index, dict[int, int], dict[int, int]]:
    """
    Change (score-critical, minimal): build robust id mapping to avoid silent row misalignment.
    If fold outputs are keyed by spectrogram_id instead of eeg_id, map to eeg_id via test.csv.
    """
    test = pd.read_csv(test_csv_path, usecols=["eeg_id", "spectrogram_id"])
    test["eeg_id"] = pd.to_numeric(test["eeg_id"], errors="coerce")
    test["spectrogram_id"] = pd.to_numeric(test["spectrogram_id"], errors="coerce")
    test = test.dropna(subset=["eeg_id", "spectrogram_id"])
    test["eeg_id"] = test["eeg_id"].astype(np.int64)
    test["spectrogram_id"] = test["spectrogram_id"].astype(np.int64)

    spec_to_eeg = dict(
        zip(test["spectrogram_id"].to_numpy(), test["eeg_id"].to_numpy())
    )
    eeg_to_spec = dict(
        zip(test["eeg_id"].to_numpy(), test["spectrogram_id"].to_numpy())
    )
    return test.index, spec_to_eeg, eeg_to_spec


def _load_and_align_fold(
    fold_path: str,
    sample_index: pd.Index,
    spec_to_eeg: dict[int, int] | None,
    fill_with_prior: np.ndarray,
) -> np.ndarray | None:
    """
    Change (score-critical, minimal): accept folds keyed by either eeg_id or spectrogram_id,
    and do NOT skip folds due to partial coverage—fill missing rows with prior instead.
    This prevents catastrophic KL from mis-keyed ids or missing ids.
    """
    try:
        sol = pd.read_csv(fold_path)
    except Exception as e:
        print(f"[WARN] Failed reading {fold_path}: {e}")
        return None

    missing_cols = [c for c in TARGET_COLS if c not in sol.columns]
    if missing_cols:
        print(f"[WARN] {fold_path} missing columns {missing_cols} (skipping)")
        return None

    id_col = None
    if "eeg_id" in sol.columns:
        id_col = "eeg_id"
    elif "spectrogram_id" in sol.columns and spec_to_eeg is not None:
        id_col = "spectrogram_id"
    else:
        print(
            f"[WARN] {fold_path} has no usable id column ('eeg_id' or 'spectrogram_id'); skipping"
        )
        return None

    sol_small = sol[[id_col] + TARGET_COLS].copy()
    sol_small[TARGET_COLS] = sol_small[TARGET_COLS].apply(
        pd.to_numeric, errors="coerce"
    )

    if id_col == "spectrogram_id":
        sol_small[id_col] = pd.to_numeric(sol_small[id_col], errors="coerce")
        sol_small = sol_small.dropna(subset=[id_col])
        sol_small[id_col] = sol_small[id_col].astype(np.int64)
        sol_small["eeg_id"] = sol_small[id_col].map(spec_to_eeg)
        sol_small = sol_small.dropna(subset=["eeg_id"])
        sol_small["eeg_id"] = sol_small["eeg_id"].astype(np.int64)
        key = "eeg_id"
    else:
        sol_small[id_col] = pd.to_numeric(sol_small[id_col], errors="coerce")
        sol_small = sol_small.dropna(subset=[id_col])
        sol_small[id_col] = sol_small[id_col].astype(np.int64)
        sol_small = sol_small.rename(columns={id_col: "eeg_id"})
        key = "eeg_id"

    sol_g = sol_small.groupby(key, sort=False, as_index=True)[TARGET_COLS].mean()
    sol_g = sol_g.reindex(sample_index)

    raw = sol_g.to_numpy(dtype=np.float64)

    nan_rows = np.isnan(raw).any(axis=1)
    if np.any(nan_rows):
        n_missing = int(nan_rows.sum())
        print(
            f"[WARN] {fold_path}: {n_missing} missing eeg_id(s) after alignment; filling with prior for stability."
        )
        raw[nan_rows] = fill_with_prior[None, :]

    preds = _maybe_logits_to_probs(raw)
    preds = _normalize_probs(preds, eps=EPS_FLOOR)
    return preds


def _apply_temperature_smoothing(
    probs: np.ndarray, t: float, eps: float = 1e-12
) -> np.ndarray:
    """
    Soften overconfident distributions by applying p_i <- p_i^(1/t) with t>1.
    """
    probs = _normalize_probs(probs, eps=eps)
    if t is None or float(t) <= 1.0:
        return probs
    power = 1.0 / float(t)
    probs = np.power(np.clip(probs, eps, 1.0), power)
    probs = _normalize_probs(probs, eps=eps)
    return probs


def _adaptive_prior_blend(probs: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """
    Change (KL-oriented, minimal): slightly lighter adaptive blend than before to reduce bias.
    """
    probs = _normalize_probs(probs, eps=1e-12)
    maxp = probs.max(axis=1)  # (n,)
    alpha = ALPHA_BASE + ALPHA_MAX_EXTRA * np.clip((maxp - 0.75) / 0.25, 0.0, 1.0)
    blended = (1.0 - alpha[:, None]) * probs + alpha[:, None] * prior[None, :]
    return _normalize_probs(blended, eps=1e-12)


def merge_preds(
    folds=(0, 1, 2, 3, 4),
    sample_sub_path="/kaggle/input/hms-harmful-brain-activity-classification/sample_submission.csv",
    test_csv_path="/kaggle/input/hms-harmful-brain-activity-classification/test.csv",
):
    """
    Align by eeg_id to sample_submission order, log-mean ensemble across folds, then apply
    minimal KL-oriented smoothing/calibration.
    """
    sample = pd.read_csv(sample_sub_path)
    if "eeg_id" not in sample.columns:
        raise ValueError("sample_submission.csv is missing 'eeg_id' column.")
    for c in TARGET_COLS:
        if c not in sample.columns:
            raise ValueError(
                f"sample_submission.csv is missing required target column: {c}"
            )

    try:
        test = pd.read_csv(test_csv_path, usecols=["eeg_id"])
        if len(test) == len(sample):
            if not np.array_equal(
                test["eeg_id"].to_numpy(), sample["eeg_id"].to_numpy()
            ):
                print(
                    "[WARN] test.csv eeg_id order differs from sample_submission; using sample_submission order."
                )
        else:
            print(
                "[WARN] test.csv row count differs from sample_submission; using sample_submission order."
            )
    except Exception as e:
        print(f"[WARN] Could not validate test.csv alignment: {e}")

    try:
        _, spec_to_eeg, _ = _build_id_maps(test_csv_path)
    except Exception as e:
        print(f"[WARN] Could not build spectrogram->eeg id map from test.csv: {e}")
        spec_to_eeg = None

    sample_ids = (
        pd.to_numeric(sample["eeg_id"], errors="coerce").astype(np.int64).to_numpy()
    )
    sample_index = pd.Index(sample_ids, name="eeg_id")

    eps_log = EPS_FLOOR

    log_accum = None
    used_folds = 0
    used_fold_ids = []
    fold_signatures = {}

    fill_prior = TRAIN_PRIOR.astype(np.float64, copy=False)

    for fold in folds:
        fold_path = f"/kaggle/working/submission_fold{fold}.csv"
        if not os.path.exists(fold_path):
            print(f"[WARN] Missing fold prediction file: {fold_path} (skipping)")
            continue

        fold_preds = _load_and_align_fold(
            fold_path=fold_path,
            sample_index=sample_index,
            spec_to_eeg=spec_to_eeg,
            fill_with_prior=fill_prior,
        )
        if fold_preds is None:
            continue

        fold_signatures[fold] = float(np.round(fold_preds[:50].mean(), 12))
        fold_log = np.log(np.clip(fold_preds, eps_log, 1.0))

        if log_accum is None:
            log_accum = fold_log
        else:
            log_accum += fold_log

        used_folds += 1
        used_fold_ids.append(fold)

    if len(set(fold_signatures.values())) <= 1 and len(fold_signatures) > 1:
        print(
            "[WARN] Fold predictions appear nearly identical by quick signature; "
            "this can hurt ensemble quality. (Not failing; just warning.)"
        )

    if used_folds == 0:
        print(
            "[WARN] No fold predictions loaded; falling back to sample_submission probabilities."
        )
        preds = sample[TARGET_COLS].to_numpy(dtype=np.float64)
        preds = _normalize_probs(preds, eps=EPS_FLOOR)
    else:
        print(f"[INFO] Ensembling folds (groupby+aligned, log-mean): {used_fold_ids}")
        log_mean = log_accum / used_folds
        preds = np.exp(log_mean)
        preds = _normalize_probs(preds, eps=EPS_FLOOR)

    preds = _apply_temperature_smoothing(preds, t=TEMP_T, eps=EPS_FLOOR)
    preds = _adaptive_prior_blend(preds, TRAIN_PRIOR)
    preds = _normalize_probs(preds, eps=EPS_FLOOR)

    out = sample.copy()
    out["eeg_id"] = sample_ids
    out[TARGET_COLS] = preds
    return out




## === cell 8
sol = merge_preds(folds=(0, 1, 2, 3, 4))


## === cell 9
sub_path = "/kaggle/working/submission.csv"
sol.to_csv(sub_path, index=False)

assert (
    sol.columns.tolist() == ["eeg_id"] + TARGET_COLS
), "Submission columns do not match required format."
row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
if not np.all(np.isfinite(row_sums)):
    raise ValueError("Non-finite row sums detected in submission.")
max_abs_deviation = float(np.max(np.abs(row_sums - 1.0)))
print(
    f"Wrote submission: {sub_path}  shape={sol.shape}  max|sum-1|={max_abs_deviation:.3e}"
)
print(sol.head())


## === cell 10
sol
