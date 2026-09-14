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

0.3390572951476089

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'Your current run likely doesn’t yield a score because the pipeline can silently skip inference (missing `/kaggle/input/hms-mk-codes` or checkpoints) and/or produce a submission with `eeg_id` order/rows not guaranteed to match `sample_submission.csv`. I make minimal changes to (1) always produce a valid `submission.csv` aligned exactly to the sample submission’s `eeg_id` list, (2) robustly merge fold predictions onto that index (so missing/extra IDs can’t break validity), and (3) guarantee numeric stability and row-wise normalization for KL (clip + renormalize). This preserves your core approach (fold CSV averaging) while ensuring Kaggle accepts the file and you can obtain a real score to move toward the target. If inference assets are missing, it still create a correct uniform submission.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is far from the target (0.33906), so we should improve performance without changing your core approach (fold checkpoint inference → CSV averaging). The most likely cause of a very bad KL score here is a mismatch between what the fold scripts output and what Kaggle expects: your pipeline currently averages already-normalized “vote-like” columns across folds, but it doesn’t enforce consistent column order, handle duplicate `eeg_id` rows, or (critically) apply a final safety calibration to avoid overconfident near-zeros which are heavily penalized by KL. I make minimal, metric-aligned changes: (1) aggregate fold predictions by `eeg_id` (mean if duplicates), (2) enforce exact sample-submission `eeg_id` order and columns, and (3) apply a very small epsilon-smoothing (“label smoothing” on predictions) before renormalization to reduce KL blow-ups from tiny probabilities. These changes keep your model and inference intact and only adjust robust post-processing to move the score down toward the target.'
- What this solution (achieved 1.40995) has done: 'Your KL score is much worse than the target (1.40995 vs 0.33906, lower-is-better), so we should improve without changing your model/inference core. The most likely “cheap win” here is fixing probability calibration for KL: your current epsilon smoothing is extremely small and can still leave near-zero probabilities that KL penalizes heavily. I keep your exact fold-CSV averaging logic, but (1) add a robust probability sanitizer (clip + renormalize + optional temperature on logits converted from probabilities) and (2) tune smoothing strength slightly upward in a controlled way to reduce KL blow-ups while preserving fold outputs. I also make the pipeline resilient if some folds are missing (use available folds instead of erroring), which avoids accidentally submitting a partial/invalid ensemble.'
- What this solution (achieved 1.40995) has done: 'Your score (1.40995, lower-is-better) is far from the target (0.33906), and the simplest way to move toward it without touching your model/inference core is to reduce KL “blow-ups” caused by overconfident near-zero probabilities and to avoid applying the calibration twice. I make two minimal, metric-aligned post-processing changes: (1) remove the duplicate second call to `_sanitize_and_calibrate_probs` (currently applied both inside `merge_preds` and again right before saving), and (2) make the probability smoothing a bit stronger and simpler (clip + uniform-mix only), which is safer for KL than the current log/temperature transform on already-averaged probabilities. Everything else (fold file loading, alignment to sample submission, averaging) stays the same, and the script still always emit a valid `submission.csv` with row-wise sums exactly 1.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.33906), so we should improve with minimal risk and without touching your model/inference. The most impactful safe lever here is post-processing calibration for KL: reduce overconfident near-zeros and mild overconfidence by using a slightly stronger but still simple uniform-mix smoothing plus a light power/temperature flattening on probabilities (not logits), then renormalize. I also make the fold-merge more robust by weighting folds equally but optionally incorporating a “prior” class distribution estimated from `train.csv` votes (very small weight) to stabilize predictions when some folds are missing or miscalibrated. These changes keep your fold CSV ensemble core logic intact while directly targeting KL’s sensitivity to tiny probabilities and miscalibration.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.33906), so we should improve with minimal, metric-aligned post-processing while keeping your fold-CSV ensemble core unchanged. The most likely issue is that the current calibration is too aggressive/unhelpful (uniform/prior mixing + power) and can distort already-good fold probabilities, plus it doesn’t explicitly handle “votes vs probabilities” scale in a principled way. I make a small, contained change: replace the current calibration with a simple Dirichlet-style pseudocount smoothing toward the empirical train vote prior (no power transform), and tune the strength so probabilities never get near-zero (reducing KL blow-ups) without washing out signal. I keep all I/O, fold alignment, averaging, and submission formatting identical, and still enforce strict row-wise normalization.'
- What this solution (achieved 1.40995) has done: 'Your score is much worse than the target (1.40995 vs 0.33906, lower-is-better), so we should *improve* while keeping your inference/ensemble core unchanged. The most likely cheap win is that the current post-processing (`alpha_prior=6.0` plus a relatively large `eps=5e-4`) is washing out model signal and pushing predictions too close to the train prior/uniform, which can be very bad for KL on this task. I make a minimal, metric-aligned adjustment: (1) reduce the probability floor `eps` to a safer small value to avoid KL blow-ups without over-flattening, and (2) reduce the Dirichlet pseudocount strength so we keep more of the fold ensemble’s information. Everything else (fold reading, alignment to sample submission, duplicate handling, averaging, final renorm, and CSV output) stays the same.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is still far from the target (0.33906), so we should improve it with minimal, metric-aligned post-processing while keeping your fold-CSV ensemble logic intact. The most likely remaining issue is that the current prior-mixing strength (`ALPHA_PRIOR=0.8`) can still wash out useful model signal, while the probability floor can be kept tiny just to prevent KL blow-ups without flattening. I make a small adjustment by reducing `ALPHA_PRIOR` (less bias toward the global prior) and keeping `CALIB_EPS` small, while leaving all I/O, fold alignment, duplicate handling, averaging, and final normalization exactly the same. This is a contained change that should move KL down (better) toward your target without altering your inference/model core.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.33906), so we should improve it with minimal, metric-aligned post-processing while keeping your fold-CSV ensemble core unchanged. The biggest safe lever here is calibration for KL: reduce overconfident probabilities (especially near-zeros) without washing out model signal. I replace the current train-prior Dirichlet mixing (which can bias predictions incorrectly) with a small, symmetric Dirichlet/pseudocount smoothing toward uniform plus a light probability “flattening” (power < 1) applied once after ensembling, then renormalize. This preserves your inference, fold averaging, alignment, and submission formatting, while directly addressing KL sensitivity.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target (0.33906), so we should improve with the smallest safe change while keeping your fold-CSV ensemble core intact. The biggest likely issue is that the current post-processing (uniform mixing + power flattening) can distort already-good fold probabilities and push them away from what the model learned, hurting KL. I remove the power transform and reduce the amount of uniform smoothing to a tiny “probability floor” style Dirichlet smoothing that mainly prevents near-zeros (KL blow-ups) without washing out signal. I also keep the strict alignment/duplicate handling unchanged and add a sanity check that no fold file contains “vote counts” (sum >> 1), in which case we normalize them once before ensembling.'
- What this solution (achieved 1.40995) has done: 'Your KL (lower-is-better) is far above the target, so we should *improve* it with the smallest change that directly targets KL sensitivity while preserving your fold-CSV ensemble core. The most likely low-risk gain is to reduce overconfident near-zeros without washing out signal: your current uniform-mix smoothing (`ALPHA_UNI=0.02`) can still distort predictions; instead we switch to a tiny *probability floor via symmetric Dirichlet pseudocount* that mainly prevents zeros and keeps the original shape. We also make the “vote-count detection” actually normalize only when needed (currently it prints info but always normalizes anyway), and we optionally ensemble duplicate fold outputs more robustly by renormalizing after the groupby mean. All I/O, fold inference, alignment to `sample_submission.csv`, and averaging remain unchanged, and the script still guarantees row sums of 1.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is far above the target (0.33906), so we should improve with the smallest change that directly targets KL sensitivity while preserving your fold-CSV ensemble core. The most likely issue is that your post-processing is too weak to prevent near-zero probabilities (KL blow-ups) when some fold outputs are overconfident/miscalibrated; a slightly stronger but still minimal symmetric Dirichlet smoothing (probability floor) is a safe lever. I only adjust the calibration strength and make the “vote-count detection” actually conditional (so we don’t unnecessarily renormalize already-good probabilities beyond the required safety normalization). Everything else—fold inference, alignment to `sample_submission.csv`, averaging, and strict row-sum=1—stays the same and still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
from pathlib import Path

CODE_PATH = "/kaggle/input/hms-mk-codes"
print("Python:", sys.version)
print("CODE_PATH exists:", os.path.exists(CODE_PATH))
if os.path.exists(CODE_PATH) and CODE_PATH not in sys.path:
    sys.path.append(CODE_PATH)
print("sys.path contains CODE_PATH:", CODE_PATH in sys.path)



## === cell 1
import subprocess, shlex, os


def _run(cmd):
    print("RUN:", cmd)
    subprocess.check_call(shlex.split(cmd))


REQ_DIR = "/kaggle/input/requirements-mk"
if os.path.isdir(REQ_DIR):
    _run(
        f"pip install {REQ_DIR}/antlr4_python3_runtime-4.9.2-py3-none-any.whl --no-index --no-deps --force-reinstall"
    )
    _run(f"pip install {REQ_DIR}/omegaconf-2.3.0-py3-none-any.whl --no-index --no-deps")
    _run(
        f"pip install {REQ_DIR}/hydra_core-1.3.2-py3-none-any.whl --no-index --no-deps"
    )
    _run(f"pip install {REQ_DIR}/lightning-2.2.1-py3-none-any.whl --no-deps --no-index")
else:
    print(f"WARNING: {REQ_DIR} not found; skipping custom wheel installs.")



## === cell 2
from pathlib import Path

DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"

Path(OUT_PATH).mkdir(parents=True, exist_ok=True)
print("DATA_PATH exists:", Path(DATA_PATH).exists())
print("OUT_PATH exists:", Path(OUT_PATH).exists())



## === cell 3
import subprocess
import os

if os.path.exists(CODE_PATH):
    subprocess.check_call(
        [
            "python",
            "-m",
            "src.convert_parquet_to_npy",
            f"--data_dir={DATA_PATH}",
            f"--out_dir={OUT_PATH}",
        ],
        cwd=CODE_PATH,
    )
else:
    print("WARNING: CODE_PATH not available; skipping parquet->npy conversion step.")



## === cell 4
import os

print("Listing /kaggle/input/hms-mk-data (if present):")
print(
    os.listdir("/kaggle/input/hms-mk-data")[:50]
    if os.path.exists("/kaggle/input/hms-mk-data")
    else "NOT FOUND"
)



## === cell 5
checkpoints = [
    "epoch_012_val_loss_0.5037.ckpt",
    "epoch_013_val_loss_0.5103.ckpt",
    "epoch_014_val_loss_0.4728.ckpt",
    "epoch_014_val_loss_0.4944.ckpt",
    "epoch_014_val_loss_0.5053.ckpt",
]



## === cell 6
import subprocess
from pathlib import Path
import os

fold_ckpts = [
    "/kaggle/input/hms-mk-data/fold0_pseudo_log.ckpt",
    "/kaggle/input/hms-mk-data/fold1_pseudo_log.ckpt",
    "/kaggle/input/hms-mk-data/fold2_pseudo_log.ckpt",
    "/kaggle/input/hms-mk-data/fold3_pseudo_log.ckpt",
    "/kaggle/input/hms-mk-data/fold4_pseudo_log.ckpt",
]

if os.path.exists(CODE_PATH):
    for fold, ckpt_path in enumerate(fold_ckpts):
        if not os.path.exists(ckpt_path):
            print(
                f"WARNING: Missing ckpt for fold {fold}: {ckpt_path} (skipping this fold)"
            )
            continue

        cmd = [
            "python",
            "-m",
            "test",
            f"paths.data_dir={DATA_PATH}",
            f"data.test_eegs_dir={OUT_PATH}",
            f"ckpt_path={ckpt_path}",
            "hydra=test",
            f"+model.test_output_dir={OUT_PATH}",
            "+model.net.pretrained=False",
        ]
        subprocess.check_call(cmd, cwd=CODE_PATH)

        src_sub = Path("/kaggle/working/submission.csv")
        dst_sub = Path(f"/kaggle/working/submission_fold{fold}.csv")
        if not src_sub.exists():
            raise FileNotFoundError(
                f"Expected {src_sub} to be created by fold {fold} inference."
            )
        src_sub.replace(dst_sub)

    print(
        "Created fold submissions:",
        [p.name for p in Path("/kaggle/working").glob("submission_fold*.csv")],
    )
else:
    print("WARNING: CODE_PATH not available; skipping fold inference.")



## === cell 7
import numpy as np
import pandas as pd
import os

try:
    from src.settings import TARGET_COLS  # type: ignore
except Exception as e:
    print("WARNING: Could not import TARGET_COLS from src.settings due to:", repr(e))
    TARGET_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

TARGET_COLS = list(TARGET_COLS)
ID_COL = "eeg_id"


def _load_sample_index():
    ss_path = f"{DATA_PATH}/sample_submission.csv"
    ss = pd.read_csv(ss_path, usecols=[ID_COL] + TARGET_COLS)
    ss = ss[[ID_COL] + TARGET_COLS].copy()
    return ss


def _normalize_probs(arr, eps=1e-12):
    arr = np.asarray(arr, dtype="float64")
    arr = np.clip(arr, eps, None)
    s = arr.sum(axis=1, keepdims=True)
    s = np.where(s <= 0, 1.0, s)
    arr = arr / s
    return arr


CALIB_EPS = 1e-12

DIRICHLET_ALPHA = 5e-3  # was 1e-3


def _sanitize_and_calibrate_probs(arr, eps=CALIB_EPS, alpha=DIRICHLET_ALPHA):
    """
    Apply calibration ONCE after ensembling:
      1) clip + renorm
      2) symmetric Dirichlet pseudocount smoothing:
           p' = (p + alpha) / sum(p + alpha)
         (reduces KL sensitivity to tiny probs)
      3) renorm
    """
    p = np.asarray(arr, dtype="float64")
    p = _normalize_probs(p, eps=eps)

    if alpha and alpha > 0:
        p = p + float(alpha)
        p = _normalize_probs(p, eps=eps)

    return p


def _read_and_align_fold(fp, base_ids):
    sol = pd.read_csv(fp)

    missing = [c for c in ([ID_COL] + TARGET_COLS) if c not in sol.columns]
    if missing:
        raise ValueError(f"{fp} missing columns: {missing}. Found: {list(sol.columns)}")

    sol = sol[[ID_COL] + TARGET_COLS].copy()
    sol[TARGET_COLS] = sol[TARGET_COLS].astype("float64")

    if sol.duplicated(subset=[ID_COL]).any():
        sol = sol.groupby(ID_COL, as_index=False)[TARGET_COLS].mean()
        sol[TARGET_COLS] = _normalize_probs(sol[TARGET_COLS].values, eps=CALIB_EPS)

    merged = base_ids.merge(sol, on=ID_COL, how="left")
    if merged[TARGET_COLS].isna().any().any():
        merged[TARGET_COLS] = merged[TARGET_COLS].fillna(1.0 / len(TARGET_COLS))

    vals = merged[TARGET_COLS].values

    row_sum_median = float(np.median(np.sum(vals, axis=1)))
    if row_sum_median > 1.5:
        print(
            f"INFO: {fp} looks like vote-counts (median row-sum={row_sum_median:.3f}); normalizing to probs."
        )
        vals = _normalize_probs(vals, eps=CALIB_EPS)
    else:
        vals = _normalize_probs(vals, eps=CALIB_EPS)

    return vals


def merge_preds(folds=(0, 1, 2, 3, 4)):
    ss = _load_sample_index()
    base_ids = ss[[ID_COL]].copy()

    preds_list = []
    used_folds = []
    for fold in folds:
        fp = f"/kaggle/working/submission_fold{fold}.csv"
        if not os.path.exists(fp):
            print(f"WARNING: Missing fold prediction file: {fp} (skipping fold {fold})")
            continue
        preds_list.append(_read_and_align_fold(fp, base_ids))
        used_folds.append(fold)

    if len(preds_list) == 0:
        raise FileNotFoundError("No fold prediction files found to merge.")

    preds = np.mean(np.stack(preds_list, axis=0), axis=0)

    preds = _sanitize_and_calibrate_probs(preds, eps=CALIB_EPS, alpha=DIRICHLET_ALPHA)
    preds = _normalize_probs(preds, eps=CALIB_EPS)

    out = base_ids.copy()
    out[TARGET_COLS] = preds
    print("Merged folds:", used_folds)
    return out[[ID_COL] + TARGET_COLS]


def make_uniform_fallback_submission():
    ss = _load_sample_index()
    out = ss[[ID_COL]].copy()
    out[TARGET_COLS] = 1.0 / len(TARGET_COLS)
    out[TARGET_COLS] = _normalize_probs(out[TARGET_COLS].values, eps=CALIB_EPS)
    return out[[ID_COL] + TARGET_COLS]




## === cell 8
from pathlib import Path
import numpy as np

available_folds = []
for f in (0, 1, 2, 3, 4):
    if Path(f"/kaggle/working/submission_fold{f}.csv").exists():
        available_folds.append(f)

if len(available_folds) > 0:
    sol = merge_preds(folds=tuple(available_folds))
    print("Using fold ensemble from folds:", available_folds)
else:
    sol = make_uniform_fallback_submission()
    print("Using uniform fallback submission (no fold prediction files found).")

sol[TARGET_COLS] = sol[TARGET_COLS].astype("float64")
sol[TARGET_COLS] = _normalize_probs(sol[TARGET_COLS].values, eps=CALIB_EPS)

sub_path = "/kaggle/working/submission.csv"
sol.to_csv(sub_path, index=False)

row_sums = sol[TARGET_COLS].sum(axis=1).values
print("submission path:", sub_path)
print("submission shape:", sol.shape)
print("min/max row sum:", float(row_sums.min()), float(row_sums.max()))
print(sol.head())



## === cell 9
sol
