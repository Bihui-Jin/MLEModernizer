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

0.319348589155934

# 6. Current score

1.40995

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.40995) has done: 'The merge step fails because the earlier `python -m test` commands likely didn’t generate `/kaggle/working/submission.csv` (or the moves failed), so the expected `submission_fold*_*.csv` files don’t exist. I (1) make all shell steps fail-fast and print diagnostics so missing checkpoints/paths are caught early, and (2) make ensembling robust by discovering whatever fold/version submission files actually exist and averaging them, falling back to the sample_submission uniform distribution only if none are found. This keeps the core model/inference logic unchanged (still uses the provided MK code/ckpts) but ensures an end-to-end run always produces a valid `submission.csv` with rows summing to 1. Finally, I write the submission and re-check schema and probability normalization to avoid Kaggle “submission invalid” errors.'
- What this solution (achieved 1.40995) has done: 'I fix the two blockers causing your pipeline to fall back to uniform predictions (which explains the very poor 1.40995 score): the missing `/kaggle/input/hms-mk-codes` path and the failing offline `pip install` wheel step. The patch makes the code auto-detect the correct input directory for the MK code bundle, and it conditionally skips wheel installs if they’re unavailable (while still running them when present). With those fixed, the existing conversion + fold inference + ensembling logic can actually run and should move the score substantially toward your 0.319 target without changing model logic. I also keep the submission validation/normalization to guarantee a valid `.csv` output.'
- What this solution (achieved 1.40995) has done: 'I fix the code-directory auto-detection so it finds the MK `src/` package reliably (your current search often returns a directory without `src`, causing the early FileNotFoundError). Then I make `run_one_test()` fail with a readable error message by printing the `python -m test` stderr when it happens, and I ensure the inference command uses an absolute `PYTHONPATH` pointing at the detected code root so module resolution works in Kaggle. Finally, I keep your ensembling logic the same but make it robust to partial fold outputs (so you don’t fall back to uniform predictions, which explains the very poor KL score) and always write a valid `submission.csv` with normalized probabilities.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
from pathlib import Path

CODES_DIR_CANDIDATES = [
    "/kaggle/input/hms-mk-codes",
    "/kaggle/input/hms-mk-code",
    "/kaggle/input/hms-mk",
]
SEARCH_ROOTS = ["/kaggle/input", "/kaggle/data/input", "/kaggle/data"]


def _is_mk_code_root(p: Path) -> bool:
    return p.is_dir() and (p / "src").is_dir()


def _find_codes_dir() -> str | None:
    for p in CODES_DIR_CANDIDATES:
        pp = Path(p)
        if _is_mk_code_root(pp):
            return str(pp)

    for root in SEARCH_ROOTS:
        rp = Path(root)
        if not rp.exists():
            continue

        for level1 in rp.glob("*"):
            if _is_mk_code_root(level1):
                return str(level1)
            if not level1.is_dir():
                continue
            for level2 in level1.glob("*"):
                if _is_mk_code_root(level2):
                    return str(level2)
                if not level2.is_dir():
                    continue
                for level3 in level2.glob("*"):
                    if _is_mk_code_root(level3):
                        return str(level3)

    return None


CODES_DIR = _find_codes_dir()
print("Python:", sys.version)
print("Detected CODES_DIR:", CODES_DIR)

if CODES_DIR is not None and CODES_DIR not in sys.path:
    sys.path.append(CODES_DIR)


def run_cmd(cmd: str, *, cwd: str | None = None, extra_env: dict | None = None) -> None:
    print(f"\n[RUN] {cmd}\n  cwd={cwd}")
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    r = subprocess.run(
        cmd, shell=True, cwd=cwd, text=True, capture_output=True, env=env
    )
    if r.stdout:
        print(r.stdout)
    if r.returncode != 0:
        if r.stderr:
            print(r.stderr)
        raise RuntimeError(f"Command failed with return code {r.returncode}: {cmd}")
    if r.stderr:
        print(r.stderr)




## === cell 1
REQ_DIR = Path("/kaggle/input/requirements-mk")
wheels = [
    (
        "antlr4_python3_runtime-4.9.2-py3-none-any.whl",
        "--no-index --no-deps --force-reinstall",
    ),
    ("omegaconf-2.3.0-py3-none-any.whl", "--no-index --no-deps"),
    ("hydra_core-1.3.2-py3-none-any.whl", "--no-index --no-deps"),
    ("lightning-2.2.1-py3-none-any.whl", "--no-deps --no-index"),
]

if REQ_DIR.exists():
    for whl, flags in wheels:
        p = REQ_DIR / whl
        if p.exists():
            run_cmd(f"pip install {p} {flags}")
        else:
            print(f"WARNING: wheel not found, skipping: {p}")
else:
    print(f"WARNING: requirements dir not found, skipping offline installs: {REQ_DIR}")



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"
os.makedirs(OUT_PATH, exist_ok=True)

print("DATA_PATH exists:", Path(DATA_PATH).exists())
print("OUT_PATH:", OUT_PATH)



## === cell 3
if CODES_DIR is None or not (Path(CODES_DIR) / "src").is_dir():
    print(
        "WARNING: MK codes directory not found (or missing src/). "
        "Conversion/inference will be skipped; will fall back to ensembling existing files or uniform."
    )
else:
    extra_env = {
        "PYTHONPATH": CODES_DIR
        + (":" + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else "")
    }
    run_cmd(
        f"python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}",
        cwd=CODES_DIR,
        extra_env=extra_env,
    )

print("Working dir files (top-level):")
run_cmd("ls -la /kaggle/working | head -n 200")



## === cell 4
from pathlib import Path


def run_one_test(ckpt_path: str, experiment: str, out_csv: str) -> None:
    if CODES_DIR is None or not (Path(CODES_DIR) / "src").is_dir():
        raise FileNotFoundError(
            "MK code root with src/ was not detected; cannot run inference."
        )

    extra_env = {
        "PYTHONPATH": CODES_DIR
        + (":" + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else "")
    }

    cmd = (
        f"python -m test "
        f"paths.data_dir={DATA_PATH} "
        f"data.test_eegs_dir={OUT_PATH} "
        f"ckpt_path={ckpt_path} "
        f"hydra=test +model.test_output_dir={OUT_PATH} "
        f"experiment={experiment} "
        f"+model.net.pretrained=False"
    )
    run_cmd(cmd, cwd=CODES_DIR, extra_env=extra_env)

    produced = Path(OUT_PATH) / "submission.csv"
    if not produced.exists():
        run_cmd("ls -la /kaggle/working | head -n 200")
        raise FileNotFoundError(
            f"Expected {produced} to be created by inference but it was not found."
        )

    target = Path(out_csv)
    if target.exists():
        target.unlink()
    produced.rename(target)
    print(f"Moved {produced} -> {target}")


CKPT_ROOT = Path("/kaggle/input/hms-mk-data")
can_infer = (
    (CODES_DIR is not None)
    and (Path(CODES_DIR) / "src").is_dir()
    and CKPT_ROOT.exists()
)

if not can_infer:
    print(
        "WARNING: Skipping inference because either MK code root or checkpoints are missing."
    )
else:
    for fold in range(5):
        ckpt = CKPT_ROOT / f"fold{fold}_levit.ckpt"
        if ckpt.exists():
            run_one_test(
                ckpt_path=str(ckpt),
                experiment="conv1d_tfm2d",
                out_csv=f"/kaggle/working/submission_fold{fold}_v0.csv",
            )
        else:
            print(f"WARNING: missing ckpt, skipping: {ckpt}")

    for fold in range(5):
        ckpt = CKPT_ROOT / f"fold{fold}_pseudo_resv2.ckpt"
        if ckpt.exists():
            run_one_test(
                ckpt_path=str(ckpt),
                experiment="conv1d_resv2",
                out_csv=f"/kaggle/working/submission_fold{fold}_v2.csv",
            )
        else:
            print(f"WARNING: missing ckpt, skipping: {ckpt}")

print("\nGenerated submission files (if any):")
run_cmd("ls -la /kaggle/working | grep submission_fold | head -n 200")



## === cell 5
import pandas as pd
import numpy as np

try:
    from src.settings import TARGET_COLS  # type: ignore
except Exception as e:
    print(
        "Warning: could not import src.settings.TARGET_COLS; using fallback. Error:",
        repr(e),
    )
    TARGET_COLS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]


def _load_and_align_pred(path: str, base_eeg_ids: pd.Series) -> np.ndarray:
    """Load a submission-like csv and align rows to base_eeg_ids."""
    df = pd.read_csv(path)
    if "eeg_id" not in df.columns:
        raise ValueError(f"{path} missing eeg_id column")

    missing = [c for c in TARGET_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing target columns: {missing}")

    df = df[["eeg_id"] + TARGET_COLS].copy()
    df = df.set_index("eeg_id").reindex(base_eeg_ids.values)

    if df.isna().any().any():
        na_rows = int(df.isna().any(axis=1).sum())
        raise ValueError(
            f"{path} has NaNs after reindexing; likely missing eeg_ids. NaN rows: {na_rows}"
        )

    arr = df[TARGET_COLS].to_numpy(dtype=np.float64)
    return arr


def merge_preds(
    folds=(0, 1, 2, 3, 4),
    versions=("v0", "v2"),
    weights=(0.3, 0.7),
    sample_path=f"{DATA_PATH}/sample_submission.csv",
):
    sol = pd.read_csv(sample_path)
    base_eeg_ids = sol["eeg_id"].copy()
    preds_sum = np.zeros((len(sol), len(TARGET_COLS)), dtype=np.float64)

    used = 0
    for fold in folds:
        for weight, version in zip(weights, versions):
            path = f"/kaggle/working/submission_fold{fold}_{version}.csv"
            if not Path(path).exists():
                continue
            arr = _load_and_align_pred(path, base_eeg_ids)
            preds_sum += arr * float(weight)
            used += 1

    if used == 0:
        print("Warning: no fold submission files found; using uniform probabilities.")
        uniform = np.full(
            (len(sol), len(TARGET_COLS)), 1.0 / len(TARGET_COLS), dtype=np.float64
        )
        sol[TARGET_COLS] = uniform
        return sol

    preds_sum = np.clip(preds_sum, 1e-15, None)
    preds_sum = preds_sum / preds_sum.sum(axis=1, keepdims=True)
    sol[TARGET_COLS] = preds_sum
    print(f"Ensembled from {used} available fold/version files.")
    return sol




## === cell 6
sol = merge_preds(folds=[0, 1, 2, 3, 4], versions=["v0", "v2"], weights=[0.3, 0.7])

row_sums = sol[TARGET_COLS].sum(axis=1).values
print("Row sum min/max:", float(row_sums.min()), float(row_sums.max()))
assert np.all(
    np.isfinite(sol[TARGET_COLS].values)
), "Non-finite probabilities in submission"
assert np.allclose(row_sums, 1.0, atol=1e-6), "Probabilities do not sum to 1"
assert (
    list(sol.columns) == ["eeg_id"] + TARGET_COLS
), f"Unexpected columns: {sol.columns.tolist()}"



## === cell 7
sub_path = "/kaggle/working/submission.csv"
sol.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape:", sol.shape)
print(pd.read_csv(sub_path).head())



## === cell 8
sol.head()
