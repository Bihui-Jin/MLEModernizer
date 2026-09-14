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
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) is consistent with often falling back to uniform predictions or producing a badly-calibrated ensemble when some fold/version files are missing. I keep your MK conversion + `python -m test` inference flow unchanged, but make two minimal scoring-relevant fixes: (1) correct `PYTHONPATH` to point at `CODES_DIR/src` so imports resolve reliably during conversion/inference, and (2) fix the ensemble averaging so it properly averages across the actually-found prediction files (instead of summing unnormalized weights, which can distort probabilities when not all folds exist). Finally, I keep strict probability normalization and submission schema checks so the output remains valid.'
- What this solution (achieved 1.40995) has done: 'Your current score (1.40995, lower-is-better) strongly suggests you’re still often submitting the uniform fallback or a poorly-formed ensemble when some fold outputs are missing/misaligned. I keep your MK convert→`python -m test` inference flow and model choices unchanged, but (1) fix `sys.path`/`PYTHONPATH` to point to `CODES_DIR/src` (your code currently appends `CODES_DIR`, which can break imports in some runs), (2) make ensembling discover and use *all* produced `submission_fold*.csv` files (not only hardcoded `v0/v2` pairs) and weight-average by model-version reliably, and (3) add a tiny “probability smoothing” mix with uniform (1–2%) to reduce extreme probabilities that can hurt KL divergence while preserving ranking/semantics. These are minimal, scoring-relevant changes that should move you substantially toward the ~0.319 target without altering architecture/training. The script still always writes a valid `/kaggle/working/submission.csv` with correct columns and row sums of 1.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.3193), and the most likely reason is that your inference outputs are not being found/used correctly, causing a fallback to uniform-ish probabilities or a badly averaged ensemble. I make two minimal scoring-relevant fixes while preserving the same convert→`python -m test` inference and ensembling approach: (1) ensure the produced prediction files are actually discovered even if the test script writes `submission*.csv` into a subdirectory, and (2) ensemble by averaging across all found fold files per version with proper normalization (so weight isn’t inadvertently multiplied by number of folds found). Finally, I keep strict probability normalization and schema checks so the submission is always valid.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.3193), and the most likely cause is that the ensemble is still often dominated by missing/misaligned inference outputs (or overly-smoothed toward uniform), which produces near-uniform or poorly calibrated probabilities. I keep your convert→`python -m test` inference flow and model choices unchanged, but make the output capture deterministic by forcing each run to write into a unique per-fold/per-version directory and then reading that exact `submission.csv` (instead of “latest submission*.csv”, which can grab the wrong file). Then I reduce the uniform-mix smoothing to 0 (it can actively hurt KL when predictions are already reasonable), and I average versions with correct normalization as you already do. These are minimal changes aimed at ensuring you actually submit the intended model predictions and avoid accidental uniform fallbacks, which should move the score substantially toward 0.319.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.3193), and with your setup that most commonly happens when inference outputs are missing or not being picked up correctly, causing the ensemble to be weak/near-uniform. I keep the convert→`python -m test` inference and the same two-version ensemble, but (1) fix the inference command so `+model.test_output_dir` is passed as a proper Hydra override (no extra `model.test_output_dir=` token), and (2) broaden file discovery to also pick up submissions written inside the per-run output directories (not only top-level `/kaggle/working/submission_fold*_v*.csv`). Finally, I keep strict normalization and add a tiny per-row epsilon floor before renormalization to avoid overconfident zeros which can inflate KL, without changing model semantics.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far from the target (0.3193), and with this pipeline that most often means the script is not actually ensembling the intended fold outputs (e.g., it’s skipping files written into `mk_out_*/submission.csv` because they don’t encode `_v0/_v2` in the filename). I make a minimal change so that after inference we also write a “versioned” copy of each produced `submission.csv` (e.g., `submission_fold0_v0.csv` / `submission_fold0_v2.csv`) while keeping the same inference command and checkpoints. Then ensembling reliably pick up the correct files and apply your intended `v0/v2` weights instead of falling back to uniform or near-uniform behavior. I also keep the strict normalization and add a tiny epsilon floor (already present) to avoid zeros that can explode KL, without changing core model logic.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995, lower-is-better) is far above the target, and with this pipeline the most likely reason is that the ensemble is accidentally using only a subset of the real fold/version predictions (or none), because `mk_out_*/submission.csv` files don’t encode `_v0/_v2` so they get skipped and you effectively drift toward a weak/unintended prediction. I make a minimal, scoring-relevant change: after each inference run, also write a versioned copy `submission_fold{fold}_{version}.csv` and (optionally) a `latest_submission_{version}.csv`, so the existing ensembling logic can reliably find and weight v0 vs v2 predictions. I also add a small robustness tweak in discovery to pick up `submission_fold{fold}_v?.csv` and ignore unversioned files for weighting (no change to model/inference). This keeps your core convert→`python -m test` inference and your (v0=0.3, v2=0.7) averaging unchanged, but should move the score substantially toward the target by ensuring you actually submit the intended ensemble.'
- What this solution (achieved 1.40995) has done: 'Your KL score (1.40995, lower-is-better) is far from the target (0.3193), and with this pipeline the most likely scoring issue is that your ensembling is silently skipping the real per-run predictions (`mk_out_*/submission.csv`) because they don’t encode `_v0/_v2` in the filename, so you end up using few/none of the intended model outputs (drifting toward a weak/uniform submission). I make a minimal, scoring-relevant fix: after each inference run, copy the produced `mk_out_*/submission.csv` into an explicitly versioned filename (`submission_fold{fold}_v0.csv` / `submission_fold{fold}_v2.csv`) so your existing ensembling logic reliably picks them up with the correct weights. I also slightly broaden version parsing so it can infer `v0/v2` from either the filename or the parent folder name (for robustness), without changing model/inference logic. Everything else (convert → `python -m test` command, weights 0.3/0.7, normalization, schema checks) remains the same to preserve core semantics while moving the score toward the target.'
- What this solution (achieved 1.40995) has done: 'You’re far worse than the target (KL 1.40995 vs 0.3193; lower is better), which strongly indicates you’re still not actually using the real model outputs for most/all rows (often because `python -m test` doesn’t run successfully under missing deps, leaving you with a near-uniform fallback). The minimal score-improving fix is to (1) ensure the MK dependencies are installed in this environment even when offline wheels aren’t provided, and (2) make the inference stage fail-fast (instead of silently continuing to the ensemble fallback) so you don’t accidentally submit uniform predictions. I keep your convert→`python -m test` inference and the same v0/v2 weighted ensemble unchanged, but I add a deterministic “dependency bootstrap” (online pip if possible; otherwise wheels) and explicit verification that at least one fold/version prediction file exists before ensembling. This should move the score substantially toward the target by ensuring the intended model predictions are actually generated and used.'
- What this solution (achieved 1.40995) has done: 'Your current KL (1.40995; lower is better) is far from the target (0.3193), which strongly suggests the ensemble being submitted is still not the intended model outputs (often due to inference failing, missing deps, or picking up the wrong CSVs). I make two minimal, scoring-relevant changes: (1) ensure the MK inference environment can import required deps by adding a fail-fast dependency check (and installing `torch` only if it’s missing, since without it the MK `test` module cannot run), and (2) make prediction discovery/versioning deterministic by copying each run’s `submission.csv` to an explicit `submission_fold{fold}_v?.csv` and restricting ensembling to those versioned fold files only (so we don’t accidentally average unrelated/unversioned files). These changes keep your convert → `python -m test` flow and the same v0/v2 weighted averaging (0.3/0.7) intact, but greatly reduce the chance of silently falling back to weak/uniform predictions. The script still always write a valid `/kaggle/working/submission.csv` with correct columns and normalized row sums.'

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

if CODES_DIR is not None:
    codes_src = str(Path(CODES_DIR) / "src")
    if Path(codes_src).is_dir() and codes_src not in sys.path:
        sys.path.insert(0, codes_src)


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


def _ensure_required_imports_or_install():
    """
    Change (score-relevant): fail-fast/auto-install critical runtime deps used by MK test-time code.
    If torch/lightning/hydra are missing, inference won't run and we'd drift toward uniform predictions (high KL).
    We only install missing packages to keep changes minimal and avoid unintended score shifts.
    """
    missing = []
    try:
        import torch  # noqa: F401
    except Exception:
        missing.append("torch")
    try:
        import lightning  # noqa: F401
    except Exception:
        missing.append("lightning==2.2.1")
    try:
        import hydra  # noqa: F401
        import omegaconf  # noqa: F401
        import antlr4  # noqa: F401
    except Exception:
        for p in [
            "antlr4-python3-runtime==4.9.2",
            "omegaconf==2.3.0",
            "hydra-core==1.3.2",
        ]:
            missing.append(p)

    seen = set()
    missing = [x for x in missing if not (x in seen or seen.add(x))]

    if not missing:
        print(
            "All required imports already available (torch/lightning/hydra/omegaconf/antlr)."
        )
        return

    print("Missing deps detected, attempting install (only missing):", missing)

    try:
        run_cmd("pip -q install " + " ".join(missing))
        return
    except Exception as e:
        print(
            "WARNING: online pip install failed; will try offline wheels if available. Error:",
            repr(e),
        )

    wheel_map = {
        "antlr4-python3-runtime==4.9.2": "antlr4_python3_runtime-4.9.2-py3-none-any.whl",
        "omegaconf==2.3.0": "omegaconf-2.3.0-py3-none-any.whl",
        "hydra-core==1.3.2": "hydra_core-1.3.2-py3-none-any.whl",
        "lightning==2.2.1": "lightning-2.2.1-py3-none-any.whl",
    }

    if REQ_DIR.exists():
        for pkg in missing:
            whl = wheel_map.get(pkg)
            if whl is None:
                print(
                    f"WARNING: no offline wheel mapping for {pkg}; skipping offline install attempt."
                )
                continue
            p = REQ_DIR / whl
            if p.exists():
                run_cmd(f"pip -q install {p} --no-index --no-deps")
            else:
                print(f"WARNING: wheel not found, skipping: {p}")
    else:
        print(
            f"WARNING: requirements dir not found, skipping offline installs: {REQ_DIR}"
        )

    try:
        import torch  # noqa: F401
        import lightning  # noqa: F401
        import hydra  # noqa: F401
        import omegaconf  # noqa: F401
        import antlr4  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Required dependencies still missing after install attempts; inference would fail and hurt KL. "
            f"Last import error: {repr(e)}"
        )


_ensure_required_imports_or_install()



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
    codes_src = str(Path(CODES_DIR) / "src")
    extra_env = {
        "PYTHONPATH": codes_src
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


def run_one_test(
    ckpt_path: str, experiment: str, out_csv: str, run_tag: str, version_tag: str
) -> None:
    if CODES_DIR is None or not (Path(CODES_DIR) / "src").is_dir():
        raise FileNotFoundError(
            "MK code root with src/ was not detected; cannot run inference."
        )

    codes_src = str(Path(CODES_DIR) / "src")
    extra_env = {
        "PYTHONPATH": codes_src
        + (":" + os.environ["PYTHONPATH"] if "PYTHONPATH" in os.environ else "")
    }

    run_out_dir = Path(OUT_PATH) / f"mk_out_{run_tag}"
    run_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = (
        f"python -m test "
        f"paths.data_dir={DATA_PATH} "
        f"data.test_eegs_dir={OUT_PATH} "
        f"ckpt_path={ckpt_path} "
        f"hydra=test "
        f"+model.test_output_dir={run_out_dir} "
        f"experiment={experiment} "
        f"+model.net.pretrained=False"
    )
    run_cmd(cmd, cwd=CODES_DIR, extra_env=extra_env)

    produced = run_out_dir / "submission.csv"
    if not produced.exists():
        run_cmd(f"ls -la {run_out_dir} | head -n 200")
        raise FileNotFoundError(
            f"Expected {produced} to be created by inference but it was not found."
        )

    target = Path(out_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(produced, target)
    print(f"Copied {produced} -> {target}")

    latest = Path(OUT_PATH) / f"latest_submission_{version_tag}.csv"
    shutil.copy2(produced, latest)
    print(f"Copied {produced} -> {latest}")


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
                run_tag=f"fold{fold}_v0",
                version_tag="v0",
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
                run_tag=f"fold{fold}_v2",
                version_tag="v2",
            )
        else:
            print(f"WARNING: missing ckpt, skipping: {ckpt}")

print("\nGenerated submission files (if any):")
run_cmd(
    "ls -la /kaggle/working | grep -E 'submission_fold|mk_out_|latest_submission' | head -n 200"
)



## === cell 5
import pandas as pd
import numpy as np
import re

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


def _discover_fold_files() -> list[Path]:
    """
    Change (score-relevant): only ensemble explicitly versioned fold files.
    This prevents accidentally picking up unrelated mk_out submissions (unversioned) which can distort averaging,
    and ensures we use the intended v0/v2 weighted ensemble.
    """
    wdir = Path("/kaggle/working")
    files = sorted(set(wdir.glob("submission_fold*_v*.csv")))
    return files


def _parse_version_from_name(p: Path) -> str | None:
    s = p.as_posix()
    m = re.search(r"_v(\d+)(?:\.csv)?$", s)
    if m:
        return f"v{m.group(1)}"
    return None


def merge_preds(
    weights_by_version: dict[str, float] | None = None,
    sample_path: str = f"{DATA_PATH}/sample_submission.csv",
    uniform_mix: float = 0.0,
):
    if weights_by_version is None:
        weights_by_version = {"v0": 0.3, "v2": 0.7}

    sol = pd.read_csv(sample_path)
    base_eeg_ids = sol["eeg_id"].copy()

    files = _discover_fold_files()

    if len(files) == 0:
        if can_infer:
            raise FileNotFoundError(
                "Inference was enabled (MK code + checkpoints found) but no versioned fold prediction files were produced. "
                "Refusing to fall back to uniform predictions because that would severely hurt KL."
            )
        print("Warning: no fold submission files found; using uniform probabilities.")
        uniform = np.full(
            (len(sol), len(TARGET_COLS)), 1.0 / len(TARGET_COLS), dtype=np.float64
        )
        sol[TARGET_COLS] = uniform
        return sol

    preds_sum = np.zeros((len(sol), len(TARGET_COLS)), dtype=np.float64)
    version_used: dict[str, list[np.ndarray]] = {}
    skipped = 0

    for p in files:
        v = _parse_version_from_name(p)
        if v is None or v not in weights_by_version:
            skipped += 1
            continue
        arr = _load_and_align_pred(str(p), base_eeg_ids)
        version_used.setdefault(v, [])
        version_used[v].append(arr)

    if not version_used:
        if can_infer:
            raise ValueError(
                f"Discovered {len(files)} versioned fold files but none matched weights_by_version={weights_by_version}. "
                "This would cause an unintended fallback/unusable ensemble."
            )
        print(
            f"Warning: discovered {len(files)} files but none matched weights_by_version={weights_by_version}; "
            "using uniform probabilities."
        )
        uniform = np.full(
            (len(sol), len(TARGET_COLS)), 1.0 / len(TARGET_COLS), dtype=np.float64
        )
        sol[TARGET_COLS] = uniform
        return sol

    weight_sum = 0.0
    used_files = 0
    for v, arr_list in version_used.items():
        v_mean = np.mean(np.stack(arr_list, axis=0), axis=0)
        w = float(weights_by_version[v])
        preds_sum += v_mean * w
        weight_sum += w
        used_files += len(arr_list)

    preds = preds_sum / max(weight_sum, 1e-12)

    if uniform_mix and uniform_mix > 0:
        uni = np.full_like(preds, 1.0 / len(TARGET_COLS))
        preds = (1.0 - float(uniform_mix)) * preds + float(uniform_mix) * uni

    eps = 1e-6
    preds = np.clip(preds, eps, None)
    preds = preds / preds.sum(axis=1, keepdims=True)
    sol[TARGET_COLS] = preds

    print(
        f"Ensembled from {used_files} files across versions={sorted(version_used.keys())} (skipped={skipped}). "
        f"Total version-weight used={weight_sum:.3f}. uniform_mix={uniform_mix}"
    )
    return sol




## === cell 6
sol = merge_preds(weights_by_version={"v0": 0.3, "v2": 0.7}, uniform_mix=0.0)

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
