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

0.3407056092205677

# 6. Current score

0.81712

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I make the script robust to missing private inputs by (1) removing the failing offline `pip install` steps and instead using a safe, optional import path, and (2) making the parquet→npy conversion and multi-fold inference conditional on the required code/checkpoints actually existing. If the pretrained pipeline can’t run, I still produce a valid `submission.csv` by falling back to a calibrated label-prior baseline derived from `train.csv` (score won’t be great, but it be valid and should run end-to-end). I also fix the cell numbering to start at 1 (your current “cell 0” breaks the required format) and ensure the final submission has the exact required columns, row count, and per-row probabilities summing to 1.'
- What this solution (achieved 0.76634) has done: 'I fix the crash in the fallback path by removing the incorrect `validate="one_to_one"` merge: `sample_submission.csv` can contain duplicate `eeg_id`, so we must map `patient_id` from `test.csv` via a de-duplicated lookup to keep row counts intact. I also renumber cells to start at 1 (your current “cell 0” violates the required format) and keep the rest of the logic identical. Finally, I make the fallback prior computation deterministic and ensure the submission is always written even when inference artifacts are missing, with strict probability normalization so the file passes Kaggle checks.'
- What this solution (achieved 0.78445) has done: 'Your current fallback is a patient-level label-prior, which is a reasonable baseline but can be improved toward the target by (1) computing priors at the more specific `eeg_id` level when possible (train has many rows per `eeg_id`, and test has unique `eeg_id`), and (2) applying a small, deterministic shrinkage mix between the more specific prior and the global prior to reduce KL penalties from overconfident priors. This keeps the same “prior-from-train-votes” core logic (no model/feature changes), but makes the fallback predictions better calibrated for the KL metric. I also fix the merge/alignment in the inference path to preserve row order exactly as `sample_submission.csv` (avoid accidental row drops/NaNs that can inflate KL). The changes are minimal and still guarantee a valid `submission.csv` with per-row probabilities summing to 1.'
- What this solution (achieved 0.81793) has done: 'Your current score (0.78445, lower-is-better) is far from the target (0.3407), so we should improve the fallback baseline while preserving the same “prior-from-train-votes” core logic. I keep your exact inference path unchanged, and only adjust the fallback calibration by (1) switching from linear-probability mixing to log-space (geometric) mixing, which is typically better for KL, and (2) making the shrinkage strength depend deterministically on how many votes support the specific prior (patient/eeg), so we rely more on global prior when data is sparse. I also fix a subtle issue where the “eeg_id prior” almost never applies in test (test eeg_ids are unseen), by explicitly preferring patient priors (when available) and treating eeg-level priors as train-only. These are minimal changes, still deterministic, still produce a valid submission with rows summing to 1.'
- What this solution (achieved 0.84316) has done: 'Your current score is worse than the target (lower is better), so we should improve the fallback predictions while keeping the same “priors-from-train-votes” core logic (no model/feature/training changes). The biggest likely issue is that the fallback currently keys only on `patient_id` (since test `eeg_id` are unseen), and it uses a fixed-strength Dirichlet smoothing plus a geometric mix that can still be miscalibrated for KL. I make two minimal, metric-aligned calibration tweaks: (1) use a data-driven global prior that matches the *per-row normalized vote distribution* (not raw summed votes), and (2) tune the patient prior smoothing and mixing weight deterministically based on patient vote mass, with slightly stronger shrinkage to global for low-mass patients. This preserves identical semantics (prior baseline) but should reduce overconfident wrong distributions and move KL down toward your target.'
- What this solution (achieved 0.7973) has done: 'Your current score (0.84316, lower-is-better) is still far above the target (0.3407), so we should improve the fallback (prior-based) predictions without touching the core “train-vote-prior → test probability” logic. The biggest safe win for KL here is better calibration: use a hierarchical prior that backs off from patient→global using an evidence-based weight (posterior mean under a Dirichlet-multinomial), rather than a fixed geometric mix that can stay overconfident. I also compute the global prior from total vote mass (Dirichlet posterior mean) to better match the competition’s label generation, and add a small deterministic “temperature-to-uniform” shrinkage for low-evidence patients to reduce KL penalties from spiky wrong distributions. Inference path and submission formatting are kept intact; changes only affect the fallback branch when private checkpoints aren’t available (your current situation).'
- What this solution (achieved 0.78398) has done: 'Your current score (0.7973, lower-is-better) is far above the target (0.3407), so we should improve the fallback prior-based predictions (the path you’re effectively using) without changing any model/training logic. The biggest safe gain for KL is to (1) compute the global prior in a way that matches the label-generation process (average of per-row vote distributions, not raw summed votes), and (2) replace the ad-hoc uniform shrink with a proper hierarchical Dirichlet-multinomial posterior mean that backs off patient→global with evidence-based strength. This keeps the same “train votes → priors → test probabilities” core logic, but reduces overconfident wrong distributions that are heavily penalized by KL. All submission formatting, row order, and per-row normalization are preserved.'
- What this solution (achieved 0.96669) has done: 'You’re far above the target KL (lower is better), and the inference path likely isn’t being used, so the only safe way to move toward the target is to improve the fallback prior-based probabilities while preserving the same “train votes → hierarchical priors → test probabilities” logic. I keep your model/conversion/inference code unchanged, and only refine the fallback calibration by (1) computing the global prior from total vote mass (closer to how labels are generated) and (2) using patient-level empirical Bayes shrinkage with a stronger, evidence-weighted backoff plus a small uniform mix to avoid overconfident spiky distributions (which KL heavily penalizes). I also fix the cell numbering to start at 1 (your provided script starts at cell 0) while keeping paths and outputs identical. The result still deterministically writes a valid `submission.csv` with the exact required columns and per-row probabilities summing to 1.'
- What this solution (achieved 0.85121) has done: 'Your current KL (0.96669, lower-is-better) is still far from the target (0.3407), and given the environment constraints you’re effectively using the fallback branch, so the safest way to move toward the target is to improve calibration of that prior-based fallback without changing the overall “train votes → hierarchical prior → test probabilities” logic. I (1) compute the global prior as the mean of per-row vote distributions (closer to how the targets are formed) instead of from raw summed votes, and (2) replace the linear backoff with a Dirichlet posterior mean per patient (and per eeg_id as a secondary fallback) using the global prior as the base measure—this reduces overconfident spiky predictions that are heavily penalized by KL. I also fix the cell numbering to start at 1 (your current script starts at cell 0) while keeping all paths and the inference branch intact. The script still always write a valid `/kaggle/working/submission.csv` with correct columns, row count, and per-row probabilities summing to 1.'
- What this solution (achieved 0.79517) has done: 'Your current score is much worse than the target (KL: 0.85121 vs 0.3407; lower is better), and given the environment it’s very likely the fallback prior branch is what’s being used. I keep the same core “train votes → hierarchical prior → test probabilities” logic, but make two minimal, metric-aligned calibration tweaks: (1) compute patient/eeg posteriors from the **mean per-row vote distribution** (not summed counts) so the posterior matches the label-generation process better, and (2) replace the fixed uniform-mix with a **deterministic evidence-based backoff** toward global (and a tiny uniform only when evidence is extremely low), which reduces overconfident wrong distributions that KL heavily penalizes. I also ensure inference-branch alignment uses the sample_submission row order without introducing NaNs, while preserving all paths and always writing a valid `submission.csv`.'
- What this solution (achieved 0.81712) has done: 'Your current score is far above the target (KL is lower-better), and in this environment you’re almost certainly using the fallback prior-based branch, so the safest path is to improve *calibration* of that fallback without changing the overall “train votes → patient prior → test probabilities” logic. I keep your inference branch intact, but in the fallback I (1) compute patient posteriors from **summed vote counts** (not sums of per-row-normalized distributions) so the Bayesian update matches the true vote-generation process, and (2) replace the per-row loop + ad-hoc uniform mix with a fully vectorized **Dirichlet posterior mean** plus a small deterministic temperature smoothing, which typically reduces overconfident errors that KL punishes. I also fix the cell numbering to start at 1 (your provided script starts at cell 0) while keeping all paths and the submission schema identical. These are minimal, metric-aligned changes intended to move KL downward toward your target, not to maximize performance.'

# 9. Code solution

## === cell 0
import os
import sys
import numpy as np
import pandas as pd

CODES_PATH = "/kaggle/input/hms-mk-codes"
if os.path.isdir(CODES_PATH) and CODES_PATH not in sys.path:
    sys.path.append(CODES_PATH)

INNER_PATH = os.path.join(CODES_PATH, "src")
if os.path.isdir(INNER_PATH) and INNER_PATH not in sys.path:
    sys.path.append(INNER_PATH)



## === cell 1
import subprocess


def _run(cmd: str, check: bool = True):
    print(cmd)
    return subprocess.run(cmd, shell=True, check=check, text=True, capture_output=False)




## === cell 2
REQ_PATH = "/kaggle/input/requirements-mk"
wheels = [
    "antlr4_python3_runtime-4.9.2-py3-none-any.whl",
    "omegaconf-2.3.0-py3-none-any.whl",
    "hydra_core-1.3.2-py3-none-any.whl",
    "lightning-2.2.1-py3-none-any.whl",
]

if os.path.isdir(REQ_PATH):
    for whl in wheels:
        fp = os.path.join(REQ_PATH, whl)
        if os.path.exists(fp):
            _run(f"pip install {fp} --no-index --no-deps", check=False)
        else:
            print(f"Wheel not found, skipping: {fp}")
else:
    print(f"Requirements directory not found, skipping pip installs: {REQ_PATH}")



## === cell 3
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"
os.makedirs(OUT_PATH, exist_ok=True)



## === cell 4
convert_ok = False
if os.path.isdir(CODES_PATH):
    cmd = f"cd {CODES_PATH} && python -m src.convert_parquet_to_npy --data_dir={DATA_PATH} --out_dir={OUT_PATH}"
    r = _run(cmd, check=False)
    convert_ok = r.returncode == 0
    print(f"convert_parquet_to_npy returncode={r.returncode}")
else:
    print(f"CODES_PATH not available, skipping conversion: {CODES_PATH}")



## === cell 5
checkpoints = [
    "epoch_012_val_loss_0.5037.ckpt",
    "epoch_013_val_loss_0.5103.ckpt",
    "epoch_014_val_loss_0.4728.ckpt",
    "epoch_014_val_loss_0.4944.ckpt",
    "epoch_014_val_loss_0.5053.ckpt",
]



## === cell 6
_run("ls /kaggle/input/hms-mk-data || true", check=False)



## === cell 7
mk_data_path = "/kaggle/input/hms-mk-data"
fold_sub_paths = []
inference_ok = False

if os.path.isdir(CODES_PATH) and os.path.isdir(mk_data_path):
    for fold in range(5):
        ckpt = os.path.join(mk_data_path, f"fold{fold}_pseudo_log.ckpt")
        if not os.path.exists(ckpt):
            print(f"Missing checkpoint, skipping fold {fold}: {ckpt}")
            continue
        cmd = (
            f"cd {CODES_PATH} && python -m test "
            f"paths.data_dir={DATA_PATH} data.test_eegs_dir={OUT_PATH} "
            f"ckpt_path={ckpt} hydra=test +model.test_output_dir={OUT_PATH} "
            f"experiment=conv1d_pseudo +model.net.pretrained=False"
        )
        r = _run(cmd, check=False)
        if r.returncode != 0:
            print(
                f"Fold {fold} inference failed (returncode={r.returncode}); will fall back if needed."
            )
            continue

        src_fp = "/kaggle/working/submission.csv"
        dst_fp = f"/kaggle/working/submission_fold{fold}.csv"
        if os.path.exists(src_fp):
            _run(f"mv {src_fp} {dst_fp}", check=False)
            if os.path.exists(dst_fp):
                fold_sub_paths.append(dst_fp)

    inference_ok = len(fold_sub_paths) > 0
else:
    print(
        f"Private code or checkpoints not available; skipping model inference. CODES_PATH={CODES_PATH}, mk_data_path={mk_data_path}"
    )

print(f"inference_ok={inference_ok}, found_fold_submissions={len(fold_sub_paths)}")



## === cell 8
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


def merge_preds(fold_paths):
    preds = []
    sol0 = None
    for fp in fold_paths:
        sol = pd.read_csv(fp)
        if sol0 is None:
            sol0 = sol.copy()
        missing = [c for c in TARGET_COLS if c not in sol.columns]
        if missing:
            raise ValueError(f"Fold submission {fp} missing columns: {missing}")
        preds.append(sol[TARGET_COLS].to_numpy(dtype=np.float64))

    preds = np.mean(np.stack(preds, axis=0), axis=0)

    eps = 1e-12
    preds = np.clip(preds, eps, 1.0)
    preds = preds / preds.sum(axis=1, keepdims=True)

    sol0[TARGET_COLS] = preds
    return sol0




## === cell 9
sample_fp = f"{DATA_PATH}/sample_submission.csv"
sample_sub = pd.read_csv(sample_fp)

if inference_ok:
    sol_pred = merge_preds(fold_sub_paths)
    if "eeg_id" not in sol_pred.columns:
        raise ValueError("Merged predictions missing `eeg_id` column.")

    sol = sample_sub[["eeg_id"]].copy()
    pred_lookup = sol_pred.drop_duplicates(subset=["eeg_id"], keep="first").set_index(
        "eeg_id"
    )[TARGET_COLS]
    for c in TARGET_COLS:
        sol[c] = sol["eeg_id"].map(pred_lookup[c])

    vals = sol[TARGET_COLS].to_numpy(dtype=np.float64)
    missing_rows = ~np.isfinite(vals).all(axis=1)
    if missing_rows.any():
        fallback = np.full(
            (len(TARGET_COLS),), 1.0 / len(TARGET_COLS), dtype=np.float64
        )
        vals[missing_rows] = fallback[None, :]
        sol[TARGET_COLS] = vals

else:
    train_fp = f"{DATA_PATH}/train.csv"
    test_fp = f"{DATA_PATH}/test.csv"

    train = pd.read_csv(train_fp, usecols=["eeg_id", "patient_id"] + TARGET_COLS)
    test = pd.read_csv(test_fp, usecols=["eeg_id", "patient_id"])

    eps = 1e-12
    K = len(TARGET_COLS)

    votes = train[TARGET_COLS].to_numpy(dtype=np.float64)
    vote_mass = votes.sum(axis=1)
    valid = vote_mass > 0

    if valid.any():
        total_counts = votes[valid].sum(axis=0)
        prior_global = total_counts / max(total_counts.sum(), eps)
    else:
        prior_global = np.full((K,), 1.0 / K, dtype=np.float64)

    prior_global = np.clip(prior_global, eps, 1.0)
    prior_global = prior_global / prior_global.sum()

    grp_pat = train.loc[valid].groupby("patient_id", sort=False)[TARGET_COLS].sum()
    pat_ids = grp_pat.index.to_numpy()
    counts_pat = grp_pat.to_numpy(dtype=np.float64)
    mass_pat = counts_pat.sum(axis=1)

    alpha0_pat = (
        48.0  # slightly stronger smoothing to reduce KL from overconfident priors
    )
    alpha_pat = alpha0_pat * prior_global[None, :]
    post_pat = (counts_pat + alpha_pat) / (mass_pat[:, None] + alpha0_pat)
    post_pat = np.clip(post_pat, eps, 1.0)
    post_pat = post_pat / post_pat.sum(axis=1, keepdims=True)
    patient_to_post = {int(pid): post_pat[i] for i, pid in enumerate(pat_ids)}
    patient_mass_map = {
        int(pat_ids[i]): float(mass_pat[i]) for i in range(len(pat_ids))
    }

    grp_eeg = train.loc[valid].groupby("eeg_id", sort=False)[TARGET_COLS].sum()
    eeg_ids = grp_eeg.index.to_numpy()
    counts_eeg = grp_eeg.to_numpy(dtype=np.float64)
    mass_eeg = counts_eeg.sum(axis=1)
    alpha0_eeg = 48.0
    alpha_eeg = alpha0_eeg * prior_global[None, :]
    post_eeg = (counts_eeg + alpha_eeg) / (mass_eeg[:, None] + alpha0_eeg)
    post_eeg = np.clip(post_eeg, eps, 1.0)
    post_eeg = post_eeg / post_eeg.sum(axis=1, keepdims=True)
    eeg_to_post = {int(eid): post_eeg[i] for i, eid in enumerate(eeg_ids)}
    eeg_mass_map = {int(eeg_ids[i]): float(mass_eeg[i]) for i in range(len(eeg_ids))}

    test_lookup = test.drop_duplicates(subset=["eeg_id"], keep="first").set_index(
        "eeg_id"
    )["patient_id"]

    sol = sample_sub.copy()
    sol["patient_id"] = sol["eeg_id"].map(test_lookup)

    eegs = sol["eeg_id"].to_numpy()
    pids = sol["patient_id"].to_numpy()
    preds = np.zeros((len(sol), K), dtype=np.float64)

    uniform = np.full((K,), 1.0 / K, dtype=np.float64)

    for i, (eid, pid) in enumerate(zip(eegs, pids)):
        p_group = None
        m = 0.0

        if not pd.isna(pid):
            pid_int = int(pid)
            p_group = patient_to_post.get(pid_int, None)
            if p_group is not None:
                m = patient_mass_map.get(pid_int, 0.0)

        if p_group is None and not pd.isna(eid):
            eid_int = int(eid)
            p_group = eeg_to_post.get(eid_int, None)
            if p_group is not None:
                m = eeg_mass_map.get(eid_int, 0.0)

        if p_group is None:
            p = prior_global
        else:
            tau = 12.0
            w = m / (m + tau) if m > 0 else 0.0
            p = (1.0 - w) * prior_global + w * p_group

        lam_u = 0.002 if m < 5.0 else 0.0
        if lam_u > 0:
            p = (1.0 - lam_u) * p + lam_u * uniform

        p = np.clip(p, eps, 1.0)
        preds[i] = p / p.sum()

    for j, c in enumerate(TARGET_COLS):
        sol[c] = preds[:, j]

    sol = sol[["eeg_id"] + TARGET_COLS]

vals = sol[TARGET_COLS].to_numpy(dtype=np.float64)
nan_mask = ~np.isfinite(vals)
if nan_mask.any():
    fallback = np.full((len(TARGET_COLS),), 1.0 / len(TARGET_COLS), dtype=np.float64)
    vals[nan_mask] = np.take(fallback, np.where(nan_mask)[1])

vals = np.clip(vals, 1e-12, 1.0)
vals = vals / vals.sum(axis=1, keepdims=True)
sol[TARGET_COLS] = vals

out_fp = "/kaggle/working/submission.csv"
sol.to_csv(out_fp, index=False)
print(f"Wrote: {out_fp}  shape={sol.shape}")
print(sol.head())



## === cell 10
assert os.path.exists("/kaggle/working/submission.csv")
sub = pd.read_csv("/kaggle/working/submission.csv")
assert list(sub.columns) == ["eeg_id"] + TARGET_COLS
assert len(sub) == len(sample_sub)
row_sums = sub[TARGET_COLS].sum(axis=1).to_numpy()
assert np.all(np.isfinite(row_sums))
assert np.max(np.abs(row_sums - 1.0)) < 1e-6
print("Submission sanity checks passed.")
