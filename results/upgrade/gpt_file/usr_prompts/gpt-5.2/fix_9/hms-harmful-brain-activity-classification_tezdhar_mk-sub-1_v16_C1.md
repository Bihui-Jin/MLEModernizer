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

0.3626664036379633

# 6. Current score

0.7647

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.41937) has done: 'I remove the hard dependency on missing private Kaggle inputs (`/kaggle/input/hms-mk-codes`, `/kaggle/input/requirements-mk`, `/kaggle/input/hms-mk-data`) by making those steps optional and falling back to a simple, deterministic baseline that still produces a valid `submission.csv`. I also ensure the script always writes a correctly formatted CSV with probabilities that sum to 1 per row and uses the required column names/order. Finally, I keep your existing ensembling/normalization logic intact when the fold prediction files exist, but won’t crash if they don’t—so you always get a submission file end-to-end (score won’t match target without the missing model artifacts, but it run and submit).'
- What this solution (achieved 1.40995) has done: 'I fix the baseline submission creation that currently explodes due to duplicated `eeg_id` rows in `sample_submission.csv` (causing a huge `.loc[...]` result and a length mismatch). The minimal change is to make `fast_baseline_from_sample_submission()` robust by dropping duplicate `eeg_id` and reindexing safely, with a fallback to global train priors if any ids are missing. This make the notebook run end-to-end, always write `/kaggle/working/submission.csv`, and pass the required probability-sum checks without altering the intended ensemble logic when fold prediction files exist. No model/training logic is changed; only the fallback I/O alignment bug is fixed.'
- What this solution (achieved 0.75933) has done: 'Your current run is falling back to a global-prior baseline (since the private checkpoints aren’t available), which is typically overconfident for this KL metric and yields a worse score than necessary. I keep your pipeline intact but improve only the fallback predictions by (1) computing per-patient priors from train votes (less mismatch than a single global prior) and (2) applying a small, deterministic probability-smoothing (mixture with uniform) to reduce extreme probabilities, which tends to lower KL. I also keep all submission formatting/normalization checks identical so it still writes a valid `/kaggle/working/submission.csv` end-to-end. These changes only affect the baseline path; if fold prediction files exist, your existing ensembling logic remains unchanged.'
- What this solution (achieved 0.76193) has done: 'Your current score (0.75933, lower-is-better) is still far from the target (0.36267), so we should improve the fallback baseline without changing the private-model path. The biggest low-risk gain for KL here is to better match the true label distribution per EEG by using training metadata: aggregate train votes at the `eeg_id` level (not per row), then condition the prior on the test `spectrogram_id` when available, with a patient-based fallback and global fallback. This keeps the same “prior baseline” core idea but reduces distribution shift vs. using only patient priors. Finally, we tune the existing smoothing to be slightly stronger (still deterministic) and keep the same normalization/checks to ensure a valid submission.'
- What this solution (achieved 0.7647) has done: 'To move your KL score down toward the target without changing any private-inference/ensemble logic, I only improve the fallback “prior baseline” path. Specifically, I (1) build priors at the correct granularity by aggregating train votes to `eeg_id` and mapping them directly to test `eeg_id` when possible (best match), then backing off to spectrogram/patient/global; and (2) tune the existing uniform-smoothing strength slightly upward to reduce overconfidence (a common KL failure mode). These are deterministic, metadata-only changes that preserve your overall approach and keep the submission formatting/normalization checks identical.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import numpy as np
import pandas as pd

CODE_PATH = "/kaggle/input/hms-mk-codes"
if CODE_PATH not in sys.path:
    sys.path.append(CODE_PATH)



## === cell 1
import subprocess


def _pip_install(path, extra_args=None):
    """
    Keep original behavior: only attempt offline installs if wheel exists.
    """
    if not os.path.exists(path):
        print(f"[skip pip] Missing wheel: {path}")
        return
    cmd = [sys.executable, "-m", "pip", "install", path, "--no-index", "--no-deps"]
    if extra_args:
        cmd += extra_args
    subprocess.run(cmd, check=True)


_pip_install(
    "/kaggle/input/requirements-mk/antlr4_python3_runtime-4.9.2-py3-none-any.whl",
    extra_args=["--force-reinstall"],
)
_pip_install("/kaggle/input/requirements-mk/omegaconf-2.3.0-py3-none-any.whl")
_pip_install("/kaggle/input/requirements-mk/hydra_core-1.3.2-py3-none-any.whl")

lightning_whl = "/kaggle/input/requirements-mk/lightning-2.2.1-py3-none-any.whl"
if os.path.exists(lightning_whl):
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            lightning_whl,
            "--no-deps",
            "--no-index",
        ],
        check=True,
    )
else:
    print(f"[skip pip] Missing wheel: {lightning_whl}")



## === cell 2
DATA_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
OUT_PATH = "/kaggle/working"
os.makedirs(OUT_PATH, exist_ok=True)

train_csv = os.path.join(DATA_PATH, "train.csv")
test_csv = os.path.join(DATA_PATH, "test.csv")
sample_sub_csv = os.path.join(DATA_PATH, "sample_submission.csv")

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)
sample_sub = pd.read_csv(sample_sub_csv)

print("train:", train_df.shape, "test:", test_df.shape, "sample_sub:", sample_sub.shape)



## === cell 3
mk_data_path = "/kaggle/input/hms-mk-data"
can_run_private = os.path.isdir(CODE_PATH) and os.path.isdir(mk_data_path)

if can_run_private and os.path.isdir(CODE_PATH):
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.convert_parquet_to_npy",
                "--data_dir",
                DATA_PATH,
                "--out_dir",
                OUT_PATH,
            ],
            cwd=CODE_PATH,
            check=True,
        )
    except Exception as e:
        print("[warn] convert_parquet_to_npy failed; continuing without it:", repr(e))
else:
    print(
        "[skip] parquet->npy conversion skipped (private inference assets unavailable)."
    )



## === cell 4
print("Listing /kaggle/input/hms-mk-data (first 50 entries):")
if os.path.isdir(mk_data_path):
    entries = sorted(os.listdir(mk_data_path))
    print("\n".join(entries[:50]))
else:
    print("Directory not found:", mk_data_path)




## === cell 5
def run_test_and_move(ckpt_path, experiment, out_csv):
    cmd = [
        sys.executable,
        "-m",
        "test",
        f"paths.data_dir={DATA_PATH}",
        f"data.test_eegs_dir={OUT_PATH}",
        f"ckpt_path={ckpt_path}",
        "hydra=test",
        f"+model.test_output_dir={OUT_PATH}",
        f"experiment={experiment}",
        "+model.net.pretrained=False",
    ]
    subprocess.run(cmd, cwd=CODE_PATH, check=True)
    src_csv = os.path.join(OUT_PATH, "submission.csv")
    if not os.path.exists(src_csv):
        raise FileNotFoundError(f"Expected inference output not found: {src_csv}")
    os.replace(src_csv, os.path.join(OUT_PATH, out_csv))


if can_run_private:
    resv2_ckpts = []
    pseudo_ckpts = []
    for fold in range(5):
        ckpt_res = f"/kaggle/input/hms-mk-data/fold{fold}_resv2.ckpt"
        ckpt_ps = f"/kaggle/input/hms-mk-data/fold{fold}_pseudo_log.ckpt"
        if os.path.exists(ckpt_res):
            resv2_ckpts.append((fold, ckpt_res))
        if os.path.exists(ckpt_ps):
            pseudo_ckpts.append((fold, ckpt_ps))

    if len(resv2_ckpts) > 0:
        for fold, ckpt in resv2_ckpts:
            run_test_and_move(
                ckpt_path=ckpt,
                experiment="conv1d_resv2",
                out_csv=f"submission_fold{fold}_v1.csv",
            )
    elif len(pseudo_ckpts) > 0:
        for fold, ckpt in pseudo_ckpts:
            run_test_and_move(
                ckpt_path=ckpt,
                experiment="conv1d_pseudo",
                out_csv=f"submission_fold{fold}_v0.csv",
            )
    else:
        print("[skip] No checkpoints found in mk-data; will use baseline submission.")
else:
    print(
        "[skip] Private inference assets not available; will use baseline submission."
    )



## === cell 6
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


def merge_preds(folds=(0, 1, 2), versions=("v0",), work_dir=OUT_PATH):
    """
    Averages predictions across all (fold, version) files provided.
    """
    all_preds = []
    base_sol = None

    for fold in folds:
        for version in versions:
            fn = os.path.join(work_dir, f"submission_fold{fold}_{version}.csv")
            if not os.path.exists(fn):
                raise FileNotFoundError(f"Missing prediction file: {fn}")
            sol = pd.read_csv(fn)
            if base_sol is None:
                base_sol = sol[["eeg_id"]].copy()
            else:
                if not np.array_equal(base_sol["eeg_id"].values, sol["eeg_id"].values):
                    sol = (
                        sol.set_index("eeg_id")
                        .loc[base_sol["eeg_id"].values]
                        .reset_index()
                    )
            all_preds.append(sol[TARGET_COLS].to_numpy(dtype=np.float64))

    preds = np.mean(np.stack(all_preds, axis=0), axis=0)
    preds = np.clip(preds, 1e-15, None)
    preds = preds / preds.sum(axis=1, keepdims=True)

    out = base_sol.copy()
    out[TARGET_COLS] = preds
    return out


def baseline_from_train_priors(train_df, test_df, target_cols):
    """
    Deterministic fallback: global class distribution from train votes.
    """
    votes = train_df[target_cols].to_numpy(np.float64)
    votes_sum = votes.sum()
    if not np.isfinite(votes_sum) or votes_sum <= 0:
        p = np.ones(len(target_cols), dtype=np.float64) / len(target_cols)
    else:
        p = votes.sum(axis=0) / votes_sum
        p = np.clip(p, 1e-15, None)
        p = p / p.sum()

    out = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
    out[target_cols] = np.tile(p, (len(out), 1))
    return out


def spectrogram_priors_from_train_eeg_agg(train_df, test_df, target_cols, alpha=0.5):
    """
    Prior baseline with backoff:
    1) eeg_id-level prior (from aggregated train votes by eeg_id) when test eeg_id exists in train,
    2) else spectrogram_id prior,
    3) else patient_id prior,
    4) else global prior.

    Change (score-improving for KL, same 'prior baseline' idea):
    Adding eeg_id-level mapping is a minimal extension that often reduces KL because many test eeg_ids
    also appear in train (different subsegments), so their aggregated label distribution is a much
    better prior than patient/spectrogram/global.
    """
    global_counts = train_df[target_cols].sum(axis=0).to_numpy(np.float64)
    global_p = (global_counts + alpha) / (
        global_counts.sum() + alpha * len(target_cols)
    )
    global_p = np.clip(global_p, 1e-15, None)
    global_p = global_p / global_p.sum()

    eeg_agg = train_df.groupby("eeg_id", as_index=False).agg(
        {
            "spectrogram_id": "first",
            "patient_id": "first",
            **{c: "sum" for c in target_cols},
        }
    )

    eeg_grp = eeg_agg.set_index("eeg_id")[target_cols]
    eeg_counts = eeg_grp.to_numpy(np.float64)
    eeg_p = (eeg_counts + alpha) / (
        eeg_counts.sum(axis=1, keepdims=True) + alpha * len(target_cols)
    )
    eeg_to_p = pd.DataFrame(eeg_p, index=eeg_grp.index, columns=target_cols)

    spec_grp = eeg_agg.groupby("spectrogram_id")[target_cols].sum()
    spec_counts = spec_grp.to_numpy(np.float64)
    spec_p = (spec_counts + alpha) / (
        spec_counts.sum(axis=1, keepdims=True) + alpha * len(target_cols)
    )
    spec_to_p = pd.DataFrame(spec_p, index=spec_grp.index, columns=target_cols)

    pat_grp = eeg_agg.groupby("patient_id")[target_cols].sum()
    pat_counts = pat_grp.to_numpy(np.float64)
    pat_p = (pat_counts + alpha) / (
        pat_counts.sum(axis=1, keepdims=True) + alpha * len(target_cols)
    )
    pat_to_p = pd.DataFrame(pat_p, index=pat_grp.index, columns=target_cols)

    out = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})

    test_eeg = test_df["eeg_id"].values
    test_spec = test_df["spectrogram_id"].values
    test_pat = test_df["patient_id"].values

    p_eeg = eeg_to_p.reindex(test_eeg).to_numpy(dtype=np.float64)
    p_spec = spec_to_p.reindex(test_spec).to_numpy(dtype=np.float64)
    p_pat = pat_to_p.reindex(test_pat).to_numpy(dtype=np.float64)

    eeg_ok = np.isfinite(p_eeg).all(axis=1)
    spec_ok = np.isfinite(p_spec).all(axis=1)
    pat_ok = np.isfinite(p_pat).all(axis=1)

    p_out = np.empty((len(test_df), len(target_cols)), dtype=np.float64)
    p_out[eeg_ok] = p_eeg[eeg_ok]

    fb_spec = (~eeg_ok) & spec_ok
    p_out[fb_spec] = p_spec[fb_spec]

    fb_pat = (~eeg_ok) & (~spec_ok) & pat_ok
    p_out[fb_pat] = p_pat[fb_pat]

    fb_global = (~eeg_ok) & (~spec_ok) & (~pat_ok)
    if fb_global.any():
        p_out[fb_global] = global_p

    out[target_cols] = p_out
    return out


def smooth_probs(p, eps=0.12):
    """
    Change (score-improving for KL): slightly stronger uniform mixing reduces overconfidence
    from prior-based predictions, which typically lowers KL for this competition.
    """
    k = p.shape[1]
    u = np.ones((1, k), dtype=np.float64) / k
    p2 = (1.0 - eps) * p + eps * u
    p2 = np.clip(p2, 1e-15, None)
    p2 = p2 / p2.sum(axis=1, keepdims=True)
    return p2


def fast_baseline_from_sample_submission(
    sample_sub, test_df, target_cols, train_df=None
):
    """
    Bugfix retained: sample_submission.csv can contain duplicate eeg_id; .loc with duplicates
    returns repeated rows causing a length mismatch. We de-duplicate by eeg_id and
    then reindex to test eeg_ids. If any ids are missing, fall back to train priors.
    """
    out = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})

    ss = sample_sub[["eeg_id"] + list(target_cols)].copy()
    ss = ss.drop_duplicates(subset=["eeg_id"], keep="first").set_index("eeg_id")

    idx = out["eeg_id"].values
    base_df = ss.reindex(idx)

    if base_df.isna().any().any():
        if train_df is not None:
            pri = baseline_from_train_priors(train_df, test_df, target_cols).set_index(
                "eeg_id"
            )
            base_df = base_df.fillna(pri.loc[idx, target_cols])
        else:
            base_df = base_df.fillna(1.0 / len(target_cols))

    out[target_cols] = base_df.to_numpy(dtype=np.float64)
    return out




## === cell 7
pred_files = glob.glob(os.path.join(OUT_PATH, "submission_fold*_v*.csv"))
if len(pred_files) > 0:
    folds_found = set()
    versions_found = set()
    for fn in pred_files:
        base = os.path.basename(fn)
        try:
            fold_part = base.split("submission_fold", 1)[1]
            fold_str, ver_csv = fold_part.split("_", 1)
            ver = ver_csv.rsplit(".csv", 1)[0]  # like "v0" or "v1"
            folds_found.add(int(fold_str))
            versions_found.add(ver)
        except Exception:
            pass

    folds_list = sorted(folds_found)
    versions_list = sorted(versions_found)
    print(
        "Found prediction files. Using folds:", folds_list, "versions:", versions_list
    )
    sol = merge_preds(folds=folds_list, versions=versions_list, work_dir=OUT_PATH)
else:
    print(
        "No fold prediction files found. Using eeg->spectrogram->patient->global prior baseline with smoothing."
    )
    sol = spectrogram_priors_from_train_eeg_agg(
        train_df, test_df, TARGET_COLS, alpha=0.5
    )

sol = sol[["eeg_id"] + TARGET_COLS].copy()
vals = sol[TARGET_COLS].to_numpy(dtype=np.float64)

vals = smooth_probs(vals, eps=0.12)

vals = np.clip(vals, 1e-15, None)
vals = vals / vals.sum(axis=1, keepdims=True)
sol[TARGET_COLS] = vals

sub_path = "/kaggle/working/submission.csv"
sol.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "shape:", sol.shape)
print(sol.head())



## === cell 8
assert sol.shape[0] == test_df.shape[0], "Row count mismatch vs test.csv"
assert list(sol.columns) == ["eeg_id"] + TARGET_COLS, "Column order/name mismatch"
row_sums = sol[TARGET_COLS].sum(axis=1).to_numpy()
assert np.allclose(row_sums, 1.0, atol=1e-6), "Probabilities do not sum to 1"
assert np.isfinite(sol[TARGET_COLS].to_numpy()).all(), "Non-finite probabilities found"
print("Submission checks passed.")
