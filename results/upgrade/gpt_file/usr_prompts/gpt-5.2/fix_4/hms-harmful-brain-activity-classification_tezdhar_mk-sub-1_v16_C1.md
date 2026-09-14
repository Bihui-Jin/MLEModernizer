# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
if os.path.isdir(CODE_PATH):
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
    print("[skip] CODE_PATH not found; skipping parquet->npy conversion:", CODE_PATH)



## === cell 4
print("Listing /kaggle/input/hms-mk-data (first 50 entries):")
mk_data_path = "/kaggle/input/hms-mk-data"
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


can_run_private = os.path.isdir(CODE_PATH) and os.path.isdir(mk_data_path)
if can_run_private:
    for fold in range(5):
        ckpt = f"/kaggle/input/hms-mk-data/fold{fold}_pseudo_log.ckpt"
        if os.path.exists(ckpt):
            run_test_and_move(
                ckpt_path=ckpt,
                experiment="conv1d_pseudo",
                out_csv=f"submission_fold{fold}_v0.csv",
            )
        else:
            print("[skip] Missing ckpt:", ckpt)

    for fold in range(5):
        ckpt = f"/kaggle/input/hms-mk-data/fold{fold}_resv2.ckpt"
        if os.path.exists(ckpt):
            run_test_and_move(
                ckpt_path=ckpt,
                experiment="conv1d_resv2",
                out_csv=f"submission_fold{fold}_v1.csv",
            )
        else:
            print("[skip] Missing ckpt:", ckpt)
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


def spectro_lr_baseline(
    train_df, test_df, data_path, target_cols, max_files_per_split=None
):
    import warnings

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
    except Exception as e:
        print(
            "[warn] sklearn not available for LR baseline; falling back to priors:",
            repr(e),
        )
        return baseline_from_train_priors(train_df, test_df, target_cols)

    spec_dir = os.path.join(data_path, "train_spectrograms")
    test_spec_dir = os.path.join(data_path, "test_spectrograms")
    if not (os.path.isdir(spec_dir) and os.path.isdir(test_spec_dir)):
        print("[warn] spectrogram directories missing; falling back to priors")
        return baseline_from_train_priors(train_df, test_df, target_cols)

    def _safe_read_parquet(path):
        try:
            return pd.read_parquet(path)
        except Exception as e:
            raise RuntimeError(f"Failed to read parquet {path}: {e}")

    def _spec_features_from_df(df):
        x = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        if x.size == 0:
            return np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = np.log1p(np.maximum(x, 0.0))
        flat = x.ravel()
        mean = flat.mean()
        std = flat.std()
        q10, q50, q90 = np.quantile(flat, [0.1, 0.5, 0.9])
        mx = flat.max() if flat.size else 0.0
        return np.array([mean, std, q10, q50, q90, mx], dtype=np.float64)

    tr = train_df[["spectrogram_id"] + target_cols].copy()
    tr_sum = tr.groupby("spectrogram_id", as_index=False)[target_cols].sum()
    votes = tr_sum[target_cols].to_numpy(np.float64)
    votes = np.clip(votes, 0.0, None)
    row_sums = votes.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.divide(votes, row_sums, out=np.zeros_like(votes), where=row_sums > 0)
    y = probs.argmax(axis=1).astype(int)

    spec_ids = tr_sum["spectrogram_id"].values
    if max_files_per_split is not None:
        spec_ids = spec_ids[:max_files_per_split]

    X_list = []
    y_list = []
    kept_ids = []
    for sid, yi in zip(spec_ids, y[: len(spec_ids)]):
        p = os.path.join(spec_dir, f"{sid}.parquet")
        if not os.path.exists(p):
            continue
        try:
            dfp = _safe_read_parquet(p)
            feat = _spec_features_from_df(dfp)
        except Exception:
            continue
        X_list.append(feat)
        y_list.append(yi)
        kept_ids.append(sid)

    if len(X_list) < 100:
        print(
            "[warn] too few spectrogram features read; falling back to priors:",
            len(X_list),
        )
        return baseline_from_train_priors(train_df, test_df, target_cols)

    X_train = np.vstack(X_list)
    y_train = np.array(y_list, dtype=int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                (
                    "lr",
                    LogisticRegression(
                        multi_class="multinomial",
                        solver="lbfgs",
                        C=0.5,  # mild regularization for better calibration vs overfit
                        max_iter=200,
                        n_jobs=None,
                        random_state=0,
                    ),
                ),
            ]
        )
        clf.fit(X_train, y_train)

    test_spec_ids = test_df["spectrogram_id"].values
    if max_files_per_split is not None:
        test_spec_ids = test_spec_ids[:max_files_per_split]

    Xte_list = []
    ok_mask = []
    for sid in test_df["spectrogram_id"].values:
        p = os.path.join(test_spec_dir, f"{sid}.parquet")
        if not os.path.exists(p):
            Xte_list.append(np.zeros(6, dtype=np.float64))
            ok_mask.append(False)
            continue
        try:
            dfp = _safe_read_parquet(p)
            feat = _spec_features_from_df(dfp)
            Xte_list.append(feat)
            ok_mask.append(True)
        except Exception:
            Xte_list.append(np.zeros(6, dtype=np.float64))
            ok_mask.append(False)

    X_test = np.vstack(Xte_list)
    pred = clf.predict_proba(X_test).astype(np.float64)

    pred = np.clip(pred, 1e-15, None)
    pred = pred / pred.sum(axis=1, keepdims=True)

    out = pd.DataFrame({"eeg_id": test_df["eeg_id"].values})
    out[target_cols] = pred
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
            ver = ver_csv.rsplit(".csv", 1)[0]  # like "v0"
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
        "No fold prediction files found. Using spectrogram LR baseline (fallback to priors if needed)."
    )
    sol = spectro_lr_baseline(train_df, test_df, DATA_PATH, TARGET_COLS)

sol = sol[["eeg_id"] + TARGET_COLS].copy()
vals = sol[TARGET_COLS].to_numpy(dtype=np.float64)
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
