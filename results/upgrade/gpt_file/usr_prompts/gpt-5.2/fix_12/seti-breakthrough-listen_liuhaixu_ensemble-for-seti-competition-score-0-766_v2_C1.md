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
Identify technosignature signals in cadence snippets taken from a digital spectrometer.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `id` in the test set, you must predict a probability for the `target` variable. The file should contain a header and have the following format:

```
id,target
00034abb3629,0.5
0004be0baf70,0.5
0005be4d0752,0.5
etc.

```

## Dataset
The data is from a digital spectrometer, which takes incoming raw data from the telescope (amounting to hundreds of TB per day) and performs a Fourier Transform to generate a spectrogram. These spectrograms, also referred to as filterbank files, or dynamic spectra, consist of measurements of signal intensity as a function of frequency and time.

Below is an example of an FM radio signal. This is not from the GBT, but from a small antenna attached to a software defined radio dongle (a $20 piece of kit that you can plug into your laptop to pick up signals). The data we get from the GBT are very similar, but split into larger numbers of frequency channels, covering a much broader instantaneous frequency range, and with much better sensitivity.

![frequency-time-plot](https://prod-files-secure.s3.us-west-2.amazonaws.com/667f1cbf-826f-4641-a321-96054292638d/b59a57f3-11a7-4493-8268-55c3fa632f7e/Untitled.png)

The screenshot above shows frequency on the horizontal axis (running from around 88.2 to 89.8 MHz) and time on the vertical axis. The bright orange feature at 88.5 MHz is the FM signal from KQED, a radio station in the San Francisco Bay Area. The solid yellow blocks on either side (one highlighted by the pointer in the screenshot) are the KQED “HD radio” signal (the same data as the FM signal, but encoded digitally). Additional FM stations are visible at different frequencies, including another obvious FM signal (without the corresponding digital sidebands) at 89.5 MHz.

The spectrometer generates similar spectrograms to the one shown above, but typically spanning several GHz of the radio spectrum (rather than the approx. 2 MHz shown above). The data are stored either as filterbank format or HDF5 format files, but essentially are arrays of intensity as a function of frequency and time, accompanied by headers containing metadata such as the direction the telescope was pointed in, the frequency scale, and so on. We generate over 1 PB of spectrograms per year; individual filterbank files can be tens of GB in size. We have discarded the majority of the metadata and are simply presenting numpy arrays consisting of small regions of the spectrograms that we refer to as “snippets”.

The spectrometer is searching for candidate signatures of extraterrestrial technology - so-called technosignatures. The main obstacle to doing so is that our own human technology (not just radio stations, but wifi routers, cellphones, and even electronics that are not deliberately designed to transmit radio signals) also gives off radio signals. We refer to these human-generated signals as “radio frequency interference”, or RFI.

One method we use to isolate candidate technosignatures from RFI is to look for signals that appear to be coming from particular positions on the sky. Typically we do this by alternating observations of our primary target star with observations of three nearby stars: 5 minutes on star “A”, then 5 minutes on star “B”, then back to star “A” for 5 minutes, then “C”, then back to “A”, then finishing with 5 minutes on star “D”. One set of six observations (ABACAD) is referred to as a “cadence”. Since we're just giving you a small range of frequencies for each cadence, we refer to the datasets you'll be analyzing as “cadence snippets”.

An example of an extraterrestrial signal:

![voyager-signal](https://storage.googleapis.com/kaggle-media/competitions/SETI-Berkeley/Screen%20Shot%202021-05-03%20at%2011.39.42.png)

As the plot title suggests, this is the Voyager 1 spacecraft. Even though it's 20 billion kilometers from Earth, it's picked up clearly by the GBT. The first, third, and fifth panels are the “A” target (the spacecraft, in this case). The yellow diagonal line is the radio signal coming from Voyager. It's detected when we point at the spacecraft, and it disappears when we point away. It's a diagonal line in this plot because the relative motion of the Earth and the spacecraft imparts a Doppler drift, causing the frequency to change over time. As it happens, that's another possible way to reject RFI, which has a higher tendency to remain at a fixed frequency over time.

While it would be nice to train our algorithms entirely on observations of interplanetary spacecraft, there are not many examples of them, and we also want to be able to find a wider range of signal types. So we've turned to simulating technosignature candidates.

We've taken tens of thousands of cadence snippets, which we're calling the haystack, and we've hidden needles among them. Some of these needles look similar to the Voyager 1 signal above and should be easy to detect, even with classical detection algorithms. Others are hidden in noisy regions of the spectrum and will be harder, even though they might be relatively obvious on visual inspection:

![needle-signal](https://storage.googleapis.com/kaggle-media/competitions/SETI-Berkeley/Screen%20Shot%202021-05-03%20at%2011.34.06.png)

After we perform the signal injections, we normalize each snippet, so you probably can't identify most of the needles just by looking for excess energy in the corresponding array. You'll likely need a more subtle algorithm that looks for patterns that appear only in the on-target observations.

Not all of the “needle” signals look like diagonal lines, and they may not be present for the entirety of all three “A” observations, but what they do have in common is that they are only present in some or all of the “A” observations (panels 1, 3, and 5 in the cadence snippets). Your challenge is to train an algorithm to find as many needles as you can, while minimizing the number of false positives from the haystack.

- **train/** - a training set of cadence snippet files stored in `numpy` `float16` format (v1.20.1), one file per cadence snippet `id`, with corresponding labels found in the `train_labels.csv` file. Each file has dimension `(6, 273, 256)`, with the 1st dimension representing the 6 positions of the cadence, and the 2nd and 3rd dimensions representing the 2D spectrogram.
- **test/** - the test set cadence snippet files; you must predict whether or not the cadence contains a "needle", which is the `target` for this competition
- **sample_submission.csv** - a sample submission file in the correct format
- **train_labels** - targets corresponding (by `id`) to the cadence snippet files found in the `train/` folder
- **old_leaky_data** - full pre-relaunch data, including test labels; you should not assume this data is helpful (it may or may not be).

# 2. Python version

3.9

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
            description.md (112 lines)
            old_leaky_data.zip (23.6 GB)
            sample_submission.csv (6001 lines)
            sample_submission.csv.zip (60.0 kB)
            test.zip (4.5 GB)
            train.zip (4.7 GB)
            train_labels.csv (54001 lines)
            train_labels.csv.zip (529.5 kB)
            old_leaky_data/
                test_labels_old.csv (35848 lines)
                train_labels_old.csv (50166 lines)
                test_old/
                    0/
                        00034db451c4.npy (838.8 kB)
                        0006316b5ca0.npy (838.8 kB)
                        ... and 2197 other files
                    1/
                        10038983cab1.npy (838.8 kB)
                        100865aff453.npy (838.8 kB)
                        ... and 2278 other files
                    ... and 14 other folders
                train_old/
                    0/
                        00034abb3629.npy (838.8 kB)
                        0004300a0b9b.npy (838.8 kB)
                        ... and 3143 other files
                    1/
                        1000e00b26db.npy (838.8 kB)
                        100148224705.npy (838.8 kB)
                        ... and 3142 other files
                    ... and 14 other folders
            seti-breakthrough-listen/
                description.md (112 lines)
                old_leaky_data.zip (23.6 GB)
                ... and 6 other files
                old_leaky_data/
                    test_labels_old.csv (35848 lines)
                    train_labels_old.csv (50166 lines)
                    test_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    train_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                seti-breakthrough-listen/
                test/
                    0/
                        0016fd6c09d476d.npy (838.8 kB)
                        0017643c1c5c254.npy (838.8 kB)
                        ... and 374 other files
                    1/
                        1001ca1d08f9235.npy (838.8 kB)
                        1016de9cec2dc8a.npy (838.8 kB)
                        ... and 353 other files
                    ... and 15 other folders
                train/
                    0/
                        0000799a2b2c42d.npy (838.8 kB)
                        00042890562ff68.npy (838.8 kB)
                        ... and 3335 other files
                    1/
                        100105755d4c5b1.npy (838.8 kB)
                        1001a55ebce86f2.npy (838.8 kB)
                        ... and 3392 other files
                    ... and 15 other folders
            test/
                0/
                    0016fd6c09d476d.npy (838.8 kB)
                    0017643c1c5c254.npy (838.8 kB)
                    ... and 374 other files
                1/
                    1001ca1d08f9235.npy (838.8 kB)
                    1016de9cec2dc8a.npy (838.8 kB)
                    ... and 353 other files
                ... and 15 other folders
            train/
                0/
                    0000799a2b2c42d.npy (838.8 kB)
                    00042890562ff68.npy (838.8 kB)
                    ... and 3335 other files
                1/
                    100105755d4c5b1.npy (838.8 kB)
                    1001a55ebce86f2.npy (838.8 kB)
                    ... and 3392 other files
                ... and 15 other folders
        input/
            description.md (112 lines)
            old_leaky_data.zip (23.6 GB)
            sample_submission.csv (6001 lines)
            sample_submission.csv.zip (60.0 kB)
            test.zip (4.5 GB)
            train.zip (4.7 GB)
            train_labels.csv (54001 lines)
            train_labels.csv.zip (529.5 kB)
            old_leaky_data/
                test_labels_old.csv (35848 lines)
                train_labels_old.csv (50166 lines)
                test_old/
                    0/
                        00034db451c4.npy (838.8 kB)
                        0006316b5ca0.npy (838.8 kB)
                        ... and 2197 other files
                    1/
                        10038983cab1.npy (838.8 kB)
                        100865aff453.npy (838.8 kB)
                        ... and 2278 other files
                    ... and 14 other folders
                train_old/
                    0/
                        00034abb3629.npy (838.8 kB)
                        0004300a0b9b.npy (838.8 kB)
                        ... and 3143 other files
                    1/
                        1000e00b26db.npy (838.8 kB)
                        100148224705.npy (838.8 kB)
                        ... and 3142 other files
                    ... and 14 other folders
            seti-breakthrough-listen/
                description.md (112 lines)
                old_leaky_data.zip (23.6 GB)
                ... and 6 other files
                old_leaky_data/
                    test_labels_old.csv (35848 lines)
                    train_labels_old.csv (50166 lines)
                    test_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    train_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                seti-breakthrough-listen/
                test/
                    0/
                        0016fd6c09d476d.npy (838.8 kB)
                        0017643c1c5c254.npy (838.8 kB)
                        ... and 374 other files
                    1/
                        1001ca1d08f9235.npy (838.8 kB)
                        1016de9cec2dc8a.npy (838.8 kB)
                        ... and 353 other files
                    ... and 15 other folders
                train/
                    0/
                        0000799a2b2c42d.npy (838.8 kB)
                        00042890562ff68.npy (838.8 kB)
                        ... and 3335 other files
                    1/
                        100105755d4c5b1.npy (838.8 kB)
                        1001a55ebce86f2.npy (838.8 kB)
                        ... and 3392 other files
                    ... and 15 other folders
            test/
                0/
                    0016fd6c09d476d.npy (838.8 kB)
                    0017643c1c5c254.npy (838.8 kB)
                    ... and 374 other files
                1/
                    1001ca1d08f9235.npy (838.8 kB)
                    1016de9cec2dc8a.npy (838.8 kB)
                    ... and 353 other files
                ... and 15 other folders
            train/
                0/
                    0000799a2b2c42d.npy (838.8 kB)
                    00042890562ff68.npy (838.8 kB)
                    ... and 3335 other files
                1/
                    100105755d4c5b1.npy (838.8 kB)
                    1001a55ebce86f2.npy (838.8 kB)
                    ... and 3392 other files
                ... and 15 other folders
        working/
            seti-breakthrough-listen/
                description.md (112 lines)
                old_leaky_data.zip (23.6 GB)
                ... and 6 other files
                old_leaky_data/
                    test_labels_old.csv (35848 lines)
                    train_labels_old.csv (50166 lines)
                    test_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                    train_old/
                        0/
                            ... (max depth reached)
                        1/
                            ... (max depth reached)
                        ... and 14 other folders
                seti-breakthrough-listen/
                test/
                    0/
                        0016fd6c09d476d.npy (838.8 kB)
                        0017643c1c5c254.npy (838.8 kB)
                        ... and 374 other files
                    1/
                        1001ca1d08f9235.npy (838.8 kB)
                        1016de9cec2dc8a.npy (838.8 kB)
                        ... and 353 other files
                    ... and 15 other folders
                train/
                    0/
                        0000799a2b2c42d.npy (838.8 kB)
                        00042890562ff68.npy (838.8 kB)
                        ... and 3335 other files
                    1/
                        100105755d4c5b1.npy (838.8 kB)
                        1001a55ebce86f2.npy (838.8 kB)
                        ... and 3392 other files
                    ... and 15 other folders
```

-> data/old_leaky_data/test_labels_old.csv has 35847 rows and 2 columns.
The columns are: id, target

-> data/old_leaky_data/train_labels_old.csv has 50165 rows and 2 columns.
The columns are: id, target

-> data/sample_submission.csv has 6000 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/old_leaky_data/test_labels_old.csv has 35847 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/old_leaky_data/train_labels_old.csv has 50165 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/sample_submission.csv has 6000 rows and 2 columns.
The columns are: id, target

-> data/seti-breakthrough-listen/train_labels.csv has 54000 rows and 2 columns.
The columns are: id, target

-> data/train_labels.csv has 54000 rows and 2 columns.
The columns are: id, target

-> (stopped after 10 files for performance)

# 5. Target score

0.762598927697958

# 6. Current score

0.50892

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.50892) has done: 'The timeout is dominated by reading and featurizing tens of thousands of `.npy` files twice (train + test) in pure Python multiprocessing, plus an expensive full recursive scan of `train/` and `test/`. I keep the exact same features and logistic-regression CV logic, but make file discovery and loading significantly faster by (1) removing the full directory scan (paths are deterministic from `id`), (2) switching to a threads-based pool to overlap disk I/O and avoid process-spawn/pickle overhead, and (3) avoiding unnecessary copies when converting `float16` to `float32`. I also reduce per-item Python overhead by batching results collection and by precomputing feature dimension once. These changes are equivalent in outputs (same inputs, same feature math, same model training/prediction), but cut constant factors enough to fit within 600s on typical Kaggle CPUs.'
- What this solution (achieved 0.50892) has done: 'Your current score (0.50892) is far below the target (0.7626), so we should increase performance while keeping your core logic intact. The biggest issue is that your pipeline prioritizes loading external “submission.csv” files (which likely aren’t present here); when they are present, the hard-coded ensemble weights can yield weak predictions and drag AUC down. I keep the same ensemble mechanism but add an automatic safeguard: if external submissions are missing or look low-quality/unaligned, we fall back to your in-notebook logistic-regression features model (unchanged). Additionally, I make the external ensemble more robust by (1) requiring full alignment to sample_submission ids and (2) using a simple equal-weight average (instead of the current weight vector that can overweight a single bad file), which should move AUC upward toward the target without changing the modeling approach.'
- What this solution (achieved 0.50892) has done: 'Your current AUC (0.50892) is far below the target (0.7626), so we should improve predictions while keeping your exact feature extraction and LogisticRegression CV approach intact. The largest score drag in your script is using an external ensemble whenever any external submissions exist (these may be weak/misaligned for your environment), so I add a lightweight quality gate using *only training data* to decide whether to use that ensemble or fall back to your internal logistic-regression model. Specifically, I score each external submission by AUC on the known `old_leaky_data/test_labels_old.csv` (legitimate historical labels provided in the dataset) after id-alignment, and only ensemble submissions that clear a conservative threshold; otherwise we keep your existing internal model unchanged. This is a minimal change (no architecture/training changes) but should move AUC upward toward the target by avoiding low-quality external predictions.'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(42)



## === cell 1
candidate_paths = [
    "/kaggle/input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv",
    "/kaggle/input/lb-0-980-efficientnet-b0-more-epoch/submission.csv",
    "/kaggle/input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv",
    "/kaggle/input/seti-learned-image-resizing/submission.csv",
    "/kaggle/input/rerun-seti-e-t-resnet18d-baseline/submission.csv",
    "/kaggle/input/ensemble-for-seti-competition/submission.csv",
    "/kaggle/input/fixed-gradual-warmup-custom-head/submission.csv",
]

loaded = []
loaded_paths = []
for p in candidate_paths:
    if os.path.exists(p):
        df = pd.read_csv(p, usecols=["id", "target"])
        loaded.append(df.copy())
        loaded_paths.append(p)

print(f"Found {len(loaded)} external submission files.")
if loaded_paths:
    print("Loaded:", loaded_paths[:3], "..." if len(loaded_paths) > 3 else "")



## === cell 2
submission = None

sample_sub_candidates = [
    "/kaggle/input/seti-breakthrough-listen/sample_submission.csv",
    "/kaggle/data/seti-breakthrough-listen/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
]
sample_path = next((p for p in sample_sub_candidates if os.path.exists(p)), None)
sample_ids = None
if sample_path is not None:
    sample_ids = pd.read_csv(sample_path, usecols=["id"])["id"].astype(str).to_numpy()


def _fast_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)
    n = y_true.size
    if n == 0:
        return np.nan
    n_pos = int(y_true.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    sorted_scores = y_score[order]
    diffs = np.diff(sorted_scores)
    if np.any(diffs == 0):
        start = 0
        while start < n:
            end = start
            while end + 1 < n and sorted_scores[end + 1] == sorted_scores[start]:
                end += 1
            if end > start:
                avg_rank = 0.5 * (start + 1 + end + 1)
                ranks[order[start : end + 1]] = avg_rank
            start = end + 1
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def _find_old_test_labels():
    candidates = [
        "/kaggle/input/seti-breakthrough-listen/old_leaky_data/test_labels_old.csv",
        "/kaggle/data/seti-breakthrough-listen/old_leaky_data/test_labels_old.csv",
        "/kaggle/data/old_leaky_data/test_labels_old.csv",
        "/kaggle/input/old_leaky_data/test_labels_old.csv",
    ]
    return next((p for p in candidates if os.path.exists(p)), None)


old_test_labels_path = _find_old_test_labels()
old_test_labels = None
if old_test_labels_path is not None:
    old_test_labels = pd.read_csv(old_test_labels_path, usecols=["id", "target"])
    old_test_labels["id"] = old_test_labels["id"].astype(str)
    old_test_labels["target"] = old_test_labels["target"].astype(int)
    print(
        f"Found old test labels for external-submission gating: {old_test_labels_path} (n={len(old_test_labels)})"
    )
else:
    print("Did not find old test labels; external ensemble gating will be skipped.")


def try_build_external_ensemble(loaded_dfs, ids_ref, old_labels_df=None):
    if ids_ref is None or len(loaded_dfs) == 0:
        return None

    ids_ref = np.asarray(ids_ref, dtype=str)
    ref_index = pd.Index(ids_ref)

    cols = []
    used_paths = []
    aucs = []

    gate_auc_threshold = 0.75

    old_index = None
    old_y = None
    if old_labels_df is not None and len(old_labels_df) > 0:
        old_index = pd.Index(old_labels_df["id"].to_numpy(dtype=str))
        old_y = old_labels_df["target"].to_numpy(dtype=np.int8)

    for df, pth in zip(loaded_dfs, loaded_paths):
        if not {"id", "target"}.issubset(df.columns):
            continue
        d = df.copy()
        d["id"] = d["id"].astype(str)
        d = d.drop_duplicates("id", keep="first")
        d = d.set_index("id")["target"]

        aligned = d.reindex(ref_index)
        if aligned.isna().any():
            continue

        col = aligned.to_numpy(dtype=np.float64, copy=False)
        col = np.clip(col, 0.0, 1.0)
        if not np.isfinite(col).all():
            continue
        if float(np.std(col)) < 1e-6:
            continue

        if old_index is not None:
            old_pred = d.reindex(old_index)
            if old_pred.isna().any():
                continue
            old_pred = np.clip(
                old_pred.to_numpy(dtype=np.float64, copy=False), 0.0, 1.0
            )
            auc = _fast_auc(old_y, old_pred)
            if not np.isfinite(auc):
                continue
            if auc < gate_auc_threshold:
                continue
            aucs.append(auc)

        cols.append(col)
        used_paths.append(pth)

    if len(cols) == 0:
        return None

    preds = np.vstack(cols).T  # (n_ids, n_models)
    ens = preds.mean(axis=1)

    out = pd.DataFrame({"id": ids_ref, "target": np.clip(ens, 0.0, 1.0)})
    return out, used_paths, aucs


ext = try_build_external_ensemble(loaded, sample_ids, old_test_labels)
if ext is not None:
    submission, used_paths, aucs = ext
    print(f"Using external ensemble with {len(used_paths)} files.")
    if aucs:
        print(
            f"External files passed gate AUC>={0.75:.2f} on old labels; AUCs (first 5): {aucs[:5]}"
        )
    print("Ensembled:", used_paths[:3], "..." if len(used_paths) > 3 else "")



## === cell 3
if submission is None:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.linear_model import LogisticRegression
    from multiprocessing.pool import ThreadPool

    CANDIDATE_ROOTS = [
        "/kaggle/input/seti-breakthrough-listen",
        "/kaggle/data/seti-breakthrough-listen",
        "/kaggle/data",
        "/kaggle/input",
    ]

    def resolve_comp_root():
        for r in CANDIDATE_ROOTS:
            if not os.path.exists(r):
                continue
            if (
                os.path.exists(os.path.join(r, "train_labels.csv"))
                and os.path.exists(os.path.join(r, "sample_submission.csv"))
                and os.path.isdir(os.path.join(r, "train"))
                and os.path.isdir(os.path.join(r, "test"))
            ):
                return r
            sub = os.path.join(r, "seti-breakthrough-listen")
            if (
                os.path.exists(os.path.join(sub, "train_labels.csv"))
                and os.path.exists(os.path.join(sub, "sample_submission.csv"))
                and os.path.isdir(os.path.join(sub, "train"))
                and os.path.isdir(os.path.join(sub, "test"))
            ):
                return sub
        return None

    root = resolve_comp_root()
    if root is None:
        raise FileNotFoundError(
            "Could not find competition data root. Looked for train_labels.csv/sample_submission.csv "
            "and train/test folders under: " + ", ".join(CANDIDATE_ROOTS)
        )

    train_labels_path = os.path.join(root, "train_labels.csv")
    sample_sub_path = os.path.join(root, "sample_submission.csv")
    train_dir = os.path.join(root, "train")
    test_dir = os.path.join(root, "test")

    labels = pd.read_csv(train_labels_path, usecols=["id", "target"])
    sample_sub = pd.read_csv(sample_sub_path, usecols=["id"])

    def paths_for_ids_fast(base_dir: str, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids, dtype=str)
        first = np.fromiter((s[0] for s in ids), dtype="<U1", count=len(ids))
        return np.char.add(
            np.char.add(np.char.add(np.char.add(base_dir, os.sep), first), os.sep),
            np.char.add(ids, ".npy"),
        )

    def extract_features_from_path(path: str) -> np.ndarray:
        x16 = np.load(path, mmap_mode="r")  # float16 on disk
        x = x16.astype(np.float32, copy=False)

        means = x.mean(axis=(1, 2))
        stds = x.std(axis=(1, 2))
        maxs = x.max(axis=(1, 2))

        A = x[0::2]  # (3,273,256)
        BCD = x[1::2]  # (3,273,256)

        a_mean = A.mean()
        b_mean = BCD.mean()
        a_std = A.std()
        b_std = BCD.std()
        a_max = A.max()
        b_max = BCD.max()

        A_freq_mean = A.mean(axis=1).mean(axis=0)
        B_freq_mean = BCD.mean(axis=1).mean(axis=0)
        freq_contrast = A_freq_mean - B_freq_mean
        fc_mean = freq_contrast.mean()
        fc_std = freq_contrast.std()
        fc_max = freq_contrast.max()
        fc_min = freq_contrast.min()

        return np.concatenate(
            [
                means,
                stds,
                maxs,
                np.array(
                    [
                        a_mean,
                        b_mean,
                        a_std,
                        b_std,
                        a_max,
                        b_max,
                        a_mean - b_mean,
                        a_max - b_max,
                    ],
                    dtype=np.float32,
                ),
                np.array([fc_mean, fc_std, fc_max, fc_min], dtype=np.float32),
            ],
            axis=0,
        )

    def featurize_paths_in_order(paths, max_workers=None, chunksize=256):
        n = len(paths)
        if n == 0:
            return np.empty((0, 0), dtype=np.float32)

        d0 = extract_features_from_path(paths[0]).shape[0]
        X_out = np.empty((n, d0), dtype=np.float32)

        if max_workers is None:
            cpu = os.cpu_count() or 2
            max_workers = min(16, max(4, cpu))

        with ThreadPool(processes=max_workers) as pool:
            for i, feat in enumerate(
                pool.imap(extract_features_from_path, paths, chunksize)
            ):
                X_out[i] = feat
        return X_out

    train_ids = labels["id"].to_numpy(dtype=str)
    train_paths = paths_for_ids_fast(train_dir, train_ids)
    if len(train_paths) > 0:
        rng = np.random.RandomState(42)
        k = min(256, len(train_paths))
        sample_idx = rng.choice(len(train_paths), size=k, replace=False)
        missing = [
            train_paths[i] for i in sample_idx if not os.path.exists(train_paths[i])
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing training .npy files (e.g., {missing[0]})."
            )

    X = featurize_paths_in_order(train_paths)
    y = labels["target"].astype(int).to_numpy()

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y), dtype=np.float64)

    models = []
    for tr_idx, va_idx in skf.split(X, y):
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            n_jobs=None,
            class_weight=None,
        )
        model.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(X[va_idx])[:, 1]
        models.append(model)

    test_ids = sample_sub["id"].to_numpy(dtype=str)
    test_paths = paths_for_ids_fast(test_dir, test_ids)

    missing_mask = np.fromiter(
        (not os.path.exists(p) for p in test_paths), dtype=bool, count=len(test_paths)
    )
    if missing_mask.any():
        missing = test_ids[np.flatnonzero(missing_mask)][:3].tolist()
        raise FileNotFoundError(
            f"Missing {int(missing_mask.sum())} test .npy files (e.g., {missing})."
        )

    X_test = featurize_paths_in_order(test_paths)
    pred = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)

    submission = pd.DataFrame({"id": test_ids, "target": np.clip(pred, 0.0, 1.0)})



## === cell 4
if submission is None:
    raise RuntimeError("submission was not created; cannot write submission.csv")

submission = submission[["id", "target"]].copy()

sample_sub_candidates = [
    "/kaggle/input/seti-breakthrough-listen/sample_submission.csv",
    "/kaggle/data/seti-breakthrough-listen/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/input/sample_submission.csv",
]
sample_path = next((p for p in sample_sub_candidates if os.path.exists(p)), None)
if sample_path is not None:
    ss = pd.read_csv(sample_path, usecols=["id"])
    submission["id"] = submission["id"].astype(str)
    ss["id"] = ss["id"].astype(str)
    submission = ss.merge(submission, on="id", how="left")
    if submission["target"].isna().any():
        submission["target"] = submission["target"].fillna(0.5)
    submission["target"] = submission["target"].astype(float).clip(0.0, 1.0)

submission.to_csv("submission.csv", index=False)

print(submission.head())
print(
    f"Wrote submission.csv with shape={submission.shape} and columns={submission.columns.tolist()}"
)
