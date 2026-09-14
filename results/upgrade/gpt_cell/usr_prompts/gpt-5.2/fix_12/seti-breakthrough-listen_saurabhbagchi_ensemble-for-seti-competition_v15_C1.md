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

0.75713

# 6. Current score

0.51552

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 1 crashes because it tries to read multiple Kaggle “../input/.../submission.csv” files that do not exist in this environment, causing a `FileNotFoundError` on the first `pd.read_csv`. The notebook expects six DataFrames (`data1`…`data6`) with `id` and `target` columns to exist for later blending/inspection.  
Patch summary: In cell 1 only, add a tiny safe loader that (a) reads the file if present, otherwise (b) falls back to the provided local `sample_submission.csv` so the DataFrames exist with the expected schema. This preserves downstream interfaces without inventing new modeling logic.  
Updated cells: Only cell 1 is modified.  
Compatibility notes for cell k+1: `data1` (and the others) remain pandas DataFrames; `data1.head()` in cell 2 work unchanged and show the standard `id/target` columns.  
Assumptions: Using `../input` paths is optional in this environment; falling back to `../data/sample_submission.csv` (or its existing equivalent) is acceptable to keep the notebook runnable when external submissions are absent.'
- What this solution (achieved 0.5214) has done: 'The timeout is dominated by `_train_and_predict_submission()`, which computes features by loading tens of thousands of `.npy` files one-by-one and also repeatedly scans the entire directory tree to build `id2path`. To keep identical core logic and predictions, the key speedups are: avoid full directory scans by using direct `_id_to_path()` plus a tiny on-demand cache for missing files, and parallelize feature extraction across CPU cores while keeping deterministic ordering. We also keep memory-mapped loads and the same feature function, model, and ensemble math, but make caching more robust so reruns don’t recompute. These changes reduce wall-clock time substantially without changing the algorithm or accuracy (only negligible float differences possible due to parallel execution order, but each sample is computed independently).'
- What this solution (achieved 0.52137) has done: 'You’re currently far below the target AUC, so the smallest reliable way to move toward 0.757 is to improve the blended submission while keeping your “fallback” model and blending logic intact. The main issue is that `data2..data5` can all end up being identical to the heuristic fallback, so your blend effectively collapses to the same weak predictions; we make the blend robust by (1) training multiple deterministic logistic-regression models (same feature extractor and same training approach) on different regularization strengths and averaging probabilities, and (2) using those averaged predictions as the fallback base that populates any missing external submissions. This preserves the same core semantics (feature extraction → logistic regression → predict_proba → blend) but increases signal enough to move the score toward the target. We also ensure the final `submission.csv` has the correct columns, order, and row alignment with `sample_submission.csv`.'
- What this solution (achieved 0.51552) has done: 'We keep your exact “blend external submissions, otherwise fall back to the heuristic logistic-regression model” core logic, but make the fallback a bit stronger (to move AUC up toward the 0.75713 target) by (1) adding a few simple, cheap-to-compute features that better capture “A-only” signal structure, and (2) standardizing features before logistic regression so regularization behaves sensibly across feature scales. We also make `_align_on_id` robust to missing IDs by filling with the base mean instead of propagating NaNs that later become 0.5, which tends to hurt AUC. Everything still runs CPU-only under sklearn/numpy/pandas and writes a valid `submission.csv` with `id,target`.'

# 9. Code solution

## === cell 0
import os
import hashlib
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from concurrent.futures import ThreadPoolExecutor


def _find_base_dir() -> str:
    for cand in ("/kaggle/data", "../data", "/kaggle/input", "../input"):
        if os.path.exists(cand):
            if os.path.exists(os.path.join(cand, "train")) or os.path.exists(
                os.path.join(cand, "seti-breakthrough-listen")
            ):
                return cand
    return "/kaggle/data"


BASE = _find_base_dir()

TRAIN_DIR = os.path.join(BASE, "train")
TEST_DIR = os.path.join(BASE, "test")
LABELS_PATH = os.path.join(BASE, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(BASE, "sample_submission.csv")

if not (
    os.path.exists(TRAIN_DIR)
    and os.path.exists(TEST_DIR)
    and os.path.exists(LABELS_PATH)
):
    alt = os.path.join(BASE, "seti-breakthrough-listen")
    TRAIN_DIR = os.path.join(alt, "train")
    TEST_DIR = os.path.join(alt, "test")
    LABELS_PATH = os.path.join(alt, "train_labels.csv")
    SAMPLE_SUB_PATH = os.path.join(alt, "sample_submission.csv")


def _extract_features_from_array(x: np.ndarray) -> np.ndarray:
    """
    x shape: (6, 273, 256), float16.

    Change (score-improving, minimal): add a few additional "A-only structure" features
    while keeping the same fast summary-feature approach.
    These extra features are cheap and often improve separability for ROC-AUC.
    """
    x = x.astype(np.float32, copy=False)

    A = x[[0, 2, 4]]
    N = x[[1, 3, 5]]

    mean_all = x.mean()
    std_all = x.std()

    mean_A = A.mean()
    mean_N = N.mean()
    std_A = A.std()
    std_N = N.std()

    eA = (A * A).mean()
    eN = (N * N).mean()
    diff_e = eA - eN
    ratio_e = eA / (eN + 1e-6)

    x2 = x.reshape(6, -1)
    per_panel_max = x2.max(axis=1)
    max_mean = per_panel_max.mean()
    max_std = per_panel_max.std()
    max_A = per_panel_max[[0, 2, 4]].mean()
    max_N = per_panel_max[[1, 3, 5]].mean()
    max_diff = max_A - max_N

    per_panel_p99 = np.percentile(x2, 99.0, axis=1)
    occ = np.array(
        [(x2[i] > per_panel_p99[i]).mean() for i in range(6)], dtype=np.float32
    )
    occ_A = occ[[0, 2, 4]].mean()
    occ_N = occ[[1, 3, 5]].mean()
    occ_diff = occ_A - occ_N

    gA = np.abs(np.diff(A, axis=1)).mean()
    gN = np.abs(np.diff(N, axis=1)).mean()
    gdiff = gA - gN

    return np.array(
        [
            mean_all,
            std_all,
            mean_A,
            mean_N,
            std_A,
            std_N,
            eA,
            eN,
            diff_e,
            ratio_e,
            max_mean,
            max_std,
            max_A,
            max_N,
            max_diff,
            occ_A,
            occ_N,
            occ_diff,
            gA,
            gN,
            gdiff,
        ],
        dtype=np.float32,
    )


def _id_to_path(root_dir: str, _id: str) -> str:
    return os.path.join(root_dir, _id[0], f"{_id}.npy")


def _cache_paths(prefix: str):
    base = "/kaggle/working"
    return (
        os.path.join(base, f"{prefix}_X.npy"),
        os.path.join(base, f"{prefix}_ids_fingerprint.txt"),
    )


def _ids_fingerprint(ids) -> str:
    joined = ("\n".join(ids) + "\n").encode("utf-8")
    return hashlib.sha256(joined).hexdigest()


def _load_or_compute_features(
    ids, root_dir: str, cache_prefix: str, n_features: int
) -> np.ndarray:
    X_path, fp_path = _cache_paths(cache_prefix)
    ids = [str(x) for x in ids]
    fp = _ids_fingerprint(ids)

    if os.path.exists(X_path) and os.path.exists(fp_path):
        try:
            with open(fp_path, "r", encoding="utf-8") as f:
                cached_fp = f.read().strip()
            if cached_fp == fp:
                X_loaded = np.load(X_path, allow_pickle=False)
                if X_loaded.ndim == 2 and X_loaded.shape[1] == n_features:
                    return X_loaded
        except Exception:
            pass

    X = np.zeros((len(ids), n_features), dtype=np.float32)

    def _compute_one(i_id):
        i, _id = i_id
        p = _id_to_path(root_dir, _id)
        if not os.path.exists(p):
            return i, None
        arr = np.load(p, mmap_mode="r", allow_pickle=False)
        feat = _extract_features_from_array(arr)
        return i, feat

    max_workers = min(32, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, feat in ex.map(_compute_one, enumerate(ids), chunksize=64):
            if feat is not None:
                X[i] = feat

    np.save(X_path, X, allow_pickle=False)
    with open(fp_path, "w", encoding="utf-8") as f:
        f.write(fp)
    return X


def _train_and_predict_submission() -> pd.DataFrame:
    labels = pd.read_csv(LABELS_PATH)
    sample = pd.read_csv(SAMPLE_SUB_PATH)

    train_ids = labels["id"].astype(str).tolist()
    y = labels["target"].values.astype(np.int32)

    n_features = 21
    X = _load_or_compute_features(
        train_ids, TRAIN_DIR, cache_prefix="train_features_v2", n_features=n_features
    )

    test_ids = sample["id"].astype(str).tolist()
    X_test = _load_or_compute_features(
        test_ids, TEST_DIR, cache_prefix="test_features_v2", n_features=n_features
    )

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xts = scaler.transform(X_test)

    Cs = [0.25, 1.0, 4.0]
    proba_accum = np.zeros(len(test_ids), dtype=np.float64)

    for C in Cs:
        clf = LogisticRegression(
            solver="liblinear",
            max_iter=200,
            C=float(C),
            random_state=0,
        )
        clf.fit(Xs, y)
        proba_accum += clf.predict_proba(Xts)[:, 1].astype(np.float64)

    proba = (proba_accum / len(Cs)).astype(np.float32)
    proba = np.clip(proba, 1e-6, 1 - 1e-6)

    return pd.DataFrame({"id": test_ids, "target": proba})


_HEURISTIC_SUBMISSION_CACHE = None


def _safe_read_submission(path: str) -> pd.DataFrame:
    global _HEURISTIC_SUBMISSION_CACHE
    if os.path.exists(path):
        df = pd.read_csv(path)
        df = df[["id", "target"]].copy()
        df["id"] = df["id"].astype(str)
        df["target"] = df["target"].astype(np.float32)
        return df

    if _HEURISTIC_SUBMISSION_CACHE is None:
        _HEURISTIC_SUBMISSION_CACHE = _train_and_predict_submission()

    return _HEURISTIC_SUBMISSION_CACHE.copy()


data1 = _safe_read_submission(
    "../input/rerun-seti-e-t-volo-d1-baseline-inference/submission.csv"
)
data2 = _safe_read_submission("../input/seti-bl-spatial-info-tf-tpu/submission.csv")
data3 = _safe_read_submission("../input/seti-bl-tf-starter-tpu/submission.csv")
data4 = _safe_read_submission("../input/seti-learned-image-resizing/submission.csv")
data5 = _safe_read_submission(
    "../input/lb-0-980-efficientnet-b0-more-epoch/submission.csv"
)
data6 = _safe_read_submission(
    "../input/inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv"
)



## === cell 1
data1.head()



## === cell 2
data2.head()




## === cell 3
def _align_on_id(base: pd.DataFrame, other: pd.DataFrame) -> np.ndarray:
    m = other.set_index("id")["target"]
    aligned = base["id"].map(m).astype(np.float32)
    fill_value = np.float32(other["target"].astype(np.float32).mean())
    return aligned.fillna(fill_value).values


base = data1[["id"]].copy()
t5 = _align_on_id(base, data5)
t4 = _align_on_id(base, data4)
t6 = _align_on_id(base, data6)
t2 = _align_on_id(base, data2)
t3 = _align_on_id(base, data3)

data6 = base.copy()
data6["target"] = (0.75 * t5 + 0.125 * t4 + 0.125 * t6 + 0.00 * t2 + 0.0 * t3).astype(
    np.float32
)



## === cell 4
sample = pd.read_csv(SAMPLE_SUB_PATH)
data6 = sample[["id"]].merge(data6, on="id", how="left")
data6["target"] = data6["target"].astype(np.float32).fillna(0.5).clip(1e-6, 1 - 1e-6)

data6.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", data6.shape)
print(data6.head())
