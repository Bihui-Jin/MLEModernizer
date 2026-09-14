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

0.7627043135520268

# 6. Current score

0.48505

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your notebook fails because it tries to read multiple external Kaggle Dataset submissions that are not present in your environment, so none of the blend inputs exist and the pipeline never writes a submission. To keep the core “blend submissions” logic intact while making it runnable end-to-end, I add a small loader that (a) uses those files if they exist, otherwise (b) falls back to a simple, deterministic baseline prediction built from the provided `sample_submission.csv`. The blend then operate on whatever inputs are available, always producing a valid `submission.csv` with `id,target` and probabilities clipped to `[0,1]`. This is score-neutral relative to “no submission,” and provides a consistent baseline while preserving the ensemble-weighting semantics.'
- What this solution (achieved 0.5) has done: 'The timeout is dominated by the fallback path: extracting features by loading thousands of `.npy` files (train + test) and repeatedly computing `p99` via `np.partition` per file. To keep identical semantics while reducing wall time, I (1) compute p99 using a single `np.partition` call, (2) avoid copying/`astype` and compute stats on the memory-mapped float16 directly (NumPy reductions already upcast to float32/float64), (3) reduce overhead in multiprocessing by batching paths and returning stacked arrays, and (4) make caching keys include the dataset root so cached features are reliably reused across runs. The ensemble logic is unchanged; only the fallback feature extraction/training/inference path is optimized.'
- What this solution (achieved 0.5) has done: 'The timeout is almost certainly caused by the fallback path doing heavy per-file feature extraction over thousands of `.npy` files, where each file computes a 99th percentile via `np.partition` on ~420k elements—this dominates runtime even with multiprocessing. I keep the exact same features and model, but make them faster by computing the percentile in-place on a reusable float32 buffer (avoids repeated large allocations) and by using `np.quantile(..., method="linear")`, which is mathematically equivalent to your current interpolation logic. I also avoid the expensive multiprocessing “spawn” overhead by switching to “fork” when available (Linux), and speed up ID→prediction alignment using a single merge-based path rather than repeated `Series.map`. All I/O paths and the ensemble weighting remain unchanged.'
- What this solution (achieved 0.48459) has done: 'I fix the crash in the fallback model training by ensuring the extracted feature matrix contains only finite values (replace NaN/Inf and clip extreme magnitudes) before fitting LogisticRegression; this addresses the “infinity or too large” error. I also make feature extraction itself more robust by guarding against NaNs/Infs from reductions/quantiles and returning safe finite features. With fallback now working, the downstream blend variables (t1–t7) be defined so the NameError in the blending cell disappears. These changes preserve the existing “blend submissions or fallback” logic while producing a valid `submission.csv` and should improve score above the degenerate 0.5 baseline by enabling non-constant predictions.'
- What this solution (achieved 0.48505) has done: 'Your current score (0.48459) is far below the target (0.7627), so we should improve the fallback model a bit while keeping the same “simple feature extraction + LogisticRegression + optional blending” core logic. The main minimal win is to stop throwing away most training data via `np.linspace` subsampling and instead use a deterministic stratified subsample (still capped at 12000) so the classifier sees a balanced, more representative set and improves AUC. I also standardize the 7 numeric features before LogisticRegression (same model family and training approach; just better-conditioned inputs), which typically yields a noticeable AUC gain for linear models without changing semantics. Finally, I only blend in external submissions if they actually exist; otherwise the code effectively uses the improved fallback predictions (instead of blending multiple identical fallbacks).'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)

INPUT_BASES = [
    "../input",  # typical Kaggle notebooks
    "/kaggle/input",  # some environments
    "/kaggle/data/input",  # as provided in this environment tree
]


def _find_existing_path(rel_path: str):
    for base in INPUT_BASES:
        p = os.path.join(base, rel_path)
        if os.path.exists(p):
            return p
    return None




## === cell 1
from sklearn.linear_model import LogisticRegression


def _first_existing_dir(rel_dir: str):
    for base in INPUT_BASES:
        p = os.path.join(base, rel_dir)
        if os.path.isdir(p):
            return p
    return None


_HEX2INT = {c: i for i, c in enumerate("0123456789abcdef")}


def _resolve_id_to_npy_path(root_dir: str, _id: str) -> str:
    if not _id:
        return None
    c = _id[0].lower()
    if c in _HEX2INT:
        p = os.path.join(root_dir, str(_HEX2INT[c]), f"{_id}.npy")
        if os.path.exists(p):
            return p
    p2 = os.path.join(root_dir, f"{_id}.npy")
    if os.path.exists(p2):
        return p2
    return None


_P99_Q = 0.99
_P99_BUF = None
_P99_BUF_SIZE = 0


def _finite_or_default(v, default=0.0) -> np.float32:
    v = np.float32(v)
    if not np.isfinite(v):
        return np.float32(default)
    return v


def _p99_equivalent_from_x(x: np.ndarray) -> np.float32:
    global _P99_BUF, _P99_BUF_SIZE
    n = x.size
    if n == 0:
        return np.float32(0.0)

    if _P99_BUF is None or _P99_BUF_SIZE < n:
        _P99_BUF = np.empty(n, dtype=np.float32)
        _P99_BUF_SIZE = n

    np.copyto(_P99_BUF[:n], x.ravel(), casting="unsafe")

    q = np.quantile(_P99_BUF[:n], _P99_Q, method="linear")
    return _finite_or_default(q, default=0.0)


def _extract_features_from_npy(path: str) -> np.ndarray:
    x = np.load(path, mmap_mode="r")

    mu = _finite_or_default(x.mean(), default=0.0)
    sd = _finite_or_default(x.std(), default=0.0)

    on = _finite_or_default(x[[0, 2, 4]].mean(), default=0.0)
    off = _finite_or_default(x[[1, 3, 5]].mean(), default=0.0)
    contrast = _finite_or_default(on - off, default=0.0)

    mx = _finite_or_default(x.max(), default=0.0)
    p99 = _p99_equivalent_from_x(x)

    t_var = _finite_or_default(x.mean(axis=2).var(), default=0.0)
    f_var = _finite_or_default(x.mean(axis=1).var(), default=0.0)

    feats = np.array([mu, sd, contrast, mx, p99, t_var, f_var], dtype=np.float32)

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats = np.clip(feats, -1e6, 1e6).astype(np.float32, copy=False)
    return feats


def _feature_cache_path(tag: str) -> str:
    os.makedirs("/kaggle/working/feat_cache", exist_ok=True)
    return f"/kaggle/working/feat_cache/{tag}.npz"


def _worker_batch_extract(batch_paths):
    out = np.empty((len(batch_paths), 7), dtype=np.float32)
    for j, p in enumerate(batch_paths):
        out[j] = _extract_features_from_npy(p)
    return out


def _extract_feature_matrix(
    paths, n_workers: int = 0, chunksize: int = 256, batch_size: int = 256
):
    n = len(paths)
    feats = np.empty((n, 7), dtype=np.float32)
    if n == 0:
        return feats

    if n_workers and n_workers > 1:
        import multiprocessing as mp

        batches = [paths[i : i + batch_size] for i in range(0, n, batch_size)]

        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        ctx = mp.get_context(start_method)
        with ctx.Pool(processes=n_workers) as pool:
            k = 0
            for arr in pool.imap(
                _worker_batch_extract,
                batches,
                chunksize=max(1, chunksize // batch_size),
            ):
                m = arr.shape[0]
                feats[k : k + m] = arr
                k += m
    else:
        for i, p in enumerate(paths):
            feats[i] = _extract_features_from_npy(p)

    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    feats = np.clip(feats, -1e6, 1e6).astype(np.float32, copy=False)
    return feats


def _standardize_train_test(X_train: np.ndarray, X_test: np.ndarray):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0).astype(np.float32, copy=False)
    Xtr = ((X_train - mu) / sd).astype(np.float32, copy=False)
    Xte = ((X_test - mu) / sd).astype(np.float32, copy=False)
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)
    Xtr = np.clip(Xtr, -50.0, 50.0)
    Xte = np.clip(Xte, -50.0, 50.0)
    return Xtr, Xte


def build_fallback_predictions(sample_sub: pd.DataFrame) -> pd.DataFrame:
    train_labels_path = _find_existing_path("train_labels.csv") or _find_existing_path(
        "seti-breakthrough-listen/train_labels.csv"
    )
    train_dir = _first_existing_dir("train") or _first_existing_dir(
        "seti-breakthrough-listen/train"
    )
    test_dir = _first_existing_dir("test") or _first_existing_dir(
        "seti-breakthrough-listen/test"
    )
    if train_labels_path is None or train_dir is None or test_dir is None:
        df = sample_sub.copy()
        df["target"] = 0.5
        return df

    labels = pd.read_csv(train_labels_path, usecols=["id", "target"]).copy()
    train_ids_all = labels["id"].astype(str).to_numpy()
    y_all = labels["target"].astype(np.int32).to_numpy()

    max_train = 12000  # unchanged cap
    if train_ids_all.shape[0] > max_train:
        rng = np.random.RandomState(0)
        pos_idx = np.flatnonzero(y_all == 1)
        neg_idx = np.flatnonzero(y_all == 0)
        n_pos = min(len(pos_idx), max_train // 2)
        n_neg = min(len(neg_idx), max_train - n_pos)
        if n_pos > 0 and n_neg > 0:
            pick_pos = rng.choice(pos_idx, size=n_pos, replace=False)
            pick_neg = rng.choice(neg_idx, size=n_neg, replace=False)
            idx = np.concatenate([pick_pos, pick_neg])
            rng.shuffle(idx)
        else:
            idx = rng.choice(np.arange(len(y_all)), size=max_train, replace=False)
        train_ids = train_ids_all[idx]
        y_train = y_all[idx]
    else:
        train_ids = train_ids_all
        y_train = y_all

    train_paths = []
    y_keep = []
    for _id, y in zip(train_ids, y_train):
        p = _resolve_id_to_npy_path(train_dir, _id)
        if p is None:
            continue
        train_paths.append(p)
        y_keep.append(int(y))
    y_keep = np.asarray(y_keep, dtype=np.int32)

    if len(train_paths) < 100 or np.unique(y_keep).size < 2:
        df = sample_sub.copy()
        df["target"] = 0.5
        return df

    cpu_cnt = os.cpu_count() or 2
    n_workers = max(1, min(8, cpu_cnt))  # unchanged cap intent
    chunksize = 256
    batch_size = 256

    train_cache = _feature_cache_path(
        f"train_{os.path.basename(train_dir)}_{len(train_paths)}"
    )
    if os.path.exists(train_cache):
        cached = np.load(train_cache)
        X_train = cached["X"]
        if X_train.shape != (len(train_paths), 7):
            X_train = _extract_feature_matrix(
                train_paths,
                n_workers=n_workers,
                chunksize=chunksize,
                batch_size=batch_size,
            )
            np.savez_compressed(train_cache, X=X_train)
    else:
        X_train = _extract_feature_matrix(
            train_paths, n_workers=n_workers, chunksize=chunksize, batch_size=batch_size
        )
        np.savez_compressed(train_cache, X=X_train)

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_train = np.clip(X_train, -1e6, 1e6).astype(np.float32, copy=False)

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=300,
        n_jobs=1,
        class_weight="balanced",
        random_state=0,
    )

    ids_list = sample_sub["id"].astype(str).to_numpy()

    test_paths = []
    exist_indices = []
    for i, _id in enumerate(ids_list):
        p = _resolve_id_to_npy_path(test_dir, _id)
        if p is not None:
            exist_indices.append(i)
            test_paths.append(p)

    preds = np.full((len(ids_list),), 0.5, dtype=np.float64)

    if test_paths:
        test_cache = _feature_cache_path(
            f"test_{os.path.basename(test_dir)}_{len(test_paths)}"
        )
        if os.path.exists(test_cache):
            cached = np.load(test_cache)
            X_test = cached["X"]
            if X_test.shape != (len(test_paths), 7):
                X_test = _extract_feature_matrix(
                    test_paths,
                    n_workers=n_workers,
                    chunksize=chunksize,
                    batch_size=batch_size,
                )
                np.savez_compressed(test_cache, X=X_test)
        else:
            X_test = _extract_feature_matrix(
                test_paths,
                n_workers=n_workers,
                chunksize=chunksize,
                batch_size=batch_size,
            )
            np.savez_compressed(test_cache, X=X_test)

        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.clip(X_test, -1e6, 1e6).astype(np.float32, copy=False)

        X_train_std, X_test_std = _standardize_train_test(X_train, X_test)

        clf.fit(X_train_std, y_keep)
        proba = clf.predict_proba(X_test_std)[:, 1].astype(np.float64, copy=False)
        preds[np.asarray(exist_indices, dtype=np.int64)] = proba
    else:
        X_train_std, _ = _standardize_train_test(X_train, X_train)
        clf.fit(X_train_std, y_keep)

    df = sample_sub.copy()
    df["target"] = preds
    df["target"] = (
        df["target"].replace([np.inf, -np.inf], np.nan).fillna(0.5).clip(0.0, 1.0)
    )
    return df


def load_submission_or_fallback(
    rel_path: str, fallback_df: pd.DataFrame
) -> pd.DataFrame:
    p = _find_existing_path(rel_path)
    if p is None:
        return fallback_df.copy()
    df = pd.read_csv(p)
    if "id" not in df.columns or "target" not in df.columns:
        raise ValueError(f"Submission file missing required columns id/target: {p}")
    df = df[["id", "target"]].copy()
    return df




## === cell 2
sample_path = _find_existing_path("sample_submission.csv") or _find_existing_path(
    "seti-breakthrough-listen/sample_submission.csv"
)
if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv under known input paths."
    )

sample_sub = pd.read_csv(sample_path, usecols=["id", "target"]).copy()

fallback = build_fallback_predictions(sample_sub)

data1 = load_submission_or_fallback(
    "rerun-seti-e-t-volo-d1-baseline-inference/submission.csv", fallback
)
data2 = load_submission_or_fallback(
    "lb-0-980-efficientnet-b0-more-epoch/submission.csv", fallback
)
data3 = load_submission_or_fallback(
    "inference-5x-ensemble-vanilla-resnet34d-seti/submission.csv", fallback
)
data4 = load_submission_or_fallback(
    "seti-learned-image-resizing/submission.csv", fallback
)
data5 = load_submission_or_fallback(
    "rerun-seti-e-t-resnet18d-baseline/submission.csv", fallback
)
data6 = load_submission_or_fallback(
    "ensemble-for-seti-competition/submission.csv", fallback
)
data7 = load_submission_or_fallback(
    "fixed-gradual-warmup-custom-head/submission.csv", fallback
)


def align_to_ids(df: pd.DataFrame, ids: pd.Series) -> np.ndarray:
    s = df[["id", "target"]].copy()
    s["id"] = s["id"].astype(str)
    s = s.drop_duplicates("id", keep="last").set_index("id")["target"]
    out = s.reindex(ids.astype(str), fill_value=0.5).to_numpy(dtype=float, copy=False)
    return out


ids = sample_sub["id"]
t1 = align_to_ids(data1, ids)
t2 = align_to_ids(data2, ids)
t3 = align_to_ids(data3, ids)
t4 = align_to_ids(data4, ids)
t5 = align_to_ids(data5, ids)
t6 = align_to_ids(data6, ids)
t7 = align_to_ids(data7, ids)




## === cell 3
data11 = sample_sub.copy()




## === cell 4
all_same = (
    np.allclose(t1, t6)
    and np.allclose(t2, t6)
    and np.allclose(t3, t6)
    and np.allclose(t4, t6)
    and np.allclose(t5, t6)
    and np.allclose(t7, t6)
)

if all_same:
    data11["target"] = t6
else:
    data11["target"] = (
        0.12 * t1 + 0.10 * t2 + 0.10 * t3 + 0.10 * t4 + 0.12 * t5 + 0.60 * t6
    )

data11["target"] = data11["target"].replace([np.inf, -np.inf], np.nan).fillna(0.5)
data11["target"] = data11["target"].clip(0.0, 1.0)




## === cell 5
data11.to_csv("submission.csv", index=False)
print(data11.head())
print(
    f"Wrote submission.csv with shape {data11.shape} and columns {list(data11.columns)}"
)
