# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.7626591465113453

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your notebook fails because it tries to read other Kaggle notebooks’ `../input/.../submission.csv` files that are not available in this environment, so none of the `data1..data7` DataFrames exist and the later blend code crashes. To keep the “blend multiple submissions” core idea intact but make it runnable, I load whatever submission-like CSVs exist under `/kaggle/input/` and `/kaggle/data/`, validate they match the required `id,target` format, and then perform the same weighted blend (renormalized if some files are missing). If none are found, it falls back to a safe baseline that outputs 0.5 for every test id using `sample_submission.csv`, ensuring a valid `submission.csv` is always produced. This is score-neutral relative to “not yielded” and allow you to submit; if you later add the missing input submissions, the blend automatically use them.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score comes from the fallback path (no real model predictions found), which effectively outputs constant probabilities; AUC then collapses to ~0.5. To move toward the target with minimal change while keeping the “blend multiple submissions” idea, I add a lightweight, fully-local backup model that generates a valid probability per test `id` from the provided `.npy` cadence snippets using simple aggregate statistics and a logistic regression (scikit-learn). This backup is only used when no external submission-like CSVs are found, so it won’t interfere if you later provide real submissions to blend. The submission format, paths, and overall pipeline (blend if available, else fallback) remain the same, but now the fallback is meaningfully predictive and should increase AUC toward the target.'
- What this solution (achieved 0.50246) has done: 'The timeout is almost certainly coming from the fallback path that featurizes **all** train and test `.npy` files using a `ProcessPoolExecutor`, which is dominated by disk I/O + per-file numpy work and also pays heavy multiprocessing overhead. I keep the exact same feature definitions and LogisticRegression training, but speed it up by (1) eliminating the expensive process-based parallelism (which re-reads files in separate processes and adds IPC overhead) in favor of an efficient thread-based loader (NumPy releases the GIL in heavy ops and this workload is I/O bound), (2) reducing Python-loop overhead by batching path construction and using `os.path.join` minimally, and (3) ensuring BLAS threads stay pinned to 1 **before** importing NumPy/sklearn to avoid CPU oversubscription. These changes preserve evaluation semantics and produce the same kind of predictions (up to negligible FP differences), just faster and with much less overhead.'
- What this solution (achieved 0.50448) has done: 'Your current 0.50246 indicates the fallback model is running but is essentially non-informative; the most likely cause is that the train/test `.npy` path mapping is wrong (your `_id_to_npy_path` uses only the first character, but these datasets are sharded by the first hex digit into `0..f` folders, so many ids won’t be found, leading to failures/degenerate predictions). I fix the shard logic to use `id_str[0].lower()` and add a safe, deterministic fallback when a file is missing (so the run completes and still produces a valid CSV). To move AUC upward toward your 0.7627 target with minimal semantic change, I also standardize features (same features, just scaled) before LogisticRegression—this typically improves LR performance without changing the “simple aggregate-statistics + LR” core approach. Finally, I keep the blend behavior unchanged: if any valid external submissions are found, we blend; otherwise we run the improved local fallback.'
- What this solution (achieved 0.49869) has done: 'Your current score is still ~0.5 because the fallback model is producing near-constant / non-informative predictions; the biggest likely cause is that many `.npy` files are not being found (wrong shard mapping) and missing files are silently turned into all-zero features, which collapses AUC. I make one minimal, directly-relevant fix: correct the shard logic to match the dataset layout (folders `0..15`, not hex `0..f`), and add a fast existence check so we don’t waste time trying to mmap missing files. To move performance upward toward your target without changing the “simple aggregate stats + StandardScaler + LogisticRegression” core logic, I also set `class_weight="balanced"` (common for this imbalanced dataset) and increase `max_iter` slightly to ensure proper convergence. Everything else (feature definitions, training approach, blending behavior, output format/path) stays the same and the script still writes `submission.csv`.'
- What this solution (achieved 0.50665) has done: 'Your fallback model is still scoring ~0.5 because it likely can’t find most `.npy` files: the dataset is sharded by the first hex digit into folders `0..f`, not by decimal `0..15`, so your current `_id_to_npy_path()` mapping sends many ids to non-existent paths and collapses features to zeros. I make the minimal fix to map shards directly to the first hex character (lowercased) and add a fast missing-rate printout to confirm we’re actually loading files. To nudge AUC upward toward the target without changing the “simple aggregate stats + StandardScaler + LogisticRegression” core logic, I also set `C` to a slightly less-regularized value (often helps LR on this feature set) while keeping solver/training approach the same. The blend behavior and submission format/path remain unchanged, and it still always write a valid `submission.csv`.'
- What this solution (achieved 0.51001) has done: 'Your current score is still ~0.5 because the fallback model is effectively non-informative on this competition; with such weak features, LR collapses toward constant predictions and AUC stays near random. Keeping the same core approach (simple aggregate statistics + StandardScaler + LogisticRegression, same training loop/solver), the smallest change that typically yields a real lift is to add a few more aggregate, panel-wise “A vs others” contrast features that better reflect the known signal structure (needles appear only in panels 0/2/4). I also set `C` back to the default (1.0) to avoid overshooting regularization on the expanded feature set, while keeping `class_weight="balanced"` and everything else intact. The blend logic, paths, and submission writing remain unchanged; this only strengthens the fallback so it can move AUC upward toward your 0.7627 target when no external submissions are found.'
- What this solution (achieved 0.51166) has done: 'Your current score (~0.51 AUC) indicates the fallback model is still only weakly informative; without changing the core “simple aggregate stats + StandardScaler + LogisticRegression” approach, the smallest reliable lift is to add a few more physically-relevant, low-cost aggregate features that capture the expected “drift line” structure (time/frequency gradients) and “A-panels only” behavior. I keep the same training loop/solver and keep blending behavior unchanged, but extend `_extract_features_from_npy()` with (1) per-panel time and frequency gradient energy, and (2) A-vs-O contrasts of those gradient energies. This preserves evaluation semantics, avoids heavy compute, and should move AUC upward toward your target while staying within the 600s constraint. The script still always write a valid `submission.csv` with `id,target`.'
- What this solution (achieved 0.5) has done: 'The timeout is overwhelmingly caused by the fallback path training on the full 54k train set with expensive per-file feature extraction (multiple full-array passes plus per-panel quantiles) and then fitting LogisticRegression, all under single-threaded BLAS. The core logic (same features + StandardScaler + LogisticRegression, and same blending behavior when submissions exist) is preserved, but we remove unnecessary filesystem checks, avoid repeated `os.path.exists` calls, speed up quantile/median computation with `np.partition` (order-statistics; equivalent selection, negligible FP diffs), and parallelize feature extraction safely with multiprocessing (deterministic; no randomness in featurization). We also reduce Python overhead by precomputing paths and batching loads, while keeping all I/O paths and model hyperparameters unchanged. If external submission-like CSVs are found, the fast blending path remains essentially unchanged and finish quickly.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import glob
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression




## === cell 1
def _read_submission_csv(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    cols = [c.strip() for c in df.columns.tolist()]
    if "id" not in cols or "target" not in cols:
        return None
    df = df[["id", "target"]].copy()
    df["id"] = df["id"].astype(str)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    if df["target"].isna().any():
        return None
    return df


SEARCH_ROOTS = [
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/working",
]


def _candidate_submission_paths() -> list[str]:
    paths = []
    for root in SEARCH_ROOTS:
        if not os.path.isdir(root):
            continue
        direct = os.path.join(root, "submission.csv")
        if os.path.exists(direct):
            paths.append(direct)
        nested = os.path.join(root, "seti-breakthrough-listen", "submission.csv")
        if os.path.exists(nested):
            paths.append(nested)

        try:
            for name in os.listdir(root):
                p = os.path.join(root, name, "submission.csv")
                if os.path.exists(p):
                    paths.append(p)
        except Exception:
            pass

    if not paths:
        for root in SEARCH_ROOTS:
            if os.path.isdir(root):
                paths.extend(
                    glob.glob(
                        os.path.join(
                            root, "seti-breakthrough-listen", "**", "submission.csv"
                        ),
                        recursive=True,
                    )
                )
    return sorted(set(paths))


candidate_paths = _candidate_submission_paths()

loaded = []
loaded_paths = []
for p in candidate_paths:
    df = _read_submission_csv(p)
    if df is not None:
        loaded.append(df)
        loaded_paths.append(p)

data1 = data2 = data3 = data4 = data5 = data6 = data7 = None

for i, df in enumerate(loaded[:7], start=1):
    locals()[f"data{i}"] = df

print(f"Found {len(loaded)} valid submission-like CSV(s). Using up to 7 for blending.")
for i, p in enumerate(loaded_paths[:7], start=1):
    print(f"data{i} <- {p}")

sample_path = None
for root in SEARCH_ROOTS:
    p = os.path.join(root, "sample_submission.csv")
    if os.path.exists(p):
        sample_path = p
        break
    p2 = os.path.join(root, "seti-breakthrough-listen", "sample_submission.csv")
    if os.path.exists(p2):
        sample_path = p2
        break

if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv under expected Kaggle paths."
    )

sample = pd.read_csv(sample_path)[["id", "target"]].copy()
sample["id"] = sample["id"].astype(str)



## === cell 2
if data1 is not None:
    data11 = data1.copy()
else:
    data11 = sample.copy()

data11 = data11.merge(sample[["id"]], on="id", how="right")



## === cell 3
from sklearn.preprocessing import StandardScaler


def _find_data_root_with_subdirs(name: str) -> str | None:
    """Find a root that contains `{name}/0`, `{name}/1`, ... subdirs."""
    for root in SEARCH_ROOTS:
        p = os.path.join(root, name)
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, "0")):
            return p
        p2 = os.path.join(root, "seti-breakthrough-listen", name)
        if os.path.isdir(p2) and os.path.isdir(os.path.join(p2, "0")):
            return p2
    return None


def _id_to_npy_path(id_str: str, base_dir: str) -> str:
    s = str(id_str).strip()
    shard = s[0].lower()  # "0".."9","a".."f"
    return os.path.join(base_dir, shard, f"{s}.npy")


def _panel_order_stats(
    flat_1d: np.ndarray,
) -> tuple[np.float32, np.float32, np.float32]:
    n = flat_1d.size
    if n == 0:
        z = np.float32(0.0)
        return z, z, z

    k_med = n // 2
    k10 = int(0.10 * (n - 1))
    k90 = int(0.90 * (n - 1))

    kth = np.array([k10, k_med, k90], dtype=np.int64)
    part = np.partition(flat_1d, kth)
    q10 = np.float32(part[k10])
    med = np.float32(part[k_med])
    q90 = np.float32(part[k90])
    return med, q10, q90


def _extract_features_from_npy(arr: np.ndarray) -> np.ndarray:
    x = np.asarray(arr, dtype=np.float32)  # (6, 273, 256)

    means = x.mean(axis=(1, 2))
    stds = x.std(axis=(1, 2))
    maxs = x.max(axis=(1, 2))
    mins = x.min(axis=(1, 2))

    energies = np.mean(x * x, axis=(1, 2)).astype(np.float32)

    dt = x[:, 1:, :] - x[:, :-1, :]
    df = x[:, :, 1:] - x[:, :, :-1]
    grad_t_energy = np.mean(dt * dt, axis=(1, 2)).astype(np.float32)
    grad_f_energy = np.mean(df * df, axis=(1, 2)).astype(np.float32)

    flat = x.reshape(6, -1)
    medians = np.empty((6,), dtype=np.float32)
    q10 = np.empty((6,), dtype=np.float32)
    q90 = np.empty((6,), dtype=np.float32)
    for i in range(6):
        med, qq10, qq90 = _panel_order_stats(flat[i])
        medians[i] = med
        q10[i] = qq10
        q90[i] = qq90

    A_idx = np.array([0, 2, 4])
    O_idx = np.array([1, 3, 5])

    d_mean = float(means[A_idx].mean() - means[O_idx].mean())
    d_std = float(stds[A_idx].mean() - stds[O_idx].mean())
    d_max = float(maxs[A_idx].max() - maxs[O_idx].max())
    d_min = float(mins[A_idx].min() - mins[O_idx].min())

    A_energy = float(energies[A_idx].mean())
    O_energy = float(energies[O_idx].mean())
    d_energy = float(A_energy - O_energy)

    d_median = float(medians[A_idx].mean() - medians[O_idx].mean())

    A_q10 = float(q10[A_idx].mean())
    O_q10 = float(q10[O_idx].mean())
    d_q10 = float(A_q10 - O_q10)

    A_q90 = float(q90[A_idx].mean())
    O_q90 = float(q90[O_idx].mean())
    d_q90 = float(A_q90 - O_q90)

    A_dt = dt[A_idx]
    O_dt = dt[O_idx]
    A_df = df[A_idx]
    O_df = df[O_idx]
    A_grad_t = float(np.mean(A_dt * A_dt))
    O_grad_t = float(np.mean(O_dt * O_dt))
    d_grad_t = float(A_grad_t - O_grad_t)

    A_grad_f = float(np.mean(A_df * A_df))
    O_grad_f = float(np.mean(O_df * O_df))
    d_grad_f = float(A_grad_f - O_grad_f)

    A_means = means[A_idx]
    O_means = means[O_idx]
    A_mean_pairdiff = float(
        (
            abs(A_means[0] - A_means[1])
            + abs(A_means[0] - A_means[2])
            + abs(A_means[1] - A_means[2])
        )
        / 3.0
    )
    A_vs_O_mean_diff = float(abs(A_means.mean() - O_means.mean()))
    A_consistency_score = float(A_vs_O_mean_diff - A_mean_pairdiff)

    feat = np.concatenate(
        [
            means,
            stds,
            maxs,
            mins,
            medians,
            energies,
            grad_t_energy,
            grad_f_energy,
            q10,
            q90,
            np.array(
                [
                    d_mean,
                    d_std,
                    d_max,
                    d_min,
                    A_energy,
                    O_energy,
                    d_energy,
                    d_median,
                    A_q10,
                    O_q10,
                    d_q10,
                    A_q90,
                    O_q90,
                    d_q90,
                    A_grad_t,
                    O_grad_t,
                    d_grad_t,
                    A_grad_f,
                    O_grad_f,
                    d_grad_f,
                    A_mean_pairdiff,
                    A_vs_O_mean_diff,
                    A_consistency_score,
                ],
                dtype=np.float32,
            ),
        ]
    )
    return feat.astype(np.float32)


_DUMMY = np.zeros((6, 273, 256), dtype=np.float32)
_N_FEATS = int(_extract_features_from_npy(_DUMMY).shape[0])


def _featurize_path(path: str) -> np.ndarray:
    try:
        arr = np.load(path, mmap_mode="r")
        return _extract_features_from_npy(arr)
    except Exception:
        return np.zeros((_N_FEATS,), dtype=np.float32)


def _build_features_for_ids(
    ids: np.ndarray, base_dir: str, n_jobs: int | None = None
) -> np.ndarray:
    from concurrent.futures import ProcessPoolExecutor

    n = len(ids)
    X = np.empty((n, _N_FEATS), dtype=np.float32)

    paths = [_id_to_npy_path(s, base_dir) for s in ids]

    def _work(p: str) -> np.ndarray:
        if os.path.exists(p):
            return _featurize_path(p)
        return np.zeros((_N_FEATS,), dtype=np.float32)

    if n_jobs is None:
        n_jobs = min(4, (os.cpu_count() or 2))

    if n_jobs <= 1:
        for i, p in enumerate(paths):
            X[i] = _work(p)
        return X

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for i, feat in enumerate(ex.map(_work, paths, chunksize=64)):
            X[i] = feat
    return X


def _missing_rate(ids: np.ndarray, base_dir: str, max_check: int = 2000) -> float:
    n = min(len(ids), max_check)
    if n == 0:
        return 1.0
    miss = 0
    for s in ids[:n]:
        if not os.path.exists(_id_to_npy_path(s, base_dir)):
            miss += 1
    return miss / n


def _train_predict_fallback(sample_df: pd.DataFrame) -> np.ndarray:
    """
    Train a simple model on train/ + train_labels.csv and predict on test/.
    Only used when no external submission-like CSVs were found.
    """
    train_dir = _find_data_root_with_subdirs("train")
    test_dir = _find_data_root_with_subdirs("test")

    labels_path = None
    for root in SEARCH_ROOTS:
        p = os.path.join(root, "train_labels.csv")
        if os.path.exists(p):
            labels_path = p
            break
        p2 = os.path.join(root, "seti-breakthrough-listen", "train_labels.csv")
        if os.path.exists(p2):
            labels_path = p2
            break

    if train_dir is None or test_dir is None or labels_path is None:
        return np.full(len(sample_df), 0.5, dtype=np.float64)

    ydf = pd.read_csv(labels_path, usecols=["id", "target"])
    ydf["id"] = ydf["id"].astype(str)
    y = ydf["target"].to_numpy(dtype=np.int64)

    train_ids = ydf["id"].to_numpy()
    test_ids = sample_df["id"].to_numpy()

    print("Diagnostic missing-rate (lower is better):")
    print("  train missing rate ~", round(_missing_rate(train_ids, train_dir), 4))
    print("  test  missing rate ~", round(_missing_rate(test_ids, test_dir), 4))
    print("  feature dimension  =", _N_FEATS)

    n_jobs = min(4, (os.cpu_count() or 2))
    X_train = _build_features_for_ids(train_ids, train_dir, n_jobs=n_jobs)
    X_test = _build_features_for_ids(test_ids, test_dir, n_jobs=n_jobs)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=800,
        n_jobs=None,
        random_state=0,
        class_weight="balanced",
        C=1.0,
    )
    clf.fit(X_train_s, y)
    p = clf.predict_proba(X_test_s)[:, 1].astype(np.float64)
    return np.clip(p, 0.0, 1.0)


weights = {
    "data1": 0.12,
    "data2": 0.12,
    "data3": 0.12,
    "data4": 0.12,
    "data5": 0.15,
    "data6": 0.70,
}

sources = {}
for name in ["data1", "data2", "data3", "data4", "data5", "data6"]:
    df = locals().get(name, None)
    if df is None:
        continue
    aligned = sample[["id"]].merge(df[["id", "target"]], on="id", how="left")
    if aligned["target"].isna().any():
        aligned["target"] = aligned["target"].fillna(0.5)
    sources[name] = aligned["target"].to_numpy(dtype=np.float64)

if len(sources) == 0:
    data11["target"] = _train_predict_fallback(sample)
else:
    w = np.array([weights.get(k, 1.0) for k in sources.keys()], dtype=np.float64)
    w_sum = float(w.sum())
    if w_sum <= 0:
        w = np.ones_like(w)
        w_sum = float(w.sum())
    w = w / w_sum
    preds = np.zeros(len(sample), dtype=np.float64)
    for (k, arr), wk in zip(sources.items(), w):
        preds += wk * arr
    preds = np.clip(preds, 0.0, 1.0)
    data11["target"] = preds

data11 = data11[["id", "target"]].copy()



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
AttributeError: Can't pickle local object '_build_features_for_ids.<locals>._work'
"""

The above exception was the direct cause of the following exception:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2804884571.py in <cell line: 0>()
    296 
    297 if len(sources) == 0:
--> 298     data11["target"] = _train_predict_fallback(sample)
    299 else:
    300     w = np.array([weights.get(k, 1.0) for k in sources.keys()], dtype=np.float64)

/tmp/ipykernel_11/2804884571.py in _train_predict_fallback(sample_df)
    256 
    257     n_jobs = min(4, (os.cpu_count() or 2))
--> 258     X_train = _build_features_for_ids(train_ids, train_dir, n_jobs=n_jobs)
    259     X_test = _build_features_for_ids(test_ids, test_dir, n_jobs=n_jobs)
    260 

/tmp/ipykernel_11/2804884571.py in _build_features_for_ids(ids, base_dir, n_jobs)
    203 
    204     with ProcessPoolExecutor(max_workers=n_jobs) as ex:
--> 205         for i, feat in enumerate(ex.map(_work, paths, chunksize=64)):
    206             X[i] = feat
    207     return X

/usr/lib/python3.11/concurrent/futures/process.py in _chain_from_iterable_of_lists(iterable)
    618     careful not to keep references to yielded objects.
    619     """
--> 620     for element in iterable:
    621         element.reverse()
    622         while element:

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/multiprocessing/queues.py in _feed(buffer, notempty, send_bytes, writelock, reader_close, writer_close, ignore_epipe, onerror, queue_sem)
    242 
    243                         # serialize the data before acquiring the lock
--> 244                         obj = _ForkingPickler.dumps(obj)
    245                         if wacquire is None:
    246                             send_bytes(obj)

/usr/lib/python3.11/multiprocessing/reduction.py in dumps(cls, obj, protocol)
     49     def dumps(cls, obj, protocol=None):
     50         buf = io.BytesIO()
---> 51         cls(buf, protocol).dump(obj)
     52         return buf.getbuffer()
     53 

AttributeError: Can't pickle local object '_build_features_for_ids.<locals>._work'

## === cell 4
data11.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", data11.shape)
print(data11.head())
print("target summary:", data11["target"].describe())
