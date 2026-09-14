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

0.7625534441480373

# 6. Current score

0.50506

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I remove the hard dependency on external Kaggle dataset paths that don’t exist in your environment (the cause of the FileNotFoundError) and replace it with a robust ensembling loader that uses any available `sample_submission.csv` plus any local `submission*.csv` files if present. If none of the external submissions are available, the notebook still run end-to-end by producing a valid baseline submission (constant 0.5 probabilities) in the required `id,target` format. I also fix the incorrect weighted-sum logic (weights summed to >1 and ignored `data7`) to a safe normalized-average blend when multiple submissions exist, while preserving the “blend submissions” core intent. Finally, I add strict validation to ensure ID alignment and correct output shape before writing `submission.csv`.'
- What this solution (achieved 0.49519) has done: 'Your current 0.5 score comes from emitting constant predictions when no other `submission*.csv` files are found, so the smallest legitimate improvement is to generate non-constant predictions directly from the provided training data and test snippets. To preserve “blend submissions” core intent, I keep your ensembling loader, but add a fallback that trains a simple sklearn LogisticRegression on lightweight, deterministic summary features extracted from each `(6,273,256)` snippet and then predicts probabilities for the test ids. This keeps the approach minimal (no deep nets, no new packages) and should move AUC up toward your target without relying on external datasets. The script still always writes a valid `submission.csv` with `id,target` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.49294) has done: 'Your current score (~0.495) suggests the fallback model is learning almost nothing; the most common cause here is that raw per-snippet summary statistics are not on a comparable scale across features, so LogisticRegression underfits badly. I keep your exact feature extraction and LogisticRegression core logic, but add a minimal `StandardScaler` fit on train features and applied to test features to make optimization well-conditioned. I also set `C` slightly higher and `max_iter` a bit larger to better fit without changing the model family, and add a quick in-script train AUC check to confirm the model is learning (no metric leakage, just a sanity check). The submission format and ID alignment remain unchanged.'
- What this solution (achieved 0.4959) has done: 'Your current score indicates the fallback model still isn’t capturing the key “A vs off-target” cadence pattern, so we minimally strengthen the *same* LogisticRegression approach by making features more cadence-aware without changing the training loop or model family. Specifically, we keep your existing per-panel summary stats, but add a small set of additional handcrafted features that measure (1) consistency across the three A panels vs off panels and (2) robust “excess in A” signals (median/quantiles and variability), which are commonly predictive for this competition. We also add `class_weight="balanced"` to address class imbalance (common in SETI) while keeping the same solver and objective, which typically improves AUC when the model was near-random. Everything else (paths, ensembling behavior, output format, and always writing `submission.csv`) remains unchanged.'
- What this solution (achieved 0.50132) has done: 'Your current AUC (~0.496) is essentially random, so we need a small but meaningful signal boost without changing the overall “extract lightweight features → StandardScaler → LogisticRegression → predict_proba” core logic. The minimal high-impact fix is to correct feature scaling/robustness by applying a log1p transform to strictly-nonnegative “energy-like” features and by adding a tiny epsilon-stabilized ratio feature for A vs off-target (these preserve semantics but better match the metric). I also add a deterministic stratified train/validation split AUC check (no effect on training/predictions) to confirm the model is actually learning something beyond random. Everything else (paths, ensembling behavior, submission alignment/format, and writing `submission.csv`) remains unchanged.'
- What this solution (achieved 0.49827) has done: 'Your current score (0.50132) is far below the target (0.76255), so we should improve the fallback model while keeping the same “handcrafted features → StandardScaler → LogisticRegression” core logic. The main minimal fix is to extract a few additional cadence-aware features that better reflect the competition’s key signal (energy/structure present in A panels but not in off-target panels), without changing the model family or training approach. Concretely, we add (1) A-minus-off **per-panel difference vectors** (mean/std/energy/absmean) and (2) simple **row/column projection** statistics to capture diagonal drift/line structure, both still lightweight summary features. Everything else (paths, ensembling behavior, scaling, LR training, submission alignment/format) remains unchanged and the script still always write a valid `submission.csv`.'
- What this solution (achieved 0.50621) has done: 'The timeout is dominated by per-file feature extraction across ~54k train + 6k test `.npy` files, with heavy Python overhead (loops calling `_proj_stats`/`_panel_basic_stats`) and expensive process-based parallelism that re-imports state and increases disk contention. I keep the exact same feature definitions and logistic regression training, but make extraction faster by (1) enumerating file paths once and passing paths to workers (avoids repeated path joins), (2) switching to `ThreadPoolExecutor` for I/O-heavy `np.load(mmap_mode="r")` workloads while keeping deterministic ordering, (3) precomputing repeated reductions inside `_extract_features_from_array` without changing outputs, and (4) ensuring cache loads are true memmaps without extra copies until scaling. These changes are provably equivalent in semantics (same math, same ids ordering, same model), but reduce constant factors substantially so it can finish within 600s when the fallback path is triggered.'
- What this solution (achieved 0.5) has done: 'The timeout is overwhelmingly caused by the fallback path doing per-file feature extraction over ~54k train + 6k test `.npy` files with heavy per-panel computations and Python-level loops, plus extra work from repeatedly building file lists and hashing all IDs for cache names. To keep identical core logic/semantics, the main speedups are: (1) avoid the expensive fallback entirely when any candidate submission exists by hard-short-circuiting before locating/validating/reading train/test; (2) when fallback is needed, eliminate repeated array passes and Python loops by computing per-panel projections for all 6 panels in one vectorized block, reusing precomputed means/abs, and avoiding `np.quantile` overhead for 3-element arrays (replace with exact sorting-based equivalent). Additionally, caching is made O(1) by hashing only stable metadata rather than concatenating all IDs, and feature extraction parallelism is preserved but fed with precomputed file paths and chunking to reduce overhead. These changes preserve the exact model, training loop, feature definitions, and evaluation semantics, while cutting constant factors enough to fit within 600 seconds in typical Kaggle CPU environments.'
- What this solution (achieved 0.50589) has done: 'I fix the runtime error by making `_proj_stats_all` handle both 6-panel inputs and the 3-panel A-minus-off diffs (your current code allocates a `(6,16)` output unconditionally, causing the broadcast failure). This is a minimal, semantics-preserving bug fix: it computes the exact same projection statistics, just for `n_panels = x.shape[0]`. I also make the projection helper robust to small lengths by ensuring `np.partition` is called with a valid `kth` index. After this, the fallback feature extraction + StandardScaler + LogisticRegression path run end-to-end and produce a valid `submission.csv`, which should improve AUC beyond the current near-random 0.5.'
- What this solution (achieved 0.50506) has done: 'Your current score (~0.506) is far below the target (~0.763), so we should cautiously increase AUC while keeping the same “handcrafted features → StandardScaler → LogisticRegression → predict_proba” core pipeline unchanged. The smallest high-impact fix here is to ensure the feature extractor isn’t being fed the wrong scale: these snippets are typically log-scaled in strong solutions, so we add a deterministic per-snippet standardization step (mean/std over the whole 6×273×256 cube) inside `_extract_features_from_array` before computing the *same* downstream features. This preserves the architecture/training approach (still LogisticRegression on summary stats) but makes the engineered ratios/differences comparable across samples, which is often the difference between random and meaningful AUC. Everything else (paths, caching, ensembling behavior, submission schema, and always writing `submission.csv`) remains the same.'

# 9. Code solution

## === cell 0
import os
import glob
import hashlib
import numpy as np
import pandas as pd

os.environ.setdefault("PYTHONHASHSEED", "0")
np.random.seed(0)



## === cell 1
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


def _read_submission_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if not {"id", "target"}.issubset(df.columns):
        raise ValueError(
            f"{path} missing required columns; found {df.columns.tolist()}"
        )
    df = df[["id", "target"]].copy()
    df["id"] = df["id"].astype(str)
    df["target"] = pd.to_numeric(df["target"], errors="coerce")
    if df["target"].isna().any():
        raise ValueError(f"{path} has non-numeric target values.")
    return df


ref_paths = [
    "/kaggle/input/sample_submission.csv",
    "/kaggle/data/sample_submission.csv",
    "/kaggle/data/seti-breakthrough-listen/sample_submission.csv",
    "../input/sample_submission.csv",
    "../input/seti-breakthrough-listen/sample_submission.csv",
]
ref_path = next((p for p in ref_paths if os.path.exists(p)), None)
if ref_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations."
    )

sample_sub = pd.read_csv(ref_path)[["id", "target"]].copy()
sample_sub["id"] = sample_sub["id"].astype(str)



## === cell 2
candidate_globs = [
    "/kaggle/working/submission*.csv",
    "/kaggle/working/*submission*.csv",
    "/kaggle/input/*/submission*.csv",
    "../input/*/submission*.csv",
]
candidate_paths = []
for pattern in candidate_globs:
    candidate_paths.extend(glob.glob(pattern))

candidate_paths = sorted(
    set(
        p
        for p in candidate_paths
        if os.path.basename(p) not in {"sample_submission.csv", "submission.csv"}
    )
)

loaded = []
for p in candidate_paths:
    try:
        df = _read_submission_csv(p)
        m = sample_sub[["id"]].merge(df, on="id", how="left")
        if m["target"].isna().any():
            continue
        loaded.append((p, m["target"].to_numpy(dtype=np.float64)))
    except Exception:
        continue

print(f"Found {len(loaded)} candidate submission(s) to blend.")
for p, _ in loaded[:10]:
    print(" -", p)




## === cell 3
def _find_first_existing(paths):
    return next((p for p in paths if os.path.exists(p)), None)


train_root = _find_first_existing(
    [
        "/kaggle/data/train",
        "/kaggle/data/seti-breakthrough-listen/train",
        "/kaggle/input/train",
        "/kaggle/input/seti-breakthrough-listen/train",
        "../input/train",
        "../input/seti-breakthrough-listen/train",
    ]
)
test_root = _find_first_existing(
    [
        "/kaggle/data/test",
        "/kaggle/data/seti-breakthrough-listen/test",
        "/kaggle/input/test",
        "/kaggle/input/seti-breakthrough-listen/test",
        "../input/test",
        "../input/seti-breakthrough-listen/test",
    ]
)
labels_path = _find_first_existing(
    [
        "/kaggle/data/train_labels.csv",
        "/kaggle/data/seti-breakthrough-listen/train_labels.csv",
        "/kaggle/input/train_labels.csv",
        "/kaggle/input/seti-breakthrough-listen/train_labels.csv",
        "../input/train_labels.csv",
        "../input/seti-breakthrough-listen/train_labels.csv",
    ]
)

A_IDX = np.array([0, 2, 4], dtype=np.int64)
O_IDX = np.array([1, 3, 5], dtype=np.int64)
EPS32 = np.float32(1e-6)


def _id_to_npy_path(root_dir: str, fid: str) -> str:
    return os.path.join(root_dir, fid[0], fid + ".npy")


def _validate_ids_exist(root_dir: str, ids):
    missing = []
    for fid in ids:
        fp = _id_to_npy_path(root_dir, fid)
        if not os.path.exists(fp):
            missing.append(fid)
            if len(missing) >= 10:
                break
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} .npy files under {root_dir}; first few: {missing[:10]}"
        )


def _q75_len3(v3: np.ndarray) -> np.float32:
    s = np.sort(v3.astype(np.float32, copy=False))
    return np.float32(0.5 * (s[1] + s[2]))


def _proj_stats_all(xp: np.ndarray) -> np.ndarray:
    r = xp.mean(axis=2, dtype=np.float32)  # (n_panels,H)
    c = xp.mean(axis=1, dtype=np.float32)  # (n_panels,W)
    ar = np.abs(r)
    ac = np.abs(c)

    n_panels = r.shape[0]
    out = np.empty((n_panels, 16), dtype=np.float32)

    def _summ(v2: np.ndarray, dst2: np.ndarray):
        L = int(v2.shape[1])
        k = 10 if L >= 10 else max(1, L // 4)
        kth = max(0, L - k)
        pv = np.partition(v2, kth, axis=1)[:, L - k :]  # (n_panels,k)

        dst2[:, 0] = v2.mean(axis=1, dtype=np.float32)
        dst2[:, 1] = v2.std(axis=1, dtype=np.float32)
        dst2[:, 2] = v2.max(axis=1, initial=-np.inf)
        dst2[:, 3] = pv.mean(axis=1, dtype=np.float32)

    _summ(r, out[:, 0:4])
    _summ(c, out[:, 4:8])
    _summ(ar, out[:, 8:12])
    _summ(ac, out[:, 12:16])
    return out


def _panel_basic_stats(img2d: np.ndarray) -> np.ndarray:
    x = img2d.astype(np.float32, copy=False)
    m = x.mean(dtype=np.float32)
    s = x.std(dtype=np.float32)
    mx = x.max(initial=-np.inf)
    mn = x.min(initial=np.inf)
    ax = np.abs(x)
    am = ax.mean(dtype=np.float32)
    en = (x * x).mean(dtype=np.float32)
    return np.array(
        [m, s, mx, mn, am, en, np.log1p(en), np.log1p(am)], dtype=np.float32
    )


def _extract_features_from_array(arr: np.ndarray) -> np.ndarray:
    x = arr.astype(np.float32, copy=False)
    mu = x.mean(dtype=np.float32)
    sig = x.std(dtype=np.float32)
    x = (x - mu) / (sig + EPS32)

    abs_x = np.abs(x)
    x2 = x * x

    panel_mean = x.mean(axis=(1, 2), dtype=np.float32)  # (6,)
    panel_std = x.std(axis=(1, 2), dtype=np.float32)  # (6,)
    panel_max = x.max(axis=(1, 2), initial=-np.inf)  # (6,)
    panel_min = x.min(axis=(1, 2), initial=np.inf)  # (6,)
    panel_absmean = abs_x.mean(axis=(1, 2), dtype=np.float32)  # (6,)
    panel_energy = x2.mean(axis=(1, 2), dtype=np.float32)  # (6,)

    A_mean = panel_mean[A_IDX].mean(dtype=np.float32)
    O_mean = panel_mean[O_IDX].mean(dtype=np.float32)
    A_energy = panel_energy[A_IDX].mean(dtype=np.float32)
    O_energy = panel_energy[O_IDX].mean(dtype=np.float32)
    A_absmean = panel_absmean[A_IDX].mean(dtype=np.float32)
    O_absmean = panel_absmean[O_IDX].mean(dtype=np.float32)

    A_mean_std = panel_mean[A_IDX].std(dtype=np.float32)
    O_mean_std = panel_mean[O_IDX].std(dtype=np.float32)
    A_energy_std = panel_energy[A_IDX].std(dtype=np.float32)
    O_energy_std = panel_energy[O_IDX].std(dtype=np.float32)
    A_absmean_std = panel_absmean[A_IDX].std(dtype=np.float32)
    O_absmean_std = panel_absmean[O_IDX].std(dtype=np.float32)

    A_mean_med = np.median(panel_mean[A_IDX]).astype(np.float32, copy=False)
    O_mean_med = np.median(panel_mean[O_IDX]).astype(np.float32, copy=False)
    A_energy_med = np.median(panel_energy[A_IDX]).astype(np.float32, copy=False)
    O_energy_med = np.median(panel_energy[O_IDX]).astype(np.float32, copy=False)

    A_mean_q75 = _q75_len3(panel_mean[A_IDX])
    O_mean_q75 = _q75_len3(panel_mean[O_IDX])
    A_energy_q75 = _q75_len3(panel_energy[A_IDX])
    O_energy_q75 = _q75_len3(panel_energy[O_IDX])

    A_panel_max_of_max = panel_max[A_IDX].max(initial=-np.inf)
    O_panel_max_of_max = panel_max[O_IDX].max(initial=-np.inf)

    energy_ratio = (A_energy + EPS32) / (O_energy + EPS32)
    absmean_ratio = (A_absmean + EPS32) / (O_absmean + EPS32)
    energy_diff_norm = (A_energy - O_energy) / (A_energy + O_energy + EPS32)
    absmean_diff_norm = (A_absmean - O_absmean) / (A_absmean + O_absmean + EPS32)

    log_panel_energy = np.log1p(panel_energy).astype(np.float32, copy=False)
    log_panel_absmean = np.log1p(panel_absmean).astype(np.float32, copy=False)
    log_A_energy = np.float32(np.log1p(A_energy))
    log_O_energy = np.float32(np.log1p(O_energy))
    log_A_absmean = np.float32(np.log1p(A_absmean))
    log_O_absmean = np.float32(np.log1p(O_absmean))

    pair_d_mean = (panel_mean[A_IDX] - panel_mean[O_IDX]).astype(np.float32, copy=False)
    pair_d_std = (panel_std[A_IDX] - panel_std[O_IDX]).astype(np.float32, copy=False)
    pair_d_energy = (panel_energy[A_IDX] - panel_energy[O_IDX]).astype(
        np.float32, copy=False
    )
    pair_d_absmean = (panel_absmean[A_IDX] - panel_absmean[O_IDX]).astype(
        np.float32, copy=False
    )

    pair_log_pos = np.log1p(np.maximum(pair_d_energy, 0.0)).astype(
        np.float32, copy=False
    )
    pair_log_neg = np.log1p(np.maximum(-pair_d_energy, 0.0)).astype(
        np.float32, copy=False
    )

    proj = _proj_stats_all(x)  # (6,16)
    proj_flat = proj.reshape(-1)
    proj_A = proj[A_IDX].mean(axis=0, dtype=np.float32)
    proj_O = proj[O_IDX].mean(axis=0, dtype=np.float32)
    proj_agg = np.concatenate([proj_A, proj_O, proj_A - proj_O]).astype(
        np.float32, copy=False
    )

    diffs = (x[A_IDX] - x[O_IDX]).astype(np.float32, copy=False)  # (3,273,256)
    diff_stats = np.empty((3, 8), dtype=np.float32)
    for i in range(3):
        diff_stats[i] = _panel_basic_stats(diffs[i])
    diff_proj = _proj_stats_all(diffs)  # (3,16)

    diff_stats_flat = diff_stats.reshape(-1)
    diff_proj_flat = diff_proj.reshape(-1)

    diff_agg = np.concatenate(
        [
            diff_stats.mean(axis=0, dtype=np.float32),
            diff_stats.std(axis=0, dtype=np.float32),
            diff_proj.mean(axis=0, dtype=np.float32),
            diff_proj.std(axis=0, dtype=np.float32),
        ]
    ).astype(np.float32, copy=False)

    base = np.concatenate(
        [
            panel_mean,
            panel_std,
            panel_max.astype(np.float32, copy=False),
            panel_min.astype(np.float32, copy=False),
            panel_absmean,
            panel_energy,
            np.array(
                [
                    A_mean,
                    O_mean,
                    A_mean - O_mean,
                    A_energy,
                    O_energy,
                    A_energy - O_energy,
                    A_absmean,
                    O_absmean,
                    A_absmean - O_absmean,
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32, copy=False)

    extra = np.array(
        [
            A_mean_std,
            O_mean_std,
            A_mean_std - O_mean_std,
            A_energy_std,
            O_energy_std,
            A_energy_std - O_energy_std,
            A_absmean_std,
            O_absmean_std,
            A_absmean_std - O_absmean_std,
            A_mean_med,
            O_mean_med,
            A_mean_med - O_mean_med,
            A_energy_med,
            O_energy_med,
            A_energy_med - O_energy_med,
            A_mean_q75,
            O_mean_q75,
            A_mean_q75 - O_mean_q75,
            A_energy_q75,
            O_energy_q75,
            A_energy_q75 - O_energy_q75,
            A_panel_max_of_max,
            O_panel_max_of_max,
            A_panel_max_of_max - O_panel_max_of_max,
        ],
        dtype=np.float32,
    )

    robust = np.concatenate(
        [
            log_panel_energy,
            log_panel_absmean,
            np.array(
                [
                    log_A_energy,
                    log_O_energy,
                    log_A_energy - log_O_energy,
                    log_A_absmean,
                    log_O_absmean,
                    log_A_absmean - log_O_absmean,
                    energy_ratio,
                    absmean_ratio,
                    energy_diff_norm,
                    absmean_diff_norm,
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32, copy=False)

    pairwise = np.concatenate(
        [
            pair_d_mean,
            pair_d_std,
            pair_d_energy,
            pair_d_absmean,
            pair_log_pos,
            pair_log_neg,
        ]
    ).astype(np.float32, copy=False)

    return np.concatenate(
        [
            base,
            extra,
            robust,
            pairwise,
            proj_flat,
            proj_agg,
            diff_stats_flat,
            diff_proj_flat,
            diff_agg,
        ]
    ).astype(np.float32, copy=False)


def _feature_cache_path(split: str, root_dir: str, n_ids: int, n_feats: int) -> str:
    safe_root = os.path.basename(root_dir.rstrip("/"))
    key = f"{split}|{safe_root}|n{n_ids}|f{n_feats}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
    return (
        f"/kaggle/working/feat_cache_{split}_{safe_root}_h{h}_n{n_ids}_f{n_feats}.npy"
    )


def _extract_features_from_path(fp: str) -> np.ndarray:
    arr = np.load(fp, mmap_mode="r")
    return _extract_features_from_array(arr)


def _make_feature_matrix(ids, cache_split: str, root_dir: str):
    if len(ids) == 0:
        raise ValueError("No ids provided to build feature matrix.")
    first_arr = np.load(_id_to_npy_path(root_dir, ids[0]), mmap_mode="r")
    n_feats = _extract_features_from_array(first_arr).shape[0]

    cache_fp = _feature_cache_path(cache_split, root_dir, len(ids), n_feats)
    if os.path.exists(cache_fp):
        Xmm = np.load(cache_fp, mmap_mode="r")
        if Xmm.shape == (len(ids), n_feats):
            return np.asarray(Xmm, dtype=np.float32)

    fps = [_id_to_npy_path(root_dir, fid) for fid in ids]
    X = np.empty((len(ids), n_feats), dtype=np.float32)

    try:
        from concurrent.futures import ThreadPoolExecutor

        max_workers = min(16, (os.cpu_count() or 2) * 2)
        chunksize = 64
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for i, feat in enumerate(
                ex.map(_extract_features_from_path, fps, chunksize=chunksize)
            ):
                X[i] = feat
    except Exception as e:
        print(
            f"Thread feature extraction failed ({type(e).__name__}: {e}); falling back to sequential."
        )
        for i, fp in enumerate(fps):
            X[i] = _extract_features_from_path(fp)

    np.save(cache_fp, X)
    return X


fallback_pred = None

if len(loaded) == 0:
    if train_root is None or test_root is None or labels_path is None:
        raise FileNotFoundError(
            "No external submissions found and could not locate train/test/labels to build a model fallback."
        )

    labels = pd.read_csv(labels_path)
    labels["id"] = labels["id"].astype(str)

    train_ids_all = labels["id"].tolist()
    test_ids = sample_sub["id"].tolist()

    _validate_ids_exist(train_root, train_ids_all)
    _validate_ids_exist(test_root, test_ids)

    y = labels.set_index("id").loc[train_ids_all, "target"].to_numpy(dtype=np.int32)

    print("Building features:")
    print(" - train_root:", train_root, "using:", len(train_ids_all))
    print(" - test_root :", test_root, "predicting:", len(test_ids))

    X_train = _make_feature_matrix(
        train_ids_all, cache_split="train", root_dir=train_root
    )
    X_test = _make_feature_matrix(test_ids, cache_split="test", root_dir=test_root)

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=1200,
        n_jobs=None,
        random_state=0,
        class_weight="balanced",
        C=3.0,
    )

    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        tr_idx, va_idx = next(sss.split(X_train_s, y))
        clf_tmp = LogisticRegression(
            solver="lbfgs",
            max_iter=1200,
            n_jobs=None,
            random_state=0,
            class_weight="balanced",
            C=3.0,
        )
        clf_tmp.fit(X_train_s[tr_idx], y[tr_idx])
        va_pred = clf_tmp.predict_proba(X_train_s[va_idx])[:, 1]
        print("Holdout AUC (sanity check):", roc_auc_score(y[va_idx], va_pred))
    except Exception as e:
        print("Could not compute holdout AUC sanity check:", repr(e))

    clf.fit(X_train_s, y)

    try:
        train_pred = clf.predict_proba(X_train_s)[:, 1]
        print("Train AUC (sanity check):", roc_auc_score(y, train_pred))
    except Exception as e:
        print("Could not compute train AUC sanity check:", repr(e))

    fallback_pred = clf.predict_proba(X_test_s)[:, 1].astype(np.float64)



## === cell 4
if len(loaded) > 0:
    preds = np.vstack([arr for _, arr in loaded])
    pred = preds.mean(axis=0)
else:
    if fallback_pred is None:
        pred = np.full(len(sample_sub), 0.5, dtype=np.float64)
    else:
        pred = fallback_pred

pred = np.clip(pred, 0.0, 1.0)
data1 = pd.DataFrame({"id": sample_sub["id"].values, "target": pred})



## === cell 5
data11 = data1.copy()



## === cell 6
data11 = data11[["id", "target"]].copy()

if len(data11) != len(sample_sub):
    raise ValueError(
        f"Submission length mismatch: {len(data11)} vs expected {len(sample_sub)}"
    )
if data11["id"].duplicated().any():
    raise ValueError("Duplicate ids found in submission.")
if not np.isfinite(data11["target"]).all():
    raise ValueError("Non-finite target values found in submission.")



## === cell 7
data11.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", data11.shape)
print(data11.head())
print("target summary:", data11["target"].describe())
